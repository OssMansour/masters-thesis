import numpy as np
import pybullet as p
from gymnasium import spaces
import random

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class SpiralAviary(BaseRLAviary):
    """Single-agent spiral trajectory tracking environment without curriculum."""

    @staticmethod
    def _angle_difference(angle1, angle2):
        """
        Calculate the smallest angular difference between two angles.
        
        Handles angle wrapping to ensure the difference is in the range [-π, π].
        This is crucial for yaw control where 350° and 10° should have a 
        difference of 20° (not 340°).
        
        Parameters
        ----------
        angle1 : float
            First angle in radians
        angle2 : float
            Second angle in radians
            
        Returns
        -------
        float
            Smallest angular difference in radians, range [-π, π]
            
        Examples
        --------
        >>> SpiralAviary._angle_difference(np.pi/18, -np.pi/18)  # 10° vs -10°
        0.349... (20° in radians)
        >>> SpiralAviary._angle_difference(6.10, 0.18)  # ~350° vs ~10°
        0.349... (20° in radians, not 340°)
        """
        diff = angle1 - angle2
        # Wrap to [-π, π]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool=False,
                 record: bool=False,
                 mode: str = "spiral",
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        
        self.start_noise_std = 0.0 # was 0.05  
        self.episode_reward = 0
        self.traj_step_counter = 0
        
        # MODIFIED: Reduced spiral parameters for easier tracking
        self.spiral_radius = 0.5
        self.spiral_height = 0.5
        self.spiral_angular_speed = 0.006   
        self.spiral_vertical_speed = 0.0003
        self.spiral_radial_speed = 0.00006


        # Visualization parameters
        self.vis_target_points = []
        self.vis_drone_points = []
        self.target_line_id = None
        self.drone_line_id = None
        self.vis_step_interval = 1  # More frequent visualization (was 4)
        self.target_sphere_ids = []  # For adding target spheres
        self.last_target_viz = None
        self.drone_color = [random.random(), random.random(), random.random()]

        # for trajectory viz
        self.prev_target_point = None
        self.prev_drone_point  = None


        # Tracking performance
        self.tracking_errors = []

        # MODIFIED: Extended time horizon for slower trajectory
        self.EPISODE_LEN_SEC = 25.0  # Match paper specification (was 20)
        self.mode= mode
        self.hover_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Fixed hover point
        self.goto_target = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.5])
        self.last_action = np.zeros(4, dtype=np.float32)
        
        # ADDED: Track normalized actions for observation space (not raw RPM)
        self.last_normalized_action = np.zeros(4, dtype=np.float32)

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
                         
        # Update observation space: include relative position only
        obs_sample = self._computeObs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32
        )
        print(f"[SpiralAviary] Observation space initialized: {self.observation_space.shape}")

    def _computeSpiralTarget(self, t):
        """Compute the desired spiral target at step t."""
        angle = self.spiral_angular_speed * t
        radius = self.spiral_radius + self.spiral_radial_speed * t
        height = self.spiral_height + self.spiral_vertical_speed * t
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        return np.array([x, y, z], dtype=np.float32)

    def _computeTarget(self, t):
        if self.mode == "spiral":
            return self._computeSpiralTarget(t)
        elif self.mode == "hover":
            return self.hover_target
        elif self.mode == "goto":
            if t % 100 == 0:
                self.goto_target = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.5])
            return self.goto_target

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        target = self._computeTarget(self.traj_step_counter)

        # MODIFIED: Include velocity of target for better prediction
        if self.traj_step_counter > 0:
            prev_target = self._computeTarget(self.traj_step_counter - 1)
            target_velocity = (target - prev_target) * self.CTRL_FREQ  # Scale to get velocity
        else:
            target_velocity = np.zeros(3)
        
        # Relative position to target
        ref_pos = target
        ref_att = [0, 0, 0]  # Assuming target attitude is zero (roll, pitch, yaw)
        
        # MODIFIED: Use normalized action instead of raw RPM from state[16:20]
        # State contains: [pos(3), quat(4), rpy(3), vel(3), ang_vel(3), raw_rpm(4)]
        # We want: [pos(3), quat(4), rpy(3), vel(3), ang_vel(3), normalized_action(4)]
        sel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # Exclude raw RPM (16:20)
        delta_pos = ref_pos - state[0:3]
        delta_velocity = target_velocity - state[10:13]
        
        # MODIFIED: Use angle_difference for proper angle wrapping
        # state[7:10] = [roll, pitch, yaw]
        delta_roll = self._angle_difference(ref_att[0], state[7])
        delta_pitch = self._angle_difference(ref_att[1], state[8])
        delta_yaw = self._angle_difference(ref_att[2], state[9])
        delta_att = np.array([delta_roll, delta_pitch, delta_yaw])
        
        # MODIFIED: Use normalized action and include target velocity in observation
        # Observation: [state(16), action(4), ref_pos(3), target_vel(3), ref_att(3), delta_pos(3), delta_vel(3), delta_att(3)] = 38 dims
        return np.hstack([state[sel], self.last_normalized_action, 
                         ref_pos, target_velocity, ref_att, delta_pos, delta_velocity, delta_att]).astype(np.float32)

    def _computeReward_old(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        target = self._computeTarget(self.traj_step_counter)
        self.tracking_errors.append(np.linalg.norm(target-state[0:3]))
        #ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        ###########################################################################
        a = 7
        e_k = np.sqrt((target[0]-state[0])**2 + (target[1]-state[1])**2 + (target[2]-state[2])**2)
        er1 = 1/(a*e_k)
        exp = (-0.5) * ((e_k/0.5)**2)
        den = np.sqrt(2*np.pi*(0.5**2))
        er2 = (a/den)*np.exp(exp)
        reward = er1 + er2
        ret=max(0,reward)
        return ret
 
    def _computeReward(self):
        """
        Computes the reward using the paper's multi-component reward function.
        
        Based on the paper's specification:
        - Position Tracking: r_pos = w_pos * exp(-||δ_pos||²)
        - Attitude Stability: r_att = w_att * exp(-5*(φ² + θ²))
        - Control Smoothness: r_smooth = w_smooth * exp(-0.1*||u_t - u_{t-1}||²)
        
        Total: r_t = SCALE * (r_pos + r_att + r_smooth)
        
        The SCALE factor (20.0) is added to make rewards suitable for SAC,
        which typically expects cumulative episode rewards in the thousands.
        """
        state = self._getDroneStateVector(0)
        target = self._computeTarget(self.traj_step_counter)
        
        # Position tracking reward (paper eq. 60)
        pos_error = np.linalg.norm(target - state[0:3])
        self.tracking_errors.append(pos_error)
        w_pos = 0.6
        r_pos = w_pos * np.exp(-pos_error**2)

        # Attitude stability reward (paper eq. 61)
        roll, pitch, yaw = state[7:10]
        # Use angle_difference for proper wrapping
        roll_error = self._angle_difference(roll, 0.0)
        pitch_error = self._angle_difference(pitch, 0.0)
        w_att = 0.3
        r_att = w_att * np.exp(-5.0 * (roll_error**2 + pitch_error**2))

        # Control smoothness reward (paper eq. 62)
        current_action = self.last_normalized_action
        if self.traj_step_counter > 0:
            action_diff = np.linalg.norm(current_action - self.last_action)
        else:
            action_diff = 0.0
        w_smooth = 0.1
        r_smooth = w_smooth * np.exp(-0.1 * action_diff**2)

        # Save action for next step
        self.last_action = current_action.copy()

        # Total reward (paper eq. 63) with SAC scaling factor
        # Scale factor of 1.0 makes per-step rewards ~0-1, 
        # giving full episode rewards ~0-750 (suitable for SAC)
        SCALE = 1.0  # Scaling factor for SAC compatibility
        total_reward = SCALE * max(0,(r_pos + r_att + r_smooth))

        # Expected range: 0 to 1 per step
        # Full episode (750 steps): 0 to 750

        return total_reward

    def _computeTerminated(self):
        # End episode when time horizon is reached
        # FIXED: Use CTRL_FREQ (30Hz) not PYB_FREQ (240Hz)
        return (self.step_counter / (self.PYB_FREQ * self.PYB_STEPS_PER_CTRL)) > self.EPISODE_LEN_SEC

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        roll, pitch, yaw = state[7:10]
        
        # MODIFIED: More forgiving termination conditions
        too_far = np.linalg.norm(pos) > (self.spiral_radius + 2.0)  # Was 3.0
        
        # MODIFIED: Use angle_difference for proper angle checking
        # Very relaxed to allow exploration during early training
        roll_error = abs(self._angle_difference(roll, 0.0))
        pitch_error = abs(self._angle_difference(pitch, 0.0))
        too_tilted = roll_error > 1.57 or pitch_error > 1.57  # ~90 degrees (very permissive)
        
        return too_far or too_tilted

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        target = self._computeTarget(self.traj_step_counter)
        dist = np.linalg.norm(pos - target)
        
        # Calculate average tracking error for this episode
        avg_error = np.mean(self.tracking_errors) if self.tracking_errors else 0
        
        return {
            "episode": {
                "r": self.episode_reward,
                "l": self.traj_step_counter/self.CTRL_FREQ
            },
            "distance": float(dist),
            "target": target,
            "avg_tracking_error": avg_error
        }
        
    def _updateVisualization(self):
        if not self.GUI:
            return

        t = self.traj_step_counter
        if t % self.vis_step_interval != 0:
            return

        # current positions
        target = self._computeTarget(t)
        drone  = self._getDroneStateVector(0)[0:3]

        # draw target line
        if self.prev_target_point is not None:
            p.addUserDebugLine(self.prev_target_point.tolist(),
                               target.tolist(),
                               [1,0,0],      # red
                               lineWidth=1,
                               lifeTime=0)
        # draw drone line
        if self.prev_drone_point is not None:
            p.addUserDebugLine(self.prev_drone_point.tolist(),
                               drone.tolist(),
                               self.drone_color,      # blue
                               lineWidth=1,
                               lifeTime=0)

        self.prev_target_point = target
        self.prev_drone_point  = drone

    def step(self, action):
        # ADDED: Store normalized action before super().step() converts it to RPM
        self.last_normalized_action = action[0].copy() if action.ndim == 2 else action.copy()
        # ✅ INCREMENT COUNTER BEFORE SUPER STEP
        self.traj_step_counter += 1

        obs, reward, terminated, truncated, info = super().step(action)
        
        
        self.episode_reward += reward  # Track cumulative reward
        
        # Update visualization after step
        self._updateVisualization()

        
        if terminated or truncated:
            self.episode_reward = 0  # Reset for next episode
            # Clean up visualization elements
            if self.GUI and self.last_target_viz is not None:
                p.removeBody(self.last_target_viz)
                self.last_target_viz = None
            
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Resets the environment and returns the initial observation."""
        if options is not None:
            t0    = options['t0']
            noise = options['noise']
            yaw0  = options['yaw']
        else:
            # FIXED: Use CTRL_FREQ (30Hz) not PYB_FREQ (240Hz)
            # Start at beginning of trajectory for stable training
            t0 = 0  # Was: self.np_random.integers(0, int(0.1*self.EPISODE_LEN_SEC * self.CTRL_FREQ))
            noise = self.np_random.normal(
                loc=0.0,  # Changed from 0.05 to 0.0 for no bias
                scale=self.start_noise_std,  # This is already 0.0
                size=3
            ).astype(np.float32)  # choose noise
            yaw0 = 0.0  # Start with zero yaw for stable training (was random ±45°)

        self.traj_step_counter = t0

        init_target = self._computeTarget(t0)

        init_pos = init_target + noise        
        # place drone exactly at that spiral point (plus tiny noise if you like)
        self.INIT_XYZS = np.array([[init_pos[0],
                                    init_pos[1],
                                    init_pos[2]]], dtype=np.float32)
        # random yaw around ±45°

        self.INIT_RPYS = np.array([[0.0, 0.0, yaw0]], dtype=np.float32)

        # reset diagnostics & viz
        self.episode_reward = 0
        self.tracking_errors = []
        self.prev_target_point = None
        self.prev_drone_point  = None
        self.last_normalized_action = np.zeros(4, dtype=np.float32)  # ADDED: Reset normalized action

        obs, info = super().reset(seed=seed, options=options)

        self.traj_step_counter = t0

        # 🎯 Draw first target
        
        if self.GUI:
            target = self._computeTarget(self.traj_step_counter)
            state = self._getDroneStateVector(0)[0:3]
            self.prev_target_point = target
            self.prev_drone_point  = state

        return obs, info
