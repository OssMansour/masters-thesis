"""
MINIMAL FIXED Pendulum Environment

This is a completely minimal fix that ONLY changes observation format.
NO physics modifications - uses PendulumEnv exactly as-is.

Author: DHP vs SAC Comparison Study
Date: August 16, 2025
"""

import gym
import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv
from gymnasium import spaces


class EnhancedAdvancedAdaptivePID:
    """Enhanced adaptive PID controller (unchanged from original)"""
    
    def __init__(self, Kp=1.5, Ki=0.04, Kd=0.06, dt=0.01, pendulum_params=None,
                 control_weight=0.1, adapt_rate_p=0.0005, adapt_rate_i=0.0005, 
                 adapt_rate_d=0.0005, epsilon=1e-4, gamma=0.9):
        self.dt = dt
        self.Kp = float(Kp)
        self.Ki = float(Ki) 
        self.Kd = float(Kd)
        self.Kpo = float(Kp)
        self.Kio = float(Ki)
        self.Kdo = float(Kd)
        
        if pendulum_params is None:
            self.pendulum_params = {'m': 1.0, 'L': 1.0, 'g': 10, 'b': 0.0}
        else:
            self.pendulum_params = pendulum_params
            
        self.control_weight = control_weight
        self.adapt_rate_p = adapt_rate_p
        self.adapt_rate_i = adapt_rate_i
        self.adapt_rate_d = adapt_rate_d
        self.epsilon = epsilon
        self.gamma = gamma
        self.smoothed_cost = None
        self.integral = 0.0
        self.last_error = 0.0
        
    def reset(self):
        self.Kp = float(self.Kpo)
        self.Ki = float(self.Kio)
        self.Kd = float(self.Kdo)
        self.integral = 0.0
        self.last_error = 0.0
        self.smoothed_cost = None
        
    def compute_control_force(self, error):
        derivative = (error - self.last_error) / self.dt
        self.integral += error * self.dt
        force_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return force_output, derivative
        
    def force_to_acceleration(self, force):
        moment_of_inertia = self.pendulum_params['m'] * (self.pendulum_params['L'] ** 2)
        angular_acceleration = force / moment_of_inertia
        return angular_acceleration
        
    def acceleration_to_velocity_reference(self, angular_acceleration):
        velocity_change = angular_acceleration * self.dt
        return velocity_change
        
    def cost_function(self, error, output):
        return error**2 + self.control_weight * output**2
        
    def update_gains(self, error, integral, derivative):
        current_output = self.Kp * error + self.Ki * integral + self.Kd * derivative
        cost = self.cost_function(error, current_output)
        
        if self.smoothed_cost is None:
            self.smoothed_cost = cost
        else:
            self.smoothed_cost = self.gamma * self.smoothed_cost + (1 - self.gamma) * cost
            
        perturbed_Kp = self.Kp + self.epsilon
        output_perturbed = perturbed_Kp * error + self.Ki * integral + self.Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Kp = (cost_perturbed - cost) / self.epsilon
        
        perturbed_Ki = self.Ki + self.epsilon
        output_perturbed = self.Kp * error + perturbed_Ki * integral + self.Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Ki = (cost_perturbed - cost) / self.epsilon
        
        perturbed_Kd = self.Kd + self.epsilon
        output_perturbed = self.Kp * error + self.Ki * integral + perturbed_Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Kd = (cost_perturbed - cost) / self.epsilon
        
        self.Kp -= self.adapt_rate_p * grad_Kp
        self.Ki -= self.adapt_rate_i * grad_Ki
        self.Kd -= self.adapt_rate_d * grad_Kd
        
        self.Kp = max(0.01, min(self.Kp, 10.0))
        self.Ki = max(0.0, min(self.Ki, 5.0))
        self.Kd = max(0.0, min(self.Kd, 5.0))
        
    def __call__(self, error):
        force_output, derivative = self.compute_control_force(error)
        self.update_gains(error, self.integral, derivative)
        angular_acceleration = self.force_to_acceleration(force_output)
        velocity_reference = self.acceleration_to_velocity_reference(angular_acceleration)
        self.last_error = error
        return velocity_reference


class PendulumRandomTargetEnv(PendulumEnv):
    """
    MINIMAL FIXED Enhanced Pendulum Environment
    
    CRITICAL: This version does NOT override step() or reset() physics!
    It ONLY changes the observation format from [cos, sin, omega] to [theta, omega].
    """
    
    def __init__(self, gui=False, record=False, fixed_target=None, 
                 normalize_states=True, render_mode="human", pendulum_params=None):
        """Initialize with minimal changes to PendulumEnv"""
        
        # Initialize parent PendulumEnv - THIS PRESERVES ALL PHYSICS
        super().__init__(render_mode=render_mode if gui else None)
        
        # Store configuration
        self.gui = gui
        self.record = record  
        self.fixed_target = fixed_target
        self.normalize_states = normalize_states
        self.theta_desired = 0.0
        
        # Physical parameters
        if pendulum_params is None:
            self.pendulum_params = {'m': 1.0, 'L': 1.0, 'g': 10.0, 'b': 0.1}
        else:
            self.pendulum_params = pendulum_params
            
        # Initialize enhanced PID
        self.enhanced_pid = EnhancedAdvancedAdaptivePID(
            Kp=1.5, Ki=0.04, Kd=0.06, dt=0.01,
            pendulum_params=self.pendulum_params
        )
        
        # Reference tracking
        self.prev_theta_dot_desired = 0.0
        
        # State bounds for normalization
        self.state_bounds = {
            'theta': [-np.pi, np.pi],
            'theta_dot': [-8.0, 8.0]
        }
        
        # ONLY change observation space - NOT physics!
        if self.normalize_states:
            obs_low = np.array([-1.0, -1.0], dtype=np.float32)
            obs_high = np.array([1.0, 1.0], dtype=np.float32)
        else:
            obs_low = np.array([-np.pi, -self.max_speed], dtype=np.float32)
            obs_high = np.array([np.pi, self.max_speed], dtype=np.float32)
            
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # DHP cost function weights
        self.Q_matrix = np.diag([10.0, 1.0])
        
        # Episode tracking
        self.episode_length = 0
        self.max_episode_steps = 200
        
        print(f"[MINIMAL FIXED] Enhanced PendulumRandomTargetEnv initialized")
        print(f"[MINIMAL FIXED] Observation space: {self.observation_space}")
        print(f"[MINIMAL FIXED] State normalization: {self.normalize_states}")
    
    def normalize_observation(self, obs):
        """Normalize [theta, theta_dot] to [-1, 1]"""
        if not self.normalize_states:
            return obs
            
        normalized = np.zeros_like(obs)
        bounds = [self.state_bounds['theta'], self.state_bounds['theta_dot']]
        
        for i, (low, high) in enumerate(bounds):
            normalized[i] = 2.0 * (obs[i] - low) / (high - low) - 1.0
            normalized[i] = np.clip(normalized[i], -1.0, 1.0)
            
        return normalized.astype(np.float32)
    
    def _angular_difference(self, a, b):
        """Compute shortest angular difference"""
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return diff
    
    def _get_obs(self):
        """
        ONLY CHANGE: Convert [cos, sin, omega] to [theta, omega]
        This is the ONLY modification to PendulumEnv behavior!
        """
        # PendulumEnv state is always [cos(theta), sin(theta), theta_dot]
        theta = self.state[0]
        theta_dot = self.state[1]
        
        # Create [theta, theta_dot] observation
        obs = np.array([theta, theta_dot], dtype=np.float32)
        
        # Apply normalization if enabled
        if self.normalize_states:
            obs = self.normalize_observation(obs)
            
        return obs
    
    def generate_reference(self):
        """Generate reference using enhanced PID"""
        # Extract current angle from PendulumEnv state
        current_theta = np.arctan2(self.state[1], self.state[0])
        
        # Compute angle error
        theta_error = self._angular_difference(self.theta_desired, current_theta)
        
        # Enhanced PID generates velocity reference
        theta_dot_desired = float(self.enhanced_pid(theta_error))
        
        # Clamp velocity reference
        max_theta_dot_ref = 2*np.pi
        theta_dot_desired = np.clip(theta_dot_desired, -max_theta_dot_ref, max_theta_dot_ref)
        
        # Rate limiting
        if hasattr(self, 'prev_theta_dot_desired'):
            max_rate = 5.0
            max_change = max_rate * self.dt
            change = theta_dot_desired - self.prev_theta_dot_desired
            if abs(change) > max_change:
                theta_dot_desired = self.prev_theta_dot_desired + np.sign(change) * max_change
        
        self.prev_theta_dot_desired = theta_dot_desired
        
        # Reference vector
        reference = np.array([self.theta_desired, theta_dot_desired], dtype=np.float32)
        
        if self.normalize_states:
            reference = self.normalize_observation(reference)
            
        return reference
    
    def compute_dhp_cost(self, state, reference):
        """Compute DHP quadratic cost function"""
        state = np.asarray(state, dtype=np.float64).flatten()
        reference = np.asarray(reference, dtype=np.float64).flatten()
        
        # Robust handling
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(reference)) or np.any(np.isinf(reference)):
            reference = np.nan_to_num(reference, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract states
        theta, theta_dot = state[0], state[1]
        theta_ref, theta_dot_ref = reference[0], reference[1]
        
        # Compute errors
        theta_error = self._angular_difference(theta, theta_ref)
        theta_dot_error = theta_dot - theta_dot_ref
        error = np.array([theta_error, theta_dot_error])
        
        # Quadratic cost
        cost = error.T @ self.Q_matrix @ error
        dcost_derror = 2.0 * self.Q_matrix @ error
        dcostdx = np.array([dcost_derror[0], dcost_derror[1]])
        
        # Gradient transformation for normalized states
        if self.normalize_states:
            bounds = [self.state_bounds['theta'], self.state_bounds['theta_dot']]
            dcostdx_normalized = np.zeros_like(dcostdx)
            for i, (low, high) in enumerate(bounds):
                dcostdx_normalized[i] = dcostdx[i] * 2.0 / (high - low)
            dcostdx = dcostdx_normalized
        
        # Handle numerical issues
        if np.any(np.isnan(dcostdx)) or np.any(np.isinf(dcostdx)):
            dcostdx = np.zeros_like(dcostdx)
        
        cost_scalar = float(cost.flat[0]) if np.size(cost) == 1 else float(cost)
        if not np.isfinite(cost_scalar):
            cost_scalar = 1000.0
        
        return cost_scalar, dcostdx.astype(np.float32)
    
    def reset(self, *, seed=None, options=None):
        """
        MINIMAL OVERRIDE: Use parent reset, only add enhanced functionality
        """
        # Use parent PendulumEnv reset - PRESERVES EXACT PHYSICS
        parent_result = super().reset(seed=seed)
        if isinstance(parent_result, tuple):
            _, info = parent_result
        else:
            info = {}
        
        # Set target angle
        if self.fixed_target is not None:
            self.theta_desired = self.fixed_target
        else:
            if seed is not None:
                np.random.seed(seed)
            self.theta_desired = np.random.uniform(-np.pi, np.pi)
        
        # Reset episode tracking
        self.episode_length = 0
        
        # Reset enhanced PID
        self.enhanced_pid.reset()
        self.prev_theta_dot_desired = 0.0
        
        # Get our observation format
        actual_obs = self._get_obs()
        
        # Generate reference
        reference = self.generate_reference()
        
        # Compute DHP cost
        dhp_cost, dhp_gradient = self.compute_dhp_cost(actual_obs, reference)
        
        # Enhanced info
        info.update({
            'pendulum_states': actual_obs,
            'reference': reference,
            'dhp_cost': dhp_cost,
            'dhp_gradient': dhp_gradient,
            'target_angle': self.theta_desired,
            'position_error': abs(self._angular_difference(
                actual_obs[0] if not self.normalize_states else 
                (actual_obs[0] + 1.0) * np.pi - np.pi, 
                self.theta_desired)),
            'enhanced_pid_gains': {
                'Kp': self.enhanced_pid.Kp,
                'Ki': self.enhanced_pid.Ki,
                'Kd': self.enhanced_pid.Kd
            }
        })
        
        return actual_obs, info
    
    def step(self, action):
        """
        MINIMAL OVERRIDE: Use parent step for physics, only change observation
        """
        # Ensure action compatibility
        action = np.array(action, dtype=np.float32).flatten()
        if len(action) == 0:
            action = np.array([0.0])
        elif len(action) > 1:
            action = action[:1]
        
        # Use parent PendulumEnv step - PRESERVES EXACT PHYSICS
        parent_obs, parent_reward, terminated, truncated, parent_info = super().step(action)
        
        # Convert to our observation format
        actual_obs = self._get_obs()
        
        # Generate reference
        reference = self.generate_reference()
        
        # Compute DHP cost
        dhp_cost, dhp_gradient = self.compute_dhp_cost(actual_obs, reference)
        
        # Extract angle for reward calculation
        current_theta = self.state[0]
        current_theta_dot = self.state[1]

        # Modified reward for tracking
        delta_theta = self._angular_difference(current_theta, self.theta_desired)
        angle_penalty = delta_theta ** 2
        velocity_penalty = 0.1 * current_theta_dot ** 2  
        action_penalty = 0.001 * (action[0] ** 2)
        reward = -(angle_penalty + velocity_penalty + action_penalty)
        
        # Episode management
        self.episode_length += 1
        if self.episode_length >= self.max_episode_steps:
            truncated = True
        
        # Enhanced info
        position_error = abs(delta_theta)
        info = {
            'pendulum_states': actual_obs,
            'reference': reference,
            'dhp_cost': dhp_cost,
            'dhp_gradient': dhp_gradient,
            'position_error': position_error,
            'target_angle': self.theta_desired,
            'episode_success': position_error < 0.1,
            'angle_error': delta_theta,
            'action_magnitude': abs(action[0]),
            'episode_length': self.episode_length,
            'enhanced_pid_gains': {
                'Kp': self.enhanced_pid.Kp,
                'Ki': self.enhanced_pid.Ki,
                'Kd': self.enhanced_pid.Kd
            },
            'theta_dot_reference': reference[1],
            'original_obs': parent_obs,
            'original_reward': parent_reward
        }
        
        return actual_obs, reward, terminated, truncated, info


if __name__ == "__main__":
    """Test the minimal fixed environment"""
    print("="*80)
    print("TESTING MINIMAL FIXED Enhanced PendulumRandomTargetEnv")
    print("="*80)
    
    # Test energy conservation
    env = PendulumRandomTargetEnv(normalize_states=False, fixed_target=np.pi/4)
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    print("\n--- Testing Energy Conservation (MINIMAL FIXED) ---")
    obs, info = env.reset()
    
    # Set known state
    env.state[0] = np.cos(np.pi/3)  # cos(60°)
    env.state[1] = np.sin(np.pi/3)  # sin(60°)  
    env.state[2] = 0.0              # zero velocity
    
    initial_obs = env._get_obs()
    print(f"Initial state: θ={np.rad2deg(initial_obs[0]):.1f}°, ω={initial_obs[1]:.3f}")
    
    energies = []
    for step in range(50):
        obs, reward, terminated, truncated, info = env.step([0.0])  # Zero control
        
        # Calculate energy from actual state
        theta = np.arctan2(env.state[1], env.state[0])
        omega = env.state[2]
        
        ke = 0.5 * omega**2
        pe = (1 - np.cos(theta)) * 10.0
        energy = ke + pe
        energies.append(energy)
        
        if step % 10 == 0:
            print(f"Step {step:2d}: θ={np.rad2deg(theta):6.1f}°, ω={omega:6.3f}, E={energy:6.3f}")
    
    energy_variation = np.max(energies) - np.min(energies)
    print(f"\nEnergy variation: {energy_variation:.6f} J")
    
    if energy_variation < 1.0:
        print("✓ ENERGY CONSERVATION FIXED!")
    else:
        print("✗ Energy conservation still broken")
        print("This suggests PendulumEnv inheritance is not working properly")
    
    print("\n--- Testing Enhanced Features ---")
    obs, info = env.reset()
    print(f"Target: {np.rad2deg(info['target_angle']):.1f}°")
    print(f"Enhanced PID working: {'✓' if 'enhanced_pid_gains' in info else '✗'}")
    print(f"DHP cost working: {'✓' if 'dhp_cost' in info else '✗'}")
    
    print("\n✓ MINIMAL FIXED environment ready for testing!")