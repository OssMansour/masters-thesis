"""
Enhanced Pendulum Random Target Environment for DHP vs SAC Comparison Study

This environment extends the classic Pendulum-v1 to support both DHP and SAC algorithms
with random target tracking, following the same architectural principles as our
successful CF2X quadrotor implementation.

ENHANCEMENT: Proper PID reference generation for theta_dot following the sophisticated 
approach from the training script - using PID to generate force, then converting to 
acceleration and integrating to get velocity reference.

Key Features:
- Random target angle per episode (prevents overfitting)
- State normalization system (critical for DHP success)
- Gradient transformation (essential for DHP learning)
- Clean separation of states and references (matches CF2X design)
- DHP cost function with proper angular error handling
- SAC-compatible reward structure
- ENHANCED: Proper PID controller for theta_dot reference generation

Author: DHP vs SAC Comparison Study
Date: August 15, 2025
"""

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium import spaces


class EnhancedAdvancedAdaptivePID:
    """
    Enhanced adaptive PID controller based on the training script implementation.
    This version uses proper physics: PID -> Force -> Acceleration -> Velocity
    """
    
    def __init__(self, 
                 Kp=3.0, Ki=0.08, Kd=0.12, dt=0.02,
                 pendulum_params=None,
                 control_weight=0.1,
                 adapt_rate_p=0.001, adapt_rate_i=0.001, adapt_rate_d=0.001,
                 epsilon=1e-4, gamma=0.9):
        self.dt = dt
        self.Kp = float(Kp)
        self.Ki = float(Ki) 
        self.Kd = float(Kd)
        self.Kpo = float(Kp)
        self.Kio = float(Ki)
        self.Kdo = float(Kd)
        
        # Pendulum physical parameters (for force->acceleration conversion)
        if pendulum_params is None:
            self.pendulum_params = {
                'm': 1.0,    # mass [kg]
                'L': 1.0,    # length [m] 
                'g': 10,   # gravity [m/s²]
                'b': 0.0     # damping coefficient
            }
        else:
            self.pendulum_params = pendulum_params
            
        # Cost function parameter: weight on control effort
        self.control_weight = control_weight
        
        # Adaptation rates for each gain
        self.adapt_rate_p = adapt_rate_p
        self.adapt_rate_i = adapt_rate_i
        self.adapt_rate_d = adapt_rate_d
        self.epsilon = epsilon
        
        # Smoothing factor for cost (exponential moving average)
        self.gamma = gamma
        self.smoothed_cost = None
        
        # PID internal states
        self.integral = 0.0
        self.last_error = 0.0
        
    def reset(self):
        """Reset the PID internal states and smoothed cost."""
        self.Kp = float(self.Kpo)
        self.Ki = float(self.Kio)
        self.Kd = float(self.Kdo)
        self.integral = 0.0
        self.last_error = 0.0
        self.smoothed_cost = None
        
    def compute_control_force(self, error):
        """
        Compute the PID control force output given the current error.
        This follows the training script approach: PID -> Force
        """
        derivative = (error - self.last_error) / self.dt
        self.integral += error * self.dt
        
        # PID force output (not velocity!)
        force_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        return force_output, derivative
        
    def force_to_acceleration(self, force):
        """
        Convert PID force to angular acceleration using pendulum dynamics.
        This is the key enhancement from the training script approach.
        """
        # From pendulum dynamics: τ = I*α where I = m*L²
        # So: α = τ / (m*L²)
        moment_of_inertia = self.pendulum_params['m'] * (self.pendulum_params['L'] ** 2)
        angular_acceleration = force / moment_of_inertia
        return angular_acceleration
        
    def acceleration_to_velocity_reference(self, angular_acceleration):
        """
        Integrate angular acceleration to get velocity reference.
        This matches the training script physics approach.
        """
        # Integrate acceleration to get velocity change
        velocity_change = angular_acceleration * self.dt
        return velocity_change
        
    def cost_function(self, error, output):
        """
        Define the instantaneous cost as a combination of squared error and control effort.
        """
        return error**2 + self.control_weight * output**2
        
    def update_gains(self, error, integral, derivative):
        """
        Estimate the gradient of the cost with respect to each gain using finite differences,
        and update the gains with a gradient-descent-like rule.
        """
        # Current control output and cost
        current_output = self.Kp * error + self.Ki * integral + self.Kd * derivative
        cost = self.cost_function(error, current_output)
        
        # Smooth the cost with exponential moving average
        if self.smoothed_cost is None:
            self.smoothed_cost = cost
        else:
            self.smoothed_cost = self.gamma * self.smoothed_cost + (1 - self.gamma) * cost
            
        # Finite-difference gradient for Kp
        perturbed_Kp = self.Kp + self.epsilon
        output_perturbed = perturbed_Kp * error + self.Ki * integral + self.Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Kp = (cost_perturbed - cost) / self.epsilon
        
        # Finite-difference gradient for Ki
        perturbed_Ki = self.Ki + self.epsilon
        output_perturbed = self.Kp * error + perturbed_Ki * integral + self.Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Ki = (cost_perturbed - cost) / self.epsilon
        
        # Finite-difference gradient for Kd
        perturbed_Kd = self.Kd + self.epsilon
        output_perturbed = self.Kp * error + self.Ki * integral + perturbed_Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Kd = (cost_perturbed - cost) / self.epsilon
        
        # Update gains using a gradient descent step
        self.Kp -= self.adapt_rate_p * grad_Kp
        self.Ki -= self.adapt_rate_i * grad_Ki
        self.Kd -= self.adapt_rate_d * grad_Kd
        
        # Clip gains to avoid negative values and keep them within reasonable bounds
        self.Kp = max(0.01, min(self.Kp, 10.0))
        self.Ki = max(0.0, min(self.Ki, 5.0))
        self.Kd = max(0.0, min(self.Kd, 5.0))
        
    def __call__(self, error):
        """
        Enhanced PID call following training script approach:
        Error -> PID Force -> Acceleration -> Velocity Reference
        
        This is the key improvement over the simple PID approach.
        """
        # Compute control force and derivative term
        force_output, derivative = self.compute_control_force(error)
        
        # Update gains using the current error, accumulated integral, and derivative
        self.update_gains(error, self.integral, derivative)
        
        # ENHANCEMENT: Convert force to acceleration using pendulum physics
        angular_acceleration = self.force_to_acceleration(force_output)
        
        # ENHANCEMENT: Integrate acceleration to get velocity reference
        velocity_reference = self.acceleration_to_velocity_reference(angular_acceleration)
        
        # Update last_error for next derivative computation
        self.last_error = error
        
        return velocity_reference


class PendulumRandomTargetEnv(PendulumEnv):
    """
    Enhanced Pendulum environment with random target tracking for DHP vs SAC comparison.
    
    ENHANCEMENT: Proper PID reference generation following training script approach.
    COMPATIBILITY: Maintains original PendulumEnv physics and timing.
    
    Architecture matches CF2X_FastStates_HoverAviary:
    - Observation: States [theta, theta_dot] (converted from original [cos, sin, thetadot])
    - Reference: Generated separately via generate_reference()
    - DHP Interface: compute_dhp_cost(state, reference)
    - State normalization and gradient transformation included
    - ENHANCED: Physics-based PID for theta_dot reference
    - COMPATIBLE: Uses original PendulumEnv physics integration
    """
    
    def __init__(self, 
                 gui=False,
                 record=False,
                 fixed_target=None,
                 normalize_states=True,
                 render_mode="human",
                 pendulum_params=None):
        """
        Initialize Enhanced Pendulum Random Target Environment
        
        Args:
            gui: Whether to render environment (for compatibility)
            record: Whether to record episodes (for compatibility)
            fixed_target: Fixed target angle for repeatable experiments (None for random)
            normalize_states: Whether to apply state normalization (recommended: True)
            render_mode: Gym/Gymnasium render mode (e.g., "human")
            pendulum_params: Physical parameters for enhanced PID controller
        """
        super().__init__(render_mode=render_mode if gui else None)
        
        # Physical parameters for enhanced PID (match original PendulumEnv exactly)
        if pendulum_params is None:
            self.pendulum_params = {
                'm': 1.0,    # mass [kg] - matches original PendulumEnv
                'L': 1.0,    # length [m] - matches original PendulumEnv  
                'g': 10.0,   # gravity [m/s²] - matches original PendulumEnv default
                'b': 0.1     # damping coefficient (not in original, but needed for PID)
            }
        else:
            self.pendulum_params = pendulum_params
            
        # Initialize enhanced PID controller (with original dt timing)
        self._init_enhanced_pid()
        
        # Store original physics parameters for consistency
        self.dt = 0.05  # Match original PendulumEnv timestep
        self.g = self.pendulum_params['g']  # Match gravity setting
        self.m = self.pendulum_params['m']  # Match mass setting
        self.l = self.pendulum_params['L']  # Match length setting
        
        # Environment configuration
        self.gui = gui
        self.record = record
        self.fixed_target = fixed_target
        self.normalize_states = normalize_states
        self.theta_desired = 0.0
        
        # Initialize reference rate limiting
        self.prev_theta_dot_desired = 0.0
        
        # State bounds for normalization (following CF2X approach)
        self.state_bounds = {
            'theta': [-np.pi, np.pi],     # Angle bounds [-π, π]
            'theta_dot': [-8.0, 8.0]      # Angular velocity limit (pendulum max_speed)
        }
        
        # Observation space: ONLY actual pendulum states (like CF2X)
        if self.normalize_states:
            # Normalized state space [-1, 1] for all states
            obs_low = np.array([-1.0, -1.0], dtype=np.float32)
            obs_high = np.array([1.0, 1.0], dtype=np.float32)
        else:
            # Raw state space
            obs_low = np.array([-np.pi, -self.max_speed], dtype=np.float32)
            obs_high = np.array([np.pi, self.max_speed], dtype=np.float32)
            
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # DHP cost function weights for [theta, theta_dot] state representation
        # Cost computed on [theta_error, theta_dot_error] for direct pendulum control
        self.Q_matrix = np.diag([
            10.0,      # theta_error - angle tracking (high importance)
            100.0       # theta_dot_error - velocity damping (lower weight)
        ])
        
        # Episode statistics (match original episode length)
        self.episode_length = 0
        self.max_episode_steps = 200  # Match original PendulumEnv truncation
        
        print(f"[INFO] Enhanced PendulumRandomTargetEnv initialized")
        print(f"[INFO] Observation space: {self.observation_space}")
        print(f"[INFO] Action space: {self.action_space}")
        print(f"[INFO] State normalization: {self.normalize_states}")
        print(f"[INFO] States: [theta, theta_dot]")
        print(f"[INFO] ENHANCED: Physics-based PID for theta_dot reference")
        print(f"[INFO] Reference generated separately (matches CF2X design)")
        print(f"[INFO] DHP cost weights: theta_error={self.Q_matrix[0,0]}, theta_dot_error={self.Q_matrix[1,1]}")
    
    def _init_enhanced_pid(self):
        """Initialize the enhanced PID controller with original PendulumEnv timing"""
        self.enhanced_pid = EnhancedAdvancedAdaptivePID(
            Kp=1.5,    # Reduced from 3.0 for smoother response
            Ki=0.04,   # Reduced from 0.08 for less aggressive integral action
            Kd=0.06,   # Reduced from 0.12 for less aggressive derivative action
            dt=0.05,   # Match original PendulumEnv timestep
            pendulum_params=self.pendulum_params,
            control_weight=0.1,
            adapt_rate_p=0.0005,  # Reduced adaptation rates for stability
            adapt_rate_i=0.0005, 
            adapt_rate_d=0.0005,
            epsilon=1e-4, 
            gamma=0.9
        )
    
    def normalize_observation(self, obs):
        """
        Normalize observation states to [-1, 1] range (critical for DHP success)
        
        Args:
            obs: Raw observation [theta, theta_dot]
            
        Returns:
            Normalized observation with all values in [-1, 1]
        """
        if not self.normalize_states:
            return obs
            
        normalized = np.zeros_like(obs)
        bounds = [
            self.state_bounds['theta'],      # [-π, π]
            self.state_bounds['theta_dot']   # [-8, 8]
        ]
        
        for i, (low, high) in enumerate(bounds):
            # Normalize to [-1, 1]: 2 * (x - low) / (high - low) - 1
            normalized[i] = 2.0 * (obs[i] - low) / (high - low) - 1.0
            # Clamp to bounds for safety
            normalized[i] = np.clip(normalized[i], -1.0, 1.0)
            
        return normalized.astype(np.float32)
    
    def _angular_difference(self, a, b):
        """
        Compute shortest angular difference between two angles (like CF2X)
        
        Args:
            a, b: Angles in radians
            
        Returns:
            Shortest angular difference (a - b) wrapped to [-π, π]
        """
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return diff
    
    def generate_reference(self):
        """
        ENHANCED: Generate reference signal using physics-based PID approach.
        
        This follows the training script method:
        1. Compute angle error
        2. PID controller generates FORCE (not velocity)
        3. Convert force to angular acceleration using pendulum physics
        4. Integrate acceleration to get velocity reference
        
        Returns:
            Reference vector [theta_desired, theta_dot_desired]
        """
        # Get current angle from pendulum state
        theta = self.state[0]  # Current angle from pendulum state
        
        # Compute angle error (wrap to [-pi, pi])
        theta_error = self._angular_difference(self.theta_desired, theta)
        
        # ENHANCEMENT: Use physics-based PID to get velocity reference
        # This follows training script approach: Error -> Force -> Acceleration -> Velocity
        theta_dot_desired = float(self.enhanced_pid(theta_error))
        
        # FIXED: Clamp velocity reference to reasonable bounds (much lower limit)
        max_theta_dot_ref = 1.0  # reduced from 2π to 1 rad/s for smooth control
        theta_dot_desired = np.clip(theta_dot_desired, -max_theta_dot_ref, max_theta_dot_ref)
        
        # FIXED: Add rate limiting to prevent discontinuous jumps
        if hasattr(self, 'prev_theta_dot_desired'):
            max_rate = 5.0  # Maximum change: 5 rad/s² 
            dt = self.dt  # Use actual environment timestep (0.05s)
            max_change = max_rate * dt  # 0.25 rad/s per step
            
            change = theta_dot_desired - self.prev_theta_dot_desired
            if abs(change) > max_change:
                theta_dot_desired = self.prev_theta_dot_desired + np.sign(change) * max_change
        
        # Store for next iteration
        self.prev_theta_dot_desired = theta_dot_desired
        
        # Reference vector in same format as states
        reference = np.array([
            self.theta_desired,    # theta_desired
            theta_dot_desired      # theta_dot_desired (from enhanced PID)
        ], dtype=np.float32)
        
        # Apply same normalization as states for consistency
        if self.normalize_states:
            reference = self.normalize_observation(reference)
            
        return reference
    
    def compute_dhp_cost(self, state, reference):
        """
        Compute DHP quadratic cost function with gradient transformation
        
        Args:
            state: Current pendulum states [theta, theta_dot]
            reference: Reference states [theta_des, theta_dot_des]
            
        Returns:
            tuple: (cost_scalar, cost_gradient_array)
            
        Note: Follows same structure as CF2X compute_dhp_cost for consistency
        """
        # Ensure inputs are numpy arrays
        state = np.asarray(state, dtype=np.float64).flatten()
        reference = np.asarray(reference, dtype=np.float64).flatten()
        
        # Check for NaN or infinite values (robust like CF2X)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Warning: Invalid state values detected: {state}")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if np.any(np.isnan(reference)) or np.any(np.isinf(reference)):
            print(f"Warning: Invalid reference values detected: {reference}")
            reference = np.nan_to_num(reference, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract states directly (much simpler now!)
        theta, theta_dot = state[0], state[1]
        theta_ref, theta_dot_ref = reference[0], reference[1]
        
        # Compute tracking errors (like CF2X angular error handling)
        theta_error = self._angular_difference(theta, theta_ref)
        theta_dot_error = theta_dot - theta_dot_ref
        
        # Error vector for quadratic cost
        error = np.array([theta_error, theta_dot_error])
        
        # Quadratic cost: J = e^T * Q * e (same as CF2X)
        cost = error.T @ self.Q_matrix @ error
        
        # Cost gradient w.r.t. [theta_error, theta_dot_error]
        dcost_derror = 2.0 * self.Q_matrix @ error
        
        # Transform gradient from [theta_error, theta_dot_error] to [theta, theta_dot]
        # For theta: dcost/dtheta = dcost/dtheta_error (direct since theta_error = theta - theta_ref)
        # For theta_dot: dcost/dtheta_dot = dcost/dtheta_dot_error (direct since theta_dot_error = theta_dot - theta_dot_ref)
        dcostdx = np.array([
            dcost_derror[0],  # dcost/dtheta
            dcost_derror[1]   # dcost/dtheta_dot
        ])
        
        # Apply gradient transformation for normalized states (critical for DHP)
        if self.normalize_states:
            bounds = [
                self.state_bounds['theta'],      # [-π, π]
                self.state_bounds['theta_dot']   # [-8, 8]
            ]
            
            dcostdx_normalized = np.zeros_like(dcostdx)
            for i, (low, high) in enumerate(bounds):
                # Transform gradient for normalized coordinates
                dcostdx_normalized[i] = dcostdx[i] * 2.0 / (high - low)
            
            dcostdx = dcostdx_normalized
        
        # Check for numerical issues (robust like CF2X)
        if np.any(np.isnan(dcostdx)) or np.any(np.isinf(dcostdx)):
            print(f"Warning: Invalid cost gradient detected, using zero gradient")
            dcostdx = np.zeros_like(dcostdx)
        
        # Extract scalar cost value
        if np.ndim(cost) == 0:
            cost_scalar = float(cost)
        elif np.size(cost) == 1:
            cost_scalar = float(cost.flat[0])
        else:
            print(f"Warning: cost has unexpected shape {cost.shape}, using first element")
            cost_scalar = float(cost.flat[0])
        
        # Ensure cost is finite
        if not np.isfinite(cost_scalar):
            print(f"Warning: Non-finite cost {cost_scalar}, setting to 1000.0")
            cost_scalar = 1000.0
        
        return cost_scalar, dcostdx.astype(np.float32)
    
    def _get_obs(self):
        """
        Get current pendulum observation (actual states only, like CF2X)
        
        Returns:
            Observation [theta, theta_dot]
        """
        theta = self.state[0]
        theta_dot = self.state[1]
        
        # Wrap theta to [-π, π] for consistency
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        obs = np.array([
            theta,
            theta_dot
        ], dtype=np.float32)
        
        # Apply state normalization
        if self.normalize_states:
            obs = self.normalize_observation(obs)
            
        return obs
    
    def reset(self, *, seed=None, options=None):
        """
        Reset environment with new random target (or fixed target for experiments)
        
        Returns:
            tuple: (observation, info_dict)
        """
        # Reset pendulum to random initial state
        obs_tuple = super().reset(seed=seed)
        if isinstance(obs_tuple, tuple):
            obs, info = obs_tuple
        else:
            obs, info = obs_tuple, {}
        
        # Set target angle for this episode
        if self.fixed_target is not None:
            self.theta_desired = self.fixed_target
        else:
            # Random target angle (prevents overfitting)
            np.random.seed(seed)
            self.theta_desired = np.random.uniform(-np.pi, np.pi)
        
        # Reset episode statistics
        self.episode_length = 0

        # Reset enhanced PID controller state
        self._init_enhanced_pid()
        
        # Reset reference rate limiting
        self.prev_theta_dot_desired = 0.0
        
        # Get actual observation (normalized states only)
        actual_obs = self._get_obs()
        
        # Generate initial reference using enhanced PID
        reference = self.generate_reference()
        
        # Compute initial DHP cost
        dhp_cost, dhp_gradient = self.compute_dhp_cost(actual_obs, reference)
        
        # Add DHP-specific information (matches CF2X info structure)
        info.update({
            'pendulum_states': actual_obs,
            'reference': reference,
            'dhp_cost': dhp_cost,
            'dhp_gradient': dhp_gradient,
            'target_angle': self.theta_desired,
            'position_error': abs(self._angular_difference(actual_obs[0] if not self.normalize_states else 
                                                           # Denormalize theta for error calculation
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
        Execute one step in environment using original PendulumEnv physics
        Args:
            action: Control torque (scalar or 1D array)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure action is compatible with original PendulumEnv (expects array)
        action = np.array(action, dtype=np.float32).flatten()
        if len(action) == 0:
            action = np.array([0.0])
        elif len(action) > 1:
            action = action[:1]  # Take only first element
        
        # Execute original pendulum physics step
        orig_obs, orig_reward, terminated, truncated, orig_info = super().step(action)
        
        # Get our custom observation (2D state representation)
        actual_obs = self._get_obs()
        
        # Generate reference for current step using enhanced PID
        reference = self.generate_reference()
        
        # Compute DHP cost and gradient
        dhp_cost, dhp_gradient = self.compute_dhp_cost(actual_obs, reference)
        
        # Compute tracking error and reward
        theta = self.state[0]
        theta_dot = self.state[1]
        
        # Angular error to target (wrapped to [-π, π])
        delta_theta = self._angular_difference(theta, self.theta_desired)
        
        # Use original reward as base but modify for target tracking
        # Original: -(theta^2 + 0.1*thetadot^2 + 0.001*u^2) where theta is angle from upright
        # Modified: -(delta_theta^2 + 0.1*thetadot^2 + 0.001*u^2) where delta_theta is error from target
        angle_penalty = delta_theta ** 2
        velocity_penalty = 0.1 * theta_dot ** 2
        action_penalty = 0.001 * (action[0] ** 2)
        reward = -(angle_penalty + velocity_penalty + action_penalty)
        
        # Episode termination (use original logic but with our episode length)
        self.episode_length += 1
        if self.episode_length >= self.max_episode_steps:
            truncated = True
        
        # Success criteria (for analysis)
        position_error = abs(delta_theta)
        episode_success = position_error < 0.1  # 0.1 rad ≈ 5.7° tolerance
        
        # Update info dict with DHP and analysis information (matches CF2X structure)
        info = {
            'pendulum_states': actual_obs,
            'reference': reference,
            'dhp_cost': dhp_cost,
            'dhp_gradient': dhp_gradient,
            'position_error': position_error,
            'target_angle': self.theta_desired,
            'episode_success': episode_success,
            'angle_error': delta_theta,
            'action_magnitude': abs(action[0]),
            'episode_length': self.episode_length,
            'enhanced_pid_gains': {
                'Kp': self.enhanced_pid.Kp,
                'Ki': self.enhanced_pid.Ki,
                'Kd': self.enhanced_pid.Kd
            },
            'theta_dot_reference': reference[1],  # For debugging enhanced PID
            'original_obs': orig_obs,  # Keep original observation for debugging
            'original_reward': orig_reward  # Keep original reward for comparison
        }
        
        return actual_obs, reward, terminated, truncated, info


def register_env():
    """
    Register enhanced environment with Gym for easy instantiation
    """
    gym.envs.registration.register(
        id='PendulumRandomTargetEnhanced-v0',
        entry_point='pendulum_env_enhanced:PendulumRandomTargetEnv',
        max_episode_steps=200,  # Match original PendulumEnv
        kwargs={'normalize_states': True}
    )
    print("[INFO] PendulumRandomTargetEnhanced-v0 registered with Gym")


if __name__ == "__main__":
    """
    Test the enhanced environment to verify functionality
    """
    print("="*80)
    print("TESTING ENHANCED PendulumRandomTargetEnv")
    print("="*80)
    
    # Test environment creation
    env = PendulumRandomTargetEnv(
        normalize_states=True,
        fixed_target=np.pi/4  # 45 degrees for testing
    )
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test episode
    print("\n--- Testing Enhanced PID Episode ---")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial reference: {info['reference']}")
    print(f"Initial DHP cost: {info['dhp_cost']:.4f}")
    print(f"Target angle: {info['target_angle']:.4f} rad ({np.rad2deg(info['target_angle']):.1f}°)")
    print(f"Enhanced PID gains: {info['enhanced_pid_gains']}")
    
    # Test several steps
    print("\nStep-by-step enhanced PID testing:")
    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1:2d}: "
              f"cost={info['dhp_cost']:8.4f}, "
              f"error={info['position_error']:6.4f} rad, "
              f"reward={reward:8.4f}, "
              f"action={action:6.3f}, "
              f"θ̇_ref={info['theta_dot_reference']:6.3f}")
        
        if terminated or truncated:
            print("Episode ended")
            break
    
    # Test enhanced PID physics
    print("\n--- Testing Enhanced PID Physics ---")
    test_errors = [-np.pi/2, -np.pi/4, 0.0, np.pi/4, np.pi/2]
    env.enhanced_pid.reset()
    
    for error in test_errors:
        velocity_ref = env.enhanced_pid(error)
        print(f"Angle error: {error:6.3f} rad ({np.rad2deg(error):6.1f}°) "
              f"-> Velocity ref: {velocity_ref:8.4f} rad/s")
    
    # Test normalization
    print("\n--- Testing State Normalization ---")
    raw_obs = np.array([0.5, 0.866, 4.0])  # cos(60°), sin(60°), 4 rad/s
    normalized = env.normalize_observation(raw_obs)
    print(f"Raw observation: {raw_obs}")
    print(f"Normalized: {normalized}")
    print(f"All values in [-1,1]: {np.all(np.abs(normalized) <= 1.0)}")
    
    # Test gradient computation
    print("\n--- Testing DHP Cost and Gradient ---")
    test_state = np.array([1.0, 0.0, 0.0])  # cos(0°), sin(0°), 0 rad/s
    test_ref = np.array([0.0, 1.0, 0.0])    # cos(90°), sin(90°), 0 rad/s
    cost, gradient = env.compute_dhp_cost(test_state, test_ref)
    print(f"Test state: {test_state}")
    print(f"Test reference: {test_ref}")
    print(f"DHP cost: {cost:.4f}")
    print(f"DHP gradient: {gradient}")
    print(f"Gradient finite: {np.all(np.isfinite(gradient))}")
    