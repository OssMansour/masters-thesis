"""
DHP-Compatible Enhanced Pendulum Environment

This environment is specifically designed to work with your DHP training script
while providing proper physics simulation and reference generation.

Key Features:
1. Compatible with your existing DHP training loop
2. Proper pendulum physics with RK4 integration
3. Enhanced PID for reference generation (theta_dot_ref)
4. Normalized state space for stable training
5. Proper reward function for tracking
6. DHP cost function with gradients
7. Compatible observation/action spaces

Author: Based on your DHP implementation
Date: August 16, 2025
"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt


class EnhancedAdaptivePID:
    """
    Enhanced Adaptive PID Controller for Reference Generation
    
    This generates theta_dot_reference from theta_error using proper pendulum physics.
    Based on your AdvancedAdaptivePID but optimized for reference generation.
    """
    
    def __init__(self, Kp=2.0, Ki=0.05, Kd=0.8, dt=0.02, 
                 pendulum_params=None, control_weight=0.1,
                 adapt_rate_p=0.0005, adapt_rate_i=0.0005, adapt_rate_d=0.0005,
                 epsilon=1e-4, gamma=0.9):
        
        self.dt = dt
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.Kpo = float(Kp)  # Original gains for reset
        self.Kio = float(Ki)
        self.Kdo = float(Kd)
        
        # Pendulum parameters for physics-based reference generation
        if pendulum_params is None:
            self.pendulum_params = {
                'm': 0.1,    # mass [kg] - match your pendulum
                'L': 0.5,    # length [m] - match your pendulum
                'g': 9.81,   # gravity [m/s²] - match your pendulum
                'b': 0.3     # damping [N⋅s/m] - match your pendulum
            }
        else:
            self.pendulum_params = pendulum_params
        
        # PID adaptation parameters
        self.control_weight = control_weight
        self.adapt_rate_p = adapt_rate_p
        self.adapt_rate_i = adapt_rate_i
        self.adapt_rate_d = adapt_rate_d
        self.epsilon = epsilon
        self.gamma = gamma
        self.smoothed_cost = None
        
        # PID internal states
        self.integral = 0.0
        self.last_error = 0.0
        
        # Reference limiting
        self.max_velocity_ref = 2.0 * np.pi  # Maximum velocity reference
        self.max_rate_change = 5.0  # Maximum rate of change for velocity reference
        self.prev_velocity_ref = 0.0
        
    def reset(self):
        """Reset PID internal states and gains"""
        self.Kp = float(self.Kpo)
        self.Ki = float(self.Kio)
        self.Kd = float(self.Kdo)
        self.integral = 0.0
        self.last_error = 0.0
        self.smoothed_cost = None
        self.prev_velocity_ref = 0.0
        
    def compute_control_force(self, error):
        """Compute PID control force output"""
        derivative = (error - self.last_error) / self.dt
        self.integral += error * self.dt
        
        # Anti-windup: limit integral accumulation
        max_integral = 2.0 / (self.Ki + 1e-6)
        self.integral = np.clip(self.integral, -max_integral, max_integral)

        force_output = np.clip(self.Kp * error + self.Ki * self.integral + self.Kd * derivative,-4.0,4.0)
        return force_output, derivative
        
    def force_to_acceleration(self, force):
        """Convert control force to angular acceleration using pendulum dynamics"""
        # For pendulum: τ = I*α where I = m*L²
        moment_of_inertia = self.pendulum_params['m'] * (self.pendulum_params['L'] ** 2)
        angular_acceleration = force / moment_of_inertia
        return angular_acceleration
        
    def acceleration_to_velocity_reference(self, angular_acceleration):
        """Integrate angular acceleration to get velocity reference"""
        velocity_change = angular_acceleration * self.dt
        return velocity_change
        
    def cost_function(self, error, output):
        """Cost function for gain adaptation"""
        return error**2 + self.control_weight * output**2
        
    def update_gains(self, error, integral, derivative):
        """Update PID gains using gradient descent on cost function"""
        current_output = self.Kp * error + self.Ki * integral + self.Kd * derivative
        cost = self.cost_function(error, current_output)
        
        # Smooth cost with exponential moving average
        if self.smoothed_cost is None:
            self.smoothed_cost = cost
        else:
            self.smoothed_cost = self.gamma * self.smoothed_cost + (1 - self.gamma) * cost
            
        # Finite-difference gradients
        # Gradient w.r.t. Kp
        perturbed_Kp = self.Kp + self.epsilon
        output_perturbed = perturbed_Kp * error + self.Ki * integral + self.Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Kp = (cost_perturbed - cost) / self.epsilon
        
        # Gradient w.r.t. Ki
        perturbed_Ki = self.Ki + self.epsilon
        output_perturbed = self.Kp * error + perturbed_Ki * integral + self.Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Ki = (cost_perturbed - cost) / self.epsilon
        
        # Gradient w.r.t. Kd
        perturbed_Kd = self.Kd + self.epsilon
        output_perturbed = self.Kp * error + self.Ki * integral + perturbed_Kd * derivative
        cost_perturbed = self.cost_function(error, output_perturbed)
        grad_Kd = (cost_perturbed - cost) / self.epsilon
        
        # Update gains
        self.Kp -= self.adapt_rate_p * grad_Kp
        self.Ki -= self.adapt_rate_i * grad_Ki
        self.Kd -= self.adapt_rate_d * grad_Kd
        
        # Clip gains to reasonable bounds
        self.Kp = max(0.1, min(self.Kp, 10.0))
        self.Ki = max(0.0, min(self.Ki, 2.0))
        self.Kd = max(0.0, min(self.Kd, 5.0))
        
    def __call__(self, angle_error):
        """
        Generate velocity reference from angle error
        
        Args:
            angle_error: Angular error [rad] (theta_desired - theta_current)
            
        Returns:
            velocity_reference: Desired angular velocity [rad/s]
        """
        # Compute control force and derivative
        force_output, derivative = self.compute_control_force(angle_error)
        
        # Update gains based on performance
        self.update_gains(angle_error, self.integral, derivative)
        
        # Convert force to acceleration using pendulum physics
        angular_acceleration = self.force_to_acceleration(force_output)
        
        # Integrate to get velocity change
        velocity_change = self.acceleration_to_velocity_reference(angular_acceleration)
        
        # Add to previous velocity reference (integration)
        velocity_reference = self.prev_velocity_ref + velocity_change
        
        # Apply velocity limits
        velocity_reference = np.clip(velocity_reference, -self.max_velocity_ref, self.max_velocity_ref)
        
        # Apply rate limiting to prevent discontinuous jumps
        max_change = self.max_rate_change * self.dt
        if abs(velocity_reference - self.prev_velocity_ref) > max_change:
            velocity_reference = self.prev_velocity_ref + np.sign(velocity_reference - self.prev_velocity_ref) * max_change
        
        # Update for next iteration
        self.last_error = angle_error
        self.prev_velocity_ref = velocity_reference
        
        return float(velocity_reference)


class DHPCompatiblePendulumEnv(gym.Env):
    """
    DHP-Compatible Enhanced Pendulum Environment
    
    Designed to work seamlessly with your existing DHP training script while
    providing proper physics and reference generation.
    """

    def __init__(self, m=0.1, g=9.81, L=0.5, b=0.3, Fmax=4.0, dt=0.02,
                 normalize_states=True, max_episode_steps=200):
        
        super(DHPCompatiblePendulumEnv, self).__init__()
        
        # Physical parameters (match your Pendulum class exactly)
        self.m = m          # mass [kg]
        self.g = g          # gravity [m/s²]
        self.L = L          # length [m]
        self.b = b          # damping [N⋅s/m]
        self.Fmax = Fmax    # maximum force [N]
        self.dt = dt        # time step [s]
        
        # State variables
        self.x1 = 0.0       # theta [rad]
        self.x2 = 0.0       # theta_dot [rad/s]
        self.t = 0.0        # time [s]
        self.steps = 0      # step counter
        
        # Environment configuration
        self.normalize_states = normalize_states
        self.max_episode_steps = max_episode_steps
        
        # State bounds for normalization (match original PendulumEnv)
        self.max_angle = np.pi
        self.max_velocity = 8.0  # Match original PendulumEnv max_speed
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        if normalize_states:
            # Normalized state space [-1, 1]
            self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            # Raw state space [theta, theta_dot]
            self.observation_space = spaces.Box(
                low=np.array([-self.max_angle, -self.max_velocity]), 
                high=np.array([self.max_angle, self.max_velocity]), 
                shape=(2,), dtype=np.float32
            )
        
        # Enhanced PID for reference generation
        pendulum_params = {
            'm': self.m,
            'L': self.L, 
            'g': self.g,
            'b': self.b
        }
        self.enhanced_pid = EnhancedAdaptivePID(
            Kp=2.0, Ki=0.05, Kd=0.8, dt=self.dt,
            pendulum_params=pendulum_params
        )
        
        # Trajectory and reference tracking
        self.theta_desired = 0.0
        self.velocity_desired = 0.0
        self.desired_traj = []
        self.actions_history = [0]
        
        # DHP cost function weights (IMPROVED for velocity control)
        # Higher velocity weight to encourage stopping at target
        self.Q_matrix = np.diag([100.0, 1.0])  # [theta_error_weight, velocity_error_weight]
        # Note: Increased velocity weight from 0.0001 to 50.0 for better braking control

        # Random number generator
        self.np_random = None
        self.seed()
        
        print("[DHP-Compatible] Enhanced Pendulum Environment Initialized")
        print(f"[DHP-Compatible] Physics: m={m}, L={L}, g={g}, b={b}, Fmax={Fmax}")
        print(f"[DHP-Compatible] State normalization: {normalize_states}")
        print(f"[DHP-Compatible] Observation space: {self.observation_space}")
        print(f"[DHP-Compatible] Action space: {self.action_space}")
        
    def seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def _dynamics(self, state, F):
        """
        Compute pendulum dynamics using proper physics
        
        Equation of motion: θ̈ = (F/(mL)) - (g/L)sin(θ) - (b/(mL))θ̇
        
        Args:
            state: [theta, theta_dot]
            F: Applied force [N]
            
        Returns:
            state_dot: [theta_dot, theta_ddot]
        """
        theta, theta_dot = state
        F = float(F)
        
        dtheta = theta_dot
        dtheta_dot = (F / (self.m * self.L)) - ((self.g / self.L) * np.sin(theta)) - ((self.b / (self.m * self.L)) * theta_dot)
        
        return np.array([dtheta, dtheta_dot])
        
    def _integrate_rk4(self, state, F):
        """
        Integrate dynamics using 4th-order Runge-Kutta (RK4)
        
        This provides more accurate and stable integration than Euler method.
        """
        k1 = self._dynamics(state, F)
        k2 = self._dynamics(state + 0.5 * self.dt * k1, F)
        k3 = self._dynamics(state + 0.5 * self.dt * k2, F)
        k4 = self._dynamics(state + self.dt * k3, F)
        
        new_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Wrap angle to [-π, π]
        new_state[0] = ((new_state[0] + np.pi) % (2 * np.pi)) - np.pi
        
        return new_state
        
    def angular_difference(self, a, b):
        """Compute shortest angular difference (a - b) wrapped to [-π, π]"""
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return diff
        
    def normalize_observation(self, obs):
        if not self.normalize_states:
            return obs
        
        normalized = np.zeros_like(obs)
        # Proper linear normalization: [-π,π] → [-1,1] and [-max_vel,max_vel] → [-1,1]
        normalized[0] = obs[0] / np.pi  # Simple linear mapping without discontinuity
        normalized[1] = obs[1] / self.max_velocity  # Simple linear mapping
        
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
        
    def denormalize_observation(self, normalized_obs):
        """Convert normalized observation back to physical units"""
        if not self.normalize_states:
            return normalized_obs
            
        obs = np.zeros_like(normalized_obs)
        obs[0] = normalized_obs[0] * self.max_angle
        obs[1] = normalized_obs[1] * self.max_velocity
        
        return obs
        
    def get_pendulum_state(self):
        """Get current pendulum state [theta, theta_dot]"""
        # Wrap angle to [-π, π]
        theta_wrapped = ((self.x1 + np.pi) % (2 * np.pi)) - np.pi
        return np.array([float(theta_wrapped), float(self.x2)])

    def desired_trajectory(self, traj="fixed"):
        """
        Generate desired trajectory for INVERTED PENDULUM control
        
        For inverted pendulum: target is π radians (180° upright position)
        """
        if traj=="fixed" or self.steps == 0:  # Initial training phase
            self.theta_desired = np.pi  # Inverted pendulum target (upright position)
        elif traj == "random angular velocities":
            if self.steps % 50 == 0:
                self.velocity_desired = self.np_random.uniform(-0.1*np.pi, 0.1*np.pi)
            self.theta_desired += self.velocity_desired * self.dt
            self.theta_desired = ((self.theta_desired + np.pi) % (2 * np.pi)) - np.pi
        elif traj == "random points":
            if self.steps % 70 == 0:
                # Keep targets near the inverted position for stabilization
                self.theta_desired = np.pi + self.np_random.uniform(-0.3, 0.3)  # ±17° around upright
                self.theta_desired = ((self.theta_desired + np.pi) % (2 * np.pi)) - np.pi
            self.velocity_desired = 0
        elif traj == "sin traj":
            t = self.steps * self.dt
            # Sinusoidal around the inverted position
            self.theta_desired = np.pi + 0.5 * (np.sin(0.06 * np.pi * t) + np.cos(0.04 * np.pi * t))
            self.theta_desired = ((self.theta_desired + np.pi) % (2 * np.pi)) - np.pi
            self.velocity_desired = 0.5 * (0.06 * np.pi * np.cos(0.06 * np.pi * t) - 
                                               0.04 * np.pi * np.sin(0.04 * np.pi * t))
        
        return self.theta_desired
        
    def generate_reference(self):
        """
        Generate reference [theta_desired, theta_dot_desired] with SMART BRAKING
        
        Key improvements:
        1. Distance-dependent velocity reference
        2. Explicit braking when close to target
        3. Velocity direction awareness
        """
        # Get current state (raw)
        current_state = self.get_pendulum_state()
        current_theta = current_state[0]
        current_theta_dot = current_state[1]
        
        # Update desired trajectory
        self.theta_desired = self.desired_trajectory()
        
        # Compute angle error (shortest path)
        angle_error = self.angular_difference(self.theta_desired, current_theta)
        distance_to_target = abs(angle_error)
        
        # PREDICTIVE ULTRA-AGGRESSIVE BRAKING VELOCITY REFERENCE GENERATION
        # Calculate if current velocity will cause overshoot
        time_to_target = distance_to_target / max(abs(current_theta_dot), 0.1)
        stopping_distance = 0.5 * abs(current_theta_dot) * time_to_target  # Rough stopping distance
        will_overshoot = stopping_distance > distance_to_target * 0.8  # Conservative threshold
        
        if distance_to_target < 0.05:  # Extremely close (< 3°)
            # EMERGENCY STOP: Maximum counter-braking
            self.velocity_desired = -2.0 * np.sign(current_theta_dot)  # Maximum braking
            
        elif distance_to_target < 0.1:  # Very close to target (< 5.7°)
            # IMMEDIATE STOP: Strong counter-braking for any velocity
            if abs(current_theta_dot) > 0.2:  # Any significant velocity
                self.velocity_desired = -1.5 * np.sign(current_theta_dot)  # Very strong counter-rotation
            else:
                self.velocity_desired = 0.0  # Complete stop when slow
                
        elif distance_to_target < 0.3 or will_overshoot:  # Expanded braking zone or overshoot prediction
            # PREDICTIVE BRAKING: Start braking early if overshoot is predicted
            if abs(current_theta_dot) > 1.0:  # If moving fast, apply counter-braking
                self.velocity_desired = -1.2 * np.sign(current_theta_dot)  # Strong counter-rotation
            else:
                self.velocity_desired = -0.5 * np.sign(current_theta_dot)  # Moderate braking
            
        elif distance_to_target < 0.8:  # Extended approach zone (< 45.8°)
            # PROGRESSIVE BRAKING: Reduce velocity as we approach target
            # Check if we're moving towards or away from target
            moving_towards_target = (angle_error * current_theta_dot) < 0
            
            if moving_towards_target:
                # Moving towards target: very conservative approach
                max_approach_velocity = 0.5 * distance_to_target  # Much slower approach
                if abs(current_theta_dot) > max_approach_velocity:
                    # Need strong braking
                    self.velocity_desired = -0.8 * np.sign(current_theta_dot)  # Counter-rotating brake
                else:
                    # Allow very gentle approach
                    self.velocity_desired = current_theta_dot * 0.3  # Strong deceleration
            else:
                # Moving away from target: correct direction with very limited speed
                self.velocity_desired = 0.2 * np.sign(angle_error) * min(distance_to_target, 0.5)
                
        else:  # Far from target (> 45.8°)
            # Use PID for velocity reference (normal operation)
            self.velocity_desired = self.enhanced_pid(angle_error)
            
            # Limit velocity based on distance (prevent overshooting)
            max_velocity = min(1.5, 1.0 * distance_to_target)  # Much more conservative
            self.velocity_desired = np.clip(self.velocity_desired, -max_velocity, max_velocity)
        
        # STRONGER: Prevent continuous rotation with more aggressive intervention
        if abs(current_theta_dot) > 4.0:  # Reduced threshold from 6.28 to 4.0
            self.velocity_desired = -0.8 * np.sign(current_theta_dot)  # Strong counter-rotating brake
        
        # ANTI-OSCILLATION: Stronger velocity reference smoothing
        if not hasattr(self, 'prev_velocity_desired'):
            self.prev_velocity_desired = 0.0
        
        # Enhanced low-pass filter for velocity reference (more aggressive smoothing)
        smoothing_factor = 0.8  # Increased from 0.7 for stronger smoothing
        self.velocity_desired = (1 - smoothing_factor) * self.velocity_desired + smoothing_factor * self.prev_velocity_desired
        self.prev_velocity_desired = self.velocity_desired
        
        # Create reference vector (RAW, not normalized)
        reference = np.array([self.theta_desired, self.velocity_desired], dtype=np.float32)
        
        return reference
        
    def compute_dhp_cost(self, state, reference):
        """
        Compute DHP cost function and gradient for your training loop
        
        This matches the cost structure used in your reward function.
        
        Args:
            state: Current state [theta, theta_dot]
            reference: Reference state [theta_ref, theta_dot_ref]
            
        Returns:
            cost: Scalar cost value
            gradient: Cost gradient w.r.t. state
        """
        # Ensure inputs are arrays
        state = np.asarray(state, dtype=np.float64).flatten()
        reference = np.asarray(reference, dtype=np.float64).flatten()
        
        # Handle normalization
        if self.normalize_states:
            state = self.denormalize_observation(state)
            reference = self.denormalize_observation(reference)
        
        # Extract states
        theta, theta_dot = state[0], state[1]
        theta_ref, theta_dot_ref = reference[0], reference[1]
        
        # Compute tracking errors
        theta_error = self.angular_difference(theta, theta_ref)
        theta_dot_error = theta_dot - theta_dot_ref
        distance_to_target = abs(theta_error)
        
        # Error vector
        error = np.array([theta_error, theta_dot_error])
        
        # Simple quadratic cost: J = e^T * Q * e (back to original but with stronger velocity damping)
        cost = error.T @ self.Q_matrix @ error
        
        # MODERATE: Prevent overshoot with reasonable penalties
        if distance_to_target < 0.5:  # Within approach zone
            # Predictive overshoot detection: will we overshoot with current velocity?
            time_to_target = distance_to_target / max(abs(theta_dot), 0.1)  # Time to reach target
            predicted_overshoot = abs(theta_dot) * time_to_target > distance_to_target * 2  # Rough overshoot prediction
            
            if predicted_overshoot or abs(theta_dot) > 2.0:
                # Strong but not extreme penalty for dangerous approach
                overshoot_risk_penalty = 100.0 * (abs(theta_dot) ** 2)  # Quadratic penalty (reduced from quartic)
                cost += overshoot_risk_penalty
        
        # REASONABLE: Velocity damping when close to target
        if distance_to_target < 0.3:  # Expanded penalty zone to 0.3 rad (< 17.2°)
            # Progressive velocity penalty that gets stronger as we get closer
            proximity_factor = (0.3 - distance_to_target) / 0.3  # 1.0 when at target, 0.0 at boundary
            velocity_damping = 20.0 * proximity_factor * (theta_dot ** 2)  # Reasonable penalty (reduced from 100.0)
            cost += velocity_damping
        
        # Cost gradient: ∂J/∂e = 2 * Q * e
        dcost_derror = 2.0 * self.Q_matrix @ error
        
        # Additional gradient terms for velocity damping
        additional_velocity_gradient = 0.0
        
        # Gradient for extreme overshoot prevention
        if distance_to_target < 0.5:
            time_to_target = distance_to_target / max(abs(theta_dot), 0.1)
            predicted_overshoot = abs(theta_dot) * time_to_target > distance_to_target * 2
            
            if predicted_overshoot or abs(theta_dot) > 2.0:
                # Gradient of 1000.0 * theta_dot^4
                additional_velocity_gradient += 4000.0 * (theta_dot ** 3)
        
        # Gradient for critical velocity damping
        if distance_to_target < 0.3:
            proximity_factor = (0.3 - distance_to_target) / 0.3
            additional_velocity_gradient += 200.0 * proximity_factor * theta_dot  # Gradient of 100.0 * proximity * theta_dot^2
        
        # Transform gradient to state coordinates
        dcostdx = np.array([dcost_derror[0], dcost_derror[1] + additional_velocity_gradient])
        
        # Apply gradient transformation for normalized states
        if self.normalize_states:
            dcostdx[0] = dcostdx[0] / self.max_angle
            dcostdx[1] = dcostdx[1] / self.max_velocity
        
        # Ensure finite values
        if not np.all(np.isfinite(dcostdx)):
            dcostdx = np.zeros_like(dcostdx)
            
        cost_scalar = float(cost)
        if not np.isfinite(cost_scalar):
            cost_scalar = 1000.0
            
        return cost_scalar, dcostdx.astype(np.float32)
        
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Control action in [-1, 1] (will be scaled to force)
            
        Returns:
            observation: Next state [theta, theta_dot]
            reward: Reward for the transition
            done: Episode termination flag
            info: Additional information dictionary
        """
        # Ensure action is properly formatted
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        else:
            action = np.array(action, dtype=np.float32).flatten()
            
        if len(action) == 0:
            action = np.array([0.0])
        elif len(action) > 1:
            action = action[:1]
            
        # SMART EMERGENCY ACTION CONSTRAINT: Only intervene when necessary
        current_state = np.array([self.x1, self.x2])
        distance_to_target = abs(self.angular_difference(self.x1, self.theta_desired))
        current_velocity = self.x2
        
        # Calculate if we're in danger of overshooting
        time_to_target = distance_to_target / max(abs(current_velocity), 0.1)
        stopping_distance = 0.5 * abs(current_velocity) * time_to_target
        imminent_overshoot = stopping_distance > distance_to_target * 0.7  # Conservative threshold
        
        # Only constrain actions that would make the situation worse
        action_makes_worse = False
        if current_velocity > 0 and action[0] > 0:  # Moving toward target, accelerating more
            action_makes_worse = True
        elif current_velocity < 0 and action[0] < 0:  # Moving away from target, accelerating away
            action_makes_worse = True
        
        # Multi-level emergency intervention (ONLY when action makes things worse) - REDUCED THRESHOLDS
        if distance_to_target < 0.02 and action_makes_worse:  # Very close (< 1.15°, was 3°)
            # EMERGENCY STOP: Force maximum braking only if action is harmful
            action[0] = -0.5 * np.sign(current_velocity)  # Moderate counter-torque (was 0.8)
            
        elif distance_to_target < 0.08 and abs(current_velocity) > 2.5 and action_makes_worse:  # Reduced zone, higher velocity threshold
            # EMERGENCY BRAKING: Force braking only if action would accelerate
            action[0] = -0.3 * np.sign(current_velocity)  # Light counter-torque (was 0.6)
            
        elif imminent_overshoot and action_makes_worse and abs(current_velocity) > 3.0:  # Only very high velocity
            # OVERSHOOT PREVENTION: Only prevent harmful actions at high speeds
            if current_velocity > 0 and action[0] > 0.3:  # Higher threshold for intervention
                action[0] = min(action[0], 0.1)  # Allow small positive actions
            elif current_velocity < 0 and action[0] < -0.3:  # Higher threshold for intervention
                action[0] = max(action[0], -0.1)  # Allow small negative actions
            
        # Scale action to force
        force = action[0] * self.Fmax
        
        # Apply force limits
        force = np.clip(force, -self.Fmax, self.Fmax)
        
        # Get current state before update
        current_state = np.array([self.x1, self.x2])
        
        # Integrate dynamics using RK4
        next_state = self._integrate_rk4(current_state, force)
        
        # Update internal state
        self.x1, self.x2 = next_state[0], next_state[1]
        
        # Update time and step counters
        self.t += self.dt
        self.steps += 1
        
        # Store action for derivative calculation
        self.actions_history.append(action[0])
        
        # Get observation (raw physical units)
        observation = self.get_pendulum_state()
        
        # Generate reference BEFORE normalization
        reference = self.generate_reference()
        
        # Store desired trajectory for visualization
        self.desired_traj.append(self.theta_desired)
        
        # Compute reward using raw observation
        reward = self._get_reward(observation, action[0])
        
        # Check termination conditions
        done = self._is_done()
        truncated = False  # Add truncated flag for new Gymnasium API
        
        # Compute DHP cost using raw states and reference
        dhp_cost, dhp_gradient = self.compute_dhp_cost(observation, reference)
        
        # Apply normalization AFTER all computations
        if self.normalize_states:
            observation_normalized = self.normalize_observation(observation)
            reference_normalized = self.normalize_observation(reference)
        else:
            observation_normalized = observation.copy()
            reference_normalized = reference.copy()
        
        # Create info dict with BOTH raw and normalized states
        info = {
            'reference': reference_normalized,  # This is what your training script expects
            'raw_reference': reference,         # Raw reference for debugging
            'raw_states': observation,          # Raw states for debugging
            'pendulum_states': observation_normalized,
            'dhp_cost': dhp_cost,
            'dhp_gradient': dhp_gradient,
            'theta_desired': self.theta_desired,
            'velocity_desired': self.velocity_desired,
            'position_error': abs(self.angular_difference(self.x1, self.theta_desired)),
            'enhanced_pid_gains': {
                'Kp': self.enhanced_pid.Kp,
                'Ki': self.enhanced_pid.Ki,
                'Kd': self.enhanced_pid.Kd
            },
            'force_applied': force,
            'episode_step': self.steps,
            'success': abs(self.angular_difference(self.x1, self.theta_desired)) < 0.1
        }
        
        return observation_normalized, reward, done, info
        
    def predict(self, action):
        """
        Predict next state without updating environment (for model learning)
        
        This is used by your approximation model during training.
        """
        # Ensure action is properly formatted
        action = np.array(action, dtype=np.float32).flatten()
        if len(action) == 0:
            action = np.array([0.0])
        elif len(action) > 1:
            action = action[:1]
            
        # Scale action to force
        force = action[0] * self.Fmax
        force = np.clip(force, -self.Fmax, self.Fmax)
        
        # Get current state
        current_state = np.array([self.x1, self.x2])
        
        # Predict next state using RK4
        predicted_state = self._integrate_rk4(current_state, force)
        
        return predicted_state
        
    def _get_reward(self, state, action):
        """
        Compute reward function (match your original implementation)
        
        This implements the same reward structure as your original code.
        """
        # Denormalize state if needed
        if self.normalize_states:
            state = self.denormalize_observation(state)
            
        current_theta = state[0]
        current_theta_dot = state[1]
        
        # Compute tracking errors
        delta_theta = self.angular_difference(self.theta_desired, current_theta)
        
        # Action derivative for smoothness penalty
        if len(self.actions_history) > 1:
            dactiondt = (self.actions_history[-1] - self.actions_history[-2]) / self.dt
        else:
            dactiondt = 0.0
            
        # Velocity reference from Enhanced PID
        velocity_error = self.velocity_desired - current_theta_dot
        
        # Reward components (match your original structure)
        position_penalty = 10.0 * (delta_theta ** 2)
        velocity_penalty = 0.001 * (velocity_error ** 2)
        action_smoothness_penalty = 0.0001 * (dactiondt ** 2)

        reward = -(position_penalty + velocity_penalty + action_smoothness_penalty)
        
        return float(reward)
        
    def _is_done(self):
        """Check if episode should terminate"""
        # Early termination for excellent tracking
        position_error = abs(self.angular_difference(self.x1, self.theta_desired))
        
        if (self.steps >= self.max_episode_steps or 
            position_error < np.pi/1000):  # Very precise tracking
            return True
        else:
            return False
            
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        Returns:
            tuple: (observation, info) - NEW GYM API FORMAT
        """
        # Reset time and counters
        self.t = 0.0
        self.steps = 0
        
        # Reset trajectory and action history
        self.desired_traj = []
        self.actions_history = [0]
        
        # Reset Enhanced PID
        self.enhanced_pid.reset()
        
        # Better initial state for inverted pendulum learning
        # Start near upright position to make the task learnable
        # self.x1 = np.pi + self.np_random.uniform(-0.2, 0.2)  # Start near inverted position ±11.5°
        self.x1 = self.np_random.uniform(-np.pi/10.0, np.pi/10.0)  # Start near upright position
        self.x2 = self.np_random.uniform(-0.5, 0.5)          # Small initial velocity

        # Initialize desired trajectory
        self.theta_desired = self.desired_trajectory()
        self.desired_traj.append(self.theta_desired)
        
        # Get initial observation (raw)
        observation = self.get_pendulum_state()
        
        # Generate initial reference BEFORE normalization
        reference = self.generate_reference()
        
        # Apply normalization for return value
        if self.normalize_states:
            observation_normalized = self.normalize_observation(observation)
            reference_normalized = self.normalize_observation(reference)
        else:
            observation_normalized = observation.copy()
            reference_normalized = reference.copy()
        
        # Create info dict with initial reference (CRITICAL FIX)
        info = {
            'reference': reference_normalized,  # This is what training script expects
            'raw_reference': reference,         # Raw reference for debugging
            'raw_states': observation,          # Raw states for debugging
            'theta_desired': self.theta_desired,
            'velocity_desired': self.velocity_desired,
            'target_angle': self.theta_desired,  # Alias for compatibility
            'position_error': abs(self.angular_difference(self.x1, self.theta_desired)),
            'episode_step': self.steps,
            'success': False  # Episode just started
        }
        
        return observation_normalized, info  # Return tuple as expected by new gym API
        
    def render(self, mode='human'):
        """
        Render the environment (compatible with your visualization)
        """
        if mode == 'human':
            # Clear previous plot
            plt.clf()
            
            # Calculate mass position
            x_mass = self.L * np.sin(self.x1)
            y_mass = -self.L * np.cos(self.x1)
            
            # Set plot limits
            plt.xlim(-self.L - 0.1, self.L + 0.1)
            plt.ylim(-self.L - 0.1, self.L + 0.1)
            
            # Draw the rod
            plt.plot([0, x_mass], [0, y_mass], 'b-', linewidth=3, label='Rod')
            
            # Draw the mass
            plt.plot(x_mass, y_mass, 'ro', markersize=12, label='Mass')
            
            # Draw desired position
            if hasattr(self, 'theta_desired'):
                x_des = self.L * np.sin(self.theta_desired)
                y_des = -self.L * np.cos(self.theta_desired)
                plt.plot(x_des, y_des, 'gx', markersize=15, markeredgewidth=3, label='Target')
                
                # Draw trajectory history
                if len(self.desired_traj) > 1:
                    theta_hist = self.desired_traj[-20:]  # Last 20 points
                    x_hist = [self.L * np.sin(th) for th in theta_hist]
                    y_hist = [-self.L * np.cos(th) for th in theta_hist]
                    plt.plot(x_hist, y_hist, 'gray', alpha=0.5, linewidth=1, label='Trajectory')
            
            # Add labels and legend
            plt.xlabel('X Position [m]')
            plt.ylabel('Y Position [m]')
            plt.title(f'DHP Pendulum Control (t={self.t:.2f}s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # Add state information
            plt.figtext(0.02, 0.95, f'θ = {np.rad2deg(self.x1):.1f}°', fontsize=10)
            plt.figtext(0.02, 0.90, f'θ̇ = {self.x2:.2f} rad/s', fontsize=10)
            plt.figtext(0.02, 0.85, f'θ_des = {np.rad2deg(self.theta_desired):.1f}°', fontsize=10)
            plt.figtext(0.02, 0.80, f'PID: Kp={self.enhanced_pid.Kp:.2f}', fontsize=10)
            
            plt.pause(self.dt / 2)
            
    def close(self):
        """Close the environment"""
        plt.close()
    
    def test_visualization(self, steps=100):
        """
        Test the visualization by running a simple demonstration
        """
        print("Testing pendulum visualization...")
        print("Close the plot window to continue.")
        
        # Enable interactive mode
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        
        # Reset environment
        obs, info = self.reset()
        
        for step in range(steps):
            # Simple control: try to reach target
            action = np.array([0.1 * np.sin(step * 0.1)])  # Oscillating action
            obs, reward, done, info = self.step(action)
            
            # Render
            self.render(mode='human')

            if done:
                print(f"Episode ended at step {step}")
                break
                
        plt.ioff()
        plt.show()
        print("Visualization test completed!")
        
    def get_velocity(self):
        """Get desired velocity (compatibility with your training script)"""
        return self.velocity_desired


# Registration for gym compatibility
try:
    from gym.envs.registration import register
    register(
        id='DHPPendulum-v0',
        entry_point='dhp_compatible_pendulum_env:DHPCompatiblePendulumEnv',
        max_episode_steps=3000,
    )
    print("[DHP-Compatible] Environment registered as 'DHPPendulum-v0'")
except:
    pass  # Handle case where environment is already registered


def create_dhp_pendulum(**kwargs):
    """
    Factory function to create DHP-compatible pendulum environment
    
    Args:
        **kwargs: Environment parameters
        
    Returns:
        DHPCompatiblePendulumEnv: Configured environment
    """
    default_params = {
        'm': 0.1,           # mass [kg] - match original PendulumEnv
        'g': 9.81,          # gravity [m/s²] - match original PendulumEnv
        'L': 0.5,           # length [m] - match original PendulumEnv
        'b': 0.3,           # damping [N⋅s/m] - match original PendulumEnv
        'Fmax': 4.0,        # max force [N] - match original PendulumEnv
        'dt': 0.02,         # time step [s] - match original PendulumEnv (0.02s)
        'normalize_states': True,  # Enable normalization for stable training
        'max_episode_steps': 200   # Match episode length (200 steps)
    }
    
    # Update with user parameters
    params = {**default_params, **kwargs}
    
    return DHPCompatiblePendulumEnv(**params)


if __name__ == "__main__":
    """
    Test the DHP-compatible environment with focus on training compatibility
    """
    print("="*80)
    print("TESTING DHP-COMPATIBLE PENDULUM ENVIRONMENT")
    print("="*80)
    
    # Test 1: Basic functionality
    print("\n--- Test 1: Basic Environment Creation ---")
    env = create_dhp_pendulum()
    print(f"✓ Environment created successfully")
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space: {env.action_space}")
    
    # Test 2: Training script compatibility (the failing case)
    print("\n--- Test 2: Training Script Compatibility ---")
    obs = env.reset()
    print(f"✓ Reset successful, initial observation shape: {obs.shape}")
    print(f"✓ Initial observation: {obs}")
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print(f"✓ Step successful")
    print(f"  - Next observation shape: {obs.shape}")
    print(f"  - Reward: {reward:.4f}")
    print(f"  - Done: {done}")
    print(f"  - Info keys: {list(info.keys())}")
    
    # Test the specific line that was failing
    print("\n--- Test 3: Reference Access (The Failing Line) ---")
    try:
        reference = info['reference']
        print(f"✓ Reference access successful: {reference}")
        print(f"✓ Reference shape: {reference.shape}")
        print(f"✓ Reference dtype: {reference.dtype}")
        
        # Test the specific way your training script uses it
        print(f"✓ Reference[0] (theta_des): {reference[0]}")
        print(f"✓ Reference[1] (velocity_des): {reference[1]}")
        
    except Exception as e:
        print(f"✗ Reference access failed: {e}")
        print(f"✗ Info dict: {info}")
        
    # Test 3: Multiple steps (simulate training loop)
    print("\n--- Test 4: Multiple Steps (Training Loop Simulation) ---")
    env.reset()
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        reference = info['reference']
        raw_reference = info.get('raw_reference', 'N/A')
        
        print(f"Step {step}: ref={reference}, raw_ref={raw_reference}, reward={reward:.3f}")
        
        if done:
            break
    
    # Test 4: Reference generation
    print("\n--- Test 5: Reference Generation Details ---")
    reference = env.generate_reference()
    print(f"✓ Raw reference generated: {reference}")
    print(f"  - Theta desired: {np.rad2deg(env.theta_desired):.2f}°")
    print(f"  - Velocity desired: {env.velocity_desired:.4f} rad/s")
    print(f"  - Enhanced PID gains: Kp={env.enhanced_pid.Kp:.3f}, Ki={env.enhanced_pid.Ki:.3f}, Kd={env.enhanced_pid.Kd:.3f}")
    
    # Test 5: DHP cost function
    print("\n--- Test 6: DHP Cost Function ---")
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    cost, gradient = env.compute_dhp_cost(obs, info['reference'])
    print(f"✓ DHP cost computed: {cost:.6f}")
    print(f"✓ DHP gradient: {gradient}")
    print(f"✓ Gradient finite: {np.all(np.isfinite(gradient))}")
    
    # Test 6: Normalization consistency
    print("\n--- Test 7: Normalization Consistency ---")
    env_norm = create_dhp_pendulum(normalize_states=True)
    env_raw = create_dhp_pendulum(normalize_states=False)
    
    # Reset both to same seed
    env_norm.seed(42)
    env_raw.seed(42)
    
    obs_norm = env_norm.reset()
    obs_raw = env_raw.reset()
    
    print(f"Normalized obs: {obs_norm}")
    print(f"Raw obs: {obs_raw}")
    
    # Apply same action
    action = 0.5
    obs_norm_next, _, _, info_norm = env_norm.step(action)
    obs_raw_next, _, _, info_raw = env_raw.step(action)
    
    print(f"Normalized reference: {info_norm['reference']}")
    print(f"Raw reference: {info_raw['reference']}")
    
    # Test 7: Success criteria
    print("\n--- Test 8: Success Criteria Check ---")
    
    tests_passed = []
    
    # Test reference access
    try:
        env = create_dhp_pendulum()
        obs = env.reset()
        obs, reward, done, info = env.step(0.0)
        reference = info['reference']
        tests_passed.append(True)
        print("✓ Reference access: PASS")
    except:
        tests_passed.append(False)
        print("✗ Reference access: FAIL")
    
    # Test reference format
    try:
        assert len(reference) == 2, f"Reference should have length 2, got {len(reference)}"
        assert isinstance(reference, np.ndarray), f"Reference should be numpy array, got {type(reference)}"
        tests_passed.append(True)
        print("✓ Reference format: PASS")
    except:
        tests_passed.append(False)
        print("✗ Reference format: FAIL")
    
    # Test DHP compatibility
    try:
        cost, gradient = env.compute_dhp_cost(obs, reference)
        assert np.isfinite(cost), f"Cost should be finite, got {cost}"
        assert np.all(np.isfinite(gradient)), f"Gradient should be finite, got {gradient}"
        tests_passed.append(True)
        print("✓ DHP compatibility: PASS")
    except:
        tests_passed.append(False)
        print("✗ DHP compatibility: FAIL")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPATIBILITY TEST SUMMARY")
    print("="*80)
    
    success_rate = np.mean(tests_passed) * 100
    print(f"Tests passed: {sum(tests_passed)}/{len(tests_passed)} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("🎉 ALL TESTS PASSED - READY FOR DHP TRAINING! 🎉")
        print("\nThe environment should now work with your training script!")
        print("The IndexError should be fixed.")
    else:
        print("⚠ SOME TESTS FAILED - NEEDS DEBUGGING")
        
    print("\nTo use in your training script:")
    print("1. Replace your environment creation with:")
    print("   from dhp_compatible_pendulum_env import create_dhp_pendulum")
    print("   env = create_dhp_pendulum()")
    print("2. Your existing training loop should work without changes!")
    
    print("\n" + "="*80)