"""
Robust Pendulum Controller for Broken Physics Environment

Since the basic PendulumEnv has energy conservation issues (3.1J variation),
we need a robust control approach that can handle:
1. Non-conservative physics (energy drift)
2. Unpredictable dynamics
3. Large disturbances

This implements multiple robust control strategies that work even with
imperfect physics.

Author: DHP vs SAC Comparison Study
Date: August 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pendulum_env import PendulumRandomTargetEnv


class RobustPendulumController:
    """
    Robust controller designed to work with imperfect physics
    
    Uses multiple strategies:
    1. Adaptive gains based on performance
    2. Disturbance rejection
    3. Energy-aware control
    4. Robust estimation
    """
    
    def __init__(self, dt=0.01):
        self.dt = dt
        
        # Adaptive PID gains (start conservative)
        self.Kp = 2.0    # Position gain
        self.Ki = 0.01   # Integral gain (very small)
        self.Kd = 1.0    # Derivative gain
        
        # Adaptive gain bounds
        self.Kp_min, self.Kp_max = 0.5, 8.0
        self.Ki_min, self.Ki_max = 0.0, 0.1
        self.Kd_min, self.Kd_max = 0.2, 3.0
        
        # Adaptation rates
        self.adaptation_rate = 0.001
        
        # PID state
        self.integral = 0.0
        self.last_error = 0.0
        self.error_history = []
        
        # Robust estimation
        self.angle_filter = 0.0
        self.velocity_filter = 0.0
        self.filter_alpha = 0.3  # Low-pass filter coefficient
        
        # Performance tracking
        self.performance_history = []
        self.control_effort_history = []
        
        # Disturbance rejection
        self.disturbance_estimate = 0.0
        self.disturbance_filter = 0.8
        
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.last_error = 0.0
        self.error_history = []
        self.angle_filter = 0.0
        self.velocity_filter = 0.0
        self.performance_history = []
        self.control_effort_history = []
        self.disturbance_estimate = 0.0
        
        # Reset gains to conservative values
        self.Kp = 2.0
        self.Ki = 0.01
        self.Kd = 1.0
    
    def angular_difference(self, a, b):
        """Compute shortest angular difference"""
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return diff
    
    def robust_state_estimation(self, raw_angle, raw_velocity):
        """
        Apply robust filtering to handle noisy/inconsistent state measurements
        """
        # Low-pass filter for angle (reduces noise)
        self.angle_filter = (self.filter_alpha * raw_angle + 
                            (1 - self.filter_alpha) * self.angle_filter)
        
        # Low-pass filter for velocity (reduces noise)
        self.velocity_filter = (self.filter_alpha * raw_velocity + 
                               (1 - self.filter_alpha) * self.velocity_filter)
        
        return self.angle_filter, self.velocity_filter
    
    def estimate_disturbance(self, expected_response, actual_response):
        """
        Estimate external disturbances affecting the system
        """
        prediction_error = actual_response - expected_response
        
        # Update disturbance estimate with low-pass filter
        self.disturbance_estimate = (self.disturbance_filter * self.disturbance_estimate +
                                   (1 - self.disturbance_filter) * prediction_error)
        
        return self.disturbance_estimate
    
    def adapt_gains(self, error, control_effort):
        """
        Adapt PID gains based on performance
        """
        # Track recent performance
        recent_error = abs(error)
        self.performance_history.append(recent_error)
        self.control_effort_history.append(abs(control_effort))
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
            self.control_effort_history.pop(0)
        
        if len(self.performance_history) < 10:
            return  # Need some history first
        
        # Calculate performance metrics
        avg_error = np.mean(self.performance_history[-10:])
        avg_control = np.mean(self.control_effort_history[-10:])
        error_trend = np.mean(self.performance_history[-5:]) - np.mean(self.performance_history[-10:-5])
        
        # Adaptation logic
        if avg_error > 0.5:  # Large error - increase gains
            if avg_control < 1.5:  # Not saturating
                self.Kp = min(self.Kp + self.adaptation_rate, self.Kp_max)
                self.Kd = min(self.Kd + self.adaptation_rate * 0.5, self.Kd_max)
        elif avg_error < 0.1 and avg_control > 1.0:  # Good tracking but high effort - reduce gains
            self.Kp = max(self.Kp - self.adaptation_rate * 0.5, self.Kp_min)
            self.Kd = max(self.Kd - self.adaptation_rate * 0.25, self.Kd_min)
        
        # Integral gain adaptation (more conservative)
        if error_trend > 0.1:  # Error increasing - add integral
            self.Ki = min(self.Ki + self.adaptation_rate * 0.1, self.Ki_max)
        elif abs(error_trend) < 0.05:  # Stable - reduce integral to prevent windup
            self.Ki = max(self.Ki - self.adaptation_rate * 0.05, self.Ki_min)
    
    def compute_control(self, raw_angle, raw_velocity, target_angle):
        """
        Compute robust control with adaptive gains and disturbance rejection
        """
        # Robust state estimation
        filtered_angle, filtered_velocity = self.robust_state_estimation(raw_angle, raw_velocity)
        
        # Compute tracking error
        error = self.angular_difference(target_angle, filtered_angle)
        
        # Store error history for analysis
        self.error_history.append(error)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # PID computation with robust features
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with windup protection
        self.integral += error * self.dt
        
        # Anti-windup: limit integral based on current gains
        max_integral = 2.0 / (self.Ki + 1e-6)
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        I = self.Ki * self.integral
        
        # Derivative term (use filtered velocity)
        derivative = -filtered_velocity  # Target velocity is zero
        D = self.Kd * derivative
        
        # Basic PID control
        control_basic = P + I + D
        
        # Add disturbance compensation
        disturbance_compensation = -0.5 * self.disturbance_estimate
        
        # Final control signal
        control = control_basic + disturbance_compensation
        
        # Control saturation with soft limiting
        max_control = 2.0
        if abs(control) > max_control:
            control = max_control * np.tanh(control / max_control)
        
        # Estimate disturbance for next iteration
        expected_velocity_change = control * self.dt  # Simplified expectation
        actual_velocity_change = filtered_velocity - getattr(self, 'last_velocity', 0.0)
        self.estimate_disturbance(expected_velocity_change, actual_velocity_change)
        
        # Adapt gains based on performance
        self.adapt_gains(error, control)
        
        # Update for next iteration
        self.last_error = error
        self.last_velocity = filtered_velocity
        
        return control, {
            'error': error,
            'P': P,
            'I': I,
            'D': D,
            'disturbance_est': self.disturbance_estimate,
            'filtered_angle': filtered_angle,
            'filtered_velocity': filtered_velocity,
            'gains': {'Kp': self.Kp, 'Ki': self.Ki, 'Kd': self.Kd}
        }


def test_robust_control():
    """
    Test the robust controller on the broken physics environment
    """
    print("="*80)
    print("TESTING ROBUST CONTROL ON BROKEN PHYSICS ENVIRONMENT")
    print("="*80)
    
    # Create environment (known to have physics issues)
    env = PendulumRandomTargetEnv(gui=True,
        normalize_states=False,
        fixed_target=np.pi/4  # 45 degree target
    )
    
    # Create robust controller
    controller = RobustPendulumController()
    
    print(f"Target: {np.rad2deg(np.pi/4):.1f}°")
    print("Testing robust control with adaptive gains...")
    
    # Data collection
    time_data = []
    angle_data = []
    velocity_data = []
    control_data = []
    error_data = []
    gain_data = {'Kp': [], 'Ki': [], 'Kd': []}
    performance_data = []
    
    # Reset and run
    obs, info = env.reset()
    controller.reset()
    
    print(f"Initial angle: {np.rad2deg(obs[0]):.1f}°")
    print(f"Physics energy variation: ~3.1J (broken but manageable)")
    
    print("\nStep | Angle [°] | Error [°] | Control | Gains [P,I,D] | Performance")
    print("-" * 75)
    
    for step in range(600):  # 30 seconds - longer test for adaptation
        current_angle = obs[0]
        current_velocity = obs[1]
        
        # Compute robust control
        control, control_info = controller.compute_control(
            current_angle, current_velocity, np.pi/4
        )
        
        # Apply control
        obs, reward, terminated, truncated, info = env.step([control])
        
        # Store data
        time_data.append(step * env.dt)
        angle_data.append(current_angle)
        velocity_data.append(current_velocity)
        control_data.append(control)
        error_data.append(abs(control_info['error']))
        gain_data['Kp'].append(control_info['gains']['Kp'])
        gain_data['Ki'].append(control_info['gains']['Ki'])
        gain_data['Kd'].append(control_info['gains']['Kd'])
        
        # Calculate performance metric
        performance = 1.0 / (1.0 + abs(control_info['error']))  # Higher is better
        performance_data.append(performance)
        
        # Print progress
        if step % 100 == 0:
            gains = control_info['gains']
            print(f"{step:4d} | {np.rad2deg(current_angle):8.1f} | "
                  f"{np.rad2deg(control_info['error']):8.1f} | {control:7.3f} | "
                  f"[{gains['Kp']:.2f},{gains['Ki']:.3f},{gains['Kd']:.2f}] | "
                  f"{performance:.3f}")
        
        if terminated or truncated:
            break
    
    # Analysis
    final_error = error_data[-1]
    avg_error_last_100 = np.mean(error_data[-100:])  # Last 5 seconds
    max_control = np.max(np.abs(control_data))
    
    # Check for improvement over time
    early_performance = np.mean(performance_data[50:150])  # Steps 50-150
    late_performance = np.mean(performance_data[-100:])    # Last 100 steps
    improvement = late_performance - early_performance
    
    # Settling analysis
    tolerance = np.deg2rad(15.0)  # 15 degree tolerance (relaxed for broken physics)
    settled_indices = [i for i, e in enumerate(error_data) if e < tolerance]
    settling_time = time_data[settled_indices[0]] * 0.05 if settled_indices else None
    
    print(f"\n--- ROBUST CONTROL ANALYSIS ---")
    print(f"Final error: {np.rad2deg(final_error):.2f}°")
    print(f"Average error (last 5s): {np.rad2deg(avg_error_last_100):.2f}°")
    print(f"Maximum control: {max_control:.2f} N⋅m")
    print(f"Performance improvement: {improvement:.3f} ({improvement*100:.1f}%)")
    print(f"Settling time (15° tolerance): {settling_time:.1f}s" if settling_time else "Did not settle")
    
    # Gain adaptation analysis
    initial_gains = [gain_data['Kp'][10], gain_data['Ki'][10], gain_data['Kd'][10]]
    final_gains = [gain_data['Kp'][-1], gain_data['Ki'][-1], gain_data['Kd'][-1]]
    print(f"Initial gains: Kp={initial_gains[0]:.2f}, Ki={initial_gains[1]:.3f}, Kd={initial_gains[2]:.2f}")
    print(f"Final gains:   Kp={final_gains[0]:.2f}, Ki={final_gains[1]:.3f}, Kd={final_gains[2]:.2f}")
    
    # Success criteria (adjusted for broken physics)
    success = (avg_error_last_100 < np.deg2rad(20.0) and  # 20 degree tolerance
               max_control < 1.9 and                        # Reasonable control effort
               improvement > 0.05)                          # Some adaptation occurred
    
    print(f"\nROBUST CONTROL SUCCESS: {'YES' if success else 'NO'}")
    
    if success:
        print("✓ Controller successfully adapted to broken physics!")
        print("✓ Reasonable tracking performance achieved despite energy drift")
        print("✓ Adaptive gains helped compensate for system issues")
    else:
        print("✗ Even robust control struggled with the physics issues")
        print("✗ May need even more conservative approach or environment fix")
    
    # Create plots
    plot_robust_results(time_data, angle_data, velocity_data, control_data, 
                       error_data, gain_data, performance_data)
    
    return {
        'success': success,
        'final_error': final_error,
        'avg_error': avg_error_last_100,
        'improvement': improvement,
        'settling_time': settling_time,
        'max_control': max_control
    }


def plot_robust_results(time_data, angle_data, velocity_data, control_data, 
                       error_data, gain_data, performance_data):
    """
    Plot comprehensive results of robust control test
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    time_array = np.array(time_data)
    
    # Plot 1: Angle tracking
    axes[0,0].plot(time_array, np.rad2deg(angle_data), 'b-', linewidth=2, label='Actual')
    axes[0,0].axhline(y=45.0, color='r', linestyle='--', linewidth=2, label='Target')
    axes[0,0].set_xlabel('Time [s]')
    axes[0,0].set_ylabel('Angle [°]')
    axes[0,0].set_title('Robust Angle Tracking')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Control signal
    axes[0,1].plot(time_array, control_data, 'g-', linewidth=2)
    axes[0,1].axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Limit')
    axes[0,1].axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Time [s]')
    axes[0,1].set_ylabel('Control [N⋅m]')
    axes[0,1].set_title('Adaptive Control Signal')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Error evolution
    axes[0,2].plot(time_array, np.rad2deg(error_data), 'r-', linewidth=2)
    axes[0,2].axhline(y=15.0, color='g', linestyle='--', alpha=0.5, label='15° tolerance')
    axes[0,2].set_xlabel('Time [s]')
    axes[0,2].set_ylabel('Error [°]')
    axes[0,2].set_title('Tracking Error')
    axes[0,2].set_yscale('log')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Adaptive gains
    axes[1,0].plot(time_array, gain_data['Kp'], 'r-', linewidth=2, label='Kp')
    axes[1,0].plot(time_array, gain_data['Ki'], 'g-', linewidth=2, label='Ki×10')
    axes[1,0].plot(time_array, gain_data['Kd'], 'b-', linewidth=2, label='Kd')
    axes[1,0].set_xlabel('Time [s]')
    axes[1,0].set_ylabel('Gain Values')
    axes[1,0].set_title('Adaptive Gain Evolution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Performance metric
    axes[1,1].plot(time_array, performance_data, 'purple', linewidth=2)
    axes[1,1].set_xlabel('Time [s]')
    axes[1,1].set_ylabel('Performance (0-1)')
    axes[1,1].set_title('Controller Performance')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Phase portrait
    axes[1,2].plot(np.rad2deg(angle_data), velocity_data, 'b-', linewidth=2, alpha=0.7)
    axes[1,2].scatter(np.rad2deg(angle_data[0]), velocity_data[0], color='green', s=100, label='Start', zorder=5)
    axes[1,2].scatter(np.rad2deg(angle_data[-1]), velocity_data[-1], color='red', s=100, label='End', zorder=5)
    axes[1,2].axvline(x=45.0, color='gray', linestyle='--', alpha=0.7, label='Target')
    axes[1,2].set_xlabel('Angle [°]')
    axes[1,2].set_ylabel('Angular Velocity [rad/s]')
    axes[1,2].set_title('Phase Portrait')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to test robust control on broken physics environment
    """
    print("ROBUST PENDULUM CONTROL FOR BROKEN PHYSICS")
    print("=" * 80)
    print("This tests whether robust, adaptive control can work")
    print("despite the 3.1J energy variation in the basic pendulum.")
    print("Key features:")
    print("- Adaptive PID gains")
    print("- Disturbance estimation and rejection")
    print("- Robust state filtering")
    print("- Performance-based adaptation")
    
    try:
        results = test_robust_control()
        
        print("\n" + "="*80)
        print("ROBUST CONTROL SUMMARY")
        print("="*80)
        
        if results['success']:
            print("🎯 SUCCESS: Robust control worked despite broken physics!")
            print(f"   Final error: {np.rad2deg(results['final_error']):.1f}°")
            print(f"   Performance improvement: {results['improvement']*100:.1f}%")
            print(f"   Settling time: {results['settling_time']:.1f}s")
            print("\n✅ CONCLUSION: Your enhanced PID environment should work!")
            print("✅ The key is using robust, adaptive control strategies")
            print("✅ Even broken physics can be overcome with good control design")
        else:
            print("⚠ PARTIAL SUCCESS: Some improvement but still challenging")
            print(f"   Final error: {np.rad2deg(results['final_error']):.1f}°")
            print(f"   Best average error: {np.rad2deg(results['avg_error']):.1f}°")
            print("\n💡 RECOMMENDATION: Consider environment fixes:")
            print("   - Smaller timesteps (dt=0.01 instead of 0.05)")
            print("   - Different integration methods")
            print("   - Or switch to Gymnasium for proper physics")
        
    except Exception as e:
        print(f"\n!!! ERROR DURING ROBUST TESTING !!!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()