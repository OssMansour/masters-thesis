"""
Test SpiralAviary Environment with PID Controller

This script validates the SpiralAviary environment by:
1. Testing observation space consistency
2. Validating reward function behavior
3. Checking termination/truncation logic
4. Evaluating PID tracking performance
5. Generating diagnostic plots for environment assessment

Purpose: Ensure environment is ready for SAC training by identifying potential issues.
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add gym-pybullet-drones to path
gym_path = Path(__file__).parent.parent / "gym-pybullet-drones"
sys.path.append(str(gym_path))

from gym_pybullet_drones.envs.SpiralAviary import SpiralAviary
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# ==============================================================================
# Configuration
# ==============================================================================
EPISODE_LEN_SEC = 25.0  # Match paper specification
CTRL_FREQ = 30
PYB_FREQ = 240
GUI = True  # Set to False for faster testing
RECORD = False
MODE = "spiral"  # "spiral", "hover", or "goto"

# PID Controller RPM conversion (from DSLPIDControl)
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM = 20000
MAX_PWM = 65535
RPM_MIN = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
RPM_MAX = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST

print("=" * 80)
print("SpiralAviary Environment Test with PID Controller")
print("=" * 80)
print(f"Episode Length: {EPISODE_LEN_SEC}s")
print(f"Control Frequency: {CTRL_FREQ}Hz")
print(f"Physics Frequency: {PYB_FREQ}Hz")
print(f"Mode: {MODE}")
print(f"GUI: {GUI}")
print("=" * 80)

# ==============================================================================
# Create Environment
# ==============================================================================
print("\n[1/6] Creating SpiralAviary environment...")
env = SpiralAviary(
    drone_model=DroneModel.CF2X,
    gui=GUI,
    record=RECORD,
    mode=MODE,
    physics=Physics.PYB,
    pyb_freq=PYB_FREQ,
    ctrl_freq=CTRL_FREQ,
    act=ActionType.RPM
)

print(f"✓ Environment created successfully")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")
print(f"  Expected obs dim: 29 (20 state + 3 ref_pos + 3 target_vel + 3 delta_pos)")

# ==============================================================================
# Create PID Controller
# ==============================================================================
print("\n[2/6] Initializing DSL PID controller...")
pid = DSLPIDControl(drone_model=DroneModel.CF2X)
print("✓ PID controller initialized")

# ==============================================================================
# Data Logging
# ==============================================================================
log = {
    'time': [],
    'observations': [],   # Full observation vector
    'states': [],         # Full 20-dim state
    'targets': [],        # Target positions
    'positions': [],      # Drone positions [x, y, z]
    'velocities': [],     # Drone velocities
    'attitudes': [],      # [roll, pitch, yaw]
    'angular_rates': [],  # [wx, wy, wz]
    'actions': [],        # Motor RPMs (raw)
    'actions_norm': [],   # Motor RPMs (normalized)
    'rewards': [],        # Per-step rewards
    'reward_components': [],  # [pos_reward, att_reward, smooth_reward]
    'position_errors': [],
    'attitude_errors': [],
    'terminated': [],
    'truncated': [],
}

# ==============================================================================
# Run Episode
# ==============================================================================
print("\n[3/6] Running PID control episode...")
obs, info = env.reset()
print(f"✓ Environment reset")
print(f"  Initial observation shape: {obs.shape}")
print(f"  Initial observation sample: {obs[:5]}... (first 5 elements)")

max_steps = int(EPISODE_LEN_SEC * PYB_FREQ)
max_control_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)
episode_reward = 0.0
step = 0
control_step_counter = 0  # Counter for control timesteps (30Hz)

print(f"\nEpisode Configuration:")
print(f"  Max Physics Steps (240Hz): {max_steps}")
print(f"  Max Control Steps (30Hz): {max_control_steps}")
print(f"  Expected Episode Duration: {EPISODE_LEN_SEC}s")

for step in range(max_steps):
    # Get full state from observation
    # Observation structure: [state(20), ref_pos(3), target_vel(3), delta_pos(3)]
    state = obs[0:20]  # First 20 elements are full state
    
    # Extract state components for PID
    pos = state[0:3]        # [x, y, z]
    quat = state[3:7]       # [qw, qx, qy, qz]
    rpy = state[7:10]       # [roll, pitch, yaw]
    vel = state[10:13]      # [vx, vy, vz]
    ang_vel = state[13:16]  # [wx, wy, wz]
    
    # Extract target from observation
    ref_pos = obs[20:23]    # Target position
    
    # PID control - compute RPM commands
    target_rpy = np.zeros(3)  # Level flight
    rpm_tuple = pid.computeControl(
        control_timestep=1.0/CTRL_FREQ,
        cur_pos=pos,
        cur_quat=quat,
        cur_vel=vel,
        cur_ang_vel=ang_vel,
        target_pos=ref_pos,
        target_rpy=target_rpy
    )
    rpm = rpm_tuple[0]  # Extract RPM array from tuple
    
    # Normalize RPM to [-1, 1] for environment
    rpm_normalized = 2 * (rpm - RPM_MIN) / (RPM_MAX - RPM_MIN) - 1
    rpm_normalized = np.clip(rpm_normalized, -1, 1)
    
    # Reshape to (1, 4) for single drone environment
    rpm_normalized = rpm_normalized.reshape(1, 4)
    
    # Step environment (this happens every physics step at 240Hz)
    obs, reward, terminated, truncated, info = env.step(rpm_normalized)
    
    # Increment control step counter (happens at control frequency 30Hz)
    # Note: In gym-pybullet-drones, step() is called at physics freq but control happens at ctrl_freq
    if step % int(PYB_FREQ / CTRL_FREQ) == 0:
        control_step_counter += 1
    
    # Compute reward components for analysis (match environment's computation)
    target = ref_pos
    pos_error = np.linalg.norm(target - pos)
    pos_reward = np.exp(-pos_error**2)
    
    roll, pitch, yaw = rpy
    att_reward = np.exp(-5.0 * (roll**2 + pitch**2))
    
    # Smoothness (approximate - we don't have previous action in first step)
    if step > 0:
        prev_rpm = log['actions'][-1]
        action_diff = np.linalg.norm(rpm - prev_rpm)
        smooth_reward = np.exp(-0.1 * action_diff)
    else:
        smooth_reward = 1.0
    
    reward_components = [pos_reward, att_reward, smooth_reward]
    
    # Log data
    t = step / PYB_FREQ
    log['time'].append(t)
    log['observations'].append(obs.copy())
    log['states'].append(state.copy())
    log['targets'].append(ref_pos.copy())
    log['positions'].append(pos.copy())
    log['velocities'].append(vel.copy())
    log['attitudes'].append(rpy.copy())
    log['angular_rates'].append(ang_vel.copy())
    log['actions'].append(rpm.copy())
    log['actions_norm'].append(rpm_normalized.copy())
    log['rewards'].append(reward)
    log['reward_components'].append(reward_components)
    log['position_errors'].append(pos_error)
    log['attitude_errors'].append(np.linalg.norm(rpy))
    log['terminated'].append(terminated)
    log['truncated'].append(truncated)
    
    episode_reward += reward
    
    # Print progress every 2 seconds
    if step % (CTRL_FREQ * 2) == 0 and step > 0:
        print(f"  t={t:.1f}s: pos_err={pos_error:.3f}m, reward={reward:.3f}, "
              f"total_reward={episode_reward:.2f}")
    
    if terminated or truncated:
        print(f"\n✓ Episode ended at t={t:.2f}s")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        break

# Convert to arrays
for key in log:
    log[key] = np.array(log[key])

print(f"\n✓ Episode completed")
print(f"  Total physics steps (240Hz): {step + 1}")
print(f"  Total control steps (30Hz): {control_step_counter}")
print(f"  Actual episode duration: {(step + 1) / PYB_FREQ:.2f}s")
print(f"  Episode reward: {episode_reward:.2f}")
print(f"  Mean position error: {np.mean(log['position_errors']):.4f}m")
print(f"  Max position error: {np.max(log['position_errors']):.4f}m")
print(f"  Mean reward: {np.mean(log['rewards']):.4f}")

# ==============================================================================
# Environment Validation Checks
# ==============================================================================
print("\n[4/6] Validating environment properties...")

# Check 1: Observation space consistency
obs_shapes = [obs.shape[0] for obs in log['observations']]
if len(set(obs_shapes)) == 1:
    print(f"✓ Observation space consistent: {obs_shapes[0]} dimensions")
else:
    print(f"✗ WARNING: Observation space inconsistent: {set(obs_shapes)}")

# Check 2: Reward bounds
reward_min, reward_max = np.min(log['rewards']), np.max(log['rewards'])
print(f"✓ Reward range: [{reward_min:.4f}, {reward_max:.4f}]")
if reward_min >= 0 and reward_max <= 1.0:
    print(f"  ✓ Rewards properly bounded in [0, 1]")
else:
    print(f"  ⚠ WARNING: Rewards outside expected [0, 1] range")

# Check 3: Termination logic
num_terminated = np.sum(log['terminated'])
num_truncated = np.sum(log['truncated'])
print(f"✓ Termination events: {num_terminated} terminated, {num_truncated} truncated")

# Check 4: Action space
action_min, action_max = np.min(log['actions_norm']), np.max(log['actions_norm'])
print(f"✓ Normalized action range: [{action_min:.4f}, {action_max:.4f}]")
if action_min >= -1.0 and action_max <= 1.0:
    print(f"  ✓ Actions properly normalized")
else:
    print(f"  ✗ WARNING: Actions outside [-1, 1] range")

# Check 5: Reward component analysis
pos_rewards = log['reward_components'][:, 0]
att_rewards = log['reward_components'][:, 1]
smooth_rewards = log['reward_components'][:, 2]
print(f"✓ Reward components (mean):")
print(f"  Position: {np.mean(pos_rewards):.4f} (weight 0.6)")
print(f"  Attitude: {np.mean(att_rewards):.4f} (weight 0.3)")
print(f"  Smoothness: {np.mean(smooth_rewards):.4f} (weight 0.1)")

# ==============================================================================
# Generate Diagnostic Plots
# ==============================================================================
print("\n[5/6] Generating diagnostic plots...")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    colors = sns.color_palette("husl", 8)
except ImportError:
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

t = log['time']
positions = log['positions']
targets = log['targets']
attitudes = log['attitudes']
rewards = log['rewards']
pos_errors = log['position_errors']

# Figure 1: Position Tracking
fig1 = plt.figure(figsize=(14, 10))

ax1 = plt.subplot(4, 1, 1)
ax1.plot(t, positions[:, 0], color=colors[0], linewidth=2, label='x (PID)')
ax1.plot(t, targets[:, 0], '--', color=colors[1], linewidth=2, label='x_ref')
ax1.set_ylabel('X Position [m]')
ax1.set_title('SpiralAviary Environment Test - PID Control')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(4, 1, 2, sharex=ax1)
ax2.plot(t, positions[:, 1], color=colors[2], linewidth=2, label='y (PID)')
ax2.plot(t, targets[:, 1], '--', color=colors[3], linewidth=2, label='y_ref')
ax2.set_ylabel('Y Position [m]')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.plot(t, positions[:, 2], color=colors[4], linewidth=2, label='z (PID)')
ax3.plot(t, targets[:, 2], '--', color=colors[5], linewidth=2, label='z_ref')
ax3.set_ylabel('Z Position [m]')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(4, 1, 4, sharex=ax1)
ax4.plot(t, pos_errors, color='red', linewidth=2, label='Position Error')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Error [m]')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spiral_test_position_tracking.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: spiral_test_position_tracking.png")

# Figure 2: Reward Analysis
fig2 = plt.figure(figsize=(14, 10))

ax1 = plt.subplot(4, 1, 1)
ax1.plot(t, rewards, color=colors[0], linewidth=2, label='Total Reward')
ax1.axhline(y=np.mean(rewards), color='red', linestyle='--', label=f'Mean={np.mean(rewards):.3f}')
ax1.set_ylabel('Reward')
ax1.set_title('Reward Function Analysis')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(4, 1, 2, sharex=ax1)
ax2.plot(t, pos_rewards, color=colors[1], linewidth=2, label='Position Reward (w=0.6)')
ax2.set_ylabel('Position Reward')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.plot(t, att_rewards, color=colors[2], linewidth=2, label='Attitude Reward (w=0.3)')
ax3.set_ylabel('Attitude Reward')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(4, 1, 4, sharex=ax1)
ax4.plot(t, smooth_rewards, color=colors[3], linewidth=2, label='Smoothness Reward (w=0.1)')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Smoothness Reward')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spiral_test_reward_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: spiral_test_reward_analysis.png")

# Figure 3: Attitude Control
fig3 = plt.figure(figsize=(14, 8))

ax1 = plt.subplot(3, 1, 1)
ax1.plot(t, np.rad2deg(attitudes[:, 0]), color=colors[0], linewidth=2, label='Roll (PID)')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.set_ylabel('Roll [deg]')
ax1.set_title('Attitude Control')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(t, np.rad2deg(attitudes[:, 1]), color=colors[1], linewidth=2, label='Pitch (PID)')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax2.set_ylabel('Pitch [deg]')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(t, np.rad2deg(attitudes[:, 2]), color=colors[2], linewidth=2, label='Yaw (PID)')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Yaw [deg]')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spiral_test_attitude_control.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: spiral_test_attitude_control.png")

# Figure 4: 3D Trajectory
fig4 = plt.figure(figsize=(12, 10))
ax_3d = fig4.add_subplot(111, projection='3d')

ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
          color=colors[0], linewidth=3, label='Actual Trajectory (PID)')
ax_3d.plot(targets[:, 0], targets[:, 1], targets[:, 2], 
          '--', color=colors[1], linewidth=3, label='Reference Trajectory')
ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
             color='green', s=100, label='Start', marker='o')
ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
             color='red', s=100, label='End', marker='s')

ax_3d.set_xlabel('X Position [m]')
ax_3d.set_ylabel('Y Position [m]')
ax_3d.set_zlabel('Z Position [m]')
ax_3d.set_title('3D Spiral Trajectory - PID Control')
ax_3d.legend(loc='upper left')
ax_3d.grid(True, alpha=0.3)

# Set equal aspect ratio
max_range = max(
    np.max(positions[:, 0]) - np.min(positions[:, 0]),
    np.max(positions[:, 1]) - np.min(positions[:, 1]),
    np.max(positions[:, 2]) - np.min(positions[:, 2])
) / 2.0
mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) * 0.5
mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) * 0.5
mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) * 0.5
ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.savefig('spiral_test_3d_trajectory.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: spiral_test_3d_trajectory.png")

# ==============================================================================
# Summary Report
# ==============================================================================
print("\n[6/6] Environment Test Summary")
print("=" * 80)
print("ENVIRONMENT VALIDATION:")
print(f"  Observation Space: {env.observation_space.shape[0]} dimensions ✓")
print(f"  Action Space: {env.action_space.shape[0]} dimensions ✓")
print(f"  Episode Length: {(step+1)/PYB_FREQ:.2f}s / {EPISODE_LEN_SEC}s")
print(f"  Physics Steps Executed: {step+1} / {max_steps} (240Hz)")
print(f"  Control Steps Executed: {control_step_counter} / {max_control_steps} (30Hz)")
print(f"  Termination Logic: {'✓ Working' if num_terminated > 0 or num_truncated > 0 else '⚠ Check logic'}")
print()
print("TIMESTEP ANALYSIS:")
print(f"  Physics Frequency: {PYB_FREQ}Hz → {max_steps} steps per {EPISODE_LEN_SEC}s episode")
print(f"  Control Frequency: {CTRL_FREQ}Hz → {max_control_steps} steps per {EPISODE_LEN_SEC}s episode")
print(f"  Steps per control action: {int(PYB_FREQ / CTRL_FREQ)} physics steps")
print(f"  ✓ For SAC training at {CTRL_FREQ}Hz: expect {max_control_steps} timesteps/episode")
print()
print("TRACKING PERFORMANCE:")
print(f"  Mean Position Error: {np.mean(pos_errors):.4f}m")
print(f"  Max Position Error: {np.max(pos_errors):.4f}m")
print(f"  Final Position Error: {pos_errors[-1]:.4f}m")
print(f"  Mean Attitude Error: {np.mean(log['attitude_errors']):.4f}rad")
print()
print("REWARD FUNCTION:")
print(f"  Mean Total Reward: {np.mean(rewards):.4f}")
print(f"  Reward Range: [{reward_min:.4f}, {reward_max:.4f}]")
print(f"  Mean Position Component: {np.mean(pos_rewards):.4f}")
print(f"  Mean Attitude Component: {np.mean(att_rewards):.4f}")
print(f"  Mean Smoothness Component: {np.mean(smooth_rewards):.4f}")
print()
print("CONVERGENCE ASSESSMENT:")
if np.mean(pos_errors) < 0.5:
    print("  ✓ Position tracking: GOOD - SAC should converge well")
elif np.mean(pos_errors) < 1.0:
    print("  ⚠ Position tracking: MODERATE - SAC may need tuning")
else:
    print("  ✗ Position tracking: POOR - Check environment setup")

if np.mean(rewards) > 0.5:
    print("  ✓ Reward function: GOOD - Clear learning signal")
elif np.mean(rewards) > 0.3:
    print("  ⚠ Reward function: MODERATE - Learning may be slower")
else:
    print("  ✗ Reward function: POOR - May need reward tuning")

if reward_min >= 0 and reward_max <= 1.0:
    print("  ✓ Reward bounds: CORRECT - SAC-compatible")
else:
    print("  ✗ Reward bounds: INCORRECT - Fix reward clipping")

print()
print("RECOMMENDATIONS FOR SAC TRAINING:")
print("  1. Environment appears", "READY ✓" if np.mean(pos_errors) < 0.5 else "NEEDS REVIEW ⚠")
print("  2. Start with conservative learning rate (3e-4)")
print("  3. Monitor position error convergence in TensorBoard")
print("  4. Expected convergence: ~500k-1M timesteps")
print("=" * 80)

if GUI:
    plt.show()

env.close()
print("\n✓ Test complete!")
