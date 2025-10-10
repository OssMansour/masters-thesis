"""
Evaluate Trained SAC Model with Real-Time Visualization and Video Recording

This script:
1. Loads the best trained SAC model
2. Runs evaluation with REAL-TIME visualization (synchronized to 30Hz)
3. Records video of the evaluation
4. Generates comprehensive diagnostic plots
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import pybullet as p

# Add gym-pybullet-drones to path
gym_path = Path(__file__).parent.parent / "gym-pybullet-drones"
sys.path.append(str(gym_path))

from gym_pybullet_drones.envs.SpiralAviary import SpiralAviary
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, Physics
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==============================================================================
# Configuration
# ==============================================================================
EPISODE_LEN_SEC = 25.0
CTRL_FREQ = 30
PYB_FREQ = 240
GUI = True
NUM_EVAL_EPISODES = 1
REALTIME_SYNC = True  # Enable real-time synchronization
RECORD_VIDEO = True   # Enable video recording

# Model paths
MODEL_PATH = Path("logs/sac_spiral/best_model/best_model.zip")
VECNORM_PATH = Path("logs/sac_spiral/vec_normalize.pkl")
VIDEO_PATH = Path("logs/sac_spiral/evaluation_video.mp4")

print("=" * 80)
print("SAC Model Evaluation - Real-Time with Video Recording")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Real-time sync: {REALTIME_SYNC}")
print(f"Record video: {RECORD_VIDEO}")
print(f"Video output: {VIDEO_PATH}")
print("=" * 80)

# ==============================================================================
# Load Model
# ==============================================================================
print("\n[1/5] Loading trained SAC model...")

# Create environment with recording
env = SpiralAviary(
    drone_model=DroneModel.CF2X,
    gui=GUI,
    record=RECORD_VIDEO,
    mode="spiral",
    physics=Physics.PYB,
    pyb_freq=PYB_FREQ,
    ctrl_freq=CTRL_FREQ,
    act=ActionType.RPM
)

# Wrap in DummyVecEnv
env = DummyVecEnv([lambda: env])

# Load VecNormalize if exists
if VECNORM_PATH.exists():
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False  # Don't update stats during eval
    env.norm_reward = False  # Use raw rewards for logging
    print("✓ Loaded VecNormalize statistics")
else:
    print("⚠ No VecNormalize stats found, using raw observations")

# Load SAC model
model = SAC.load(MODEL_PATH, env=env)
print(f"✓ SAC model loaded successfully")

# ==============================================================================
# Data Logging
# ==============================================================================
log = {
    'time': [],
    'observations': [],
    'states': [],
    'targets': [],
    'positions': [],
    'velocities': [],
    'attitudes': [],
    'angular_rates': [],
    'actions': [],
    'rewards': [],
    'reward_components': [],
    'position_errors': [],
    'attitude_errors': [],
}

# ==============================================================================
# Run Evaluation with Real-Time Sync
# ==============================================================================
print("\n[2/5] Running evaluation episode with real-time visualization...")
print("⏱ Real-time sync enabled - simulation will run at 30Hz control frequency")

# Start video recording if enabled
if RECORD_VIDEO:
    # Get the unwrapped environment to access PyBullet client
    unwrapped_env = env.envs[0].env if hasattr(env.envs[0], 'env') else env.envs[0]
    p.startStateLogging(
        loggingType=p.STATE_LOGGING_VIDEO_MP4,
        fileName=str(VIDEO_PATH.absolute()),
        physicsClientId=unwrapped_env.CLIENT
    )
    print(f"🎥 Recording video to: {VIDEO_PATH}")

obs = env.reset()
episode_reward = 0.0
max_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)
step_duration = 1.0 / CTRL_FREQ  # 0.0333 seconds per step at 30Hz

start_time = time.time()

for step in range(max_steps):
    step_start_time = time.time()
    
    # Get action from SAC policy (deterministic for evaluation)
    action, _states = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, done, info = env.step(action)
    
    # Extract state from observation (unwrap from VecEnv)
    obs_unwrapped = obs[0] if len(obs.shape) > 1 else obs
    
    # State components
    state = obs_unwrapped[0:20]
    pos = state[0:3]
    quat = state[3:7]
    rpy = state[7:10]
    vel = state[10:13]
    ang_vel = state[13:16]
    
    # Target from observation
    ref_pos = obs_unwrapped[20:23]
    
    # Compute reward components (match environment computation)
    pos_error = np.linalg.norm(ref_pos - pos)
    pos_reward = np.exp(-pos_error**2)
    
    roll, pitch, yaw = rpy
    att_reward = np.exp(-5.0 * (roll**2 + pitch**2))
    
    # Smoothness (approximate)
    if step > 0:
        prev_action = log['actions'][-1]
        action_unwrapped = action[0] if len(action.shape) > 1 else action
        action_diff = np.linalg.norm(action_unwrapped - prev_action)
        smooth_reward = np.exp(-0.1 * action_diff)
    else:
        smooth_reward = 1.0
    
    reward_components = [pos_reward, att_reward, smooth_reward]
    
    # Log data
    t = step / CTRL_FREQ
    action_unwrapped = action[0] if len(action.shape) > 1 else action
    reward_unwrapped = reward[0] if isinstance(reward, np.ndarray) else reward
    
    log['time'].append(t)
    log['observations'].append(obs_unwrapped.copy())
    log['states'].append(state.copy())
    log['targets'].append(ref_pos.copy())
    log['positions'].append(pos.copy())
    log['velocities'].append(vel.copy())
    log['attitudes'].append(rpy.copy())
    log['angular_rates'].append(ang_vel.copy())
    log['actions'].append(action_unwrapped.copy())
    log['rewards'].append(reward_unwrapped)
    log['reward_components'].append(reward_components)
    log['position_errors'].append(pos_error)
    log['attitude_errors'].append([np.abs(roll), np.abs(pitch)])
    
    episode_reward += reward_unwrapped
    
    # Real-time synchronization
    if REALTIME_SYNC:
        step_elapsed = time.time() - step_start_time
        sleep_time = step_duration - step_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Progress indicator every 2 seconds
    if step % (CTRL_FREQ * 2) == 0:
        elapsed = time.time() - start_time
        print(f"  Step {step}/{max_steps} | Time: {t:.1f}s | Pos Error: {pos_error:.3f}m | Reward: {reward_unwrapped:.3f}")
    
    if done:
        print(f"⚠ Episode terminated early at step {step}")
        break

# Stop video recording
if RECORD_VIDEO:
    p.stopStateLogging(unwrapped_env.CLIENT)
    print(f"✓ Video saved to: {VIDEO_PATH}")

total_time = time.time() - start_time
print(f"\n✓ Evaluation complete!")
print(f"  Total simulation time: {total_time:.1f}s")
print(f"  Episode reward: {episode_reward:.3f}")
print(f"  Steps completed: {len(log['time'])}/{max_steps}")

# Convert lists to arrays
for key in log:
    log[key] = np.array(log[key])

# ==============================================================================
# Compute Statistics
# ==============================================================================
print("\n[3/5] Computing performance statistics...")

# Position tracking
pos_errors = log['position_errors']
mean_pos_error = np.mean(pos_errors)
std_pos_error = np.std(pos_errors)
max_pos_error = np.max(pos_errors)

# Attitude stability
roll_errors = log['attitude_errors'][:, 0]
pitch_errors = log['attitude_errors'][:, 1]
mean_roll = np.mean(roll_errors)
mean_pitch = np.mean(pitch_errors)
max_roll = np.max(roll_errors)
max_pitch = np.max(pitch_errors)

# Rewards
mean_reward = np.mean(log['rewards'])
std_reward = np.std(log['rewards'])
pos_rewards = log['reward_components'][:, 0]
att_rewards = log['reward_components'][:, 1]
smooth_rewards = log['reward_components'][:, 2]

print(f"\n📊 Performance Metrics:")
print(f"  Position Error: {mean_pos_error:.4f}m ± {std_pos_error:.4f}m (max: {max_pos_error:.4f}m)")
print(f"  Roll Error: {np.degrees(mean_roll):.2f}° (max: {np.degrees(max_roll):.2f}°)")
print(f"  Pitch Error: {np.degrees(mean_pitch):.2f}° (max: {np.degrees(max_pitch):.2f}°)")
print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
print(f"  Episode Reward: {episode_reward:.2f}")

# ==============================================================================
# Generate Plots
# ==============================================================================
print("\n[4/5] Generating diagnostic plots...")

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SAC Spiral Tracking Performance Analysis', fontsize=16, fontweight='bold')

# 1. Position Tracking
ax1 = axes[0, 0]
ax1.plot(log['time'], log['positions'][:, 0], label='X Position', linewidth=2)
ax1.plot(log['time'], log['positions'][:, 1], label='Y Position', linewidth=2)
ax1.plot(log['time'], log['positions'][:, 2], label='Z Position', linewidth=2)
ax1.plot(log['time'], log['targets'][:, 0], '--', label='X Target', alpha=0.7)
ax1.plot(log['time'], log['targets'][:, 1], '--', label='Y Target', alpha=0.7)
ax1.plot(log['time'], log['targets'][:, 2], '--', label='Z Target', alpha=0.7)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Position (m)', fontsize=12)
ax1.set_title('Position Tracking', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Reward Analysis
ax2 = axes[0, 1]
ax2.plot(log['time'], log['rewards'], label='Total Reward', linewidth=2, color='black')
ax2.plot(log['time'], pos_rewards, label='Position Reward (0.6)', alpha=0.7)
ax2.plot(log['time'], att_rewards, label='Attitude Reward (0.3)', alpha=0.7)
ax2.plot(log['time'], smooth_rewards, label='Smoothness Reward (0.1)', alpha=0.7)
ax2.axhline(y=mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.3f}')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Reward', fontsize=12)
ax2.set_title(f'Reward Analysis (Episode Total: {episode_reward:.2f})', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.1])

# 3. Attitude Tracking
ax3 = axes[1, 0]
ax3.plot(log['time'], np.degrees(log['attitudes'][:, 0]), label='Roll', linewidth=2)
ax3.plot(log['time'], np.degrees(log['attitudes'][:, 1]), label='Pitch', linewidth=2)
ax3.plot(log['time'], np.degrees(log['attitudes'][:, 2]), label='Yaw', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Angle (degrees)', fontsize=12)
ax3.set_title('Attitude Tracking', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. 3D Trajectory
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot(log['positions'][:, 0], log['positions'][:, 1], log['positions'][:, 2], 
         label='Actual', linewidth=2, color='blue')
ax4.plot(log['targets'][:, 0], log['targets'][:, 1], log['targets'][:, 2], 
         '--', label='Target', linewidth=2, color='red', alpha=0.7)
ax4.scatter([log['positions'][0, 0]], [log['positions'][0, 1]], [log['positions'][0, 2]], 
            color='green', s=100, marker='o', label='Start')
ax4.scatter([log['positions'][-1, 0]], [log['positions'][-1, 1]], [log['positions'][-1, 2]], 
            color='red', s=100, marker='X', label='End')
ax4.set_xlabel('X (m)', fontsize=12)
ax4.set_ylabel('Y (m)', fontsize=12)
ax4.set_zlabel('Z (m)', fontsize=12)
ax4.set_title('3D Trajectory', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
plot_path = Path("logs/sac_spiral/sac_spiral_evaluation_plots.png")
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Plots saved to: {plot_path}")

# ==============================================================================
# Comparison with PID Baseline
# ==============================================================================
print("\n[5/5] Comparison with PID Baseline:")
print("-" * 60)
print("Metric                    | PID Baseline  | SAC Result")
print("-" * 60)
print(f"Mean Reward               | 0.990         | {mean_reward:.3f}")
print(f"Position Error (m)        | 0.048         | {mean_pos_error:.3f}")
print(f"Roll Error (deg)          | ±4.0          | ±{np.degrees(mean_roll):.1f}")
print(f"Pitch Error (deg)         | ±4.0          | ±{np.degrees(mean_pitch):.1f}")
print(f"Episode Completion        | 750/750       | {len(log['time'])}/{max_steps}")
print("-" * 60)

# Close environment
env.close()

print("\n✅ Evaluation complete! Check:")
print(f"   - Video: {VIDEO_PATH}")
print(f"   - Plots: {plot_path}")
print("=" * 80)

plt.show()
