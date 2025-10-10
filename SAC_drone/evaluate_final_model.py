"""
Evaluate the final SAC model (4M steps) without VecNormalize
Since observation space changed during training, we skip the old VecNormalize
"""
import numpy as np
import sys
from pathlib import Path

# Add gym-pybullet-drones to path
gym_path = Path(__file__).parent.parent / "gym-pybullet-drones"
sys.path.append(str(gym_path))

from gym_pybullet_drones.envs.SpiralAviary import SpiralAviary
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, Physics
from stable_baselines3 import SAC

# ==============================================================================
# Configuration
# ==============================================================================
EPISODE_LEN_SEC = 25.0
CTRL_FREQ = 30
PYB_FREQ = 240
NUM_EVAL_EPISODES = 5

# Model path - use the model trained with 38-dim observation space
MODEL_PATH = "logs/SAC/Drone/best_model/best_model.zip"  # This one has 38 dims!

print("=" * 80)
print("SAC Final Model Evaluation (4M timesteps)")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Episodes: {NUM_EVAL_EPISODES}")
print(f"Episode length: {EPISODE_LEN_SEC}s")
print("=" * 80)

# ==============================================================================
# Create environment and load model
# ==============================================================================
print("\n[1/2] Loading model...")

# Create evaluation environment with GUI and recording
env = SpiralAviary(
    drone_model=DroneModel.CF2X,
    gui=True,
    record=True,
    mode="spiral",
    physics=Physics.PYB,
    pyb_freq=PYB_FREQ,
    ctrl_freq=CTRL_FREQ,
    act=ActionType.RPM
)

# Load model directly (no VecNormalize since obs space changed)
model = SAC.load(MODEL_PATH)
print(f"✓ Model loaded successfully")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")

# ==============================================================================
# Run evaluation episodes
# ==============================================================================
print(f"\n[2/2] Running {NUM_EVAL_EPISODES} evaluation episodes...")

episode_rewards = []
episode_lengths = []
tracking_errors = []

for episode in range(NUM_EVAL_EPISODES):
    print(f"\n--- Episode {episode + 1}/{NUM_EVAL_EPISODES} ---")
    
    # Gymnasium API returns (obs, info)
    obs, info = env.reset()
    episode_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Get action from policy (deterministic)
        action, _states = model.predict(obs, deterministic=True)
        
        # Gymnasium API returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        step_count += 1
        
        # Print progress every 5 seconds
        if step_count % (CTRL_FREQ * 5) == 0:
            elapsed_time = step_count / CTRL_FREQ
            print(f"  t={elapsed_time:.1f}s | Reward: {reward:.3f} | Total: {episode_reward:.2f}")
    
    # Get final tracking error
    if hasattr(env, 'tracking_errors') and len(env.tracking_errors) > 0:
        avg_tracking_error = np.mean(env.tracking_errors)
        tracking_errors.append(avg_tracking_error)
    else:
        tracking_errors.append(0.0)
    
    episode_rewards.append(episode_reward)
    episode_lengths.append(step_count)
    
    elapsed_time = step_count / CTRL_FREQ
    print(f"✓ Episode {episode + 1} complete:")
    print(f"    Duration: {elapsed_time:.2f}s ({step_count} steps)")
    print(f"    Reward: {episode_reward:.2f}")
    if tracking_errors[-1] > 0:
        print(f"    Avg tracking error: {tracking_errors[-1]:.4f}m")

# ==============================================================================
# Summary statistics
# ==============================================================================
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
mean_length = np.mean(episode_lengths)
mean_duration = mean_length / CTRL_FREQ

print(f"\nPerformance Metrics:")
print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Mean episode length: {mean_length:.1f} steps ({mean_duration:.2f}s)")
print(f"  Episode rewards: {episode_rewards}")

if any(err > 0 for err in tracking_errors):
    mean_tracking_error = np.mean([e for e in tracking_errors if e > 0])
    print(f"  Mean tracking error: {mean_tracking_error:.4f}m")

# Check if episodes completed full duration
expected_steps = int(EPISODE_LEN_SEC * CTRL_FREQ)
completion_rate = np.mean([length >= expected_steps for length in episode_lengths]) * 100
print(f"\nEpisode Completion:")
print(f"  Full episodes: {sum(length >= expected_steps for length in episode_lengths)}/{NUM_EVAL_EPISODES}")
print(f"  Completion rate: {completion_rate:.1f}%")

print("\n" + "=" * 80)
print("✅ Evaluation complete! Videos saved to results/ directory")
print("=" * 80)

env.close()
