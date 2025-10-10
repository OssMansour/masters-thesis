"""
Quick test script for SAC trajectory plotting

Tests if the trajectory plotting works with existing trained model
"""
import sys
sys.path.append("C:\\Projects\\masters-thesis\\gym-pybullet-drones")

from plot_sac_trajectories import evaluate_and_plot_trajectories

print("Testing SAC Trajectory Plotting...")
print("=" * 80)

try:
    evaluate_and_plot_trajectories(
        model_path="logs/SAC/Drone/best_model/best_model.zip",
        vec_normalize_path="logs/final/vec_normalize.pkl",
        output_dir="plots/trajectories",
        n_episodes=3
    )
    print("\n✅ Trajectory plotting test successful!")
except Exception as e:
    print(f"\n❌ Trajectory plotting test failed: {e}")
    import traceback
    traceback.print_exc()
