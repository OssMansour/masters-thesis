"""
SAC Trajectory Visualization Module

Generates comprehensive 3D trajectory plots and state analysis for SAC-trained
spiral tracking, matching the detailed visualization from SpiralEnvTest.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
from pathlib import Path

# Add gym-pybullet-drones to path
sys.path.append("C:\\Projects\\masters-thesis\\gym-pybullet-drones")

from gym_pybullet_drones.envs.SpiralAviary import SpiralAviary
from gym_pybullet_drones.utils.enums import ActionType, Physics
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

def evaluate_and_plot_trajectories(
    model_path="logs/SAC/Drone/best_model/best_model.zip",
    vec_normalize_path="logs/final/vec_normalize.pkl",
    output_dir="plots/trajectories",
    n_episodes=3,
    episode_len_sec=24.0
):
    """
    Evaluate trained SAC model and generate detailed trajectory plots
    
    Args:
        model_path: Path to trained SAC model
        vec_normalize_path: Path to VecNormalize stats
        output_dir: Directory to save plots
        n_episodes: Number of episodes to evaluate
        episode_len_sec: Episode length in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SAC TRAJECTORY VISUALIZATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Episode Length: {episode_len_sec}s")
    print("=" * 80)
    
    # Create evaluation environment
    print("\n[1/5] Creating evaluation environment...")
    def make_eval_env():
        return SpiralAviary(
            gui=False,
            record=False,
            act=ActionType.RPM,
            mode="spiral",
            pyb_freq=240,
            ctrl_freq=30,
            physics=Physics.PYB
        )
    
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize.load(vec_normalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    print("✓ Environment created")
    
    # Load trained model
    print("\n[2/5] Loading trained model...")
    model = SAC.load(model_path, env=eval_env)
    print("✓ Model loaded")
    
    # Run episodes and collect data
    print(f"\n[3/5] Running {n_episodes} evaluation episodes...")
    
    all_episodes_data = []
    
    for ep in range(n_episodes):
        print(f"\n  Episode {ep+1}/{n_episodes}...")
        
        episode_data = {
            'time': [],
            'positions': [],
            'velocities': [],
            'targets': [],
            'attitudes': [],  # [roll, pitch, yaw]
            'angular_rates': [],
            'actions': [],
            'rewards': [],
            'position_errors': [],
            'attitude_errors': [],
        }
        
        obs = eval_env.reset()
        done = False
        step = 0
        max_steps = int(episode_len_sec * 240)  # 240 Hz physics frequency
        episode_reward = 0.0
        
        while not done and step < max_steps:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = eval_env.step(action)
            
            # Extract state from observation (pre-normalization)
            # Get raw observation before normalization
            raw_obs = eval_env.get_original_obs()[0]
            
            # Parse observation structure: [state(20), ref_pos(3), target_vel(3), ref_att(3), 
            #                                delta_pos(3), delta_vel(3), delta_att(3)]
            state = raw_obs[0:20]
            
            pos = state[0:3]        # [x, y, z]
            quat = state[3:7]       # [qw, qx, qy, qz]
            rpy = state[7:10]       # [roll, pitch, yaw]
            vel = state[10:13]      # [vx, vy, vz]
            ang_vel = state[13:16]  # [wx, wy, wz]
            
            ref_pos = raw_obs[20:23]    # Target position
            
            # Calculate errors
            pos_error = np.linalg.norm(ref_pos - pos)
            att_error = np.linalg.norm(rpy)
            
            # Log data
            t = step / 30.0  # 30 Hz control frequency
            episode_data['time'].append(t)
            episode_data['positions'].append(pos.copy())
            episode_data['velocities'].append(vel.copy())
            episode_data['targets'].append(ref_pos.copy())
            episode_data['attitudes'].append(rpy.copy())
            episode_data['angular_rates'].append(ang_vel.copy())
            
            # Extract action properly - handle various shapes
            # Could be (4,), (1, 4), or (1, 1, 4) depending on environment
            action_flat = action.squeeze()  # Remove all singleton dimensions
            if action_flat.ndim == 1:
                episode_data['actions'].append(action_flat.copy())
            else:
                # If still not 1D, take first element
                episode_data['actions'].append(action_flat[0].copy())
            
            episode_data['rewards'].append(reward[0])  # Extract from batch
            episode_data['position_errors'].append(pos_error)
            episode_data['attitude_errors'].append(att_error)
            
            episode_reward += reward[0]
            step += 1
            
            if done[0]:
                break
        
        # Convert to arrays
        for key in episode_data:
            episode_data[key] = np.array(episode_data[key])
        
        # Debug: Check shapes
        if episode_data['actions'].ndim == 1:
            print(f"    ⚠ Warning: Actions are 1D (shape: {episode_data['actions'].shape})")
            print(f"       First action sample: {episode_data['actions'][:3]}")
        
        episode_data['episode_reward'] = episode_reward
        episode_data['episode_number'] = ep + 1
        
        all_episodes_data.append(episode_data)
        
        print(f"    Steps: {step}, Reward: {episode_reward:.2f}, "
              f"Mean Error: {np.mean(episode_data['position_errors']):.4f}m")
    
    eval_env.close()
    print("\n✓ Episode evaluation complete")
    
    # Generate plots for each episode
    print(f"\n[4/5] Generating trajectory plots...")
    
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        colors = sns.color_palette("husl", 8)
    except ImportError:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for ep_idx, ep_data in enumerate(all_episodes_data):
        ep_num = ep_data['episode_number']
        print(f"\n  Episode {ep_num}...")
        
        t = ep_data['time']
        positions = ep_data['positions']
        targets = ep_data['targets']
        attitudes = ep_data['attitudes']
        rewards = ep_data['rewards']
        pos_errors = ep_data['position_errors']
        velocities = ep_data['velocities']
        actions = ep_data['actions']
        
        # =====================================================================
        # Figure 1: 3D Trajectory (matching SpiralEnvTest.py)
        # =====================================================================
        fig1 = plt.figure(figsize=(12, 10))
        ax_3d = fig1.add_subplot(111, projection='3d')
        
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                  color=colors[0], linewidth=3, label='Actual Trajectory (SAC)')
        ax_3d.plot(targets[:, 0], targets[:, 1], targets[:, 2], 
                  '--', color=colors[1], linewidth=3, label='Reference Trajectory')
        ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                     color='green', s=150, label='Start', marker='o')
        ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                     color='red', s=150, label='End', marker='s')
        
        ax_3d.set_xlabel('X Position [m]', fontsize=12)
        ax_3d.set_ylabel('Y Position [m]', fontsize=12)
        ax_3d.set_zlabel('Z Position [m]', fontsize=12)
        ax_3d.set_title(f'Episode {ep_num}: 3D Spiral Trajectory - SAC Control\n'
                       f'Reward: {ep_data["episode_reward"]:.2f} | '
                       f'Mean Error: {np.mean(pos_errors):.4f}m',
                       fontsize=14, fontweight='bold')
        ax_3d.legend(loc='upper left', fontsize=11)
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
        plot1_path = os.path.join(output_dir, f"episode_{ep_num}_3d_trajectory.png")
        plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved: {plot1_path}")
        plt.close()
        
        # =====================================================================
        # Figure 2: Position Tracking (matching SpiralEnvTest.py)
        # =====================================================================
        fig2 = plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(t, positions[:, 0], color=colors[0], linewidth=2, label='x (SAC)')
        ax1.plot(t, targets[:, 0], '--', color=colors[1], linewidth=2, label='x_ref')
        ax1.set_ylabel('X Position [m]', fontsize=11)
        ax1.set_title(f'Episode {ep_num}: Position Tracking Analysis', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(4, 1, 2, sharex=ax1)
        ax2.plot(t, positions[:, 1], color=colors[2], linewidth=2, label='y (SAC)')
        ax2.plot(t, targets[:, 1], '--', color=colors[3], linewidth=2, label='y_ref')
        ax2.set_ylabel('Y Position [m]', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t, positions[:, 2], color=colors[4], linewidth=2, label='z (SAC)')
        ax3.plot(t, targets[:, 2], '--', color=colors[5], linewidth=2, label='z_ref')
        ax3.set_ylabel('Z Position [m]', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(4, 1, 4, sharex=ax1)
        ax4.plot(t, pos_errors, color='red', linewidth=2, label='Position Error')
        ax4.axhline(y=0.1, color='green', linestyle='--', linewidth=2, 
                   label='Target (0.1m)', alpha=0.5)
        ax4.set_xlabel('Time [s]', fontsize=11)
        ax4.set_ylabel('Error [m]', fontsize=11)
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, f"episode_{ep_num}_position_tracking.png")
        plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved: {plot2_path}")
        plt.close()
        
        # =====================================================================
        # Figure 3: Attitude Control (matching SpiralEnvTest.py)
        # =====================================================================
        fig3 = plt.figure(figsize=(14, 8))
        
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t, np.rad2deg(attitudes[:, 0]), color=colors[0], linewidth=2, label='Roll (SAC)')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Roll [deg]', fontsize=11)
        ax1.set_title(f'Episode {ep_num}: Attitude Control', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(t, np.rad2deg(attitudes[:, 1]), color=colors[1], linewidth=2, label='Pitch (SAC)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Pitch [deg]', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(t, np.rad2deg(attitudes[:, 2]), color=colors[2], linewidth=2, label='Yaw (SAC)')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time [s]', fontsize=11)
        ax3.set_ylabel('Yaw [deg]', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot3_path = os.path.join(output_dir, f"episode_{ep_num}_attitude_control.png")
        plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved: {plot3_path}")
        plt.close()
        
        # =====================================================================
        # Figure 4: Actions and Velocities
        # =====================================================================
        fig4 = plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot(3, 1, 1)
        # Plot motor commands - actions should now be (N, 4)
        if actions.ndim == 2 and actions.shape[1] == 4:
            for i in range(4):
                ax1.plot(t, actions[:, i], linewidth=2, label=f'Motor {i+1}')
            ax1.legend(loc='upper right', fontsize=9, ncol=4)
        elif actions.ndim == 1:
            # Single action dimension - shouldn't happen but handle gracefully
            ax1.plot(t, actions, linewidth=2, label='Action')
            ax1.legend(loc='upper right', fontsize=9)
        else:
            print(f"    ⚠ Warning: Unexpected action shape {actions.shape}")
            ax1.text(0.5, 0.5, f'Action shape: {actions.shape}\n(Expected: (N, 4))', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        ax1.set_ylabel('Normalized RPM [-1, 1]', fontsize=11)
        ax1.set_title(f'Episode {ep_num}: Control Actions & State Velocities', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(t, velocities[:, 0], color=colors[0], linewidth=2, label='vx')
        ax2.plot(t, velocities[:, 1], color=colors[1], linewidth=2, label='vy')
        ax2.plot(t, velocities[:, 2], color=colors[2], linewidth=2, label='vz')
        ax2.set_ylabel('Velocity [m/s]', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10, ncol=3)
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(t, rewards, color=colors[3], linewidth=2, label='Reward')
        ax3.axhline(y=np.mean(rewards), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rewards):.3f}')
        ax3.set_xlabel('Time [s]', fontsize=11)
        ax3.set_ylabel('Reward', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot4_path = os.path.join(output_dir, f"episode_{ep_num}_actions_velocities.png")
        plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved: {plot4_path}")
        plt.close()
    
    # =========================================================================
    # Generate Combined Summary Plot
    # =========================================================================
    print(f"\n  Generating combined summary...")
    
    fig5 = plt.figure(figsize=(16, 12))
    
    # 3D trajectories comparison
    ax_3d = fig5.add_subplot(2, 2, 1, projection='3d')
    for ep_idx, ep_data in enumerate(all_episodes_data):
        positions = ep_data['positions']
        targets = ep_data['targets']
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                  linewidth=2, label=f'Episode {ep_idx+1}', alpha=0.8)
    
    # Plot reference once
    ax_3d.plot(targets[:, 0], targets[:, 1], targets[:, 2], 
              '--', color='black', linewidth=3, label='Reference', alpha=0.5)
    ax_3d.set_xlabel('X [m]', fontsize=10)
    ax_3d.set_ylabel('Y [m]', fontsize=10)
    ax_3d.set_zlabel('Z [m]', fontsize=10)
    ax_3d.set_title('3D Trajectories Comparison', fontsize=12, fontweight='bold')
    ax_3d.legend(fontsize=9)
    ax_3d.grid(True, alpha=0.3)
    
    # Position errors comparison
    ax2 = fig5.add_subplot(2, 2, 2)
    for ep_idx, ep_data in enumerate(all_episodes_data):
        t = ep_data['time']
        pos_errors = ep_data['position_errors']
        ax2.plot(t, pos_errors, linewidth=2, label=f'Episode {ep_idx+1}', alpha=0.8)
    ax2.axhline(y=0.1, color='green', linestyle='--', linewidth=2, 
               label='Target (0.1m)', alpha=0.5)
    ax2.set_xlabel('Time [s]', fontsize=10)
    ax2.set_ylabel('Position Error [m]', fontsize=10)
    ax2.set_title('Position Tracking Errors', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Rewards comparison
    ax3 = fig5.add_subplot(2, 2, 3)
    for ep_idx, ep_data in enumerate(all_episodes_data):
        t = ep_data['time']
        rewards = ep_data['rewards']
        ax3.plot(t, rewards, linewidth=2, label=f'Episode {ep_idx+1}', alpha=0.8)
    ax3.set_xlabel('Time [s]', fontsize=10)
    ax3.set_ylabel('Reward', fontsize=10)
    ax3.set_title('Reward Evolution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Performance statistics table
    ax4 = fig5.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_data = [['Episode', 'Reward', 'Mean Error', 'Final Error']]
    for ep_idx, ep_data in enumerate(all_episodes_data):
        stats_data.append([
            f'{ep_idx+1}',
            f'{ep_data["episode_reward"]:.2f}',
            f'{np.mean(ep_data["position_errors"]):.4f}m',
            f'{ep_data["position_errors"][-1]:.4f}m'
        ])
    
    # Add average row
    avg_reward = np.mean([ep['episode_reward'] for ep in all_episodes_data])
    avg_error = np.mean([np.mean(ep['position_errors']) for ep in all_episodes_data])
    avg_final = np.mean([ep['position_errors'][-1] for ep in all_episodes_data])
    stats_data.append(['Average', f'{avg_reward:.2f}', f'{avg_error:.4f}m', f'{avg_final:.4f}m'])
    
    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style average row
    for i in range(4):
        table[(len(stats_data)-1, i)].set_facecolor('#4CAF50')
        table[(len(stats_data)-1, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)-1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Performance Statistics', fontsize=12, fontweight='bold', pad=20)
    
    fig5.suptitle('SAC Trajectory Evaluation Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plot5_path = os.path.join(output_dir, "combined_trajectory_summary.png")
    plt.savefig(plot5_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {plot5_path}")
    plt.close()
    
    # Print summary
    print("\n[5/5] Evaluation Summary")
    print("=" * 80)
    for ep_idx, ep_data in enumerate(all_episodes_data):
        print(f"Episode {ep_idx+1}:")
        print(f"  Total Reward: {ep_data['episode_reward']:.2f}")
        print(f"  Mean Position Error: {np.mean(ep_data['position_errors']):.4f}m")
        print(f"  Final Position Error: {ep_data['position_errors'][-1]:.4f}m")
        print(f"  Mean Attitude Error: {np.mean(ep_data['attitude_errors']):.4f}rad")
    
    print(f"\nAverage Performance:")
    print(f"  Mean Reward: {avg_reward:.2f}")
    print(f"  Mean Position Error: {avg_error:.4f}m")
    print(f"  Mean Final Error: {avg_final:.4f}m")
    print("=" * 80)
    print(f"\n✅ All trajectory plots saved to: {output_dir}/")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SAC trajectory plots')
    parser.add_argument('--model', default='logs/SAC/Drone/best_model/best_model.zip',
                       help='Path to trained model')
    parser.add_argument('--vec-normalize', default='logs/final/vec_normalize.pkl',
                       help='Path to VecNormalize stats')
    parser.add_argument('--output', default='plots/trajectories',
                       help='Output directory for plots')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to evaluate')
    
    args = parser.parse_args()
    
    evaluate_and_plot_trajectories(
        model_path=args.model,
        vec_normalize_path=args.vec_normalize,
        output_dir=args.output,
        n_episodes=args.episodes
    )
