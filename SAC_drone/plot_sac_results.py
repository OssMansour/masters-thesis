"""
SAC Training Results Plotting Module

Generates comprehensive plots for SAC spiral tracking training results,
matching the style and structure of DHP plotting for fair comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_sac_training_results(log_dir="logs", output_dir="plots"):
    """
    Generate comprehensive training plots from SAC training logs
    
    Args:
        log_dir: Directory containing training logs
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    training_log_path = os.path.join(log_dir, "training_log.txt")
    eval_log_path = os.path.join(log_dir, "eval_log.txt")
    
    if not os.path.exists(training_log_path):
        print(f"❌ Training log not found: {training_log_path}")
        return
    
    # Parse training log
    print("Loading training data...")
    training_data = np.loadtxt(training_log_path, delimiter=',', skiprows=1)
    
    if len(training_data) == 0:
        print("❌ No training data found")
        return
    
    timesteps = training_data[:, 0]
    episodes = training_data[:, 1]
    mean_rewards = training_data[:, 2]
    mean_lengths = training_data[:, 3]
    mean_errors = training_data[:, 4]
    
    # Parse evaluation log if exists
    eval_data = None
    if os.path.exists(eval_log_path):
        print("Loading evaluation data...")
        eval_data = np.loadtxt(eval_log_path, delimiter=',', skiprows=1)
        # Handle case where only one evaluation exists (1D array)
        if eval_data.ndim == 1:
            eval_data = eval_data.reshape(1, -1)
    
    # Create figure with multiple subplots (matching DHP style)
    print("Generating plots...")
    
    # =========================================================================
    # Figure 1: Training Progress (4 subplots)
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('SAC Training Progress - Spiral Trajectory Tracking', 
                  fontsize=16, fontweight='bold')
    
    # Subplot 1: Episode Rewards
    ax = axes1[0, 0]
    ax.plot(episodes, mean_rewards, 'b-', linewidth=2, label='Mean Reward (100 eps)')
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Episode Rewards Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add moving average
    if len(mean_rewards) > 50:
        window = min(50, len(mean_rewards) // 10)
        moving_avg = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, 'r--', linewidth=2, 
                label=f'Moving Avg ({window} eps)', alpha=0.7)
        ax.legend(fontsize=10)
    
    # Subplot 2: Episode Lengths (Duration)
    ax = axes1[0, 1]
    ax.plot(episodes, mean_lengths, 'g-', linewidth=2, label='Mean Length (100 eps)')
    ax.axhline(y=25.0, color='r', linestyle='--', linewidth=2, 
               label='Target Duration (25s)', alpha=0.7)
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Episode Duration (seconds)', fontsize=12)
    ax.set_title('Episode Duration Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 30])
    
    # Subplot 3: Position Tracking Error
    ax = axes1[1, 0]
    ax.plot(episodes, mean_errors, 'r-', linewidth=2, label='Mean Tracking Error')
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.set_title('Position Tracking Error Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add target error threshold
    ax.axhline(y=0.1, color='g', linestyle='--', linewidth=2, 
               label='Good Performance (0.1m)', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Subplot 4: Timesteps vs Reward
    ax = axes1[1, 1]
    ax.plot(timesteps, mean_rewards, 'purple', linewidth=2)
    ax.set_xlabel('Training Timesteps', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Reward vs Training Timesteps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show millions
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(6,6))
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "sac_training_progress.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot1_path}")
    plt.close()
    
    # =========================================================================
    # Figure 2: Evaluation Results (if available)
    # =========================================================================
    if eval_data is not None and len(eval_data) > 0:
        eval_timesteps = eval_data[:, 0]
        eval_mean_rewards = eval_data[:, 1]
        eval_std_rewards = eval_data[:, 2]
        eval_mean_errors = eval_data[:, 3]
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('SAC Evaluation Results', fontsize=16, fontweight='bold')
        
        # Subplot 1: Evaluation Rewards with Std
        ax = axes2[0]
        ax.plot(eval_timesteps, eval_mean_rewards, 'b-', linewidth=2, marker='o', 
                markersize=6, label='Mean Reward')
        ax.fill_between(eval_timesteps, 
                        eval_mean_rewards - eval_std_rewards,
                        eval_mean_rewards + eval_std_rewards,
                        alpha=0.3, color='b', label='±1 Std Dev')
        ax.set_xlabel('Training Timesteps', fontsize=12)
        ax.set_ylabel('Evaluation Reward', fontsize=12)
        ax.set_title('Evaluation Reward Over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(6,6))
        
        # Subplot 2: Evaluation Tracking Errors
        ax = axes2[1]
        ax.plot(eval_timesteps, eval_mean_errors, 'r-', linewidth=2, marker='s',
                markersize=6, label='Mean Tracking Error')
        ax.axhline(y=0.1, color='g', linestyle='--', linewidth=2, 
                   label='Target (0.1m)', alpha=0.7)
        ax.set_xlabel('Training Timesteps', fontsize=12)
        ax.set_ylabel('Position Error (m)', fontsize=12)
        ax.set_title('Evaluation Tracking Error', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(6,6))
        
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, "sac_evaluation_results.png")
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot2_path}")
        plt.close()
    
    # =========================================================================
    # Figure 3: Training Statistics Summary
    # =========================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    fig3.suptitle('SAC Training Statistics Summary', fontsize=16, fontweight='bold')
    
    # Subplot 1: Reward Distribution (Histogram)
    ax = axes3[0, 0]
    ax.hist(mean_rewards, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(mean_rewards), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(mean_rewards):.2f}')
    ax.axvline(np.median(mean_rewards), color='g', linestyle='--', linewidth=2,
               label=f'Median: {np.median(mean_rewards):.2f}')
    ax.set_xlabel('Mean Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Error Distribution (Histogram)
    ax = axes3[0, 1]
    ax.hist(mean_errors, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(mean_errors), color='b', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(mean_errors):.4f}m')
    ax.axvline(np.median(mean_errors), color='g', linestyle='--', linewidth=2,
               label=f'Median: {np.median(mean_errors):.4f}m')
    ax.set_xlabel('Position Error (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Tracking Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Learning Curve (Smoothed)
    ax = axes3[1, 0]
    if len(mean_rewards) > 100:
        window = 100
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], smoothed, 'b-', linewidth=2, label='Smoothed Reward (100 eps)')
        ax.fill_between(episodes[window-1:], 
                        smoothed - np.std(mean_rewards),
                        smoothed + np.std(mean_rewards),
                        alpha=0.2, color='b')
    else:
        ax.plot(episodes, mean_rewards, 'b-', linewidth=2)
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Learning Curve (Smoothed)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Subplot 4: Performance Metrics Table
    ax = axes3[1, 1]
    ax.axis('off')
    
    # Calculate statistics
    stats_data = [
        ['Metric', 'Value'],
        ['Total Episodes', f'{int(episodes[-1])}'],
        ['Total Timesteps', f'{int(timesteps[-1]):,}'],
        ['Final Mean Reward', f'{mean_rewards[-1]:.2f}'],
        ['Best Mean Reward', f'{np.max(mean_rewards):.2f}'],
        ['Final Tracking Error', f'{mean_errors[-1]:.4f}m'],
        ['Best Tracking Error', f'{np.min(mean_errors):.4f}m'],
        ['Mean Episode Duration', f'{np.mean(mean_lengths):.2f}s'],
        ['Episodes > 20s', f'{np.sum(mean_lengths > 20.0) / len(mean_lengths) * 100:.1f}%'],
        ['Convergence Rate', f'{np.sum(mean_errors < 0.1) / len(mean_errors) * 100:.1f}%']
    ]
    
    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Training Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "sac_training_statistics.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot3_path}")
    plt.close()
    
    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "="*80)
    print("SAC TRAINING SUMMARY")
    print("="*80)
    print(f"Total Episodes: {int(episodes[-1])}")
    print(f"Total Timesteps: {int(timesteps[-1]):,}")
    print(f"\nReward Performance:")
    print(f"  Final Mean Reward: {mean_rewards[-1]:.2f}")
    print(f"  Best Mean Reward: {np.max(mean_rewards):.2f}")
    print(f"  Average Reward: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
    print(f"\nTracking Performance:")
    print(f"  Final Error: {mean_errors[-1]:.4f}m")
    print(f"  Best Error: {np.min(mean_errors):.4f}m")
    print(f"  Average Error: {np.mean(mean_errors):.4f}m ± {np.std(mean_errors):.4f}m")
    print(f"\nEpisode Completion:")
    print(f"  Mean Duration: {np.mean(mean_lengths):.2f}s")
    print(f"  Target Duration: 25.0s")
    print(f"  Completion Rate (>20s): {np.sum(mean_lengths > 20.0) / len(mean_lengths) * 100:.1f}%")
    print(f"\nConvergence:")
    print(f"  Episodes with error < 0.1m: {np.sum(mean_errors < 0.1) / len(mean_errors) * 100:.1f}%")
    print("="*80)
    
    return {
        'episodes': episodes,
        'timesteps': timesteps,
        'mean_rewards': mean_rewards,
        'mean_lengths': mean_lengths,
        'mean_errors': mean_errors,
        'eval_data': eval_data
    }


if __name__ == "__main__":
    print("SAC Training Results Plotter")
    print("="*80)
    plot_sac_training_results()
    print("\n✅ Plotting complete!")
