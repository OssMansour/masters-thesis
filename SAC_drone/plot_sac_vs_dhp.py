"""
SAC vs DHP Comparison Plotting Module

Generates side-by-side comparison plots for SAC and DHP training results
to enable fair performance analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path

def load_dhp_metrics(dhp_log_dir):
    """Load DHP training metrics from JSON file"""
    json_path = os.path.join(dhp_log_dir, "training_metrics.json")
    
    if not os.path.exists(json_path):
        print(f"❌ DHP metrics not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def load_sac_metrics(sac_log_dir):
    """Load SAC training metrics from log files"""
    training_log_path = os.path.join(sac_log_dir, "training_log.txt")
    
    if not os.path.exists(training_log_path):
        print(f"❌ SAC training log not found: {training_log_path}")
        return None
    
    data = np.loadtxt(training_log_path, delimiter=',', skiprows=1)
    
    metrics = {
        'episodes': data[:, 1].tolist(),
        'timesteps': data[:, 0].tolist(),
        'episode_rewards': data[:, 2].tolist(),
        'episode_durations': data[:, 3].tolist(),
        'position_errors': data[:, 4].tolist(),
    }
    
    return metrics

def plot_sac_vs_dhp_comparison(sac_log_dir="logs", dhp_log_dir=None, output_dir="comparison_plots"):
    """
    Generate comprehensive SAC vs DHP comparison plots
    
    Args:
        sac_log_dir: Directory containing SAC training logs
        dhp_log_dir: Directory containing DHP training logs and metrics
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics
    print("Loading SAC metrics...")
    sac_metrics = load_sac_metrics(sac_log_dir)
    
    if sac_metrics is None:
        print("❌ Cannot load SAC metrics")
        return
    
    dhp_metrics = None
    if dhp_log_dir is not None:
        print("Loading DHP metrics...")
        dhp_metrics = load_dhp_metrics(dhp_log_dir)
        
        if dhp_metrics is None:
            print("⚠ DHP metrics not available, generating SAC-only plots")
    
    # =========================================================================
    # Figure 1: Training Performance Comparison
    # =========================================================================
    if dhp_metrics is not None:
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('SAC vs DHP: Training Performance Comparison', 
                      fontsize=16, fontweight='bold')
        
        # Subplot 1: Episode Rewards
        ax = axes1[0, 0]
        sac_episodes = np.array(sac_metrics['episodes'])
        dhp_episodes = np.array(dhp_metrics['episode_numbers'])
        
        ax.plot(sac_episodes, sac_metrics['episode_rewards'], 
                'b-', linewidth=2, label='SAC', alpha=0.7)
        ax.plot(dhp_episodes, dhp_metrics['episode_rewards'], 
                'r-', linewidth=2, label='DHP', alpha=0.7)
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.set_title('Episode Rewards Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Subplot 2: Position Tracking Error
        ax = axes1[0, 1]
        ax.plot(sac_episodes, sac_metrics['position_errors'], 
                'b-', linewidth=2, label='SAC', alpha=0.7)
        ax.plot(dhp_episodes, dhp_metrics['episode_position_errors'], 
                'r-', linewidth=2, label='DHP', alpha=0.7)
        ax.axhline(y=0.1, color='g', linestyle='--', linewidth=2, 
                   label='Target (0.1m)', alpha=0.5)
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Position Error (m)', fontsize=12)
        ax.set_title('Position Tracking Error Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Subplot 3: Episode Duration
        ax = axes1[1, 0]
        ax.plot(sac_episodes, sac_metrics['episode_durations'], 
                'b-', linewidth=2, label='SAC', alpha=0.7)
        
        # Convert DHP durations from steps to seconds (assuming 30Hz)
        dhp_durations_sec = np.array(dhp_metrics['episode_durations']) / 30.0
        ax.plot(dhp_episodes, dhp_durations_sec, 
                'r-', linewidth=2, label='DHP', alpha=0.7)
        ax.axhline(y=25.0, color='g', linestyle='--', linewidth=2, 
                   label='Target (25s)', alpha=0.5)
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Episode Duration (seconds)', fontsize=12)
        ax.set_title('Episode Duration Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        ax.set_ylim([0, 30])
        
        # Subplot 4: Learning Efficiency (Smoothed Rewards)
        ax = axes1[1, 1]
        if len(sac_metrics['episode_rewards']) > 50:
            window = 50
            sac_smooth = np.convolve(sac_metrics['episode_rewards'], 
                                    np.ones(window)/window, mode='valid')
            ax.plot(sac_episodes[window-1:], sac_smooth, 
                   'b-', linewidth=2, label='SAC (smoothed)')
        
        if len(dhp_metrics['episode_rewards']) > 50:
            window = 50
            dhp_smooth = np.convolve(dhp_metrics['episode_rewards'], 
                                    np.ones(window)/window, mode='valid')
            ax.plot(dhp_episodes[window-1:], dhp_smooth, 
                   'r-', linewidth=2, label='DHP (smoothed)')
        
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Smoothed Reward', fontsize=12)
        ax.set_title('Learning Curves (50-episode moving average)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        plt.tight_layout()
        plot1_path = os.path.join(output_dir, "sac_vs_dhp_training_comparison.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot1_path}")
        plt.close()
    
    # =========================================================================
    # Figure 2: Performance Statistics Comparison Table
    # =========================================================================
    if dhp_metrics is not None:
        fig2, ax = plt.subplots(figsize=(12, 8))
        fig2.suptitle('SAC vs DHP: Performance Statistics', 
                      fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Calculate statistics
        sac_rewards = np.array(sac_metrics['episode_rewards'])
        sac_errors = np.array(sac_metrics['position_errors'])
        sac_durations = np.array(sac_metrics['episode_durations'])
        
        dhp_rewards = np.array(dhp_metrics['episode_rewards'])
        dhp_errors = np.array(dhp_metrics['episode_position_errors'])
        dhp_durations = np.array(dhp_metrics['episode_durations']) / 30.0  # Convert to seconds
        
        stats_data = [
            ['Metric', 'SAC', 'DHP', 'Winner'],
            ['Final Episode Reward', 
             f'{sac_rewards[-1]:.2f}', 
             f'{dhp_rewards[-1]:.2f}',
             'SAC' if sac_rewards[-1] > dhp_rewards[-1] else 'DHP'],
            ['Best Episode Reward', 
             f'{np.max(sac_rewards):.2f}', 
             f'{np.max(dhp_rewards):.2f}',
             'SAC' if np.max(sac_rewards) > np.max(dhp_rewards) else 'DHP'],
            ['Mean Episode Reward', 
             f'{np.mean(sac_rewards):.2f}', 
             f'{np.mean(dhp_rewards):.2f}',
             'SAC' if np.mean(sac_rewards) > np.mean(dhp_rewards) else 'DHP'],
            ['Final Tracking Error (m)', 
             f'{sac_errors[-1]:.4f}', 
             f'{dhp_errors[-1]:.4f}',
             'SAC' if sac_errors[-1] < dhp_errors[-1] else 'DHP'],
            ['Best Tracking Error (m)', 
             f'{np.min(sac_errors):.4f}', 
             f'{np.min(dhp_errors):.4f}',
             'SAC' if np.min(sac_errors) < np.min(dhp_errors) else 'DHP'],
            ['Mean Tracking Error (m)', 
             f'{np.mean(sac_errors):.4f}', 
             f'{np.mean(dhp_errors):.4f}',
             'SAC' if np.mean(sac_errors) < np.mean(dhp_errors) else 'DHP'],
            ['Mean Episode Duration (s)', 
             f'{np.mean(sac_durations):.2f}', 
             f'{np.mean(dhp_durations):.2f}',
             'SAC' if np.mean(sac_durations) > np.mean(dhp_durations) else 'DHP'],
            ['Episodes Completed (>20s)', 
             f'{np.sum(sac_durations > 20.0) / len(sac_durations) * 100:.1f}%', 
             f'{np.sum(dhp_durations > 20.0) / len(dhp_durations) * 100:.1f}%',
             'SAC' if np.sum(sac_durations > 20.0) > np.sum(dhp_durations > 20.0) else 'DHP'],
            ['Convergence Rate (<0.1m)', 
             f'{np.sum(sac_errors < 0.1) / len(sac_errors) * 100:.1f}%', 
             f'{np.sum(dhp_errors < 0.1) / len(dhp_errors) * 100:.1f}%',
             'SAC' if np.sum(sac_errors < 0.1) > np.sum(dhp_errors < 0.1) else 'DHP'],
        ]
        
        table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                        colWidths=[0.35, 0.2, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color winner column
        for i in range(1, len(stats_data)):
            winner = stats_data[i][3]
            if winner == 'SAC':
                table[(i, 3)].set_facecolor('#4CAF50')
                table[(i, 3)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 3)].set_facecolor('#FF5722')
                table[(i, 3)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors for first 3 columns
            if i % 2 == 0:
                for j in range(3):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, "sac_vs_dhp_statistics.png")
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot2_path}")
        plt.close()
    
    # =========================================================================
    # Print Comparison Summary
    # =========================================================================
    if dhp_metrics is not None:
        print("\n" + "="*80)
        print("SAC vs DHP COMPARISON SUMMARY")
        print("="*80)
        
        sac_rewards = np.array(sac_metrics['episode_rewards'])
        dhp_rewards = np.array(dhp_metrics['episode_rewards'])
        sac_errors = np.array(sac_metrics['position_errors'])
        dhp_errors = np.array(dhp_metrics['episode_position_errors'])
        
        print("\nReward Performance:")
        print(f"  SAC Mean: {np.mean(sac_rewards):.2f} | DHP Mean: {np.mean(dhp_rewards):.2f}")
        print(f"  SAC Best: {np.max(sac_rewards):.2f} | DHP Best: {np.max(dhp_rewards):.2f}")
        print(f"  Winner: {'SAC' if np.mean(sac_rewards) > np.mean(dhp_rewards) else 'DHP'}")
        
        print("\nTracking Accuracy:")
        print(f"  SAC Mean Error: {np.mean(sac_errors):.4f}m | DHP Mean Error: {np.mean(dhp_errors):.4f}m")
        print(f"  SAC Best Error: {np.min(sac_errors):.4f}m | DHP Best Error: {np.min(dhp_errors):.4f}m")
        print(f"  Winner: {'SAC' if np.mean(sac_errors) < np.mean(dhp_errors) else 'DHP'}")
        
        print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare SAC and DHP training results')
    parser.add_argument('--sac-logs', default='logs', help='SAC logs directory')
    parser.add_argument('--dhp-logs', default=None, help='DHP logs directory')
    parser.add_argument('--output', default='comparison_plots', help='Output directory')
    
    args = parser.parse_args()
    
    print("SAC vs DHP Comparison Plotter")
    print("="*80)
    plot_sac_vs_dhp_comparison(
        sac_log_dir=args.sac_logs,
        dhp_log_dir=args.dhp_logs,
        output_dir=args.output
    )
    print("\n✅ Comparison plotting complete!")
