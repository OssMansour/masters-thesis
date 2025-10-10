import numpy as np
import sys
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt

# Import your environment
sys.path.append("C:\\Projects\\masters-thesis\\gym-pybullet-drones")

from gym_pybullet_drones.envs.SpiralAviary import SpiralAviary
from gym_pybullet_drones.utils.enums import ActionType, Physics, ObservationType

class SpiralTrainingCallback(BaseCallback):
    """
    Custom callback for tracking SAC training progress on spiral tracking task.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.tracking_errors = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track episode statistics
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            
            # Log episode metrics
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                if self.verbose > 0:
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"Reward={ep_reward:.2f}, Length={ep_length:.2f}s")
            
            # Track average tracking error
            if 'avg_tracking_error' in info:
                self.tracking_errors.append(info['avg_tracking_error'])
                if self.verbose > 0:
                    print(f"  Avg Tracking Error: {info['avg_tracking_error']:.4f}m")
        
        return True
    
    def plot_training_progress(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        
        # Episode rewards
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) > 50:
            window = 50
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(episodes[:len(moving_avg)], moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards Over Training')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(episodes, self.episode_lengths, alpha=0.6)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length (s)')
        axes[0, 1].set_title('Episode Duration')
        axes[0, 1].grid(True)
        
        # Tracking errors
        if self.tracking_errors:
            axes[1, 0].plot(episodes[:len(self.tracking_errors)], self.tracking_errors, alpha=0.6)
            if len(self.tracking_errors) > 50:
                window = 50
                moving_avg = np.convolve(self.tracking_errors, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(episodes[:len(moving_avg)], moving_avg, 'r-', linewidth=2, label=f'MA({window})')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Avg Tracking Error (m)')
            axes[1, 0].set_title('Average Tracking Error Over Training')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale('log')
        
        # Success rate (low tracking error episodes)
        if self.tracking_errors:
            threshold = 0.3  # Success if avg error < 30cm
            success_window = 100
            successes = [1 if err < threshold else 0 for err in self.tracking_errors]
            if len(successes) >= success_window:
                success_rate = []
                for i in range(success_window, len(successes)):
                    rate = np.mean(successes[i-success_window:i]) * 100
                    success_rate.append(rate)
                axes[1, 1].plot(range(success_window, len(successes)), success_rate)
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Success Rate (%)')
                axes[1, 1].set_title(f'Success Rate (Error < {threshold}m, window={success_window})')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")
        
        return fig


def create_spiral_env(gui=False, mode="spiral"):
    """Create and wrap the SpiralAviary environment."""
    env = SpiralAviary(
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=30,
        gui=gui,
        record=False,
        mode=mode,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    
    # Wrap with Monitor for episode statistics
    env = Monitor(env)
    
    return env


def train_sac_spiral(
    total_timesteps=1_000_000,
    save_dir="./sac_spiral_models",
    log_dir="./sac_spiral_logs",
    gui=False,
    mode="spiral"
):
    """
    Train SAC agent on SpiralAviary environment.
    
    Parameters
    ----------
    total_timesteps : int
        Total training timesteps
    save_dir : str
        Directory to save trained models
    log_dir : str
        Directory for tensorboard logs
    gui : bool
        Whether to show PyBullet GUI during training
    mode : str
        Training mode: "spiral", "hover", or "goto"
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_{mode}_{timestamp}"
    
    print("="*60)
    print(f"Training SAC on SpiralAviary - Mode: {mode}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Save directory: {save_dir}")
    print("="*60)
    
    # Create training environment
    env = create_spiral_env(gui=gui, mode=mode)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment (without GUI)
    eval_env = create_spiral_env(gui=False, mode=mode)
    
    # Print environment information
    print(f"\nEnvironment Information:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Episode length: {env.EPISODE_LEN_SEC}s")
    print(f"  Control frequency: {env.CTRL_FREQ}Hz")
    print(f"  Steps per episode: {int(env.EPISODE_LEN_SEC * env.CTRL_FREQ)}")
    
    # SAC Hyperparameters (tuned for drone control)
    sac_config = {
        'policy': 'MlpPolicy',
        'env': vec_env,
        'learning_rate': 3e-4,
        'buffer_size': 200_000,  # Large buffer for off-policy learning
        'learning_starts': 1000,  # Start learning after 1000 steps
        'batch_size': 256,
        'tau': 0.005,  # Soft update coefficient
        'gamma': 0.99,  # Discount factor
        'train_freq': 1,  # Train after every step
        'gradient_steps': 1,  # One gradient step per env step
        'ent_coef': 'auto',  # Automatic entropy tuning
        'target_update_interval': 1,
        'use_sde': False,  # State-dependent exploration (can try True)
        'verbose': 1,
        'device': 'cpu',  # Use 'cuda' if GPU available
        'tensorboard_log': log_dir,
        'policy_kwargs': {
            'net_arch': {
                'pi': [256, 256],  # Actor network (2 layers, 256 units each)
                'qf': [256, 256]   # Q-network (2 layers, 256 units each)
            }
        }
    }
    
    print(f"\nSAC Configuration:")
    for key, value in sac_config.items():
        if key not in ['env', 'policy_kwargs']:
            print(f"  {key}: {value}")
    print(f"  Actor network: {sac_config['policy_kwargs']['net_arch']['pi']}")
    print(f"  Q-network: {sac_config['policy_kwargs']['net_arch']['qf']}")
    
    # Create SAC model
    model = SAC(**sac_config)
    
    # Setup callbacks
    training_callback = SpiralTrainingCallback(verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, run_name),
        log_path=os.path.join(log_dir, run_name),
        eval_freq=max(2500, int(env.EPISODE_LEN_SEC * env.CTRL_FREQ)),  # Eval every ~3-4 episodes
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([training_callback, eval_callback])
    
    # Train the model
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=4,  # Log every 4 episodes
            tb_log_name=run_name,
            reset_num_timesteps=True
        )
        
        # Save final model
        final_model_path = os.path.join(save_dir, f"{run_name}_final")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Plot and save training progress
        fig = training_callback.plot_training_progress(
            save_path=os.path.join(save_dir, f"{run_name}_training_progress.png")
        )
        plt.show()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total episodes: {len(training_callback.episode_rewards)}")
        if training_callback.episode_rewards:
            print(f"Final 10 episodes avg reward: {np.mean(training_callback.episode_rewards[-10:]):.2f}")
        if training_callback.tracking_errors:
            print(f"Final 10 episodes avg error: {np.mean(training_callback.tracking_errors[-10:]):.4f}m")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        # Save interrupted model
        interrupt_path = os.path.join(save_dir, f"{run_name}_interrupted")
        model.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
    
    finally:
        env.close()
        eval_env.close()
    
    return model, training_callback


if __name__ == "__main__":
    # Train SAC on spiral trajectory tracking
    model, callback = train_sac_spiral(
        total_timesteps=1_000_000,  # 1M timesteps (~1300 episodes)
        save_dir="./sac_spiral_models",
        log_dir="./sac_spiral_logs",
        gui=False,  # Set to True to watch training (slows down training)
        mode="spiral"  # Can also try "hover" or "goto"
    )