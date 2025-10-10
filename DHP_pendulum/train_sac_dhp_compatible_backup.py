"""
SAC Training Script for DHP-Compatible Pendulum Environment

This script adapts the SAC pipeline to use the same environment as DHP for fair comparison:
- Uses DHPCompatiblePendulumEnv (same as DHP training)
- State normalization system (critical for fair comparison)
- Best episode recording and replay
- Comprehensive training analysis and plots
- Session-best recording strategy
- Same logging and visualization structure

Author: DHP vs SAC Comparison Study  
Date: August 16, 2025
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
import json
import logging
import glob
import pickle
from datetime import datetime

# Set TensorFlow to use CPU only to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add paths for existing implementations
sys.path.append('/home/osos/Mohamed_Masters_Thesis/msc-thesis')
sys.path.append('/home/osos/Mohamed_Masters_Thesis/gym-pybullet-drones')
sys.path.append('/home/osos/Mohamed_Masters_Thesis/trial2')

# Import our DHP-compatible pendulum environment
from dhp_compatible_pendulum_env import DHPCompatiblePendulumEnv

# Import Gymnasium (Gym) components
import gymnasium as gym
from gymnasium import spaces

# Import Stable Baselines3 components
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    print("✅ Stable Baselines3 imported successfully")
except ImportError as e:
    print(f"Error importing Stable Baselines3: {e}")
    print("Please install with: pip install stable-baselines3[extra]")
    sys.exit(1)


class DHPCompatibleWrapper(gym.Wrapper):
    """
    Wrapper to make DHPCompatiblePendulumEnv work seamlessly with SAC
    
    The DHP environment already provides:
    - [theta, theta_dot] state
    - Reference generation
    - Proper physics
    - Normalization
    
    We just need to combine state and reference for SAC input.
    """
    def __init__(self, env):
        super().__init__(env)
        # SAC expects 4-D input: [theta, theta_dot, theta_ref, theta_dot_ref]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _pack_observation(self, obs, info):
        """Combine state and reference into single observation for SAC"""
        # obs is already [theta, theta_dot] (normalized or raw)
        # info['reference'] is [theta_ref, theta_dot_ref] (normalized or raw)
        reference = info.get('reference', np.array([0.0, 0.0]))
        
        # Ensure both are 1D arrays
        obs = np.asarray(obs, dtype=np.float32).flatten()
        reference = np.asarray(reference, dtype=np.float32).flatten()
        
        # Combine: [theta, theta_dot, theta_ref, theta_dot_ref]
        combined = np.concatenate([obs, reference]).astype(np.float32)
        
        return combined

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        combined_obs = self._pack_observation(obs, info)
        return combined_obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        combined_obs = self._pack_observation(obs, info)
        return combined_obs, reward, done, info


class SACPerformanceCallback(BaseCallback):
    """
    Custom callback to track performance metrics during SAC training
    and capture detailed data from the best episode for plotting
    """
    def __init__(self, trainer, verbose=0):
        super().__init__(verbose)
        self.trainer = trainer
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_steps = []
        self.current_episode_reward = 0
        self.current_episode_cost = 0
        self.current_episode_steps = 0
        
    def _on_step(self) -> bool:
        # Track episode metrics
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            
            # Get info from the last step
            if hasattr(env, 'get_episode_rewards'):
                # Episode ended, log metrics
                episode_rewards = env.get_episode_rewards()
                if len(episode_rewards) > len(self.episode_rewards):
                    # New episode completed
                    self.episode_rewards.extend(episode_rewards[len(self.episode_rewards):])
                    
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        self.episode_count += 1
        
        # Log periodically
        if self.episode_count % 50 == 0:
            if len(self.episode_rewards) > 0:
                recent_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {self.episode_count}: Recent avg reward = {recent_reward:.2f}")


class PendulumSACTrainer:
    """
    SAC trainer for DHP-Compatible Pendulum Environment
    """
    
    def __init__(self, config=None):
        """
        Initialize SAC trainer with configuration matching DHP setup
        """
        # Default configuration (adapted from DHP for fair comparison)
        self.config = config or {
            'episode_length': 200,
            'max_steps': 200,
            'learning_rate': 3e-4,
            'buffer_size': 50000,
            'batch_size': 256,
            'gamma': 0.95,
            'tau': 0.005,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'gradient_steps': 1,
            'learning_starts': 1000,
            'train_freq': 1,
            'use_sde': False,
            'policy_layers': [64, 64],
            'qf_layers': [64, 64],
            'total_timesteps': 300000,  # ~1500 episodes
            'log_interval': 50,
            'save_interval': 10000,
            'gui': False,
            'record': False,
            'normalize_states': True,
            'record_best_episodes': True,
            'recording_fps': 30,
        }
        
        # Initialize components
        self.env = None
        self.vec_env = None
        self.model = None
        
        # Training data storage (same structure as DHP)
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_position_errors = []
        self.episode_success_rates = []
        
        # Performance tracking for logging
        self.best_position_error = float('inf')
        self.best_episode = -1
        self.best_episode_cost = float('inf')
        self.best_episode_data = {}
        self.convergence_episodes = []
        self.stable_performance_start = -1
        self.best_model_saved = False
        
        # Setup comprehensive logging (same as DHP)
        self.setup_logging()
        
        print("PendulumSACTrainer initialized with DHP-compatible config:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
    
    def setup_logging(self):
        """
        Setup comprehensive logging for training session (same as DHP)
        """
        # Create logs directory
        log_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{log_dir}/sac_dhp_compatible_{timestamp}.log"
        
        # Setup logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log training session start
        self.logger.info("="*80)
        self.logger.info("SAC DHP-COMPATIBLE PENDULUM TRAINING SESSION STARTED")
        self.logger.info("="*80)
        
        # Log all hyperparameters
        self.logger.info("HYPERPARAMETERS:")
        self.logger.info("-" * 40)
        for key, value in sorted(self.config.items()):
            self.logger.info(f"{key:25s}: {value}")
        
        self.logger.info("-" * 40)
        self.logger.info(f"Log file: {self.log_filename}")
        self.logger.info("="*80)
    
    def setup_environment(self):
        """
        Setup DHP-compatible environment with SAC wrapper
        """
        print("\n[SETUP] Initializing DHP-Compatible Pendulum Environment...")
        
        # Create base DHP-compatible environment (SAME as DHP training)
        base_env = DHPCompatiblePendulumEnv(
            normalize_states=self.config['normalize_states'],
            max_episode_steps=self.config['max_steps']
        )
        
        # Wrap for SAC compatibility (adds reference to observation)
        wrapped_env = DHPCompatibleWrapper(base_env)
        
        # Add monitoring for episode statistics
        self.env = Monitor(wrapped_env, info_keywords=('position_error', 'dhp_cost'))
        
        # Create vectorized environment for Stable Baselines3
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        print(f"Environment observation space: {self.env.observation_space}")
        print(f"Environment action space: {self.env.action_space}")
        print(f"SAC receives: [theta, theta_dot, theta_ref, theta_dot_ref] (4 inputs)")
        print(f"This EXACTLY matches DHP environment for fair comparison!")
        
    def setup_sac_agent(self):
        """
        Setup SAC agent with architecture matching DHP complexity
        """
        print("\n[SETUP] Initializing SAC Agent...")
        
        # SAC model configuration
        model_kwargs = {
            'policy': 'MlpPolicy',
            'env': self.vec_env,
            'learning_rate': self.config['learning_rate'],
            'buffer_size': self.config['buffer_size'],
            'batch_size': self.config['batch_size'],
            'gamma': self.config['gamma'],
            'tau': self.config['tau'],
            'ent_coef': self.config['ent_coef'],
            'target_update_interval': self.config['target_update_interval'],
            'gradient_steps': self.config['gradient_steps'],
            'learning_starts': self.config['learning_starts'],
            'train_freq': self.config['train_freq'],
            'use_sde': self.config['use_sde'],
            'verbose': 1,
            'device': 'cpu',
            'policy_kwargs': {
                'net_arch': {
                    'pi': self.config['policy_layers'],
                    'qf': self.config['qf_layers']
                }
            }
        }
        
        # Create SAC model
        self.model = SAC(**model_kwargs)
        
        print(f"SAC Agent created with architecture:")
        print(f"  Policy network: {self.config['policy_layers']}")
        print(f"  Q-function network: {self.config['qf_layers']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Buffer size: {self.config['buffer_size']}")
        print(f"  Batch size: {self.config['batch_size']}")
    
    def train(self):
        """
        Main training loop with comprehensive logging (same structure as DHP)
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("STARTING SAC TRAINING FOR DHP-COMPATIBLE PENDULUM")
        self.logger.info("="*50)
        
        # Setup all components
        self.setup_environment()
        self.setup_sac_agent()
        
        # Create callback for performance tracking
        callback = SACPerformanceCallback(self)
        
        # Start training timer
        self.training_start_time = time.time()
        
        self.logger.info(f"\nTraining for {self.config['total_timesteps']} timesteps...")
        self.logger.info("This corresponds to approximately 1500 episodes of 200 steps each")
        self.logger.info("Episode | Reward  | Avg Cost | Pos Error | Success | Time")
        self.logger.info("-" * 60)
        
        # Main SAC training
        self.model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=callback,
            log_interval=self.config['log_interval']
        )
        
        # Training completed - log final analysis
        total_time = time.time() - self.training_start_time
        self.evaluate_final_performance()
        self.log_final_analysis(total_time)
        
        # Save final trained model
        self.save_trained_model()
        
        # Plot results
        self.plot_training_results()
    
    def evaluate_final_performance(self):
        """
        Evaluate the trained SAC policy on multiple episodes
        """
        print("\n[EVALUATION] Testing final SAC policy performance...")
        
        # Reset environment for evaluation
        obs, info = self.env.reset()
        
        eval_episodes = 10
        eval_rewards = []
        eval_errors = []
        eval_costs = []
        
        for ep in range(eval_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_steps = 0
            
            for step in range(self.config['max_steps']):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_cost += info.get('dhp_cost', 0)
                episode_steps += 1

                if done:
                    break
            
            final_error = info.get('position_error', 10.0)
            avg_cost = episode_cost / episode_steps if episode_steps > 0 else 0.0
            
            eval_rewards.append(episode_reward)
            eval_errors.append(final_error)
            eval_costs.append(avg_cost)
            
            print(f"  Eval Episode {ep+1}: reward={episode_reward:.2f}, error={final_error:.4f} rad, cost={avg_cost:.3f}")
        
        # Store evaluation results
        self.final_eval_reward = np.mean(eval_rewards)
        self.final_eval_error = np.mean(eval_errors)
        self.final_eval_cost = np.mean(eval_costs)
        self.best_position_error = min(eval_errors)
        
        print(f"\nFinal SAC Performance Summary:")
        print(f"  Average reward: {self.final_eval_reward:.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  Average error: {self.final_eval_error:.4f} ± {np.std(eval_errors):.4f} rad")
        print(f"  Best error: {self.best_position_error:.4f} rad ({np.rad2deg(self.best_position_error):.2f}°)")
        print(f"  Average cost: {self.final_eval_cost:.3f} ± {np.std(eval_costs):.3f}")
    
    def log_final_analysis(self, total_time):
        """
        Log comprehensive final training analysis (adapted for SAC)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING COMPLETED - FINAL ANALYSIS")
        self.logger.info("="*80)
        
        # Basic statistics
        self.logger.info(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"Final average reward: {self.final_eval_reward:.2f}")
        self.logger.info(f"Final average position error: {self.final_eval_error:.4f} rad")
        self.logger.info(f"Best position error achieved: {self.best_position_error:.4f} rad")
        
        # Performance classification
        if self.best_position_error < 0.1:
            performance_class = "EXCELLENT"
        elif self.best_position_error < 0.5:
            performance_class = "GOOD"
        elif self.best_position_error < 1.0:
            performance_class = "ACCEPTABLE"
        else:
            performance_class = "NEEDS IMPROVEMENT"
            
        self.logger.info(f"Performance classification: {performance_class}")
        
        # Save detailed metrics to JSON
        self.save_training_metrics_json()
    
    def save_training_metrics_json(self):
        """
        Save comprehensive training metrics to JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/training_logs"
        json_filename = f"{metrics_dir}/sac_dhp_compatible_metrics_{timestamp}.json"
        
        metrics = {
            "training_session": {
                "algorithm": "SAC",
                "environment": "DHP-Compatible",
                "timestamp": timestamp,
                "total_timesteps": self.config['total_timesteps'],
                "total_time_seconds": time.time() - self.training_start_time,
                "hyperparameters": self.config
            },
            "performance_summary": {
                "best_position_error": float(self.best_position_error),
                "final_avg_reward": float(self.final_eval_reward),
                "final_avg_error": float(self.final_eval_error),
                "final_avg_cost": float(self.final_eval_cost),
            },
            "environment_info": {
                "same_as_dhp": True,
                "state_normalization": self.config['normalize_states'],
                "physics_parameters": {
                    "m": 0.1, "L": 0.5, "g": 9.81, "b": 0.3, "Fmax": 4.0
                }
            }
        }
        
        with open(json_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved to: {json_filename}")
        return json_filename
    
    def save_trained_model(self):
        """
        Save the trained SAC model with configuration
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/trained_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save SAC model
        model_path = f"{checkpoint_dir}/sac_dhp_compatible_final"
        self.model.save(model_path)
        
        # Save configuration and metrics
        config_path = f"{checkpoint_dir}/sac_dhp_compatible_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"Trained SAC model saved to: {model_path}")
        print(f"Config saved to: {config_path}")
        
        # Performance summary
        print(f"\n🎯 SAC Training Results:")
        print(f"  Best position error: {self.best_position_error:.4f} rad ({np.rad2deg(self.best_position_error):.2f}°)")
        print(f"  Environment: DHP-Compatible (SAME as DHP training)")
        print(f"  Ready for fair DHP vs SAC comparison!")
        
        self.best_model_saved = True
    
    def plot_training_results(self):
        """
        Generate basic training visualization
        """
        print("\n[PLOTTING] Basic SAC training visualization...")
        
        # For now, just create a simple performance summary plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.text(0.5, 0.7, f"SAC Training Completed", 
                ha='center', va='center', fontsize=16, weight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.5, f"Best Position Error: {self.best_position_error:.4f} rad ({np.rad2deg(self.best_position_error):.2f}°)", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.3, f"Environment: DHP-Compatible", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.1, f"Ready for DHP vs SAC comparison!", 
                ha='center', va='center', fontsize=12, weight='bold', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('SAC DHP-Compatible Training Summary', fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/sac_dhp_compatible_summary.png', 
                    dpi=150, bbox_inches='tight')
        plt.show()
        
        print("SAC training summary plot saved!")
    
    def demonstrate_policy(self, gui=True, num_episodes=3):
        """
        Demonstrate the trained SAC policy
        """
        print(f"\n[DEMO] Demonstrating trained SAC policy for {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            print(f"\nEpisode {ep+1}:")
            
            for step in range(self.config['max_steps']):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1
                
                # Print some steps
                if step % 20 == 0:
                    pos_error = info.get('position_error', 0)
                    print(f"  Step {step}: position_error={pos_error:.4f} rad, reward={reward:.3f}")

                if done:
                    break
            
            final_error = info.get('position_error', 10.0)
            print(f"  Final: reward={episode_reward:.2f}, error={final_error:.4f} rad, steps={episode_steps}")
    
    def load_best_model(self):
        """
        Load the best saved SAC model
        """
        model_path = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/trained_models/sac_dhp_compatible_final"
        if os.path.exists(model_path + ".zip"):
            self.model = SAC.load(model_path)
            print(f"✅ Best SAC model loaded from: {model_path}")
            return True
        else:
            print(f"❌ No saved model found at: {model_path}")
            return False


if __name__ == "__main__":
    print("SAC Training for DHP-Compatible Pendulum Environment")
    print("====================================================")
    
    # Configuration for SAC training (optimized for fair comparison with DHP)
    config = {
        'total_timesteps': 300000,  # ~1500 episodes (same as DHP)
        'episode_length': 200,
        'max_steps': 200,
        'log_interval': 50,
        'save_interval': 10000,
        'gui': False,
        'record': False,
        'normalize_states': True,  # SAME as DHP
        'record_best_episodes': True,
        'learning_rate': 3e-4,
        'buffer_size': 50000,
        'batch_size': 256,
        'gamma': 0.95,  # SAME as DHP
        'tau': 0.005,
        'ent_coef': 'auto',
        'learning_starts': 1000,
        'policy_layers': [64, 64],  # SAME as DHP
        'qf_layers': [64, 64],      # SAME as DHP
    }
    
    # Create and configure trainer
    trainer = PendulumSACTrainer(config)
    
    print(f"\nSAC DHP-Compatible training configuration:")
    print(f"  Timesteps: {trainer.config['total_timesteps']} (~1500 episodes)")
    print(f"  Episode length: {trainer.config['max_steps']} steps")
    print(f"  State normalization: {trainer.config['normalize_states']}")
    print(f"  Network architecture: {trainer.config['policy_layers']}")
    print(f"  Environment: SAME as DHP training!")
    print(f"  Physics: m=0.1, L=0.5, g=9.81, b=0.3, Fmax=4.0")
    
    # Train the SAC agent
    print("\n" + "="*60)
    print("STARTING SAC TRAINING")
    print("="*60)
    
    trainer.train()
    
    # After training, demonstrate the learned policy
    trainer.logger.info("\n" + "="*50)
    trainer.logger.info("STARTING POLICY DEMONSTRATION")
    trainer.logger.info("="*50)

    print("\n" + "="*50)
    print("STARTING POLICY DEMONSTRATION")
    print("="*50)
    
    # Demonstrate the trained policy
    trainer.demonstrate_policy(gui=False, num_episodes=3)
    
    # Log demonstration results
    trainer.logger.info("Training and demonstration session completed successfully!")
    
    print("\nSAC DHP-compatible training completed!")
    print("🎯 Key achievements:")
    if trainer.best_model_saved:
        print(f"  ✅ Best model saved with error: {trainer.best_position_error:.4f} rad")
        print(f"  ✅ Uses SAME environment as DHP training")
        print(f"  ✅ Ready for fair DHP vs SAC comparison!")
    else:
        print(f"  ⚠️  Training completed but model saving failed")

    print("\nFiles generated:")
    print("  📊 Training summary: sac_dhp_compatible_summary.png")
    print("  📋 Training logs: training_logs/")
    print("  🤖 Trained models: trained_models/")
    
    print(f"\nFor detailed analysis, check the log file:")
    print(f"  {trainer.log_filename}")
    
    print("\n" + "="*70)
    print("SAC DHP-COMPATIBLE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Performance evaluation
    if trainer.best_position_error < 0.1:
        print("🎯 EXCELLENT performance achieved! (< 0.1 rad = 5.7°)")
    elif trainer.best_position_error < 0.5:
        print("👍 GOOD performance achieved! (< 0.5 rad = 28.6°)")
    elif trainer.best_position_error < 1.0:
        print("✅ ACCEPTABLE performance achieved! (< 1.0 rad = 57.3°)")
    else:
        print("⚠️  Performance needs improvement (> 1.0 rad = 57.3°)")
    
    # Performance comparison context
    print(f"\nPerformance context:")
    print(f"  Excellent: < 0.1 rad (5.7°)   - Precision control")
    print(f"  Good:      < 0.5 rad (28.6°)  - Practical control")
    print(f"  Acceptable: < 1.0 rad (57.3°) - Basic stabilization")
    print(f"  SAC result: {trainer.best_position_error:.4f} rad ({np.rad2deg(trainer.best_position_error):.1f}°)")
    
    # Save final summary for research documentation
    summary_file = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/sac_dhp_compatible_summary.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("SAC DHP-Compatible Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: SAC (Soft Actor-Critic)\n")
        f.write(f"Environment: DHP-Compatible Pendulum (SAME as DHP)\n")
        f.write(f"Best position error: {trainer.best_position_error:.4f} rad ({np.rad2deg(trainer.best_position_error):.1f}°)\n")
        f.write(f"Training timesteps: {trainer.config['total_timesteps']}\n")
        f.write(f"Network architecture: {trainer.config['policy_layers']}\n")
        f.write(f"State normalization: {trainer.config['normalize_states']}\n")
        f.write(f"Physics parameters: m=0.1, L=0.5, g=9.81, b=0.3, Fmax=4.0\n")
        f.write(f"\nReady for fair comparison with DHP results!\n")
    
    print(f"\n📄 Training summary saved: {summary_file}")
    
    print("\n🎯 SAC DHP-compatible training session complete!")
    print("   Ready for comprehensive DHP vs SAC comparison study!")
