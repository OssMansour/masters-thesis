"""
SAC Training Script for Pendulum Random Target Environment

This script adapts the SAC pipeline for pendulum control using the PendulumRandomTargetEnv
with the same architectural principles as the DHP implementation for fair comparison:
- State normalization system (critical for fair comparison)
- Best episode recording and replay
- Comprehensive training analysis and plots
- Session-best recording strategy
- Same logging and visualization structure

Author: DHP vs SAC Comparison Study  
Date: August 15, 2025
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

# Import our custom pendulum environment
from pendulum_env import PendulumRandomTargetEnv

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
    import torch
    print("✅ Stable Baselines3 imported successfully")
except ImportError as e:
    print(f"Error importing Stable Baselines3: {e}")
    print("Please install with: pip install stable-baselines3[extra]")
    sys.exit(1)


class StateReferenceWrapper(gym.Wrapper):
    """
    Combine env states [theta, theta_dot] and reference [theta_ref, theta_dot_ref]
    into a single 4-D observation for SAC.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _pack(self, obs, info=None):
        # Prefer the environment's reference from info (keeps it perfectly in sync)
        if info is not None and isinstance(info, dict) and 'reference' in info:
            reference = info['reference']
        elif hasattr(self.env, 'generate_reference'):
            reference = self.env.generate_reference()
        else:
            reference = np.array([0.0, 0.0], dtype=np.float32)

        obs = np.asarray(obs, dtype=np.float32).flatten()
        reference = np.asarray(reference, dtype=np.float32).flatten()

        # Safety: if an old 3D [cos,sin,thetadot] sneaks in, convert to [theta, theta_dot]
        if obs.shape[0] == 3:
            theta = np.arctan2(obs[1], obs[0])
            obs = np.array([theta, obs[2]], dtype=np.float32)
        if reference.shape[0] == 3:
            theta_ref = np.arctan2(reference[1], reference[0])
            reference = np.array([theta_ref, reference[2]], dtype=np.float32)

        return np.concatenate([obs, reference]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pack(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pack(obs, info), reward, terminated, truncated, info


class SACPerformanceCallback(BaseCallback):
    """
    Custom callback to track performance metrics during SAC training
    and capture detailed data from the best episode for plotting
    """
    def __init__(self, trainer, verbose=0):
        super().__init__(verbose)
        self.trainer = trainer
        self.current_episode_data = {
            'states': [],
            'references': [],
            'actions': [],
            'costs': [],
            'times': []
        }
        self.episode_start_time = 0.0
        
    def _on_step(self) -> bool:
        # Capture detailed data during training for potential best episode
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            try:
                # Get current observation from vectorized environment
                env = self.training_env.envs[0]
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'unwrapped'):
                    # Get the StateReferenceWrapper
                    wrapper_env = env.unwrapped
                    if hasattr(wrapper_env, 'observation_space') and wrapper_env.observation_space.shape[0] == 4:
                        last_obs = getattr(wrapper_env, '_last_obs', None)
                        if last_obs is not None and len(last_obs) == 4:
                            states = last_obs[:2]       # [theta, theta_dot]
                            references = last_obs[2:]   # [theta_ref, theta_dot_ref]
                            
                            # Get last action
                            last_action = getattr(wrapper_env, '_last_action', np.array([0.0]))
                            
                            # Store episode data
                            step_time = len(self.current_episode_data['states']) * 0.02
                            self.current_episode_data['states'].append(states.copy())
                            self.current_episode_data['references'].append(references.copy())
                            self.current_episode_data['actions'].append(last_action.copy())
                            self.current_episode_data['times'].append(step_time)
                            
                            # Get cost if available
                            base_env = wrapper_env.unwrapped
                            if hasattr(base_env, 'compute_dhp_cost'):
                                cost, _ = base_env.compute_dhp_cost(states, references)
                                self.current_episode_data['costs'].append(cost)
                            else:
                                self.current_episode_data['costs'].append(0.0)
            except Exception as e:
                # Silently continue if data capture fails
                pass
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each episode"""
        # Check if this episode had better performance
        if len(self.current_episode_data['states']) > 0:
            # Calculate episode metrics
            episode_length = len(self.current_episode_data['states'])
            avg_cost = np.mean(self.current_episode_data['costs']) if self.current_episode_data['costs'] else 1.0
            
            # Update trainer data for best episode tracking
            if avg_cost < getattr(self.trainer, 'best_episode_cost', float('inf')):
                self.trainer.best_episode_cost = avg_cost
                self.trainer.best_episode_data = {
                    'states': [s.copy() for s in self.current_episode_data['states']],
                    'references': [r.copy() for r in self.current_episode_data['references']],
                    'actions': [a.copy() for a in self.current_episode_data['actions']],
                    'costs': self.current_episode_data['costs'].copy(),
                    'times': self.current_episode_data['times'].copy()
                }
        
        # Clear current episode data
        self.current_episode_data = {
            'states': [],
            'references': [],
            'actions': [],
            'costs': [],
            'times': []
        }


class PendulumSACTrainer:
    """
    SAC trainer for Pendulum Random Target adapted from DHP implementation structure
    """
    
    def __init__(self, config=None):
        """
        Initialize SAC trainer with configuration
        """
        # Default configuration (adapted from DHP but optimized for SAC)
        self.config = config or {
            # Environment settings
            'fixed_target': None,         # None for random targets each episode
            'episode_length': 200,        # Standard pendulum episode length
            'max_steps': 200,             # Same as episode_length

            # SAC agent settings
            'learning_rate': 3e-4,        # Standard SAC learning rate
            'buffer_size': 50000,         # Replay buffer size
            'batch_size': 256,            # Batch size for updates
            'gamma': 0.95,                # Discount factor (same as DHP)
            'tau': 0.005,                 # Soft update coefficient
            'ent_coef': 'auto',           # Automatic entropy coefficient tuning
            'target_update_interval': 1,   # Update target networks every step
            'gradient_steps': 1,          # Gradient steps per environment step
            'learning_starts': 1000,      # Steps before learning starts
            'train_freq': 1,              # Training frequency
            'use_sde': False,             # State dependent exploration
            
            # Network architecture (matching DHP complexity)
            'policy_layers': [64, 64],    # Policy network layers
            'qf_layers': [64, 64],        # Q-function network layers
            
            # Training settings
            'total_timesteps': 300000,    # Total training timesteps (1500 episodes * 200 steps)
            'log_interval': 50,           # Episodes between logging
            'save_interval': 10000,       # Timesteps between saves
            'gui': False,                 # Training without GUI for speed
            'record': False,
            
            # State normalization settings (CRITICAL for fair comparison)
            'normalize_states': True,     # Enable state normalization (same as DHP)
            
            # Best episode recording settings
            'record_best_episodes': True,   # Enable recording of best episodes
            'recording_fps': 30,            # Recording frame rate
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
        
        print("PendulumSACTrainer initialized with config:")
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
        self.log_filename = f"{log_dir}/sac_pendulum_training_{timestamp}.log"
        
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
        self.logger.info("SAC PENDULUM TRAINING SESSION STARTED")
        self.logger.info("="*80)
        
        # Log all hyperparameters
        self.logger.info("HYPERPARAMETERS:")
        self.logger.info("-" * 40)
        for key, value in sorted(self.config.items()):
            self.logger.info(f"{key:25}: {value}")
        
        self.logger.info("-" * 40)
        self.logger.info(f"Log file: {self.log_filename}")
        self.logger.info("="*80)
    
    def setup_environment(self):
        """
        Setup Pendulum Random Target environment with state-reference wrapper
        """
        print("\n[SETUP] Initializing Pendulum Random Target Environment...")
        
        # Create base environment
        base_env = PendulumRandomTargetEnv(
            fixed_target=self.config['fixed_target'],
            normalize_states=self.config['normalize_states'],
            gui=self.config['gui'],
            record=self.config['record']
        )
        
        # Wrap base environment to include references
        wrapped_env = StateReferenceWrapper(base_env)
        
        # Add monitoring for episode statistics
        self.env = Monitor(wrapped_env, info_keywords=('position_error', 'dhp_cost'))
        
        # Create vectorized environment for Stable Baselines3
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        print(f"Environment observation space: {self.env.observation_space}")
        print(f"Environment action space: {self.env.action_space}")
        print(f"SAC now receives BOTH states AND references (6 inputs total)")
        print(f"This matches DHP input structure for fair comparison")
        
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
            'device': 'cpu',  # Use CPU to match DHP setup
            'policy_kwargs': {
                'net_arch': {
                    'pi': self.config['policy_layers'],  # Policy network
                    'qf': self.config['qf_layers']       # Q-function network
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
    
    def log_performance_metrics(self, episode, episode_reward, avg_cost, pos_error, success_rate):
        """
        Log performance metrics and track convergence indicators
        """
        # Update best performance and save best model
        if pos_error < self.best_position_error:
            self.best_position_error = pos_error
            self.best_episode = episode
            self.logger.info(f"NEW BEST PERFORMANCE! Episode {episode}: {pos_error:.4f} rad")
            
            # Automatically save the best model
            self.save_best_model(episode, pos_error)
        
        # Track convergence episodes (< 0.5 rad position error ≈ 28.6 degrees)
        if pos_error < 0.5:
            self.convergence_episodes.append(episode)
            self.logger.info(f"CONVERGENCE EPISODE {episode}: {pos_error:.4f} rad")
        
        # Check for stable performance (10 consecutive episodes < 0.5 rad)
        if len(self.convergence_episodes) >= 10:
            recent_convergent = self.convergence_episodes[-10:]
            if all(recent_convergent[i] == recent_convergent[0] + i for i in range(10)):
                if self.stable_performance_start == -1:
                    self.stable_performance_start = recent_convergent[0]
                    self.logger.info(f"STABLE PERFORMANCE ACHIEVED starting at episode {self.stable_performance_start}")
    
    def save_best_model(self, episode, pos_error):
        """
        Save the best performing model
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/trained_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save best model
        best_model_path = f"{checkpoint_dir}/sac_pendulum_best"
        self.model.save(best_model_path)
        
        # Save best performance metadata
        best_metadata = {
            'best_episode': episode,
            'best_position_error': float(pos_error),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{checkpoint_dir}/sac_pendulum_best_metadata.pkl", 'wb') as f:
            pickle.dump(best_metadata, f)
        
        self.logger.info(f"BEST MODEL SAVED! Episode {episode}, Error: {pos_error:.4f} rad")
        self.logger.info(f"Best model path: {best_model_path}")
        
        # Mark that best model has been saved
        self.best_model_saved = True
    
    def train(self):
        """
        Main training loop with comprehensive logging (same structure as DHP)
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("STARTING SAC TRAINING FOR PENDULUM")
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
        
        # Custom training loop to track episodes
        episode_count = 0
        timestep_count = 0
        
        while timestep_count < self.config['total_timesteps']:
            # Train for one episode worth of timesteps
            episode_timesteps = min(self.config['episode_length'], 
                                   self.config['total_timesteps'] - timestep_count)
            
            try:
                # Learn for this batch of timesteps
                self.model.learn(
                    total_timesteps=episode_timesteps,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=False
                )
            except RuntimeError as e:
                if "Tried to step environment that needs reset" in str(e):
                    self.logger.error(f"Environment reset error at episode {episode_count}. Attempting to recover...")
                    # Force environment reset
                    try:
                        self.env.reset()
                        self.logger.info("Environment reset successful, continuing training...")
                        continue  # Skip this training iteration and continue
                    except Exception as reset_error:
                        self.logger.error(f"Failed to reset environment: {reset_error}")
                        break  # Exit training loop
                else:
                    # Re-raise other RuntimeErrors
                    raise e
            except Exception as e:
                self.logger.error(f"Unexpected error during training: {e}")
                break
            
            timestep_count += episode_timesteps
            episode_count += 1
            
            # Evaluate current policy every few episodes
            if episode_count % self.config['log_interval'] == 0:
                self.evaluate_current_policy(episode_count)
        
        # Training completed - log final analysis
        total_time = time.time() - self.training_start_time
        self.log_final_analysis(total_time)
        
        # Save final trained model
        self.save_trained_model()
        
        # Plot results
        self.plot_training_results()
    
    def evaluate_current_policy(self, episode_num):
        """
        Evaluate current policy performance
        """
        # Reset environment for evaluation
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        
        for step in range(self.config['max_steps']):
            # Get action from current policy
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Accumulate metrics
            episode_reward += reward
            episode_cost += info.get('dhp_cost', 0.0)
            episode_steps += 1
            
            # Check termination
            if terminated or truncated:
                break
        
        # Store episode metrics
        self.episode_rewards.append(episode_reward)
        if episode_steps > 0:
            self.episode_costs.append(episode_cost / episode_steps)
        else:
            self.episode_costs.append(1.0)
        
        final_pos_error = info.get('position_error', 10.0)
        final_success = info.get('episode_success', False)
        self.episode_position_errors.append(final_pos_error)
        self.episode_success_rates.append(1.0 if final_success else 0.0)
        
        # Log performance metrics
        self.log_performance_metrics(
            episode_num, episode_reward, 
            episode_cost / episode_steps if episode_steps > 0 else 1.0,
            final_pos_error, 1.0 if final_success else 0.0
        )
        
        # Periodic logging
        elapsed_time = time.time() - self.training_start_time
        log_msg = (f"{episode_num:7d} | {episode_reward:7.2f} | {float(episode_cost / episode_steps if episode_steps > 0 else 1.0):8.3f} | "
                  f"{float(final_pos_error):9.4f} | {float(1.0 if final_success else 0.0):7.3f} | {elapsed_time:6.1f}s")
        self.logger.info(log_msg)
    
    def log_final_analysis(self, total_time):
        """
        Log comprehensive final training analysis (adapted for SAC)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING COMPLETED - FINAL ANALYSIS")
        self.logger.info("="*80)
        
        # Basic statistics
        self.logger.info(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        final_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
        self.logger.info(f"Final episode reward: {final_reward:.2f}")
        self.logger.info(f"Final position error: {self.episode_position_errors[-1] if self.episode_position_errors else 0.0:.4f} rad")
        self.logger.info(f"Final success rate: {self.episode_success_rates[-1] if self.episode_success_rates else 0.0:.3f}")
        self.logger.info(f"Best position error achieved: {self.best_position_error:.4f} rad at episode {self.best_episode}")
        
        # Convergence analysis
        total_convergent = len(self.convergence_episodes)
        convergence_rate = total_convergent / len(self.episode_position_errors) if self.episode_position_errors else 0
        avg_success = np.mean(self.episode_success_rates[-100:]) if len(self.episode_success_rates) >= 100 else np.mean(self.episode_success_rates) if self.episode_success_rates else 0
        
        self.logger.info(f"Total convergent episodes (< 0.5 rad): {total_convergent}/{len(self.episode_position_errors)} ({convergence_rate:.2%})")
        self.logger.info(f"Recent success rate (last 100 episodes): {avg_success:.2%}")
        
        # Save detailed metrics to JSON
        self.save_training_metrics_json()
    
    def save_training_metrics_json(self):
        """
        Save comprehensive training metrics to JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/training_logs"
        json_filename = f"{metrics_dir}/sac_pendulum_metrics_{timestamp}.json"
        
        metrics = {
            "training_session": {
                "algorithm": "SAC",
                "timestamp": timestamp,
                "total_episodes": len(self.episode_position_errors),
                "total_time_seconds": time.time() - self.training_start_time,
                "hyperparameters": self.config
            },
            "performance_summary": {
                "best_position_error": float(self.best_position_error),
                "best_episode": int(self.best_episode),
                "final_position_error": float(self.episode_position_errors[-1]) if self.episode_position_errors else None,
                "final_reward": float(self.episode_rewards[-1]) if self.episode_rewards else None,
                "final_success_rate": float(self.episode_success_rates[-1]) if self.episode_success_rates else None,
                "convergent_episodes_count": len(self.convergence_episodes),
                "convergence_rate": len(self.convergence_episodes) / len(self.episode_position_errors) if self.episode_position_errors else 0,
                "stable_performance_start": self.stable_performance_start
            },
            "time_series_data": {
                "episode_position_errors": [float(x) for x in self.episode_position_errors],
                "episode_rewards": [float(x) for x in self.episode_rewards],
                "episode_costs": [float(x) for x in self.episode_costs],
                "episode_success_rates": [float(x) for x in self.episode_success_rates],
                "convergent_episodes": [int(x) for x in self.convergence_episodes]
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
        
        # Save final SAC model
        model_path = f"{checkpoint_dir}/sac_pendulum_final"
        self.model.save(model_path)
        
        # Save configuration and metrics
        np.savez(f"{checkpoint_dir}/sac_pendulum_training_summary.npz",
                 config=self.config,
                 rewards=self.episode_rewards,
                 costs=self.episode_costs,
                 position_errors=self.episode_position_errors,
                 success_rates=self.episode_success_rates)
        
        # Save complete config for easy loading
        with open(f"{checkpoint_dir}/sac_pendulum_config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"Trained SAC model saved to: {model_path}")
        print(f"Training summary saved to: {checkpoint_dir}/sac_pendulum_training_summary.npz")
        print(f"Config saved to: {checkpoint_dir}/sac_pendulum_config.pkl")
        
        # Print normalization status for user
        if self.config['normalize_states']:
            print("✅ Model saved with state normalization ENABLED")
            print("  Demo script will automatically apply the same normalization")
            self.logger.info("Model saved with state normalization ENABLED")
        else:
            print("✅ Model saved with state normalization DISABLED") 
            print("  Demo script will use raw states")
            self.logger.info("Model saved with state normalization DISABLED")
        
        # Print best model information
        if self.best_model_saved:
            print(f"✅ Best model automatically saved from episode {self.best_episode}")
            print(f"  Best position error: {self.best_position_error:.4f} rad ({np.rad2deg(self.best_position_error):.1f}°)")
            print(f"  Use sac_pendulum_best.zip for best performance")
        else:
            print("⚠️ No best model saved - no recordings available")
    
    def plot_training_results(self):
        """
        Plot training metrics in the same style as DHP for comparison
        """
        print("\n[PLOTTING] Generating SAC pendulum training analysis plots...")
        
        if len(self.episode_rewards) == 0:
            print("No training data available for plotting")
            return
        
        # Set colors
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        plt.rcParams['figure.figsize'] = 25/2.54, 50/2.54
        
        ### Figure 1: Training Progress - Episode Rewards, Costs, and Success Rates ###
        fig1 = plt.figure()
        episodes = np.arange(len(self.episode_rewards))
        
        ax1 = fig1.add_subplot(3,1,1)
        ax1.plot(episodes, self.episode_rewards, color=colors[0], linewidth=2, label='Episode Rewards (SAC)')
        ax1.set_ylabel(r'Episode Reward')
        ax1.set_title('SAC Pendulum Training Progress - Rewards, Costs, and Success Rates')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        ax2.plot(episodes, self.episode_costs, color=colors[1], linewidth=2, label='Episode Average Cost (SAC)')
        ax2.set_ylabel(r'Episode Average Cost')
        ax2.set_yscale('log')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3,1,3, sharex=ax1)
        ax3.plot(episodes, self.episode_success_rates, color=colors[2], linewidth=2, label='Episode Success Rate (SAC)')
        ax3.set_ylabel(r'Episode Success Rate')
        ax3.set_xlabel(r'Episode Number')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        ### Figure 2: Performance Analysis ###
        fig2 = plt.figure(figsize=(12, 10))
        
        # Position error over time
        ax1 = fig2.add_subplot(2,2,1)
        ax1.plot(episodes, self.episode_position_errors, color=colors[0], alpha=0.7, label='Position Error (SAC)')
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Convergence Threshold (0.5 rad)')
        # Mark convergence episodes
        if self.convergence_episodes:
            conv_errors = [self.episode_position_errors[i] for i in self.convergence_episodes if i < len(self.episode_position_errors)]
            conv_episodes = [i for i in self.convergence_episodes if i < len(self.episode_position_errors)]
            ax1.scatter(conv_episodes, conv_errors, color='green', s=20, alpha=0.7, label='Convergent Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Position Error [rad]')
        ax1.set_title('SAC Learning Curve with Convergence Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Position error distribution
        ax2 = fig2.add_subplot(2,2,2)
        ax2.hist(np.array(self.episode_position_errors)*180/np.pi, bins=30, alpha=0.7, color=colors[0])
        ax2.set_xlabel('Position Error [degrees]')
        ax2.set_ylabel('Frequency')
        ax2.set_title('SAC Position Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Success rate over time
        ax3 = fig2.add_subplot(2,2,3)
        # Calculate rolling success rate (50-episode window)
        if len(self.episode_success_rates) >= 50:
            rolling_success = []
            for i in range(49, len(self.episode_success_rates)):
                rolling_success.append(np.mean(self.episode_success_rates[i-49:i+1]))
            ax3.plot(range(49, len(self.episode_success_rates)), rolling_success, color=colors[2], linewidth=2)
        ax3.plot(self.episode_success_rates, alpha=0.3, color=colors[2])
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('SAC Success Rate Evolution (50-episode rolling mean)')
        ax3.grid(True, alpha=0.3)
        
        # Episode rewards distribution
        ax4 = fig2.add_subplot(2,2,4)
        ax4.hist(self.episode_rewards, bins=30, alpha=0.7, color=colors[1])
        ax4.set_xlabel('Episode Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('SAC Episode Reward Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        ### Figure 3: Best Episode Analysis (if available) ###
        if hasattr(self, 'best_episode_data') and self.best_episode_data:
            fig3 = plt.figure(figsize=(12, 10))
            
            # Extract best episode data
            states = np.array(self.best_episode_data['states'])         # shape [T, 2]
            references = np.array(self.best_episode_data['references']) # shape [T, 2]
            actions = np.array(self.best_episode_data['actions'])       # shape [T, 1]
            costs = np.array(self.best_episode_data['costs'])
            times = np.array(self.best_episode_data['times'])


            theta = states[:, 0]
            theta_ref = references[:, 0]
            theta_dot = states[:, 1]
            theta_dot_ref = references[:, 1]

            
            # Angle tracking
            ax1 = fig3.add_subplot(4,1,1)
            ax1.plot(times, theta*180/np.pi, 'b-', label=r'$\theta$ (SAC Best Episode)')
            ax1.plot(times, theta_ref*180/np.pi, 'r--', label=r'$\theta_{ref}$ (SAC Best Episode)')
            ax1.set_ylabel(r'Angle $[deg]$')
            ax1.set_title(f'SAC Pendulum Best Episode Performance (Ep {self.best_episode}, Error: {np.rad2deg(self.best_position_error):.1f}°)')
            ax1.legend(loc='upper right')
            ax1.grid(True)
            
            # Angular velocity tracking
            ax2 = plt.subplot(4,1,2, sharex=ax1)
            ax2.plot(times, theta_dot*180/np.pi, 'b-', label=r'$\dot{\theta}$ (SAC Best Episode)')
            ax2.plot(times, theta_dot_ref*180/np.pi, 'r--', label=r'$\dot{\theta}_{ref}$ (SAC Best Episode)')
            ax2.set_ylabel(r'Angular Velocity $[deg/s]$')
            ax2.legend(loc='upper right')
            ax2.grid(True)
            
            # Control actions
            ax3 = plt.subplot(4,1,3, sharex=ax1)
            ax3.plot(times, actions[:, 0], 'g-', label=r'Torque (SAC Best Episode)')
            ax3.set_ylabel(r'Control Torque $[Nm]$')
            ax3.legend(loc='upper right')
            ax3.grid(True)
            
            # Costs
            ax4 = plt.subplot(4,1,4, sharex=ax1)
            ax4.plot(times, costs, 'r-', label=r'Cost (SAC Best Episode)')
            ax4.set_xlabel(r'$t [s]$')
            ax4.set_ylabel(r'Cost $[-]$')
            ax4.set_yscale('log')
            ax4.legend(loc='upper right')
            ax4.grid(True)
            
            plt.tight_layout()
            fig3.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/sac_pendulum_best_episode.png', dpi=150, bbox_inches='tight')
        
        # Align labels and show plots
        fig1.align_labels()
        fig2.align_labels()
        
        # Save figures
        fig1.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/sac_pendulum_training_progress.png', dpi=150, bbox_inches='tight')
        fig2.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/sac_pendulum_performance_analysis.png', dpi=150, bbox_inches='tight')
        
        plt.show()
        
        print("SAC PENDULUM training analysis plots generated and saved!")
        print("Figures saved:")
        print("  - Training Progress: sac_pendulum_training_progress.png")
        print("  - Performance Analysis: sac_pendulum_performance_analysis.png")
        if hasattr(self, 'best_episode_data') and self.best_episode_data:
            print(f"  - Best Episode Analysis: sac_pendulum_best_episode.png (Episode {self.best_episode})")
        print(f"All plots show SAC performance for comparison with DHP")
    
    def demonstrate_policy(self, gui=True, record=True, episode_length=200, real_time=True):
        """
        Demonstrate the trained SAC policy with visualization
        """
        print("\n" + "="*50)
        print("DEMONSTRATING TRAINED SAC PENDULUM POLICY")
        print("="*50)
        
        # Load the best model for demonstration
        self.load_best_model()
        
        # Create demonstration environment with GUI
        demo_env = PendulumRandomTargetEnv(
            fixed_target=np.pi/4,  # Random target for demonstration
            normalize_states=self.config['normalize_states'],
            gui=gui,
            record=record
        )
        
        # Wrap for SAC (same as training)
        demo_env = StateReferenceWrapper(demo_env)
        
        # Reset environment
        obs, info = demo_env.reset()
        
        max_steps = episode_length
        demo_states = []
        demo_actions = []
        demo_costs = []
        demo_times = []
        
        print(f"Running demonstration for {episode_length} steps...")
        if real_time:
            print("Running in real-time mode")
        else:
            print("Running in fast mode")
        
        # Track real time for proper visualization
        start_real_time = time.time()
        
        for step in range(max_steps):
            # Get action from trained SAC policy
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = demo_env.step(action)
            
            # Store demo data (extract states from combined observation)
            states = obs[:3]  # First 3 elements are states
            demo_states.append(states.copy())
            demo_actions.append(action.copy())
            demo_costs.append(info.get('dhp_cost', 0.0))
            demo_times.append(step * 0.02)  # Assuming ~0.02s per step
            
            # Update for next step
            obs = next_obs
            
            # Real-time synchronization for smooth visualization
            if real_time and gui:
                # Calculate how much time should have passed
                expected_time = step * 0.02
                elapsed_real_time = time.time() - start_real_time
                
                # Sleep if we're running too fast
                if elapsed_real_time < expected_time:
                    time.sleep(expected_time - elapsed_real_time)
            
            # Print progress
            if step % 40 == 0:  # Every ~2 seconds
                pos_error = info.get('position_error', 0.0)
                print(f"Step: {step:3d}, Angle Error: {pos_error:.3f} rad ({np.rad2deg(pos_error):.1f}°), Cost: {info.get('dhp_cost', 0.0):.3f}")
            
            # Check termination
            if terminated or truncated:
                break
        
        demo_env.close()
        
        # Print final statistics
        final_pos_error = info.get('position_error', 0.0)
        final_success = info.get('episode_success', False)
        avg_cost = np.mean(demo_costs) if demo_costs else 0.0
        print(f"\nDemonstration completed!")
        print(f"Final position error: {final_pos_error:.4f} rad ({np.rad2deg(final_pos_error):.1f}°)")
        print(f"Episode success: {final_success}")
        print(f"Average cost: {avg_cost:.3f}")
        
        if record:
            print("Recording saved (if environment supports it)")
        
        return demo_states, demo_actions, demo_costs, demo_times
    
    def load_best_model(self):
        """
        Load the best saved SAC model for demonstration
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/trained_models"
        
        # Try to load best model first
        best_model_path = f"{checkpoint_dir}/sac_pendulum_best.zip"
        if os.path.exists(best_model_path):
            print("Loading best saved SAC model for demonstration...")
            try:
                with open(f"{checkpoint_dir}/sac_pendulum_best_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                    print(f"Best model: Episode {metadata['best_episode']}, Error: {np.rad2deg(metadata['best_position_error']):.1f}°")
                
                self.model = SAC.load(best_model_path)
                print("✅ Best SAC model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error loading best model: {e}")
        
        # Fallback to final model
        final_model_path = f"{checkpoint_dir}/sac_pendulum_final.zip"
        if os.path.exists(final_model_path):
            print("Loading final SAC model for demonstration...")
            try:
                self.model = SAC.load(final_model_path)
                print("✅ Final SAC model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error loading final model: {e}")
        
        print("⚠️ No saved model found, using current model...")
        return False


if __name__ == "__main__":
    print("SAC Training for Pendulum Random Target Environment")
    print("===================================================")
    
    # Configuration for pendulum SAC training (optimized for fair comparison with DHP)
    config_override = {
        # Training settings optimized for pendulum
        'total_timesteps': 300000,      # 1500 episodes * 200 steps
        'episode_length': 200,          # Standard pendulum episode length
        'max_steps': 200,
        'log_interval': 50,             # More frequent logging for shorter episodes
        'save_interval': 10000,
        'gui': False,                   # Training without GUI for speed
        'record': False,
        
        # State normalization settings (CRITICAL for fair comparison)
        'normalize_states': True,       # ENABLE normalization (same as DHP)
        
        # Best episode recording
        'record_best_episodes': True,   # Enable recording of best episodes
        
        # SAC hyperparameters (optimized for pendulum)
        'learning_rate': 3e-4,          # Standard SAC learning rate
        'buffer_size': 50000,           # Sufficient for 300k timesteps
        'batch_size': 256,              # Standard batch size
        'gamma': 0.95,                  # Same as DHP for fair comparison
        'tau': 0.005,                   # Standard SAC soft update
        'ent_coef': 'auto',             # Automatic entropy tuning
        'learning_starts': 1000,        # Start learning after 1000 steps
        'policy_layers': [64, 64],      # Same complexity as DHP
        'qf_layers': [64, 64],          # Same complexity as DHP
    }
    
    # Create trainer with default config, then update with overrides
    trainer = PendulumSACTrainer()
    trainer.config.update(config_override)
    
    # Log the configuration update
    trainer.logger.info("\nCONFIGURATION OVERRIDES APPLIED:")
    for key, value in config_override.items():
        trainer.logger.info(f"  {key}: {value}")
    
    print("\nUpdated config for pendulum SAC training:")
    for key, value in config_override.items():
        print(f"  {key}: {value}")
    
    print(f"\nPendulum SAC training configuration:")
    print(f"  Timesteps: {trainer.config['total_timesteps']} (~1500 episodes)")
    print(f"  Episode length: {trainer.config['max_steps']} steps")
    print(f"  State normalization: {trainer.config['normalize_states']}")
    print(f"  Random targets: {trainer.config['fixed_target'] is None}")
    print(f"  Success threshold: < 0.1 rad (5.7°)")
    print(f"  Convergence threshold: < 0.5 rad (28.6°)")
    print(f"  Network architecture: {trainer.config['policy_layers']} (matches DHP)")
    
    # Train the SAC agent
    trainer.train()
    
    # After training, demonstrate the learned policy with GUI
    trainer.logger.info("\n" + "="*50)
    trainer.logger.info("STARTING POLICY DEMONSTRATION")
    trainer.logger.info("="*50)

    print("\n" + "="*50)
    print("STARTING POLICY DEMONSTRATION")
    print("="*50)
    
    # Demonstrate the trained policy with visualization
    demo_states, demo_actions, demo_costs, demo_times = trainer.demonstrate_policy(
        gui=True,           # Show GUI for visualization
        record=True,        # Record if environment supports it
        episode_length=200, # Standard episode length
        real_time=False     # Fast mode for quick demonstration
    )
    
    # Log demonstration results
    if demo_states:
        # Calculate final angle from cos/sin
        final_cos = demo_states[-1][0]
        final_sin = demo_states[-1][1]
        final_angle = np.arctan2(final_sin, final_cos)
        demo_error = abs(final_angle)  # Error from vertical (target would be variable)
        trainer.logger.info(f"Demonstration final angle error: {np.rad2deg(demo_error):.1f}°")
    trainer.logger.info("Training and demonstration session completed successfully!")
    
    print("\nSAC pendulum training and demonstration completed!")
    print("Check the saved plots for detailed analysis results.")
    print(f"Detailed training log: {trainer.log_filename}")
    print("🎯 Key achievements:")
    if trainer.best_model_saved:
        print(f"  ✅ Best model saved: Episode {trainer.best_episode}")
        print(f"  ✅ Best position error: {trainer.best_position_error:.4f} rad ({np.rad2deg(trainer.best_position_error):.1f}°)")
        print(f"  ✅ Convergent episodes: {len(trainer.convergence_episodes)}")
        
        if trainer.convergence_episodes:
            convergence_rate = len(trainer.convergence_episodes) / len(trainer.episode_position_errors)
            print(f"  ✅ Convergence rate: {convergence_rate:.1%}")
        
        if trainer.stable_performance_start != -1:
            print(f"  ✅ Stable performance since episode: {trainer.stable_performance_start}")
    else:
        print("  ⚠️ No significant improvement achieved during training")
        print("  💡 Consider adjusting hyperparameters or extending training")

    print("\nFiles generated:")
    print("  📊 Training plots: sac_pendulum_*.png")
    print("  📋 Training logs: training_logs/")
    print("  🤖 Trained models: trained_models/")
    
    print(f"\nFor detailed analysis, check the log file:")
    print(f"  {trainer.log_filename}")
    
    print("\n" + "="*60)
    print("SAC PENDULUM TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Additional analysis for research purposes
    if trainer.best_position_error < 0.1:  # Excellent performance (< 5.7°)
        print("\n🏆 EXCELLENT PERFORMANCE ACHIEVED!")
        print(f"   Final error: {trainer.best_position_error:.4f} rad ({np.rad2deg(trainer.best_position_error):.1f}°)")
        print("   This represents high-precision pendulum control suitable for real-world applications.")
    elif trainer.best_position_error < 0.5:  # Good performance (< 28.6°)
        print("\n✅ GOOD PERFORMANCE ACHIEVED!")
        print(f"   Final error: {trainer.best_position_error:.4f} rad ({np.rad2deg(trainer.best_position_error):.1f}°)")
        print("   This represents acceptable pendulum control for most applications.")
    else:
        print("\n⚠️ PERFORMANCE COULD BE IMPROVED")
        print(f"   Final error: {trainer.best_position_error:.4f} rad ({np.rad2deg(trainer.best_position_error):.1f}°)")
        print("   Consider extending training or adjusting hyperparameters.")
    
    # Performance comparison context
    print(f"\nPerformance context:")
    print(f"  Excellent: < 0.1 rad (5.7°)   - Precision control")
    print(f"  Good:      < 0.5 rad (28.6°)  - Practical control")
    print(f"  Acceptable: < 1.0 rad (57.3°) - Basic stabilization")
    print(f"  Your result: {trainer.best_position_error:.4f} rad ({np.rad2deg(trainer.best_position_error):.1f}°)")
    
    # Save final summary for research documentation
    summary_file = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/sac_training_summary.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("SAC Pendulum Training Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Episodes trained: {len(trainer.episode_position_errors)}\n")
        f.write(f"Best episode: {trainer.best_episode}\n")
        f.write(f"Best position error: {trainer.best_position_error:.6f} rad ({np.rad2deg(trainer.best_position_error):.2f}°)\n")
        f.write(f"Convergent episodes: {len(trainer.convergence_episodes)}\n")
        if trainer.convergence_episodes:
            convergence_rate = len(trainer.convergence_episodes) / len(trainer.episode_position_errors)
            f.write(f"Convergence rate: {convergence_rate:.1%}\n")
        f.write(f"State normalization: {trainer.config['normalize_states']}\n")
        f.write(f"Network layers: {trainer.config['policy_layers']}\n")
        f.write(f"Learning rate: {trainer.config['learning_rate']}\n")
        f.write(f"Training time: {time.time() - trainer.training_start_time:.1f} seconds\n")
        f.write(f"Log file: {trainer.log_filename}\n")
    
    print(f"\n📄 Training summary saved: {summary_file}")
    
    # Optional: Quick demonstration if requested
    user_input = input("\nWould you like to run a quick demonstration of the trained SAC policy? (y/n): ").strip().lower()
    if user_input in ['y', 'yes']:
        print("\n" + "="*50)
        print("RUNNING QUICK SAC DEMONSTRATION")
        print("="*50)
        
        try:
            demo_states, demo_actions, demo_costs, demo_times = trainer.demonstrate_policy(
                gui=True,           # Show visualization
                record=False,       # Don't record (quick demo)
                episode_length=200, # Standard episode length
                real_time=True      # Real-time for smooth visualization
            )
            
            if demo_states:
                # Quick demo analysis
                final_cos = demo_states[-1][0]
                final_sin = demo_states[-1][1]
                final_angle = np.arctan2(final_sin, final_cos)
                demo_error = abs(final_angle)
                
                print(f"\n📊 Quick demo results:")
                print(f"  Final angle error: {demo_error:.4f} rad ({np.rad2deg(demo_error):.1f}°)")
                print(f"  Average cost: {np.mean(demo_costs):.3f}")
                print(f"  Demo completed successfully!")
            else:
                print("Demo completed but no state data captured.")
                
        except Exception as e:
            print(f"Demo failed: {e}")
            print("The trained model is still saved and can be used later.")
    
    print("\n🎯 SAC training session complete! Ready for DHP vs SAC comparison study.")