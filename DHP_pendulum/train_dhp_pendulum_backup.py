"""
DHP Training Script for Pendulum Random Target Environment

This script adapts the successful CF2X DHP training pipeline for pendulum control
using the PendulumRandomTargetEnv with the same architectural principles:
- State normalization system (critical for DHP success)
- Best episode recording and replay
- Comprehensive training analysis and plots
- Session-best recording strategy
- Same logging and visualization as CF2X

Author: DHP vs SAC Comparison Study  
Date: August 15, 2025
"""
# Import utilities
import tensorflow as tf
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

# Import existing DHP components from msc-thesis
from agents.dhp import Agent as DHP_Agent
from agents.model import RecursiveLeastSquares

# Import our custom pendulum environment
from pendulum_env import PendulumRandomTargetEnv


class PendulumDHPTrainer:
    """
    DHP trainer for Pendulum Random Target adapted from successful CF2X implementation
    """
    
    def __init__(self, config=None):
        """
        Initialize DHP trainer with configuration
        """
        # Default configuration (adapted from CF2X but optimized for pendulum)
        self.config = config or {
            # Environment settings
            'fixed_target': np.pi,         # None for random targets each episode
            'episode_length': 200,        # Standard pendulum episode length
            'max_steps': 200,             # Same as episode_length

            # DHP agent settings (2D state space: [theta, theta_dot])
            'state_size': 2,              # Pendulum states [theta, theta_dot] 
            'reference_size': 2,          # Reference vector [theta_ref, theta_dot_ref]
            'action_size': 1,             # Single torque control
            'hidden_layer_size': [64, 64, 32],  # Same architecture as CF2X
            'lr_critic': 0.01,           # Same proven learning rates as CF2X
            'lr_actor': 0.005,           # Same proven learning rates as CF2X
            'activation': 'relu',  # or 'relu', or whatever your agent expects
            'gamma': 0.95,               # Discount factor (slightly lower than CF2X for faster episodes)
            'split': False,              # Start with unified architecture for pendulum
            'target_network': True,
            'tau': 0.001,               # Same as CF2X
            
            # RLS model settings
            'rls_state_size': 2,          # RLS model state size
            'rls_action_size': 1,
            'rls_gamma': 0.9995,
            'rls_covariance': 100.0,
            'predict_delta': False,
            
            # Training settings (adapted for pendulum learning)
            'num_episodes': 1500,        # More episodes for pendulum convergence
            'update_cycles': 2,          # Same as CF2X
            'excitation_steps': 7500,    # Exploration phase (50% of training)
            'excitation_amplitude': 0.2, # Reasonable excitation for pendulum torque range
            
            # State normalization settings (CRITICAL for DHP success)
            'normalize_states': True,     # Enable state normalization
            # State normalization bounds (pendulum-specific)
            'state_bounds': {
                'theta': [-np.pi, np.pi],     # theta natural bounds
                'theta_dot': [-8.0, 8.0]      # Angular velocity limit (pendulum max_speed)
            },
            
            # Logging
            'log_interval': 50,          # More frequent logging for shorter episodes
            'save_interval': 200,
            'gui': False,                # No GUI during training for speed
            'record': False,             # Don't record all episodes
            
            # Best episode recording settings
            'record_best_episodes': True,   # Enable recording of best episodes
            'best_episode_gui': False,      # Show GUI only for best episodes
            'recording_fps': 30,            # Recording frame rate
        }
        
        # Initialize components
        self.env = None
        self.agent = None
        self.model = None
        
        # Training data storage (same structure as CF2X)
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_position_errors = []
        self.episode_success_rates = []  # Additional metric for pendulum
        
        # Performance tracking for logging
        self.best_position_error = float('inf')
        self.best_episode = -1
        self.convergence_episodes = []  # Episodes with pos_error < 0.5 rad
        self.stable_performance_start = -1  # Episode when stable performance begins
        self.best_model_saved = False  # Track if best model has been saved
        
        # Setup comprehensive logging (same as CF2X)
        self.setup_logging()
        
        # Detailed training data for dhp_main.py style plots (from BEST episode only)
        self.detailed_states = []
        self.detailed_references = []
        self.detailed_actions = []
        self.detailed_costs = []
        self.detailed_model_errors = []
        self.detailed_actor_grads = []
        self.detailed_critic_grads = []
        self.detailed_F_matrices = []
        self.detailed_G_matrices = []
        self.detailed_rls_variance = []
        self.detailed_time = []
        
        # Best episode data storage (for plotting the best performance)
        self.best_episode_states = []
        self.best_episode_references = []
        self.best_episode_actions = []
        self.best_episode_costs = []
        self.best_episode_model_errors = []
        self.best_episode_actor_grads = []
        self.best_episode_critic_grads = []
        self.best_episode_F_matrices = []
        self.best_episode_G_matrices = []
        self.best_episode_rls_variance = []
        self.best_episode_time = []
        
        # Best episode RLS prediction data (for model accuracy analysis)
        self.best_episode_rls_predictions = []
        self.best_episode_rls_ground_truth = []
        
        # Session-best recording data (for replay-at-end strategy)
        self.session_best_episode_num = -1
        self.session_best_error = float('inf')
        self.session_best_actions_sequence = []  # Action sequence for replay
        self.session_best_initial_conditions = None  # Initial conditions for perfect replay
        self.session_best_target_angle = None  # Target angle for replay
        
        # Current episode data collection (temporary storage)
        self.current_episode_data = {
            'states': [],
            'references': [],
            'actions': [],
            'costs': [],
            'model_errors': [],
            'actor_grads': [],
            'critic_grads': [],
            'F_matrices': [],
            'G_matrices': [],
            'rls_variance': [],
            'rls_predictions': [],
            'rls_ground_truth': [],
            'times': [],
            'target_angle': None,
            'success': False
        }
        
        print("PendulumDHPTrainer initialized with config:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
    
    def setup_logging(self):
        """
        Setup comprehensive logging for training session (same as CF2X)
        """
        # Create logs directory
        log_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{log_dir}/dhp_pendulum_training_{timestamp}.log"
        
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
        self.logger.info("DHP PENDULUM TRAINING SESSION STARTED")
        self.logger.info("="*80)
        
        # Log all hyperparameters
        self.logger.info("HYPERPARAMETERS:")
        self.logger.info("-" * 40)
        for key, value in sorted(self.config.items()):
            self.logger.info(f"{key:25}: {value}")
        
        self.logger.info("-" * 40)
        self.logger.info(f"Log file: {self.log_filename}")
        self.logger.info("="*80)
    
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
        
        # Track convergence episodes (< 0.5 rad position error ~ 28.6 degrees)
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
        
        # Log milestone episodes
        if episode in [100, 250, 500, 750, 1000, 1250, 1500]:
            self.log_milestone(episode)
    
    def log_milestone(self, episode):
        """
        Log detailed milestone analysis (same structure as CF2X)
        """
        if len(self.episode_position_errors) == 0:
            return
            
        recent_errors = self.episode_position_errors[-50:] if len(self.episode_position_errors) >= 50 else self.episode_position_errors
        recent_costs = self.episode_costs[-50:] if len(self.episode_costs) >= 50 else self.episode_costs
        recent_success = self.episode_success_rates[-50:] if len(self.episode_success_rates) >= 50 else self.episode_success_rates
        
        avg_pos_error = np.mean(recent_errors)
        std_pos_error = np.std(recent_errors)
        avg_cost = np.mean(recent_costs)
        avg_success = np.mean(recent_success)
        convergent_ratio = len([e for e in recent_errors if e < 0.5]) / len(recent_errors)
        
        self.logger.info("")
        self.logger.info(f"MILESTONE {episode} ANALYSIS:")
        self.logger.info(f"  Recent 50 episodes avg position error: {avg_pos_error:.4f} ± {std_pos_error:.4f} rad")
        self.logger.info(f"  Recent 50 episodes avg cost: {avg_cost:.3f}")
        self.logger.info(f"  Recent 50 episodes success rate: {avg_success:.2%}")
        self.logger.info(f"  Convergent episodes ratio (< 0.5 rad): {convergent_ratio:.2%}")
        self.logger.info(f"  Total convergent episodes so far: {len(self.convergence_episodes)}")
        self.logger.info(f"  Best performance: {self.best_position_error:.4f} rad at episode {self.best_episode}")
        
        if self.stable_performance_start != -1:
            self.logger.info(f"  Stable performance maintained since episode {self.stable_performance_start}")
        else:
            self.logger.info("  Stable performance not yet achieved")
        self.logger.info("")
    
    def save_best_model(self, episode, pos_error):
        """
        Save the best performing model weights and record the best episode
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/trained_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save best model with fixed naming (no episode number for easy loading)
        best_model_path = f"{checkpoint_dir}/dhp_pendulum_best"
        
        # Save best model without episode number - this gets overwritten each time we find a better model
        self.agent.save(file_path=best_model_path, global_step=None)
        
        # Save best performance metadata
        best_metadata = {
            'best_episode': episode,
            'best_position_error': float(pos_error),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{checkpoint_dir}/dhp_pendulum_best_metadata.pkl", 'wb') as f:
            pickle.dump(best_metadata, f)
        
        self.logger.info(f"BEST MODEL SAVED! Episode {episode}, Error: {pos_error:.4f} rad")
        self.logger.info(f"Best model path: {best_model_path} (overwritten each time)")
        
        # Mark that best model has been saved
        self.best_model_saved = True
        
        # Session-best tracking for end-of-training recording
        if self.config.get('record_best_episodes', True):
            if pos_error < self.session_best_error:
                self.session_best_episode_num = episode
                self.session_best_error = pos_error
                
                # Save action sequence and initial conditions from current episode data
                if len(self.current_episode_data['actions']) > 0:
                    self.session_best_actions_sequence = [a.copy() for a in self.current_episode_data['actions']]
                    self.session_best_target_angle = self.current_episode_data['target_angle']
                    
                    self.logger.info(f"SESSION BEST UPDATED! Episode {episode}: {pos_error:.4f} rad")
                    self.logger.info(f"  Saved {len(self.session_best_actions_sequence)} actions for end-of-training recording")
        
        self.best_model_saved = True
    
    def setup_environment(self):
        """
        Setup Pendulum Random Target environment
        """
        print("\n[SETUP] Initializing Pendulum Random Target Environment...")
        
        self.env = PendulumRandomTargetEnv(
            fixed_target=self.config['fixed_target'],
            normalize_states=self.config['normalize_states'],
            gui=self.config['gui'],
            record=self.config['record']
        )
        
        print(f"Environment observation space: {self.env.observation_space}")
        print(f"Environment action space: {self.env.action_space}")
        
    def setup_dhp_agent(self):
        """
        Setup DHP agent using existing msc-thesis implementation (same as CF2X)
        """
        print("\n[SETUP] Initializing DHP Agent...")
        
        # Agent configuration (following msc-thesis format)
        agent_kwargs = {
            'input_size': [self.config['state_size'], self.config['reference_size']],
            'output_size': self.config['action_size'],
            'hidden_layer_size': self.config['hidden_layer_size'],
            'lr_critic': self.config['lr_critic'],
            'lr_actor': self.config['lr_actor'],
            'gamma': self.config['gamma'],
            'target_network': self.config['target_network'],
            'tau': self.config['tau'],
            'split': self.config['split'],
            'activation': self.config['activation']
        }
        
        # Create DHP agent (reusing msc-thesis implementation)
        self.agent = DHP_Agent(**agent_kwargs)
        
        # Set trim to zero for pendulum (no steady-state torque needed)
        self.agent.trim = np.zeros(self.config['action_size'])
        
        print(f"DHP Agent created with architecture: {self.config['hidden_layer_size']}")
        print("Note: Using unified architecture for pendulum control")
    
    def setup_rls_model(self):
        """
        Setup RLS dynamics model using existing msc-thesis implementation
        """
        print("\n[SETUP] Initializing RLS Dynamics Model...")
        
        # Model configuration (adjusted for 2D pendulum state)
        model_kwargs = {
            'state_size': self.config['rls_state_size'],
            'action_size': self.config['rls_action_size'],
            'gamma': 0.999,     # Slightly higher forgetting factor for stability
            'covariance': 10,   # Lower initial covariance for better stability
            'predict_delta': self.config['predict_delta']
        }
        
        # Create RLS model (reusing msc-thesis implementation)
        self.model = RecursiveLeastSquares(**model_kwargs)
        
        print(f"RLS Model created with forgetting factor: {self.config['rls_gamma']}")
        print(f"Initial covariance: {self.config['rls_covariance']}")
    
    def check_rls_stability(self):
        """Check if RLS model is still stable"""
        if hasattr(self.model, 'cov'):
            cov_trace = np.trace(self.model.cov)
            cov_max = np.max(self.model.cov)
            
            # Reset RLS if covariance explodes
            if cov_trace > 1000.0 or cov_max > 500.0 or np.any(np.isnan(self.model.cov)):
                print(f"WARNING: RLS covariance explosion detected! Resetting RLS model...")
                print(f"  Trace: {cov_trace:.2f}, Max: {cov_max:.2f}")
                self.setup_rls_model()  # Reset the model
                return False
        return True

    def normalize_state(self, state):
        """
        Normalize state vector to [-1, 1] range (same approach as CF2X)
        """
        if not self.config['normalize_states']:
            return state.copy()
        
        normalized_state = state.copy()
        state_names = ['theta', 'theta_dot']
        
        for i, name in enumerate(state_names):
            if name in self.config['state_bounds']:
                min_val, max_val = self.config['state_bounds'][name]
                # Clip to bounds first
                clipped_val = np.clip(state[i], min_val, max_val)
                # Normalize to [-1, 1]
                normalized_state[i] = 2.0 * (clipped_val - min_val) / (max_val - min_val) - 1.0
        
        return normalized_state
    
    def denormalize_state(self, normalized_state):
        """
        Denormalize state vector from [-1, 1] range back to physical units
        """
        if not self.config['normalize_states']:
            return normalized_state.copy()
        
        state = normalized_state.copy()
        state_names = ['theta', 'theta_dot']
        
        for i, name in enumerate(state_names):
            if name in self.config['state_bounds']:
                min_val, max_val = self.config['state_bounds'][name]
                # Denormalize from [-1, 1] to physical range
                state[i] = min_val + (normalized_state[i] + 1.0) * (max_val - min_val) / 2.0
        
        return state
    
    def normalize_reference(self, reference):
        """
        Normalize reference vector using same bounds as states
        """
        return self.normalize_state(reference)
        
    def _transform_gradient_to_normalized_space(self, dcostdx_raw):
        """
        Transform cost gradient from raw physical space to normalized space (same as CF2X)
        """
        if not self.config['normalize_states']:
            return dcostdx_raw.copy()
        
        dcostdx_normalized = dcostdx_raw.copy()
        state_names = ['theta', 'theta_dot']

        for i, name in enumerate(state_names):
            if name in self.config['state_bounds']:
                min_val, max_val = self.config['state_bounds'][name]
                # Chain rule: d(cost)/d(normalized) = d(cost)/d(raw) * d(raw)/d(normalized)
                scaling_factor = (max_val - min_val) / 2.0
                dcostdx_normalized[i] = dcostdx_raw[i] * scaling_factor
        
        return dcostdx_normalized
        
    def generate_excitation_signal(self, step):
        """
        Generate exploration signal for initial learning (adapted for pendulum torque range)
        """
        if step < self.config['excitation_steps']:
            # Sinusoidal excitation for pendulum system identification
            t = step * 0.02  # Assuming ~0.02s per step for 200-step episodes
            excitation = self.config['excitation_amplitude'] * np.array([
                np.sin(2.0 * np.pi * 0.2 * t)  # 0.2 Hz excitation frequency
            ])
            return excitation
        else:
            return np.zeros(self.config['action_size'])
    
    def train_episode(self, episode_num):
        """
        Train single episode following dhp_main.py structure (adapted for pendulum)
        """
        # Reset environment
        state, info = self.env.reset()
        reference = info['reference']
        
        # Store episode information
        self.current_episode_data['target_angle'] = info['target_angle']
        
        # Safety check for initial state
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Warning: Invalid initial state at episode {episode_num}")
            return 0.0, 1.0, 10.0, 0.0  # Return safe default values
        
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        max_steps = self.config['max_steps']
        nan_detected = False
        
        for step in range(max_steps):
            # Store current state and reference (raw values)
            X_raw = state.copy()
            R_sig_raw = reference.copy()
            
            # Apply normalization if enabled
            X = self.normalize_state(X_raw)
            R_sig = self.normalize_reference(R_sig_raw)
            
            # Prepare arrays for different uses
            X_flat = X.flatten()  # 1D for RLS model
            X_shaped = X.reshape(1, -1, 1)  # 3D for DHP agent
            R_sig_shaped = R_sig.reshape(1, -1, 1)
            
            # DHP Update cycles (following dhp_main.py structure)
            for update_cycle in range(self.config['update_cycles']):
                # Forward prediction using RLS model (use normalized states)
                action = self.agent.action(X_shaped, reference=R_sig_shaped)
                action_flat = action.flatten()  # Convert back to 1D for RLS model
                X_next_pred = self.model.predict(X_flat, action_flat)
                
                # Safety check for RLS prediction
                if np.any(np.isnan(X_next_pred)) or np.any(np.isinf(X_next_pred)):
                    print(f"Warning: Invalid RLS prediction at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                # Cost computation using environment cost function (use raw states for cost)
                X_next_pred_raw = self.denormalize_state(X_next_pred) if self.config['normalize_states'] else X_next_pred
                cost, dcostdx_raw = self.env.compute_dhp_cost(X_next_pred_raw, R_sig_raw)
                
                # Safety check for cost computation
                if np.any(np.isnan(dcostdx_raw)) or np.any(np.isinf(dcostdx_raw)):
                    print(f"Warning: Invalid cost gradient at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                # Transform cost gradient to normalized space if needed
                if self.config['normalize_states']:
                    dcostdx = self._transform_gradient_to_normalized_space(dcostdx_raw)
                else:
                    dcostdx = dcostdx_raw
                
                # Model gradients for DHP updates
                A = self.model.gradient_state(X_flat, action_flat)         # ∂x_{t+1}/∂x_t
                B = self.model.gradient_action(X_flat, action_flat)        # ∂x_{t+1}/∂u_t
                
                # Safety check for model gradients
                if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(np.isnan(B)) or np.any(np.isinf(B)):
                    print(f"Warning: Invalid model gradients at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                dactiondx = self.agent.gradient_actor(X_shaped, reference=R_sig_shaped)  # ∂u/∂x
                dactiondx_flat = dactiondx.reshape(self.config['action_size'], self.config['state_size'])
                
                # Safety check for actor gradients
                if np.any(np.isnan(dactiondx_flat)) or np.any(np.isinf(dactiondx_flat)):
                    print(f"Warning: Invalid actor gradients at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                # Critic update (Bellman residual gradient)
                lmbda = self.agent.value_derivative(X_shaped, reference=R_sig_shaped)
                X_next_pred_shaped = X_next_pred.reshape(1, -1, 1)
                target_lmbda = self.agent.target_value_derivative(X_next_pred_shaped, reference=R_sig_shaped)
                
                # Check for NaN values before computation
                if np.any(np.isnan(lmbda)) or np.any(np.isnan(target_lmbda)) or np.any(np.isnan(dcostdx)):
                    print(f"Warning: NaN detected in critic update at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                # Stabilize the critic gradient computation
                bellman_target = dcostdx + self.config['gamma'] * target_lmbda.flatten()
                model_term = A + B @ dactiondx_flat
                
                # Additional safety checks
                if np.any(np.isnan(bellman_target)) or np.any(np.isnan(model_term)):
                    print(f"Warning: NaN in Bellman computation at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                grad_critic = lmbda.flatten() - bellman_target @ model_term
                
                # Clip gradients for stability
                grad_critic = np.clip(grad_critic, -0.5, 0.5)
                
                # Additional safety check before update
                if np.any(np.isnan(grad_critic)) or np.any(np.isinf(grad_critic)):
                    print(f"Warning: Invalid critic gradient after computation at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                self.agent.update_critic(X_shaped, reference=R_sig_shaped, gradient=grad_critic.reshape(1, 1, -1))
                
                # Actor update (Policy gradient)
                lmbda = self.agent.value_derivative(X_next_pred_shaped, reference=R_sig_shaped)
                
                # Check for NaN values
                if np.any(np.isnan(lmbda)) or np.any(np.isnan(B)):
                    print(f"Warning: NaN detected in actor update at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                grad_actor = (dcostdx + self.config['gamma'] * lmbda.flatten()) @ B

                # Clip gradients for stability
                grad_actor = np.clip(grad_actor, -0.5, 0.5)

                # Additional safety check before update
                if np.any(np.isnan(grad_actor)) or np.any(np.isinf(grad_actor)):
                    print(f"Warning: Invalid actor gradient after computation at episode {episode_num}, step {step}")
                    nan_detected = True
                    break

                self.agent.update_actor(X_shaped, reference=R_sig_shaped, gradient=grad_actor.reshape(1, 1, -1))
            
            # If NaN was detected during DHP updates, skip the rest of this step
            if nan_detected:
                print(f"Skipping step {step} in episode {episode_num} due to NaN detection")
                break
            
            # Environment step
            action = self.agent.action(X_shaped, reference=R_sig_shaped)
            action_flat = action.flatten()  # Convert to 1D for environment
            
            # Add excitation for initial exploration
            excitation = self.generate_excitation_signal(episode_num * max_steps + step)
            action_with_excitation = action_flat + excitation
            action_clipped = np.clip(action_with_excitation, -2.0, 2.0)  # Pendulum torque range
            
            # Save action for potential session-best replay
            self.current_episode_data['actions'].append(action_clipped.copy())
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action_clipped)
            
            # Normalize next state for model error calculation and RLS update
            next_state_normalized = self.normalize_state(next_state) if self.config['normalize_states'] else next_state
            
            # Calculate model error (in normalized space if normalization is enabled)
            model_error = np.mean((X_next_pred - next_state_normalized)**2)
            
            # Safety check for model error
            if np.isnan(model_error) or np.isinf(model_error):
                print(f"Warning: Invalid model error at episode {episode_num}, step {step}")
                model_error = 1.0  # Set to a safe default value
            
            # Store RLS prediction vs ground truth for model analysis
            self.current_episode_data['rls_predictions'].append(X_next_pred.copy())
            self.current_episode_data['rls_ground_truth'].append(next_state_normalized.copy())
            
            # Store detailed data for current episode (for potential best episode capture)
            episode_time = step * 0.02  # Assuming ~0.02s per step for 200-step episodes
            self.current_episode_data['states'].append(X_raw.copy())  # Store raw state for plotting
            self.current_episode_data['references'].append(R_sig_raw.copy())  # Store raw reference for plotting
            self.current_episode_data['costs'].append(info['dhp_cost'])
            self.current_episode_data['model_errors'].append(model_error)
            self.current_episode_data['times'].append(episode_time)
            
            # Store gradients and matrices from last update cycle
            if 'grad_actor' in locals() and 'grad_critic' in locals():
                self.current_episode_data['actor_grads'].append(grad_actor.copy())
                self.current_episode_data['critic_grads'].append(grad_critic.copy())
                self.current_episode_data['F_matrices'].append(A.copy())
                self.current_episode_data['G_matrices'].append(B.copy())
                # RLS variance (diagonal of covariance matrix)
                rls_var = np.diag(self.model.cov) if hasattr(self.model, 'cov') else np.zeros(self.config['rls_state_size'])
                self.current_episode_data['rls_variance'].append(rls_var.copy())
                
                # Store RLS variance for overall training analysis
                self.detailed_rls_variance.append(rls_var.copy())
                self.detailed_time.append(episode_time)
            
            # Update RLS model with real data (use normalized states if normalization enabled)
            try:
                if self.config['normalize_states']:
                    self.model.update(X_flat, action_clipped, next_state_normalized.flatten())
                else:
                    self.model.update(X_flat, action_clipped, next_state.flatten())
            except Exception as e:
                print(f"Warning: RLS update failed at episode {episode_num}, step {step}: {e}")
                # Continue without updating RLS this step
            
            # Update for next iteration (use raw states for environment)
            state = next_state
            reference = info['reference']
            
            # Accumulate metrics
            episode_reward += reward
            episode_cost += info['dhp_cost']
            episode_steps += 1
            
            # Check termination
            if terminated or truncated or nan_detected:
                if nan_detected:
                    print(f"Episode {episode_num} terminated early due to numerical instability")
                break
        
        # Store episode metrics
        self.episode_rewards.append(episode_reward)
        if episode_steps > 0:
            self.episode_costs.append(episode_cost / episode_steps)  # Average cost per step
        else:
            self.episode_costs.append(1.0)  # Safe default value
        
        final_pos_error = info.get('position_error', 10.0) if not nan_detected else 10.0
        final_success = info.get('episode_success', False) if not nan_detected else False
        self.episode_position_errors.append(final_pos_error)
        self.episode_success_rates.append(1.0 if final_success else 0.0)
        
        # Check if this is the best episode and save detailed data if so
        is_best_episode = final_pos_error < self.best_position_error
        if is_best_episode:
            self.best_position_error = final_pos_error
            self.best_episode = episode_num
            
            # Save current episode data as best episode data for plotting
            self.best_episode_states = [s.copy() for s in self.current_episode_data['states']]
            self.best_episode_references = [r.copy() for r in self.current_episode_data['references']]
            self.best_episode_actions = [a.copy() for a in self.current_episode_data['actions']]
            self.best_episode_costs = self.current_episode_data['costs'].copy()
            self.best_episode_model_errors = self.current_episode_data['model_errors'].copy()
            self.best_episode_actor_grads = [g.copy() for g in self.current_episode_data['actor_grads']]
            self.best_episode_critic_grads = [g.copy() for g in self.current_episode_data['critic_grads']]
            self.best_episode_F_matrices = [f.copy() for f in self.current_episode_data['F_matrices']]
            self.best_episode_G_matrices = [g.copy() for g in self.current_episode_data['G_matrices']]
            self.best_episode_rls_variance = [v.copy() for v in self.current_episode_data['rls_variance']]
            self.best_episode_time = self.current_episode_data['times'].copy()
            
            # Save RLS prediction data for model analysis
            self.best_episode_rls_predictions = [p.copy() for p in self.current_episode_data['rls_predictions']]
            self.best_episode_rls_ground_truth = [g.copy() for g in self.current_episode_data['rls_ground_truth']]
            
            # Update session-best recording data if recording is enabled
            if self.config.get('record_best_episodes', True):
                self.session_best_episode_num = episode_num
                self.session_best_error = final_pos_error
                # Store action sequence for replay (already captured in current_episode_data)
                self.session_best_actions_sequence = [a.copy() for a in self.current_episode_data['actions']]
                self.session_best_target_angle = self.current_episode_data['target_angle']
                
                print(f"🎯 NEW SESSION BEST EPISODE: {episode_num}, Error: {final_pos_error:.4f} rad")
                print(f"   Target angle: {self.session_best_target_angle:.3f} rad ({np.rad2deg(self.session_best_target_angle):.1f}°)")
                print(f"   Action sequence captured: {len(self.session_best_actions_sequence)} steps")
            
            print(f"[BEST EPISODE] Episode {episode_num}: New best position error {final_pos_error:.4f} rad")
            print(f"               Captured {len(self.best_episode_states)} data points for plotting")
        
        # Clear current episode data for next episode
        self.current_episode_data = {
            'states': [],
            'references': [],
            'actions': [],
            'costs': [],
            'model_errors': [],
            'actor_grads': [],
            'critic_grads': [],
            'F_matrices': [],
            'G_matrices': [],
            'rls_variance': [],
            'rls_predictions': [],
            'rls_ground_truth': [],
            'times': [],
            'target_angle': None,
            'success': False
        }
        
        return (float(episode_reward), 
        float(episode_cost / episode_steps) if episode_steps > 0 else 1.0, 
        float(final_pos_error), 
        float(1.0 if final_success else 0.0))
    
    def train(self):
        """
        Main training loop with comprehensive logging (same structure as CF2X)
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("STARTING DHP TRAINING FOR PENDULUM")
        self.logger.info("="*50)
        
        # Setup all components
        self.setup_environment()
        self.setup_dhp_agent()
        self.setup_rls_model()
        
        # Start training timer
        self.training_start_time = time.time()
        
        self.logger.info(f"\nTraining for {self.config['num_episodes']} episodes...")
        self.logger.info("Episode | Reward  | Avg Cost | Pos Error | Success | Time")
        self.logger.info("-" * 60)
        
        # Training loop
        for episode in range(self.config['num_episodes']):
            episode_reward, avg_cost, pos_error, success_rate = self.train_episode(episode)
            
            # Log performance metrics and track convergence
            self.log_performance_metrics(episode, episode_reward, avg_cost, pos_error, success_rate)
            
            # Periodic logging
            if episode % self.config['log_interval'] == 0:
                elapsed_time = time.time() - self.training_start_time
                log_msg = (f"{episode:7d} | {episode_reward:7.2f} | {float(avg_cost):8.3f} | "
                          f"{float(pos_error):9.4f} | {float(success_rate):7.3f} | {elapsed_time:6.1f}s")
                self.logger.info(log_msg)
            
            # Save checkpoint
            if episode % self.config['save_interval'] == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        # Training completed - log final analysis
        total_time = time.time() - self.training_start_time
        self.log_final_analysis(total_time)
        
        # Record the session-best episode at the end of training
        print("\n" + "="*60)
        print("FINALIZING SESSION-BEST EPISODE RECORDING")
        print("="*60)
        self.finalize_session_best_recording()
        
        # Save final trained model
        self.save_trained_model()
        
        # Plot results
        self.plot_training_results()
    
    def finalize_session_best_recording(self):
        """
        At the end of training, replay the session-best episode with recording enabled
        """
        if (self.session_best_episode_num == -1 or 
            len(self.session_best_actions_sequence) == 0 or 
            self.session_best_target_angle is None):
            self.logger.warning("No session-best data available for recording")
            print("❌ No session-best episode data to record")
            return
        
        print(f"\n🎥 RECORDING SESSION BEST EPISODE {self.session_best_episode_num}")
        print(f"   Error: {self.session_best_error:.4f} rad ({np.rad2deg(self.session_best_error):.1f}°)")
        print(f"   Target: {self.session_best_target_angle:.3f} rad ({np.rad2deg(self.session_best_target_angle):.1f}°)")
        print(f"   Actions to replay: {len(self.session_best_actions_sequence)}")
        
        # Create recordings directory
        recordings_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/best_episode_recordings"
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Create recording environment with GUI and fixed target for perfect repeatability
        record_env = PendulumRandomTargetEnv(
            fixed_target=self.session_best_target_angle,  # Use exact same target
            normalize_states=self.config['normalize_states'],
            gui=True,  # Enable rendering for recording
            record=True  # Enable recording
        )
        
        try:
            # Reset environment with same target
            state, info = record_env.reset()
            reference = info['reference']
            
            print(f"✅ Recording environment initialized with target: {info['target_angle']:.3f} rad")
            print(f"Replaying {len(self.session_best_actions_sequence)} actions...")
            
            # Replay the saved action sequence
            episode_cost = 0.0
            for step, saved_action in enumerate(self.session_best_actions_sequence):
                # Execute the saved action
                next_state, reward, terminated, truncated, info = record_env.step(saved_action)
                
                # Update for next iteration
                state = next_state
                reference = info['reference']
                episode_cost += info['dhp_cost']
                
                # Check termination
                if terminated or truncated:
                    break
            
            final_error = info['position_error']
            avg_cost = episode_cost / len(self.session_best_actions_sequence)
            final_success = info['episode_success']
            
            print(f"✅ Session-best recording completed!")
            print(f"   Final position error: {final_error:.4f} rad ({np.rad2deg(final_error):.1f}°)")
            print(f"   Expected error was: {self.session_best_error:.4f} rad")
            print(f"   Average cost: {avg_cost:.3f}")
            print(f"   Episode success: {final_success}")
            
            # Log the recording
            self.logger.info(f"SESSION-BEST EPISODE RECORDED: Episode {self.session_best_episode_num}")
            self.logger.info(f"  Replay error: {final_error:.4f} rad vs training error: {self.session_best_error:.4f} rad")
            self.logger.info(f"  Target angle: {self.session_best_target_angle:.3f} rad")
            
        except Exception as e:
            print(f"❌ Error during session-best recording: {e}")
            self.logger.error(f"Failed to record session-best episode {self.session_best_episode_num}: {e}")
        
        finally:
            record_env.close()
            print("🎥 Session-best recording completed\n")
    
    def log_final_analysis(self, total_time):
        """
        Log comprehensive final training analysis (adapted for pendulum)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING COMPLETED - FINAL ANALYSIS")
        self.logger.info("="*80)
        
        # Basic statistics
        self.logger.info(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        final_reward = self.episode_rewards[-1]
        if isinstance(final_reward, np.ndarray):
            final_reward = float(final_reward) if final_reward.size == 1 else float(final_reward[0])
        self.logger.info(f"Final episode reward: {final_reward:.2f}")
        self.logger.info(f"Final position error: {self.episode_position_errors[-1]:.4f} rad")
        self.logger.info(f"Final success rate: {self.episode_success_rates[-1]:.3f}")
        self.logger.info(f"Best position error achieved: {self.best_position_error:.4f} rad at episode {self.best_episode}")
        
        # Convergence analysis
        total_convergent = len(self.convergence_episodes)
        convergence_rate = total_convergent / len(self.episode_position_errors)
        avg_success = np.mean(self.episode_success_rates[-100:]) if len(self.episode_success_rates) >= 100 else np.mean(self.episode_success_rates)
        
        self.logger.info(f"Total convergent episodes (< 0.5 rad): {total_convergent}/{len(self.episode_position_errors)} ({convergence_rate:.2%})")
        self.logger.info(f"Recent success rate (last 100 episodes): {avg_success:.2%}")
        
        if self.stable_performance_start != -1:
            stable_duration = len(self.episode_position_errors) - self.stable_performance_start
            self.logger.info(f"Stable performance achieved: Yes (from episode {self.stable_performance_start}, duration: {stable_duration} episodes)")
        else:
            self.logger.info("Stable performance achieved: No")
        
        # Performance distribution analysis
        all_errors = np.array(self.episode_position_errors)
        self.logger.info(f"Position error statistics:")
        self.logger.info(f"  Mean: {np.mean(all_errors):.4f} rad ({np.rad2deg(np.mean(all_errors)):.1f}°)")
        self.logger.info(f"  Std:  {np.std(all_errors):.4f} rad ({np.rad2deg(np.std(all_errors)):.1f}°)")
        self.logger.info(f"  Min:  {np.min(all_errors):.4f} rad ({np.rad2deg(np.min(all_errors)):.1f}°)")
        self.logger.info(f"  Max:  {np.max(all_errors):.4f} rad ({np.rad2deg(np.max(all_errors)):.1f}°)")
        self.logger.info(f"  Median: {np.median(all_errors):.4f} rad ({np.rad2deg(np.median(all_errors)):.1f}°)")
        
        # Performance thresholds (pendulum-specific)
        excellent = len(all_errors[all_errors < 0.1])    # < 5.7°
        good = len(all_errors[(all_errors >= 0.1) & (all_errors < 0.3)])  # 5.7° - 17.2°
        acceptable = len(all_errors[(all_errors >= 0.3) & (all_errors < 0.5)])  # 17.2° - 28.6°
        poor = len(all_errors[all_errors >= 0.5])  # > 28.6°
        
        self.logger.info(f"Performance distribution:")
        self.logger.info(f"  Excellent (< 0.1 rad / 5.7°): {excellent} episodes ({excellent/len(all_errors):.1%})")
        self.logger.info(f"  Good (0.1-0.3 rad / 5.7°-17.2°): {good} episodes ({good/len(all_errors):.1%})")
        self.logger.info(f"  Acceptable (0.3-0.5 rad / 17.2°-28.6°): {acceptable} episodes ({acceptable/len(all_errors):.1%})")
        self.logger.info(f"  Poor (> 0.5 rad / 28.6°): {poor} episodes ({poor/len(all_errors):.1%})")
        
        # Training configuration effectiveness
        self.logger.info("\nTRAINING CONFIGURATION EFFECTIVENESS:")
        self.logger.info("-" * 40)
        self.logger.info(f"Episodes needed for first convergence: {self.convergence_episodes[0] if self.convergence_episodes else 'N/A'}")
        self.logger.info(f"Excitation phase ended at episode: {int(self.config['excitation_steps'] / self.config['max_steps'])}")
        
        # Save detailed metrics to JSON
        self.save_training_metrics_json()
        
        # Final best model confirmation
        if self.best_episode != -1 and self.best_position_error != float('inf'):
            self.best_model_saved = True
        
        if self.best_model_saved:
            self.logger.info(f"\nBEST MODEL CONFIRMED:")
            self.logger.info(f"  Episode: {self.best_episode}")
            self.logger.info(f"  Position Error: {self.best_position_error:.4f} rad ({np.rad2deg(self.best_position_error):.1f}°)")
            self.logger.info(f"  Model files: dhp_pendulum_best_*")
        else:
            self.logger.info("\nWARNING: No significant improvement found - no best model saved")
        
        self.logger.info("="*80)
        self.logger.info("TRAINING SESSION COMPLETED")
        self.logger.info(f"Detailed log saved to: {self.log_filename}")
        self.logger.info("="*80)
    
    def save_training_metrics_json(self):
        """
        Save comprehensive training metrics to JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/training_logs"
        json_filename = f"{metrics_dir}/dhp_pendulum_metrics_{timestamp}.json"
        
        metrics = {
            "training_session": {
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
            "statistics": {
                "position_errors": {
                    "mean": float(np.mean(self.episode_position_errors)) if self.episode_position_errors else None,
                    "std": float(np.std(self.episode_position_errors)) if self.episode_position_errors else None,
                    "min": float(np.min(self.episode_position_errors)) if self.episode_position_errors else None,
                    "max": float(np.max(self.episode_position_errors)) if self.episode_position_errors else None,
                    "median": float(np.median(self.episode_position_errors)) if self.episode_position_errors else None
                },
                "rewards": {
                    "mean": float(np.mean(self.episode_rewards)) if self.episode_rewards else None,
                    "std": float(np.std(self.episode_rewards)) if self.episode_rewards else None,
                    "min": float(np.min(self.episode_rewards)) if self.episode_rewards else None,
                    "max": float(np.max(self.episode_rewards)) if self.episode_rewards else None
                },
                "success_rates": {
                    "mean": float(np.mean(self.episode_success_rates)) if self.episode_success_rates else None,
                    "std": float(np.std(self.episode_success_rates)) if self.episode_success_rates else None,
                    "final_100_episodes": float(np.mean(self.episode_success_rates[-100:])) if len(self.episode_success_rates) >= 100 else float(np.mean(self.episode_success_rates)) if self.episode_success_rates else None
                }
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
        Save the trained DHP agent with normalization configuration
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/trained_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save DHP agent
        model_path = f"{checkpoint_dir}/dhp_pendulum_final"
        
        # First save hyperparameters (with global_step=None to force .pkl creation)
        self.agent.save(file_path=model_path, global_step=None)
        
        # Then save the trained weights with the step number
        self.agent.save(file_path=model_path, global_step=len(self.episode_rewards))
        
        # Save configuration and metrics
        np.savez(f"{checkpoint_dir}/pendulum_training_summary.npz",
                 config=self.config,
                 rewards=self.episode_rewards,
                 costs=self.episode_costs,
                 position_errors=self.episode_position_errors,
                 success_rates=self.episode_success_rates)
        
        # Also save a simple pickle file with just the complete config for easy loading
        with open(f"{checkpoint_dir}/dhp_pendulum_config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"Trained model saved to: {model_path}")
        print(f"Training summary saved to: {checkpoint_dir}/pendulum_training_summary.npz")
        print(f"Config saved to: {checkpoint_dir}/dhp_pendulum_config.pkl")
        
        # Print normalization status for user
        if self.config['normalize_states']:
            print("✓ Model saved with state normalization ENABLED")
            print("  Demo script will automatically apply the same normalization")
            self.logger.info("Model saved with state normalization ENABLED")
        else:
            print("✓ Model saved with state normalization DISABLED") 
            print("  Demo script will use raw states")
            self.logger.info("Model saved with state normalization DISABLED")
        
        # Print best model information
        if self.best_model_saved:
            print(f"✓ Best model automatically saved from episode {self.best_episode}")
            print(f"  Best position error: {self.best_position_error:.4f} rad ({np.rad2deg(self.best_position_error):.1f}°)")
            print(f"  Use dhp_pendulum_best_* files for best performance")
        else:
            print("⚠️  No best model saved - no recordings available")
        
    def save_checkpoint(self, episode):
        """
        Save training checkpoint
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save training metrics
        np.savez(f"{checkpoint_dir}/pendulum_metrics_episode_{episode}.npz",
                 rewards=self.episode_rewards,
                 costs=self.episode_costs,
                 position_errors=self.episode_position_errors,
                 success_rates=self.episode_success_rates)
        
        print(f"Checkpoint saved at episode {episode}")
    
    def plot_training_results(self):
        """
        Plot training metrics in dhp_main.py style with data from the BEST episode (adapted for pendulum)
        """
        # Import seaborn for enhanced styling
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
            sns.set_palette("husl") 
            sns.set_context("notebook", font_scale=1.1)
            seaborn_available = True
            print("Using seaborn styling for enhanced plots")
        except ImportError:
            print("Seaborn not available - using matplotlib defaults")
            seaborn_available = False
        
        print("\n[PLOTTING] Generating DHP pendulum training analysis plots...")
        
        if len(self.best_episode_states) == 0:
            print("No best episode data available for plotting")
            print("This means no episode completed successfully during training")
            return
        
        print(f"Using BEST episode data from episode {self.best_episode}")
        print(f"Best episode position error: {self.best_position_error:.4f} rad ({np.rad2deg(self.best_position_error):.1f}°)")
        print(f"Data points: {len(self.best_episode_states)}")
            
        # Convert best episode data to arrays for easier manipulation
        states = np.array(self.best_episode_states)
        references = np.array(self.best_episode_references) 
        actions = np.array(self.best_episode_actions)
        costs = np.array(self.best_episode_costs)
        model_errors = np.array(self.best_episode_model_errors)
        t = np.array(self.best_episode_time)

        # Extract state components [theta, theta_dot]
        theta = states[:, 0]
        theta_dot = states[:, 1]
        
        
        # Extract references
        theta_ref = references[:, 0]
        theta_dot_ref = references[:, 1]
        
        
        # Extract actions (single torque)
        torque = actions[:, 0]
        
        # Set colors
        if seaborn_available:
            try:
                colors = sns.color_palette("husl", 8)
            except:
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        else:
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
        plt.rcParams['figure.figsize'] = 25/2.54, 50/2.54
        
        ### Figure 0: Training Progress - Episode Rewards, Costs, and Success Rates ###
        fig0 = plt.figure()
        episodes = np.arange(len(self.episode_rewards))
        
        ax1 = fig0.add_subplot(3,1,1)
        ax1.plot(episodes, self.episode_rewards, color=colors[0], linewidth=2, label='Episode Rewards (DHP)')
        ax1.set_ylabel(r'Episode Reward')
        ax1.set_title('DHP Pendulum Training Progress - Rewards, Costs, and Success Rates')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        ax2.plot(episodes, self.episode_costs, color=colors[1], linewidth=2, label='Episode Average Cost (DHP)')
        ax2.set_ylabel(r'Episode Average Cost')
        ax2.set_yscale('log')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3,1,3, sharex=ax1)
        ax3.plot(episodes, self.episode_success_rates, color=colors[2], linewidth=2, label='Episode Success Rate (DHP)')
        ax3.set_ylabel(r'Episode Success Rate')
        ax3.set_xlabel(r'Episode Number')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        ### Figure 1: Primary Control - Pendulum Angle Tracking ###
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(4,1,1)
        ax1.plot(t, theta*180/np.pi, 'b-', label=r'$\theta$ (DHP Best Episode)')  
        ax1.plot(t, theta_ref*180/np.pi, 'r--', label=r'$\theta_{ref}$ (DHP Best Episode)')  
        ax1.set_ylabel(r'Angle $[deg]$')
        ax1.set_title(f'DHP Pendulum Best Episode Performance (Ep {self.best_episode}, Error: {np.rad2deg(self.best_position_error):.1f}°)')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        ax2 = plt.subplot(4,1,2, sharex=ax1)
        ax2.plot(t, theta_dot*180/np.pi, 'b-', label=r'$\dot{\theta}$ (DHP Best Episode)')
        ax2.plot(t, theta_dot_ref*180/np.pi, 'r--', label=r'$\dot{\theta}_{ref}$ (DHP Best Episode)')
        ax2.set_ylabel(r'Angular Velocity $[deg/s]$')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        ax3 = plt.subplot(4,1,3, sharex=ax1)
        ax3.plot(t, costs, 'b-', label=r'$c_{actual}$ (DHP Best Episode)')  
        ax3.set_ylabel(r'Cost $[-]$')
        ax3.set_yscale('log')
        ax3.legend(loc='upper right')
        ax3.grid(True)

        ax4 = plt.subplot(4,1,4, sharex=ax1)
        ax4.plot(t, model_errors, 'b-', label=r'$e_{model}$ (DHP Best Episode)') 
        ax4.set_xlabel(r'$t [s]$')
        ax4.set_ylabel(r'Model Error $[-]$')
        ax4.set_yscale('log')
        ax4.legend(loc='upper right')
        ax4.grid(True)
        
        ### Figure 2: Control Actions and State Space Analysis ###
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(4,1,1)
        ax1.plot(t, torque, 'g-', label=r'Torque (DHP Best Episode)')
        ax1.set_ylabel(r'Control Torque $[Nm]$')
        ax1.set_title(f'DHP Pendulum Control Analysis - Best Episode {self.best_episode}')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        ax2 = plt.subplot(4,1,2, sharex=ax1)
        # Angular error plot
        theta_error = theta - theta_ref
        # Wrap angular errors to [-π, π]
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
        ax2.plot(t, theta_error*180/np.pi, 'r-', label=r'$\theta_{error}$ (DHP Best Episode)')
        ax2.set_ylabel(r'Angular Error $[deg]$')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        ax3 = plt.subplot(4,1,3, sharex=ax1)
        # Phase portrait in cos/sin space
        ax3.plot(np.cos(theta), np.sin(theta), 'b-', label='Actual Trajectory (cos/sin)', linewidth=2)
        ax3.plot(np.cos(theta_ref), np.sin(theta_ref), 'r--', label='Reference Trajectory', linewidth=2)
        ax3.plot(np.cos(theta[0]), np.sin(theta[0]), 'go', markersize=8, label='Start')
        ax3.plot(np.cos(theta[-1]), np.sin(theta[-1]), 'ro', markersize=8, label='End')
        ax3.set_xlabel(r'$\cos(\theta)$')
        ax3.set_ylabel(r'$\sin(\theta)$')
        ax3.set_title('Phase Space (Unit Circle)')
        ax3.legend(loc='upper right')
        ax3.grid(True)
        ax3.axis('equal')

        ax4 = plt.subplot(4,1,4, sharex=ax1)
        # Angular velocity vs angle phase portrait
        ax4.plot(theta*180/np.pi, theta_dot*180/np.pi, 'b-', label='Actual Phase Portrait', linewidth=2)
        ax4.plot(theta_ref*180/np.pi, theta_dot_ref*180/np.pi, 'r--', label='Reference Phase Portrait', linewidth=2)
        ax4.plot(theta[0]*180/np.pi, theta_dot[0]*180/np.pi, 'go', markersize=8, label='Start')
        ax4.plot(theta[-1]*180/np.pi, theta_dot[-1]*180/np.pi, 'ro', markersize=8, label='End')
        ax4.set_xlabel(r'Angle $[deg]$')
        ax4.set_ylabel(r'Angular Velocity $[deg/s]$')
        ax4.set_title('Phase Portrait (θ vs θ̇)')
        ax4.legend(loc='upper right')
        ax4.grid(True)

        ### Figure 3: Neural Network Analysis (Best Episode) ###
        if len(self.best_episode_actor_grads) > 0:
            fig3 = plt.figure()
            actor_grads = np.array(self.best_episode_actor_grads)
            critic_grads = np.array(self.best_episode_critic_grads)
            
            ax1 = fig3.add_subplot(4,1,1)
            # Plot actor gradients
            if actor_grads.ndim == 1:
                ax1.plot(t[:len(actor_grads)], actor_grads, 'b-', label='Actor Grad (Best Episode)')
            elif actor_grads.ndim == 2:
                # Plot norm of gradient vector
                grad_norm = np.linalg.norm(actor_grads, axis=1)
                ax1.plot(t[:len(grad_norm)], grad_norm, 'b-', label='Actor Grad Norm (Best Episode)')
            ax1.set_ylabel(r'Actor Gradients')
            ax1.set_title(f'DHP Neural Network Analysis - Best Episode {self.best_episode}')
            ax1.legend(loc='upper right')
            ax1.grid(True)

            ax2 = plt.subplot(4,1,2, sharex=ax1)
            # Plot critic gradients
            if critic_grads.ndim == 1:
                ax2.plot(t[:len(critic_grads)], critic_grads, 'r-', label='Critic Grad (Best Episode)')
            elif critic_grads.ndim == 2:
                # Plot norm of gradient vector
                grad_norm = np.linalg.norm(critic_grads, axis=1)
                ax2.plot(t[:len(grad_norm)], grad_norm, 'r-', label='Critic Grad Norm (Best Episode)')
            ax2.set_ylabel(r'Critic Gradients')
            ax2.legend(loc='upper right')
            ax2.grid(True)

            ax3 = plt.subplot(4,1,3, sharex=ax1)
            if len(self.best_episode_F_matrices) > 0:
                F_matrices = np.array(self.best_episode_F_matrices)
                # F_matrices shape: (time_steps, state_size, state_size)
                # Plot Frobenius norm of each matrix
                F_norms = np.linalg.norm(F_matrices.reshape(F_matrices.shape[0], -1), axis=1)
                ax3.plot(t[:len(F_norms)], F_norms, 'g-', label=r'$\|\frac{\partial x_{t+1}}{\partial x_t}\|_F$ (Best Episode)')
            ax3.set_ylabel(r'State Jacobian Norm')
            ax3.legend(loc='upper right')
            ax3.grid(True)

            ax4 = plt.subplot(4,1,4, sharex=ax1)
            if len(self.best_episode_G_matrices) > 0:
                G_matrices = np.array(self.best_episode_G_matrices)
                # G_matrices shape: (time_steps, state_size, action_size)
                # Plot Frobenius norm of each matrix
                G_norms = np.linalg.norm(G_matrices.reshape(G_matrices.shape[0], -1), axis=1)
                ax4.plot(t[:len(G_norms)], G_norms, 'orange', label=r'$\|\frac{\partial x_{t+1}}{\partial u_t}\|_F$ (Best Episode)')
            ax4.set_xlabel(r'$t [s]$')
            ax4.set_ylabel(r'Control Jacobian Norm')
            ax4.legend(loc='upper right')
            ax4.grid(True)

        ### Figure 4: RLS Model Analysis ###
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            fig4 = plt.figure(figsize=(12, 10))
            
            # Overall training RLS variance metrics (all episodes)
            if len(self.detailed_rls_variance) > 0:
                ax1 = fig4.add_subplot(2,2,1)
                rls_var_overall = np.array(self.detailed_rls_variance)
                t_overall = np.array(self.detailed_time)
                
                # Plot mean variance across all states
                mean_variance = np.mean(rls_var_overall, axis=1)
                ax1.plot(t_overall, mean_variance, 'b-', label='Mean RLS Variance (All Training)')
                ax1.set_ylabel(r'Mean RLS Variance')
                ax1.set_xlabel(r'Training Time [s]')
                ax1.set_yscale('log')
                ax1.legend(loc='upper right')
                ax1.grid(True)
                ax1.set_title('RLS Learning Progress (Whole Training)')
                
                # Individual state variances
                ax2 = fig4.add_subplot(2,2,2)
                state_names = ['cos_θ', 'sin_θ', 'θ̇']
                for i in range(min(3, rls_var_overall.shape[1])):
                    ax2.plot(t_overall, rls_var_overall[:, i], label=f'{state_names[i]}', alpha=0.7)
                ax2.set_ylabel(r'RLS Variance by State')
                ax2.set_xlabel(r'Training Time [s]')
                ax2.set_yscale('log')
                ax2.legend(loc='upper right', fontsize=8)
                ax2.grid(True)
                ax2.set_title('RLS Variance per State (Whole Training)')
            
            # Best episode RLS predictions vs ground truth
            if len(self.best_episode_rls_predictions) > 0:
                ax3 = fig4.add_subplot(2,2,3)
                predictions = np.array(self.best_episode_rls_predictions)
                ground_truth = np.array(self.best_episode_rls_ground_truth)
                
                # Ensure consistent shapes
                predictions = np.squeeze(predictions)
                ground_truth = np.squeeze(ground_truth)
                
                # Ensure both are 2D (time_steps, state_size)
                if predictions.ndim == 1:
                    predictions = predictions.reshape(1, -1)
                if ground_truth.ndim == 1:
                    ground_truth = ground_truth.reshape(1, -1)
                
                t_best = t[:len(predictions)]
                
                # Plot prediction vs ground truth for all states
                ax3.plot(t_best, ground_truth[:, 0], 'b-', label='True cos(θ)', linewidth=2)
                ax3.plot(t_best, predictions[:, 0], 'r--', label='Predicted cos(θ)', linewidth=1.5)
                ax3.plot(t_best, ground_truth[:, 1], 'g-', label='True sin(θ)', alpha=0.7)
                ax3.plot(t_best, predictions[:, 1], 'm--', label='Predicted sin(θ)', alpha=0.7)
                ax3.set_ylabel(r'State Values')
                ax3.set_xlabel(r'Time [s]')
                ax3.legend(loc='upper right', fontsize=8)
                ax3.grid(True)
                ax3.set_title(f'RLS Predictions vs Truth (Best Episode {self.best_episode})')
                
                # Prediction error analysis
                ax4 = fig4.add_subplot(2,2,4)
                prediction_errors = np.abs(predictions - ground_truth)
                mean_error = np.mean(prediction_errors, axis=1)
                ax4.plot(t_best, mean_error, 'r-', label='Mean Prediction Error')
                ax4.set_ylabel(r'Prediction Error')
                ax4.set_xlabel(r'Time [s]')
                ax4.set_yscale('log')
                ax4.legend(loc='upper right')
                ax4.grid(True)
                ax4.set_title(f'RLS Model Accuracy (Best Episode {self.best_episode})')
            
            plt.tight_layout()

        ### Figure 5: Performance Distribution Analysis ###
        fig5 = plt.figure(figsize=(12, 8))
        
        # Position error distribution
        ax1 = fig5.add_subplot(2,2,1)
        ax1.hist(np.array(self.episode_position_errors)*180/np.pi, bins=50, alpha=0.7, color=colors[0])
        ax1.set_xlabel('Position Error [degrees]')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Position Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Success rate over time
        ax2 = fig5.add_subplot(2,2,2)
        # Calculate rolling success rate (50-episode window)
        if len(self.episode_success_rates) >= 50:
            rolling_success = []
            for i in range(49, len(self.episode_success_rates)):
                rolling_success.append(np.mean(self.episode_success_rates[i-49:i+1]))
            ax2.plot(range(49, len(self.episode_success_rates)), rolling_success, color=colors[2], linewidth=2)
        ax2.plot(self.episode_success_rates, alpha=0.3, color=colors[2])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate Evolution (50-episode rolling mean)')
        ax2.grid(True, alpha=0.3)
        
        # Episode rewards distribution
        ax3 = fig5.add_subplot(2,2,3)
        rewards_array = np.array([float(r) if np.isscalar(r) else float(r[0]) if hasattr(r, '__len__') else 0.0 
                         for r in self.episode_rewards])
        ax3.hist(rewards_array, bins=50, alpha=0.7, color=colors[1])

        ax3.set_xlabel('Episode Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Episode Reward Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Learning curve with convergence threshold
        ax4 = fig5.add_subplot(2,2,4)
        ax4.plot(self.episode_position_errors, color=colors[0], alpha=0.7, label='Position Error')
        ax4.axhline(y=0.5, color='red', linestyle='--', label='Convergence Threshold (0.5 rad)')
        # Mark convergence episodes
        if self.convergence_episodes:
            conv_errors = [self.episode_position_errors[i] for i in self.convergence_episodes]
            ax4.scatter(self.convergence_episodes, conv_errors, color='green', s=20, alpha=0.7, label='Convergent Episodes')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Position Error [rad]')
        ax4.set_title('Learning Curve with Convergence Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()

        # Align labels and show plots
        fig0.align_labels()
        fig1.align_labels()
        fig2.align_labels() 
        if len(self.best_episode_actor_grads) > 0:
            fig3.align_labels()
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            fig4.align_labels()
        fig5.align_labels()
        
        # Save figures
        fig0.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/dhp_pendulum_training_progress.png', dpi=150, bbox_inches='tight')
        fig1.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/dhp_pendulum_angle_control.png', dpi=150, bbox_inches='tight')
        fig2.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/dhp_pendulum_control_analysis.png', dpi=150, bbox_inches='tight')
        if len(self.best_episode_actor_grads) > 0:
            fig3.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/dhp_pendulum_neural_analysis.png', dpi=150, bbox_inches='tight')
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            fig4.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/dhp_pendulum_rls_analysis.png', dpi=150, bbox_inches='tight')
        fig5.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/dhp_pendulum_performance_analysis.png', dpi=150, bbox_inches='tight')
        
        plt.show()
        
        print("DHP PENDULUM BEST EPISODE analysis plots generated and saved!")
        print("Figures saved:")
        print("  - Training Progress: dhp_pendulum_training_progress.png")
        print(f"  - Angle Control: dhp_pendulum_angle_control.png (Best Episode {self.best_episode})")
        print(f"  - Control Analysis: dhp_pendulum_control_analysis.png (Best Episode {self.best_episode})") 
        if len(self.best_episode_actor_grads) > 0:
            print(f"  - Neural Network Analysis: dhp_pendulum_neural_analysis.png (Best Episode {self.best_episode})")
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            print(f"  - RLS Analysis: dhp_pendulum_rls_analysis.png (Training Progress + Best Episode {self.best_episode})")
        print(f"  - Performance Analysis: dhp_pendulum_performance_analysis.png (Statistical Analysis)")
        print(f"All state vs reference plots show Episode {self.best_episode} (Position Error: {np.rad2deg(self.best_position_error):.1f}°)")
        print(f"Phase space plots show pendulum trajectory in cos/sin and θ-θ̇ coordinates")

    def demonstrate_policy(self, gui=True, record=True, episode_length=200, real_time=True):
        """
        Demonstrate the trained policy with visualization (adapted for pendulum)
        """
        print("\n" + "="*50)
        print("DEMONSTRATING TRAINED DHP PENDULUM POLICY")
        print("="*50)
        
        # Load the best model for demonstration
        self.load_best_model()
        
        # Create demonstration environment with GUI
        demo_env = PendulumRandomTargetEnv(
            fixed_target=None,  # Random target for demonstration
            normalize_states=self.config['normalize_states'],
            gui=gui,
            record=record
        )
        
        # Reset environment
        state, info = demo_env.reset()
        reference = info['reference']
        
        max_steps = episode_length
        demo_states = []
        demo_actions = []
        demo_costs = []
        demo_times = []
        target_angle = info['target_angle']
        
        print(f"Running demonstration for {episode_length} steps...")
        print(f"Target angle: {target_angle:.3f} rad ({np.rad2deg(target_angle):.1f}°)")
        if real_time:
            print("Running in real-time mode")
        else:
            print("Running in fast mode")
        
        # Track real time for proper visualization
        start_real_time = time.time()
        
        for step in range(max_steps):
            # Get action from trained policy (no excitation)
            # Apply normalization if it was used during training
            X_normalized = self.normalize_state(state) if self.config['normalize_states'] else state
            R_sig_normalized = self.normalize_reference(reference) if self.config['normalize_states'] else reference
            
            X_shaped = X_normalized.reshape(1, -1, 1)
            R_sig_shaped = R_sig_normalized.reshape(1, -1, 1)
            action = self.agent.action(X_shaped, reference=R_sig_shaped)
            action_flat = action.flatten()
            action_clipped = np.clip(action_flat, -2.0, 2.0)  # Pendulum torque range
            
            # Execute action
            next_state, reward, terminated, truncated, info = demo_env.step(action_clipped)
            
            # Store demo data (store raw states for plotting)
            demo_states.append(state.copy())
            demo_actions.append(action_clipped.copy())
            demo_costs.append(info['dhp_cost'])
            demo_times.append(step * 0.02)  # Assuming ~0.02s per step

            # Update for next step
            state = next_state
            reference = info['reference']
            
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
                print(f"Step: {step:3d}, Angle Error: {info['position_error']:.3f} rad ({np.rad2deg(info['position_error']):.1f}°), Cost: {info['dhp_cost']:.3f}")
            
            # Check termination
            if terminated or truncated:
                break
        
        demo_env.close()
        
        # Print final statistics
        final_pos_error = info['position_error']
        final_success = info['episode_success']
        avg_cost = np.mean(demo_costs)
        print(f"\nDemonstration completed!")
        print(f"Final position error: {final_pos_error:.4f} rad ({np.rad2deg(final_pos_error):.1f}°)")
        print(f"Episode success: {final_success}")
        print(f"Average cost: {avg_cost:.3f}")
        
        if record:
            print("Recording saved (if environment supports it)")
        
        return demo_states, demo_actions, demo_costs, demo_times
    
    def load_best_model(self):
        """
        Load the best saved model for demonstration
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/trained_models"
        
        # Try to load best model first
        best_model_path = f"{checkpoint_dir}/dhp_pendulum_best"
        if os.path.exists(f"{best_model_path}.pkl"):
            print("Loading best saved model for demonstration...")
            try:
                with open(f"{checkpoint_dir}/dhp_pendulum_best_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                    print(f"Best model: Episode {metadata['best_episode']}, Error: {np.rad2deg(metadata['best_position_error']):.1f}°")
                
                self.agent.load(best_model_path)
                print("✅ Best model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error loading best model: {e}")
        
        # Fallback to final model
        final_model_path = f"{checkpoint_dir}/dhp_pendulum_final"
        if os.path.exists(f"{final_model_path}.pkl"):
            print("Loading final model for demonstration...")
            try:
                self.agent.load(final_model_path)
                print("✅ Final model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error loading final model: {e}")
        
        print("⚠️  No saved model found, using current model...")
        return False


if __name__ == "__main__":
    print("DHP Training for Pendulum Random Target Environment")
    print("==================================================")
    
    # Configuration for pendulum DHP training (optimized for stability)
    config_override = {
        # Training settings
        'num_episodes': 400,
        'episode_length': 200,
        'max_steps': 200,
        'excitation_steps': 3000,      # Reduced excitation phase
        'excitation_amplitude': 0.15,   # Reduced excitation amplitude
        'log_interval': 50,
        'save_interval': 200,
        'gui': False,
        'record': False,
        
        # State normalization (critical for DHP success)
        'normalize_states': True,
        'record_best_episodes': True,
        
        # DHP hyperparameters (optimized for pendulum)
        'lr_critic': 0.008,            # Slightly lower learning rates
        'lr_actor': 0.004,
        'hidden_layer_size': [64, 64, 32],
        'gamma': 0.95,
        'update_cycles': 2,
        'tau': 0.001,
        
        # RLS stability improvements
        'rls_gamma': 0.999,            # Higher forgetting factor
        'rls_covariance': 10.0,        # Lower initial covariance
    }
    
    # Create and configure trainer
    trainer = PendulumDHPTrainer()
    trainer.config.update(config_override)
    
    print(f"\nPendulum DHP Training Configuration:")
    print(f"  Episodes: {trainer.config['num_episodes']}")
    print(f"  Episode length: {trainer.config['max_steps']} steps")
    print(f"  State format: [theta, theta_dot]")
    print(f"  State normalization: {trainer.config['normalize_states']}")
    print(f"  Target: {trainer.config['fixed_target']:.3f} rad ({np.rad2deg(trainer.config['fixed_target']):.1f}°)")
    
    # Train the DHP agent
    print("\nStarting DHP training...")
    trainer.train()
    
    print("\n🎯 DHP Pendulum training completed!")
    print("Check training_logs/ and trained_models/ for results.")
    if trainer.session_best_episode_num != -1:
        print(f"  ðŸŽ¥ Session-best episode recorded: {trainer.session_best_episode_num}")
        print(f"  ðŸŽ¥ Recorded error: {trainer.session_best_error:.4f} rad ({np.rad2deg(trainer.session_best_error):.1f}°)")
    else:
        print("  âš ï¸  No significant improvement achieved during training")
        print("  ðŸ'  Consider adjusting hyperparameters or extending training")

    print("\nFiles generated:")
    print("  📊 Training plots: dhp_pendulum_*.png")
    print("  ðŸ‹ Training logs: training_logs/")
    print("  🤖 Trained models: trained_models/")
    if trainer.session_best_episode_num != -1:
        print("  ðŸŽ¥ Best episode recording: best_episode_recordings/")
    
    print(f"\nFor detailed analysis, check the log file:")
    print(f"  {trainer.log_filename}")
    
    print("\n" + "="*60)
    print("DHP PENDULUM TRAINING COMPLETED SUCCESSFULLY!")
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
    summary_file = "/home/osos/Mohamed_Masters_Thesis/DHP_pendulum/training_summary.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("DHP Pendulum Training Summary\n")
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
        f.write(f"Hidden layers: {trainer.config['hidden_layer_size']}\n")
        f.write(f"Learning rates: critic={trainer.config['lr_critic']}, actor={trainer.config['lr_actor']}\n")
        f.write(f"Training time: {time.time() - trainer.training_start_time:.1f} seconds\n")
        f.write(f"Log file: {trainer.log_filename}\n")
    
    print(f"\n📄 Training summary saved: {summary_file}")
    
    # Optional: Quick demonstration if requested
    user_input = input("\nWould you like to run a quick demonstration of the trained policy? (y/n): ").strip().lower()
    if user_input in ['y', 'yes']:
        print("\n" + "="*50)
        print("RUNNING QUICK DEMONSTRATION")
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
                final_angle = demo_states[-1][0]
                final_theta_dot = demo_states[-1][1]
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
    
    print("\n🎯 Training session complete! Ready for DHP vs SAC comparison study.")


    print("DHP Training for Pendulum Random Target Environment")
    print("==================================================")
    
    # Check if user wants to run ablation study
    run_mode = input("Select run mode:\n1. Single training run (default)\n2. Ablation study (multiple configurations)\nEnter choice (1/2): ").strip()
    
    if run_mode == '2':
        # Run ablation study
        results = run_ablation_study()
    else:
        # Single training run (default)
        
        # Configuration for pendulum DHP training (optimized based on CF2X success)
        config_override = {
            # Training settings optimized for pendulum
            'num_episodes': 1500,           # More episodes for pendulum convergence
            'episode_length': 200,          # Standard pendulum episode length
            'max_steps': 200,
            'excitation_steps': 7500,       # 50% of total steps for exploration
            'excitation_amplitude': 0.2,    # Reasonable for pendulum torque range
            'log_interval': 50,             # More frequent logging for shorter episodes
            'save_interval': 200,
            'gui': False,                   # Training without GUI for speed
            'record': False,
            
            # State normalization settings (CRITICAL for DHP success)
            'normalize_states': True,       # ENABLE normalization for better convergence
            
            # Best episode recording
            'record_best_episodes': True,   # Enable recording of best episodes
            
            # DHP hyperparameters (proven stable from CF2X)
            'lr_critic': 0.01,             # Same as CF2X
            'lr_actor': 0.005,             # Same as CF2X  
            'hidden_layer_size': [64, 64, 32],  # Same architecture as CF2X
            'gamma': 0.95,                 # Slightly lower for faster episodes
            'update_cycles': 2,            # Same as CF2X
            'tau': 0.001,                  # Same as CF2X
        }
        
        # Create trainer with default config, then update with overrides
        trainer = PendulumDHPTrainer()
        trainer.config.update(config_override)
        
        # Log the configuration update
        trainer.logger.info("\nCONFIGURATION OVERRIDES APPLIED:")
        for key, value in config_override.items():
            trainer.logger.info(f"  {key}: {value}")
        
        print("\nUpdated config for pendulum DHP training:")
        for key, value in config_override.items():
            print(f"  {key}: {value}")
        
        print(f"\nPendulum training configuration:")
        print(f"  Episodes: {trainer.config['num_episodes']}")
        print(f"  Episode length: {trainer.config['max_steps']} steps")
        print(f"  State normalization: {trainer.config['normalize_states']}")
        print(f"  Random targets: {trainer.config['fixed_target'] is None}")
        print(f"  Success threshold: < 0.1 rad (5.7°)")
        print(f"  Convergence threshold: < 0.5 rad (28.6°)")
        
        # Train the DHP agent
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
            final_angle = demo_states[-1][0]
            final_theta_dot = demo_states[-1][1]
            demo_error = abs(final_angle)  # Error from vertical (target would be variable)
            trainer.logger.info(f"Demonstration final angle error: {np.rad2deg(demo_error):.1f}°")
        trainer.logger.info("Training and demonstration session completed successfully!")
        
        print("\nDHP pendulum training and demonstration completed!")
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
                
            # Session-best recording information
            if trainer.session_best_episode_num != -1:
                print(f"  🎥 Session-best episode recorded: {trainer.session_best_episode_num}")
                print(f"  🎥 Recorded error: {trainer.session_best_error:.4f} rad ({np.rad2deg(trainer.session_best_error):.1f}°)")
        else:
            print("  ⚠️ No significant improvement achieved during training")
            print("  💡 Consider adjusting hyperparameters or extending training")
        
        print("\nFiles generated:")
        print("  📊 Training plots: dhp_pendulum_*.png")
        print("  📋 Training logs: training_logs/")
        print("  🤖 Trained models: trained_models/")
        if trainer.session_best_episode_num != -1:
            print("  🎥 Best episode recording: best_episode_recordings/")
        
        print(f"\nFor detailed analysis, check the log file:")
        print(f"  {trainer.log_filename}")
        
        print("\n" + "="*60)
        print("DHP PENDULUM TRAINING COMPLETED SUCCESSFULLY!")
        print("\n" + "="*60)