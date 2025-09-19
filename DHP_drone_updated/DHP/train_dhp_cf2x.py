"""
DHP Training Script for CF2X Quadrotor

This script adapts the msc-thesis dhp_main.py training loop for quadrotor control
using the CF2X fast states environment and existing DHP agent implementation.

Author: DHP vs SAC Comparison Study  
Date: August 8, 2025
"""

import numpy as np
import sys
import os
import time
import importlib
import matplotlib.pyplot as plt
import json
import logging
import glob
import pickle
from datetime import datetime

# Set TensorFlow to use CPU only to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add paths for existing implementations (put our local msc-thesis first)  
sys.path.append('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/gym-pybullet-drones')
sys.path.append('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP')
sys.path.append('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/msc-thesis')

# Import existing DHP components from msc-thesis
from agents.dhp import Agent as DHP_Agent
from agents.model import RecursiveLeastSquares

# Import our custom environment
from cf2x_fast_states_env import CF2X_FastStates_HoverAviary

# Import utilities
import tensorflow as tf

# Import quadrotor DHP modes for split architecture
from quadrotor_dhp_modes import create_quadrotor_phlab_extension

class QuadrotorDHPTrainer:
    """
    DHP trainer for CF2X quadrotor adapted from msc-thesis dhp_main.py
    """
    
    def __init__(self, config=None):
        """
        Initialize DHP trainer with configuration
        """
        # Default configuration (adapted from msc-thesis)
        self.config = config or {
            # Environment settings
            'target_pos': [0.5, 0.5, 1.0],
            'episode_length': 20.0,        # 20 seconds
            'dt': 1.0/30.0,               # 30 Hz control frequency
            'use_trajectory': False,      # Enable trajectory following training
            'trajectory_type': 'spiral',  # 'spiral', 'figure8', 'circle'

            # DHP agent settings (optimized for split architecture convergence)
            'state_size': 8,              # Fast states (z, roll, pitch, yaw, vz, wx, wy, wz)
            'reference_size': 8,          # Reference vector  
            'action_size': 4,             # Motor RPMs
            'hidden_layer_size': [64, 64, 32],  # IMPROVED: Larger networks for split complexity
            'lr_critic': 0.01,           # PROVEN STABLE: Achieved 0.000093m with 90.08% convergence
            'lr_actor': 0.005,           # PROVEN STABLE: Achieved 0.000093m with 90.08% convergence  
            'lr_decay': 0.9999,          # Learning rate decay per episode to prevent late-stage NaN
            'gradient_clip_value': 0.5,  # Much stricter gradient clipping for stability
            'gamma': 0.98,               # INCREASED for better long-term coordination
            'split': False,               # ENABLE split architecture for vertical/attitude separation
            'target_network': True,
            'tau': 0.001,               # SLOWER target network updates for stability
            
            # RLS model settings (updated for extended states)
            'rls_state_size': 8,          # RLS model state size
            'rls_action_size': 4,
            'rls_gamma': 0.9995,
            'rls_covariance': 100.0,
            'predict_delta': False,
            
            # Training settings (optimized for split architecture convergence)
            'num_episodes': 2500,         # INCREASED for split architecture learning
            'update_cycles': 5,           # INCREASED for better gradient updates per step
            'excitation_steps': 12500,    # PROVEN STABLE: Achieved 90.08% convergence (not 15000!)
            'excitation_amplitude': 0.03, # REDUCED for gentler exploration
            
            # State normalization settings
            'normalize_states': True,      # Enable/disable state normalization (CORRECTED: was actually enabled during training)
            # State normalization bounds (matching environment limits)
            'state_bounds': {
                'z': [0.0, 5.0],         # Altitude bounds [m]
                'roll': [-np.pi/2, np.pi/2],     # Roll angle bounds [rad]
                'pitch': [-np.pi/2, np.pi/2],    # Pitch angle bounds [rad]
                'yaw': [-np.pi, np.pi],  # Yaw angle bounds [rad]
                'vz': [-12.0, 12.0],       # Vertical velocity bounds [m/s]
                'wx': [-10.0, 10.0],       # Roll rate bounds [rad/s]
                'wy': [-10.0, 10.0],       # Pitch rate bounds [rad/s]
                'wz': [-10.0, 10.0]        # Yaw rate bounds [rad/s]
            },
            # Note: When normalize_states=True, all states are normalized to [-1, 1] range
            # This can improve neural network training stability and convergence
            # When normalize_states=False, states remain in their original physical units
            
            # Logging
            'log_interval': 10,
            'save_interval': 100,
            'gui': False,              # Disable GUI during training for speed
            'record': False,           # Don't record all episodes
            
            # Best episode recording settings
            'record_best_episodes': True,   # Enable recording of best episodes
            'best_episode_gui': False,      # Show GUI only for best episodes (set True to see them)
            'recording_fps': 30,            # Recording frame rate
            
            # Split architecture coordination settings
            'coordination_warmup_episodes': 400,  # Episodes before enabling full coordination
            'coordination_strength': 0.1,        # Coupling strength between networks
        }
        
        # Initialize components
        self.env = None
        self.agent = None
        self.model = None
        
        # Training data storage
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_position_errors = []
        self.episode_durations = []  # Track how long each episode lasted (in steps)
        self.episode_completion_rates = []  # Track completion percentage (0.0 to 1.0)
        self.training_start_time = None
        
        # Performance tracking for logging with smarter criteria
        self.best_position_error = float('inf')
        self.best_episode = -1
        self.best_episode_score = float('-inf')  # Composite score for better selection
        self.best_completed_episode = -1  # Best among episodes that completed fully
        self.best_completed_error = float('inf')  # Error of best completed episode
        self.convergence_episodes = []  # Episodes with pos_error < 1.0m
        self.stable_performance_start = -1  # Episode when stable performance begins
        self.best_model_saved = False  # Track if best model has been saved
        
        # Early stopping criteria
        self.early_stopping_patience = 100  # Episodes to wait after no improvement
        self.early_stopping_threshold = 0.8  # Position error threshold for good performance
        self.no_improvement_counter = 0  # Count episodes without improvement
        self.early_stopping_triggered = False
        
        # Setup comprehensive logging
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
        
        # Best episode X,Y position tracking (not part of fast states but needed for analysis)
        self.best_episode_x_positions = []
        self.best_episode_y_positions = []
        self.best_episode_x_references = []
        self.best_episode_y_references = []
        
        # Session-best recording data (for replay-at-end strategy)
        self.session_best_episode_num = -1
        self.session_best_error = float('inf')
        self.session_best_actions_sequence = []  # Action sequence for replay (full precision)
        self.session_best_initial_conditions = None  # EXACT initial conditions for perfect replay
        self.session_best_replay_file = None  # Path to saved replay data
        
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
            'x_positions': [],      # Track actual X position from full state
            'y_positions': [],      # Track actual Y position from full state
            'x_references': [],     # Track X reference (from target_pos)
            'y_references': []      # Track Y reference (from target_pos)
        }
        
        print("QuadrotorDHPTrainer initialized with config:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
    
    def setup_logging(self):
        """
        Setup comprehensive logging for training session
        """
        # Create logs directory
        log_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{log_dir}/dhp_split_training_{timestamp}.log"
        
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
        self.logger.info("DHP SPLIT ARCHITECTURE TRAINING SESSION STARTED")
        self.logger.info("="*80)
        
        # Log all hyperparameters
        self.logger.info("HYPERPARAMETERS:")
        self.logger.info("-" * 40)
        for key, value in sorted(self.config.items()):
            self.logger.info(f"{key:25}: {value}")
        
        self.logger.info("-" * 40)
        self.logger.info(f"Log file: {self.log_filename}")
        self.logger.info("="*80)
    
    def log_performance_metrics(self, episode, episode_reward, avg_cost, pos_error):
        """
        Log performance metrics and track convergence indicators
        """
        # Update best performance and save best model
        if pos_error < self.best_position_error:
            self.best_position_error = pos_error
            self.best_episode = episode
            self.logger.info(f"NEW BEST PERFORMANCE! Episode {episode}: {pos_error:.4f}m")
            
            # Automatically save the best model
            self.save_best_model(episode, pos_error)
        
        # Track convergence episodes (< 1.0m position error)
        if pos_error < 1.0:
            self.convergence_episodes.append(episode)
            self.logger.info(f"CONVERGENCE EPISODE {episode}: {pos_error:.4f}m")
        
        # Check for stable performance (10 consecutive episodes < 1.0m)
        if len(self.convergence_episodes) >= 10:
            recent_convergent = self.convergence_episodes[-10:]
            if all(recent_convergent[i] == recent_convergent[0] + i for i in range(10)):
                if self.stable_performance_start == -1:
                    self.stable_performance_start = recent_convergent[0]
                    self.logger.info(f"STABLE PERFORMANCE ACHIEVED starting at episode {self.stable_performance_start}")
        
        # Log milestone episodes
        if episode in [100, 250, 500, 750, 1000, 1500, 2000, 2500]:
            self.log_milestone(episode)
    
    def log_milestone(self, episode):
        """
        Log detailed milestone analysis
        """
        if len(self.episode_position_errors) == 0:
            return
            
        recent_errors = self.episode_position_errors[-50:] if len(self.episode_position_errors) >= 50 else self.episode_position_errors
        recent_costs = self.episode_costs[-50:] if len(self.episode_costs) >= 50 else self.episode_costs
        
        avg_pos_error = np.mean(recent_errors)
        std_pos_error = np.std(recent_errors)
        avg_cost = np.mean(recent_costs)
        convergent_ratio = len([e for e in recent_errors if e < 1.0]) / len(recent_errors)
        
        self.logger.info("")
        self.logger.info(f"MILESTONE {episode} ANALYSIS:")
        self.logger.info(f"  Recent 50 episodes avg position error: {avg_pos_error:.4f} ± {std_pos_error:.4f}m")
        self.logger.info(f"  Recent 50 episodes avg cost: {avg_cost:.3f}")
        self.logger.info(f"  Convergent episodes ratio (< 1.0m): {convergent_ratio:.2%}")
        self.logger.info(f"  Total convergent episodes so far: {len(self.convergence_episodes)}")
        self.logger.info(f"  Best performance: {self.best_position_error:.4f}m at episode {self.best_episode}")
        
        if self.stable_performance_start != -1:
            self.logger.info(f"  Stable performance maintained since episode {self.stable_performance_start}")
        else:
            self.logger.info("  Stable performance not yet achieved")
        self.logger.info("")
    
    def save_best_model(self, episode, pos_error):
        """
        Save the best performing model with consistent checkpoint numbering
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/trained_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save best model using DHP's built-in save
        best_model_path = f"{checkpoint_dir}/dhp_cf2x_best"
        
        try:
            # Save with consistent numbering scheme
            # First call: saves hyperparameters (.pkl)
            # Second call: saves TensorFlow checkpoint with number
            self.agent.save(file_path=best_model_path, global_step=None)     # Creates non-numbered checkpoint

            # Save additional metadata for debugging
            metadata = {
                'episode': episode,
                'position_error': pos_error,
                'config': self.config.copy(),
                'state_size': self.config['state_size'],
                'action_size': self.config['action_size'],
                'split_architecture': self.config['split'],
                'normalization_enabled': self.config.get('normalize_states', False),
                'timestamp': time.time()
            }
            
            import pickle
            with open(f"{best_model_path}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            self.logger.info(f"BEST MODEL SAVED! Episode {episode}, Error: {pos_error:.4f}m")
            self.logger.info(f"Best model path: {best_model_path}")
            
            # Verify all expected files were created
            expected_files = [
                f"{best_model_path}.pkl",                           # Hyperparameters
                f"{best_model_path}.index",                       # TF checkpoint index
                f"{best_model_path}.meta",                        # TF checkpoint meta
                f"{best_model_path}.data-00000-of-00001",        # TF checkpoint data
                f"{best_model_path}_metadata.pkl"                  # Metadata
            ]
            
            missing_files = [f for f in expected_files if not os.path.exists(f)]
            if missing_files:
                self.logger.warning(f"Some model files were not created: {missing_files}")
            else:
                self.logger.info("All model files saved successfully!")
                
            self.best_model_saved = True
            
        except Exception as e:
            self.logger.error(f"Failed to save best model: {str(e)}")
            print(f"Error saving best model: {str(e)}")
            self.best_model_saved = False

    def setup_environment(self):
        """
        Setup CF2X fast states environment
        """
        print("\n[SETUP] Initializing CF2X Fast States Environment...")
        
        self.env = CF2X_FastStates_HoverAviary(
            target_pos=np.array(self.config['target_pos']),
            gui=self.config['gui'],
            record=self.config['record'],
            use_trajectory=self.config.get('use_trajectory', False),
            trajectory_type=self.config.get('trajectory_type', 'spiral')
        )
        
        print(f"Environment observation space: {self.env.observation_space}")
        print(f"Environment action space: {self.env.action_space}")
        if self.config.get('use_trajectory', False):
            print(f"TRAJECTORY MODE: {self.config.get('trajectory_type', 'spiral')}")
        else:
            print(f"FIXED TARGET MODE: {self.config['target_pos']}")
        
    def setup_dhp_agent(self):
        """
        Setup DHP agent using existing msc-thesis implementation with quadrotor split support
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
            'tau': self.config['tau']
        }
        
        if self.config['split']:
            # Enable split architecture with quadrotor compatibility
            agent_kwargs['split'] = True
            
            # Patch phlab module for quadrotor support
            self._setup_quadrotor_phlab_compatibility()
            agent_kwargs['mode_id'] = 300  # Use LATLON mode (now patched for quadrotor)
            # The use_delta parameter: (use_delta_flag, tracked_states_list)
            # For quadrotor, we track all 8 states but don't use delta states
            agent_kwargs['use_delta'] = (False, [True] * 8)  # Track all 8 fast states
        else:
            agent_kwargs['split'] = False
        
        # Create DHP agent (reusing msc-thesis implementation)
        self.agent = DHP_Agent(**agent_kwargs)
        
        # Set trim to hover RPMs (normalized to [-1, 1] range)
        # For CF2X: hover_rpm ≈ 16800, action range is ±5% -> trim = 0.0
        self.agent.trim = np.zeros(self.config['action_size'])
        
        print(f"DHP Agent created with split architecture: {agent_kwargs.get('split', False)}")
        print(f"Network architecture: {self.config['hidden_layer_size']}")
        if self.config['split']:
            print("Note: Using quadrotor-specific split architecture (vertical + attitude control)")
        else:
            print("Note: Using unified architecture")
    
    def _setup_quadrotor_phlab_compatibility(self):
        """
        Patch the phlab module to support quadrotor states for split architecture
        """
        import utils.phlab as phlab
        
        # Get quadrotor extension definitions
        quad_ext = create_quadrotor_phlab_extension()
        
        # Add quadrotor modes to phlab
        phlab.ID_QUAD_LON = quad_ext['ID_QUAD_LON']
        phlab.ID_QUAD_LAT = quad_ext['ID_QUAD_LAT'] 
        phlab.ID_QUAD_FULL = quad_ext['ID_QUAD_FULL']
        
        # Update phlab dictionaries
        phlab.states.update(quad_ext['states'])
        phlab.idx.update(quad_ext['idx'])
        phlab.track_states.update(quad_ext['track_states'])
        
        # Override the LATLON mode (300) with quadrotor full states
        phlab.states[300] = quad_ext['states'][quad_ext['ID_QUAD_FULL']]
        phlab.idx[300] = quad_ext['idx'][quad_ext['ID_QUAD_FULL']]
        phlab.track_states[300] = quad_ext['track_states'][quad_ext['ID_QUAD_FULL']]
        
        # Override the LON/LAT IDs for quadrotor split
        phlab.ID_LON = quad_ext['ID_QUAD_LON']
        phlab.ID_LAT = quad_ext['ID_QUAD_LAT']
        
        print("✓ PHLab module patched for quadrotor split architecture")
        print(f"  - Longitudinal states: {quad_ext['states'][quad_ext['ID_QUAD_LON']]}")
        print(f"  - Lateral states: {quad_ext['states'][quad_ext['ID_QUAD_LAT']]}")
        print(f"  - LATLON mode (300) overridden with: {quad_ext['states'][quad_ext['ID_QUAD_FULL']]}")
        
    def setup_rls_model(self):
        """
        Setup RLS dynamics model using existing msc-thesis implementation
        """
        print("\n[SETUP] Initializing RLS Dynamics Model...")
        
        # Model configuration (following msc-thesis format)
        model_kwargs = {
            'state_size': self.config['rls_state_size'],
            'action_size': self.config['rls_action_size'],
            'gamma': self.config['rls_gamma'],
            'covariance': self.config['rls_covariance'],
            'predict_delta': self.config['predict_delta']
        }
        
        # Create RLS model (reusing msc-thesis implementation)
        self.model = RecursiveLeastSquares(**model_kwargs)
        
        print(f"RLS Model created with forgetting factor: {self.config['rls_gamma']}")
        print(f"Initial covariance: {self.config['rls_covariance']}")
        
    def normalize_state(self, state):
        """
        Normalize state vector to [-1, 1] range based on configured bounds
        
        Args:
            state: Raw state vector [z, roll, pitch, yaw, vz, wx, wy, wz]
            
        Returns:
            normalized_state: Normalized state vector in [-1, 1] range
        """
        if not self.config['normalize_states']:
            return state.copy()
        
        normalized_state = state.copy()
        state_names = ['z', 'roll', 'pitch', 'yaw', 'vz', 'wx', 'wy', 'wz']
        
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
        
        Args:
            normalized_state: Normalized state vector in [-1, 1] range
            
        Returns:
            state: State vector in physical units
        """
        if not self.config['normalize_states']:
            return normalized_state.copy()
        
        state = normalized_state.copy()
        state_names = ['z', 'roll', 'pitch', 'yaw', 'vz', 'wx', 'wy', 'wz']
        
        for i, name in enumerate(state_names):
            if name in self.config['state_bounds']:
                min_val, max_val = self.config['state_bounds'][name]
                # Denormalize from [-1, 1] to physical range
                state[i] = min_val + (normalized_state[i] + 1.0) * (max_val - min_val) / 2.0
        
        return state
    
    def normalize_reference(self, reference):
        """
        Normalize reference vector using same bounds as states
        
        Args:
            reference: Raw reference vector [same structure as state]
            
        Returns:
            normalized_reference: Normalized reference vector in [-1, 1] range
        """
        # Reference has same structure as state, so use same normalization
        return self.normalize_state(reference)
        
    def _transform_gradient_to_normalized_space(self, dcostdx_raw):
        """
        Transform cost gradient from raw physical space to normalized space
        
        Args:
            dcostdx_raw: Cost gradient in raw physical space
            
        Returns:
            dcostdx_normalized: Cost gradient in normalized space
        """
        if not self.config['normalize_states']:
            return dcostdx_raw.copy()
        
        dcostdx_normalized = dcostdx_raw.copy()
        state_names = ['z', 'roll', 'pitch', 'yaw', 'vz', 'wx', 'wy', 'wz']
        
        for i, name in enumerate(state_names):
            if name in self.config['state_bounds']:
                min_val, max_val = self.config['state_bounds'][name]
                # Chain rule: d(cost)/d(normalized) = d(cost)/d(raw) * d(raw)/d(normalized)
                # where d(raw)/d(normalized) = (max_val - min_val) / 2.0
                scaling_factor = (max_val - min_val) / 2.0
                dcostdx_normalized[i] = dcostdx_raw[i] * scaling_factor
        
        return dcostdx_normalized
        
    def generate_excitation_signal(self, step, episode_num=0):
        """
        Generate exploration signal with performance-aware decay
        Reduces exploration when good performance is consistently achieved
        """
        # Check if we're in a stable performance region
        stable_performance_factor = 1.0
        if len(self.episode_position_errors) >= 10:
            # If last 10 episodes all have position error < 1.0m, reduce excitation
            recent_errors = self.episode_position_errors[-10:]
            if all(error < 1.0 for error in recent_errors):
                stable_performance_factor = 0.3  # Reduce excitation significantly
                if episode_num % 50 == 0:  # Log occasionally
                    print(f"🔒 Episode {episode_num}: Stable performance detected, reducing excitation to {stable_performance_factor*100:.0f}%")
        
        if step < self.config['excitation_steps']:
            # Multi-frequency excitation for system identification
            t = step * self.config['dt']
            base_amplitude = self.config['excitation_amplitude'] * stable_performance_factor
            excitation = base_amplitude * np.array([
                0.5 * np.sin(2.0 * np.pi * 0.1 * t),    # Low freq for motor 1
                0.5 * np.sin(2.0 * np.pi * 0.15 * t),   # Low freq for motor 2
                0.5 * np.sin(2.0 * np.pi * 0.1 * t),    # Low freq for motor 3
                0.5 * np.sin(2.0 * np.pi * 0.15 * t)    # Low freq for motor 4
            ])
            return excitation
        else:
            return np.zeros(self.config['action_size'])
    
    def evaluate_episode_performance(self, episode_num, pos_error, completion_rate, episode_steps, expected_steps):
        """
        Smart episode evaluation that considers both performance and completion
        
        Args:
            episode_num: Current episode number
            pos_error: Final position error
            completion_rate: Fraction of episode completed (0.0 to 1.0)
            episode_steps: Actual steps completed
            expected_steps: Expected steps for full episode
            
        Returns:
            tuple: (is_best_overall, is_best_completed)
        """
        # Calculate composite score that prioritizes completion and performance
        # Score formula: completion_weight * completion_rate - error_weight * pos_error
        completion_weight = 100.0  # High weight for completion
        error_weight = 10.0       # Lower weight for position error
        
        # Composite score: higher is better
        composite_score = completion_weight * completion_rate - error_weight * pos_error
        
        # Bonus for episodes that complete 90%+ of expected duration
        if completion_rate >= 0.9:
            composite_score += 50.0  # Completion bonus
        
        # Penalty for very short episodes (likely crashes)
        if completion_rate < 0.1:
            composite_score -= 200.0  # Crash penalty
        
        # Check if this is the best overall episode (considering completion + performance)
        is_best_overall = composite_score > self.best_episode_score
        if is_best_overall:
            self.best_episode_score = composite_score
            self.best_position_error = pos_error
            self.best_episode = episode_num
            
            # Log the new best episode with detailed information
            completion_pct = completion_rate * 100
            duration_sec = episode_steps * self.config['dt']
            self.logger.info(f"🏆 NEW BEST EPISODE (SMART CRITERIA)! Episode {episode_num}")
            self.logger.info(f"   Position Error: {pos_error:.4f}m")
            self.logger.info(f"   Completion: {completion_pct:.1f}% ({episode_steps}/{expected_steps} steps)")
            self.logger.info(f"   Duration: {duration_sec:.1f}s / {self.config['episode_length']:.1f}s")
            self.logger.info(f"   Composite Score: {composite_score:.2f}")
        
        # Check if this is the best among completed episodes (≥90% completion)
        is_best_completed = False
        if completion_rate >= 0.9:  # 90% completion threshold
            if pos_error < self.best_completed_error:
                self.best_completed_error = pos_error
                self.best_completed_episode = episode_num
                is_best_completed = True
                
                completion_pct = completion_rate * 100
                duration_sec = episode_steps * self.config['dt']
                self.logger.info(f"🎯 NEW BEST COMPLETED EPISODE! Episode {episode_num}")
                self.logger.info(f"   Position Error: {pos_error:.4f}m (among completed episodes)")
                self.logger.info(f"   Completion: {completion_pct:.1f}% ({episode_steps}/{expected_steps} steps)")
                self.logger.info(f"   Duration: {duration_sec:.1f}s")
        
        return is_best_overall, is_best_completed
    
    def train_episode(self, episode_num):
        """
        Train single episode following dhp_main.py structure
        """
        # Performance-aware learning rate adaptation
        # Base decay rate
        base_lr_critic = self.config['lr_critic'] * (self.config['lr_decay'] ** episode_num)
        base_lr_actor = self.config['lr_actor'] * (self.config['lr_decay'] ** episode_num)
        
        # Check for stable performance and reduce learning rate further if needed
        stable_performance_factor = 1.0
        if len(self.episode_position_errors) >= 10:
            recent_errors = self.episode_position_errors[-10:]
            if all(error < 1.0 for error in recent_errors):
                stable_performance_factor = 0.5  # Reduce learning rate for stability
                if episode_num % 50 == 0:
                    print(f"🔒 Episode {episode_num}: Stable performance detected, reducing learning rate by 50%")
        
        # Apply performance-aware adjustment
        current_lr_critic = base_lr_critic * stable_performance_factor
        current_lr_actor = base_lr_actor * stable_performance_factor
        
        # Update agent learning rates
        if hasattr(self.agent, 'critic') and hasattr(self.agent.critic, 'optimizer'):
            self.agent.critic.optimizer.learning_rate.assign(current_lr_critic)
        if hasattr(self.agent, 'actor') and hasattr(self.agent.actor, 'optimizer'):
            self.agent.actor.optimizer.learning_rate.assign(current_lr_actor)
        
        # Log learning rate changes every 100 episodes
        if episode_num % 100 == 0 and episode_num > 0:
            print(f"Episode {episode_num}: Learning rates - Critic: {current_lr_critic:.6f}, Actor: {current_lr_actor:.6f}")
            if stable_performance_factor < 1.0:
                print(f"                      Performance factor applied: {stable_performance_factor}")
        
        # Early termination if networks have become unstable
        if hasattr(self, 'consecutive_nan_episodes') and self.consecutive_nan_episodes >= 5:
            print(f"Warning: Too many consecutive NaN episodes ({self.consecutive_nan_episodes}). Terminating training.")
            return 0.0, 10.0, 10.0  # Return high error to indicate failure
        
        # Reset environment
        state, info = self.env.reset()
        reference = info['reference']
        
        # CAPTURE EXACT INITIAL CONDITIONS for potential session-best replay
        initial_full_state = self.env._getDroneStateVector(0)
        initial_pos = initial_full_state[0:3].copy()    # [x, y, z]
        initial_rpy = initial_full_state[7:10].copy()   # [roll, pitch, yaw]
        initial_vel = initial_full_state[10:13].copy()  # [vx, vy, vz] 
        initial_ang_vel = initial_full_state[13:16].copy()  # [wx, wy, wz]
        
        # Safety check for initial state
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Warning: Invalid initial state at episode {episode_num}")
            return 0.0, 1.0, 10.0  # Return safe default values
        
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        max_steps = int(self.config['episode_length'] / self.config['dt'])
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
                # Denormalize predicted state for cost computation
                X_next_pred_raw = self.denormalize_state(X_next_pred) if self.config['normalize_states'] else X_next_pred
                cost, dcostdx_raw = self.env.compute_dhp_cost(X_next_pred_raw, R_sig_raw)
                
                # Safety check for cost computation
                if np.any(np.isnan(dcostdx_raw)) or np.any(np.isinf(dcostdx_raw)):
                    print(f"Warning: Invalid cost gradient at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                # If using normalization, transform cost gradient to normalized space
                if self.config['normalize_states']:
                    # Transform gradient from raw space to normalized space
                    # dcostdx_normalized = dcostdx_raw * d(raw)/d(normalized)
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
                
                # Apply much stricter gradient clipping for stability
                grad_critic = np.clip(grad_critic, -self.config['gradient_clip_value'], self.config['gradient_clip_value'])
                
                # Additional safety check before update
                if np.any(np.isnan(grad_critic)) or np.any(np.isinf(grad_critic)):
                    print(f"Warning: Invalid critic gradient after computation at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                self.agent.update_critic(X_shaped, reference=R_sig_shaped, gradient=grad_critic.reshape(1, 1, -1), learn_rate=current_lr_critic)
                
                # Actor update (Policy gradient)
                lmbda = self.agent.value_derivative(X_next_pred_shaped, reference=R_sig_shaped)
                
                # Check for NaN values
                if np.any(np.isnan(lmbda)) or np.any(np.isnan(B)):
                    print(f"Warning: NaN detected in actor update at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                grad_actor = (dcostdx + self.config['gamma'] * lmbda.flatten()) @ B

                # Apply much stricter gradient clipping for stability
                grad_actor = np.clip(grad_actor, -self.config['gradient_clip_value'], self.config['gradient_clip_value'])

                # Additional safety check before update
                if np.any(np.isnan(grad_actor)) or np.any(np.isinf(grad_actor)):
                    print(f"Warning: Invalid actor gradient after computation at episode {episode_num}, step {step}")
                    nan_detected = True
                    break
                
                # Delayed Actor update for stability
                if update_cycle % 2 == 0:
                    self.agent.update_actor(X_shaped, reference=R_sig_shaped, gradient=grad_actor.reshape(1, 1, -1), learn_rate=current_lr_actor)
            
            # If NaN was detected during DHP updates, skip the rest of this step
            if nan_detected:
                print(f"Skipping step {step} in episode {episode_num} due to NaN detection")
                break
            
            # Environment step
            action = self.agent.action(X_shaped, reference=R_sig_shaped)
            action_flat = action.flatten()  # Convert to 1D for environment
            
            # Add excitation for initial exploration
            excitation = self.generate_excitation_signal(episode_num * max_steps + step, episode_num)
            action_with_excitation = action_flat + excitation
            action_clipped = np.clip(action_with_excitation, -1.0, 1.0)
            
            # SAVE ACTION for potential session-best replay
            # Store the action that will actually be executed (with excitation and clipping)
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
            episode_time = episode_num * max_steps * self.config['dt'] + step * self.config['dt']
            self.current_episode_data['states'].append(X_raw.copy())  # Store raw state for plotting
            self.current_episode_data['references'].append(R_sig_raw.copy())  # Store raw reference for plotting
            # Note: action already stored above after clipping
            self.current_episode_data['costs'].append(info['dhp_cost'])
            self.current_episode_data['model_errors'].append(model_error)
            self.current_episode_data['times'].append(episode_time)
            
            # Extract X,Y positions from full drone state (not part of fast states)
            current_full_state = self.env._getDroneStateVector(0)
            current_x = current_full_state[0]  # X position [m]
            current_y = current_full_state[1]  # Y position [m]
            
            # X,Y references from current target position (trajectory or fixed)
            current_target = info.get('current_target', self.config['target_pos'])
            target_x = current_target[0] 
            target_y = current_target[1]
            
            # Store X,Y position tracking data
            self.current_episode_data['x_positions'].append(current_x)
            self.current_episode_data['y_positions'].append(current_y)
            self.current_episode_data['x_references'].append(target_x)
            self.current_episode_data['y_references'].append(target_y)
            
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
        
        # Store episode metrics with duration tracking
        self.episode_rewards.append(episode_reward)
        if episode_steps > 0:
            self.episode_costs.append(episode_cost / episode_steps)  # Average cost per step
        else:
            self.episode_costs.append(1.0)  # Safe default value
        
        # Calculate episode completion metrics
        expected_steps = int(self.config['episode_length'] / self.config['dt'])
        completion_rate = episode_steps / expected_steps if expected_steps > 0 else 0.0
        episode_duration_seconds = episode_steps * self.config['dt']
        
        self.episode_durations.append(episode_steps)
        self.episode_completion_rates.append(completion_rate)
        
        final_pos_error = info.get('position_error', 10.0) if not nan_detected else 10.0
        self.episode_position_errors.append(final_pos_error)
        
        # Smarter best episode selection with composite scoring
        is_best_episode, is_best_completed = self.evaluate_episode_performance(
            episode_num, final_pos_error, completion_rate, episode_steps, expected_steps
        )
        
        if is_best_episode:
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
            
            # Save X,Y position data for plotting (not part of fast states)
            self.best_episode_x_positions = self.current_episode_data['x_positions'].copy()
            self.best_episode_y_positions = self.current_episode_data['y_positions'].copy()
            self.best_episode_x_references = self.current_episode_data['x_references'].copy()
            self.best_episode_y_references = self.current_episode_data['y_references'].copy()
            
            # SAVE EXACT INITIAL CONDITIONS for perfect replay
            self.session_best_initial_conditions = {
                'position': initial_pos.copy(),      # [x, y, z]
                'orientation': initial_rpy.copy(),   # [roll, pitch, yaw]
                'velocity': initial_vel.copy(),      # [vx, vy, vz]
                'angular_velocity': initial_ang_vel.copy()  # [wx, wy, wz]
            }
            
            # Update session-best recording data if recording is enabled
            if self.config.get('record_best_episodes', True):
                self.session_best_episode_num = episode_num
                self.session_best_error = final_pos_error
                # Store action sequence for replay (already captured in current_episode_data)
                self.session_best_actions_sequence = [a.copy() for a in self.current_episode_data['actions']]
                
                print(f"🎯 NEW SESSION BEST EPISODE: {episode_num}, Error: {final_pos_error:.4f}m")
                print(f"   Initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
                print(f"   Initial orientation: [{initial_rpy[0]:.3f}, {initial_rpy[1]:.3f}, {initial_rpy[2]:.3f}]")
                print(f"   Action sequence captured: {len(self.session_best_actions_sequence)} steps")
            self.best_episode_rls_predictions = [p.copy() for p in self.current_episode_data['rls_predictions']]
            self.best_episode_rls_ground_truth = [gt.copy() for gt in self.current_episode_data['rls_ground_truth']]
            self.best_episode_time = self.current_episode_data['times'].copy()
            
            print(f"[BEST EPISODE] Episode {episode_num}: New best position error {final_pos_error:.4f}m")
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
            'x_positions': [],
            'y_positions': [],
            'x_references': [],
            'y_references': []
        }

        return episode_reward, episode_cost / (episode_steps + 1e-6), final_pos_error

    def train(self):
        """
        Main training loop with comprehensive logging
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("STARTING DHP TRAINING FOR CF2X QUADROTOR")
        self.logger.info("="*50)
        
        # Setup all components
        self.setup_environment()
        self.setup_dhp_agent()
        self.setup_rls_model()
        
        # Start training timer
        self.training_start_time = time.time()
        
        # Initialize NaN tracking
        self.consecutive_nan_episodes = 0
        
        self.logger.info(f"\nTraining for {self.config['num_episodes']} episodes...")
        self.logger.info("Episode | Reward  | Avg Cost | Pos Error | Duration (Completion) | Time")
        self.logger.info("-" * 85)
        
        # Training loop
        for episode in range(self.config['num_episodes']):
            episode_reward, avg_cost, pos_error = self.train_episode(episode)
            
            # Track NaN episodes
            if pos_error >= 10.0:  # High error indicates NaN detection or failure
                self.consecutive_nan_episodes += 1
            else:
                self.consecutive_nan_episodes = 0  # Reset counter on successful episode
            
            # Log performance metrics and track convergence
            self.log_performance_metrics(episode, episode_reward, avg_cost, pos_error)
            
            # Early stopping check
            if pos_error < self.early_stopping_threshold:
                self.no_improvement_counter = 0  # Reset counter on good performance
            else:
                self.no_improvement_counter += 1
                
            # Check for early stopping
            if (self.no_improvement_counter >= self.early_stopping_patience and 
                episode > 200):  # Don't stop too early
                self.early_stopping_triggered = True
                self.logger.info(f"\n🛑 EARLY STOPPING TRIGGERED at episode {episode}")
                self.logger.info(f"   No improvement for {self.no_improvement_counter} episodes")
                self.logger.info(f"   Best performance: {self.best_position_error:.4f}m at episode {self.best_episode}")
                self.logger.info(f"   Threshold: {self.early_stopping_threshold}m")
                print(f"\n🛑 Early stopping triggered - no improvement for {self.no_improvement_counter} episodes")
                break
            
            # Periodic logging with episode duration information
            if episode % self.config['log_interval'] == 0:
                elapsed_time = time.time() - self.training_start_time
                
                # Get episode duration info
                episode_duration_steps = self.episode_durations[-1] if self.episode_durations else 0
                completion_rate = self.episode_completion_rates[-1] if self.episode_completion_rates else 0.0
                duration_sec = episode_duration_steps * self.config['dt']
                completion_pct = completion_rate * 100
                
                log_msg = f"{episode:7d} | {episode_reward:7.2f} | {avg_cost:8.3f} | {pos_error:9.4f} | {duration_sec:5.1f}s ({completion_pct:4.1f}%) | {elapsed_time:6.1f}s"
                self.logger.info(log_msg)
        
        # Training completed - log final analysis
        total_time = time.time() - self.training_start_time
        self.log_final_analysis(total_time)
        
        # Save best episode replay data for visualization
        if self.session_best_episode_num != -1:
            self.save_best_episode_replay_data()
        
        # Plot results
        self.plot_training_results()
    
    def log_final_analysis(self, total_time):
        """
        Log comprehensive final training analysis
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING COMPLETED - FINAL ANALYSIS")
        self.logger.info("="*80)
        
        # Basic statistics
        self.logger.info(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"Final episode reward: {self.episode_rewards[-1]:.2f}")
        self.logger.info(f"Final position error: {self.episode_position_errors[-1]:.4f}m")
        
        # Episode completion statistics
        if self.episode_completion_rates:
            avg_completion = np.mean(self.episode_completion_rates) * 100
            completed_episodes = len([rate for rate in self.episode_completion_rates if rate >= 0.9])
            completion_success_rate = completed_episodes / len(self.episode_completion_rates) * 100
            
            self.logger.info(f"\nEPISODE COMPLETION STATISTICS:")
            self.logger.info(f"Average completion rate: {avg_completion:.1f}%")
            self.logger.info(f"Episodes completed (≥90%): {completed_episodes}/{len(self.episode_completion_rates)} ({completion_success_rate:.1f}%)")
            
            if self.best_completed_episode != -1:
                self.logger.info(f"Best completed episode: #{self.best_completed_episode} with error {self.best_completed_error:.4f}m")
            else:
                self.logger.info("No episodes achieved ≥90% completion")
        
        # Best episode analysis (smart criteria)
        self.logger.info(f"\nBEST EPISODE ANALYSIS:")
        self.logger.info(f"Best overall episode (smart criteria): #{self.best_episode}")
        self.logger.info(f"  Position error: {self.best_position_error:.4f}m")
        self.logger.info(f"  Composite score: {self.best_episode_score:.2f}")
        
        # Convergence analysis
        total_convergent = len(self.convergence_episodes)
        convergence_rate = total_convergent / len(self.episode_position_errors)
        self.logger.info(f"Total convergent episodes (< 1.0m): {total_convergent}/{len(self.episode_position_errors)} ({convergence_rate:.2%})")
        
        if self.stable_performance_start != -1:
            stable_duration = len(self.episode_position_errors) - self.stable_performance_start
            self.logger.info(f"Stable performance achieved: Yes (from episode {self.stable_performance_start}, duration: {stable_duration} episodes)")
        else:
            self.logger.info("Stable performance achieved: No")
        
        # Performance distribution analysis
        all_errors = np.array(self.episode_position_errors)
        self.logger.info(f"Position error statistics:")
        self.logger.info(f"  Mean: {np.mean(all_errors):.4f}m")
        self.logger.info(f"  Std:  {np.std(all_errors):.4f}m")
        self.logger.info(f"  Min:  {np.min(all_errors):.4f}m")
        self.logger.info(f"  Max:  {np.max(all_errors):.4f}m")
        self.logger.info(f"  Median: {np.median(all_errors):.4f}m")
        
        # Performance thresholds
        excellent = len(all_errors[all_errors < 0.5])
        good = len(all_errors[(all_errors >= 0.5) & (all_errors < 1.0)])
        acceptable = len(all_errors[(all_errors >= 1.0) & (all_errors < 1.5)])
        poor = len(all_errors[all_errors >= 1.5])
        
        self.logger.info(f"Performance distribution:")
        self.logger.info(f"  Excellent (< 0.5m): {excellent} episodes ({excellent/len(all_errors):.1%})")
        self.logger.info(f"  Good (0.5-1.0m):   {good} episodes ({good/len(all_errors):.1%})")
        self.logger.info(f"  Acceptable (1.0-1.5m): {acceptable} episodes ({acceptable/len(all_errors):.1%})")
        self.logger.info(f"  Poor (> 1.5m):     {poor} episodes ({poor/len(all_errors):.1%})")
        
        # Split architecture effectiveness analysis
        if self.config['split']:
            self.logger.info("\nSPLIT ARCHITECTURE ANALYSIS:")
            self.logger.info("-" * 40)
            
            # Analyze learning phases
            if len(all_errors) >= 400:
                warmup_errors = all_errors[:400]
                coordination_errors = all_errors[400:800] if len(all_errors) > 800 else all_errors[400:]
                final_errors = all_errors[-200:] if len(all_errors) >= 200 else all_errors
                
                self.logger.info(f"Warmup phase (0-400): {np.mean(warmup_errors):.4f} ± {np.std(warmup_errors):.4f}m")
                self.logger.info(f"Coordination phase (400-800): {np.mean(coordination_errors):.4f} ± {np.std(coordination_errors):.4f}m")
                self.logger.info(f"Final phase (last 200): {np.mean(final_errors):.4f} ± {np.std(final_errors):.4f}m")
                
                # Learning improvement
                improvement = np.mean(warmup_errors) - np.mean(final_errors)
                self.logger.info(f"Overall improvement: {improvement:.4f}m ({improvement/np.mean(warmup_errors):.1%})")
        
        # Training configuration summary
        self.logger.info("\nTRAINING CONFIGURATION EFFECTIVENESS:")
        self.logger.info("-" * 40)
        self.logger.info(f"Episodes needed for first convergence: {self.convergence_episodes[0] if self.convergence_episodes else 'N/A'}")
        self.logger.info(f"Excitation phase ended at episode: {int(self.config['excitation_steps'] / (self.config['episode_length'] / self.config['dt']))}")
        
        # Save detailed metrics to JSON
        self.save_training_metrics_json()
        
        # Final best model confirmation - check if we actually have a best episode
        # If we have a valid best episode, we should have a best model saved
        if self.best_episode != -1 and self.best_position_error != float('inf'):
            self.best_model_saved = True  # Ensure flag is set if we have valid best data
        
        if self.best_model_saved:
            self.logger.info(f"\nBEST MODEL CONFIRMED:")
            self.logger.info(f"  Episode: {self.best_episode}")
            self.logger.info(f"  Position Error: {self.best_position_error:.4f}m")
            self.logger.info(f"  Model files: dhp_cf2x_best_*")
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
        metrics_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/training_logs"
        json_filename = f"{metrics_dir}/dhp_split_metrics_{timestamp}.json"
        
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
                "costs": {
                    "mean": float(np.mean(self.episode_costs)) if self.episode_costs else None,
                    "std": float(np.std(self.episode_costs)) if self.episode_costs else None,
                    "min": float(np.min(self.episode_costs)) if self.episode_costs else None,
                    "max": float(np.max(self.episode_costs)) if self.episode_costs else None
                }
            },
            "time_series_data": {
                "episode_position_errors": [float(x) for x in self.episode_position_errors],
                "episode_rewards": [float(x) for x in self.episode_rewards],
                "episode_costs": [float(x) for x in self.episode_costs],
                "convergent_episodes": [int(x) for x in self.convergence_episodes]
            },
            "performance_distribution": {
                "excellent_episodes": len([x for x in self.episode_position_errors if x < 0.5]),
                "good_episodes": len([x for x in self.episode_position_errors if 0.5 <= x < 1.0]),
                "acceptable_episodes": len([x for x in self.episode_position_errors if 1.0 <= x < 1.5]),
                "poor_episodes": len([x for x in self.episode_position_errors if x >= 1.5])
            }
        }
        
        with open(json_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved to: {json_filename}")
        return json_filename
             
    def demonstrate_policy(self, gui=True, record=True, episode_length=8.0, real_time=True):
        """
        Demonstrate the trained policy with visualization
        """
        print("\n" + "="*50)
        print("DEMONSTRATING TRAINED DHP POLICY")
        print("="*50)
        
        # Print demonstration mode information
        if self.config.get('use_trajectory', False):
            print(f"🎪 TRAJECTORY DEMONSTRATION MODE")
            print(f"   Trajectory Type: {self.config.get('trajectory_type', 'spiral')}")
            print(f"   Spiral Center: {self.config['target_pos']}")
            print(f"   The drone will demonstrate trajectory following as trained")
        else:
            print(f"📍 FIXED TARGET DEMONSTRATION MODE")
            print(f"   Target Position: {self.config['target_pos']}")
            print(f"   The drone will demonstrate hovering as trained")
        
        # Load the best model for demonstration
        self.load_best_model()
        
        # Create demonstration environment with GUI - MATCH TRAINING SETTINGS
        demo_env = CF2X_FastStates_HoverAviary(
            target_pos=np.array(self.config['target_pos']),
            gui=gui,
            record=record,
            use_trajectory=self.config.get('use_trajectory', False),  # Match training trajectory setting
            trajectory_type=self.config.get('trajectory_type', 'spiral')  # Match training trajectory type
        )
        
        # Reset environment
        state, info = demo_env.reset()
        reference = info['reference']
        
        max_steps = int(episode_length / self.config['dt'])
        demo_states = []
        demo_actions = []
        demo_costs = []
        demo_times = []
        
        print(f"Running demonstration for {episode_length} seconds...")
        if real_time:
            print("Running in real-time mode - you should see the drone flying smoothly")
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
            action_clipped = np.clip(action_flat, -1.0, 1.0)
            
            # Execute action
            next_state, reward, terminated, truncated, info = demo_env.step(action_clipped)
            
            # Store demo data (store raw states for plotting)
            demo_states.append(state.copy())
            demo_actions.append(action_clipped.copy())
            demo_costs.append(info['dhp_cost'])
            demo_times.append(step * self.config['dt'])
            
            # Update for next step
            state = next_state
            reference = info['reference']
            
            # Real-time synchronization for smooth visualization
            if real_time and gui:
                # Calculate how much time should have passed
                expected_time = step * self.config['dt']
                elapsed_real_time = time.time() - start_real_time
                
                # Sleep if we're running too fast
                if elapsed_real_time < expected_time:
                    time.sleep(expected_time - elapsed_real_time)
            
            # Print progress
            if step % 30 == 0:  # Every second at 30Hz
                print(f"Time: {step * self.config['dt']:.1f}s, Position Error: {info['position_error']:.3f}m, Cost: {info['dhp_cost']:.3f}")
            
            # Check termination
            if terminated or truncated:
                break
        
        demo_env.close()
        
        # Print final statistics
        final_pos_error = info['position_error']
        avg_cost = np.mean(demo_costs)
        print(f"\nDemonstration completed!")
        print(f"Final position error: {final_pos_error:.4f} m")
        print(f"Average cost: {avg_cost:.3f}")
        
        if record:
            print("Video recording saved in the environment's output directory")
        
        return demo_states, demo_actions, demo_costs, demo_times
    
    def load_best_model(self):
        """
        Load the best saved model using DHP's built-in load method
        """
        checkpoint_dir = f"/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/trained_models"
        best_model_path = f"{checkpoint_dir}/dhp_cf2x_best"
        
        # Check if the actual files exist (non-numbered format)
        pkl_file = f"{best_model_path}.pkl"
        meta_file = f"{best_model_path}.meta"
        index_file = f"{best_model_path}.index"
        data_file = f"{best_model_path}.data-00000-of-00001"
        
        # Check if all required files exist
        if (os.path.exists(pkl_file) and 
            os.path.exists(meta_file) and
            os.path.exists(index_file) and
            os.path.exists(data_file)):
            
            print("Loading best saved model...")
            try:
                from agents.dhp import Agent as DHP_Agent
                self.agent = DHP_Agent(load_path=best_model_path)
                print("✅ Best model loaded successfully!")
                
                # Verify the loaded model can produce actions  
                # DHP expects 3D inputs: (batch_size, sequence_length, 1) with features in sequence dimension
                test_state = np.zeros(8).reshape(1, -1, 1)  # Shape: (1, 8, 1) - 8 state features
                test_ref = np.zeros(8).reshape(1, -1, 1)    # Shape: (1, 8, 1) - 8 reference values
                test_action = self.agent.predict_actor(test_state, test_ref)
                print(f"✅ Model verification: action shape {test_action.shape}")
                return True
            except Exception as e:
                print(f"❌ Failed to load best model: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ Required model files not found:")
            print(f"   pkl: {os.path.exists(pkl_file)}")
            print(f"   meta: {os.path.exists(meta_file)}")
            print(f"   index: {os.path.exists(index_file)}")
            print(f"   data: {os.path.exists(data_file)}")
        
        print("No saved model found, using current model...")
        return False
    
    def save_best_episode_replay_data(self):
        """
        Save the best episode's action sequence and initial conditions for exact replay
        """
        if self.session_best_episode_num == -1 or len(self.session_best_actions_sequence) == 0:
            print("No best episode data to save for replay")
            return None
            
        # Create replay data directory
        replay_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/best_episode_replays"
        os.makedirs(replay_dir, exist_ok=True)
        
        # Create unique filename with timestamp and episode info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_filename = f"{replay_dir}/best_episode_replay_ep{self.session_best_episode_num}_error{self.session_best_error:.4f}_{timestamp}.pkl"
        
        # Prepare replay data with full precision
        replay_data = {
            'episode_number': self.session_best_episode_num,
            'position_error': self.session_best_error,
            'initial_conditions': self.session_best_initial_conditions.copy(),
            'action_sequence': [action.copy() for action in self.session_best_actions_sequence],  # Full precision actions
            'config': {
                'dt': self.config['dt'],
                'episode_length': self.config['episode_length'],
                'ctrl_freq': 1.0 / self.config['dt'],  # Control frequency (typically 30 Hz)
                'target_pos': self.config['target_pos'],
                'use_trajectory': self.config.get('use_trajectory', False),
                'trajectory_type': self.config.get('trajectory_type', 'spiral'),
                'normalize_states': self.config['normalize_states']
            },
            'metadata': {
                'timestamp': timestamp,
                'total_steps': len(self.session_best_actions_sequence),
                'episode_duration': len(self.session_best_actions_sequence) * self.config['dt'],
                'ctrl_frequency_hz': 1.0 / self.config['dt']
            }
        }
        
        try:
            with open(replay_filename, 'wb') as f:
                pickle.dump(replay_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.session_best_replay_file = replay_filename
            self.logger.info(f"Best episode replay data saved: {replay_filename}")
            self.logger.info(f"  Episode: {self.session_best_episode_num}, Error: {self.session_best_error:.4f}m")
            self.logger.info(f"  Action sequence: {len(self.session_best_actions_sequence)} steps")
            self.logger.info(f"  Duration: {len(self.session_best_actions_sequence) * self.config['dt']:.2f} seconds")
            self.logger.info(f"  Control frequency: {1.0 / self.config['dt']:.1f} Hz")
            
            print(f"✅ Best episode replay data saved!")
            print(f"   File: {replay_filename}")
            print(f"   Episode {self.session_best_episode_num}, Error: {self.session_best_error:.4f}m")
            print(f"   {len(self.session_best_actions_sequence)} actions, {len(self.session_best_actions_sequence) * self.config['dt']:.2f}s duration")
            
            return replay_filename
            
        except Exception as e:
            self.logger.error(f"Failed to save best episode replay data: {str(e)}")
            print(f"❌ Failed to save replay data: {str(e)}")
            return None
    
    def replay_best_episode(self, replay_file=None, gui=True, record=True, real_time=True):
        """
        Replay the best episode using saved action sequence and initial conditions
        with real-time synchronization to control frequency
        
        Args:
            replay_file: Path to replay data file (if None, uses self.session_best_replay_file)
            gui: Whether to show GUI
            record: Whether to record video
            real_time: Whether to synchronize to real-time based on control frequency
        """
        print("\n" + "="*50)
        print("REPLAYING BEST EPISODE")
        print("="*50)
        
        # Determine replay file
        if replay_file is None:
            if self.session_best_replay_file is None:
                # Try to save current session data if available
                self.save_best_episode_replay_data()
                replay_file = self.session_best_replay_file
            else:
                replay_file = self.session_best_replay_file
        
        if replay_file is None:
            print("❌ No replay data available")
            return False
        
        # Load replay data
        try:
            with open(replay_file, 'rb') as f:
                replay_data = pickle.load(f)
            print(f"✅ Loaded replay data from: {replay_file}")
        except Exception as e:
            print(f"❌ Failed to load replay data: {str(e)}")
            return False
        
        # Extract replay information
        episode_num = replay_data['episode_number']
        position_error = replay_data['position_error']
        initial_conditions = replay_data['initial_conditions']
        action_sequence = replay_data['action_sequence']
        config = replay_data['config']
        metadata = replay_data['metadata']
        
        print(f"📺 BEST EPISODE REPLAY: Episode {episode_num}")
        print(f"   Original Error: {position_error:.4f}m")
        print(f"   Duration: {metadata['episode_duration']:.2f} seconds")
        print(f"   Control Frequency: {metadata['ctrl_frequency_hz']:.1f} Hz")
        print(f"   Total Actions: {metadata['total_steps']}")
        
        if config.get('use_trajectory', False):
            print(f"   Mode: Trajectory Following ({config['trajectory_type']})")
        else:
            print(f"   Mode: Fixed Target {config['target_pos']}")
        
        # Create environment for replay (match original training settings)
        replay_env = CF2X_FastStates_HoverAviary(
            target_pos=np.array(config['target_pos']),
            gui=gui,
            record=record,
            use_trajectory=config.get('use_trajectory', False),
            trajectory_type=config.get('trajectory_type', 'spiral')
        )
        
        # Reset environment and set exact initial conditions
        state, info = replay_env.reset()
        
        # Set the drone to exact initial conditions from the best episode
        try:
            # Get the drone's current state vector and modify it
            full_state = replay_env._getDroneStateVector(0)
            
            # Set position [x, y, z]
            full_state[0:3] = initial_conditions['position']
            # Set orientation [quaternion - we need to convert from RPY]
            import pybullet as p
            quat = p.getQuaternionFromEuler(initial_conditions['orientation'])
            full_state[3:7] = quat  # [qx, qy, qz, qw]
            # Set RPY for compatibility
            full_state[7:10] = initial_conditions['orientation']
            # Set linear velocity [vx, vy, vz]
            full_state[10:13] = initial_conditions['velocity']
            # Set angular velocity [wx, wy, wz]
            full_state[13:16] = initial_conditions['angular_velocity']
            
            # Apply the exact initial conditions to the simulation
            p.resetBasePositionAndOrientation(replay_env.DRONE_IDS[0], 
                                            initial_conditions['position'], 
                                            quat)
            p.resetBaseVelocity(replay_env.DRONE_IDS[0], 
                              initial_conditions['velocity'], 
                              initial_conditions['angular_velocity'])
            
            print(f"✅ Set exact initial conditions:")
            print(f"   Position: [{initial_conditions['position'][0]:.3f}, {initial_conditions['position'][1]:.3f}, {initial_conditions['position'][2]:.3f}]")
            print(f"   Orientation: [{initial_conditions['orientation'][0]:.3f}, {initial_conditions['orientation'][1]:.3f}, {initial_conditions['orientation'][2]:.3f}]")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not set exact initial conditions: {str(e)}")
            print("   Continuing with default reset conditions...")
        
        # Get the updated state after setting initial conditions
        state, info = replay_env.reset()
        
        print(f"\n🎬 Starting replay...")
        if real_time:
            print(f"   Real-time mode: {metadata['ctrl_frequency_hz']:.1f} Hz synchronization")
        else:
            print("   Fast mode: No timing synchronization")
        
        # Replay statistics
        replay_states = []
        replay_position_errors = []
        replay_costs = []
        
        # Track timing for real-time synchronization
        start_real_time = time.time()
        dt = config['dt']
        
        # Execute the recorded action sequence
        for step, recorded_action in enumerate(action_sequence):
            # Use the exact recorded action (no agent inference needed)
            action_clipped = np.clip(recorded_action, -1.0, 1.0)
            
            # Execute the recorded action
            next_state, reward, terminated, truncated, info = replay_env.step(action_clipped)
            
            # Collect replay statistics
            replay_states.append(state.copy())
            replay_position_errors.append(info['position_error'])
            replay_costs.append(info['dhp_cost'])
            
            # Update state
            state = next_state
            
            # Real-time synchronization based on control frequency
            if real_time and gui:
                expected_time = (step + 1) * dt
                elapsed_real_time = time.time() - start_real_time
                
                # Sleep if we're running too fast to match control frequency
                if elapsed_real_time < expected_time:
                    time.sleep(expected_time - elapsed_real_time)
            
            # Progress reporting
            if step % int(metadata['ctrl_frequency_hz']) == 0:  # Every second
                print(f"   Time: {step * dt:.1f}s, Pos Error: {info['position_error']:.3f}m, Cost: {info['dhp_cost']:.3f}")
            
            # Check for early termination
            if terminated or truncated:
                print(f"   Episode terminated early at step {step}")
                break
        
        # Clean up
        replay_env.close()
        
        # Calculate replay statistics
        final_pos_error = replay_position_errors[-1] if replay_position_errors else float('inf')
        avg_cost = np.mean(replay_costs) if replay_costs else float('inf')
        max_pos_error = np.max(replay_position_errors) if replay_position_errors else float('inf')
        
        print(f"\n🏁 REPLAY COMPLETED!")
        print(f"   Original Episode Error: {position_error:.4f}m")
        print(f"   Replay Final Error: {final_pos_error:.4f}m")
        print(f"   Replay Average Cost: {avg_cost:.3f}")
        print(f"   Replay Max Error: {max_pos_error:.4f}m")
        print(f"   Completed Steps: {len(replay_position_errors)}/{len(action_sequence)}")
        
        if record:
            print(f"   📹 Video recording saved in environment recordings directory")
        
        return {
            'original_error': position_error,
            'replay_final_error': final_pos_error,
            'replay_avg_cost': avg_cost,
            'replay_max_error': max_pos_error,
            'completed_steps': len(replay_position_errors),
            'total_steps': len(action_sequence)
        }

    def plot_training_results(self):
        """
        Plot training metrics in dhp_main.py style with data from the BEST episode
        Enhanced with 3D trajectory visualization and seaborn styling
        """
        # Import seaborn for dhp_main.py style plots
        try:
            import seaborn as sns
            from mpl_toolkits.mplot3d import Axes3D
            sns.set_style("whitegrid")
            sns.set_palette("husl") 
            sns.set_context("notebook", font_scale=1.1)
            seaborn_available = True
            print("Using seaborn styling for enhanced plots")
        except ImportError:
            print("Seaborn not available - using matplotlib defaults")
            seaborn_available = False
        
        print("\n[PLOTTING] Generating DHP training analysis plots...")
        
        if len(self.best_episode_states) == 0:
            print("No best episode data available for plotting")
            print("This means no episode completed successfully during training")
            return
        
        print(f"Using BEST episode data from episode {self.best_episode}")
        print(f"Best episode position error: {self.best_position_error:.4f}m")
        print(f"Data points: {len(self.best_episode_states)}")
            
        # Convert best episode data to arrays for easier manipulation
        states = np.array(self.best_episode_states)
        references = np.array(self.best_episode_references) 
        actions = np.array(self.best_episode_actions)
        costs = np.array(self.best_episode_costs)
        model_errors = np.array(self.best_episode_model_errors)
        t = np.array(self.best_episode_time)
        
        # Extract state components (fast states: [z, roll, pitch, yaw, vz, wx, wy, wz])
        z = states[:, 0]           # altitude
        roll = states[:, 1]        # roll angle
        pitch = states[:, 2]       # pitch angle
        yaw = states[:, 3]         # yaw angle
        vz = states[:, 4]          # vertical velocity
        wx = states[:, 5]          # roll rate
        wy = states[:, 6]          # pitch rate  
        wz = states[:, 7]          # yaw rate
        
        # Extract references
        z_ref = references[:, 0]
        roll_ref = references[:, 1]
        pitch_ref = references[:, 2]
        yaw_ref = references[:, 3]
        vz_ref = references[:, 4]
        wx_ref = references[:, 5]
        wy_ref = references[:, 6]
        wz_ref = references[:, 7]
        
        # Extract actions (motor commands)
        motor1 = actions[:, 0]
        motor2 = actions[:, 1]
        motor3 = actions[:, 2]
        motor4 = actions[:, 3]
        
        # Set seaborn style and figure parameters (matching dhp_main.py)
        if seaborn_available:
            try:
                colors = sns.color_palette("husl", 10)  # More colors for multiple plots
            except:
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        else:
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
        plt.rcParams['figure.figsize'] = 25/2.54, 50/2.54
        
        ### Figure 0: Training Progress - Episode Rewards and Costs ###
        fig0 = plt.figure()
        episodes = np.arange(len(self.episode_rewards))
        
        ax1 = fig0.add_subplot(2,1,1)
        ax1.plot(episodes, self.episode_rewards, color=colors[0], linewidth=2, label='Episode Rewards (DHP)')
        ax1.set_ylabel(r'Episode Reward')
        ax1.set_title('DHP Training Progress - Rewards and Costs over Episodes')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2,1,2, sharex=ax1)
        ax2.plot(episodes, self.episode_costs, color=colors[1], linewidth=2, label='Episode Average Cost (DHP)')
        ax2.set_ylabel(r'Episode Average Cost')
        ax2.set_xlabel(r'Episode Number')
        ax2.set_yscale('log')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        ### Figure 1: Primary Control - Vertical Motion ###
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(4,1,1)
        ax1.plot(t, z, 'b-', label=r'$z$ (DHP Best Episode)')  
        ax1.plot(t, z_ref, 'r--', label=r'$z_{ref}$')  
        ax1.set_ylabel(r'Altitude $[m]$')
        ax1.set_title(f'DHP Best Episode Performance (Ep {self.best_episode}, Error: {self.best_position_error:.3f}m)')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        ax2 = plt.subplot(4,1,2, sharex=ax1)
        ax2.plot(t, vz, 'b-', label=r'$v_z$ (DHP Best Episode)')
        ax2.plot(t, vz_ref, 'r--', label=r'$v_{z_{ref}}$')  
        ax2.set_ylabel(r'Vertical Velocity $[m/s]$')
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
        
        ### Figure 2: Attitude Control - Roll ###
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(4,1,1)
        ax1.plot(t, roll*180/np.pi, 'b-', label=r'$\phi$ (DHP Best Episode)')
        ax1.plot(t, roll_ref*180/np.pi, 'r--', label=r'$\phi_{ref}$')  
        ax1.set_ylabel(r'Roll Angle $[deg]$')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        ax2 = plt.subplot(4,1,2, sharex=ax1)
        ax2.plot(t, wx*180/np.pi, 'b-', label=r'$p$ (DHP Best Episode)')
        ax2.plot(t, wx_ref*180/np.pi, 'r--', label=r'$p_{ref}$')   
        ax2.set_ylabel(r'Roll Rate $[deg/s]$')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        ax3 = plt.subplot(4,1,3, sharex=ax1)
        ax3.plot(t, motor2, 'g-', label=r'Motor 2 (DHP Best Episode)')
        ax3.plot(t, motor4, 'orange', label=r'Motor 4 (DHP Best Episode)')  
        ax3.set_ylabel(r'Motor Commands $[-]$')
        ax3.legend(loc='upper right')
        ax3.grid(True)

        ax4 = plt.subplot(4,1,4, sharex=ax1)
        ax4.plot(t, pitch*180/np.pi, 'b-', label=r'$\theta$ (DHP Best Episode)')
        ax4.plot(t, pitch_ref*180/np.pi, 'r--', label=r'$\theta_{ref}$')
        ax4.set_xlabel(r'$t [s]$')
        ax4.set_ylabel(r'Pitch Angle $[deg]$')
        ax4.legend(loc='upper right')
        ax4.grid(True)

        ### Figure 3: Attitude Control - Pitch & Yaw ###
        fig3 = plt.figure()
        ax1 = fig3.add_subplot(4,1,1)
        ax1.plot(t, wy*180/np.pi, 'b-', label=r'$q$ (DHP Best Episode)')
        ax1.plot(t, wy_ref*180/np.pi, 'r--', label=r'$q_{ref}$')   
        ax1.set_ylabel(r'Pitch Rate $[deg/s]$')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        ax2 = plt.subplot(4,1,2, sharex=ax1)
        ax2.plot(t, motor1, 'b-', label=r'Motor 1 (DHP Best Episode)')
        ax2.plot(t, motor3, 'r-', label=r'Motor 3 (DHP Best Episode)')  
        ax2.set_ylabel(r'Motor Commands $[-]$')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        ax3 = plt.subplot(4,1,3, sharex=ax1)
        ax3.plot(t, yaw*180/np.pi, 'b-', label=r'$\psi$ (DHP Best Episode)')
        ax3.plot(t, yaw_ref*180/np.pi, 'r--', label=r'$\psi_{ref}$')  
        ax3.set_ylabel(r'Yaw Angle $[deg]$')
        ax3.legend(loc='upper right')
        ax3.grid(True)

        ax4 = plt.subplot(4,1,4, sharex=ax1)
        ax4.plot(t, wz*180/np.pi, 'b-', label=r'$r$ (DHP Best Episode)')
        ax4.plot(t, wz_ref*180/np.pi, 'r--', label=r'$r_{ref}$')
        ax4.set_xlabel(r'$t [s]$')
        ax4.set_ylabel(r'Yaw Rate $[deg/s]$')
        ax4.legend(loc='upper right')
        ax4.grid(True)

        ### Figure 4: XYZ Position Control (not part of fast states but important for analysis) ###
        if len(self.best_episode_x_positions) > 0:
            fig4 = plt.figure(figsize=(14, 12))
            x_pos = np.array(self.best_episode_x_positions)
            y_pos = np.array(self.best_episode_y_positions)
            x_ref = np.array(self.best_episode_x_references)
            y_ref = np.array(self.best_episode_y_references)
            # Z position from fast states
            z_pos = states[:, 0]  # Z is first element of fast states
            z_ref = references[:, 0]  # Z reference
            
            ax1 = fig4.add_subplot(5,1,1)
            ax1.plot(t, x_pos, color=colors[0], linewidth=2, label=r'$x$ (DHP Best Episode)')
            ax1.plot(t, x_ref, '--', color=colors[1], linewidth=2, label=r'$x_{ref}$')  
            ax1.set_ylabel(r'X Position $[m]$')
            ax1.set_title(f'XYZ Position Control - Best Episode {self.best_episode} (Error: {self.best_position_error:.3f}m)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            ax2 = plt.subplot(5,1,2, sharex=ax1)
            ax2.plot(t, y_pos, color=colors[2], linewidth=2, label=r'$y$ (DHP Best Episode)')
            ax2.plot(t, y_ref, '--', color=colors[3], linewidth=2, label=r'$y_{ref}$')  
            ax2.set_ylabel(r'Y Position $[m]$')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            ax3 = plt.subplot(5,1,3, sharex=ax1)
            ax3.plot(t, z_pos, color=colors[4], linewidth=2, label=r'$z$ (DHP Best Episode)')
            ax3.plot(t, z_ref, '--', color=colors[5], linewidth=2, label=r'$z_{ref}$')  
            ax3.set_ylabel(r'Z Position $[m]$')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)

            ax4 = plt.subplot(5,1,4, sharex=ax1)
            # Calculate XYZ position errors
            x_error = x_pos - x_ref
            y_error = y_pos - y_ref
            z_error = z_pos - z_ref
            xyz_error = np.sqrt(x_error**2 + y_error**2 + z_error**2)
            
            ax4.plot(t, x_error, color=colors[0], linewidth=2, label=r'$e_x$ (DHP Best Episode)')
            ax4.plot(t, y_error, color=colors[2], linewidth=2, label=r'$e_y$ (DHP Best Episode)')
            ax4.plot(t, z_error, color=colors[4], linewidth=2, label=r'$e_z$ (DHP Best Episode)')
            ax4.plot(t, xyz_error, color='red', linewidth=3, label=r'$\|e_{xyz}\|$ (DHP Best Episode)')
            ax4.set_ylabel(r'Position Errors $[m]$')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)

            ax5 = plt.subplot(5,1,5)
            # 2D trajectory plot (XY plane)
            ax5.plot(x_pos, y_pos, color=colors[0], linewidth=3, label='Actual XY Trajectory (DHP Best)')
            ax5.plot(x_ref, y_ref, '--', color=colors[1], linewidth=3, label='Reference XY Trajectory')
            ax5.plot(x_pos[0], y_pos[0], 'o', color='green', markersize=10, label='Start')
            ax5.plot(x_pos[-1], y_pos[-1], 'o', color='red', markersize=10, label='End')
            ax5.set_xlabel(r'X Position $[m]$')
            ax5.set_ylabel(r'Y Position $[m]$')
            ax5.set_title('2D Trajectory View (XY Plane)')
            ax5.legend(loc='upper right')
            ax5.grid(True, alpha=0.3)
            ax5.axis('equal')  # Equal aspect ratio for true trajectory shape

            # Create separate 3D trajectory plot
            fig4_3d = plt.figure(figsize=(12, 10))
            ax_3d = fig4_3d.add_subplot(111, projection='3d')

            # 3D trajectory plot
            ax_3d.plot(x_pos, y_pos, z_pos, color=colors[0], linewidth=3, label='Actual 3D Trajectory (DHP Best)')
            ax_3d.plot(x_ref, y_ref, z_ref, '--', color=colors[1], linewidth=3, label='Reference 3D Trajectory')

            # Mark start and end points
            ax_3d.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=100, label='Start', marker='o')
            ax_3d.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=100, label='End', marker='s')

            # Add reference point (target)
            # ax_3d.scatter(x_ref[0], y_ref[0], z_ref[0], color='blue', s=150, label='Target', marker='^', alpha=0.7)

            ax_3d.set_xlabel(r'X Position $[m]$', fontsize=12)
            ax_3d.set_ylabel(r'Y Position $[m]$', fontsize=12)
            ax_3d.set_zlabel(r'Z Position $[m]$', fontsize=12)
            ax_3d.set_title(f'3D Quadrotor Trajectory - Best Episode {self.best_episode}', fontsize=14, fontweight='bold')
            ax_3d.legend(loc='upper left')

            # Set equal aspect ratio for 3D plot
            max_range = max(
                np.max(x_pos) - np.min(x_pos),
                np.max(y_pos) - np.min(y_pos),
                np.max(z_pos) - np.min(z_pos)
            ) / 2.0

            mid_x = (np.max(x_pos) + np.min(x_pos)) * 0.5
            mid_y = (np.max(y_pos) + np.min(y_pos)) * 0.5
            mid_z = (np.max(z_pos) + np.min(z_pos)) * 0.5

            ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
            ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

            # Add grid
            ax_3d.grid(True, alpha=0.3)

        ### Figure 5: Neural Network Analysis (Best Episode) ###
        if len(self.best_episode_actor_grads) > 0:
            fig5 = plt.figure()
            actor_grads = np.array(self.best_episode_actor_grads)
            critic_grads = np.array(self.best_episode_critic_grads)
            
            ax1 = fig5.add_subplot(4,1,1)
            # Plot actor gradients (should be 1D or 2D)
            if actor_grads.ndim == 1:
                ax1.plot(t[:len(actor_grads)], actor_grads, 'b-', label='Actor Grad (Best Episode)')
            elif actor_grads.ndim == 2:
                # Plot norm of gradient vector
                grad_norm = np.linalg.norm(actor_grads, axis=1)
                ax1.plot(t[:len(grad_norm)], grad_norm, 'b-', label='Actor Grad Norm (Best Episode)')
            ax1.set_ylabel(r'Actor Gradients')
            ax1.legend(loc='upper right')
            ax1.grid(True)

            ax2 = plt.subplot(4,1,2, sharex=ax1)
            # Plot critic gradients (should be 1D or 2D)
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

        ### Figure 6: RLS Model Analysis ###
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            fig6 = plt.figure(figsize=(12, 10))
            
            # Overall training RLS variance metrics (all episodes)
            if len(self.detailed_rls_variance) > 0:
                ax1 = fig6.add_subplot(2,2,1)
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
                ax2 = fig6.add_subplot(2,2,2)
                state_names = ['z', 'roll', 'pitch', 'yaw', 'vz', 'wx', 'wy', 'wz']
                for i in range(min(8, rls_var_overall.shape[1])):
                    ax2.plot(t_overall, rls_var_overall[:, i], label=f'{state_names[i]}', alpha=0.7)
                ax2.set_ylabel(r'RLS Variance by State')
                ax2.set_xlabel(r'Training Time [s]')
                ax2.set_yscale('log')
                ax2.legend(loc='upper right', fontsize=8)
                ax2.grid(True)
                ax2.set_title('RLS Variance per State (Whole Training)')
            
            # Best episode RLS predictions vs ground truth
            if len(self.best_episode_rls_predictions) > 0:
                ax3 = fig6.add_subplot(2,2,3)
                predictions = np.array(self.best_episode_rls_predictions)
                ground_truth = np.array(self.best_episode_rls_ground_truth)
                
                # Ensure consistent shapes by squeezing any singleton dimensions
                predictions = np.squeeze(predictions)
                ground_truth = np.squeeze(ground_truth)
                
                # Ensure both are 2D (time_steps, state_size)
                if predictions.ndim == 1:
                    predictions = predictions.reshape(1, -1)
                if ground_truth.ndim == 1:
                    ground_truth = ground_truth.reshape(1, -1)
                
                t_best = t[:len(predictions)]
                
                # Plot prediction vs ground truth for key states (z, roll, pitch)
                ax3.plot(t_best, ground_truth[:, 2], 'b-', label='True Z', linewidth=2)
                ax3.plot(t_best, predictions[:, 2], 'r--', label='Predicted Z', linewidth=1.5)
                ax3.plot(t_best, ground_truth[:, 3], 'g-', label='True Roll', alpha=0.7)
                ax3.plot(t_best, predictions[:, 3], 'm--', label='Predicted Roll', alpha=0.7)
                ax3.set_ylabel(r'State Values')
                ax3.set_xlabel(r'Time [s]')
                ax3.legend(loc='upper right', fontsize=8)
                ax3.grid(True)
                ax3.set_title(f'RLS Predictions vs Truth (Best Episode {self.best_episode})')
                
                # Prediction error analysis
                ax4 = fig6.add_subplot(2,2,4)
                prediction_errors = np.abs(predictions - ground_truth)
                mean_error = np.mean(prediction_errors, axis=1)
                ax4.plot(t_best, mean_error, 'r-', label='Mean Prediction Error')
                ax4.set_ylabel(r'Prediction Error')
                ax4.set_xlabel(r'Time [s]')
                ax4.set_yscale('log')
                ax4.legend(loc='upper right')
                ax4.grid(True)
                ax4.set_title(f'RLS Model Accuracy (Best Episode {self.best_episode})')
            else:
                # Fallback: Best episode RLS variance if predictions not available
                if len(self.best_episode_rls_variance) > 0:
                    ax3 = fig6.add_subplot(2,2,3)
                    rls_var = np.array(self.best_episode_rls_variance)
                    ax3.plot(t[:len(rls_var)], rls_var, label='RLS Variance (Best Episode)')
                    ax3.set_ylabel(r'RLS Variance')
                    ax3.set_xlabel(r'Time [s]')
                    ax3.set_yscale('log')
                    ax3.legend(loc='upper right')
                    ax3.grid(True)
                    ax3.set_title(f'RLS Variance (Best Episode {self.best_episode})')
                    
                    # Motor command variance
                    ax4 = fig6.add_subplot(2,2,4)
                    motor_commands = np.array([motor1, motor2, motor3, motor4]).T
                    ax4.plot(t, np.var(motor_commands, axis=1), 'b-', label='Motor Variance (Best Episode)')
                    ax4.set_xlabel(r'Time [s]')
                    ax4.set_ylabel(r'Control Variance')
                    ax4.legend(loc='upper right')
                    ax4.grid(True)
                    ax4.set_title(f'Control Smoothness (Best Episode {self.best_episode})')
            
            plt.tight_layout()

        # Align labels and show plots (matching dhp_main.py)
        fig0.align_labels()
        fig1.align_labels()
        fig2.align_labels() 
        fig3.align_labels()
        if len(self.best_episode_x_positions) > 0:
            fig4.align_labels()
            fig4_3d.tight_layout()
        if len(self.best_episode_actor_grads) > 0:
            fig5.align_labels()
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            fig6.align_labels()
        
        # Save figures
        fig0.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_training_progress.png', dpi=150, bbox_inches='tight')
        fig1.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_vertical_control.png', dpi=150, bbox_inches='tight')
        fig2.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_roll_control.png', dpi=150, bbox_inches='tight')
        fig3.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_pitch_yaw_control.png', dpi=150, bbox_inches='tight')
        if len(self.best_episode_x_positions) > 0:
            fig4.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_xyz_position_control.png', dpi=150, bbox_inches='tight')
            fig4_3d.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_3d_trajectory.png', dpi=150, bbox_inches='tight')
        if len(self.best_episode_actor_grads) > 0:
            fig5.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_neural_analysis.png', dpi=150, bbox_inches='tight')
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            fig6.savefig('/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/dhp_quadrotor_rls_analysis.png', dpi=150, bbox_inches='tight')
        
        plt.show()
        
        print("DHP BEST EPISODE analysis plots generated and saved!")
        print("Figures saved:")
        print("  - Training Progress: dhp_quadrotor_training_progress.png")
        print(f"  - Vertical Control: dhp_quadrotor_vertical_control.png (Best Episode {self.best_episode})")
        print(f"  - Roll Control: dhp_quadrotor_roll_control.png (Best Episode {self.best_episode})") 
        print(f"  - Pitch/Yaw Control: dhp_quadrotor_pitch_yaw_control.png (Best Episode {self.best_episode})")
        if len(self.best_episode_x_positions) > 0:
            print(f"  - XYZ Position Control: dhp_quadrotor_xyz_position_control.png (Best Episode {self.best_episode})")
            print(f"  - 3D Trajectory: dhp_quadrotor_3d_trajectory.png (Best Episode {self.best_episode})")
        if len(self.best_episode_actor_grads) > 0:
            print(f"  - Neural Network Analysis: dhp_quadrotor_neural_analysis.png (Best Episode {self.best_episode})")
        if len(self.detailed_rls_variance) > 0 or len(self.best_episode_rls_predictions) > 0:
            print(f"  - RLS Analysis: dhp_quadrotor_rls_analysis.png (Training Progress + Best Episode {self.best_episode})")
        print(f"All state vs reference plots show Episode {self.best_episode} (Position Error: {self.best_position_error:.4f}m)")
        print(f"RLS Analysis includes: Training variance progression + Best episode predictions vs ground truth")
        print(f"NEW: XYZ Position tracking, 3D trajectory visualization, and seaborn styling for all plots")

if __name__ == "__main__":
    print("DHP Training for CF2X Quadrotor")
    print("================================")
    
    # Configuration for trajectory following training
    config_override = {
        # Training settings for trajectory following
        'num_episodes': 200,            # Moderate episodes for trajectory learning
        'episode_length': 25.0,         # 25 seconds (20s trajectory + 5s buffer)
        'excitation_steps': 5000,       # Reduced exploration for trajectory training
        'log_interval': 10,
        'save_interval': 50,
        'gui': False,                   # Training without GUI for speed
        'record': False,
        
        # State normalization settings
        'normalize_states': True,       # Enable normalization for better convergence
        'record_best_episodes': True,   # Record best trajectory following episodes
        
        # Split architecture specific settings
        'coordination_warmup_episodes': 200,  # Faster coordination for trajectory
        'coordination_strength': 0.15,       # Moderate coupling strength
        
        # TRAJECTORY SETTINGS - NEW!
        'use_trajectory': True,         # Enable trajectory following
        'trajectory_type': 'spiral',    # 'spiral', 'figure8', 'circle'  
        'target_pos': [0.0, 0.0, 1.0]  # Center of spiral trajectory
    }
    
    # Create trainer with default config, then update with overrides
    trainer = QuadrotorDHPTrainer()
    trainer.config.update(config_override)
    
    # Log the configuration update
    trainer.logger.info("\nTRAJECTORY TRAINING CONFIGURATION:")
    for key, value in config_override.items():
        trainer.logger.info(f"  {key}: {value}")
    
    print("\nTrajectory following training configuration:")
    for key, value in config_override.items():
        print(f"  {key}: {value}")
    
    if config_override.get('use_trajectory', False):
        print(f"\n🎯 TRAJECTORY MODE ENABLED!")
        print(f"   Trajectory type: {config_override.get('trajectory_type', 'spiral')}")
        print(f"   Spiral center: {config_override.get('target_pos', [0, 0, 1])}")
        print(f"   The drone will learn to follow a smooth {config_override.get('trajectory_type', 'spiral')} trajectory")
        print(f"   Trajectory stays within bounds: X,Y ∈ [-4.5, 4.5], Z ∈ [0.2, 4.5]")
    else:
        print("\n📍 FIXED TARGET MODE")
        print(f"   Target position: {config_override.get('target_pos', [0, 0, 1])}")
    
    # Train the DHP agent
    trainer.train()
    
    # After training, replay the best episode with exact actions and initial conditions
    trainer.logger.info("\n" + "="*50)
    trainer.logger.info("STARTING BEST EPISODE REPLAY")
    trainer.logger.info("="*50)

    print("\n" + "="*50)
    print("STARTING BEST EPISODE REPLAY")
    print("="*50)
    
    # Replay the best episode using recorded actions and initial conditions
    replay_result = trainer.replay_best_episode(
        replay_file=None,   # Use the automatically saved replay data
        gui=True,           # Show GUI for visualization
        record=True,        # Record video
        real_time=True      # Synchronize to control frequency (30 Hz)
    )
    
    # Log replay results
    if replay_result:
        if config_override.get('use_trajectory', False):
            print(f"\n🎪 BEST EPISODE TRAJECTORY REPLAY COMPLETED!")
            print(f"   Trajectory Type: {config_override.get('trajectory_type', 'spiral')}")
            print(f"   Replayed exact best episode performance with recorded actions")
        else:
            print(f"\n📍 BEST EPISODE FIXED TARGET REPLAY COMPLETED!")
        
        print(f"   Original Best Error: {replay_result['original_error']:.4f}m")
        print(f"   Replay Final Error: {replay_result['replay_final_error']:.4f}m")
        print(f"   Replay Steps: {replay_result['completed_steps']}/{replay_result['total_steps']}")
        
        trainer.logger.info(f"Best episode replay completed successfully")
        trainer.logger.info(f"Original error: {replay_result['original_error']:.4f}m, Replay error: {replay_result['replay_final_error']:.4f}m")
    else:
        print("❌ Failed to replay best episode")
        trainer.logger.error("Failed to replay best episode")
    
    trainer.logger.info("Training and best episode replay session completed successfully!")
    
    print("\nDHP training and best episode replay completed!")
    print("✅ Exact best episode actions recorded and replayed with real-time synchronization")
    print("📹 Video recording captures the exact best performance achieved during training")
    print("Check the saved plots and video recordings for results.")
    print(f"Detailed training log: {trainer.log_filename}")
    if trainer.session_best_replay_file:
        print(f"Best episode replay data: {trainer.session_best_replay_file}")
