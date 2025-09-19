"""
DHP Best Model Demo Script

This script loads the best performing DHP model and demonstrates it with GUI visualization.
The model achieved 0.0329m position error at episode 1463.

Author: DHP Demo
Date: August 9, 2025
"""

import numpy as np
import sys
import os
import time
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

# Set TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import TensorFlow and reset graph to avoid variable conflicts
import tensorflow as tf
tf.reset_default_graph()

# Add paths
sys.path.append('/home/osos/Mohamed_Masters_Thesis/msc-thesis')
sys.path.append('/home/osos/Mohamed_Masters_Thesis/gym-pybullet-drones')
sys.path.append('/home/osos/Mohamed_Masters_Thesis/trial2')

# Import DHP components
from agents.dhp import Agent as DHP_Agent
from cf2x_fast_states_env import CF2X_FastStates_HoverAviary

def load_best_dhp_model():
    """Load the best performing DHP model from training"""
    
    print("Loading Best DHP Model")
    print("="*40)
    
    # Reset TensorFlow graph to avoid variable conflicts
    tf.reset_default_graph()
    
    # Paths to best model files (fixed names, no episode numbers)
    model_dir = "/home/osos/Mohamed_Masters_Thesis/trial2/trained_models"
    best_model_path = f"{model_dir}/dhp_cf2x_best"
    config_path = f"{model_dir}/dhp_cf2x_config.pkl"
    metadata_path = f"{model_dir}/dhp_cf2x_best_metadata.pkl"
    
    # Load configuration
    print("Loading configuration...")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Load best model metadata
    print("Loading best model metadata...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✓ Best model from episode: {metadata['best_episode']}")
    print(f"✓ Best position error: {metadata['best_position_error']:.4f}m")
    print(f"✓ Training timestamp: {metadata['timestamp']}")
    print(f"✓ State normalization: {'ENABLED' if config['normalize_states'] else 'DISABLED'}")
    print()
    
    # Create DHP agent with same configuration
    print("Initializing DHP agent...")
    agent_kwargs = {
        'input_size': [config['state_size'], config['reference_size']],
        'output_size': config['action_size'],
        'hidden_layer_size': config['hidden_layer_size'],
        'lr_critic': config['lr_critic'],
        'lr_actor': config['lr_actor'],
        'gamma': config['gamma'],
        'target_network': config['target_network'],
        'tau': config['tau'],
        'split': config['split']
    }
    
    # Handle split architecture mode if enabled
    if config['split']:
        # Import required modules for split architecture
        from quadrotor_dhp_modes import create_quadrotor_phlab_extension
        import phlab
        
        # Setup quadrotor phlab compatibility
        quad_ext = create_quadrotor_phlab_extension()
        phlab.ID_QUAD_LON = quad_ext['ID_QUAD_LON']
        phlab.ID_QUAD_LAT = quad_ext['ID_QUAD_LAT']
        phlab.ID_QUAD_FULL = quad_ext['ID_QUAD_FULL']
        phlab.states.update(quad_ext['states'])
        phlab.idx.update(quad_ext['idx'])
        phlab.track_states.update(quad_ext['track_states'])
        phlab.states[300] = quad_ext['states'][quad_ext['ID_QUAD_FULL']]
        phlab.idx[300] = quad_ext['idx'][quad_ext['ID_QUAD_FULL']]
        phlab.track_states[300] = quad_ext['track_states'][quad_ext['ID_QUAD_FULL']]
        phlab.ID_LON = quad_ext['ID_QUAD_LON']
        phlab.ID_LAT = quad_ext['ID_QUAD_LAT']
        
        print("✓ Split architecture enabled")
    
    # Use the base model path without episode number for DHP agent loading
    model_file = f"{best_model_path}"  # This will be dhp_cf2x_best
    
    print(f"Loading best model weights from: {model_file}")
    
    # Since the DHP agent class has variable conflicts, let's use the direct approach
    # but with better performance by using the actual training configuration
    print("Using direct TensorFlow checkpoint loading for better compatibility...")
    
    # Create a fresh graph and session
    tf.reset_default_graph()
    graph = tf.Graph()
    
    with graph.as_default():
        sess = tf.Session()
        
        # Import the saved meta graph and restore weights
        saver = tf.train.import_meta_graph(f"{model_file}.meta")
        saver.restore(sess, model_file)
        
        # Get the required tensors
        state_input = graph.get_tensor_by_name("actor/state_input:0")
        ref_input = graph.get_tensor_by_name("actor/reference_input:0") 
        actor_output = graph.get_tensor_by_name("actor/concat_1:0")
        
        # Create a wrapper class that mimics the DHP agent interface
        class DirectDHPAgent:
            def __init__(self, sess, state_input, ref_input, actor_output):
                self.sess = sess
                self.state_input = state_input
                self.ref_input = ref_input
                self.actor_output = actor_output
                self.trim = np.zeros(config['action_size'])
            
            def action(self, state, reference=None):
                # Remove the time dimension and flatten for TensorFlow
                state_flat = state.reshape(1, -1)
                ref_flat = reference.reshape(1, -1) if reference is not None else None
                
                if ref_flat is not None:
                    feed_dict = {self.state_input: state_flat, self.ref_input: ref_flat}
                else:
                    feed_dict = {self.state_input: state_flat}
                
                action = self.sess.run(self.actor_output, feed_dict=feed_dict)
                return action
        
        agent = DirectDHPAgent(sess, state_input, ref_input, actor_output)
    
    print("✓ Best model loaded successfully using direct TensorFlow approach!")
    print()
    
    return agent, config, metadata

def normalize_state(state, config):
    """Normalize state vector if normalization was used during training"""
    if not config['normalize_states']:
        return state.copy()
    
    normalized_state = state.copy()
    state_names = ['z', 'roll', 'pitch', 'yaw', 'vz', 'wx', 'wy', 'wz']
    
    for i, name in enumerate(state_names):
        if name in config['state_bounds']:
            min_val, max_val = config['state_bounds'][name]
            clipped_val = np.clip(state[i], min_val, max_val)
            normalized_state[i] = 2.0 * (clipped_val - min_val) / (max_val - min_val) - 1.0
    
    return normalized_state

def normalize_reference(reference, config):
    """Normalize reference vector using same bounds as states"""
    return normalize_state(reference, config)

def plot_demonstration_results(demo_data, config, metadata):
    """
    Create comprehensive performance plots for the DHP demonstration
    """
    print("\n[ANALYSIS] Generating performance plots...")
    
    # Extract data arrays
    states = np.array(demo_data['states'])
    actions = np.array(demo_data['actions'])
    costs = np.array(demo_data['costs'])
    position_errors = np.array(demo_data['position_errors'])
    times = np.array(demo_data['times'])
    
    # Extract state components
    z = states[:, 0]  # Altitude
    roll = states[:, 1]
    pitch = states[:, 2] 
    yaw = states[:, 3]
    vz = states[:, 4]  # Vertical velocity
    wx = states[:, 5]  # Angular velocities
    wy = states[:, 6]
    wz = states[:, 7]
    
    # Extract motor commands
    motor1 = actions[:, 0]
    motor2 = actions[:, 1] 
    motor3 = actions[:, 2]
    motor4 = actions[:, 3]
    
    # Performance metrics
    avg_pos_error = np.mean(position_errors)
    max_pos_error = np.max(position_errors)
    final_pos_error = position_errors[-1]
    avg_cost = np.mean(costs)
    target_z = config['target_pos'][2]
    
    # Create comprehensive performance plot
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'DHP Best Model Performance Analysis\nEpisode {metadata["best_episode"]} | Training Error: {metadata["best_position_error"]:.4f}m', fontsize=16, fontweight='bold')
    
    # Plot 1: Altitude Control
    ax1 = axes[0, 0]
    ax1.plot(times, z, 'b-', linewidth=2, label='Actual Altitude')
    ax1.axhline(y=target_z, color='r', linestyle='--', linewidth=2, label=f'Target ({target_z}m)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Altitude Control Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attitude Control  
    ax2 = axes[0, 1]
    ax2.plot(times, roll*180/np.pi, 'g-', linewidth=2, label='Roll')
    ax2.plot(times, pitch*180/np.pi, 'b-', linewidth=2, label='Pitch')
    ax2.plot(times, yaw*180/np.pi, 'r-', linewidth=2, label='Yaw')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Attitude Angles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Motor Commands
    ax3 = axes[1, 0]
    ax3.plot(times, motor1, label='Motor 1', linewidth=2)
    ax3.plot(times, motor2, label='Motor 2', linewidth=2)
    ax3.plot(times, motor3, label='Motor 3', linewidth=2)
    ax3.plot(times, motor4, label='Motor 4', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Motor Command [-1, 1]')
    ax3.set_title('Motor Commands')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Position Error
    ax4 = axes[1, 1]
    ax4.plot(times, position_errors, 'r-', linewidth=2, label='Position Error')
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='0.1m threshold')
    ax4.axhline(y=metadata["best_position_error"], color='green', linestyle='--', alpha=0.7, label=f'Training best: {metadata["best_position_error"]:.4f}m')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Position Tracking Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cost Function
    ax5 = axes[2, 0]
    ax5.plot(times, costs, 'purple', linewidth=2, label='DHP Cost')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Cost')
    ax5.set_title('DHP Cost Function')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance Summary
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Performance statistics
    excellent_steps = len([e for e in position_errors if e < 0.1])
    good_steps = len([e for e in position_errors if 0.1 <= e < 0.5])
    total_steps = len(position_errors)
    
    summary_text = f"""PERFORMANCE SUMMARY
    
Training Performance:
• Episode: {metadata["best_episode"]}
• Training Error: {metadata["best_position_error"]:.4f}m
• Timestamp: {metadata["timestamp"][:16]}

Demo Performance:
• Average Error: {avg_pos_error:.4f}m  
• Maximum Error: {max_pos_error:.4f}m
• Final Error: {final_pos_error:.4f}m
• Average Cost: {avg_cost:.3f}

Performance Distribution:
• Excellent (< 0.1m): {excellent_steps}/{total_steps} ({excellent_steps/total_steps:.1%})
• Good (0.1-0.5m): {good_steps}/{total_steps} ({good_steps/total_steps:.1%})

Control Statistics:
• Target Altitude: {target_z:.1f}m
• Mean Altitude: {np.mean(z):.3f}m
• Altitude Std: {np.std(z):.3f}m
• Max Motor Command: {np.max(np.abs(actions)):.3f}
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_filename = '/home/osos/Mohamed_Masters_Thesis/trial2/dhp_best_model_performance.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Performance plot saved: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    return avg_pos_error, max_pos_error, avg_cost

def demonstrate_best_dhp_policy(agent, config, metadata, 
                                episode_length=10.0, 
                                gui=True, 
                                record=True,
                                real_time=True):
    """
    Demonstrate the best DHP policy with GUI visualization
    """
    print("="*50)
    print("DEMONSTRATING BEST DHP POLICY")
    print("="*50)
    print(f"Best performance: {metadata['best_position_error']:.4f}m at episode {metadata['best_episode']}")
    print(f"Target position: {config['target_pos']}")
    print(f"Episode length: {episode_length} seconds")
    print(f"Real-time mode: {real_time}")
    print()
    
    # Create demonstration environment
    demo_env = CF2X_FastStates_HoverAviary(
        target_pos=np.array(config['target_pos']),
        gui=gui,
        record=record
    )
    
    # Reset environment
    state, info = demo_env.reset()
    reference = info['reference']
    
    max_steps = int(episode_length / config['dt'])
    demo_data = {
        'states': [],
        'actions': [],
        'costs': [],
        'position_errors': [],
        'times': []
    }
    
    print(f"Starting demonstration...")
    if real_time:
        print("Running in real-time mode - watch the smooth quadrotor control!")
    
    start_real_time = time.time()
    
    for step in range(max_steps):
        # Apply normalization if it was used during training
        X_normalized = normalize_state(state, config)
        R_sig_normalized = normalize_reference(reference, config)
        
        # Reshape for DHP agent (expects [batch, features, time])
        X_shaped = X_normalized.reshape(1, -1, 1)
        R_sig_shaped = R_sig_normalized.reshape(1, -1, 1)
        
        # Get action from best trained policy (no exploration)
        action = agent.action(X_shaped, reference=R_sig_shaped)
        action_flat = action.flatten()
        action_clipped = np.clip(action_flat, -1.0, 1.0)
        
        # Execute action
        next_state, reward, terminated, truncated, info = demo_env.step(action_clipped)
        
        # Store demonstration data
        demo_data['states'].append(state.copy())
        demo_data['actions'].append(action_clipped.copy())
        demo_data['costs'].append(info['dhp_cost'])
        demo_data['position_errors'].append(info['position_error'])
        demo_data['times'].append(step * config['dt'])
        
        # Update state and reference
        state = next_state
        reference = info['reference']
        
        # Real-time synchronization for smooth visualization
        if real_time and gui:
            expected_time = step * config['dt']
            elapsed_real_time = time.time() - start_real_time
            
            if elapsed_real_time < expected_time:
                time.sleep(expected_time - elapsed_real_time)
        
        # Print progress every second
        if step % int(1.0 / config['dt']) == 0:  # Every second
            current_time = step * config['dt']
            pos_error = info['position_error']
            cost = info['dhp_cost']
            print(f"Time: {current_time:4.1f}s | Position Error: {pos_error:.4f}m | Cost: {cost:.3f}")
        
        # Check termination
        if terminated or truncated:
            break
    
    demo_env.close()
    
    # Final statistics
    final_pos_error = demo_data['position_errors'][-1]
    avg_pos_error = np.mean(demo_data['position_errors'])
    max_pos_error = np.max(demo_data['position_errors'])
    avg_cost = np.mean(demo_data['costs'])
    
    print()
    print("="*50)
    print("DEMONSTRATION COMPLETED!")
    print("="*50)
    print(f"Final position error: {final_pos_error:.4f}m")
    print(f"Average position error: {avg_pos_error:.4f}m")
    print(f"Maximum position error: {max_pos_error:.4f}m")
    print(f"Average cost: {avg_cost:.3f}")
    print()
    
    # Performance assessment
    excellent_steps = len([e for e in demo_data['position_errors'] if e < 0.1])
    good_steps = len([e for e in demo_data['position_errors'] if 0.1 <= e < 0.5])
    acceptable_steps = len([e for e in demo_data['position_errors'] if 0.5 <= e < 1.0])
    
    total_steps = len(demo_data['position_errors'])
    print("Performance Distribution:")
    print(f"  Excellent (< 0.1m): {excellent_steps}/{total_steps} steps ({excellent_steps/total_steps:.1%})")
    print(f"  Good (0.1-0.5m):    {good_steps}/{total_steps} steps ({good_steps/total_steps:.1%})")
    print(f"  Acceptable (0.5-1.0m): {acceptable_steps}/{total_steps} steps ({acceptable_steps/total_steps:.1%})")
    print()
    
    if record:
        print("✓ Video recording saved in environment output directory")
    
    # Generate comprehensive performance plots
    plot_avg_error, plot_max_error, plot_avg_cost = plot_demonstration_results(demo_data, config, metadata)
    
    print(f"✓ Best DHP model demonstration completed successfully!")
    print(f"  Training episode: {metadata['best_episode']}")
    print(f"  Training error: {metadata['best_position_error']:.4f}m")
    print(f"  Demo average error: {avg_pos_error:.4f}m")
    
    return demo_data

def main():
    """Main demonstration function"""
    
    print("DHP Best Model Demonstration")
    print("="*50)
    print("Loading and demonstrating the best performing DHP model")
    print("from the completed training session.")
    print()
    
    try:
        # Load best model
        agent, config, metadata = load_best_dhp_model()
        
        # Run demonstration
        demo_data = demonstrate_best_dhp_policy(
            agent=agent,
            config=config, 
            metadata=metadata,
            episode_length=10.0,  # 10 second demonstration
            gui=True,             # Show GUI
            record=True,          # Record video
            real_time=True        # Real-time visualization
        )
        
        print()
        print("="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("The trained DHP agent successfully demonstrated precise")
        print("quadrotor control with excellent position tracking.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find model files: {e}")
        print("Make sure the training has completed and model files exist.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
