"""
Quadrotor DHP Agent

This module extends the msc-thesis DHP agent to support quadrotor-specific
split architecture with vertical and attitude control networks.

Author: DHP vs SAC Comparison Study  
Date: August 9, 2025
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add path for msc-thesis
sys.path.append('/home/osos/Mohamed_Masters_Thesis/msc-thesis')

from agents.dhp import Agent as DHP_Agent
from quadrotor_dhp_modes import get_quadrotor_lon_lat_mapping, create_quadrotor_phlab_extension


class QuadrotorDHPAgent(DHP_Agent):
    """
    DHP Agent with quadrotor-specific split architecture
    
    Extends the original DHP agent to support quadrotor control with:
    - Vertical control network: altitude and vertical velocity (thrust control)
    - Attitude control network: roll, pitch, yaw and rates (moment control)
    """
    
    def __init__(self, **kwargs):
        """
        Initialize quadrotor DHP agent
        
        Args:
            **kwargs: Standard DHP agent arguments plus:
                - quadrotor_split: Enable quadrotor-specific split architecture
        """
        self.quadrotor_split = kwargs.pop('quadrotor_split', False)
        
        if self.quadrotor_split:
            # Override split to use our custom implementation
            kwargs['split'] = False  # Disable original split
            
        # Initialize parent DHP agent
        super().__init__(**kwargs)
        
        # Build quadrotor split architecture if requested
        if self.quadrotor_split:
            with self.graph.as_default():
                self.build_quadrotor_split_actor(self.lr_actor)
                # Update session initialization
                self.session.run(tf.global_variables_initializer())
    
    def build_quadrotor_split_actor(self, learn_rate):
        """
        Build quadrotor-specific split actor with vertical and attitude networks
        """
        print("Building quadrotor split actor architecture...")
        
        # Get quadrotor state mapping
        lon_indices, lat_indices = get_quadrotor_lon_lat_mapping()
        
        # Layer parameters
        h_kwargs = self.hidden_layer_kwargs
        y_kwargs = self.output_layer_kwargs

        # Add scope to the network
        with tf.variable_scope('quadrotor_actor'):
            # Placeholders
            self.x_actor = tf.placeholder(tf.float32, shape=[None, self.state_size], name='state_input')
            self.x_ref_actor = tf.placeholder(tf.float32, shape=[None, self.reference_size], name='reference_input')
            self.x_gradient_actor = tf.placeholder(tf.float32, shape=[None, self.output_size], name='gradient_input') 
            self.x_lr_actor = tf.placeholder_with_default(learn_rate, shape=[], name='learn_rate_input')

            # Vertical control network (thrust control)
            with tf.variable_scope('vertical'):
                # Extract vertical states: z, vz, pitch, wy (altitude control)
                x_vertical = tf.gather(self.x_actor, lon_indices, axis=1)
                x_ref_vertical = tf.gather(self.x_ref_actor, lon_indices, axis=1) 
                
                # Combine state and reference
                x_vert_input = tf.concat([x_vertical, x_ref_vertical], axis=1)
                
                # Build vertical control network
                x_vert = x_vert_input
                for layer in range(len(self.hidden_layer_size)):
                    name_str = 'dense_' + str(layer)
                    x_vert = tf.layers.dense(x_vert, self.hidden_layer_size[layer], name=name_str, **h_kwargs)
                
                # Output: thrust command (collective)
                thrust_output = tf.layers.dense(x_vert, 1, activation=None, name='thrust_output', **y_kwargs)

            # Attitude control network (moment control) 
            with tf.variable_scope('attitude'):
                # Extract attitude states: roll, yaw, wx, wz (attitude control)
                x_attitude = tf.gather(self.x_actor, lat_indices, axis=1)
                x_ref_attitude = tf.gather(self.x_ref_actor, lat_indices, axis=1)
                
                # Combine state and reference
                x_att_input = tf.concat([x_attitude, x_ref_attitude], axis=1)
                
                # Build attitude control network
                x_att = x_att_input
                for layer in range(len(self.hidden_layer_size)):
                    name_str = 'dense_' + str(layer)
                    x_att = tf.layers.dense(x_att, self.hidden_layer_size[layer], name=name_str, **h_kwargs)
                
                # Output: moment commands (roll, pitch, yaw)
                roll_moment = tf.layers.dense(x_att, 1, activation=None, name='roll_moment', **y_kwargs)
                pitch_moment = tf.layers.dense(x_att, 1, activation=None, name='pitch_moment', **y_kwargs)
                yaw_moment = tf.layers.dense(x_att, 1, activation=None, name='yaw_moment', **y_kwargs)

            # Combine outputs: [thrust, roll_moment, pitch_moment, yaw_moment]
            self.output_actor = [thrust_output, roll_moment, pitch_moment, yaw_moment]
            
            # Flatten outputs for compatibility
            self.action_output = tf.concat(self.output_actor, axis=1)

            # Compute gradients
            self.actor_loss = tf.reduce_mean(tf.multiply(self.action_output, self.x_gradient_actor))
            self.actor_optimizer = tf.train.AdamOptimizer(self.x_lr_actor)
            self.actor_train_op = self.actor_optimizer.minimize(-self.actor_loss)  # Negative for maximization

            # Get trainable variables for this scope
            self.vars_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='quadrotor_actor')

        print(f"Quadrotor split actor built:")
        print(f"  - Vertical network inputs: {len(lon_indices)} states")
        print(f"  - Attitude network inputs: {len(lat_indices)} states") 
        print(f"  - Output: 4 actions [thrust, roll_moment, pitch_moment, yaw_moment]")

    def action(self, state, reference=None):
        """
        Get action from quadrotor split actor
        
        Args:
            state: Current state [batch, state_size, time]
            reference: Reference trajectory [batch, reference_size, time]
            
        Returns:
            action: Action commands [batch, action_size, time]
        """
        if self.quadrotor_split:
            # Use quadrotor split actor
            feed_dict = {self.x_actor: state.reshape(-1, self.state_size)}
            if reference is not None:
                feed_dict[self.x_ref_actor] = reference.reshape(-1, self.reference_size)
            
            action = self.session.run(self.action_output, feed_dict=feed_dict)
            return action.reshape(state.shape[0], self.output_size, state.shape[2])
        else:
            # Use parent implementation
            return super().action(state, reference)

    def update_actor(self, state, reference=None, gradient=None):
        """
        Update quadrotor split actor networks
        
        Args:
            state: Current state [batch, state_size, time]
            reference: Reference trajectory [batch, reference_size, time]  
            gradient: Policy gradient [batch, action_size, time]
        """
        if self.quadrotor_split:
            # Update quadrotor split actor
            feed_dict = {
                self.x_actor: state.reshape(-1, self.state_size),
                self.x_gradient_actor: gradient.reshape(-1, self.output_size)
            }
            if reference is not None:
                feed_dict[self.x_ref_actor] = reference.reshape(-1, self.reference_size)
            
            self.session.run(self.actor_train_op, feed_dict=feed_dict)
        else:
            # Use parent implementation
            super().update_actor(state, reference, gradient)

    def gradient_actor(self, state, reference=None):
        """
        Get actor gradients for quadrotor split architecture
        
        Args:
            state: Current state [batch, state_size, time]
            reference: Reference trajectory [batch, reference_size, time]
            
        Returns:
            gradient: Actor gradients [action_size, state_size]
        """
        if self.quadrotor_split:
            # Compute gradients for quadrotor split actor
            feed_dict = {self.x_actor: state.reshape(-1, self.state_size)}
            if reference is not None:
                feed_dict[self.x_ref_actor] = reference.reshape(-1, self.reference_size)
            
            # Compute jacobian dπ/dx for each action
            gradients = []
            for i in range(self.output_size):
                grad = tf.gradients(self.action_output[:, i], self.x_actor)[0]
                grad_value = self.session.run(grad, feed_dict=feed_dict)
                gradients.append(grad_value.flatten())
            
            return np.array(gradients)  # [action_size, state_size]
        else:
            # Use parent implementation
            return super().gradient_actor(state, reference)


if __name__ == "__main__":
    # Test quadrotor DHP agent
    print("Testing Quadrotor DHP Agent")
    print("=" * 40)
    
    # Test configuration
    agent_kwargs = {
        'input_size': [8, 8],          # 8 states, 8 references
        'output_size': 4,              # 4 motor commands
        'hidden_layer_size': [50, 50], 
        'lr_critic': 0.01,
        'lr_actor': 0.005,
        'gamma': 0.95,
        'quadrotor_split': True,       # Enable quadrotor split
        'target_network': True,
        'tau': 0.001
    }
    
    try:
        agent = QuadrotorDHPAgent(**agent_kwargs)
        print("✓ Quadrotor DHP agent created successfully!")
        print(f"  - Split architecture: {agent.quadrotor_split}")
        print(f"  - State size: {agent.state_size}")
        print(f"  - Action size: {agent.output_size}")
        
        # Test action computation
        test_state = np.random.randn(1, 8, 1)
        test_reference = np.random.randn(1, 8, 1)
        action = agent.action(test_state, test_reference)
        print(f"  - Action shape: {action.shape}")
        print(f"  - Action values: {action.flatten()}")
        
    except Exception as e:
        print(f"✗ Error creating quadrotor DHP agent: {e}")
        import traceback
        traceback.print_exc()
