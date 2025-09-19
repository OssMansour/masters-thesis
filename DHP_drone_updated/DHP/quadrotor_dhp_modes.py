"""
Quadrotor DHP Modes Extension

This module extends the msc-thesis phlab module to support quadrotor control
with split architecture adapted for vertical and attitude control.

Author: DHP vs SAC Comparison Study  
Date: August 9, 2025
"""

import numpy as np

# Quadrotor mode identifiers (extend phlab)
ID_QUAD_VERTICAL = 400   # Vertical control (altitude + vertical velocity)
ID_QUAD_ATTITUDE = 500   # Attitude control (roll, pitch, yaw + rates)
ID_QUAD_FULL = 600       # Full quadrotor state (all 8 fast states)

# Quadrotor state definitions
# Fast states: [z, roll, pitch, yaw, vz, wx, wy, wz]
QUAD_VERTICAL_STATES = ['z', 'vz']                                    # Vertical position and velocity
QUAD_ATTITUDE_STATES = ['roll', 'pitch', 'yaw', 'wx', 'wy', 'wz']    # Attitude angles and rates  
QUAD_FULL_STATES = ['z', 'roll', 'pitch', 'yaw', 'vz', 'wx', 'wy', 'wz']  # All fast states

# State tracking (which states are actively controlled)
QUAD_TRACK_STATES = {
    ID_QUAD_VERTICAL: [True, False, False, False, True, False, False, False],    # Track z and vz
    ID_QUAD_ATTITUDE: [False, True, True, True, False, True, True, True],        # Track attitude and rates
    ID_QUAD_FULL: [True, True, True, True, True, True, True, True]               # Track all states
}

# State definitions
QUAD_STATES = {
    ID_QUAD_VERTICAL: QUAD_VERTICAL_STATES,
    ID_QUAD_ATTITUDE: QUAD_ATTITUDE_STATES, 
    ID_QUAD_FULL: QUAD_FULL_STATES
}

# State indices (which indices in the full state vector correspond to each mode)
QUAD_IDX = {
    ID_QUAD_VERTICAL: [0, 4],                           # z=0, vz=4
    ID_QUAD_ATTITUDE: [1, 2, 3, 5, 6, 7],              # roll=1, pitch=2, yaw=3, wx=5, wy=6, wz=7
    ID_QUAD_FULL: [0, 1, 2, 3, 4, 5, 6, 7]             # All indices
}

def get_quadrotor_split_config():
    """
    Get quadrotor split configuration for DHP agent
    
    Returns:
        dict: Configuration dictionary with quadrotor-specific split setup
    """
    return {
        'mode_id': ID_QUAD_FULL,
        'tracked_states': QUAD_TRACK_STATES[ID_QUAD_FULL],
        'vertical_indices': QUAD_IDX[ID_QUAD_VERTICAL],      # Which states go to vertical network
        'attitude_indices': QUAD_IDX[ID_QUAD_ATTITUDE],      # Which states go to attitude network
        'vertical_actions': [0],           # Thrust (collective motor command) - first action output
        'attitude_actions': [1, 2, 3]     # Roll, pitch, yaw moments - remaining action outputs
    }

def quadrotor_state2loc(mode_id):
    """
    Create state name to local index mapping for quadrotor modes
    
    Args:
        mode_id: Quadrotor mode identifier
        
    Returns:
        dict: Mapping from state names to local indices
    """
    states = QUAD_STATES[mode_id]
    return dict(zip(states, range(len(states))))

def get_quadrotor_lon_lat_mapping():
    """
    Map quadrotor states to longitudinal/lateral concept for compatibility
    with existing split architecture
    
    In aircraft terms:
    - Longitudinal: Forward/backward motion, altitude control
    - Lateral: Side-to-side motion, roll/yaw control
    
    For quadrotor:
    - "Longitudinal": Vertical motion (z, vz) + pitch control (pitch, wy)
    - "Lateral": Roll/yaw control (roll, yaw, wx, wz)
    
    Returns:
        tuple: (longitudinal_indices, lateral_indices) in full state vector
    """
    # Map quadrotor control to aircraft concept
    longitudinal_indices = [0, 2, 4, 6]  # z, pitch, vz, wy (vertical + pitch)
    lateral_indices = [1, 3, 5, 7]       # roll, yaw, wx, wz (roll + yaw)
    
    return longitudinal_indices, lateral_indices

def create_quadrotor_phlab_extension():
    """
    Create extensions to phlab module for quadrotor support
    
    Returns:
        dict: Dictionary with phlab-compatible definitions
    """
    lon_idx, lat_idx = get_quadrotor_lon_lat_mapping()
    
    # Create phlab-compatible definitions
    extension = {
        # Mode IDs
        'ID_QUAD_LON': 101,  # Quadrotor longitudinal (vertical + pitch)
        'ID_QUAD_LAT': 201,  # Quadrotor lateral (roll + yaw)
        'ID_QUAD_FULL': 301, # Full quadrotor
        
        # States for each mode
        'states': {
            101: ['z', 'pitch', 'vz', 'wy'],        # Longitudinal states
            201: ['roll', 'yaw', 'wx', 'wz'],       # Lateral states  
            301: ['z', 'roll', 'pitch', 'yaw', 'vz', 'wx', 'wy', 'wz']  # Full states
        },
        
        # State indices in full vector
        'idx': {
            101: lon_idx,      # Longitudinal indices
            201: lat_idx,      # Lateral indices
            301: list(range(8)) # Full indices
        },
        
        # Tracking states
        'track_states': {
            101: [True, True, True, True],           # Track all longitudinal states
            201: [True, True, True, True],           # Track all lateral states
            301: [True] * 8                          # Track all states
        }
    }
    
    return extension

if __name__ == "__main__":
    # Test the quadrotor extensions
    print("Quadrotor DHP Mode Extensions")
    print("=" * 40)
    
    config = get_quadrotor_split_config()
    print(f"Split config: {config}")
    
    extension = create_quadrotor_phlab_extension()
    print(f"PHLab extension: {extension}")
    
    lon_idx, lat_idx = get_quadrotor_lon_lat_mapping()
    print(f"Longitudinal indices: {lon_idx}")
    print(f"Lateral indices: {lat_idx}")
