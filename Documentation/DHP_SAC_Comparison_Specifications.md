# DHP vs SAC Comparison Study: Technical Specifications

**Study Date:** August 8, 2025  
**Environment:** HoverAviary from gym-pybullet-drones  
**Objective:** Fair comparison between DHP and SAC algorithms using identical state/action spaces  
**Task:** Quadrotor hovering at target position [0, 0, 1]  

---

## 1. Environment Configuration

### 1.1 HoverAviary Setup
```python
env_config = {
    'drone_model': DroneModel.CF2P,        # Plus-configuration for natural decoupling
    'physics': Physics.PYB,                # Standard PyBullet physics
    'pyb_freq': 240,                       # 240 Hz physics simulation
    'ctrl_freq': 30,                       # 30 Hz control frequency
    'obs': ObservationType.KIN,            # Kinematic observations (not vision)
    'act': ActionType.RPM,                 # Direct motor RPM control
    'episode_length': 8.0,                 # 8 seconds per episode
    'target_position': [0, 0, 1]           # Hover target: 1m altitude
}
```

### 1.2 Task Specifications
- **Primary Task**: Hover at position [0, 0, 1] (1 meter altitude)
- **Success Criterion**: Distance to target < 0.0001 meters
- **Failure Conditions**: 
  - Horizontal drift > 1.5m (|x| > 1.5 or |y| > 1.5)
  - Altitude > 2.0m or < 0m
  - Excessive tilt > 0.4 radians (~23°)
  - Episode timeout (8 seconds)

---

## 2. State Space Definition

### 2.1 Full Drone State Vector (20 elements)
From `_getDroneStateVector()`:
```python
state_vector_20 = [
    # Position (3): World coordinates
    x, y, z,                                # [0:3]
    
    # Quaternion (4): Orientation representation  
    qx, qy, qz, qw,                        # [3:7]
    
    # Euler Angles (3): Roll, Pitch, Yaw
    roll, pitch, yaw,                      # [7:10]
    
    # Linear Velocity (3): World frame
    vx, vy, vz,                           # [10:13]
    
    # Angular Velocity (3): Body frame
    wx, wy, wz,                           # [13:16]
    
    # Last Action (4): Previous motor commands
    rpm1_prev, rpm2_prev, rpm3_prev, rpm4_prev  # [16:20]
]
```

### 2.2 Observation Space for RL Agents (9 elements - Fast States Only)
Inspired by msc-thesis state selection principle - only include states with fast response to actuators:
```python
# Fast states observation (9 elements) - matching aircraft DHP approach
fast_obs_9 = [
    # Altitude (1) - Medium-fast response to total thrust
    z,                                    # [0] - Current altitude
    
    # Attitude (3) - Fast response to motor torque differences
    roll, pitch, yaw,                     # [1:4] - Current orientation
    
    # Vertical velocity (1) - Fast response to thrust changes  
    vz,                                   # [5] - Current vertical velocity
    
    # Angular rates (3) - Immediate response to motor torques
    wx, wy, wz                           # [6:9] - Current angular rates
]

# EXCLUDED STATES (delegated to outer navigation loop):
# x, y - horizontal position (slow dynamics, > 500ms response)
# vx, vy - horizontal velocity (medium dynamics, navigation responsibility)
# Previous actions - not needed for inner-loop control

# Total observation space: 9 elements (same as aircraft DHP)
total_observation = fast_obs_9
```

### 2.3 State Selection Rationale
Following the aircraft DHP principle of including only states that respond quickly to actuators:

```python
# Motor Control Authority Analysis for CF2P:
control_hierarchy = {
    # IMMEDIATE RESPONSE (< 50ms): Angular accelerations
    'angular_rates': ['wx', 'wy', 'wz'],     # Direct torque → angular acceleration
    
    # FAST RESPONSE (50-200ms): Attitude changes  
    'attitude': ['roll', 'pitch', 'yaw'],    # Angular rates → attitude integration
    
    # MEDIUM RESPONSE (200-500ms): Vertical motion
    'vertical': ['vz', 'z'],                 # Thrust → vertical acceleration → velocity → position
    
    # SLOW RESPONSE (> 500ms): Horizontal motion (EXCLUDED)
    'horizontal': ['vx', 'vy', 'x', 'y']     # Attitude → horizontal forces → motion
}

# State selection criteria (from msc-thesis analysis):
# ✅ Direct control authority: Motors can immediately influence these states
# ✅ Fast dynamics: Response time compatible with control frequency (30 Hz)
# ✅ Observable: All selected states measurable by IMU/barometer
# ✅ Controllable: Full control authority over inner-loop dynamics
# ✅ Minimal: No redundant slow states that add complexity without benefit
```

### 2.4 Hierarchical Control Structure
```python
# Outer Loop (Navigation) - handles slow dynamics:
outer_loop_inputs = [x_ref, y_ref, yaw_ref]           # Position/heading commands
outer_loop_outputs = [roll_ref, pitch_ref]            # Attitude commands to inner loop

# Inner Loop (DHP/SAC) - handles fast dynamics:  
inner_loop_inputs = [z_ref, roll_ref, pitch_ref, yaw_ref]  # Altitude + attitude references
inner_loop_states = [z, roll, pitch, yaw, vz, wx, wy, wz]  # Fast states only
inner_loop_outputs = [rpm1, rpm2, rpm3, rpm4]             # Motor commands
```
### 2.5 State Normalization Ranges
```python
observation_bounds = {
    # Altitude bounds (based on task truncation limits)
    'z': [0.0, 2.0],                      # Altitude limits
    
    # Orientation bounds (based on task truncation limits)
    'roll': [-0.4, 0.4],                 # ~23° tilt limit
    'pitch': [-0.4, 0.4],                # ~23° tilt limit
    'yaw': [-π, π],                       # Full rotation
    
    # Vertical velocity bounds (estimated from physics)
    'vz': [-3.0, 3.0],                   # Vertical velocity limits
    
    # Angular velocity bounds (estimated from physics)
    'wx': [-10.0, 10.0],                 # Roll rate limits (rad/s)
    'wy': [-10.0, 10.0],                 # Pitch rate limits (rad/s)
    'wz': [-5.0, 5.0],                   # Yaw rate limits (rad/s)
}
```

---

## 3. Action Space Definition

### 3.1 Action Space (4 elements)
```python
action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

# Action interpretation for ActionType.RPM:
action_mapping = [
    rpm1_cmd,    # [0] - Front motor (CF2P: pitch control)
    rpm2_cmd,    # [1] - Right motor (CF2P: roll control) 
    rpm3_cmd,    # [2] - Back motor (CF2P: pitch control)
    rpm4_cmd     # [3] - Left motor (CF2P: roll control)
]

# Action preprocessing in _preprocessAction():
# rpm[k,:] = HOVER_RPM * (1 + 0.05 * action)
# Where HOVER_RPM ≈ 16800 RPM for CF2P
```

### 3.2 CF2P Motor Configuration
```python
# CF2P Plus-configuration motor layout:
#     FRONT (Motor 1)
#        |
# LEFT   +   RIGHT  
# (M4)   |   (M2)
#        |
#     BACK (Motor 3)

motor_control_mapping = {
    'vertical_thrust': [1, 1, 1, 1],      # All motors contribute to lift
    'pitch_control':   [1, 0, -1, 0],     # Front(+) vs Back(-) motors  
    'roll_control':    [0, 1, 0, -1],     # Right(+) vs Left(-) motors
    'yaw_control':     [1, -1, 1, -1]     # Alternating motor directions
}

# Action limits (5% variation around hover)
rpm_range = {
    'min_rpm': 16800 * (1 - 0.05) = 15960,
    'max_rpm': 16800 * (1 + 0.05) = 17640,
    'hover_rpm': 16800
}
```

---

## 4. Reference Signals and Cost Functions

### 4.1 Reference Signals (Fast States Only)
```python
# Reference for fast inner-loop control (9 elements matching state space)
fast_reference_vector = [
    z_ref,                                # [0] - Target altitude (1.0m)
    roll_ref, pitch_ref, yaw_ref,         # [1:4] - Target attitude (from outer loop or 0)
    vz_ref,                               # [5] - Target vertical velocity (0.0 m/s)
    wx_ref, wy_ref, wz_ref               # [6:9] - Target angular rates (0.0 rad/s)
]

# For hovering task:
hovering_reference = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Note: x, y position references handled by outer navigation loop
# Outer loop generates roll_ref, pitch_ref from horizontal position errors
```

### 4.2 HoverAviary Reward Function (for SAC)
```python
def compute_reward(state, target_pos=[0, 0, 1]):
    """
    HoverAviary reward function from the environment
    """
    position = state[0:3]
    distance = np.linalg.norm(target_pos - position)
    reward = max(0, 2 - distance**4)
    return reward

# Reward characteristics:
# - Maximum reward: 2.0 (when distance = 0)
# - Reward drops rapidly with distance (4th power)
# - Reward becomes 0 when distance ≥ 2^(1/4) ≈ 1.19m
```

### 4.3 DHP Cost Function (to be designed)
```python
def compute_dhp_cost(state, reference):
    """
    Quadratic cost function for DHP focusing on fast states only
    """
    # Fast state tracking errors (9 elements)
    z_error = state[0] - reference[0]                    # Altitude error
    attitude_error = state[1:4] - reference[1:4]        # Roll, pitch, yaw errors
    vz_error = state[4] - reference[4]                   # Vertical velocity error  
    rate_error = state[5:8] - reference[5:8]            # Angular rate errors
    
    # Quadratic cost with weighting matrix Q (9x9)
    Q = np.diag([
        # Altitude weight (high importance)
        100,                    # z tracking
        
        # Attitude weights (high importance, yaw less critical)
        80, 80, 20,            # roll, pitch, yaw tracking
        
        # Vertical velocity weight (medium importance)
        30,                    # vz tracking
        
        # Rate weights (for stability and damping)
        10, 10, 5              # wx, wy, wz tracking
    ])
    
    # Combined error vector
    error = np.concatenate([[z_error], attitude_error, [vz_error], rate_error])
    
    # Quadratic cost: J = e^T * Q * e
    cost = error.T @ Q @ error
    cost_gradient = 2 * Q @ error
    
    return cost, cost_gradient
```

---

## 5. Agent-Specific Configurations

### 5.1 SAC Agent Configuration
```python
sac_config = {
    # Network architecture
    'policy_layers': [256, 256],           # Actor network hidden layers
    'q_layers': [256, 256],                # Critic network hidden layers
    
    # Learning parameters
    'learning_rate': 3e-4,                 # Learning rate for all networks
    'buffer_size': 1000000,                # Replay buffer size
    'batch_size': 256,                     # Minibatch size
    'gamma': 0.99,                         # Discount factor
    'tau': 0.005,                          # Soft update rate
    'alpha': 0.2,                          # Entropy regularization (auto-tune)
    
    # Training parameters
    'train_freq': 1,                       # Train every step
    'gradient_steps': 1,                   # Gradient steps per environment step
    'learning_starts': 10000               # Steps before training starts
}
```

### 5.2 DHP Agent Configuration  
```python
dhp_config = {
    # Network architecture (based on msc-thesis)
    'state_size': 8,                       # Fast states only (without altitude reference)
    'reference_size': 8,                   # Reference vector size (without altitude)
    'action_size': 4,                      # Action space size
    'hidden_layers': [50, 50, 50],         # Network depth
    
    # Learning parameters
    'lr_critic': 0.1,                      # Critic learning rate (higher than aircraft)
    'lr_actor': 0.05,                      # Actor learning rate  
    'gamma': 0.95,                         # Discount factor (higher than aircraft)
    
    # Architecture options
    'split_architecture': True,            # Split longitudinal/lateral
    'target_network': True,                # Target network for stability
    'tau': 0.001,                          # Target network update rate
    
    # Model learning (online dynamics identification)
    'model_type': 'RLS',                   # Recursive Least Squares
    'rls_forgetting_factor': 0.9995,       # RLS forgetting factor
    'model_learning': True,                # Enable online model adaptation
    
    # Training parameters
    'update_cycles': 2,                    # Multiple updates per environment step
    'excitation_steps': 1000               # Initial exploration steps
}
```

---

## 6. Split Architecture for DHP (CF2P-Optimized Fast States)

### 6.1 Longitudinal Control (Pitch + Vertical)
```python
longitudinal_states = [
    z,           # [0] - Altitude  
    pitch,       # [1] - Pitch angle
    vz,          # [2] - Vertical velocity
    wy           # [3] - Pitch rate
]

longitudinal_actions = [
    rpm1_cmd,    # [0] - Front motor
    rpm3_cmd     # [1] - Back motor  
]

longitudinal_reference = [
    z_ref,       # [0] - Target altitude (1.0m)
    pitch_ref,   # [1] - Target pitch (0.0 rad)
    vz_ref,      # [2] - Target vertical velocity (0.0 m/s)
    wy_ref       # [3] - Target pitch rate (0.0 rad/s)
]
```

### 6.2 Lateral Control (Roll + Yaw)
```python
lateral_states = [
    roll,        # [0] - Roll angle
    yaw,         # [1] - Yaw angle  
    wx,          # [2] - Roll rate
    wz           # [3] - Yaw rate
]

lateral_actions = [
    rpm2_cmd,    # [0] - Right motor
    rpm4_cmd     # [1] - Left motor
]

lateral_reference = [
    roll_ref,    # [0] - Target roll (0.0 rad, or from outer loop)
    yaw_ref,     # [1] - Target yaw (0.0 rad, or from outer loop)
    wx_ref,      # [2] - Target roll rate (0.0 rad/s)
    wz_ref       # [3] - Target yaw rate (0.0 rad/s)
]
```

### 6.3 Split Architecture Benefits
```python
# Natural decoupling advantages:
decoupling_benefits = {
    'longitudinal': {
        'dynamics': 'z ↔ pitch ↔ vz ↔ wy (coupled through pitch)',
        'actuators': 'Front/Back motors (rpm1, rpm3)',
        'response': 'Pitch attitude → vertical forces → altitude',
        'simplicity': '4 states, 2 actuators, clear physics'
    },
    
    'lateral': {
        'dynamics': 'roll ↔ yaw ↔ wx ↔ wz (weakly coupled)',  
        'actuators': 'Left/Right motors (rpm2, rpm4)',
        'response': 'Roll/yaw torques → angular motion',
        'simplicity': '4 states, 2 actuators, independent learning'
    }
}

# Reduced complexity: 8 states total vs 16 original
# Natural physics: Matches CF2P mechanical design
# Faster learning: Smaller networks, clearer control relationships
```

---

## 7. Performance Metrics

### 7.1 Primary Metrics
```python
performance_metrics = {
    # Task performance
    'success_rate': 'Percentage of episodes reaching target',
    'mean_episode_reward': 'Average reward per episode',
    'position_rmse': 'Root mean square position error',
    'settling_time': 'Time to reach within 0.1m of target',
    
    # Learning efficiency
    'sample_efficiency': 'Episodes to reach performance threshold',
    'training_time': 'Wall-clock time for training',
    'convergence_stability': 'Variance in performance over time',
    
    # Control quality
    'control_smoothness': 'Action variation between steps',
    'energy_efficiency': 'Average motor power consumption',
    'attitude_stability': 'Roll/pitch angle deviations'
}
```

### 7.2 Comparison Criteria
```python
comparison_criteria = {
    # Learning performance
    'final_success_rate': 'Success rate after training convergence',
    'learning_speed': 'Episodes to reach 90% final performance',
    'sample_efficiency': 'Total environment steps needed',
    
    # Control performance
    'steady_state_error': 'Position error during hover',
    'disturbance_rejection': 'Recovery time from perturbations',
    'robustness': 'Performance under parameter variations',
    
    # Computational efficiency
    'training_time': 'Time to train each algorithm',
    'inference_time': 'Time per action computation',
    'memory_usage': 'Peak memory during training'
}
```

---

## 8. Expected Advantages

### 8.1 DHP Expected Advantages
1. **Model-based learning**: Better sample efficiency through dynamics learning
2. **Split architecture**: Natural control decoupling for CF2P configuration
3. **Online adaptation**: RLS model learning for real-time adaptation
4. **Gradient information**: Direct policy gradients from critic derivatives
5. **Reference tracking**: Explicit reference following capability

### 8.2 SAC Expected Advantages  
1. **Proven performance**: Well-established algorithm for continuous control
2. **Exploration**: Built-in entropy regularization for exploration
3. **Stability**: Off-policy learning with experience replay
4. **Hyperparameter robustness**: Less sensitive to parameter tuning
5. **Implementation maturity**: Well-tested stable-baselines3 implementation

---

## 9. Implementation Phases

### 9.1 Phase 1: Environment Setup and SAC Baseline
- [ ] Configure HoverAviary with ActionType.RPM
- [ ] Implement SAC training pipeline
- [ ] Establish performance baseline
- [ ] Validate observation/action spaces

### 9.2 Phase 2: DHP Implementation
- [ ] Implement DHP agent with split architecture
- [ ] Integrate online RLS model learning
- [ ] Implement reference tracking cost function
- [ ] Validate gradient computation

### 9.3 Phase 3: Comparative Evaluation
- [ ] Run controlled experiments with identical conditions
- [ ] Collect performance metrics
- [ ] Statistical analysis of results
- [ ] Ablation studies (split vs unified, model learning on/off)

---

**Document Status:** Specification Complete - Ready for Implementation  
**Next Step:** Phase 1 - Environment Setup and SAC Baseline  
**No Code Development Until User Approval**
