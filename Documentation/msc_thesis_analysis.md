# MSc-Thesis Repository Analysis: DHP Implementation for Aircraft Control

**Analysis Date:** August 8, 2025  
**Repository:** D. Kroezen MSc Thesis - Cessna Citation 550 DHP Control  
**Focus:** DHP Agent Architecture and Environment Integration  

---

## Executive Summary

The msc-thesis repository implements a **Dual Heuristic Programming (DHP) reinforcement learning agent** for flight control of a Cessna Citation 550 aircraft. The implementation demonstrates online adaptive control using split longitudinal/lateral actor networks with a sophisticated aircraft dynamics model and reference tracking system.

---

## 1. Repository Architecture Overview

### 1.1 Directory Structure

```
msc-thesis/
├── agents/                  # RL agent implementations
│   ├── base.py             # Abstract base agent class
│   ├── dhp.py              # Main DHP agent implementation (~400 lines)
│   └── model.py            # Aircraft dynamics models (NN, RLS, LS)
├── code/                   # Execution scripts
│   ├── DHP/
│   │   └── dhp_main.py     # Main training/execution script (~300 lines)
│   └── PID/                # PID baseline implementations
├── envs/                   # Environment implementations
│   ├── phlab/              # Citation aircraft simulation
│   │   ├── citation.py     # SWIG-generated environment interface
│   │   ├── fc0/, fc1/, fc2/, fc3/  # Flight condition directories
│   │   └── lin_system.mat  # Linearized system model
│   └── __init__.py
├── utils/                  # Utility functions and controllers
│   ├── phlab/              # Aircraft-specific utilities
│   │   └── __init__.py     # State management and normalization
│   ├── __init__.py         # Signal generation and PID controllers
│   └── pid.py              # PID controller implementation
└── environment.yml         # Conda environment specification
```

### 1.2 Key Components Integration

**Agent-Environment-Model Trinity:**
- **DHP Agent** (`agents/dhp.py`): Neural network-based policy and value function
- **Citation Environment** (`envs/phlab/`): High-fidelity aircraft simulation
- **Dynamics Model** (`agents/model.py`): Online model learning (RLS/Neural Network)

---

## 2. DHP Agent Architecture Analysis

### 2.1 Core Agent Structure (`agents/dhp.py`)

**Network Architecture:**
```python
class Agent(BaseAgent):
    def __init__(self, **kwargs):
        # Network Configuration
        self.input_size = [ac_state_size, reference_size]  # [9, 3] for LATLON mode
        self.output_size = action_size                      # 1-3 actuators
        self.hidden_layer_size = [50, 50, 50]              # Configurable depth
        
        # Learning Parameters
        self.lr_critic = 0.1                               # Critic learning rate
        self.lr_actor = 0.05                               # Actor learning rate
        self.gamma = 0.4                                   # Discount factor
        
        # Architecture Options
        self.split = True                                   # Split longitudinal/lateral
        self.target_network = True                          # Target network for stability
        self.tau = 0.001                                   # Target network update rate
```

**Split Actor Architecture:**
```python
def build_split_actor(self, learn_rate):
    with tf.variable_scope('actor'):
        with tf.variable_scope('longitudinal'):
            # Longitudinal states: [q, V, alpha, theta, h]
            # Controls: elevator (pitch control)
            
        with tf.variable_scope('lateral'):  
            # Lateral states: [p, r, beta, phi]
            # Controls: aileron, rudder (roll/yaw control)
```

### 2.2 Input/Output Processing

**State Vector Processing:**
- **Full State Size**: 9 elements (LATLON mode)
- **Actor-Critic State Size**: Variable based on tracked states
- **Reference Size**: Number of tracked states for reference following

**Action Processing:**
- **Action Clipping**: Physical actuator limits enforced
- **Trim Compensation**: Initial trim added as offset
- **Action Types**: Elevator, Aileron, Rudder (1-3 actuators)

### 2.3 Network Training Pipeline

**Critic Update (Value Function):**
```python
# Cost-to-go estimation
lmbda = agent.value_derivative(X, reference=R_sig)
target_lmbda = agent.target_value_derivative(X_next_pred, reference=R_sig)

# Bellman residual gradient
grad_critic = lmbda - (dcostdx + gamma*target_lmbda) @ (A + B @ dactiondx)
agent.update_critic(X, reference=R_sig, gradient=grad_critic)
```

**Actor Update (Policy):**
```python
# Policy gradient using critic information
lmbda = agent.value_derivative(X_next_pred, reference=R_sig)
grad_actor = (dcostdx + gamma*lmbda) @ B
agent.update_actor(X, reference=R_sig, gradient=grad_actor)
```

---

## 3. Environment Integration Analysis

### 3.1 Citation Aircraft Environment

**Simulation Interface:**
- **Backend**: SWIG-wrapped C++ simulation (`_citation.cp36-win_amd64.pyd`)
- **Interface**: Python wrapper (`citation.py`, `citation_act.py`)
- **State Vector**: 30+ elements including position, orientation, velocities
- **Control Inputs**: 10 actuator channels (primary: elevator, aileron, rudder)

**Flight Conditions:**
- **FC0**: 140 m/s, 5000m altitude, 0° gamma
- **FC1**: 90 m/s, 5000m altitude, 0° gamma  
- **FC2**: 140 m/s, 2000m altitude, 0° gamma
- **FC3**: 90 m/s, 2000m altitude, 0° gamma

### 3.2 State Management System

**Mode Selection (`utils/phlab/__init__.py`):**
```python
# Available modes
LON = 'lon'        # Longitudinal only [q, V, alpha, theta, h]
LAT = 'lat'        # Lateral only [p, r, V, beta, phi, h]  
LATLON = 'latlon'  # Combined [p, q, r, V, alpha, beta, phi, theta, h]
```

**State Tracking Configuration:**
```python
# Example: Track p, q, beta (ID_PQ_BETA_LATLON = 311)
TRACK = ['p', 'q', 'beta']                    # States to track
EXCLUDE = []                                  # States to exclude  
tracked_states = [True, True, False, False, False, True, False, False, False]
```

**Normalization Pipeline:**
```python
def normalize(state, id):
    # Speed normalization: [40, 200] m/s -> [0, 1]
    state[:, 'V'] = (V - 40) / (200 - 40)
    # Altitude normalization: [0, 4000] m -> [0, 1]  
    state[:, 'h'] = (h - 0) / (4000 - 0)
    return state
```

### 3.3 Reference Signal Generation

**Outer Loop Controllers:**
```python
# Altitude Controller (PI + P cascade)
h_controller = AltitudeController(dt=dt)
q_ref = h_controller.cmd(alt_ref, altitude, pitch)

# Roll Angle Controller (P control)
p_controller = RollAngleController(dt=dt)  
p_ref = p_controller.cmd(phi_ref, roll_angle)

# Reference vector construction
R_sig = np.array([p_ref, q_ref, beta_ref])  # Based on tracked states
```

---

## 4. Aircraft Dynamics Model Integration

### 4.1 Model Architecture (`agents/model.py`)

**Recursive Least Squares (Primary):**
```python
class RecursiveLeastSquares():
    def __init__(self, **kwargs):
        self.state_size = 9                    # Aircraft state dimension
        self.action_size = 3                   # Control input dimension
        self.gamma = 0.9995                    # Forgetting factor
        self.predict_delta = False             # Predict absolute vs incremental
        
    def update(self, state, action, next_state):
        # Online parameter estimation
        # Updates: self.W (parameters), self.cov (covariance)
        
    def gradient_state(self, state, action):
        return self.W[:self.state_size, :].T   # ∂x_{t+1}/∂x_t
        
    def gradient_action(self, state, action):  
        return self.W[self.state_size:, :].T   # ∂x_{t+1}/∂u_t
```

**Neural Network Alternative:**
```python
class NeuralNetwork():
    def __init__(self, **kwargs):
        self.hidden_layer_size = [100, 100, 100]  # Deep network
        self.activation = tf.nn.relu
        # Implements same interface as RLS
```

### 4.2 Model-Agent Integration

**Forward Prediction:**
```python
# In training loop (dhp_main.py)
action = agent.action(X, reference=R_sig)
action_clipped = np.clip(action, actuator_limits)
X_next_pred = ac_model.predict(X, action_clipped)  # Model prediction
```

**Gradient Computation:**
```python
# For critic/actor updates  
A = ac_model.gradient_state(X, action)    # State transition matrix
B = ac_model.gradient_action(X, action)   # Control influence matrix
dactiondx = agent.gradient_actor(X, reference=R_sig)  # Policy gradient
```

---

## 5. Training Loop Architecture

### 5.1 Main Training Pipeline (`dhp_main.py`)

**Initialization Phase:**
```python
# Environment setup
citation = importlib.import_module(f'envs.phlab.fc{FLIGHT_CONDITION}.citation_act')
citation.initialize()

# Agent creation
agent = DHP.Agent(**agent_kwargs)
agent.trim = initial_trim_commands

# Model creation  
ac_model = model.RecursiveLeastSquares(**model_kwargs)

# Reference signal generation
alt_signal, phi_signal = utils.experiment_profile(dt, alt_0=initial_altitude)
```

**Training Loop Structure:**
```python
for i in range(time_steps):
    # 1. Reference Signal Generation
    phi_ref = phi_signal[i]
    alt_ref = alt_signal[i] 
    p_ref = p_controller.cmd(phi_ref, current_phi)
    q_ref = h_controller.cmd(alt_ref, current_h, current_theta)
    
    # 2. Agent Updates (multiple cycles)
    for j in range(update_cycles):
        # Forward prediction
        action = agent.action(X, reference=R_sig)
        X_next_pred = ac_model.predict(X, action)
        
        # Cost computation
        e = P @ (X_next_pred - X_ref)
        cost = e.T @ Q @ e
        dcostdx = 2 * e.T @ Q @ P
        
        # Critic update
        grad_critic = lmbda - (dcostdx + gamma*target_lmbda) @ (A + B @ dactiondx)
        agent.update_critic(X, reference=R_sig, gradient=grad_critic)
        
        # Actor update  
        grad_actor = (dcostdx + gamma*lmbda) @ B
        agent.update_actor(X, reference=R_sig, gradient=grad_actor)
    
    # 3. Environment Step
    action = agent.action(X, reference=R_sig)
    if i < 1000: action += excitation_signal[i]  # Initial exploration
    cmd = command_interface(cmd, action)
    X_cit, _ = citation.step(cmd, env, failure)
    X_next = process_citation_state(X_cit)
    
    # 4. Model Update
    ac_model.update(X, action, X_next)
    
    # 5. State Update
    X = X_next.copy()
```

### 5.2 Cost Function Design

**Quadratic Cost Structure:**
```python
# State error weighting
P = np.diag(TRACKED)                           # Projection matrix
Q = np.diag(STATE_ERROR_WEIGHTS)               # Error weights

# Special weighting for beta tracking
if tracking_beta:
    Q[beta_index] = 100                        # High penalty for sideslip

# Cost computation
e = P @ (X_next - X_ref)                       # Tracking error
cost = e.T @ Q @ e                             # Quadratic cost
dcostdx = 2 * e.T @ Q @ P                      # Cost gradient
```

---

## 6. Key Integration Patterns

### 6.1 Agent-Environment Interface

**State Flow:**
```
Citation Simulation (30+ states) 
    ↓ [state selection & normalization]
DHP Agent Input (9 states)
    ↓ [neural network processing]  
Control Actions (1-3 actuators)
    ↓ [clipping & command interface]
Citation Actuator Commands (10 channels)
```

**Reference Tracking:**
```
High-Level Commands (altitude, bank angle)
    ↓ [outer loop controllers]
Rate References (p_ref, q_ref, beta_ref)
    ↓ [reference vector construction]
Agent Reference Input
    ↓ [tracking error computation]
Cost Function Input
```

### 6.2 Model-Agent Coupling

**Online Learning Integration:**
```python
# Prediction phase
X_next_pred = model.predict(X, action)        # Forward model

# Gradient computation phase  
A = model.gradient_state(X, action)           # ∂x_{t+1}/∂x_t
B = model.gradient_action(X, action)          # ∂x_{t+1}/∂u_t
dactiondx = agent.gradient_actor(X, ref)      # ∂u/∂x

# Update phase
grad_critic = lmbda - (dcostdx + γ*target_lmbda) @ (A + B @ dactiondx)
grad_actor = (dcostdx + γ*lmbda) @ B
```

### 6.3 Multi-Level Control Architecture

**Hierarchical Control Structure:**
```
Reference Signals (alt_ref, phi_ref)
    ↓ [Outer Loop Controllers]
Rate References (p_ref, q_ref, beta_ref)  
    ↓ [DHP Agent]
Actuator Commands (elevator, aileron, rudder)
    ↓ [Citation Simulation]
Aircraft Response
```

---

## 7. Configuration Management

### 7.1 Experiment Configuration

**Flight Mode Selection:**
```python
MODE = phlab.LATLON                            # Control mode
TRACK = ['p', 'q', 'beta']                     # Tracked states
EXCLUDE = []                                   # Excluded states
FLIGHT_CONDITION = 3                           # Aircraft configuration
```

**Training Parameters:**
```python
dt = 0.02                                      # 50 Hz control frequency
T = 300.0                                      # 5-minute episodes
update_cycles = 2                              # Multiple updates per step
```

### 7.2 Agent Hyperparameters

**Network Configuration:**
```python
agent_kwargs = {
    'input_size': [ac_state_size, reference_size],
    'output_size': action_size,
    'hidden_layer_size': [50, 50, 50],
    'lr_critic': 0.1,
    'lr_actor': 0.05, 
    'gamma': 0.4,
    'split': True,                             # Split longitudinal/lateral
    'target_network': True,                    # Target network stability
    'tau': 0.001                               # Target network update rate
}
```

**Model Configuration:**
```python
model_kwargs = {
    'state_size': ac_state_size,
    'action_size': action_size,
    'gamma': 0.9995,                           # RLS forgetting factor
    'covariance': 100,                         # Initial covariance
    'predict_delta': False                     # Absolute state prediction
}
```

---

## 8. Comparison with gym-pybullet-drones

### 8.1 Architecture Differences

**msc-thesis (Aircraft DHP):**
- Single high-fidelity aircraft simulation
- Online model learning with RLS
- Split longitudinal/lateral actor architecture
- Hierarchical control with outer loop controllers
- Reference tracking focus
- TensorFlow 1.x implementation

**gym-pybullet-drones (Quadrotor Framework):**
- Multiple drone models and physics options
- No online model learning (direct control)
- Unified actor architecture
- Direct low-level control
- Position/velocity control focus
- Modern framework compatibility

### 8.2 Integration Complexity

**msc-thesis Integration Challenges:**
- Complex state space management (9-30 dimensions)
- Multi-level reference signal processing
- Online model adaptation requirements
- Aircraft-specific normalization
- Custom control interfaces

**gym-pybullet-drones Advantages:**
- Standardized environment interface
- Pre-built action/observation spaces
- Built-in multi-agent support
- Modern RL ecosystem compatibility

---

## 9. Key Insights for Quadrotor DHP Implementation

### 9.1 Transferable Patterns

**Split Actor Architecture:**
- Longitudinal/lateral separation aligns perfectly with CF2P quadrotor
- Independent control channels reduce coupling complexity
- Enables specialized learning for each axis

**Reference Tracking System:**
- Hierarchical control structure applicable to quadrotor
- Outer loop position → inner loop rate control
- Cost function design for tracking performance

**Online Model Learning:**
- RLS model adaptation could improve quadrotor control
- Gradient computation for policy updates
- Real-time system identification

### 9.2 Adaptation Requirements

**State Vector Simplification:**
- Quadrotor: 20 elements (position, attitude, velocities, RPMs)
- Aircraft: 9-30 elements (complex aerodynamics)
- Simpler normalization and processing

**Control Simplification:**
- Quadrotor: 4 motor RPMs or PID setpoints
- Aircraft: 3-10 actuator channels with complex limits
- Direct motor control vs. hierarchical actuation

**Physics Differences:**
- Quadrotor: Rigid body + motor dynamics
- Aircraft: Complex aerodynamics + engine + control surfaces
- Simpler dynamics modeling for quadrotor

---

## Conclusion

The msc-thesis repository demonstrates a sophisticated implementation of DHP for aircraft control with several key architectural innovations:

1. **Split Actor Design**: Longitudinal/lateral separation that aligns perfectly with CF2P quadrotor natural decoupling
2. **Online Model Learning**: RLS-based system identification enabling adaptive control
3. **Hierarchical Reference Tracking**: Multi-level control structure suitable for complex tracking tasks
4. **Robust Training Pipeline**: Comprehensive integration of agent, environment, and model components

**Key Takeaways for Quadrotor Implementation:**
- The split actor architecture is directly applicable to CF2P quadrotor control
- Online model learning could significantly improve performance
- Reference tracking framework provides excellent foundation for waypoint following
- Training loop structure offers proven integration patterns

The implementation provides an excellent reference for developing DHP control systems with proper agent-environment integration patterns that can be adapted for quadrotor applications in the gym-pybullet-drones framework.

---

**Document Version:** 1.0  
**Analysis Date:** August 8, 2025  
**Source Repository:** msc-thesis (D. Kroezen)  
**Target Application:** Quadrotor DHP with gym-pybullet-drones
