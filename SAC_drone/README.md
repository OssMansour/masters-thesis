# SAC Drone Training System

This directory contains all SAC (Soft Actor-Critic) related components for drone training, separated from the DHP (Dual Heuristic Programming) system in trial2.

## 📁 Directory Structure

```
SAC_drone/
├── train_sac_drone.py          # Main SAC training script
├── cf2x_drone_env.py           # CF2X environment optimized for SAC
├── sac_trained_models/         # Trained SAC models and metadata
├── sac_training_logs/          # Training logs and metrics
└── README.md                   # This file
```

## 🚀 Main Components

### `train_sac_drone.py`
- **Purpose**: Complete SAC training pipeline for CF2X quadrotor
- **Features**: 
  - State+reference concatenation (16 inputs) to match DHP comparison
  - Performance tracking and best episode recording
  - Comprehensive logging and visualization
  - Policy demonstration capabilities
- **Class**: `DroneQuadrotorSACTrainer`

### `cf2x_drone_env.py`
- **Purpose**: CF2X environment wrapper optimized for SAC training
- **Features**:
  - Fast states extraction (8 elements): [z, roll, pitch, yaw, vz, wx, wy, wz]
  - DHP-compatible interface for fair algorithm comparison
  - Proper multirotor control structure with PID controllers
  - Reference signal generation
- **Class**: `CF2X_Drone_HoverAviary`

### `sac_trained_models/`
- **Contents**: Pre-trained SAC models, configurations, and metadata
- **Files**:
  - `sac_*_best.zip` - Best performing models
  - `sac_*_final.zip` - Final trained models
  - `sac_*_config.pkl` - Training configurations
  - `sac_*_metadata.pkl` - Training metadata and performance metrics

### `sac_training_logs/`
- **Contents**: Detailed training logs, metrics, and performance data
- **Files**:
  - `sac_drone_training_*.log` - Detailed training logs
  - `sac_drone_metrics_*.json` - Performance metrics in JSON format
  - Historical training data from different experiments

## 🎯 Usage

### Training a New SAC Model
```bash
cd /home/osos/Mohamed_Masters_Thesis/SAC_drone
python train_sac_drone.py
```

### Key Configuration Options
```python
config_override = {
    'total_timesteps': 600000,           # Training duration
    'learning_rate': 3e-4,               # SAC learning rate
    'buffer_size': 50000,                # Replay buffer size
    'batch_size': 128,                   # Training batch size
    'policy_architecture': [64, 64],     # Policy network layers
    'q_network_architecture': [64, 64],  # Q-network layers
    'gui': False,                        # Training visualization
    'record': False,                     # Video recording
}
```

## 📊 Performance Comparison with DHP

The SAC system is designed for direct comparison with the DHP system:

- **Input Structure**: Both receive 16 inputs (8 states + 8 references)
- **Action Space**: Both control 4 motor RPMs
- **Environment**: Same CF2X dynamics and physics
- **Success Metrics**: Position error, tracking performance, stability

## 🔧 Dependencies

- `stable-baselines3[extra]` - SAC implementation
- `gymnasium` - Environment interface
- `numpy`, `matplotlib` - Numerical computing and visualization
- `gym-pybullet-drones` - Drone simulation environment

## 📈 Expected Performance

- **Convergence**: ~500-1000 episodes for basic hovering
- **Position Error**: Target < 0.5m for good performance
- **Training Time**: ~30-60 minutes on CPU
- **Best Performance**: Typically achieves 0.1-0.3m position error

## 🚀 Integration Notes

This SAC system can be easily integrated back into any project requiring SAC-based drone control. The modular design allows for:

- Independent training and evaluation
- Easy hyperparameter tuning
- Clear separation from other algorithms
- Reusable environment and training components

## 📝 Author

DHP vs SAC Comparison Study  
Date: August 13, 2025

## 🔄 Migration from trial2

This directory was created by moving all SAC-related components from `/trial2/` to maintain clean separation between DHP and SAC systems while preserving all training data and model artifacts.
