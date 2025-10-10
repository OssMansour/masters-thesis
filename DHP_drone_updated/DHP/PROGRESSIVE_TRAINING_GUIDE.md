# Progressive Training for DHP Quadrotor Control

## Overview

This document describes the progressive training curriculum system implemented to address the multi-axis control challenges in DHP quadrotor training. The system uses curriculum learning to gradually introduce complexity, starting from simple altitude control and progressing to complex trajectory following.

## Problem Background

Your original DHP training worked perfectly for hovering at `[0,0,1]` but failed when attempting to reach `[0.2,0.2,1.0]`. After thorough investigation, we verified that:

1. ✅ **Coordinate transformations are mathematically correct** - Body frame transformation using cos(-ψ), sin(-ψ)
2. ✅ **Reference generation methods are valid** - Both world frame and body frame approaches work
3. ❌ **Multi-axis training is too difficult** - Direct training for 3D position control is challenging

## Solution: Progressive Curriculum Learning

The progressive training system breaks down the complex 3D control problem into manageable phases:

### Training Phases

| Phase | Description | Episodes | Success Criteria |
|-------|-------------|----------|------------------|
| **Z-Axis** | Altitude control: `[0, 0, Z]` | 1000 | 15cm error, 70% success |
| **X-Axis** | Forward/backward: `[X, 0, Z_fixed]` | 1250 | 20cm error, 60% success |
| **Y-Axis** | Left/right: `[0, Y, Z_fixed]` | 1250 | 20cm error, 60% success |
| **XYZ Fixed** | 3D positions: `[X, Y, Z]` | 1500 | 25cm error, 50% success |
| **Trajectories** | Dynamic paths: Square, Circle, etc. | 2000 | 30cm error, 40% success |

### Difficulty Levels

Each phase has 5 difficulty levels that gradually increase the challenge:

#### Z-Axis Phase
- Level 0: Fixed altitude (1.0m)
- Level 1: Small range (0.5-1.5m)
- Level 2: Medium range (0.3-2.0m)
- Level 3: Large range (0.2-3.0m)
- Level 4: Maximum range (0.1-4.0m)

#### X-Axis & Y-Axis Phases
- Level 0: ±0.5m displacement
- Level 1: ±1.0m displacement
- Level 2: ±1.5m displacement
- Level 3: ±2.0m displacement
- Level 4: ±2.5m displacement + altitude variation

#### XYZ Fixed Points
- Level 0: Small cube (±0.5m in X,Y)
- Level 1: Medium cube (±1.0m in X,Y)
- Level 2: Large cube (±1.5m in X,Y)
- Level 3: Very large (±2.0m in X,Y)
- Level 4: Maximum (±2.5m in X,Y)

#### Dynamic Trajectories
- Level 0: Simple square (slow, large)
- Level 1: Circle trajectory
- Level 2: Figure-8 trajectory
- Level 3: Spiral trajectory
- Level 4: Complex combined trajectories

## Usage Instructions

### 1. Quick Start - Run Progressive Training

```bash
cd /home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/Hover
python train_dhp_progressive.py
```

This will:
- Start with Z-axis altitude control
- Automatically advance through phases based on performance
- Save models at phase transitions
- Generate training analysis plots
- Estimate ~7000 total episodes for full curriculum

### 2. Monitor Training Progress

The training provides detailed progress reports:

```
Episode 1234 | X-Axis Control L2 | Target: X-axis: [1.5, 0, 1.00] | Error: 0.187m | Reward: 245.3

🎯 CURRICULUM ADVANCEMENT!
   Type: LEVEL
   From: x_axis Level 1
   To: x_axis Level 2
   Reason: Achieved 62.0% success rate with 0.185m average error
   Performance: 62.0% success, 0.185m error
```

### 3. Results Analysis

Training results are saved to timestamped directories:
```
dhp_progressive_results_YYYYMMDD_HHMMSS/
├── training_stats.json          # Detailed training metrics
├── final_model.pt              # Final trained model
├── training_analysis.png       # Performance plots
└── checkpoints/               # Models saved at phase transitions
    ├── best_model_epXXXX.pt
    ├── phase_transition_epXXXX.pt
    └── ...
```

### 4. Configuration Options

You can customize the training by modifying the config in `train_dhp_progressive.py`:

```python
config = {
    'gui': False,                    # Show PyBullet GUI
    'record': False,                 # Record video
    'results_dir': 'custom_results', # Results directory
    
    # DHP Agent parameters
    'lr_actor': 1e-4,              # Actor learning rate
    'lr_critic': 1e-3,             # Critic learning rate
    'lr_model': 1e-3,              # Model learning rate
    'lambda_dhp': 0.9,             # DHP lambda parameter
    'gamma': 0.99,                 # Discount factor
    'tau': 0.005,                  # Soft update rate
    
    # Environment parameters
    'episode_length': 8.0,         # Episode length in seconds
    'dt': 1.0/30.0                 # Control frequency
}
```

## Key Implementation Details

### 1. Environment Integration

The progressive system integrates with your existing `CF2X_FastStates_HoverAviary` environment:

```python
# Set new target during training
env.set_target_position([x, y, z])
obs, info = env.reset()
```

### 2. Curriculum Management

The `ProgressiveTrainingCurriculum` class handles:
- Target generation for each phase/level
- Performance tracking and advancement decisions
- Automatic progression through difficulty levels
- Fallback mechanisms for slow learners

### 3. Advancement Criteria

Advancement occurs when:
- **Performance criteria met**: Error threshold + success rate
- **Forced advancement**: After 150% of expected episodes (prevents getting stuck)

### 4. Backwards Compatibility

The system maintains compatibility with existing training scripts:
- Environment can still be used directly
- Old training methods continue to work
- Progressive features are optional

## Expected Benefits

1. **Improved Convergence**: Gradual difficulty increase should improve learning success
2. **Better Generalization**: Training across diverse targets builds robust control
3. **Systematic Progression**: Clear phases make training progress measurable
4. **Reduced Training Time**: Early phases require fewer episodes than full 3D control

## Next Steps

1. **Run the progressive training** to see if it solves the multi-axis control problem
2. **Compare results** with direct training on `[0.2, 0.2, 1.0]`
3. **Analyze phase transitions** to identify which phases are most challenging
4. **Fine-tune parameters** based on initial results

## Troubleshooting

### Common Issues

1. **Training gets stuck in early phases**
   - Check if success thresholds are too strict
   - Monitor if agent is learning (reward should improve)
   - Consider adjusting learning rates

2. **Phase advancement too slow**
   - Reduce success rate thresholds
   - Increase position error tolerance
   - Enable forced advancement earlier

3. **Memory issues with long training**
   - Reduce replay buffer size in DHP agent
   - Clear old checkpoints periodically
   - Monitor system memory usage

### Performance Monitoring

Watch for these indicators of successful training:
- ✅ Gradual reduction in position error
- ✅ Increasing success rate within each level
- ✅ Successful phase transitions
- ✅ Smooth progression through curriculum

## Files Created

1. `progressive_training_strategy.py` - Core curriculum implementation
2. `train_dhp_progressive.py` - Progressive training script
3. `test_progressive_integration.py` - Integration tests

## Mathematical Verification Completed

✅ **Coordinate Frame Analysis**: Verified body frame transformation is mathematically correct
✅ **Reference Generation**: Both world frame and body frame methods validated
✅ **State Consistency**: Confirmed states and references use compatible coordinate systems

The progressive training system represents a systematic approach to solving the multi-axis control challenge through curriculum learning rather than attempting to fix coordinate transformation issues that were already correct.
