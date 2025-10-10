# SAC Drone Setup Instructions

## Prerequisites
- **Python 3.11** (strongly recommended for optimal performance)
- CUDA-capable GPU (optional, for faster training with PyTorch)
- 16GB+ RAM recommended for large-scale training

## Installation Steps

### 1. Create a conda environment (recommended)

#### Option A: Using environment.yml file (easiest)
```bash
# Create environment from the provided environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate SAC

# Verify installation
python --version  # Should show Python 3.11.x
```

#### Option B: Manual conda environment creation
```bash
# Create conda environment named "SAC" with Python 3.11
conda create -n SAC python=3.11 -y

# Activate the environment
conda activate SAC

# Verify Python version and environment
python --version  # Should show Python 3.11.x
conda info --envs  # Should show SAC environment as active
```

#### Alternative: Using pip virtual environment
```bash
# If you prefer pip virtual environments instead of conda
python3.11 -m venv sac_drone_env

# On Windows:
sac_drone_env\Scripts\activate
# On Linux/Mac:
source sac_drone_env/bin/activate
```

### 2. Install dependencies
```bash
# Make sure your SAC environment is activated
conda activate SAC

# Upgrade pip to latest version
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Optional: Install some packages via conda for better performance
# conda install numpy scipy matplotlib pandas -y
# pip install -r requirements.txt  # Then install remaining packages
```

### 3. Install gym-pybullet-drones
Since this project depends on gym-pybullet-drones, you have two options:

#### Option A: Install from local path (if you have the source code)
```bash
cd path/to/gym-pybullet-drones
pip install -e .
```

#### Option B: Install from GitHub
```bash
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
```

### 4. Verify installation
```bash
# Make sure SAC environment is activated
conda activate SAC

# Run verification script
python -c "
import gym_pybullet_drones
import stable_baselines3
import torch
import numpy as np
print('✓ All dependencies installed successfully!')
print(f'✓ Python version: {torch.version.__version__ if hasattr(torch.version, '__version__') else 'N/A'}')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ NumPy version: {np.__version__}')
"
```

## Key Dependencies Explained

- **torch**: Deep learning framework for neural networks
- **stable-baselines3**: Reinforcement learning algorithms (SAC implementation)
- **pybullet**: Physics simulation engine
- **gym-pybullet-drones**: Drone simulation environments
- **psutil**: System process utilities for CPU affinity
- **tqdm**: Progress bars for training visualization
- **matplotlib**: Plotting and visualization
- **transforms3d**: 3D transformations for drone dynamics

## Hardware Requirements

- **RAM**: 8GB+ recommended (16GB+ for large buffer sizes)
- **GPU**: CUDA-compatible GPU recommended for faster training
- **CPU**: Multi-core processor (the script uses multiple processes)

## Running the Training

```bash
# Always activate the SAC environment first
conda activate SAC

# Run the training script
python SAC_gym_pybullet.py
```

The script will:
1. Create necessary directories for logs and models
2. Set up vectorized environments with CPU affinity
3. Train the SAC agent for 4M timesteps
4. Save the best models during training
5. Perform final evaluation with GUI visualization

## Configuration

Key parameters in the script:
- `TOTAL_TIMESTEPS = 4_000_000`: Total training steps
- `NUM_ENVS = 4`: Number of parallel environments
- `EVAL_FREQ = 100_000`: Evaluation frequency
- Learning rate, buffer size, network architecture can be modified in the SAC initialization

## Output

The script generates:
- Training logs in `logs/tensorboard/SAC/Drone/`
- Best models in `logs/SAC/Drone/best_model/`
- Final trained model in `logs/final/`
- Training progress visualized with TensorBoard
