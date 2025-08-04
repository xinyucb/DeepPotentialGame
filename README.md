# Deep Potential Game Solver

A Python implementation of deep learning-based solvers for potential games, specifically designed for multi-agent systems including flocking and aversion games.

Paper:  Xin Guo, Xinyu Li, Yufei Zhang, Distributed Games with Jumps: an α-Potential Game Approach, 2025.

## Contact

For questions and support, please contact xinyu_li@berkeley.edu. 

## Overview

This project implements a deep potential game solver using PyTorch for solving multi-agent control problems. The solver can handle various game types including:

- **Flocking Games**: Multi-agent systems where agents coordinate to achieve collective behavior
- **Aversion Games**: Multi-agent systems where agents try to avoid each other while reaching targets

## Features

- **Multi-agent Control**: Supports D-player systems with configurable state dimensions
- **Deep Neural Networks**: Uses PyTorch-based neural networks for control policy approximation
- **Multiple Game Types**: Supports flocking, aversion, and other potential game formulations
- **Visualization**: Generates trajectory plots and animations
- **Model Persistence**: Save and load trained models for reuse
- **GPU Support**: Compatible with CUDA for accelerated training

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, for acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Deep_alpha_Potential_Game_code
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Deep_alpha_Potential_Game_code/
├── DeepPGSolver.py          # Main solver implementation
├── Parameters.py            # Game configuration and parameters
├── main_test_flocking_game.py    # Flocking game test script
├── main_test_aversion_game.py    # Aversion game test script
├── requirements.txt         # Python dependencies
├── output/                 # Generated trajectory plots
├── outputLoss/             # Training loss data
├── outputNN/               # Saved neural network models
└── video/                  # Generated animation videos
```

## Usage

### Running Flocking Game

```bash
python main_test_flocking_game.py
```

This will:
- Train a 4-player 2d flocking game solver
- Generate trajectory visualizations
- Save the trained model
- Create animation videos

### Running Aversion Game

```bash
python main_test_aversion_game.py
```

This will:
- Train a 4-player 2d aversion game solver
- Generate trajectory visualizations
- Save the trained model
- Create animation videos

## Configuration

### Game Parameters

The main configuration parameters are defined in the test scripts:

- **D**: Number of players (default: 4)
- **state_dim**: State dimension (default: 2)
- **game_type**: Type of game ("flocking" or "aversion")
- **Q_type**: Interaction matrix type ("average", "twogroups", "random")
- **control_type**: Control strategy ("full")

### Training Parameters

- **iteration**: Number of training iterations
- **epochs**: Number of epochs per iteration
- **lr_actor**: Learning rate for actor networks
- **device**: Training device (CPU/GPU)

### Network Architecture

- **depth**: Number of network layers
- **width**: Number of neurons per layer
- **activation**: Activation function (ReLU, Tanh, etc.)

## Output Files

### Generated Files

- **Trajectory Plots**: PNG files showing agent trajectories
- **Loss Data**: CSV files with training loss history
- **Model Files**: PyTorch (.pth) files containing trained networks
- **Videos**: MP4 files with animated trajectories

### File Naming Convention

Files are named with parameters for easy identification:
```
trajectory_{game_type}_full_D{D}_d{state_dim}_dest{num_dest}_orig{num_orig}_{Q_type}_fiu{fi_param_u}_fix{fi_param_x}_gi{gi_param}_Depth{depth}sigma{sigma}_sigma0{sigma0}_gamma{gamma}_gamma0{gamma0}
```

## Examples

### Basic Flocking Game

```python
from DeepPGSolver import deepPG, Network
from Parameters import GameConfig

# Configure game
D = 4
state_dim = 2
game_type = "flocking"
Q_type = "average"

# Create game configuration
game = GameConfig(D, state_dim, game_type, Q_type, device)

# Set up parameters and train
# (See main_test_flocking_game.py for complete example)
```

### Custom Game Configuration

```python
# Custom interaction matrix
Q_type = "twogroups"  # Two distinct groups

# Custom diffusion parameters
sigma, sigma0 = 0.2, 0.1
gamma, gamma0 = 0.15, 0.05

game.build_diffusion(sigma_param=sigma, sigma0_param=sigma0)
game.build_jump(gamma_param=gamma, gamma0_param=gamma0)
```

## Dependencies

### Core Dependencies

- **torch==1.13.0**: PyTorch for deep learning
- **numpy==1.22.4**: Numerical computing
- **pandas==1.4.4**: Data manipulation
- **matplotlib==3.5.1**: Plotting and visualization
- **tqdm==4.67.1**: Progress bars
- **scipy==1.7.3**: Scientific computing
- **scikit-learn==1.0.2**: Machine learning utilities

## Performance

### Training Time

- **Flocking Game**: ~5-10 minutes (20 iterations, 50 epochs)
- **Aversion Game**: ~5-10 minutes (20 iterations, 50 epochs)

### Memory Requirements

- **CPU**: 4-8 GB RAM
- **GPU**: 2-4 GB VRAM (recommended)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check file paths and model compatibility
3. **Plot Generation Issues**: Ensure matplotlib backend is properly configured

### GPU Support

To enable GPU acceleration:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


