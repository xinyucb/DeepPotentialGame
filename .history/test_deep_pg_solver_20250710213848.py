#!/usr/bin/env python3
"""
Test script for DeepPGSolver
This script demonstrates how to use the DeepPGSolver for both flocking and aversion games.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from tqdm import tqdm

# Import the solver and parameter classes
from DeepPGSolver import FBSNN, Network
from Parameters import GameConfig

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Set default tensor type
d_type = torch.float32
torch.set_default_dtype(d_type)

def test_flocking_game():
    """Test the DeepPGSolver with a flocking game configuration."""
    print("=" * 50)
    print("Testing Flocking Game")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Game parameters
    D = 4  # number of players
    state_dim = 2  # dimension of each player's state
    game_type = "flocking"
    Q_type = "average"
    control_type = "full"
    num_of_dest = 4
    num_of_orig = 1
    
    # Create game configuration
    game = GameConfig(D, state_dim, game_type, Q_type, device)
    
    # Get initial and terminal conditions
    origin = game.get_origin(num_of_orig)
    terminal = game.get_terminal(num_of_dest)
    
    # Configure diffusion and jump parameters
    game.build_diffusion(sigma_param=0.1, sigma0_param=0.05)
    game.build_jump(gamma_param=0.1, gamma0_param=0.05)
    
    print(f"Origin shape: {origin.shape}")
    print(f"Terminal shape: {terminal.shape}")
    print(f"Q matrix:\n{game.Q}")
    
    # Set up control parameters
    if control_type == "distributed":
        input_param = 1
    else:
        input_param = D
    
    # Define all parameters
    params = {
        "equation": {
            "mu": game.b, 
            "sigma": game.S, 
            "sigma0": game.S0, 
            "alpha": game.Gamma, 
            "beta": game.Gamma0,
            "delta": [2] + [1] * (D-1), 
            "theta": [0.8] + [0.7] * (D-1), 
            "lambda": [0.3] + [0.2] * (D-1),
            "lambda0": 0.25, 
            "D": D, 
            "T": 1, 
            "M": 100,  # Reduced for testing
            "N": 25,   # Reduced for testing
            "jump_dim": game.jump_dim, 
            "BM_dim": game.BM_dim, 
            "u_dim": game.u_dim, 
            "state_dim": state_dim,
            "Xi": origin.clone().detach().float().to(device),
            "Yi": torch.tensor(np.zeros([state_dim * D, 1])).float().to(device),
            "terminal": terminal, 
            "origin": origin, 
            "num_of_dest": num_of_dest, 
            "num_of_orig": num_of_orig,
            "Q": game.Q, 
            "Q_type": Q_type,
            "game_type": game_type, 
            "control_type": control_type,
            "fi_param_u": 0.1,
            "fi_param_x": 1,
            "f_sigma_square": 100, 
            "gi_param": 40,
        },
        "train": {
            "lr_actor": 1e-3, 
            "gamma_actor": 0.7, 
            "milestones_actor": [30000, 40000],
            "iteration": 3,  # Reduced for testing
            "epochs": 10,    # Reduced for testing
            "actor_step": 100, 
            "device": device,
        },
        "net": {
            "inputs": 2 * input_param * state_dim + 1, 
            "width": state_dim * input_param + 10, 
            "depth": 4, 
            "output": game.u_dim, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {
                "Tanh": nn.Tanh(), 
                "Tanhshrink": nn.Tanhshrink(), 
                "ReLU": nn.ReLU(), 
                "ReLU6": nn.ReLU6()
            },
        }
    }
    
    # Create control networks for all players
    net_control = [Network(params["net"]) for _ in range(D)]
    
    # Create and train the model
    model = FBSNN(net_control, params).to(device)
    print(f"Model filename: {model.filename}")
    
    # Train the model
    print("Starting training...")
    model.train_players()
    
    # Generate trajectory plots
    print("Generating trajectory plots...")
    if state_dim == 1:
        model.generate_1d_trajectory()
    else:
        model.generate_2d_trajectory(save=True)
    
    # Save the trained model
    model.save_NN()
    
    print("Flocking game test completed!")
    return model

def test_aversion_game():
    """Test the DeepPGSolver with an aversion game configuration."""
    print("=" * 50)
    print("Testing Aversion Game")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Game parameters
    D = 4  # number of players
    state_dim = 1  # dimension of each player's state
    game_type = "aversion"
    Q_type = "average"
    control_type = "full"
    num_of_dest = 1
    num_of_orig = 1
    
    # Create game configuration
    game = GameConfig(D, state_dim, game_type, Q_type, device)
    
    # Get initial and terminal conditions
    origin = game.get_origin(num_of_orig)
    terminal = game.get_terminal(num_of_dest)
    
    # Configure diffusion and jump parameters
    game.build_diffusion(sigma_param=0.1, sigma0_param=0.05)
    game.build_jump(gamma_param=0.1, gamma0_param=0.05)
    
    print(f"Origin shape: {origin.shape}")
    print(f"Terminal shape: {terminal.shape}")
    print(f"Q matrix:\n{game.Q}")
    
    # Set up control parameters
    if control_type == "distributed":
        input_param = 1
    else:
        input_param = D
    
    # Define all parameters
    params = {
        "equation": {
            "mu": game.b, 
            "sigma": game.S, 
            "sigma0": game.S0, 
            "alpha": game.Gamma, 
            "beta": game.Gamma0,
            "delta": [2] + [1] * (D-1), 
            "theta": [0.8] + [0.7] * (D-1), 
            "lambda": [0.3] + [0.2] * (D-1),
            "lambda0": 0.25, 
            "D": D, 
            "T": 1, 
            "M": 100,  # Reduced for testing
            "N": 25,   # Reduced for testing
            "jump_dim": game.jump_dim, 
            "BM_dim": game.BM_dim, 
            "u_dim": game.u_dim, 
            "state_dim": state_dim,
            "Xi": origin.clone().detach().float().to(device),
            "Yi": torch.tensor(np.zeros([state_dim * D, 1])).float().to(device),
            "terminal": terminal, 
            "origin": origin, 
            "num_of_dest": num_of_dest, 
            "num_of_orig": num_of_orig,
            "Q": game.Q, 
            "Q_type": Q_type,
            "game_type": game_type, 
            "control_type": control_type,
            "fi_param_u": 0.1,
            "fi_param_x": 100,
            "f_sigma_square": 100, 
            "gi_param": 0,
        },
        "train": {
            "lr_actor": 1e-3, 
            "gamma_actor": 0.7, 
            "milestones_actor": [30000, 40000],
            "iteration": 3,  # Reduced for testing
            "epochs": 10,    # Reduced for testing
            "actor_step": 100, 
            "device": device,
        },
        "net": {
            "inputs": 2 * input_param * state_dim + 1, 
            "width": state_dim * input_param + 10, 
            "depth": 4, 
            "output": game.u_dim, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {
                "Tanh": nn.Tanh(), 
                "Tanhshrink": nn.Tanhshrink(), 
                "ReLU": nn.ReLU(), 
                "ReLU6": nn.ReLU6()
            },
        }
    }
    
    # Create control networks for all players
    net_control = [Network(params["net"]) for _ in range(D)]
    
    # Create and train the model
    model = FBSNN(net_control, params).to(device)
    print(f"Model filename: {model.filename}")
    
    # Train the model
    print("Starting training...")
    model.train_players()
    
    # Generate trajectory plots
    print("Generating trajectory plots...")
    if state_dim == 1:
        model.generate_1d_trajectory()
    else:
        model.generate_2d_trajectory(save=True)
    
    # Save the trained model
    model.save_NN()
    
    print("Aversion game test completed!")
    return model

def test_simple_configuration():
    """Test with a simple configuration for quick validation."""
    print("=" * 50)
    print("Testing Simple Configuration")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple parameters for quick test
    D = 2  # fewer players
    state_dim = 1  # 1D state
    game_type = "flocking"
    Q_type = "average"
    control_type = "full"
    num_of_dest = 2
    num_of_orig = 1
    
    # Create game configuration
    game = GameConfig(D, state_dim, game_type, Q_type, device)
    
    # Get initial and terminal conditions
    origin = game.get_origin(num_of_orig)
    terminal = game.get_terminal(num_of_dest)
    
    # Configure diffusion and jump parameters
    game.build_diffusion(sigma_param=0.1, sigma0_param=0.05)
    game.build_jump(gamma_param=0.1, gamma0_param=0.05)
    
    print(f"Origin shape: {origin.shape}")
    print(f"Terminal shape: {terminal.shape}")
    print(f"Q matrix:\n{game.Q}")
    
    # Set up control parameters
    if control_type == "distributed":
        input_param = 1
    else:
        input_param = D
    
    # Define all parameters with minimal training
    params = {
        "equation": {
            "mu": game.b, 
            "sigma": game.S, 
            "sigma0": game.S0, 
            "alpha": game.Gamma, 
            "beta": game.Gamma0,
            "delta": [2] + [1] * (D-1), 
            "theta": [0.8] + [0.7] * (D-1), 
            "lambda": [0.3] + [0.2] * (D-1),
            "lambda0": 0.25, 
            "D": D, 
            "T": 1, 
            "M": 50,   # Very small for quick test
            "N": 10,   # Very small for quick test
            "jump_dim": game.jump_dim, 
            "BM_dim": game.BM_dim, 
            "u_dim": game.u_dim, 
            "state_dim": state_dim,
            "Xi": origin.clone().detach().float().to(device),
            "Yi": torch.tensor(np.zeros([state_dim * D, 1])).float().to(device),
            "terminal": terminal, 
            "origin": origin, 
            "num_of_dest": num_of_dest, 
            "num_of_orig": num_of_orig,
            "Q": game.Q, 
            "Q_type": Q_type,
            "game_type": game_type, 
            "control_type": control_type,
            "fi_param_u": 0.1,
            "fi_param_x": 1,
            "f_sigma_square": 100, 
            "gi_param": 40,
        },
        "train": {
            "lr_actor": 1e-3, 
            "gamma_actor": 0.7, 
            "milestones_actor": [30000, 40000],
            "iteration": 1,  # Very small for quick test
            "epochs": 5,     # Very small for quick test
            "actor_step": 100, 
            "device": device,
        },
        "net": {
            "inputs": 2 * input_param * state_dim + 1, 
            "width": state_dim * input_param + 10, 
            "depth": 2,  # Smaller network for quick test
            "output": game.u_dim, 
            "activation": "ReLU", 
            "penalty": "Tanh",
            "params_act": {
                "Tanh": nn.Tanh(), 
                "Tanhshrink": nn.Tanhshrink(), 
                "ReLU": nn.ReLU(), 
                "ReLU6": nn.ReLU6()
            },
        }
    }
    
    # Create control networks for all players
    net_control = [Network(params["net"]) for _ in range(D)]
    
    # Create and train the model
    model = FBSNN(net_control, params).to(device)
    print(f"Model filename: {model.filename}")
    
    # Train the model
    print("Starting training...")
    model.train_players()
    
    # Generate trajectory plots
    print("Generating trajectory plots...")
    if state_dim == 1:
        model.generate_1d_trajectory()
    else:
        model.generate_2d_trajectory(save=True)
    
    # Save the trained model
    model.save_NN()
    
    print("Simple configuration test completed!")
    return model

def main():
    """Main function to run all tests."""
    print("DeepPGSolver Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Simple configuration (quick test)
        print("\nRunning simple configuration test...")
        model1 = test_simple_configuration()
        
        # Test 2: Flocking game
        print("\nRunning flocking game test...")
        model2 = test_flocking_game()
        
        # Test 3: Aversion game
        print("\nRunning aversion game test...")
        model3 = test_aversion_game()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 