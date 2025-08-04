#!/usr/bin/env python3
"""
Simple example demonstrating how to use DeepPGSolver
This script shows the basic setup and usage pattern.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

# Import the solver and parameter classes
from DeepPGSolver import FBSNN, Network
from Parameters import GameConfig

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Set default tensor type
torch.set_default_dtype(torch.float32)

def main():
    """Simple example of using DeepPGSolver."""
    
    print("DeepPGSolver Simple Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================================================
    # Step 1: Define game parameters
    # ============================================================================
    D = 2  # number of players
    state_dim = 1  # dimension of each player's state
    game_type = "flocking"  # or "aversion"
    Q_type = "average"
    control_type = "full"
    num_of_dest = 2
    num_of_orig = 1
    
    print(f"Game parameters:")
    print(f"  Players: {D}")
    print(f"  State dimension: {state_dim}")
    print(f"  Game type: {game_type}")
    print(f"  Control type: {control_type}")
    
    # ============================================================================
    # Step 2: Create game configuration
    # ============================================================================
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
    
    # ============================================================================
    # Step 3: Set up control parameters
    # ============================================================================
    if control_type == "distributed":
        input_param = 1
    else:
        input_param = D
    
    # ============================================================================
    # Step 4: Define all parameters
    # ============================================================================
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
            "M": 50,   # number of trajectories
            "N": 10,   # number of time steps
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
            "iteration": 1,  # number of training iterations
            "epochs": 5,     # number of epochs per iteration
            "actor_step": 100, 
            "device": device,
        },
        "net": {
            "inputs": 2 * input_param * state_dim + 1, 
            "width": state_dim * input_param + 10, 
            "depth": 2,  # network depth
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
    
    print(f"Network parameters:")
    print(f"  Inputs: {params['net']['inputs']}")
    print(f"  Width: {params['net']['width']}")
    print(f"  Depth: {params['net']['depth']}")
    print(f"  Output: {params['net']['output']}")
    
    # ============================================================================
    # Step 5: Create control networks for all players
    # ============================================================================
    net_control = [Network(params["net"]) for _ in range(D)]
    print(f"Created {D} control networks")
    
    # ============================================================================
    # Step 6: Create and train the model
    # ============================================================================
    model = FBSNN(net_control, params).to(device)
    print(f"Model filename: {model.filename}")
    
    # Train the model
    print("\nStarting training...")
    model.train_players()
    
    # ============================================================================
    # Step 7: Generate results
    # ============================================================================
    print("\nGenerating trajectory plots...")
    if state_dim == 1:
        model.generate_1d_trajectory()
    else:
        model.generate_2d_trajectory(save=True)
    
    # Save the trained model
    model.save_NN()
    
    print("\nExample completed successfully!")
    print("Check the 'output/' directory for plots and 'outputNN/' for saved models.")

if __name__ == "__main__":
    main() 