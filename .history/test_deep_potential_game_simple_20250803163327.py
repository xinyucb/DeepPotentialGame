#!/usr/bin/env python3
"""
Simple Deep Potential Game Solver Test
Based on Deep_Potential_Game_Solver.ipynb

This is a simplified version for quick testing with reduced training iterations.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import time
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from DeepPGSolver import deepPG, Network
from Parameters import GameConfig

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Set default tensor type
d_type = torch.float32
torch.set_default_dtype(d_type)
import warnings
warnings.filterwarnings("ignore")



def main():
    """Main function to run the simplified Deep Potential Game solver test."""
    
    print("=" * 50)
    print("Flocking Game Test")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================================================
    # Game Parameters Setup 
    # ============================================================================
    D = 4  # number of players 
    state_dim = 2  # state dimension 
    game_type = ["flocking", "aversion"][0]  # aversion
    Q_type = ["average", "twogroups", "random"][0]  # average 
    control_type = "full" 
    
    num_of_dest = 4
    num_of_orig = 1
    
    
    # ============================================================================
    # Create Game Configuration
    # ============================================================================
    game = GameConfig(D, state_dim, game_type, Q_type, device)
    
    # Get initial and terminal conditions
    origin = game.get_origin(num_of_orig) * 0
    terminal = game.get_terminal(num_of_dest) 
    
    # Configure diffusion and jump parameters
    sigma, sigma0, gamma, gamma0 = 0.1, 0, 0.1, 0.
    game.build_diffusion(sigma_param=sigma, sigma0_param=sigma0)
    game.build_jump(gamma_param=gamma, gamma0_param=gamma0)
    

    
    # ============================================================================
    # Set up control parameters
    # ============================================================================
    if control_type == "distributed": 
        input_param = 1
    else: 
        input_param = D
    
   
    
    # ============================================================================
    # Define all parameters (reduced for faster testing)
    # ============================================================================
    params = {
        # Parameters of the game setup
        "equation": {
            "mu": game.b, "sigma": game.S, "sigma0": game.S0, "alpha": game.Gamma, "beta": game.Gamma0,
            "lambda": [0.3] + [0.2] * (D-1),
            "lambda0": 0.25, "D": D, "T": 1, "M": 500, "N": 50,  
            "jump_dim": game.jump_dim, "BM_dim": game.BM_dim, "u_dim": game.u_dim, "state_dim": state_dim,
            "Xi": torch.tensor(origin.clone().detach()).float().to(device),
            "Yi": torch.tensor(np.zeros([state_dim * D, 1])).float().to(device),
            "terminal": terminal, "origin": origin, "num_of_dest": num_of_dest, "num_of_orig": num_of_orig,
            "Q": game.Q, "Q_type": Q_type,
            "game_type": game_type, "control_type": control_type, "random_initial_state": False,
            "fi_param_u": 0.1,
            "fi_param_x": 1, "f_sigma_square": 100, #only used in aversion game with Gaussian kernel
            "gi_param": 40,
        },
        "train": {
            "lr_actor": 1e-3, "gamma_actor": 0.7, "milestones_actor": [30000, 40000],
            "iteration": 20, "epochs": 50, "device": device,  # reduced iterations and epochs
        },
        "net": {
            "inputs": 2 * input_param * state_dim + 1, "width": input_param * state_dim * 1 + 10, "depth": 4, 
            "output": game.u_dim, 
            "activation": "ReLU", "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
        },
    }
    
    
    # ============================================================================
    # Create control networks
    # ============================================================================
    params_net =  params["net"]
    net_control = [Network(params_net) for _ in range(D)]  # u=(u1, u2, ..., uD)
    
    # ============================================================================
    # Create and configure the model
    # ============================================================================
    model = deepPG(net_control, params).to(device)
    
    # Update filename with diffusion parameters
    model.filename = model.filename + f"sigma{sigma}_sigma0{sigma0}_gamma{gamma}_gamma0{gamma0}".replace(".", "_")
    
    print(f"Model filename: {model.filename}")
    
    # ============================================================================
    # Train the model
    # ============================================================================
    print("\nStarting training (simplified)...")
    model.train_players()
    
    # ============================================================================
    # Generate trajectories
    # ============================================================================
    print("\nGenerating trajectories...")
    if state_dim == 1:
        _ = model.generate_1d_trajectory(plot_mean=True)
    else:
        ind, X, ber = model.generate_2d_trajectory(plot_mean=False, save=False, filename=model.filename + f"sample{1}")
        model.generate_animation()
    # ============================================================================
    # Save the model
    # ============================================================================
    print("\nSaving model...")
    model.save_NN()
    
    # ============================================================================
    # Plot and save results
    # ============================================================================
    print("\nPlotting results...")
    val_loss = np.array([t.numpy() for t in model.training_cost_list])
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss)
    plt.title("Training Loss (Simplified Test)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("outputLoss/loss" + model.filename + ".png", dpi=300)
    plt.show()
    
    # Save loss data to CSV
    df = pd.DataFrame(val_loss, columns=["val"])
    df.to_csv(f'loss_csv/{model.filename}_val_loss.csv', index=False)
    
    print(f"\nFlocking game test completed.")
    print(f"Model saved as: outputNN/{model.filename}")
    print(f"Loss plot saved as: output/loss{model.filename}.png")
    print(f"Loss data saved as: outputLoss/{model.filename}_val_loss.csv")

if __name__ == "__main__":
    main() 