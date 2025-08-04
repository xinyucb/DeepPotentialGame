import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.animation import FuncAnimation

import numpy as np
import time

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import gridspec

class Block(nn.Module):

    def __init__(self, inputs: int, params: dict, activation="Tanh"):
        super(Block, self).__init__()
        self.L1 = nn.Linear(inputs, inputs)
        self.L2 = nn.Linear(inputs, inputs)
        self.activation = activation
        if activation in params:
            self.act = params[activation]

    def forward(self, x):
        if self.activation == "Sin" or self.activation == "sin":
            a = torch.sin(self.L2(torch.sin(self.L1(x)))) + x
        else:
            a = self.act(self.L1(self.act(self.L2(x)))) + x
        return a


class Network(nn.Module):

    def __init__(self, params, penalty=None):
        super(Network, self).__init__()
        self.params = params
        self.first = nn.Linear(self.params["inputs"], self.params["width"])
        self.last = nn.Linear(self.params["width"], self.params["output"])
        self.network = nn.Sequential(*[
            self.first,
            *[Block(self.params["width"], self.params["params_act"], self.params["activation"])] * self.params["depth"],
            self.last
        ])
        self.penalty = penalty
        self.bound = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        if self.penalty is not None:
            if self.penalty == "Sigmoid":
                sigmoid = nn.Sigmoid()
                return sigmoid(self.network(x)) * self.bound
            elif self.penalty == "Tanh":
                tanh = nn.Tanh()
                return tanh(self.network(x)) * self.bound
            else:
                raise RuntimeError("Other penalty has not bee implemented!")
        else:
            return self.network(x)


class FBSNN(nn.Module):  # Forward-Backward Stochastic Neural Network
    def __init__(self, net_control, params):
        super().__init__()

        # Save equation, training, and network parameters
        self.params_equ = params["equation"]
        self.params_train = params["train"]
        self.params_net = params["net"]

        self.device = self.params_train["device"]  # Device (CPU/GPU)


        # Problem-specific constants
        self.Xi = self.params_equ["Xi"]    # Initial point of state dynamics (starting state)
        self.Yi = self.params_equ["Yi"]    # Initial point of sensitivity process (starting state)
        self.T = self.params_equ["T"]      # Terminal time (end of time horizon)
        self.M = self.params_equ["M"]      # Number of trajectories (samples/paths)
        self.N = self.params_equ["N"]      # Number of time snapshots (time steps)
        self.D = self.params_equ["D"]      # Number of players (for multi-agent problems)
        self.Q = torch.from_numpy(self.params_equ["Q"]).float().to(self.device)     # Weight matrix
        self.state_dim = self.params_equ["state_dim"]
        self.bm_dim = self.params_equ["BM_dim"]
        self.u_dim = self.params_equ["u_dim"]
        self.jump_dim =self.params_equ["jump_dim"]

        # Create a multiprocessing manager to hold shared objects
        manager = mp.Manager()
        self.dict = manager.dict()  # Shared dict to store networks, optimizers, schedulers
        self.val_loss = np.inf
        self.game_type, self.control_type,self.random_initial_state = self.params_equ["game_type"], self.params_equ["control_type"], self.params_equ["random_initial_state"]
        self.sigma_square = self.params_equ["f_sigma_square"]
        self.fi_param_u = self.params_equ["fi_param_u"]
        self.fi_param_x = self.params_equ["fi_param_x"]
         
        # For each player (or agent):
        for i in range(self.D):
            # Store control network weights
            self.dict[f'net_actor_{i}'] = net_control[i].state_dict()

            # Create optimizer for the actor
            self.dict[f'optimizer_actor_{i}'] = optim.Adam(
                net_control[i].parameters(), # network structure
                lr=self.params_train["lr_actor"]
            )

            # Learning rate scheduler for actor optimizer
            self.dict[f'scheduler_actor_{i}'] = optim.lr_scheduler.MultiStepLR(
                self.dict[f'optimizer_actor_{i}'],
                milestones=self.params_train["milestones_actor"],
                gamma=self.params_train["gamma_actor"]
            )

        # Terminal condition of the game/dynamics (e.g., terminal cost)
        self.terminal = self.params_equ["terminal"].to(self.device)
        self.origin = self.params_equ["origin"].to(self.device)
        self.num_of_dest = self.params_equ["num_of_dest"]
        self.num_of_orig = self.params_equ["num_of_orig"]
        # Lists to log errors and costs during training
        self.Y_error_list = []
        self.control_error_list = []
        self.training_cost_list = []

        # Time step size
        self.dt = self.T / self.N
        
        

        filename= f"_{self.game_type}_{self.control_type}_D{self.D}_d{self.state_dim}_dest{self.num_of_dest}"
        filename+=f"_orig{self.num_of_orig}_{self.params_equ['Q_type']}_fiu{self.fi_param_u}_fix{self.fi_param_x}_gi{self.params_equ['gi_param']}_Depth{self.params_net['depth']}"
        if self.game_type=="aversion": filename += f"_fsigmasq{self.sigma_square}"
        filename = filename.replace(".","_")
        self.filename = filename


    def generate_p_dim_poisson_jumps(self, dim, scale, max_jumps=30):
        """
        Generate jump times for p independent Poisson processes with the same intensity lambda,
        for M trajectories up to terminal time T.

        Inputs:
        - dim: dimension of the Poisson process (number of independent Poisson components)
            For individual noises, dim = D * jump_dim
            For common noise, dim = jump_dim
        - scale: intensity rate for each dimensions
            For individual: scale=[1 / self.params_equ["lambda"][i] for i in range(self.D)]
                            scale = np.repeat(np.array(scale), p)
            For common: scale = 1/self.parames_equ["lambda0]
        - max_jumps: max number of jumps to generate per trajectory and dimension (buffer)

        Outputs:
        - P: numpy array of shape (M, max_jumps, Dp) with jump times, zeros where no jump (after T)
        """

        # Sample exponential interarrival times: shape (M, max_jumps, p)
        exp_samples = np.random.exponential(scale=scale, size=(self.M, max_jumps, dim))
        # Cumulative sum along the jumps axis to get jump times
        P = np.cumsum(exp_samples, axis=1)
        # Zero out jump times that exceed terminal time T
        P[P > self.T] = 0
        P[np.where(P > self.T)] = 0
        count0 = np.count_nonzero(P, axis=1).max()
        P = P[:, 0:count0]
        return P # (Batch, ?, dim)


    def fetch_minibatch(self):
        """
        Sample mini-batch of stochastic processes for M trajectories over N time steps:
            Brownian motions of each player, common Brownian motion,
            Possion process of each player, common Poisson process

        Inputs:
        - None (samples from internal parameters and random generators).

        Outputs: t, W, B, P, P0
        - t: Tensor (M, N+1, 1) of cumulative time grid.
        - W: Tensor (M, N+1, D*bm_dim) of idiosyncratic Brownian motions for each player.
        - B: Tensor (M, N+1, bm_dim) of common Brownian motion.
        - P: Tensor (M, ?, D) of jump times for individual Poisson processes (per player).
        - P0: Tensor (M, ?, 1) of jump times for common Poisson process.

        Note:
        - No control inputs involved; only noise processes are sampled.
        - Poisson jump counts are dynamically truncated based on jump times.
        """

        # Initialize empty arrays for time steps (Dt), Brownian increments (DW, DB)
        # Dt: M trajectories, N+1 time points (including t=0), 1-dim time step at each point
        Dt = np.zeros((self.M, self.N + 1, 1))
        DW = np.zeros((self.M, self.N + 1, self.D * self.bm_dim))  # individual Brownian motion per player
        DB = np.zeros((self.M, self.N + 1, self.bm_dim))       # common noise (shared Brownian motion)

        # Set constant dt for time steps except the initial point (t=0)
        Dt[:, 1:, :] = self.dt

        # Sample Brownian increments ~ N(0, dt)
        DW[:, 1:, :] = np.sqrt(self.dt) * np.random.normal(size=(self.M, self.N, self.D * self.bm_dim))
        DB[:, 1:, :] = np.sqrt(self.dt) * np.random.normal(size=(self.M, self.N, self.bm_dim))

        # Integrate to get actual Brownian paths and time grid
        t = np.cumsum(Dt, axis=1)    # cumulative sum of time steps → time at each node
        W = np.cumsum(DW, axis=1)    # cumulative sum of Brownian increments → Brownian paths
        B = np.cumsum(DB, axis=1)    # common noise Brownian paths

        # Simulate jump times for individual Poisson processes (all players inside one matrix)
        scale=[1 / self.params_equ["lambda"][i] for i in range(self.D)]
        scale = np.repeat(np.array(scale), self.jump_dim)
        P = self.generate_p_dim_poisson_jumps(self.D * self.jump_dim, scale)

        # Simulate jump times for common Poisson process
        P0 = self.generate_p_dim_poisson_jumps(self.jump_dim, 1/self.params_equ["lambda0"])

        # Track which trajectories actually had jumps (for potential later use)
        self.jump_index = torch.unique(torch.nonzero(torch.tensor(P))[:, 0])
        self.jump_index0 = torch.unique(torch.nonzero(torch.tensor(P0))[:, 0])

        # Return tensors on the correct device
        # B is repeated along last axis to match dimensionality with W (D components)
        return (
            torch.from_numpy(t).float().to(self.device),                     # time grid (M x N+1 x 1)
            torch.from_numpy(W).float().to(self.device),                     # Brownian motion (M x N+1 x D)
            torch.from_numpy(B).float().to(self.device),# common noise (M x N+1 x D)
            torch.from_numpy(P).float().to(self.device),                     # individual Poisson times (M x ? x Dp)
            torch.from_numpy(P0).float().to(self.device)                     # common Poisson times (M x ? x p)
        )


    def loss_function(
        self, net_actor, t,
        W, B, P, P0, X0, Y0, # noise trajectory and initial point for all the players
        i, n, bernoulli_M=None):
        """
        Compute the loss for the critic and actor networks over a batch of trajectories.

        Inputs:
        - net_critic: list of neural nets approximating value function and its gradient.
        - net_actor: list of neural nets producing control given [u1(t, X, Y),...,uD(t,X,Y)].
        - t: Tensor (M, N+1, 1), time grid for each sample path.
        - W: Tensor (M, N+1, D), Brownian motion paths (idiosyncratic noise).
        - B: Tensor (M, N+1, D), Brownian motion paths (common noise).
        - P: Tensor (M, ?, Dp), jump times for Poisson processes (idiosyncratic jumps).
        - P0: Tensor (M, ?, p), jump times for Poisson process (common jumps).
        - X0: Tensor (M, state_dim), initial state for all the players.
        - i: int, starting time index for simulation window. (i=0 in the gradient descent step)
        - n: int, number of time steps to simulate in this window. (n=self.N in the gradient descent step)
        - k: int, player/index for which loss is computed.
        - bernoulli_M: (M,), Bernoulli random variable to decide the position of leader's target in leader_follower game
        Returns:
        - loss_actor: scalar tensor of actor loss.
        - X: Tensor of simulated state trajectories (M, n+1, state_dim).
        """


        # Initialize loss and buffers
        X_buffer, Y_buffer = [], []                 # to store states (X) and value estimates (Y) along path

        # Initial time and noise values
        t0, W0, B0 = t[:, i, :], W[:, i, :], B[:, i, :]

        # Require gradients for X0 for potential derivative computations
        X0.requires_grad = True

        if i == 0:
            X_buffer.append(X0)
            Y_buffer.append(Y0)

        reward = 0
        # Loop over n time steps for this loss segment
        for j in range(0, n):
            # Next time/noise
            t1, W1, B1 = t[:, i + j + 1, :], W[:, i + j + 1, :], B[:, i + j + 1, :]
            # Handle Poisson jumps between t0 and t1 (masking jumps outside interval)
            P1 = torch.where(P < t1[0, 0], P, torch.zeros_like(P, dtype=torch.float))
            P2 = torch.where(P1 > t0[0, 0], P1, torch.zeros_like(P1, dtype=torch.float))
            P3 = torch.where(torch.sum(P2, dim=1) != 0, 1, 0)  # indicator if jump occurred in interval

            P01 = torch.where(P0 < t1[0, 0], P0, torch.zeros_like(P0, dtype=torch.float))
            P02 = torch.where(P01 > t0[0, 0], P01, torch.zeros_like(P01, dtype=torch.float))
            P03 = torch.where(torch.sum(P02, dim=1) != 0, 1, 0) # common jump indicator

            # Compensated jump martingale increments
            Mi = P3 - torch.tensor(np.repeat(self.params_equ["lambda"], self.params_equ["jump_dim"]), \
                                   device=self.device).float().unsqueeze(0) * self.dt
            M0 = P03 - self.params_equ["lambda0"] * self.dt

            # Update control given the current Xt and Yt
            control_weight_list = []
            tmp = 0
            for d in range(self.D):
                if self.control_type == "full":
                    Xd, Yd = X0, Y0
                elif self.control_type == "distributed":
                    Xd = X0[:, tmp: tmp + self.state_dim]
                    Yd = Y0[:, tmp: tmp + self.state_dim]
                    tmp += self.state_dim
                
                # In the leader follower case, we need to add the bernoulli random variable to control
                if self.params_equ["Q_type"] == "leader_follower" and d== 0:
                    control_weight_list.append(net_actor[d](torch.cat([t0, bernoulli_M ,Xd, Yd], dim = 1)))
                else:
                control_weight_list.append(net_actor[d](torch.cat([t0, Xd, Yd], dim = 1)))
            control = torch.cat(
                control_weight_list, dim=1
            )

            # Forward simulate SDE dynamics (Euler + jump terms)
            drift = self.mu(t0, X0, control)
            s_tmp = self.sigma_i(t0, X0, control).expand(W0.size(0), -1, -1)
            s0_tmp = self.sigma_0(t0, X0, control).expand(B0.size(0), -1, -1)
            gamma_tmp = self.xi_i(t0, X0, control).expand(B0.size(0), -1, -1)
            gamma0_tmp = self.xi_0(t0, X0, control).expand(B0.size(0), -1, -1)


            X1 = (
                X0 + drift * self.dt
                + torch.bmm(s_tmp, (W1 - W0).unsqueeze(2)).squeeze(-1)
                + torch.bmm(s0_tmp,  (B1 - B0).unsqueeze(2)).squeeze(-1)
                + torch.bmm(gamma_tmp, Mi.unsqueeze(2)).squeeze(-1)
                + torch.bmm(gamma0_tmp, M0.unsqueeze(2)).squeeze(-1)
            )

            # Forwrad simulate SDE dynamics for Y
            Y1 = (Y0 + drift * self.dt)

            t0, W0, B0, X0, Y0 = t1, W1, B1, X1, Y1
            X_buffer.append(X0)
            Y_buffer.append(Y0)
            if t1[0,0] < self.T:
                # Calculate the running cost
                if self.game_type == "aversion":
                    reward += self.f_Phi(control, X0, Y0, self.fi_param_u, self.fi_param_x) * self.dt
                elif self.game_type == "flocking":
                    reward += self.f_quadratic(control, X0, Y0, self.fi_param_u, self.fi_param_x) * self.dt
            else:
                # Calculate the terminal cost
                reward += self.g_quadratic(control, X0, Y0, gi_param=self.params_equ["gi_param"])
        return X_buffer, Y_buffer, reward

    def Actor_step(self, net_actor, optimizer_actor):
        # Fetch a minibatch of simulated noise trajectories:
        # t: time grid, W: Brownian motions, B: common noise, P: individual Poisson jumps, P0: common Poisson jumps
        t, W, B, P, P0 = self.fetch_minibatch()

        # Xi is the initial condition, repeat it M times for the batch
        X0 = torch.cat([self.Xi.unsqueeze(0).to(self.device)] * self.M).squeeze(2)
        Y0 = torch.cat([self.Yi.unsqueeze(0).to(self.device)] * self.M).squeeze(2)  # batch size * state_dim
        if self.random_initial_state:
            X0 +=  np.random.rand(self.M, self.state_dim * self.D) - 0.5
            X0 = X0.float().to(self.device)


        # Call the loss function:
        X_buffer, Y_buffer, loss = self.loss_function(net_actor, t, W, B, P, P0, X0, Y0, i=0, n=self.N)

        # Reset gradients before backward pass
        optimizer_actor.zero_grad()
        # Compute gradients of actor loss w.r.t. actor parameters
        loss.backward()
        optimizer_actor.step()


        # Return the actor loss for monitoring
        return loss.to(self.device)

    def train_players(self):
        """
        Train player k during iteration it.

        Parameters:
        - it: current outer iteration (int)
        - k: index of the player to train (int)

        This function:
        - Initializes terminal states if it's the first iteration.
        - Creates local copies of critic and actor networks for all players and loads their states.
        - Applies dynamic learning rate scheduling.
        - Trains player k's critic and actor over several epochs.
        - Updates the stored network states.
        """
        self.start_time = time.time()


        # Create local networks for all players and load stored parameters
        net_actor = [Network(self.params_net, penalty=self.params_net["penalty"]).to(self.device)\
                      for _ in range(self.D)]
        all_weights = []

        for d in range(self.D):
            # Load previously saved parameters into local networks
            net_actor[d].load_state_dict(self.dict[f'net_actor_{d}'])
            all_weights += list(net_actor[d].parameters())

        # Set up Adam optimizers for player k's actor
        optimizer_actor = optim.Adam(all_weights, lr=self.params_train["lr_actor"])
        scheduler = ReduceLROnPlateau(optimizer_actor, mode='min', patience=5, factor=self.params_train["gamma_actor"])

        for it in range(self.params_train["iteration"]):
            # Train over specified number of epochs
            for _ in tqdm(range(self.params_train["epochs"])):
                # Perform actor update step for all players
                loss_actor = self.Actor_step(net_actor, optimizer_actor)

                # Save detached losses for monitoring
                self.dict[f"loss_actor"] = loss_actor.detach().cpu().clone()
                self.training_cost_list.append(loss_actor.detach().cpu().clone())

            for d in range(self.D):
                self.dict[f'net_actor_{d}'] = net_actor[d].state_dict()

            ## Validation
            _, _, loss, _ = self.simulation_paths()
            scheduler.step(loss) # validation  loss
            if loss < self.val_loss:
                self.val_loss= loss
                # record the network weights
                for d in range(self.D):
                    self.dict[f'best_net_actor_{d}'] = net_actor[d].state_dict()

            print('It: %d, loss_training: %.4f, loss_validation: %.4f' % (it, self.dict["loss_actor"].item(), loss))
            # Then print the updated lr
            for param_group in optimizer_actor.param_groups:
                print("Updated learning rate:", param_group['lr'])
            if it%5 == 0:
                if self.state_dim==1: self.generate_1d_trajectory()
                else:self.generate_2d_trajectory()
        # After training, save updated networks back to the shared dictionary
        for d in range(self.D):
            self.dict[f'net_actor_{d}'] = net_actor[d].state_dict()

    def f_quadratic(self, control, X, Y,  fi_param_u=0.01, fi_param_x = 1):
        """
        Input:
        - X, Y: Batch_size * D state_dim
        - control ut: Batch_size * D
        [Quadratic], used only in flocking game
        return: f(t, Xt, Yt, ut) torch number
        """
        # X, Y: (B, dn, 1), Q: (n, n)
        B, dn = X.shape
        n = self.Q.shape[0]
        d = dn // n

        # Reshape to (B, n, d)
        Xb = (X - 0.5 * Y).view(B, n, d)
        Yb = Y.view(B, n, d)

        # Expand Q to (B, n, n)
        Qb = self.Q.unsqueeze(0).expand(B, n, n)
        # Compute weighted difference: x_i * sum_j Q_ij - sum_j Q_ij x_j
        diff = Xb * self.Q.sum(dim=1)[None, :, None] - torch.bmm(Qb, Xb)  # (B, n, d)
        # Final result: sum_i y_i^T * diff_i
        result_x = (Yb * diff).sum(dim=(1, 2))  # (B,)
        result_u = control**2 * fi_param_u
        return torch.mean(result_x) * fi_param_x + torch.mean(result_u)




    
    def partial_xi_fi(self, X0, control, sigma_square=10):
        """
        X0: tensor of shape (B, D * state_dim)
        return: tensor of shape (B, D, state_dim)
        """
        B = X0.shape[0]                      # B = R * M if batching over r
        D, d = self.D, self.state_dim
        x = X0.view(B, D, d)                 # (B, D, d)

        # x_i - x_j: create pairwise diff (B, D, D, d)
        x_i = x.unsqueeze(2)                # (B, D, 1, d)
        x_j = x.unsqueeze(1)                # (B, 1, D, d)
        diff = x_i - x_j                    # (B, D, D, d)

        # Q: (D, D) -> (1, D, D) -> (B, D, D)
        Q = self.Q.to(X0.device)
        Qb = Q.unsqueeze(0).expand(B, -1, -1)  # (B, D, D)

        norm_sq = (diff ** 2).sum(-1)               # (B, D, D)
        weights = -Qb * torch.exp(-norm_sq * sigma_square)  # (B, D, D)
        fi = torch.einsum('bijd,bij->bid', diff, weights)    # (B, D, d)

        return fi

    def f_Phi(self, X0, Y0, control, fi_param_u=0.01, fi_param_x = 10):
        y = Y0.view(self.M, self.D, self.state_dim)
        r_array = torch.linspace(0, 1, 51, device=X0.device).view(-1, 1, 1, 1)  # (R, 1, 1, 1)

        X = X0.view(1, self.M, self.D, self.state_dim) - (1 - r_array) * Y0.view(1, self.M, self.D, self.state_dim)  # (R, M, D, d)
        X = X.view(-1, self.D * self.state_dim)  # (R * M, D * d)

        # Apply partial_xi_fi on all R*M at once
        partial_x_f = self.partial_xi_fi( X, control, sigma_square=self.sigma_square)  # (R*M, D, d)
        partial_x_f = partial_x_f.view(len(r_array), self.M, self.D, self.state_dim)

        result = (partial_x_f * y).sum(dim=[2, 3])  # shape: (R, M)
        mean_integral = result.mean(dim=1).mean()   # mean over batch and r
        
        
        result_u = control**2 * fi_param_u


        return (mean_integral * (r_array[1] - r_array[0])).squeeze() * fi_param_x \
            + torch.mean(result_u) 
   

    def g_quadratic(self, control, X0, Y0, gi_param=40):
        """
        - X, Y: Batch_size * D state_dim
        control: num of trajectories * num of players, ut

        return: f(t, Xt, Yt, ut) [num of players * 1]
        """

        Ter = self.terminal.T.expand(self.M, - 1) # batch * D state_dim
        tmp = X0 - 0.5 * Y0 - Ter
        result_x = torch.bmm(tmp.unsqueeze(1), Y0.unsqueeze(2)).squeeze(1)
        return torch.mean(result_x) * gi_param

    def simulation_paths(self):

        # Fetch a minibatch of simulated noise trajectories:
        # t: time grid, W: Brownian motions, B: common noise, P: individual Poisson jumps, P0: common Poisson jumps
        t, W, B, P, P0 = self.fetch_minibatch()
        
        # Xi is the initial condition, repeat it M times for the batch
        X0 = torch.cat([self.Xi.unsqueeze(0).to(self.device)] * self.M).squeeze(2)
        Y0 = torch.cat([self.Yi.unsqueeze(0).to(self.device)] * self.M).squeeze(2)  # batch size * state_dim

        if self.params_equ["Q_type"] == "leader_follower":
            # Generate M Bernoulli(0.5) random variables
            bernoulli_M = torch.bernoulli(torch.full((self.M,), 0.5))  # shape: [500]

        if self.random_initial_state:
            X0 +=  np.random.rand(self.M, self.state_dim * self.D) - 0.5
            X0 = X0.float().to(self.device)

        # Create local networks for all players and load stored parameters
        net_actor = [Network(self.params_net, penalty=self.params_net["penalty"]).to(self.device) 
                        for _ in range(self.D)]

        for d in range(self.D):
            # Load previously saved parameters into local networks
            net_actor[d].load_state_dict(self.dict[f'net_actor_{d}'])

        # Call the loss function:
        X_buffer, Y_buffer, loss = self.loss_function(net_actor, t, W, B, P, P0, X0, Y0, i=0, n=self.N)

        return X_buffer, Y_buffer, loss, net_actor

    def mu(self, t, X, control):
        mu_ = torch.tensor(self.params_equ["mu"], device=self.device).float().unsqueeze(0)
        u = control.unsqueeze(2)
        mu_u = torch.matmul(mu_, u)
        return mu_u.squeeze(2).to(self.device)

    def sigma_i(self, t, X, control):
        """
        Output: 1 * (D * state_dim) * D
        """
        return torch.tensor(self.params_equ["sigma"], device=self.device).float().unsqueeze(0)


    def sigma_0(self, t, X, control):
        return  torch.tensor(self.params_equ["sigma0"], device=self.device).float().unsqueeze(0)

    def xi_i(self, t, X, control):
        return  torch.tensor(self.params_equ["alpha"], device=self.device).float().unsqueeze(0)

    def xi_0(self, t, X, control):
        return  torch.tensor(self.params_equ["beta"], device=self.device).float().unsqueeze(0)
    

    def generate_1d_trajectory(self, X_trajectory=None, plot_mean=True, preset_sample_ind=0, \
                               save=True, plot_label="X", Title=""):
        state_dim = self.state_dim
        linestyle = ["-", "--", ":", "-."]

        if X_trajectory == None:
            # Simulate new trajectories if not provided
            X_buffer, Y_buffer, loss, net_actor = self.simulation_paths()
            if plot_label == "X":
                # Reshape into shape: (T, M, D, state_dim)
                X_trajectory = torch.stack(X_buffer).view(-1, self.M, self.D, state_dim)
            elif plot_label == "Y":
                X_trajectory = torch.stack(Y_buffer).view(-1, self.M, self.D, state_dim)
            else:
                print("Incompatible plotting label. Please choose 'X' or 'Y'.")
        if plot_mean: traj = torch.mean(X_trajectory, dim=1).detach().cpu()
        else: traj = X_trajectory[:,preset_sample_ind, :,:].detach().cpu()
        for i in range(self.D):
            plt.plot(np.linspace(0,self.T, len(traj[:,i,:]) ), traj[:,i,:], linestyle[i], label="Player " + str(i+1))
            plt.plot([self.T], [self.terminal[i].detach().cpu().numpy()], "x", markersize=10, color="red" )
            if plot_mean: marker_size = 10
            else: marker_size = 3
            plt.plot([0], [traj[0, i, :].detach().cpu().numpy()], "o", markersize=marker_size, color="red" )
        plt.grid("True")
        plt.legend()
        plt.title(Title)
        if save:
            filename = f"output/trajectory_{self.filename}.png"
            plt.savefig(filename, dpi=300)
        plt.show()
        return preset_sample_ind, X_trajectory
        
        
    def generate_2d_trajectory_with_colorbar(self, X_trajectory=None,
                            plot_mean=True, preset_sample_ind=None, plot_label="X", save=True):
        

        def get_truncated_cmap(cmap_name, minval=0.5, maxval=1, n=256):
            """
            Generate a truncated colormap by slicing the input range.
            This avoids overly bright or dark extremes.
            """
            base = cm.get_cmap(cmap_name, n)
            new_colors = base(np.linspace(minval, maxval, n))
            return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", new_colors)

        def plot_traj(ax, x, y, xT, yT, x0, y0, base_cmap, i):
            """
            Plot a colored trajectory with time-gradient coloring.
            """
            T = len(x)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t = np.linspace(0, 1, T - 1)

            cmap = get_truncated_cmap(base_cmap)
            lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
            lc.set_array(1 - t)
            lc.set_linewidth(2)
            ax.add_collection(lc)

            # Add dummy line for legend
            ax.plot([], [], color=cmap(1), linewidth=2, label=f"Player {i+1}")

            # Mark initial and terminal positions
            ax.plot(x0, y0, marker='o', markersize=10, color="red")
            ax.plot(xT, yT, marker='x', markersize=10, color="red")


        state_dim = self.state_dim
        # Only allow 2D state for this plotting function
        if state_dim != 2:
            print("Incompatible plotting function. Only works with state dimension = 2.")
            return
        
        if X_trajectory is None:
            # Simulate new trajectories if not provided
            X_buffer, Y_buffer, loss, net_actor = self.simulation_paths()
            if plot_label == "X":
                # Reshape into shape: (T, M, D, state_dim)
                X_trajectory = torch.stack(X_buffer).view(-1, self.M, self.D, state_dim)
            elif plot_label == "Y":
                X_trajectory = torch.stack(Y_buffer).view(-1, self.M, self.D, state_dim)
            else:
                print("Incompatible plotting label. Please choose 'X' or 'Y'.")
                return


        colormaps = ['Greys', 'Greens', 'Purples', 'Oranges']
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, 0.2])

        ax_traj = fig.add_subplot(gs[0, 0])
        ax_cb = fig.add_subplot(gs[0, 1])

        ## Trajectory Plotting
        for i in range(self.D):  # D players
            if plot_mean:
                traj_np = np.mean(X_trajectory[:, :, i, :].detach().numpy(), axis=1) 
            elif preset_sample_ind is not None: 
                traj_np = X_trajectory[:, preset_sample_ind, i, :].detach().numpy()
            else:
                preset_sample_ind = np.random.choice(X_trajectory.shape[1])
                traj_np = X_trajectory[:, preset_sample_ind, i, :].detach().numpy()

            x = traj_np[:, 0]
            y = traj_np[:, 1]
            xT, yT = self.terminal.view(self.D, -1).detach().numpy()[i]
            x0, y0 = self.origin.view(self.D, -1).detach().numpy()[i]
            plot_traj(ax_traj, x, y, xT, yT, x0, y0, colormaps[i], i)

        if plot_mean:
            traj = np.mean(X_trajectory[:, :, :, :].detach().numpy(), axis=1) 
        else:
            traj = X_trajectory[:, preset_sample_ind, :, :].detach().numpy()
        min_x, max_x = np.min(traj[:,:,0]), np.max(traj[:,:,0])
        min_y, max_y = np.min(traj[:,:,1]), np.max(traj[:,:,1])
        expandx= (max_x - min_x) * 0.1
        expandy = (max_y - min_y) * 0.1
        # Fix aspect ratio and visible range
        ax_traj.set_xlim(min_x-expandx, max_x+expandx)
        ax_traj.set_ylim(min_y-expandy, max_y+expandy)
        ax_traj.set_aspect('equal')
        ax_traj.grid(True)
        # ax_traj.legend(loc='upper right')

        ## Colorbar plotting
        n = 256
        gradient = np.linspace(0, 1, n).reshape(-1, 1)
        for i, cmap_name in enumerate(colormaps):
            truncated_cmap = get_truncated_cmap(cmap_name)
            ax_cb.imshow(gradient, aspect='auto', cmap=truncated_cmap,
                        extent=[i * 0.25, i * 0.25 + 0.25, 0, 1])
                        

        ax_cb.yaxis.set_label_position("right")
        ax_cb.yaxis.tick_right()
        ax_cb.set_ylabel('Time')
        ax_cb.set_ylim(0, 1)
        ax_cb.set_xlim(0, 1)
        ax_cb.set_xticks([])
        
        if save:
            filename = f"output/trajectory_{self.game_type}_D={self.D}_d={state_dim}.png"
            plt.savefig(filename, dpi=300)
        plt.show()

        return preset_sample_ind, X_trajectory

    def generate_2d_trajectory(self, X_trajectory=None, filename="",
                                plot_mean=True, preset_sample_ind=None, plot_label="X", save=True):

        def plot_traj(ax, x, y, xT, yT, x0, y0, color, i, marker_='s', label=True, linestyle="-"):
            T = len(x)
            ax.plot(x, y, linestyle, color=color, linewidth=2)

            # Mark 3 points at 1/4, 1/2, 3/4 and label them
            idxs = [T // 4, T // 2, 3 * T // 4]
            markers = [ 's', 's', 's']
            for j, (idx, marker) in enumerate(zip(idxs, markers)):
                ax.text(x[idx], y[idx], str(j+1), color=str(1), fontsize=8,
                        ha='center', va='center', zorder=2 + i)
                ax.plot(x[idx], y[idx], linestyle, marker=marker_, markersize=10, color=color, zorder=1 + i, label=f"Player {i+1}" if j==1 else None)
            
            # Mark initial and terminal positions
            ax.plot(x0, y0, marker='o', markersize=10, color="red")
            ax.plot(xT, yT, marker='x', markersize=10, color="red")


        state_dim = self.state_dim
        if state_dim != 2:
            print("Incompatible plotting function. Only works with state dimension = 2.")
            return

        if X_trajectory is None:
            X_buffer, Y_buffer, loss, net_actor = self.simulation_paths()
            if plot_label == "X":
                X_trajectory = torch.stack(X_buffer).view(-1, self.M, self.D, state_dim)
            elif plot_label == "Y":
                X_trajectory = torch.stack(Y_buffer).view(-1, self.M, self.D, state_dim)
            else:
                print("Incompatible plotting label. Please choose 'X' or 'Y'.")
                return

        # Use fixed colors per player
        # flat_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
        #                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
        # flat_colors = ['#9467bd', '#1f77b4',  '#8c564b', '#2ca02c',
        #       '#d62728',  '#7f7f7f', '#000000','#ff7f0e',]
        flat_colors =  [  "#B563AD",
            "#bd4f05",
            "#469355",  
            "#5A8AA4",   
            '#000000']  
        markers=['s', '^', 'd', 'o']
        fig, ax_traj = plt.subplots(figsize=(6, 5))
        ax_traj.grid(True)

        linestyles = [":","-.","--",'-']
        for i in range(self.D):
            if plot_mean:
                traj_np = np.mean(X_trajectory[:, :, i, :].detach().numpy(), axis=1)
            elif preset_sample_ind is not None:
                traj_np = X_trajectory[:, preset_sample_ind, i, :].detach().numpy()
            else:
                preset_sample_ind = np.random.choice(X_trajectory.shape[1])
                traj_np = X_trajectory[:, preset_sample_ind, i, :].detach().numpy()

            x = traj_np[:, 0]
            y = traj_np[:, 1]
            xT, yT = self.terminal.view(self.D, -1).detach().numpy()[i]
            x0, y0 = self.origin.view(self.D, -1).detach().numpy()[i]
            plot_traj(ax_traj, x, y, xT, yT, x0, y0, flat_colors[i % len(flat_colors)], i, linestyle=linestyles[i], marker_=markers[i])

        if plot_mean:
            traj = np.mean(X_trajectory[:, :, :, :].detach().numpy(), axis=1)
        else:
            traj = X_trajectory[:, preset_sample_ind, :, :].detach().numpy()
        min_x, max_x = np.min(traj[:, :, 0]), np.max(traj[:, :, 0])
        min_y, max_y = np.min(traj[:, :, 1]), np.max(traj[:, :, 1])
        expandx = (max_x - min_x) * 0.1
        expandy = (max_y - min_y) * 0.1

        ax_traj.set_xlim(min_x - expandx, max_x + expandx)
        ax_traj.set_ylim(min_y - expandy, max_y + expandy)
        # ax_traj.set_aspect('equal')
        ax_traj.legend()

        if save:
            filename = f"output/trajectory{self.filename}{filename}.png"
            plt.savefig(filename, dpi=300)


        plt.show()

        return preset_sample_ind, X_trajectory
    

    def generate_animation(self, X_trajectory=None, plot_mean=True, preset_sample_ind=0, filename=None, save=True):
        if filename == None: filename = self.filename

        if X_trajectory==None:
            X_buffer, Y_buffer, loss, net_actor = self.simulation_paths()
            X_trajectory = torch.stack(X_buffer).view(-1, self.M, self.D, self.state_dim)
        
        if self.state_dim == 1:
            
            if plot_mean: traj = torch.mean(X_trajectory, dim=1).detach().cpu().numpy()
            else: traj = torch(X_trajectory[:, preset_sample_ind, :, :].detach().numpy())

            for i in range(self.D):
                plt.plot(traj[:,i,:])
            plt.show()

            # Remove trailing singleton dimension: shape → (T, 4)
            data = traj.squeeze(-1)
            T = len(traj)
            time = np.linspace(0,1,T)

            # Set up figure
            fig, ax = plt.subplots()
            # colors = ['r', 'g', 'b', 'm']
            colors= ["red", "green", "blue", "orange"]

            lines = [ax.plot([], [], color=c, label=f'Player {i+1}')[0] for i, c in enumerate(colors)]

            # min_x, max_x = np.min(trajectories[:,:,0]), np.max(trajectories[:,:,0])
            # min_y, max_y = np.min(trajectories[:,:,1]), nBNVp.max(trajectories[:,:,1])
            expand=0.1
            # ax.set_xlim(min_x-expand, max_x+expand)
            # ax.set_ylim(min_y-expand, max_y+expand)

            ax.set_xlim(time[0]-expand, time[-1]+expand)
            ymin, ymax = data.min(), data.max()
            ax.set_ylim(ymin - expand, ymax + expand)
            ax.set_xlabel("Time")
            ax.set_ylabel("Position")
            ax.legend()
            # ax.set_title("1D Player Trajectories Over Time")
            fixed_points = self.terminal.view(4).detach().numpy()

            for i in range(self.D):
                fixed_scatter = ax.plot(1, fixed_points[i], 
                                        'x', markersize=8, color="red")
            lines, points = [], []

            for i in range(4):
                line, = ax.plot([], [], color=colors[i], lw=2, label=f"Player {i+1}")
                point, = ax.plot([], [], 'o', color=colors[i])
                lines.append(line)
                points.append(point)
            # Init
            def init():
                for line in lines:
                    line.set_data([], [])
                return lines
            # Update
            def update(frame):
                x = np.linspace(0, 1, T)[:frame + 1]
                for i, line in enumerate(lines):
                    y = data[:frame + 1, i]
                    line.set_data(x, y)
                    points[i].set_data(x[-1], y[-1])

                return lines + points + fixed_scatter
            
            plt.grid(True)

            # Animate
            ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=100)
            if save: ani.save("video/trajectory" + filename + ".mp4", writer='ffmpeg', fps=30)
            plt.show()

        if self.state_dim == 2:

            if plot_mean: trajectories = X_trajectory.mean(dim=1).detach().numpy()
            else: trajectories = X_trajectory[:, preset_sample_ind, :, :].detach().numpy()

            T = len(trajectories)

            # Plot setup
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            # colors = ['gray', 'seagreen', 'mediumslateblue', 'chocolate']
            colors= ["red", "green", "blue", "orange"]
            # After fig, ax setup
            fixed_points = self.terminal.view(4,2).detach().numpy()

            for i in range(self.D):
                fixed_scatter = ax.plot(fixed_points[i, 0], fixed_points[i, 1],
                                        'x', markersize=8, color="red")
            lines, points = [], []

            for i in range(4):
                line, = ax.plot([], [], color=colors[i], lw=2, label=f"Player {i+1}")
                point, = ax.plot([], [], 'o', color=colors[i])
                lines.append(line)
                points.append(point)

            min_x, max_x = np.min(trajectories[:,:,0]), np.max(trajectories[:,:,0])
            min_y, max_y = np.min(trajectories[:,:,1]), np.max(trajectories[:,:,1])
            expand=0.1
            ax.set_xlim(min_x-expand, max_x+expand)
            ax.set_ylim(min_y-expand, max_y+expand)
            # ax.set_aspect('equal')

            legend = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
            fig.canvas.draw()  # ensures the legend is rendered before animation

            # Animation
            def init():
                for line, point in zip(lines, points):
                    line.set_data([], [])
                    point.set_data([], [])
                return lines + points + fixed_scatter + [legend] 

            def update(frame):
                for i in range(4):
                    traj = trajectories[:frame+1, i]
                    lines[i].set_data(traj[:, 0], traj[:, 1])
                    points[i].set_data(traj[-1, 0], traj[-1, 1])
                return lines + points + fixed_scatter + [legend] 

            ani = FuncAnimation(fig, update, frames=T, init_func=init,
                                blit=True, interval=30)
            plt.grid(True)

            if save: ani.save("video/trajectory" + filename + ".mp4", writer='ffmpeg', fps=30)
            plt.show()
    
    def save_NN(self, filename = None):
        if filename == None: filename = self.filename
        X_buffer, Y_buffer, loss, net_actor = self.simulation_paths()
        
        # Save all models in one file
        torch.save({f'net_actor_{d}': net_actor[d].state_dict() for d in range(self.D)}, \
                   'outputNN/' + filename + '.pth')
        print("The NN weights are saved in")
        print('outputNN/' + filename + '.pth')