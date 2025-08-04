
import torch
import numpy as np

class GameConfig:
    def __init__(self, D, state_dim, game_type="flocking", Q_type="average", device="cpu"):
        """
        Initialize configuration for a D-player system with state_dim-dimensional state.

        Args:
            D (int): number of players
            state_dim (int): dimension of each player's state
            Q_type (str): type of Q interaction matrix ("average", "twogroups", "random")
            device (str): device for torch tensors
        """
        self.D = D
        self.state_dim = state_dim
        self.device = device
        self.BM_dim = state_dim
        self.u_dim = state_dim
        self.jump_dim = state_dim
        self.Q_type = Q_type
        self.game_type = game_type

        # Construct all necessary components
        self.build_drift()
        self.build_diffusion()
        self.build_jump()
        self.build_Q()
        self.load_conditions()

    def block_diag_repeat(self, A, D, weight=None):
        """
        Build a D-block block-diagonal matrix with optional per-block scaling.

        Args:
            A (np.ndarray): base matrix of shape (m, n)
            D (int): number of blocks
            weight (array-like): optional length-D weight vector

        Returns:
            np.ndarray: block-diagonal matrix of shape (D*m, D*n)
        """
        m, n = A.shape
        weight = np.ones(D) if weight is None else np.asarray(weight)
        return np.block([
            [weight[i] * A if i == j else np.zeros((m, n)) for j in range(D)]
            for i in range(D)
        ])

    def build_drift(self):
        """Construct the global drift matrix b."""
        bi = np.eye(self.state_dim, self.u_dim)  # Drift for one player
        self.b = self.block_diag_repeat(bi, self.D)

    def build_diffusion(self, sigma_param=0.1, sigma0_param=0):
        """Construct global volatility matrices S and S0."""
        Si = np.eye(self.state_dim, self.BM_dim)
        self.S = self.block_diag_repeat(Si, self.D, weight=np.arange(self.D)/self.D) * sigma_param
        Si0 = np.eye(self.state_dim, self.BM_dim)
        self.S0 = np.tile(Si0, (self.D, 1)) * sigma0_param

    def build_jump(self, gamma_param=0.1, gamma0_param=0):
        """Construct jump diffusion matrices Gamma and Gamma0."""
        Gamma_i = np.eye(self.state_dim, self.jump_dim)
        self.Gamma = self.block_diag_repeat(Gamma_i, self.D) * gamma_param
        Gamma0_i = np.eye(self.state_dim, self.jump_dim)
        self.Gamma0 = np.tile(Gamma0_i, (self.D, 1)) * gamma0_param

    def build_Q(self):
        """Construct the D×D interaction matrix Q based on the specified type."""
        D = self.D
        if self.Q_type == "average":
            self.Q = np.ones([D, D])
        elif self.Q_type == "twogroups":
            self.Q = np.ones([D, D])
            group1 = list(np.arange(int(D / 2)))
            group2 = list(np.arange(int(np.ceil(D / 2)), D))
            for i in range(D):
                for j in range(D):
                    if (i in group1 and j in group1) or (i in group2 and j in group2):
                        self.Q[i, j] = 0
        elif self.Q_type == "random":
            Q = np.random.rand(D, D)
            self.Q = ((Q + Q.T) / 2 > 0.5).astype(float)
        elif self.Q_type == "leader_follower":
            pass
        else:
            raise ValueError(f"Unknown Q_type: {self.Q_type}")
        print("Q:", self.Q)

    def load_conditions(self):
        """Predefine initial and terminal conditions for various game modes."""
        D, state_dim = self.D, self.state_dim
        device = self.device

        self.origin_conditions = {
            "flocking": {
                1: {1: torch.zeros(D, 1).float()},
                2: {
                    4: torch.tensor([[0, 0.3, 0, 0.1, 0, -0.1, 0, -0.3]]).T.float().to(device),
                    1: torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]]).T.float().to(device)},
            },
            "aversion": {
                1: {1: torch.zeros(D, 1).float()},
                2: {1: (torch.ones([D * state_dim, 1]) / state_dim).float().to(device),
                    4: torch.tensor([[0, 0.3, 0, 0.1, 0, -0.1, 0, -0.3]]).T.float().to(device)},
            },
        }

        self.terminal_conditions = {
            "flocking": {
                1: {4: (torch.arange(D) / D - 0.5).unsqueeze(1).float()},
                2: {
                    4: torch.tensor([[0.25, 0, 0, 0.5, -0.75, 0, 0, -1]]).T.float().to(device),
                    2: torch.tensor([[1, 0.2, 1, -0.2, 1, 0.2, 1, -0.2]]).T.float().to(device),
                },
            },
            "aversion": {
                1: {1: torch.ones(D, 1).float(),
                    4: (torch.arange(D) / D - 0.5).unsqueeze(1).float()},
                2: {1: (torch.ones([D * state_dim, 1]) / state_dim).float().to(device),
                    2: torch.tensor([[1, 0.2, 1, -0.2, 1, 0.2, 1, -0.2]]).T.float().to(device),}
            },
        }

    def get_origin(self,  num_origins):
        """ 
        Get origin condition for a given mode and number of players.

        Returns:
            torch.Tensor or None
        """
        try:
            return self.origin_conditions[self.game_type][self.state_dim][num_origins]
        except KeyError:
            print(f"[Info] No origin for (mode={self.game_type}, dim={self.state_dim}, num={num_origins})")
            return None

    def get_terminal(self, num_destinations):
        """
        Get terminal condition for a given mode and number of destinations.

        Returns:
            torch.Tensor or None
        """
        try:
            return self.terminal_conditions[self.game_type][self.state_dim][num_destinations]
        except KeyError:
            print(f"[Info] No terminal for (mode={self.game_type}, dim={self.state_dim}, num={num_destinations})")
            return None
