import torch
import torch.nn.functional as F

class ChemicalFields:
    """
    GPU-accelerated environment for 4 isolated chemical layers:
    [0: Food, 1: Toxin, 2: Pheromone A, 3: Pheromone B]
    """

    def __init__(self, channels= 4, size=128, gamma = (0.999, 0.980, 0.920, 0.900), device="cuda"):
        self.channels = channels
        self.size = size
        self.device = torch.device(device)

        # 1. The World Grid
        # Shape: (Batch=1, Channels=4, Height=size, Width=size)
        self.grid = torch.zeros((1, channels, size, size), dtype=torch.float32, device=self.device)

        # 2. The Decay Vector (Temporal Decay)
        # Reshaped to (1, 4, 1, 1) to trigger PyTorch's zero-stride broadcasting across the spatial grid
        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=self.device).view(1, channels, 1, 1)

        # 3. The Diffusion Kernel (Spatial Spread)
        # Base 3x3 Gaussian matrix ensuring mass conservation (sum = 1.0)
        base_kernel = torch.tensor([
            [0.05, 0.10, 0.05],
            [0.10, 0.40, 0.10],
            [0.05, 0.10, 0.05]
        ], dtype=torch.float32, device=self.device)

        # Reshape and repeat to form the 4D Depthwise Weight Tensor
        # Shape: (out_channels=4, in_channels_per_group=1, H=3, W=3)
        self.kernel = base_kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

    def diffuse(self, pad, mode):
        padded_grid = F.pad(self.grid, pad, mode=mode)
        self.grid = F.conv2d(padded_grid, self.kernel, groups=self.channels, padding=0) # double check

    def update(self):
        """Executes one physics frame: Toroidal Diffusion -> Channel Decay"""
        # Step A: Circular padding to create a seamless toroidal universe (wraps edges)
        self.diffuse((1, 1, 1, 1), "circular")

        # Step C: Multiply by broadcasted decay tensor
        self.grid = self.grid * self.gamma

    def add_chemical(
            self,
            channel_idx: int | torch.Tensor,
            x_coords: torch.Tensor,
            y_coords: torch.Tensor,
            amounts: torch.Tensor
    ):
        """
        Injects chemical mass at specific coordinates in parallel.

        Supports dropping a single chemical type across all agents (pass int)
        or individual chemical types per agent (pass 1D Tensor).
        Uses PyTorch's atomic accumulation to prevent GPU race conditions when
        multiple agents occupy the same pixel.
        """
        # 1. Bounds check and wrap coordinates into valid integer grid indices
        x_coords = x_coords.long() % self.size
        y_coords = y_coords.long() % self.size

        # 2. Normalize channel selection into a 1D index tensor matching (N,) coordinates
        if isinstance(channel_idx, int):
            channel_coords = torch.full_like(x_coords, fill_value=channel_idx, dtype=torch.long)
        else:
            channel_coords = channel_idx.long()

        # 3. Perform atomic GPU addition across 3D coordinates (channel, y, x)
        # self.grid[0] accesses the 3D tensor of shape (4, height, width)
        self.grid[0].index_put_(
            indices=(channel_coords, y_coords, x_coords),
            values=amounts,
            accumulate=True
        )

    def get_concentrations(self, x_coords: torch.Tensor, y_coords: torch.Tensor):
        """
        Reads the chemical concentrations across all 4 channels for a batch of (x, y) coordinates.
        Returns a tensor of shape (N, 4) containing the environmental data for N agents.
        """
        x_coords = x_coords.long() % self.size
        y_coords = y_coords.long() % self.size

        # Returns all 4 channels for the requested spatial coordinates
        return self.grid[0, :, y_coords, x_coords].T