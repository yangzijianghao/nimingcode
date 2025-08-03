import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeAwareResidualPredictor(nn.Module):
    def __init__(self, channel_dim=32, signal_length=400, time_embed_dim=128, max_timesteps=1000, num_layers=2):
        super().__init__()
        self.channel_dim = channel_dim
        self.signal_length = signal_length
        self.max_timesteps = max_timesteps
        self.num_layers = num_layers

        # Sinusoidal time embedding layer
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        # Project time embedding to match signal dimensions
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, channel_dim * signal_length),
            nn.ReLU()
        )

        # Deep convolutional block for residual prediction
        layers = []
        in_channels = 2 * channel_dim  # Input channels (signal + time projection)
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_channels, channel_dim, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(4, channel_dim))  # Normalization
            layers.append(nn.ReLU())
            in_channels = channel_dim
        self.conv_net = nn.Sequential(*layers)

    def forward(self, z50, t):
        """
        Predict residual at timestep t
        
        Args:
            z50: Input tensor (B, channel_dim, signal_length)
            t: Timestep tensor (B,) or scalar integer
        Returns:
            Predicted residual tensor (B, channel_dim, signal_length)
        """
        B, C, L = z50.shape

        # Embed timestep
        t_embed = self.time_embedding(t)  # (B, D)
        t_proj = self.time_proj(t_embed)  # (B, C*L)
        t_proj = t_proj.view(B, C, L)     # Reshape to (B, C, L)

        # Concatenate input and time projection
        x = torch.cat([z50, t_proj], dim=1)  # (B, 2C, L)

        # Predict residual
        res_t_pred = self.conv_net(x)        # (B, C, L)
        return res_t_pred


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        Create sinusoidal embeddings for timesteps
        
        Args:
            t: Timestep tensor (B,) or scalar integer
        Returns:
            Embedding tensor (B, dim)
        """
        # Handle scalar input
        if isinstance(t, int):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            t = torch.tensor([t], dtype=torch.float32, device=device)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        freqs = torch.exp(-emb * torch.arange(half_dim, device=t.device))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        
        # Create sinusoidal embeddings
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            # Add zero padding for odd dimensions
            emb = torch.cat([sin_emb, cos_emb, torch.zeros_like(sin_emb[:, :1])], dim=1)
        else:
            emb = torch.cat([sin_emb, cos_emb], dim=1)
            
        return emb