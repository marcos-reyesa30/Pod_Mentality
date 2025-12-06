import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GERF_BRA(nn.Module):
    """
    Optimized Gaussian-based Effective Receptive Field with Bi-Level Routing Attention
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads (default: 8)
        window_size (int): Size of the window for splitting feature maps (default: 7)
        topk (int): Number of top-k regions to route (default: 4)
        qkv_bias (bool): Whether to add bias to qkv projection (default: True)
    """
    
    def __init__(self, dim, num_heads=8, window_size=7, topk=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.topk = min(topk, window_size * window_size)  # Ensure topk doesn't exceed total windows
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Simplified Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Simplified Gaussian parameters (only predict scale, not full covariance)
        # Make sure this matches the dimension of the input features
        self.gaussian_params = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 2)
        )
        
        # Depth-wise convolution for local feature enhancement
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # Cache for positional encodings
        self.register_buffer('mu_cache', None, persistent=False)
        
    def get_simplified_gaussian_weights(self, sigma_params, H, W, device):
        """
        Optimized 2D Gaussian distribution weights (assuming ρ=0 for diagonal covariance)
        
        Args:
            sigma_params: Tensor of shape (B, S^2, 2) containing [σ_x, σ_y]
            H, W: Height and width of feature map in windows
            device: torch device
            
        Returns:
            Gaussian weight matrix of shape (B, S^2, S^2)
        """
        B, S_sq, _ = sigma_params.shape
        S = int(math.sqrt(S_sq))
        
        # Extract and constrain parameters
        sigma_x = F.softplus(sigma_params[..., 0]) + 0.5  # Ensure positive, min 0.5
        sigma_y = F.softplus(sigma_params[..., 1]) + 0.5
        
        # Create or reuse cached mu values
        if self.mu_cache is None or self.mu_cache.shape[0] != S_sq:
            y_coords = torch.arange(S, dtype=torch.float32, device=device)
            x_coords = torch.arange(S, dtype=torch.float32, device=device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            self.mu_cache = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (S^2, 2)
        
        mu = self.mu_cache.to(device)
        
        # Vectorized distance calculation
        # mu: (S^2, 2), reshape to (1, S^2, 1, 2)
        # grid_coords: (S^2, 2), reshape to (1, 1, S^2, 2)
        mu_i = mu.unsqueeze(0).unsqueeze(2)  # (1, S^2, 1, 2)
        mu_j = mu.unsqueeze(0).unsqueeze(1)  # (1, 1, S^2, 2)
        
        # Calculate distances
        diff = mu_j - mu_i  # (1, S^2, S^2, 2)
        dx = diff[..., 0]  # (1, S^2, S^2)
        dy = diff[..., 1]
        
        # Expand sigma for broadcasting
        sigma_x = sigma_x.unsqueeze(2)  # (B, S^2, 1)
        sigma_y = sigma_y.unsqueeze(2)
        
        # Simplified Gaussian (diagonal covariance, ρ=0)
        z = (dx.unsqueeze(0) / sigma_x) ** 2 + (dy.unsqueeze(0) / sigma_y) ** 2
        gaussian = torch.exp(-0.5 * z)
        
        # Normalize
        gaussian = gaussian / (gaussian.sum(dim=-1, keepdim=True) + 1e-6)
        
        return gaussian  # (B, S^2, S^2)
    
    def forward(self, x):
        """
        Forward pass of GERF-BRA
        
        Args:
            x: Input tensor of shape (B, H, W, C)
            
        Returns:
            Output tensor of shape (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Ensure H and W are divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            _, H, W, _ = x.shape
        
        # Calculate number of windows
        nH = H // self.window_size
        nW = W // self.window_size
        S = nH * nW
        
        # Adjust topk if needed (can't be larger than number of windows)
        actual_topk = min(self.topk, S)
        
        # Reshape to windows: (B, S, ws*ws, C)
        x_windows = x.view(B, nH, self.window_size, nW, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_windows = x_windows.view(B, S, self.window_size * self.window_size, C)
        
        # Region-level mean for routing
        x_region = x_windows.mean(dim=2)  # (B, S, C)
        
        # Generate Q, K, V
        qkv = self.qkv(x_region).reshape(B, S, 3, C)
        q_r, k_r, v_r = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        
        # Compute adjacency matrix for routing
        A_r = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale  # (B, S, S)
        
        # Top-k routing indices (use actual_topk)
        _, topk_indices = torch.topk(A_r, actual_topk, dim=-1)  # (B, S, topk)
        
        # Generate Gaussian parameters
        gaussian_params = self.gaussian_params(q_r)  # (B, S, 2)
        
        # Compute Gaussian weights
        gaussian_weights = self.get_simplified_gaussian_weights(gaussian_params, nH, nW, x.device)
        
        # Full QKV for token-level attention
        qkv_full = self.qkv(x_windows).reshape(B, S, self.window_size * self.window_size, 3, C)
        q, k, v = qkv_full[..., 0, :], qkv_full[..., 1, :], qkv_full[..., 2, :]
        
        # Efficient gathering with Gaussian weighting
        batch_idx = torch.arange(B, device=x.device)[:, None, None].expand(B, S, actual_topk)
        window_idx = torch.arange(S, device=x.device)[None, :, None].expand(B, S, actual_topk)
        
        # Gather Gaussian weights for selected windows
        gaussian_wt = gaussian_weights[batch_idx, window_idx, topk_indices]  # (B, S, topk)
        
        # Gather K and V
        k_g = k[batch_idx, topk_indices]  # (B, S, topk, ws*ws, C)
        v_g = v[batch_idx, topk_indices]
        
        # Apply Gaussian weighting efficiently
        k_g = k_g * gaussian_wt.unsqueeze(-1).unsqueeze(-1)
        v_g = v_g * gaussian_wt.unsqueeze(-1).unsqueeze(-1)
        
        # Reshape for attention
        k_g = k_g.reshape(B, S, -1, C)  # (B, S, topk*ws*ws, C)
        v_g = v_g.reshape(B, S, -1, C)
        
        # Token-level attention
        attn = torch.matmul(q, k_g.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v_g)  # (B, S, ws*ws, C)
        
        # Add depth-wise convolution branch
        v_dwconv = v.reshape(B * S, self.window_size, self.window_size, C)
        v_dwconv = v_dwconv.permute(0, 3, 1, 2)  # (B*S, C, ws, ws)
        v_dwconv = self.dwconv(v_dwconv)
        v_dwconv = v_dwconv.permute(0, 2, 3, 1).reshape(B, S, -1, C)
        
        out = out + v_dwconv
        
        # Reshape back
        out = out.view(B, nH, nW, self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, H, W, C)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H-pad_h, :W-pad_w, :]
        
        # Output projection
        out = self.proj(out)
        
        return out


class GERF_BRA_Block(nn.Module):
    """
    Optimized GERF-BRA Block for YOLO integration
    
    Args:
        c1 (int): Number of input channels
        c2 (int): Number of output channels
        num_heads (int): Number of attention heads (default: 8)
        window_sizes (list or tuple): Window sizes (default: (7, 5, 3))
        topk (int): Number of top-k regions (default: 4)
        lightweight (bool): Use single window size for speed (default: False)
    """
    
    def __init__(self, c1, c2, num_heads=8, window_sizes=(7, 5, 3), topk=4, lightweight=False):
        super().__init__()
        
        # For speed, optionally use only one window size
        if lightweight:
            window_sizes = [window_sizes[0] if isinstance(window_sizes, (list, tuple)) else 7]
        elif isinstance(window_sizes, (list, tuple)):
            window_sizes = list(window_sizes)
        else:
            window_sizes = [7]
        
        self.gerf_bra_modules = nn.ModuleList([
            GERF_BRA(c1, num_heads, ws, topk) for ws in window_sizes
        ])
        
        self.norm = nn.LayerNorm(c1)
        
        # Projection
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Convert to (B, H, W, C) for attention
        x = x.permute(0, 2, 3, 1)
        
        # Apply GERF-BRA modules (with residual connection)
        identity = x
        for module in self.gerf_bra_modules:
            x = identity + module(self.norm(x))
            identity = x
        
        # Convert back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Apply projection
        x = self.proj(x)
        
        return x


# Lightweight version for faster training
class GERF_BRA_Lite(nn.Module):
    """
    Lightweight GERF-BRA for faster training - single window size, simplified attention
    """
    
    def __init__(self, c1, c2, num_heads=8, window_size=7, topk=4):
        super().__init__()
        
        self.gerf_bra = GERF_BRA(c1, num_heads, window_size, topk)
        self.norm = nn.LayerNorm(c1)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x + self.gerf_bra(self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return self.proj(x)