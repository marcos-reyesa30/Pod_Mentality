import torch
import torch.nn as nn
import torch.nn.functional as F


class GERF_BRA_Simple(nn.Module):
    """
    Simplified GERF-BRA that's easier to integrate with YOLO
    Uses spatial attention with Gaussian weighting
    """
    
    def __init__(self, c1, c2=None, k=None):
        """
        Args:
            c1: Input channels (auto-filled by YOLO)
            c2: Output channels (from YAML, optional)
            k: Unused compatibility parameter for YOLO parser
        """
        super().__init__()
        
        # If c2 is None or not provided, use c1
        if c2 is None:
            c2 = c1
        
        # Multi-scale depth-wise convolutions (simulating different window sizes)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=kernel, padding=kernel//2, groups=c1)
            for kernel in [7, 5, 3]
        ])
        
        # Point-wise convolutions after each DW conv
        self.pw_convs = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=1) for _ in range(3)
        ])
        
        # Gaussian parameter prediction (predicts attention weights)
        self.attn_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(c1 // 4, 8), 1),  # Ensure at least 8 channels
            nn.ReLU(),
            nn.Conv2d(max(c1 // 4, 8), 3, 1),  # 3 weights for 3 scales
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        identity = x
        
        # Get attention weights for different scales
        scale_weights = self.attn_conv(x)  # (B, 3, 1, 1)
        
        # Apply multi-scale convolutions with learned weights
        out = 0
        for i, (dw, pw) in enumerate(zip(self.dw_convs, self.pw_convs)):
            feat = dw(x)
            feat = pw(feat)
            # Apply scale-specific weight
            weight = scale_weights[:, i:i+1, :, :]
            out = out + feat * weight
        
        # Residual connection
        out = out + identity
        
        # Output projection
        out = self.proj(out)
        
        return out


# Alias for easy integration
GERF_BRA_Lite = GERF_BRA_Simple