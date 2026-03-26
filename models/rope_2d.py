import torch
import torch.nn as nn
from typing import Tuple

class Fused2DRoPE(nn.Module):
    """
    Factorized 2D Rotary Positional Encoding (RoPE).
    
    Splits the head dimension d into two equal halves (d/2). 
    Applies standard 1D RoPE to the first half using row coordinates, 
    and to the second half using column coordinates.
    """
    def __init__(self, head_dim: int, max_grid_size: int = 64, base: float = 10000.0):
        """
        Args:
            head_dim: The dimension of each attention head. Must be even.
            max_grid_size: The maximum dimension of the ARC grid (64 for AGI-3).
            base: The base for the exponential frequency decay.
        """
        super().__init__()
        
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be strictly divisible by 2, got {head_dim}")
            
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.max_grid_size = max_grid_size
        self.base = base
        
        # Precompute the inverse frequencies for the half-dimension
        # theta_i = 10000^(-2(i-1)/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-cache the cosine and sine matrices for the maximum possible grid size
        # to prevent recomputing these trigonometric functions during the forward pass.
        self._precompute_freqs_cis()

    def _precompute_freqs_cis(self):
        """
        Caches the cos and sin frequencies for absolute grid positions up to max_grid_size.
        """
        t = torch.arange(self.max_grid_size, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        
        # freqs shape: [max_grid_size, half_dim / 2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate frequencies to match the complex rotation pattern
        # emb shape: [max_grid_size, half_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cached_cos", emb.cos(), persistent=False)
        self.register_buffer("cached_sin", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                row_coords: torch.Tensor, 
                col_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D RoPE to queries and keys based on their spatial grid coordinates.
        
        Args:
            q: Query tensor. Shape: [1, total_tokens, num_heads, head_dim]
            k: Key tensor. Shape: [1, total_tokens, num_heads, head_dim]
            row_coords: 1D tensor of row indices [total_tokens]
            col_coords: 1D tensor of column indices [total_tokens]
            
        Returns:
            q_rot, k_rot: The rotated query and key tensors.
        """
        # q and k are packed sequences, so batch dim is 1
        # shape expected: [1, total_tokens, num_heads, head_dim]
        total_tokens = q.shape[1]
        
        # Ensure coordinates are safely bounded
        row_coords = row_coords.clamp(max=self.max_grid_size - 1)
        col_coords = col_coords.clamp(max=self.max_grid_size - 1)

        # Look up cached cos/sin for row and column positions
        # shapes: [total_tokens, half_dim]
        cos_row = self.cached_cos[row_coords]
        sin_row = self.cached_sin[row_coords]
        
        cos_col = self.cached_cos[col_coords]
        sin_col = self.cached_sin[col_coords]

        # Concatenate row and col components to form the full head_dim vector
        # shapes: [total_tokens, head_dim]
        cos_2d = torch.cat([cos_row, cos_col], dim=-1)
        sin_2d = torch.cat([sin_row, sin_col], dim=-1)

        # Reshape to broadcast across heads
        # shapes: [1, total_tokens, 1, head_dim]
        cos_2d = cos_2d.unsqueeze(0).unsqueeze(2)
        sin_2d = sin_2d.unsqueeze(0).unsqueeze(2)

        # Apply the complex rotation mathematically
        q_embed = (q * cos_2d) + (self._rotate_half(q) * sin_2d)
        k_embed = (k * cos_2d) + (self._rotate_half(k) * sin_2d)

        return q_embed, k_embed

# ==========================================
# Integration Hook for diffusion_prior.py
# ==========================================

def inject_2d_rope(q: torch.Tensor, 
                   k: torch.Tensor, 
                   row_coords: torch.Tensor, 
                   col_coords: torch.Tensor, 
                   rope_module: Fused2DRoPE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hook to apply 2D RoPE inside the continuous denoising step.
    In the final Kaggle environment, this wrapper will route directly to 
    the Unsloth fused Triton kernel to save VRAM reads/writes.
    """
    return rope_module(q, k, row_coords, col_coords)