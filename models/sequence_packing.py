import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import List, Tuple, Dict

class GridSequencePacker:
    """
    A utility class to pack variable-length 2D ARC grids into a contiguous 1D buffer.
    
    This enables the use of FlashAttention / xFormers memory-efficient attention
    without wasting FLOPs on <PAD> tokens. It also extracts the 2D spatial 
    coordinates (row, col) necessary for applying 2D RoPE before attention.
    """
    
    @staticmethod
    def pack_grids(grids: List[torch.Tensor], device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        Takes a list of 2D or flattened grid tensors and packs them into a 1D sequence.
        
        Args:
            grids: List of tensors. Can be raw integer grids [H, W] or embedded 
                   token sequences [H * W, embed_dim].
            device: Target torch device.
            
        Returns:
            A dictionary containing:
                - 'packed_sequence': The concatenated 1D tensor.
                - 'cu_seq_lens': Cumulative lengths array [batch_size + 1] (int32).
                - 'max_seq_len': The maximum sequence length in the batch.
                - 'grid_shapes': List of original (H, W) shapes for unpacking/RoPE.
                - 'row_coords': 1D tensor of row indices for each token.
                - 'col_coords': 1D tensor of column indices for each token.
        """
        if device is None:
            device = grids[0].device
            
        packed_tensors = []
        lengths = []
        grid_shapes = []
        row_coords_list = []
        col_coords_list = []
        
        for grid in grids:
            # Handle both raw grids [H, W] and embedded sequence grids [H, W, D]
            if grid.dim() == 2:
                h, w = grid.shape
                flattened = grid.view(-1)
            elif grid.dim() == 3:
                h, w, d = grid.shape
                flattened = grid.view(-1, d)
            else:
                raise ValueError(f"Expected 2D or 3D grid, got shape {grid.shape}")
                
            grid_shapes.append((h, w))
            seq_len = h * w
            lengths.append(seq_len)
            packed_tensors.append(flattened)
            
            # Generate 2D spatial coordinates for Unsloth 2D RoPE
            rows, cols = torch.meshgrid(
                torch.arange(h, device=device), 
                torch.arange(w, device=device), 
                indexing='ij'
            )
            row_coords_list.append(rows.flatten())
            col_coords_list.append(cols.flatten())

        # 1. Concatenate into a single contiguous buffer
        packed_sequence = torch.cat(packed_tensors, dim=0)
        
        # 2. Build cumulative sequence lengths (Must be int32 for xFormers)
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
        cu_seq_lens = torch.zeros(len(grids) + 1, dtype=torch.int32, device=device)
        cu_seq_lens[1:] = torch.cumsum(lengths_tensor, dim=0)
        
        # 3. Concatenate coordinate maps
        row_coords = torch.cat(row_coords_list, dim=0)
        col_coords = torch.cat(col_coords_list, dim=0)
        
        return {
            "packed_sequence": packed_sequence,
            "cu_seq_lens": cu_seq_lens,
            "max_seq_len": int(lengths_tensor.max().item()),
            "grid_shapes": grid_shapes,
            "row_coords": row_coords,
            "col_coords": col_coords
        }

    @staticmethod
    def unpack_sequence(packed_sequence: torch.Tensor, 
                        grid_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """
        Reverses the packing process, returning a list of 2D/3D grids.
        Useful when translating continuous token probabilities back into a grid output.
        """
        unpacked_grids = []
        current_idx = 0
        
        for (h, w) in grid_shapes:
            seq_len = h * w
            # Extract sequence slice
            seq_slice = packed_sequence[current_idx : current_idx + seq_len]
            current_idx += seq_len
            
            # Reshape back to original 2D or 3D geometry
            if seq_slice.dim() == 1:
                unpacked_grids.append(seq_slice.view(h, w))
            else:
                # Assuming [seq_len, embed_dim]
                d = seq_slice.shape[-1]
                unpacked_grids.append(seq_slice.view(h, w, d))
                
        return unpacked_grids

# ==========================================
# Memory-Efficient Attention Wrapper
# ==========================================

def execute_packed_attention(query: torch.Tensor, 
                             key: torch.Tensor, 
                             value: torch.Tensor, 
                             cu_seq_lens: torch.Tensor, 
                             max_seq_len: int) -> torch.Tensor:
    """
    Executes scaled dot-product attention on packed 1D sequences.
    
    Args:
        query, key, value: Tensors of shape [total_tokens, num_heads, head_dim].
        cu_seq_lens: int32 tensor of shape [batch_size + 1] outlining sequence boundaries.
        max_seq_len: The maximum single sequence length in the packed batch.
        
    Returns:
        attn_output: Tensor of shape [total_tokens, num_heads, head_dim].
    """
    
    # xFormers / memory-efficient backend requires a dummy batch dimension of 1
    # Shape becomes [1, total_tokens, num_heads, head_dim]
    q = query.unsqueeze(0)
    k = key.unsqueeze(0)
    v = value.unsqueeze(0)
    
    # We strictly force the PyTorch SDPA dispatcher to use the Memory Efficient Backend.
    # On a T4, this maps to the highly optimized xFormers kernel which respects cu_seq_lens.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        # We must pass the mask as a sequence of boolean flags or utilize the 
        # customized xFormers BlockDiagonal mask if working outside native SDPA.
        # In PyTorch 2.2+, NestedTensors inherently route through the block-diagonal logic.
        
        # NOTE: If query/key/value were cast as true torch.nested.nested_tensor earlier,
        # SDPA handles this implicitly. Because we manage a raw 1D contiguous tensor,
        # we format it using the explicit Flash/xFormers signature wrapper if needed.
        # Here we rely on the native SDPA handling assuming inputs are appropriately nested or masked.
        
        # For raw 1D packed tensors in modern PyTorch SDPA, the easiest native integration 
        # is passing the NestedTensor representation:
        pass
    
    raise NotImplementedError("Requires PyTorch NestedTensor conversion prior to SDPA call.")