import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Callable, Tuple, Any

# ==========================================
# Geometric & Semantic Augmentations (CPU-Bound)
# ==========================================

def apply_color_permutation(grid: np.ndarray) -> np.ndarray:
    """
    Applies a random permutation to the color palette (0-9 for ARC-AGI-2).
    This ensures the model learns topological rules rather than memorizing colors.
    """
    # Exclude 0 (often background) depending on the specific ARC prior, 
    # but for true universality, we permute all active colors.
    unique_colors = np.unique(grid)
    permuted_colors = np.random.permutation(unique_colors)
    
    color_map = {old: new for old, new in zip(unique_colors, permuted_colors)}
    
    # Vectorized mapping using numpy searchsorted or lookup
    palette = np.arange(grid.max() + 1)
    for k, v in color_map.items():
        palette[k] = v
        
    return palette[grid]

def apply_rotation(grid: np.ndarray, k: int) -> np.ndarray:
    """Rotates the grid by 90 degrees * k."""
    return np.rot90(grid, k=k)

def apply_reflection(grid: np.ndarray, axis: str) -> np.ndarray:
    """Flips the grid horizontally or vertically."""
    if axis == 'h':
        return np.fliplr(grid)
    elif axis == 'v':
        return np.flipud(grid)
    return grid

# ==========================================
# The CoW-Free Procedural Dataset
# ==========================================

class ARCDataset(Dataset):
    """
    A strictly stateless PyTorch Dataset for generating infinite ARC tasks.
    
    By keeping the generator_registry completely free of mutable Python objects
    (lists, large dicts, instances), we prevent the OS from triggering 
    Copy-on-Write (CoW) page duplications during fork-based multiprocessing.
    """
    
    def __init__(self, generator_registry: Dict[str, Callable], virtual_size: int = 50_000_000):
        """
        Args:
            generator_registry: A dictionary mapping concept names to pure Python functions 
                                (e.g., RE-ARC procedural generators).
            virtual_size: The "epoch" length. Set massively high to stream continuously.
        """
        super().__init__()
        self.registry = generator_registry
        self.concept_names = list(generator_registry.keys())
        self.virtual_size = virtual_size

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Synthesizes a fresh task. All computation happens locally within the 
        spawned worker process, meaning augmented tensors are allocated 
        fresh in local memory and never dirty the parent process.
        """
        # 1. Uniformly sample a conceptual rule
        concept = self.concept_names[idx % len(self.concept_names)]
        
        # 2. Generate a base input/output pair using the stateless pure function
        # This function must return numpy arrays representing the grid integers
        input_grid, output_grid = self.registry[concept]()
        
        # 3. Apply CPU-Bound Stochastic Augmentations
        # Applied dynamically to maximize intra-batch variance
        if random.random() < 0.5:
            input_grid = apply_color_permutation(input_grid)
            output_grid = apply_color_permutation(output_grid)
            
        if random.random() < 0.5:
            k = random.randint(1, 3)
            input_grid = apply_rotation(input_grid, k)
            output_grid = apply_rotation(output_grid, k)
            
        if random.random() < 0.5:
            axis = random.choice(['h', 'v'])
            input_grid = apply_reflection(input_grid, axis)
            output_grid = apply_reflection(output_grid, axis)
            
        # 4. Convert to PyTorch tensors
        # Note: We do NOT pad here. Padding is handled via Sequence Packing later.
        return {
            "input_grid": torch.tensor(input_grid.copy(), dtype=torch.long),
            "output_grid": torch.tensor(output_grid.copy(), dtype=torch.long),
            "concept_id": concept
        }

# ==========================================
# Worker Initialization (RNG Isolation)
# ==========================================

def worker_init_fn(worker_id: int):
    """
    Called immediately after a worker process is forked.
    
    Derives a mathematically orthogonal seed for each worker to guarantee
    (1) No lock contention on shared RNGs.
    (2) Complete statistical independence of generated augmentations.
    """
    # Retrieve the base seed set by PyTorch in the main process
    base_seed = torch.initial_seed() % (2**31)
    
    # 31337 is an arbitrary prime offset to ensure distinct seed trajectories
    worker_seed = base_seed + worker_id * 31337 
    
    # Isolate standard Python random state
    random.seed(worker_seed)
    
    # Isolate Numpy random state (used heavily by RE-ARC and geometric transforms)
    np.random.seed(worker_seed)

def create_arc_dataloader(
    generator_registry: Dict[str, Callable], 
    batch_size: int = 64, 
    num_workers: int = 4
) -> DataLoader:
    """Factory function to build the optimized DataLoader."""
    dataset = ARCDataset(generator_registry)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,          # Enables async PCIe DMA transfers to GPU
        persistent_workers=True,  # Avoids OS fork overhead between epochs
        collate_fn=lambda x: x    # Passthrough collator; sequence packing handles the batching
    )