import torch
import torch.multiprocessing as mp
import ctypes
import numpy as np
from typing import Tuple, Dict, List

# Enforce the 'spawn' start method for PyTorch multiprocessing.
# 'fork' causes CUDA context corruption and Copy-on-Write memory leaks.
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class IPCMemoryManager:
    """
    Manages the shared memory segments and communication queues for the 
    Asynchronous MCTS / GPU Batch Server architecture.
    """
    
    def __init__(self, 
                 num_slots: int = 256, 
                 state_bytes: int = 4096, 
                 num_workers: int = 8):
        """
        Allocates the shared memory buffers and IPC queues.
        
        Args:
            num_slots: Maximum number of concurrent neural evaluations in flight.
            state_bytes: Maximum byte size of a serialized ARC-AGI-3 game state.
            num_workers: Number of CPU MCTS workers.
        """
        self.num_slots = num_slots
        self.state_bytes = state_bytes
        self.num_workers = num_workers
        
        # 1. Allocate Shared Memory Buffers (Zero-Copy across processes)
        # Flat byte array for game states (Inputs)
        self.shared_states = mp.Array(ctypes.c_uint8, num_slots * state_bytes, lock=False)
        
        # Flat float array for TRM stability scores / value estimates (Outputs)
        self.shared_scores = mp.Array(ctypes.c_float, num_slots, lock=False)
        
        # 2. Allocate IPC Queues
        # Workers push (worker_id, slot_id) here to request a GPU evaluation
        self.request_queue = mp.Queue()
        
        # GPU pushes the completed slot_id back to the specific worker's queue
        self.result_queues = {i: mp.Queue() for i in range(num_workers)}
        
        # 3. Slot Management (Collision Prevention)
        # Thread-safe queue containing all available slot indices [0, 1, ..., num_slots - 1]
        self.available_slots = mp.Queue()
        for i in range(num_slots):
            self.available_slots.put(i)

    def get_worker_interfaces(self) -> List[Dict]:
        """
        Returns a list of interface dictionaries, one for each CPU worker.
        These contain only the specific queues and memory pointers that worker needs.
        """
        interfaces = []
        for worker_id in range(self.num_workers):
            interfaces.append({
                "worker_id": worker_id,
                "request_queue": self.request_queue,
                "result_queue": self.result_queues[worker_id],
                "available_slots": self.available_slots,
                "shared_states": self.shared_states,
                "shared_scores": self.shared_scores,
                "state_bytes": self.state_bytes
            })
        return interfaces
        
    def get_gpu_server_interface(self) -> Dict:
        """
        Returns the interface dictionary for the GPU Batch Server process.
        """
        return {
            "request_queue": self.request_queue,
            "result_queues": self.result_queues,
            "shared_states": self.shared_states,
            "shared_scores": self.shared_scores,
            "state_bytes": self.state_bytes
        }

# ==========================================
# Worker-Side Helper Functions
# ==========================================

class IPCWorkerClient:
    """
    A lightweight client instantiated inside each CPU worker process to interact
    with the shared memory buffers safely.
    """
    def __init__(self, interface: Dict):
        self.worker_id = interface["worker_id"]
        self.request_queue = interface["request_queue"]
        self.result_queue = interface["result_queue"]
        self.available_slots = interface["available_slots"]
        self.state_bytes = interface["state_bytes"]
        
        # Create fast NumPy views over the raw multiprocessing memory.
        # This prevents allocating new memory when writing the state bytes.
        self._states_buffer = np.frombuffer(interface["shared_states"], dtype=np.uint8)
        self._scores_buffer = np.frombuffer(interface["shared_scores"], dtype=np.float32)

    def evaluate_state(self, serialized_state: np.ndarray) -> float:
        """
        Synchronous wrapper around the async IPC architecture. 
        The CPU worker calls this, writes to memory, and sleeps until the GPU 
        notifies it that the score is ready.
        """
        if len(serialized_state) > self.state_bytes:
            raise ValueError(f"State size {len(serialized_state)} exceeds max {self.state_bytes}")
            
        # 1. Checkout an available memory slot (blocks if GPU is completely backed up)
        slot_id = self.available_slots.get()
        
        # 2. Write the state directly into the shared RAM buffer (Zero-copy)
        start_idx = slot_id * self.state_bytes
        end_idx = start_idx + len(serialized_state)
        self._states_buffer[start_idx:end_idx] = serialized_state
        
        # 3. Notify the GPU Batch Server
        self.request_queue.put((self.worker_id, slot_id))
        
        # 4. Yield the CPU and sleep until the GPU finishes the batch and pings us back
        # The result queue will yield the slot_id once computation is done
        returned_slot_id = self.result_queue.get()
        assert returned_slot_id == slot_id, "IPC routing error: received mismatched slot ID."
        
        # 5. Read the float score calculated by the TRM
        score = float(self._scores_buffer[slot_id])
        
        # 6. Return the slot to the available pool
        self.available_slots.put(slot_id)
        
        return score