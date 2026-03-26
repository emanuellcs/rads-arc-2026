# RADS: Recursive Active-Diffusion Synthesis

[![ARC Prize 2026](https://img.shields.io/badge/ARC_Prize_2026-Competitor-blue)](https://arcprize.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

RADS (Recursive Active-Diffusion Synthesis) is a unified neuro-symbolic architecture designed to solve the ARC Prize 2026 competitions. This single repository contains the code to compete in **ARC-AGI-2** (Static Prediction), **ARC-AGI-3** (Interactive Agency), and the **ARC Paper Track** without altering the core neural weights.

## Overview

Standard autoregressive language models struggle with true fluid intelligence because they rely on sequential pattern matching and lack internal spatial geometry. RADS solves this by framing abstract reasoning as hypothesis synthesis followed by falsification-driven convergence.

The system couples two primary components:
1. **The Dreamer:** An 8-billion parameter Masked Diffusion Language Model (MDLM) loaded in 4-bit NF4 precision. It uses continuous token algebra and Unsloth's fused 2D RoPE to generate highly structural grid hypotheses.
2. **The Verifier:** A 7-million parameter Tiny Recursive Model (TRM). Operating as a Banach contraction mapping, it evaluates hypotheses by checking for fixed-point convergence (Aizawa attractor) or chaotic divergence.

## Core Engineering Features

Built strictly for heavily constrained Kaggle Notebooks and Google Colab environments (e.g., Dual T4, L4, or P100 GPUs).

* **Sequence Packing:** Drops `<PAD>` tokens entirely, utilizing 1D `NestedTensors` and `cu_seq_lens` with xFormers to maximize Tensor Core utilization.
* **Compiled Verification:** The TRM is wrapped in `torch.compile(mode="reduce-overhead")` with CUDA Graphs, executing hundreds of verification passes per second.
* **Copy-on-Write (CoW) Defeat:** A highly optimized PyTorch `DataLoader` employing stateless RE-ARC generators, dynamically seeded RNGs (`worker_init_fn`), and CPU-bound augmentations to prevent RAM fragmentation.
* **GIL-Bypassing Async MCTS:** A multi-process Inter-Process Communication (IPC) architecture where CPU workers execute Monte Carlo Tree Search (MCTS) via an internal Python simulator, while a dedicated GPU server dynamically batches neural evaluations.

## Repository Structure

```text
rads-arc-2026/
│
├── data/                      # Procedural generation and CoW-free DataLoaders
│   ├── dataset.py             
│   ├── transforms.py          
│   └── re_arc_generators/     
│
├── models/                    # Neural architectures
│   ├── diffusion_prior.py     # 8B Masked Diffusion & Token Algebra
│   ├── rope_2d.py             # Fused 2D RoPE
│   ├── sequence_packing.py    # NestedTensor logic
│   └── trm_verifier.py        # 7M Recursive Thermodynamic Verifier
│
├── agent/                     # ARC-AGI-3 Agentic Framework
│   ├── mcts.py                # Monte Carlo Tree Search
│   ├── epistemic_foraging.py  # MVP probes & HPC stopping criteria
│   └── physics_simulator.py   # Internal Python world model
│
├── orchestrator/              # IPC and Hardware Management
│   ├── gpu_batch_server.py    # Async dynamic batching
│   └── shared_memory.py       # IPC queues
│
├── scripts/                   # Competition Entry Points
│   ├── run_arc_agi_2_ttt.py   # Test-Time Training (12-hour limit)
│   └── run_arc_agi_3_agent.py # Interactive Swarm (6-hour limit)
│        
└── requirements.txt
```

## Getting Started

*Note: Full installation instructions and dependency locking will be provided in `requirements.txt`.*

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/emanuellcs/rads-arc-2026.git](https://github.com/emanuellcs/rads-arc-2026.git)
   cd rads-arc-2026
   ```
2. **Install dependencies:**
   Ensure you have a CUDA 12.1+ compatible environment.
   ```bash
   pip install -r requirements.txt
   ```

## Competition Targets

* **ARC-AGI-2:** Target runtime $< 12$ hours (Offline).
* **ARC-AGI-3:** Target runtime $< 6$ hours (Offline). Target RHAE score $= 1.0x$ (via the Decoupled Thinking Loop and RESET exploit).