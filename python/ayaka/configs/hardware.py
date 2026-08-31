from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from typing import Final, TypedDict

CC_LIMITS: dict[tuple[int, int], dict[str, int]] = {
    # Pascal
    # GP100 (Tesla P100)
    (6, 0): dict(
        shared_per_block=64 << 10,
        shared_per_sm=64 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GP102/104/106 (GTX 1080 Ti, Tesla P40, P4)
    (6, 1): dict(
        shared_per_block=48 << 10,
        shared_per_sm=96 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GP10B (Tegra Parker)
    (6, 2): dict(
        shared_per_block=48 << 10,
        shared_per_sm=48 << 10,
        regs_per_sm=32768,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # Volta 
    # GV100 (Tesla V100, Titan V)
    (7, 0): dict(
        shared_per_block=96 << 10,
        shared_per_sm=96 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GV10B (Jetson AGX Xavier)
    (7, 2): dict(
        shared_per_block=96 << 10,
        shared_per_sm=96 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # Turing 
    # TU102/104/106/116 (RTX 2080 Ti, Tesla T4)
    (7, 5): dict(
        shared_per_block=64 << 10,
        shared_per_sm=64 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1024,
        max_threads_per_block=1024,
        max_blocks_per_sm=16,
    ),
    # Ampere 
    # GA100 (A100, A30)
    (8, 0): dict(
        shared_per_block=163 << 10,
        shared_per_sm=164 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GA102/104/106 (RTX 3090, A10, A40, RTX A6000)
    (8, 6): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=16,
    ),
    # GA10B (Jetson AGX Orin)
    (8, 7): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=16,
    ),
    # Ada Lovelace 
    # AD102/103/104 (RTX 4090, L40, L40S, L4)
    (8, 9): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=24,
    ),
    # Hopper 
    # GH100 (H100, H200, GH200)
    (9, 0): dict(
        shared_per_block=227 << 10,
        shared_per_sm=228 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # Blackwell
    # GB100/GB200 (B100, B200, GB200 - Data Center)
    (10, 0): dict(
        shared_per_block=227 << 10,
        shared_per_sm=228 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GB20x (RTX 5090, RTX 5080 - Client / Workstation)
    (12, 0): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=24,
    ),
}
