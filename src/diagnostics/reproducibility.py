"""Seed and deterministic execution helpers."""

import random

import numpy as np
import torch

TORCH_CPU_THREADS = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_CPU_THREADS)
    torch.use_deterministic_algorithms(True)
