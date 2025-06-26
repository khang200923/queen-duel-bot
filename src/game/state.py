from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

@dataclass
class GameState:
    blockers_board: torch.Tensor
    selfq_board: torch.Tensor
    oppq_board: torch.Tensor
    is_self_turn: torch.Tensor
