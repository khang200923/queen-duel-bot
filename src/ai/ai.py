from dataclasses import dataclass
from typing import Any
import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn

from src.ai.oppos import IntentionallyDumbAgent, SimpleCleverAgent
import src.game.game
import src.game.state
from src.game.game import Game
from src.game.state import GameState, GameStateTorch

import torch
import torch.nn as nn

class SpatialBias(nn.Module): # credit to ChatGPT
    def __init__(self, board_size, num_channels):
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels

        # Learnable biases
        self.row_bias = nn.Parameter(torch.zeros(num_channels, board_size))               # (C, H)
        self.col_bias = nn.Parameter(torch.zeros(num_channels, board_size))               # (C, W)
        self.diag_bias = nn.Parameter(torch.zeros(num_channels, 2 * board_size - 1))      # (C, 2H-1)
        self.anti_diag_bias = nn.Parameter(torch.zeros(num_channels, 2 * board_size - 1)) # (C, 2H-1)

        # Precompute diagonal and anti-diagonal index maps
        i = torch.arange(board_size).view(board_size, 1)  # (H, 1)
        j = torch.arange(board_size).view(1, board_size)  # (1, W)

        diag_indices = i - j + (board_size - 1)           # (H, W), values in [0, 2H-2]
        anti_diag_indices = i + j                         # (H, W), values in [0, 2H-2]

        # Register as buffers so they move with the model
        self.register_buffer("diag_indices", diag_indices)
        self.register_buffer("anti_diag_indices", anti_diag_indices)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device
        assert H == W == self.board_size, "Input spatial dims must match board_size"
        assert C == self.num_channels, "Input channels must match num_channels"

        # Row and column biases
        row = self.row_bias.view(1, C, H, 1)  # (1, C, H, 1)
        col = self.col_bias.view(1, C, 1, W)  # (1, C, 1, W)

        # Expand diag and anti-diag index maps to (1, C, H, W)
        diag_idx = self.diag_indices.unsqueeze(0).expand(C, H, W).unsqueeze(0)       # (1, C, H, W)
        anti_diag_idx = self.anti_diag_indices.unsqueeze(0).expand(C, H, W).unsqueeze(0)  # (1, C, H, W)

        # Expand diag and anti-diag bias to (1, C, H, 2H-1)
        diag_bias_exp = self.diag_bias.view(1, C, 1, 2 * H - 1).expand(1, C, H, 2 * H - 1)
        anti_diag_bias_exp = self.anti_diag_bias.view(1, C, 1, 2 * H - 1).expand(1, C, H, 2 * H - 1)

        # Gather biases using the index maps
        diag = torch.gather(diag_bias_exp, 3, diag_idx)             # (1, C, H, W)
        anti_diag = torch.gather(anti_diag_bias_exp, 3, anti_diag_idx)  # (1, C, H, W)

        # Total bias
        total_bias = row + col + diag + anti_diag  # (1, C, H, W)

        return x + total_bias


class SmolQAgent(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.bias = SpatialBias(board_size=8, num_channels=4)
        self.proj = nn.Linear(4, 64)
        self.layernorm = nn.LayerNorm(64)
        self.attn1 = nn.MultiheadAttention(64, 8, batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 64)
        )
        self.attn2 = nn.MultiheadAttention(64, 8, batch_first=True)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 64)
        )
        self.attn3 = nn.MultiheadAttention(64, 8, batch_first=True)
        self.out = nn.Linear(64, 1)
        self.head_value = nn.Linear(64, 1)
        self.head_advantage = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def core(x):
            B = x.size(0)
            x = self.bias(x)
            x = x.permute(0, 2, 3, 1).reshape(B, 64, 4)
            x = self.proj(x)

            x = self.layernorm(x)
            attn1_out = self.attn1(x, x, x)[0]
            x = x + attn1_out
            x = x + self.ffn1(x)

            x = self.layernorm(x)
            attn2_out = self.attn2(x, x, x)[0]
            x = x + attn2_out
            x = x + self.ffn2(x)

            x = self.layernorm(x)
            attn3_out = self.attn3(x, x, x)[0]
            x = x + attn3_out

            x = self.out(x)
            x = x.reshape(B, 64)
            v = self.head_value(x)
            a = x + self.head_advantage(x)
            x = v + a - a.mean(dim=1, keepdim=True)
            return x.view(B, 8, 8)
        x = core(x)
        return x.view(-1, 64)

    def play(self, state: GameState) -> np.ndarray:
        state_torch = state.convert_to_torch()
        with torch.no_grad():
            if torch.randn(1) < self.epsilon:
                return SimpleCleverAgent().play(state)
            else:
                res = self(state_torch.all_boards().unsqueeze(0)).squeeze(0)
                illegal_mask = 1 - state_torch.legal_moves_mask.flatten()
                res = res - illegal_mask * 100
                best_ind = torch.argmax(res)
                best_action_probs = torch.full_like(res, 0.0)
                best_action_probs.scatter_(0, best_ind.unsqueeze(0), 1.0)
        return best_action_probs.view(8,8).cpu().numpy()
    
@dataclass
class ReplayEntry:
    state: GameStateTorch
    action: torch.Tensor
    reward: float
    next_state: GameStateTorch
    done: bool