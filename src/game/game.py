from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from src.game.state import GameState

@dataclass
class Game:
    white: nn.Module
    black: nn.Module
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    blockers_board: np.ndarray = field(init=False, repr=False)
    whiteq_board: np.ndarray = field(init=False, repr=False)
    blackq_board: np.ndarray = field(init=False, repr=False)
    is_white_turn: bool = field(default=True)

    def __post_init__(self):
        self.white.to(self.device)
        self.black.to(self.device)

        self.reset()

    def reset(self):
        self.blockers_board = np.zeros((8, 8), dtype=np.int32)
        self.whiteq_board = np.zeros((8, 8), dtype=np.int32)
        self.blackq_board = np.zeros((8, 8), dtype=np.int32)
        self.whiteq_board[0, 4] = 1
        self.blackq_board[7, 3] = 1

    def get_state(self, turn: bool) -> GameState:
        blockers_board_tensor = torch.tensor(self.blockers_board, dtype=torch.float32, device=self.device)
        whiteq_board_tensor = torch.tensor(self.whiteq_board, dtype=torch.float32, device=self.device)
        blackq_board_tensor = torch.tensor(self.blackq_board, dtype=torch.float32, device=self.device)
        is_white_turn_tensor = torch.tensor([turn == self.is_white_turn], dtype=torch.bool, device=self.device)

        # Flip the board according to the player's perspective
        if not turn:
            blockers_board_tensor = blockers_board_tensor.flip(0)
            whiteq_board_tensor = whiteq_board_tensor.flip(0)
            blackq_board_tensor = blackq_board_tensor.flip(0)

        return GameState(
            blockers_board=blockers_board_tensor,
            selfq_board=whiteq_board_tensor if turn else blackq_board_tensor,
            oppq_board=blackq_board_tensor if turn else whiteq_board_tensor,
            is_self_turn=is_white_turn_tensor
        )
