from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from src.game.state import GameState

@dataclass
class Game:
    white: Any
    black: Any
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    blockers_board: np.ndarray = field(init=False, repr=False)
    whiteq_board: np.ndarray = field(init=False, repr=False)
    blackq_board: np.ndarray = field(init=False, repr=False)
    is_white_turn: bool = field(default=True)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.blockers_board = np.zeros((8, 8), dtype=np.int32)
        self.whiteq_board = np.zeros((8, 8), dtype=np.int32)
        self.blackq_board = np.zeros((8, 8), dtype=np.int32)
        self.whiteq_board[0, 4] = 1
        self.blackq_board[7, 3] = 1

    def get_state(self, turn: bool | None = None) -> GameState:
        if turn is None:
            turn = self.is_white_turn

        blockers_board_tensor = torch.tensor(self.blockers_board, dtype=torch.float32, device=self.device)
        whiteq_board_tensor = torch.tensor(self.whiteq_board, dtype=torch.float32, device=self.device)
        blackq_board_tensor = torch.tensor(self.blackq_board, dtype=torch.float32, device=self.device)

        return GameState(
            blockers_board=blockers_board_tensor,
            selfq_board=whiteq_board_tensor if turn else blackq_board_tensor,
            oppq_board=blackq_board_tensor if turn else whiteq_board_tensor,
            is_self_turn=turn
        )

    def make_move(self, move: tuple[int, int], turn: bool | None = None):
        assert self.result() is None, "Game is already over!"

        if turn is None:
            turn = self.is_white_turn

        assert turn == self.is_white_turn, "It's not your turn!"

        state = self.get_state(turn)
        current_queen_pos = state.find_queen_position()
        legal_moves_mask = state.mask_legal_moves()
        assert legal_moves_mask[move], "Illegal move!"

        self.blockers_board[current_queen_pos] = 1
        if turn:
            self.whiteq_board = np.zeros((8, 8), dtype=np.int32)
            self.whiteq_board[move] = 1
            self.blackq_board[move] = 0
        else:
            self.blackq_board = np.zeros((8, 8), dtype=np.int32)
            self.blackq_board[move] = 1
            self.whiteq_board[move] = 0
        self.is_white_turn = not self.is_white_turn

    def agent_move(self) -> tuple[np.ndarray, tuple[int, int], bool]:
        state = self.get_state()
        agent = self.white if self.is_white_turn else self.black

        action_probs = agent.play(state)
        legal_moves_mask = state.mask_legal_moves().cpu().numpy()
        legal_action_probs = action_probs * legal_moves_mask.astype(np.float32)
        legal = True
        try:
            legal_action_probs /= legal_action_probs.sum()
        except ZeroDivisionError:
            legal = False
            legal_action_probs.fill(1.0 / legal_moves_mask.sum())
            legal_action_probs *= legal_moves_mask.astype(np.float32)
        if np.isnan(legal_action_probs).any():
            legal = False
            legal_action_probs.fill(1.0 / legal_moves_mask.sum())
            legal_action_probs *= legal_moves_mask.astype(np.float32)

        move = np.random.choice(np.arange(64), p=legal_action_probs.flatten())
        move = (move // 8, move % 8)
        self.make_move(move)

        return action_probs, move, legal

    def result(self) -> bool | None:
        state = self.get_state()
        if (state.selfq_board == 1).sum() == 0:
            return False
        if (state.oppq_board == 1).sum() == 0:
            return True
        if not state.mask_legal_moves().any():
            return not self.is_white_turn # suffocation -> ded
        return None

    def repr_board(self) -> str:
        board = np.zeros((8, 8), dtype=str)
        for i in range(8):
            for j in range(8):
                if self.blockers_board[i, j] == 1:
                    board[i, j] = '█'
                elif self.whiteq_board[i, j] == 1:
                    board[i, j] = 'W'
                elif self.blackq_board[i, j] == 1:
                    board[i, j] = 'B'
                else:
                    board[i, j] = '.'
        return '\n'.join([' '.join(row) for row in board])
