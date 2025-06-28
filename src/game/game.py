from dataclasses import dataclass, field
from typing import Any
import numpy as np
from src.game.state import GameState

@dataclass
class Game:
    white: Any
    black: Any
    blockers_board: np.ndarray = field(init=False, repr=False)
    whiteq_board: np.ndarray = field(init=False, repr=False)
    blackq_board: np.ndarray = field(init=False, repr=False)
    is_white_turn: bool = field(default=True)
    result: bool | None = field(default=None, init=False, repr=False)
    _state_cache_white: GameState | None = field(default=None, init=False, repr=False)
    _state_cache_black: GameState | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.blockers_board = np.zeros((8, 8), dtype=np.int32)
        self.whiteq_board = np.zeros((8, 8), dtype=np.int32)
        self.blackq_board = np.zeros((8, 8), dtype=np.int32)
        self.reset()

    def reset(self):
        self.blockers_board.fill(0)
        self.whiteq_board.fill(0)
        self.blackq_board.fill(0)
        self.whiteq_board[0, 4] = 1
        self.blackq_board[7, 3] = 1
        self.is_white_turn = True
        self.result = None
        self._state_cache_white = None
        self._state_cache_black = None

    def get_state(self, turn: bool | None = None) -> GameState:
        if turn is None:
            turn = self.is_white_turn
        if turn and self._state_cache_white is not None:
            return self._state_cache_white
        if not turn and self._state_cache_black is not None:
            return self._state_cache_black

        res = GameState(
            blockers_board=self.blockers_board,
            selfq_board=self.whiteq_board if turn else self.blackq_board,
            oppq_board=self.blackq_board if turn else self.whiteq_board,
            is_self_turn=turn
        )
        if turn:
            self._state_cache_white = res
        else:
            self._state_cache_black = res
        return res

    def make_move(self, move: tuple[int, int], turn: bool | None = None, _mask_cache: np.ndarray | None = None):
        assert self.result is None, "Game is already over!"

        if turn is None:
            turn = self.is_white_turn

        assert turn == self.is_white_turn, "It's not your turn!"

        state = self.get_state(turn)
        current_queen_pos = state.find_queen_position()
        if _mask_cache is not None:
            legal_moves_mask = _mask_cache.copy()
        else:
            legal_moves_mask = state.mask_legal_moves()
        assert legal_moves_mask[move], "Illegal move!"

        self.blockers_board[current_queen_pos] = 1
        if turn:
            self.whiteq_board[current_queen_pos] = 0
            self.whiteq_board[move] = 1
            if self.blackq_board[move] == 1:
                self.result = True
            self.blackq_board[move] = 0
        else:
            self.blackq_board[current_queen_pos] = 0
            self.blackq_board[move] = 1
            if self.whiteq_board[move] == 1:
                self.result = False
            self.whiteq_board[move] = 0
        self.is_white_turn = not self.is_white_turn

        self._state_cache_white = None
        self._state_cache_black = None

        # check if there is suffocation
        if self.result is not None:
            return
        new_state = self.get_state(self.is_white_turn)
        if new_state.mask_legal_moves().sum() == 0:
            self.result = not self.is_white_turn

    def agent_move(self) -> tuple[np.ndarray, int, bool]:
        state = self.get_state()
        agent = self.white if self.is_white_turn else self.black

        action_probs = agent.play(state)
        legal_moves_mask = state.mask_legal_moves()
        mask = legal_moves_mask.astype(np.float32)
        legal_action_probs = action_probs * mask
        legal = True

        legal_action_probs /= legal_action_probs.sum()
        if np.isnan(legal_action_probs).any():
            legal = False
            legal_action_probs.fill(1.0 / legal_moves_mask.sum())
            legal_action_probs *= mask

        move = np.random.choice(np.arange(64), p=legal_action_probs.flatten())
        movee = (move // 8, move % 8)
        self.make_move(movee, _mask_cache=legal_moves_mask)

        return action_probs, move, legal

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

    def heuristic_reward(self) -> float:
        if self.result is not None:
            return 0.0
        state = self.get_state()
        legal_moves_num = state.mask_legal_moves().sum()
        oppo_state = self.get_state(not self.is_white_turn)
        oppo_legal_moves_num = oppo_state.mask_legal_moves().sum()
        return legal_moves_num / 64 - oppo_legal_moves_num / 64
