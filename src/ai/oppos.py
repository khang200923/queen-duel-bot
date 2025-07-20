import numpy as np

from src.game.state import GameState

class RandomAgent:
    def __init__(self):
        pass

    def play(self, state: GameState) -> np.ndarray:
        return np.ones((8, 8), dtype=np.float32) / 64.0

class SimpleCleverAgent:
    def __init__(self):
        pass

    def play(self, state: GameState) -> np.ndarray:
        hmm = state.mask_legal_moves() * state.oppq_board
        if hmm.sum() > 0.001:
            return hmm
        return np.ones((8, 8), dtype=np.float32) / 64.0

class BetterSimpleCleverAgent:
    def __init__(self):
        pass

    def play(self, state: GameState) -> np.ndarray:
        hmm = state.mask_legal_moves() * state.oppq_board
        if hmm.sum() > 0.001:
            return hmm
        oppo_state = GameState(
            blockers_board=state.blockers_board,
            selfq_board=state.oppq_board,
            oppq_board=state.selfq_board,
            is_self_turn=not state.is_self_turn
        )
        hmm = (np.ones((8, 8), dtype=np.float32) - oppo_state.mask_legal_moves()) / 64.0
        if hmm.sum() < 0.001:
            return np.ones((8, 8), dtype=np.float32) / 64.0
        return hmm
    
class IntentionallyDumbAgent:
    def __init__(self):
        pass

    def play(self, state: GameState) -> np.ndarray:
        oppo_state = GameState(
            blockers_board=state.blockers_board,
            selfq_board=state.oppq_board,
            oppq_board=state.selfq_board,
            is_self_turn=not state.is_self_turn
        )
        hmm = state.mask_legal_moves() * oppo_state.mask_legal_moves()
        if hmm.sum() > 0.001:
            return hmm
        return np.ones((8, 8), dtype=np.float32) / 64.
