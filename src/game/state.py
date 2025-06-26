from dataclasses import dataclass, field
import numpy as np

@dataclass
class GameState:
    blockers_board: np.ndarray
    selfq_board: np.ndarray
    oppq_board: np.ndarray
    is_self_turn: bool

    def find_queen_position(self) -> tuple[int, int]:
        assert (self.selfq_board == 1).sum() == 1, "There should be exactly one queen on the self board."
        assert (self.selfq_board == 0).sum() == 63, "There should be no other queens on the self board."
        queen_pos = np.argwhere(self.selfq_board == 1)
        res = (queen_pos[0].item(), queen_pos[1].item())
        assert isinstance(res[0], int) and isinstance(res[1], int), "Queen position should be integers."
        return res # type: ignore

    def mask_legal_moves(self) -> np.ndarray:
        mask = np.zeros_like(self.selfq_board, dtype=bool)
        r0, c0 = self.find_queen_position()

        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ])

        for dr, dc in directions:
            dr_it: int = dr.item() # type: ignore
            dc_it: int = dc.item() # type: ignore
            r, c = r0 + dr_it, c0 + dc_it
            while 0 <= r < 8 and 0 <= c < 8:
                if self.blockers_board[r, c] == 1:
                    break
                mask[r, c] = True
                if self.oppq_board[r, c] == 1:
                    break
                r += dr_it
                c += dc_it

        return mask
