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

    def find_queen_position(self) -> tuple[int, int]:
        assert (self.selfq_board == 1).sum() == 1, "There should be exactly one queen on the self board."
        assert (self.selfq_board == 0).sum() == 63, "There should be no other queens on the self board."
        queen_pos = torch.nonzero(self.selfq_board, as_tuple=False).squeeze(0)
        res = (queen_pos[0].item(), queen_pos[1].item())
        assert isinstance(res[0], int) and isinstance(res[1], int), "Queen position should be integers."
        return res # type: ignore

    def mask_legal_moves(self) -> torch.Tensor:
        mask = torch.zeros_like(self.selfq_board, dtype=torch.bool)
        selfq_pos = self.find_queen_position()
        selfq_row, selfq_col = selfq_pos

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1)
        ]

        for dr, dc in directions:
            r, c = selfq_row, selfq_col
            r += dr
            c += dc
            while 0 <= r < 8 and 0 <= c < 8:
                if self.blockers_board[r, c] == 1:
                    break
                if self.oppq_board[r, c] == 1:
                    mask[r, c] = True
                    break
                mask[r, c] = True
                r += dr
                c += dc

        return mask
