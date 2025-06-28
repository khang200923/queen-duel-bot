from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class GameState:
    blockers_board: np.ndarray # shape (8, 8)
    selfq_board: np.ndarray # shape (8, 8)
    oppq_board: np.ndarray # shape (8, 8)
    is_self_turn: bool
    _mask_cache: np.ndarray | None = field(default=None, repr=False)

    def find_queen_position(self) -> tuple[int, int]:
        assert (self.selfq_board == 1).sum() == 1, "There should be exactly one queen on the self board."
        assert (self.selfq_board == 0).sum() == 63, "There should be no other queens on the self board."
        # print("fdaadf", np.argwhere(self.selfq_board == 1))
        queen_pos = np.argwhere(self.selfq_board == 1).squeeze(0)
        res = (queen_pos[0].item(), queen_pos[1].item())
        assert isinstance(res[0], int) and isinstance(res[1], int), "Queen position should be integers."
        return res # type: ignore

    def mask_legal_moves(self) -> np.ndarray:
        if self._mask_cache is not None:
            return self._mask_cache

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

        self._mask_cache = mask.copy()
        return mask

    def convert_to_torch(self, device: torch.device | None = None) -> 'GameStateTorch':
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blockers_board_tensor = torch.tensor(self.blockers_board, dtype=torch.float32, device=device)
        selfq_board_tensor = torch.tensor(self.selfq_board, dtype=torch.float32, device=device)
        oppq_board_tensor = torch.tensor(self.oppq_board, dtype=torch.float32, device=device)
        return GameStateTorch(
            blockers_board=blockers_board_tensor,
            selfq_board=selfq_board_tensor,
            oppq_board=oppq_board_tensor
        )

@dataclass
class GameStateTorch:
    blockers_board: torch.Tensor
    selfq_board: torch.Tensor
    oppq_board: torch.Tensor

    def all_boards(self) -> torch.Tensor:
        return torch.stack([self.selfq_board, self.oppq_board, self.blockers_board], dim=0)
