from dataclasses import dataclass, field
from typing import Any, List, Tuple
import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn

from src.ai.ai import ReplayEntry
from src.ai.utils import empty_game_state_torch, get_reward, train_step
import src.game.game
import src.game.state
from src.game.game import Game
from src.game.state import GameState, GameStateTorch

@dataclass
class Player:
    agent: nn.Module
    target: nn.Module = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    buffer: List[ReplayEntry] = field(init=False)

    def __post_init__(self):
        self.target = copy.deepcopy(self.agent)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=5e-4, betas=(0.9, 0.999))
        self.buffer = []

    def train_step(self, batch_size: int):
        sampled_buffer = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return train_step(self.agent, self.target, self.optimizer, sampled_buffer)

    def purge_buffer(self, factor: float = 0.01):
        random.shuffle(self.buffer)
        self.buffer = self.buffer[int(len(self.buffer) * factor):]

    def fight_with(self, oppo: "Player", k: int = 1):
        a_buffer, b_buffer = get_buffer_from_fights(self.agent, oppo.agent, k)
        self.buffer.extend(a_buffer)
        oppo.buffer.extend(b_buffer)

    def fight_with_non_player(self, oppo: Any, k: int = 1):
        a_buffer, b_buffer = get_buffer_from_fights(self.agent, oppo, k)
        self.buffer.extend(a_buffer)

    def get_win_rate(self, oppo: Any, k: int = 100) -> float:
        wins = 0
        for _ in range(k):
            game = Game(self.agent, oppo)
            while game.result is None:
                game.agent_move()
            if game.result:
                wins += 1

            game = Game(oppo, self.agent)
            while game.result is None:
                game.agent_move()
            if not game.result:
                wins += 1
        return wins / k / 2 if k > 0 else 0.5

def get_buffer_from_fights(a: Any, b: Any, k: int = 1) -> Tuple[List[ReplayEntry], List[ReplayEntry]]:
    a_buffer = []
    b_buffer = []
    is_a_white = True
    counter = 0
    while counter < k:
        game = Game(a, b)
        entry_a = None
        entry_b = None
        while game.result is None:
            if counter >= k:
                break
            counter += 1
            is_a_playing = game.is_white_turn == is_a_white
            state = game.get_state().convert_to_torch()
            _, move, legal = game.agent_move()
            reward_a = get_reward(game, is_a_white, legal)
            reward_b = get_reward(game, not is_a_white, legal)
            done = game.result is not None
            if is_a_playing:
                entry_a = ReplayEntry(
                    state=state,
                    action=torch.tensor(move, dtype=torch.int64, device=state.selfq_board.device),
                    reward=reward_a,
                    next_state=empty_game_state_torch(),
                    done=done,
                )
                if entry_b:
                    entry_b.reward += reward_b
                    entry_b.next_state = game.get_state().convert_to_torch() if not done else empty_game_state_torch()
                    b_buffer.append(entry_b)
            else:
                entry_b = ReplayEntry(
                    state=state,
                    action=torch.tensor(move, dtype=torch.int64, device=state.selfq_board.device),
                    reward=reward_b,
                    next_state=empty_game_state_torch(),
                    done=done,
                )
                if entry_a:
                    entry_a.reward += reward_a
                    entry_a.next_state = game.get_state().convert_to_torch() if not done else empty_game_state_torch()
                    a_buffer.append(entry_a)
            if done:
                if is_a_playing:
                    a_buffer.append(entry_a)
                else:
                    b_buffer.append(entry_b)
    return a_buffer, b_buffer
    