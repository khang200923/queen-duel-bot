import torch
import torch.nn as nn
from src.ai.ai import ReplayEntry
from src.game.game import Game
from src.game.state import GameStateTorch

def empty_game_state_torch() -> GameStateTorch:
    return GameStateTorch(
        blockers_board=torch.zeros((8, 8), dtype=torch.float32),
        selfq_board=torch.zeros((8, 8), dtype=torch.float32),
        oppq_board=torch.zeros((8, 8), dtype=torch.float32),
        legal_moves_mask=torch.ones((8, 8), dtype=torch.float32)
    )

def get_reward(game: Game, side: bool, legal: bool) -> float:
    reward = 0
    if game.result is not None:
        reward = 1 if game.result == side else -1
    else:
        # state = game.get_state(side)
        # legal_moves_num = state.mask_legal_moves().sum()
        # oppo_state = game.get_state(not side)
        # oppo_legal_moves_num = oppo_state.mask_legal_moves().sum()
        # reward = (legal_moves_num / 64 - oppo_legal_moves_num / 64) / 4
        pass
    return reward * 10

def train_step(agent: nn.Module, target: nn.Module, optimizer: torch.optim.Optimizer, buffer: list[ReplayEntry], GAMMA: float = 0.99) -> float:
    optimizer.zero_grad()
    device = buffer[0].state.selfq_board.device

    states = torch.stack([entry.state.all_boards().to(device) for entry in buffer]).to(device)
    actions = torch.stack([entry.action for entry in buffer]).to(device)
    rewards = torch.as_tensor([entry.reward for entry in buffer], dtype=torch.float32, device=device)
    next_states = torch.stack([entry.next_state.all_boards().to(device) for entry in buffer]).to(device)
    dones = torch.as_tensor([entry.done for entry in buffer], dtype=torch.float32, device=device)

    q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        best_actions = agent(next_states).argmax(1, keepdim=True)
        next_q_values = target(next_states).gather(1, best_actions).squeeze(1)
    target_q_values = rewards + (1 - dones) * next_q_values * GAMMA

    assert q_values.shape == target_q_values.shape, f"q_values shape {q_values.shape} does not match target_q_values shape {target_q_values.shape}"
    loss = nn.functional.mse_loss(q_values, target_q_values)
    loss.backward()
    optimizer.step()

    return loss.item()

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )