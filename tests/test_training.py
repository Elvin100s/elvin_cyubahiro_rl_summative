import copy

import numpy as np
import torch

from environment.custom_env import ACTIONS, INVALID_ACTION, DroneDeliveryEnv
from training.common import FeasibleOnly, feasible_actions, make_mlp


def test_mask_matches_the_environment_rules():
    env = DroneDeliveryEnv()
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(0)
    for _ in range(600):
        mask = feasible_actions(torch.as_tensor(obs)).numpy()
        for action in range(len(ACTIONS)):
            trial = copy.deepcopy(env)
            assert mask[action] == (trial._apply_action(action) != INVALID_ACTION), (ACTIONS[action], obs)
        obs, _, terminated, truncated, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        if terminated or truncated:
            obs, _ = env.reset()


def test_masked_network_never_picks_an_infeasible_action():
    net = FeasibleOnly(make_mlp((16,), len(ACTIONS)))
    env = DroneDeliveryEnv()
    obs, _ = env.reset(seed=5)
    for _ in range(300):
        tensor = torch.as_tensor(obs)
        action = int(net(tensor).argmax())
        assert feasible_actions(tensor)[action]
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
