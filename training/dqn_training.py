"""DQN with TorchRL on the drone delivery environment.

uv run python -m training.dqn_training --workers 3     # full sweep
uv run python -m training.dqn_training --runs 1 10     # selected sweep entries
"""
from dataclasses import dataclass

import numpy as np
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import (
    RunLogger, feasible_actions, finished_episode_rewards, gradient_step, make_env, make_policy_net, quiet, sweep_cli,
    sweep_configs,
)


@dataclass(frozen=True)
class DQNConfig:
    algo: str = "dqn"
    run: str = ""
    note: str = "baseline"
    seed: int = 0
    frames: int = 400_000
    eval_every: int = 10_000
    shaping: float = 1.0
    mask: bool = True
    exploring_starts: float = 0.5
    reward_scale: float = 1.0
    log_every: int = 2_048
    num_envs: int = 8
    frames_per_batch: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    hidden: tuple = (128, 128)
    activation: str = "relu"
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    replay_ratio: float = 0.25
    eps_end: float = 0.05
    eps_decay_frames: int = 100_000
    target_update: str = "hard"
    target_interval: int = 1_000
    tau: float = 0.005
    double_dqn: bool = False
    max_grad_norm: float = 10.0


SWEEP = [
    ("baseline", {}),
    ("lr 1e-4", {"lr": 1e-4}),
    ("lr 2e-3", {"lr": 2e-3}),
    ("gamma 0.95", {"gamma": 0.95}),
    ("fast epsilon decay (30k, end 0.01)", {"eps_decay_frames": 30_000, "eps_end": 0.01}),
    ("slow epsilon decay (250k, end 0.10)", {"eps_decay_frames": 250_000, "eps_end": 0.10}),
    ("small replay buffer 10k", {"buffer_size": 10_000}),
    ("replay ratio 0.5", {"replay_ratio": 0.5}),
    ("hard target every 250 steps", {"target_interval": 250}),
    ("soft target tau 0.005", {"target_update": "soft"}),
    ("double DQN", {"double_dqn": True}),
    ("no action mask (ablation)", {"mask": False}),
    ("weaker shaping k=0.3", {"shaping": 0.3}),
]


def build_configs(algo, runs=None, seeds=(0,), frames=None):
    return sweep_configs(DQNConfig(), SWEEP, runs, seeds, frames)


def train(cfg):
    quiet()
    torch.manual_seed(cfg.seed)
    env = make_env(cfg.num_envs, cfg.seed, cfg.shaping, cfg.exploring_starts, cfg.reward_scale)
    q_net = make_policy_net(cfg.hidden, cfg.activation, cfg.mask)
    actor = QValueActor(q_net, in_keys=["observation"], spec=env.action_spec_unbatched)
    explore = EGreedyModule(env.action_spec_unbatched, eps_init=1.0, eps_end=cfg.eps_end,
                            annealing_num_steps=cfg.eps_decay_frames, action_mask_key="action_mask" if cfg.mask else None)
    mask = TensorDictModule(feasible_actions, in_keys=["observation"], out_keys=["action_mask"])

    loss_fn = DQNLoss(actor, loss_function="smooth_l1", delay_value=True, double_dqn=cfg.double_dqn,
                      action_space=env.action_spec_unbatched)
    loss_fn.make_value_estimator(gamma=cfg.gamma)
    if cfg.target_update == "hard":
        target = HardUpdate(loss_fn, value_network_update_interval=cfg.target_interval)
    else:
        target = SoftUpdate(loss_fn, tau=cfg.tau)
    params = list(loss_fn.parameters())
    optim = torch.optim.Adam(params, lr=cfg.lr)
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg.buffer_size), batch_size=cfg.batch_size)

    policy = TensorDictSequential(mask, actor, explore) if cfg.mask else TensorDictSequential(actor, explore)
    collector = Collector(env, policy, frames_per_batch=cfg.frames_per_batch,
                          total_frames=cfg.frames)
    logger = RunLogger("dqn", cfg.run, cfg, q_net)
    updates_per_batch = max(1, round(cfg.frames_per_batch * cfg.replay_ratio))
    frames, episode_rewards, losses, q_values = 0, [], [], []
    for data in collector:
        frames += data.numel()
        buffer.extend(data.reshape(-1).exclude("collector"))
        explore.step(data.numel())
        episode_rewards += finished_episode_rewards(data)

        if frames >= cfg.learning_starts:
            for _ in range(updates_per_batch):
                batch = buffer.sample()
                loss = loss_fn(batch)["loss"]
                gradient_step(optim, params, loss, cfg.max_grad_norm)
                target.step()
                losses.append(loss.item())
            with torch.no_grad():
                q_values.append(q_net(batch["observation"]).max(-1).values.mean().item())

        if frames % cfg.log_every < cfg.frames_per_batch:
            logger.log(frames, episode_rewards, epsilon=explore.eps.item(),
                       loss=np.mean(losses) if losses else np.nan, q_mean=np.mean(q_values) if q_values else np.nan)
            episode_rewards, losses, q_values = [], [], []
    collector.shutdown()
    return logger.finish()


if __name__ == "__main__":
    sweep_cli(["dqn"], build_configs, train)
