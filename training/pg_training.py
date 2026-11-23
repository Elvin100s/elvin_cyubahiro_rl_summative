"""REINFORCE, PPO and A2C with TorchRL on the drone delivery environment.

uv run python -m training.pg_training --algo ppo --workers 3   # full sweep for one algorithm
uv run python -m training.pg_training --algo a2c --runs 1 4    # selected sweep entries
"""
from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from environment.custom_env import MAX_STEPS
from training.common import (
    RunLogger, finished_episode_rewards, gradient_step, make_env, make_mlp, make_policy_net, quiet, sweep_cli,
    sweep_configs,
)


@dataclass(frozen=True)
class PGConfig:
    algo: str
    run: str = ""
    note: str = "baseline"
    seed: int = 0
    frames: int = 400_000
    eval_every: int = 10_000
    shaping: float = 1.0
    mask: bool = True
    exploring_starts: float = 0.5
    reward_scale: float = 0.1
    num_envs: int = 16
    rollout: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.9
    hidden: tuple = (64, 64)
    activation: str = "tanh"
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_adv: bool = True
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    baseline: str = "value"


BASE = {
    "ppo": PGConfig("ppo"),
    "a2c": PGConfig("a2c", rollout=8, lr=7e-4, lmbda=1.0, normalize_adv=False),
    "reinforce": PGConfig("reinforce", num_envs=8, rollout=MAX_STEPS, lr=1e-3),
}

SWEEPS = {
    "ppo": [
        ("baseline", {}),
        ("lr 1e-3", {"lr": 1e-3}),
        ("lr 1e-4", {"lr": 1e-4}),
        ("clip 0.1", {"clip": 0.1}),
        ("clip 0.3", {"clip": 0.3}),
        ("entropy 0", {"entropy_coef": 0.0}),
        ("entropy 0.05", {"entropy_coef": 0.05}),
        ("rollout 32", {"rollout": 32}),
        ("rollout 512", {"rollout": 512}),
        ("epochs 10", {"epochs": 10}),
        ("GAE lambda 0.7", {"lmbda": 0.7}),
        ("GAE lambda 0.99", {"lmbda": 0.99}),
        ("weaker shaping k=0.3", {"shaping": 0.3}),
    ],
    "a2c": [
        ("baseline", {}),
        ("lr 2.5e-4", {"lr": 2.5e-4}),
        ("lr 2e-3", {"lr": 2e-3}),
        ("rollout 32", {"rollout": 32}),
        ("rollout 3", {"rollout": 3}),
        ("entropy 0", {"entropy_coef": 0.0}),
        ("entropy 0.05", {"entropy_coef": 0.05}),
        ("GAE lambda 0.9", {"lmbda": 0.9}),
        ("value coef 1.0", {"value_coef": 1.0}),
        ("normalised advantage", {"normalize_adv": True}),
        ("gamma 0.95", {"gamma": 0.95}),
        ("hidden 128x128", {"hidden": (128, 128)}),
    ],
    "reinforce": [
        ("baseline", {}),
        ("no baseline", {"baseline": "none"}),
        ("normalised returns", {"baseline": "normalized"}),
        ("lr 3e-4", {"lr": 3e-4}),
        ("lr 3e-3", {"lr": 3e-3}),
        ("4 episodes/update", {"num_envs": 4}),
        ("32 episodes/update", {"num_envs": 32}),
        ("entropy 0", {"entropy_coef": 0.0}),
        ("entropy 0.05", {"entropy_coef": 0.05}),
        ("gamma 0.95", {"gamma": 0.95}),
        ("gamma 0.999", {"gamma": 0.999}),
        ("hidden 128x128", {"hidden": (128, 128)}),
    ],
}


def build_configs(algo, runs=None, seeds=(0,), frames=None):
    return sweep_configs(BASE[algo], SWEEPS[algo], runs, seeds, frames)


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def reinforce_batch(data, critic, cfg):
    """Monte Carlo returns over complete episodes; steps of an unfinished episode are dropped."""
    done = data["next", "done"]
    returns = reward2go(data["next", "reward"], done, cfg.gamma, time_dim=-2)
    complete = done.int().flip(-2).cummax(-2).values.flip(-2).bool().squeeze(-1)
    batch = data[complete]
    returns = returns[complete]
    if cfg.baseline == "none":
        advantage = returns
    elif cfg.baseline == "normalized":
        advantage = standardize(returns)
    else:
        with torch.no_grad():
            advantage = standardize(returns - critic(batch.clone())["state_value"])
    batch.set("advantage", advantage)
    batch.set("value_target", returns)
    return batch


def train(cfg):
    quiet()
    torch.manual_seed(cfg.seed)
    env = make_env(cfg.num_envs, cfg.seed, cfg.shaping, cfg.exploring_starts, cfg.reward_scale)
    actor_net = make_policy_net(cfg.hidden, cfg.activation, cfg.mask)
    actor = ProbabilisticActor(TensorDictModule(actor_net, ["observation"], ["logits"]), in_keys=["logits"],
                               distribution_class=Categorical, spec=env.action_spec, return_log_prob=True)
    critic = ValueOperator(make_mlp(cfg.hidden, 1, cfg.activation), in_keys=["observation"])

    if cfg.algo == "ppo":
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg.clip, entropy_coeff=cfg.entropy_coef,
                              critic_coeff=cfg.value_coef, normalize_advantage=False)
    elif cfg.algo == "a2c":
        loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg.entropy_coef, critic_coeff=cfg.value_coef)
    else:
        loss_fn = ReinforceLoss(actor, critic)
    gae = GAE(gamma=cfg.gamma, lmbda=cfg.lmbda, value_network=critic)
    params = list(loss_fn.parameters())
    optim = torch.optim.Adam(params, lr=cfg.lr)

    collector = Collector(env, actor, frames_per_batch=cfg.num_envs * cfg.rollout, total_frames=cfg.frames,
                          reset_at_each_iter=cfg.algo == "reinforce")
    logger = RunLogger(cfg.algo, cfg.run, cfg, actor_net)
    frames = 0
    for data in collector:
        frames += data.numel()
        if cfg.algo == "reinforce":
            batch = reinforce_batch(data, critic, cfg)
            out = loss_fn(batch)
            entropy = actor.get_dist(batch).entropy().mean()
            value_loss = out["loss_value"] if cfg.baseline == "value" else 0.0
            gradient_step(optim, params, out["loss_actor"] - cfg.entropy_coef * entropy + cfg.value_coef * value_loss,
                          cfg.max_grad_norm)
            stats = {"policy_loss": out["loss_actor"].item(), "value_loss": out["loss_value"].item(),
                     "entropy": entropy.item()}
        else:
            with torch.no_grad():
                gae(data)
            batch = data.reshape(-1)
            if cfg.normalize_adv:
                batch.set("advantage", standardize(batch["advantage"]))
            stats = update_actor_critic(cfg, loss_fn, optim, params, batch, actor)
        logger.log(frames, finished_episode_rewards(data), **stats)
    collector.shutdown()
    return logger.finish()


def update_actor_critic(cfg, loss_fn, optim, params, batch, actor):
    epochs, minibatch = (cfg.epochs, cfg.minibatch) if cfg.algo == "ppo" else (1, len(batch))
    history = []
    for _ in range(epochs):
        for idx in torch.randperm(len(batch)).split(minibatch):
            out = loss_fn(batch[idx])
            loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", 0.0)
            gradient_step(optim, params, loss, cfg.max_grad_norm)
            history.append(out.detach())
    stats = {"policy_loss": "loss_objective", "value_loss": "loss_critic", "entropy": "entropy",
             "clip_fraction": "clip_fraction", "kl": "kl_approx"}
    result = {name: torch.stack([h[key] for h in history]).mean().item()
              for name, key in stats.items() if key in history[0].keys()}
    if "entropy" not in result:
        with torch.no_grad():
            result["entropy"] = actor.get_dist(batch).entropy().mean().item()
    return result


if __name__ == "__main__":
    sweep_cli(SWEEPS, build_configs, train)
