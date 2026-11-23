"""Shared by every algorithm: TorchRL env, network, run logging, evaluation and checkpoints."""
import argparse
import json
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, TransformedEnv
from torchrl.modules import MLP

from environment.custom_env import ACTIONS, CLINIC_NAMES, OBS_FIELDS, SCENARIOS, DroneDeliveryEnv
from environment.evaluation import VALIDATION_SEED, evaluate

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
MODEL_SUBDIR = {"dqn": "dqn", "reinforce": "pg", "ppo": "pg", "a2c": "pg"}
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}

warnings.filterwarnings("ignore")
logging.getLogger("torchrl").setLevel(logging.WARNING)


def quiet():
    warnings.filterwarnings("ignore")
    logging.getLogger("torchrl").setLevel(logging.WARNING)
    torch.set_num_threads(1)


def make_env(num_envs, seed, shaping=0.0, exploring_starts=0.0, reward_scale=1.0, scenario="standard"):
    """Batched TorchRL env; `episode_reward` logs the unscaled reward, the learner sees reward * reward_scale."""
    vec = gym.vector.SyncVectorEnv(
        [lambda: DroneDeliveryEnv(**SCENARIOS[scenario], shaping=shaping, exploring_starts=exploring_starts)
         for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env = TransformedEnv(GymWrapper(vec, categorical_action_encoding=True),
                         Compose(RewardSum(), RewardScaling(loc=0.0, scale=reward_scale)))
    env.set_seed(seed)
    return env


def make_mlp(hidden, out_features, activation="tanh"):
    return MLP(in_features=len(OBS_FIELDS), out_features=out_features, num_cells=list(hidden),
               activation_class=ACTIVATIONS[activation])


REQUEST_COLUMNS = [OBS_FIELDS.index(f"{name}_request") for name in CLINIC_NAMES]
DEPOT_DX, DEPOT_DY = OBS_FIELDS.index("depot_dx"), OBS_FIELDS.index("depot_dy")


def feasible_actions(obs, eps=1e-4):
    """Physically possible actions, read off the observation: north, south, west, east, load, deliver, recharge."""
    x, y, packages = obs[..., 0], obs[..., 1], obs[..., 3]
    at_depot = (obs[..., DEPOT_DX].abs() < eps) & (obs[..., DEPOT_DY].abs() < eps)
    at_open_clinic = torch.zeros_like(at_depot)
    for c in REQUEST_COLUMNS:
        at_open_clinic |= (obs[..., c] > 0.5) & (obs[..., c + 3].abs() < eps) & (obs[..., c + 4].abs() < eps)
    return torch.stack([y > eps, y < 1 - eps, x > eps, x < 1 - eps, at_depot & (packages < 1 - eps),
                        at_open_clinic & (packages > eps), at_depot], dim=-1)


class FeasibleOnly(nn.Module):
    """Wraps a policy or Q network so infeasible actions can never be chosen."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net(obs).masked_fill(~feasible_actions(obs), -1e9)


def make_policy_net(hidden, activation, mask):
    net = make_mlp(hidden, len(ACTIONS), activation)
    return FeasibleOnly(net) if mask else net


def greedy_policy(net):
    """Pick the highest-scoring action (Q-value for DQN, logit for policy-gradient methods)."""
    @torch.no_grad()
    def act(env, obs):
        return int(net(torch.as_tensor(obs)).argmax())
    return act


def scored_policy(net, algo):
    """Same choice as greedy_policy, but stores the action scores on env for the visual."""
    @torch.no_grad()
    def act(env, obs):
        scores = net(torch.as_tensor(obs, dtype=torch.float32))
        feasible = scores > -1e8
        shown = scores if algo == "dqn" else scores.softmax(-1)
        env.scores = [float(s) if ok else None for s, ok in zip(shown, feasible)]
        env.score_kind = "Q-value" if algo == "dqn" else "probability"
        return int(scores.argmax())
    return act


def finished_episode_rewards(data):
    done = data["next", "done"].squeeze(-1)
    return data["next", "episode_reward"].squeeze(-1)[done].tolist()


def gradient_step(optim, params, loss, max_norm):
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, max_norm)
    optim.step()


class RunLogger:
    def __init__(self, algo, run, config, net):
        self.algo, self.run, self.config, self.net = algo, run, config, net
        self.rows, self.start = [], time.time()
        self.next_eval = 0
        (LOG_DIR / algo).mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / MODEL_SUBDIR[algo]).mkdir(parents=True, exist_ok=True)

    def log(self, frames, episode_rewards, **metrics):
        row = {"frames": frames, "time": time.time() - self.start, "episodes": len(episode_rewards),
               "train_reward": float(np.mean(episode_rewards)) if len(episode_rewards) else np.nan}
        row.update({k: float(v) for k, v in metrics.items()})
        if frames >= self.next_eval:
            row["val_reward"] = evaluate(greedy_policy(self.net), episodes=5, seed=VALIDATION_SEED)["reward_mean"]
            self.next_eval += self.config.eval_every
        self.rows.append(row)

    def finish(self):
        curve = pd.DataFrame(self.rows)
        curve.to_csv(LOG_DIR / self.algo / f"{self.run}.csv", index=False)
        policy = greedy_policy(self.net)
        summary = {
            "algo": self.algo, "run": self.run, "config": asdict(self.config),
            "train_time_s": round(time.time() - self.start, 1),
            "final_train_reward": float(curve["train_reward"].tail(max(1, len(curve) // 10)).mean()),
            "best_val_reward": float(curve["val_reward"].max()),
            "val": evaluate(policy, episodes=20, seed=VALIDATION_SEED),
            "test": {s: evaluate(policy, episodes=30, scenario=s) for s in SCENARIOS},
        }
        tests = summary["test"]
        (LOG_DIR / self.algo / f"{self.run}.json").write_text(json.dumps(summary, indent=1))
        torch.save({"algo": self.algo, "run": self.run, "config": asdict(self.config),
                    "obs_fields": OBS_FIELDS, "actions": ACTIONS, "state_dict": self.net.state_dict()},
                   MODEL_DIR / MODEL_SUBDIR[self.algo] / f"{self.run}.pt")
        t = tests["standard"]
        print(f"[{self.run}] done in {summary['train_time_s']:.0f}s | test reward {t['reward_mean']:.1f} ± {t['reward_std']:.1f}"
              f" | delivered {t['delivered']:.1f} | urgent expired {t['urgent_expired']:.1f} | crash rate {t['crash_rate']:.2f}",
              flush=True)
        return summary


def load_policy(path):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    net = make_policy_net(cfg["hidden"], cfg["activation"], cfg.get("mask", False))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def sweep_configs(base, sweep, runs=None, seeds=(0,), frames=None):
    """One config per (sweep entry, seed); each entry changes one or two settings from `base`."""
    configs = []
    for i, (note, overrides) in enumerate(sweep, start=1):
        if runs and i not in runs:
            continue
        for seed in seeds:
            name = f"{base.algo}_{i:02d}" + (f"_s{seed}" if seed else "")
            cfg = replace(base, run=name, note=note, seed=seed, **overrides)
            configs.append(replace(cfg, frames=frames) if frames else cfg)
    return configs


def best_sweep_index(algo):
    """Sweep index of the seed-0 run with the highest mean validation reward."""
    results = [json.loads(p.read_text()) for p in (LOG_DIR / algo).glob(f"{algo}_[0-9][0-9].json")]
    if not results:
        raise SystemExit(f"No finished {algo} sweep runs yet")
    return int(max(results, key=lambda r: r["val"]["reward_mean"])["run"].split("_")[1])


def sweep_cli(algos, build_configs, train_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=[*algos, "all"], default="all")
    parser.add_argument("--runs", type=int, nargs="*", help="sweep indices to run (default: all)")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--frames", type=int, help="override the frame budget, e.g. for a smoke test")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="retrain runs that already have results")
    parser.add_argument("--best", action="store_true", help="only the sweep entry with the best validation reward")
    args = parser.parse_args()
    chosen = list(algos) if args.algo == "all" else [args.algo]
    configs = [c for a in chosen for c in build_configs(a, [best_sweep_index(a)] if args.best else args.runs,
                                                         args.seeds, args.frames)]
    done = [c for c in configs if (LOG_DIR / c.algo / f"{c.run}.json").exists()]
    if done and not args.force:
        print(f"Skipping {len(done)} finished run(s) (use --force to retrain)", flush=True)
        configs = [c for c in configs if c not in done]
    print(f"Training {len(configs)} run(s) with {args.workers} worker(s): {', '.join(c.run for c in configs)}", flush=True)
    run_sweep(train_fn, configs, args.workers)


def run_sweep(train_fn, configs, workers):
    if workers <= 1:
        return [train_fn(c) for c in configs]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=quiet) as pool:
        return list(pool.map(train_fn, configs))
