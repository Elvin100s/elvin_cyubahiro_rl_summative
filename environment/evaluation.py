"""Shared evaluation loop and hand-written reference policies (run: uv run python -m environment.evaluation)."""
import numpy as np

from environment.custom_env import (
    BATTERY_MAX, CAPACITY, CLINIC_POSITIONS, DELIVER, DEPOT, LOAD, RECHARGE, SCENARIOS, DroneDeliveryEnv,
)

TEST_SEED = 10_000
VALIDATION_SEED = 5_000


def evaluate(policy, episodes=30, seed=TEST_SEED, scenario="standard"):
    """Run `policy(env, obs) -> action` on fixed seeds and return outcome statistics."""
    env = DroneDeliveryEnv(**SCENARIOS[scenario])
    rewards, infos = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        env.action_space.seed(seed + ep)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(env, obs))
            total += reward
            done = terminated or truncated
        rewards.append(total)
        infos.append(info)
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "delivered": float(np.mean([i["delivered"] for i in infos])),
        "urgent_delivered": float(np.mean([i["urgent_delivered"] for i in infos])),
        "urgent_expired": float(np.mean([i["urgent_expired"] for i in infos])),
        "expired": float(np.mean([i["expired"] for i in infos])),
        "crash_rate": float(np.mean([i["crashed"] for i in infos])),
        "rewards": rewards,
    }


def _toward(pos, target):
    (x, y), (tx, ty) = pos, target
    if tx != x:
        return 2 if tx < x else 3
    if ty != y:
        return 0 if ty < y else 1
    return None


def greedy(env, obs=None):
    """Fill up at the depot, serve the most urgent then oldest request, head home when the battery runs low."""
    pos = tuple(env.drone)
    if pos == DEPOT and env.battery < BATTERY_MAX:
        return RECHARGE
    if pos == DEPOT and env.packages < CAPACITY:
        return LOAD
    if env.battery <= abs(pos[0] - DEPOT[0]) + abs(pos[1] - DEPOT[1]) + 2:
        return _toward(pos, DEPOT)
    here = env._clinic_here()
    if here is not None and env.requests[here] and env.packages:
        return DELIVER
    open_requests = [(i, r) for i, r in enumerate(env.requests) if r]
    if env.packages and open_requests:
        target, _ = max(open_requests, key=lambda ir: (ir[1]["urgent"], ir[1]["age"]))
        return _toward(pos, CLINIC_POSITIONS[target])
    return _toward(pos, DEPOT) if pos != DEPOT else RECHARGE


def random_policy(env, obs=None):
    return env.action_space.sample()


def idle(env, obs=None):
    return RECHARGE


def deliberate_crash(env, obs=None):
    return 2 if env.steps % 2 == 0 else 3


BASELINES = {"greedy": greedy, "idle": idle, "deliberate_crash": deliberate_crash, "random": random_policy}


if __name__ == "__main__":
    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        for name, policy in BASELINES.items():
            r = evaluate(policy, scenario=scenario)
            print(f"  {name:17s} reward {r['reward_mean']:7.1f} ± {r['reward_std']:5.1f} | delivered {r['delivered']:4.1f}"
                  f" | urgent expired {r['urgent_expired']:4.1f} | crash rate {r['crash_rate']:.2f}")
