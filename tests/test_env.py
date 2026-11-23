import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.custom_env import (
    BATTERY_MAX, CLINICS, CRASH, DELIVER, DEPOT, MAX_STEPS, OBS_FIELDS, RECHARGE, ROUTINE_DEADLINE, ROUTINE_EXPIRED,
    SHAPING_GAMMA, URGENT_DELIVERY, URGENT_EXPIRED, URGENT_SPEED_BONUS, URGENT_WAITING, DroneDeliveryEnv,
)
from environment.evaluation import BASELINES, evaluate


def quiet_env(**request_at):
    """Env with no random arrivals and only the requests given, e.g. quiet_env(Kirehe=True)."""
    env = DroneDeliveryEnv(request_prob=0.0)
    env.reset(seed=0)
    names = list(CLINICS)
    env.requests = [None] * len(names)
    for name, urgent in request_at.items():
        env.requests[names.index(name)] = {"urgent": urgent, "age": 0}
    return env


def test_passes_gymnasium_checks():
    check_env(DroneDeliveryEnv(), skip_render_check=True)


def test_observations_stay_in_bounds_and_match_fields():
    env = DroneDeliveryEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (len(OBS_FIELDS),)
    for _ in range(2000):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset()


def test_start_state():
    env = DroneDeliveryEnv()
    obs, info = env.reset(seed=0)
    assert tuple(env.drone) == DEPOT
    assert info["battery"] == BATTERY_MAX and info["packages"] == 0
    assert 1 <= sum(r is not None for r in env.requests) <= 2
    assert obs[OBS_FIELDS.index("time_left")] == 1.0


def test_capacity_is_two_packages():
    env = quiet_env()
    rewards = [env.step(4)[1] for _ in range(3)]
    assert env.packages == 2 and rewards[2] < 0


def test_instant_urgent_delivery_earns_full_bonus():
    env = quiet_env(Kirehe=True)
    env.drone, env.packages = list(CLINICS["Kirehe"]), 1
    _, reward, *_ = env.step(DELIVER)
    assert reward == pytest.approx(URGENT_DELIVERY + URGENT_SPEED_BONUS)
    assert env.stats["urgent_delivered"] == 1 and env.packages == 0


def test_routine_request_expires_at_deadline():
    env = quiet_env(Musanze=False)
    env.requests[0]["age"] = ROUTINE_DEADLINE - 1
    _, reward, *_ = env.step(RECHARGE)
    assert reward == pytest.approx(ROUTINE_EXPIRED) and env.stats["expired"] == 1


def test_crash_abandons_open_requests():
    env = quiet_env(Rubavu=True)
    env.battery = 1.0
    _, reward, terminated, _, info = env.step(2)
    assert terminated and info["crashed"]
    assert reward == pytest.approx(URGENT_WAITING + CRASH + URGENT_EXPIRED)


def test_empty_battery_at_depot_is_not_a_crash():
    env = quiet_env()
    env.battery = 0.1
    _, _, terminated, _, _ = env.step(4)
    assert not terminated


def test_shift_ends_by_truncation():
    env = quiet_env()
    for step in range(MAX_STEPS):
        _, _, terminated, truncated, _ = env.step(RECHARGE)
        assert not terminated and truncated == (step == MAX_STEPS - 1)


def test_same_seed_same_episode():
    def rollout(seed):
        env = DroneDeliveryEnv()
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return np.stack([env.step(env.action_space.sample())[0] for _ in range(100)])
    assert np.array_equal(rollout(3), rollout(3))


def test_reward_design_ranks_behaviours_sensibly():
    score = {name: evaluate(policy, episodes=10)["reward_mean"] for name, policy in BASELINES.items()}
    assert score["greedy"] > 0 > score["idle"] > score["deliberate_crash"]
    assert score["random"] < score["idle"]


def test_shaping_is_potential_based_and_leaves_dynamics_unchanged():
    plain, shaped = DroneDeliveryEnv(), DroneDeliveryEnv(shaping=0.1)
    plain.reset(seed=4)
    shaped.reset(seed=4)
    rng = np.random.default_rng(0)
    for _ in range(MAX_STEPS):
        before = -0.1 * shaped.steps_to_next_delivery()
        action = int(rng.integers(7))
        obs_a, r_plain, term, trunc, _ = plain.step(action)
        obs_b, r_shaped, *_ = shaped.step(action)
        after = 0.0 if term else -0.1 * shaped.steps_to_next_delivery()
        assert np.array_equal(obs_a, obs_b)
        assert r_shaped - r_plain == pytest.approx(SHAPING_GAMMA * after - before)
        if term or trunc:
            break


def test_exploring_starts_are_training_only_and_always_survivable():
    for seed in range(200):
        env = DroneDeliveryEnv(exploring_starts=1.0)
        env.reset(seed=seed)
        home = abs(env.drone[0] - DEPOT[0]) + abs(env.drone[1] - DEPOT[1])
        assert env.battery >= min(BATTERY_MAX, home + 5)
    plain = DroneDeliveryEnv()
    for seed in range(20):
        plain.reset(seed=seed)
        assert tuple(plain.drone) == DEPOT and plain.battery == BATTERY_MAX and plain.packages == 0
