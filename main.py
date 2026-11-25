"""Fly the best trained agent through a shift.

uv run main.py                          # best model overall, 2D pygame view
uv run main.py --view 3d                # best model in the three.js world
uv run main.py --view 3d --race         # best DQN vs REINFORCE vs PPO vs A2C side by side
uv run main.py --choose                 # pick the model to fly from a table
uv run main.py --algo dqn --view none   # terminal only
"""

import argparse
import json
from pathlib import Path

from environment.custom_env import SCENARIOS, DroneDeliveryEnv
from environment.evaluation import TEST_SEED, greedy

ROOT = Path(__file__).resolve().parent
ALGO_NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}


def best_per_algorithm():
    best = {}
    for path in sorted((ROOT / "logs").glob("*/*_[0-9][0-9].json")):
        r = json.loads(path.read_text())
        if r["algo"] not in best or r["val"]["reward_mean"] > best[r["algo"]]["val"]["reward_mean"]:
            best[r["algo"]] = r
    return {a: best[a] for a in ALGO_NAMES if a in best}


def print_leaderboard(best):
    """Print the best run of each algorithm, best first, and return them in that order."""
    ranked = sorted(best.values(), key=lambda r: -r["val"]["reward_mean"])
    print(f"\n{'#':2s} {'Algorithm':10s} {'Best run':10s} {'Setting':34s} {'Val':>7s} {'Test (30 unseen shifts)':>24s} "
          f"{'Delivered':>9s} {'Urg.exp':>7s} {'Crash':>6s}")
    for i, r in enumerate(ranked, start=1):
        t = r["test"]["standard"]
        print(f"{i:<2d} {ALGO_NAMES[r['algo']]:10s} {r['run']:10s} {r['config']['note'][:34]:34s} {r['val']['reward_mean']:7.1f} "
              f"{t['reward_mean']:15.1f} ± {t['reward_std']:5.1f} {t['delivered']:9.1f} {t['urgent_expired']:7.1f} "
              f"{t['crash_rate']:6.2f}")
    return ranked


def ask_choice(ranked):
    while True:
        answer = input(f"\nChoose a model to fly [1-{len(ranked)}, Enter = 1, the best]: ").strip()
        if not answer:
            return ranked[0]
        if answer.isdigit() and 1 <= int(answer) <= len(ranked):
            return ranked[int(answer) - 1]
        print("Please type one of the numbers in the table.")


def checkpoint_path(run):
    return ROOT / "models" / ("dqn" if run["algo"] == "dqn" else "pg") / f"{run['run']}.pt"


def load_agents(args):
    """Return [(algo, label, net or None)]: one model for 2D, one per algorithm for the 3D race."""
    from training.common import load_policy

    if args.model:
        net, ckpt = load_policy(args.model)
        return [(ckpt["algo"], f"{ckpt['run']} ({ckpt['config']['note']})", net)]
    best = best_per_algorithm()
    if args.algo:
        best = {args.algo: best[args.algo]} if args.algo in best else {}
    if not best:
        print("No trained models found in logs/ - flying the hand-written greedy baseline instead.\n"
              "Train first: uv run python -m training.pg_training --algo ppo")
        return [("greedy", "hand-written baseline", None)]
    ranked = print_leaderboard(best)
    chosen = ranked[0]
    print(f"\nBest model: {ALGO_NAMES[chosen['algo']]} {chosen['run']} ({chosen['config']['note']}). Chosen because it has "
          f"the highest mean reward on 20 validation shifts ({chosen['val']['reward_mean']:+.1f}); on 30 unseen test "
          f"shifts it scored {chosen['test']['standard']['reward_mean']:+.1f}.")
    if args.choose and not (args.view == "3d" and args.race):
        chosen = ask_choice(ranked)
        print(f"Flying {ALGO_NAMES[chosen['algo']]} {chosen['run']} ({chosen['config']['note']}).")
    runs = list(best.values()) if args.view == "3d" and args.race else [chosen]
    agents = []
    for run in runs:
        path = checkpoint_path(run)
        print(f"Loading {path.relative_to(ROOT)}")
        agents.append((run["algo"], f"{run['run']} ({run['config']['note']})", load_policy(path)[0]))
    return agents


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=["2d", "3d", "none"], default="2d")
    parser.add_argument("--algo", choices=list(ALGO_NAMES), help="restrict to one algorithm")
    parser.add_argument("--model", help="path to a specific .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=1, help="shifts to fly (2d / none views)")
    parser.add_argument("--seed", type=int, default=TEST_SEED)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="standard")
    parser.add_argument("--fps", type=int, default=5, help="simulation steps per second")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hq", action="store_true", help="3D view at full quality (shadows, denser forest)")
    parser.add_argument("--race", action="store_true", help="3D: race the best model of every algorithm")
    parser.add_argument("--choose", action="store_true", help="pick which trained model to fly from the table")
    args = parser.parse_args()

    agents = load_agents(args)
    if args.view == "3d":
        run_3d(args, agents)
        return

    algo, label, net = agents[0]
    if net is None:
        policy = greedy
    else:
        from training.common import scored_policy
        policy = scored_policy(net, algo)
    env = DroneDeliveryEnv(render_mode="human" if args.view == "2d" else None, **SCENARIOS[args.scenario])
    env.metadata["render_fps"] = args.fps
    env.label = f"{ALGO_NAMES.get(algo, 'Greedy')} - {label}"
    try:
        for episode in range(args.episodes):
            fly_shift(env, policy, args.seed + episode, episode + 1)
    except (KeyboardInterrupt, SystemExit):
        print("\nStopped.")
    finally:
        env.close()


def run_3d(args, agents):
    import threading
    import webbrowser

    import uvicorn

    from environment.server import Agent, create_app

    race = [Agent(ALGO_NAMES.get(algo, "Greedy"), algo, label, net, greedy) for algo, label, net in agents]
    app = create_app(race, args.scenario, args.seed, fps=args.fps)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n3D view running at {url} (opening the browser; Ctrl+C to stop). API docs: {url}/docs")
    threading.Timer(1.5, webbrowser.open, [url + ("/?hq" if args.hq else "")]).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def fly_shift(env, policy, seed, number):
    obs, _ = env.reset(seed=seed)
    print(f"\n=== Shift {number} (seed {seed}) | {env.label} ===")
    if env.render_mode:
        env.render()
        print("Press SPACE (or click) in the window to start; SPACE pauses and resumes.", flush=True)
        env.renderer.hold(env, "Press SPACE or click to start the shift")
    done, last_event = False, env.last_event
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(env, obs))
        done = terminated or truncated
        event = env.last_event if env.last_event != last_event else ""
        last_event = env.last_event
        print(f"step {env.steps:3d} | {env.last_action:8s} | battery {env.battery:4.1f} | packages {env.packages} | "
              f"delivered {info['delivered']:2d} | reward {reward:+6.1f} | total {env.total_reward:+7.1f} | {event}")
    s = env.stats
    print(f"--- Shift over: reward {env.total_reward:+.1f} | delivered {s['delivered']} (urgent {s['urgent_delivered']}) | "
          f"expired {s['expired']} (urgent {s['urgent_expired']}) | {'CRASHED' if s['crashed'] else 'drone safe'}")
    if env.render_mode:
        env.renderer.hold(env, f"Shift over: {env.total_reward:+.1f}, {s['delivered']} deliveries  ·  SPACE to close")


if __name__ == "__main__":
    main()
