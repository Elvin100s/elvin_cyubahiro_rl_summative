"""Build the report figures and sweep tables from logs/ (run: uv run python -m training.analysis)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment.custom_env import SCENARIOS
from environment.evaluation import BASELINES, evaluate

ROOT = Path(__file__).resolve().parents[1]
LOGS, ASSETS = ROOT / "logs", ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
INK, MUTED, FAINT, GRID = "#0b0b0b", "#52514e", "#c9c8c3", "#ebeae6"
SCENARIO_NAMES = {"standard": "Standard", "quiet": "Quiet", "high_demand": "High demand", "urgent_surge": "Urgent surge"}

HYPERPARAMS = {
    "dqn": {"lr": "lr", "gamma": "γ", "buffer_size": "buffer", "replay_ratio": "replay ratio", "eps_end": "ε end",
            "eps_decay_frames": "ε decay frames", "target": "target update", "double_dqn": "double", "mask": "mask"},
    "ppo": {"lr": "lr", "gamma": "γ", "lmbda": "GAE λ", "rollout": "rollout", "epochs": "epochs", "clip": "clip ε",
            "entropy_coef": "entropy", "value_coef": "value coef"},
    "a2c": {"lr": "lr", "gamma": "γ", "lmbda": "GAE λ", "rollout": "rollout", "entropy_coef": "entropy",
            "value_coef": "value coef", "normalize_adv": "adv. norm", "hidden": "hidden"},
    "reinforce": {"lr": "lr", "gamma": "γ", "num_envs": "episodes/update", "baseline": "advantage",
                  "entropy_coef": "entropy", "hidden": "hidden"},
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelcolor": MUTED,
    "axes.edgecolor": FAINT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 1.8, "figure.facecolor": "white",
})


def load_runs():
    runs = []
    for algo in ALGOS:
        for path in sorted((LOGS / algo).glob(f"{algo}_*.json")):
            run = json.loads(path.read_text())
            run["curve"] = pd.read_csv(path.with_suffix(".csv"))
            runs.append(run)
    return runs


def frames_to_90(curve):
    """Frames until the smoothed validation reward first covers 90% of its total improvement."""
    val = curve.dropna(subset=["val_reward"])
    smooth = val["val_reward"].rolling(3, min_periods=1).mean()
    target = smooth.iloc[0] + 0.9 * (smooth.max() - smooth.iloc[0])
    return int(val["frames"][smooth >= target].iloc[0])


def results_table(runs):
    rows = []
    for r in runs:
        cfg, test = r["config"], r["test"]
        target = "soft τ={tau}" if cfg.get("target_update") == "soft" else "hard/{target_interval}"
        rows.append({
            "algo": r["algo"], "run": r["run"], "seed": cfg["seed"], "note": cfg["note"], **cfg,
            "target": target.format(**cfg) if r["algo"] == "dqn" else "",
            "val_reward": r["val"]["reward_mean"],
            "test_reward": test["standard"]["reward_mean"], "test_std": test["standard"]["reward_std"],
            "delivered": test["standard"]["delivered"], "urgent_delivered": test["standard"]["urgent_delivered"],
            "urgent_expired": test["standard"]["urgent_expired"], "crash_rate": test["standard"]["crash_rate"],
            "high_demand_reward": test["high_demand"]["reward_mean"],
            "urgent_surge_reward": test["urgent_surge"]["reward_mean"],
            "frames_to_90pct": frames_to_90(r["curve"]), "train_minutes": r["train_time_s"] / 60,
        })
    return pd.DataFrame(rows)


def best_runs(runs, table):
    sweep = table[table["seed"] == 0]
    best = sweep.loc[sweep.groupby("algo")["val_reward"].idxmax()]
    by_name = {r["run"]: r for r in runs}
    return {row.algo: by_name[row.run] for row in best.itertuples()}


def write_tables(table):
    out = ASSETS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["config"], errors="ignore").to_csv(LOGS / "results.csv", index=False)
    for algo in ALGOS:
        t = table[(table["algo"] == algo) & (table["seed"] == 0)]
        if t.empty:
            continue
        cols = {"run": "Run", "note": "Change from baseline", **HYPERPARAMS[algo]}
        view = t[list(cols)].rename(columns=cols)
        view["Val reward"] = t["val_reward"].round(1)
        view["Test reward"] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(t["test_reward"], t["test_std"])]
        view["Delivered"] = t["delivered"].round(1)
        view["Urgent expired"] = t["urgent_expired"].round(1)
        view["Crash rate"] = t["crash_rate"].round(2)
        view["Frames to 90%"] = (t["frames_to_90pct"] / 1000).round().astype(int).astype(str) + "k"
        view.to_markdown(out / f"{algo}_sweep.md", index=False)
        view.to_csv(out / f"{algo}_sweep.csv", index=False)
    seeds = table.groupby(["algo", "note"]).filter(lambda g: len(g) > 1)
    if not seeds.empty:
        summary = seeds.groupby(["algo", "note"]).agg(
            seeds=("seed", "count"), test_mean=("test_reward", "mean"), test_sd_across_seeds=("test_reward", "std"),
            delivered=("delivered", "mean"), crash_rate=("crash_rate", "mean")).round(2).reset_index()
        summary.to_markdown(out / "seed_robustness.md", index=False)


def label_end(ax, x, y, text, color=INK):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=color)


def reference_lines(ax, baselines):
    for name, style in (("greedy", "--"), ("idle", ":")):
        y = baselines[name]["standard"]["reward_mean"]
        ax.axhline(y, color=MUTED, lw=1, ls=style, zorder=1)
        ax.annotate(f"{name} baseline ({y:+.0f})", (0.01, y), xycoords=ax.get_yaxis_transform(), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.5, color=MUTED)


def smoothed(curve, metric, window_frames=10_000):
    """Rolling mean over a fixed number of environment steps, so curves logged at different rates compare fairly."""
    c = curve.dropna(subset=[metric])
    step = c["frames"].diff().median() if len(c) > 1 else window_frames
    return c["frames"] / 1000, c[metric].rolling(max(1, int(round(window_frames / step))), min_periods=1).mean()


def val_curve(run):
    return smoothed(run["curve"], "val_reward", window_frames=30_000)


def plot_comparative(best, baselines):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for algo, run in best.items():
        x, y = val_curve(run)
        left.plot(x, y, color=COLORS[algo], label=f"{NAMES[algo]} ({run['run']}, {run['config']['note']})")
        label_end(left, x.iloc[-1], y.iloc[-1], NAMES[algo])
        x, y = smoothed(run["curve"], "train_reward", window_frames=20_000)
        right.plot(x, y, color=COLORS[algo], label=NAMES[algo])
        label_end(right, x.iloc[-1], y.iloc[-1], NAMES[algo])
    reference_lines(left, baselines)
    left.set(title="Mission reward on validation shifts (best run per algorithm)",
             xlabel="Environment steps (thousands)", ylabel="Mean episode reward (unshaped)")
    right.set(title="Training episode reward (shaped, incl. exploring starts)", xlabel="Environment steps (thousands)",
              ylabel="Mean episode reward")
    left.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "comparative_rewards.png")
    plt.close(fig)


def highlight_sweep(ax, runs, metric, highlight, color, window_frames=10_000):
    others = False
    for run in runs:
        x, y = smoothed(run["curve"], metric, window_frames)
        if run["run"] == highlight:
            ax.plot(x, y, color=color, lw=2, zorder=3, label=f"best: {run['run']} ({run['config']['note']})")
        else:
            ax.plot(x, y, color=FAINT, lw=0.9, zorder=2, label=None if others else "other sweep runs")
            others = True


def plot_dqn(runs, best):
    dqn = [r for r in runs if r["algo"] == "dqn" and r["config"]["seed"] == 0]
    if not dqn:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    highlight_sweep(left, dqn, "loss", best["dqn"]["run"], COLORS["dqn"])
    left.set(title="DQN temporal-difference loss (Huber)", xlabel="Environment steps (thousands)",
             ylabel="Loss (10k-step mean)")
    highlight_sweep(right, dqn, "q_mean", best["dqn"]["run"], COLORS["dqn"])
    double = next((r for r in dqn if r["config"]["double_dqn"]), None)
    if double and double["run"] != best["dqn"]["run"]:
        x, y = smoothed(double["curve"], "q_mean")
        right.plot(x, y, color=MUTED, lw=1.6, ls="--", zorder=3, label=f"{double['run']} (double DQN)")
    right.set(title="Mean max Q-value on replay batches", xlabel="Environment steps (thousands)", ylabel="Q-value")
    left.legend(loc="upper right", fontsize=7.5)
    right.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "dqn_loss.png")
    plt.close(fig)


def plot_entropy(runs, best):
    algos = [a for a in ("reinforce", "ppo", "a2c") if a in best]
    if not algos:
        return
    fig, axes = plt.subplots(1, len(algos), figsize=(11, 3.4), sharey=True, squeeze=False)
    for ax, algo in zip(axes[0], algos):
        group = [r for r in runs if r["algo"] == algo and r["config"]["seed"] == 0]
        highlight_sweep(ax, group, "entropy", best[algo]["run"], COLORS[algo], window_frames=20_000)
        ax.axhline(np.log(7), color=MUTED, lw=1, ls=":")
        ax.annotate("uniform over 7 actions", (0.01, np.log(7)), xycoords=ax.get_yaxis_transform(), xytext=(0, -10),
                    textcoords="offset points", fontsize=7.5, color=MUTED)
        ax.set(title=f"{NAMES[algo]} policy entropy", xlabel="Environment steps (thousands)", ylim=(0, 2.1))
        ax.legend(loc="lower left", fontsize=7)
    axes[0][0].set_ylabel("Entropy (nats, 20k-step mean)")
    fig.tight_layout()
    fig.savefig(ASSETS / "pg_entropy.png")
    plt.close(fig)


def plot_convergence(runs, best, baselines):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        group = [r for r in runs if r["algo"] == algo and r["config"]["seed"] == 0]
        for run in group:
            x, y = val_curve(run)
            is_best = run["run"] == best.get(algo, {}).get("run")
            ax.plot(x, y, color=COLORS[algo] if is_best else FAINT, lw=2 if is_best else 0.9, zorder=3 if is_best else 2)
        reference_lines(ax, baselines)
        title = f"{NAMES[algo]}: {len(group)} runs"
        if algo in best:
            title += f", best {best[algo]['run']} ({best[algo]['config']['note']})"
        ax.set_title(title, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Environment steps (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation reward")
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png")
    plt.close(fig)


def plot_generalisation(best, baselines):
    entries = [(NAMES[a], COLORS[a], best[a]["test"]) for a in ALGOS if a in best]
    entries.append(("Greedy baseline", "#8a8986", baselines["greedy"]))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.8 / len(entries)
    for i, (name, color, tests) in enumerate(entries):
        x = np.arange(len(SCENARIOS)) + (i - (len(entries) - 1) / 2) * width
        means = [tests[s]["reward_mean"] for s in SCENARIOS]
        stds = [tests[s]["reward_std"] for s in SCENARIOS]
        ax.bar(x, means, width - 0.02, color=color, label=name, edgecolor="white", linewidth=1)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(SCENARIOS)), [SCENARIO_NAMES[s] for s in SCENARIOS])
    ax.set(title="Generalisation: best model per algorithm on 30 unseen test shifts per scenario",
           ylabel="Mean episode reward (± 1 sd)")
    ax.legend(ncol=len(entries), loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "generalisation.png")
    plt.close(fig)


def plot_sensitivity(table, baselines):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rng = np.random.default_rng(0)
    for i, algo in enumerate(ALGOS):
        t = table[(table["algo"] == algo) & (table["seed"] == 0)]
        if t.empty:
            continue
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(t)), t["test_reward"], s=36, color=COLORS[algo],
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.hlines(t["test_reward"].median(), i - 0.25, i + 0.25, color=INK, lw=1.2, zorder=4)
    reference_lines(ax, baselines)
    ax.set_xticks(range(len(ALGOS)), [NAMES[a] for a in ALGOS])
    ax.set(title="Hyperparameter sensitivity: test reward of every sweep run (bar = median)",
           ylabel="Mean test reward")
    fig.tight_layout()
    fig.savefig(ASSETS / "sweep_sensitivity.png")
    plt.close(fig)


def save_snapshots(best_run):
    import pygame
    from PIL import Image

    from environment.custom_env import DroneDeliveryEnv
    from environment.evaluation import TEST_SEED
    from training.common import load_policy, scored_policy

    net, ckpt = load_policy(ROOT / "models" / ("dqn" if best_run["algo"] == "dqn" else "pg") / f"{best_run['run']}.pt")
    policy = scored_policy(net, best_run["algo"])
    env = DroneDeliveryEnv(render_mode="rgb_array")
    env.label = f"{NAMES[best_run['algo']]} - {best_run['run']} ({best_run['config']['note']})"
    obs, _ = env.reset(seed=TEST_SEED)
    frames = []
    for step in range(150):
        obs, _, terminated, truncated, _ = env.step(policy(env, obs))
        frame = env.render()
        if step % 2 == 0:
            frames.append(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2)))
        if step == 90:
            Image.fromarray(frame).save(ASSETS / "environment.png")
        if terminated or truncated:
            break
    frames[0].save(ASSETS / "best_agent.gif", save_all=True, append_images=frames[1:], duration=240, loop=0)
    env.close()
    pygame.quit()


def main():
    ASSETS.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("No runs in logs/ yet. Train first, e.g. uv run python -m training.pg_training --algo ppo")
    baselines = {name: {s: evaluate(p, scenario=s) for s in SCENARIOS} for name, p in BASELINES.items()}
    table = results_table(runs)
    best = best_runs(runs, table)
    write_tables(table)
    pd.DataFrame([{"Policy": name, **{SCENARIO_NAMES[s]: f"{r[s]['reward_mean']:.1f} ± {r[s]['reward_std']:.1f}" for s in SCENARIOS},
                   "Delivered (standard)": round(r["standard"]["delivered"], 1),
                   "Crash rate (standard)": round(r["standard"]["crash_rate"], 2)}
                  for name, r in baselines.items()]).to_markdown(ASSETS / "tables" / "baselines.md", index=False)
    plot_comparative(best, baselines)
    plot_dqn(runs, best)
    plot_entropy(runs, best)
    plot_convergence(runs, best, baselines)
    plot_generalisation(best, baselines)
    plot_sensitivity(table, baselines)
    overall = max(best.values(), key=lambda r: r["val"]["reward_mean"])
    save_snapshots(overall)

    print("Best run per algorithm (selected on validation shifts, reported on unseen test shifts):")
    cols = ["algo", "run", "note", "val_reward", "test_reward", "test_std", "delivered", "urgent_expired",
            "crash_rate", "high_demand_reward", "urgent_surge_reward", "frames_to_90pct"]
    print(table.set_index("run").loc[[r["run"] for r in best.values()], cols[:1] + cols[2:]].round(2).to_string())
    print(f"\nGreedy baseline test reward: {baselines['greedy']['standard']['reward_mean']:.1f}")
    print(f"Figures -> {ASSETS.relative_to(ROOT)}/, tables -> {(ASSETS / 'tables').relative_to(ROOT)}/ and logs/results.csv")


if __name__ == "__main__":
    main()
