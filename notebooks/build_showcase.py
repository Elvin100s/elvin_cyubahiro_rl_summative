"""Write and execute notebooks/showcase.ipynb (run: uv run python notebooks/build_showcase.py)."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

CELLS = [
    ("md", """# Rwanda Medical Drone Delivery: results showcase

A drone based at the Kigali depot serves five rural health centres. Urgent blood and medicine requests expire fast and the battery lasts about three round trips. DQN, REINFORCE, PPO and A2C were trained with TorchRL on the same environment and the same 400k-step budget; this notebook summarises what they learned.

Re-run everything with `uv run python notebooks/build_showcase.py`."""),
    ("code", """import io
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import Image, Markdown, display
from PIL import Image as PILImage

from environment.custom_env import ACTIONS, OBS_FIELDS, SCENARIOS, DroneDeliveryEnv
from environment.evaluation import BASELINES, TEST_SEED, evaluate
from training.common import load_policy, scored_policy

pd.set_option("display.precision", 1)
ASSETS, LOGS = ROOT / "assets", ROOT / "logs"
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}"""),
    ("md", "## The environment"),
    ("code", """env = DroneDeliveryEnv(render_mode="rgb_array")
env.reset(seed=TEST_SEED)
env.label = "Start of a shift"
display(PILImage.fromarray(env.render()))
env.close()
print(f"{len(ACTIONS)} actions: {', '.join(ACTIONS)}")
print(f"{len(OBS_FIELDS)} observations, all scaled to [-1, 1]")"""),
    ("md", "### Where every observation comes from"),
    ("code", """display(Markdown((ASSETS / "tables" / "observation_map.md").read_text()))"""),
    ("md", "## Reference policies\n\nHand-written policies on 30 fixed test shifts per scenario. They anchor what a good and a bad score look like."),
    ("code", """baselines = pd.DataFrame({
    name: {s: evaluate(policy, scenario=s)["reward_mean"] for s in SCENARIOS} for name, policy in BASELINES.items()
}).T
baselines"""),
    ("md", "## Hyperparameter sweeps\n\nEach algorithm ran 12–13 configurations, each changing one setting from its baseline. Models are chosen on validation shifts and scored on unseen test shifts."),
    ("code", """results = pd.read_csv(LOGS / "results.csv")
sweep = results[results["seed"] == 0]
summary = sweep.groupby("algo").agg(
    runs=("run", "count"), best_test=("test_reward", "max"), median_test=("test_reward", "median"),
    runs_that_deliver=("delivered", lambda d: int((d >= 1).sum())),
).rename(index=NAMES)
summary"""),
    ("code", """best = sweep.loc[sweep.groupby("algo")["val_reward"].idxmax()].set_index("algo").rename(index=NAMES)
best[["run", "note", "val_reward", "test_reward", "test_std", "delivered", "urgent_delivered", "urgent_expired",
      "crash_rate"]]"""),
    ("code", """for algo, name in NAMES.items():
    table = ASSETS / "tables" / f"{algo}_sweep.md"
    if table.exists():
        display(Markdown(f"### {name}\\n\\n" + table.read_text()))"""),
    ("md", "## Learning curves and diagnostics"),
    ("code", """for figure in ["comparative_rewards", "convergence", "dqn_loss", "pg_entropy", "sweep_sensitivity"]:
    display(Image(filename=str(ASSETS / f"{figure}.png"), width=900))"""),
    ("md", "## Generalisation to unseen demand"),
    ("code", """display(Image(filename=str(ASSETS / "generalisation.png"), width=900))
best[["test_reward", "high_demand_reward", "urgent_surge_reward"]]"""),
    ("md", "## Seed robustness\n\nThe best configuration of each algorithm, retrained with two more random seeds."),
    ("code", """robustness = ASSETS / "tables" / "seed_robustness.md"
display(Markdown(robustness.read_text() if robustness.exists() else "_Seed re-runs not finished yet._"))"""),
    ("md", "## Watch the best agent fly a test shift"),
    ("code", """top = best.loc[best["val_reward"].idxmax()]
algo = next(a for a, n in NAMES.items() if n == top.name)
net, _ = load_policy(ROOT / "models" / ("dqn" if algo == "dqn" else "pg") / f"{top['run']}.pt")
policy = scored_policy(net, algo)

env = DroneDeliveryEnv(render_mode="rgb_array")
env.label = f"{top.name} - {top['run']} ({top['note']})"
obs, _ = env.reset(seed=TEST_SEED)
frames, done = [], False
while not done:
    obs, _, terminated, truncated, info = env.step(policy(env, obs))
    done = terminated or truncated
    if env.steps % 3 == 0 and env.steps <= 150:
        frame = env.render()
        frames.append(PILImage.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2)))
env.close()

gif = io.BytesIO()
frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:], duration=110, loop=0)
display(Image(data=gif.getvalue(), format="gif"))
print(f"Shift reward {env.total_reward:+.1f} | delivered {info['delivered']} (urgent {info['urgent_delivered']}) | "
      f"expired {info['expired']} | {'crashed' if info['crashed'] else 'drone safe'}")"""),
    ("md", "## Serving the policy\n\nThe same checkpoints sit behind a FastAPI service used by the 3D view (`uv run main.py --view 3d`). Any client can send an observation and get an action back."),
    ("code", """from fastapi.testclient import TestClient

from environment.evaluation import greedy
from environment.server import Agent, create_app

client = TestClient(create_app([Agent(top.name, algo, top["run"], net, greedy)], verbose=False))
observation = DroneDeliveryEnv().reset(seed=1)[0].tolist()
print(client.get("/api/meta").json()["agents"])
client.post("/api/act", json={"observation": observation, "agent": top.name}).json()"""),
]


def build():
    nb = new_notebook()
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, HERE / "showcase.ipynb")
    print(f"Wrote {HERE / 'showcase.ipynb'}")


if __name__ == "__main__":
    build()
