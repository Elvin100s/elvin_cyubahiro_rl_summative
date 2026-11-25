"""FastAPI bridge for the 3D view: Python steps the environment and agents, the browser only draws.

GET  /               three.js page
GET  /api/meta       map layout, actions, observation fields and loaded agents
POST /api/act        {"observation": [32 floats], "agent": "PPO"} -> chosen action and its scores
WS   /ws             one JSON frame per step; accepts {"type": "emergency", "clinic": i} | "start" | "pause" | "next" | "reset" | "speed"
"""
import asyncio
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from environment.custom_env import (
    ACTIONS, BATTERY_MAX, CAPACITY, CLINIC_NAMES, CLINIC_POSITIONS, DELIVER, DEPOT, GRID_SIZE, LOAD, MAX_STEPS,
    OBS_FIELDS, RECHARGE, ROUTINE_DEADLINE, SCENARIOS, URGENT_DEADLINE, DroneDeliveryEnv,
)

WEB = Path(__file__).parent / "web"
COLORS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100", "greedy": "#8a8986"}


@dataclass
class Agent:
    name: str
    algo: str
    label: str
    net: torch.nn.Module | None = None
    fallback: object = None

    @torch.no_grad()
    def act(self, env, obs):
        if self.net is None:
            return int(self.fallback(env, obs)), None, None
        scores = self.net(torch.as_tensor(obs, dtype=torch.float32))
        feasible = scores > -1e8
        kind = "Q-value" if self.algo == "dqn" else "probability"
        if kind == "probability":
            scores = scores.softmax(-1)
        return int(scores.argmax()), [round(float(s), 3) if ok else None for s, ok in zip(scores, feasible)], kind


class Race:
    """Every agent flies its own copy of the world with the same seed, so each faces the same demand."""

    def __init__(self, agents, scenario, seed, verbose):
        self.agents, self.scenario, self.seed, self.verbose = agents, scenario, seed, verbose
        self.shift = 0
        self.reset()

    def reset(self):
        self.envs = [DroneDeliveryEnv(**SCENARIOS[self.scenario]) for _ in self.agents]
        self.obs = [env.reset(seed=self.seed + self.shift)[0] for env in self.envs]
        self.done = [False] * len(self.agents)
        self.thinking = [(None, None)] * len(self.agents)
        if self.verbose:
            print(f"\n=== Shift {self.shift + 1} (seed {self.seed + self.shift}, {self.scenario}) | "
                  f"{', '.join(a.label for a in self.agents)} ===", flush=True)

    def next_shift(self):
        self.shift += 1
        self.reset()

    @property
    def finished(self):
        return all(self.done)

    def step(self):
        parts = []
        for i, (agent, env) in enumerate(zip(self.agents, self.envs)):
            if self.done[i]:
                continue
            event_before, delivered_before = env.last_event, env.stats["delivered"]
            action, scores, kind = agent.act(env, self.obs[i])
            self.obs[i], reward, terminated, truncated, _ = env.step(action)
            self.done[i] = terminated or truncated
            self.thinking[i] = (scores, kind)
            parts.append(f"{agent.name}: {ACTIONS[action]:8s} bat {env.battery:4.1f} del {env.stats['delivered']:2d} "
                         f"total {env.total_reward:+7.1f}")
            if self.verbose and (env.last_event != event_before or env.stats["delivered"] != delivered_before):
                parts[-1] += f"  <- {env.last_event} ({reward:+.1f})"
        if self.verbose and parts:
            print(f"step {max(e.steps for e in self.envs):3d} | " + " | ".join(parts), flush=True)
        if self.verbose and self.finished:
            ranking = sorted(zip(self.agents, self.envs), key=lambda ae: -ae[1].total_reward)
            print("--- Shift over: " + " > ".join(f"{a.name} {e.total_reward:+.1f} ({e.stats['delivered']} delivered"
                                                  f"{', CRASHED' if e.stats['crashed'] else ''})" for a, e in ranking),
                  flush=True)

    def emergency(self, clinic):
        for env, done in zip(self.envs, self.done):
            if not done:
                env.add_emergency(clinic)
        if self.verbose:
            print(f"!!! Emergency raised at {CLINIC_NAMES[clinic]} for every drone", flush=True)

    @staticmethod
    def plan(agent, env, horizon=40):
        """Roll the agent's own policy forward on a copy of the world (no new requests) to its next stop."""
        sim = copy.deepcopy(env)
        sim.request_prob = 0.0
        obs, path, moved = sim._get_obs(), [list(sim.drone)], False
        for _ in range(horizon):
            action = agent.act(sim, obs)[0]
            if action == DELIVER or (moved and action in (LOAD, RECHARGE)):
                break
            obs, _, terminated, truncated, _ = sim.step(action)
            if list(sim.drone) != path[-1]:
                path.append(list(sim.drone))
                moved = True
            if terminated or truncated:
                break
        goal = tuple(sim.drone)
        clinic = CLINIC_NAMES[CLINIC_POSITIONS.index(goal)] if goal in CLINIC_POSITIONS else None
        request = env.requests[CLINIC_NAMES.index(clinic)] if clinic else None
        kind = "home" if goal == DEPOT else "deliver" if clinic else "fly"
        return {"path": path, "kind": kind, "clinic": clinic, "urgent": bool(request and request["urgent"])}

    def frame(self, paused, fps):
        drones = []
        for agent, env, done, (scores, kind) in zip(self.agents, self.envs, self.done, self.thinking):
            drones.append({
                "name": agent.name, "label": agent.label, "color": COLORS[agent.algo],
                "x": env.drone[0], "y": env.drone[1], "battery": env.battery, "packages": env.packages,
                "reward": round(env.total_reward, 2), "last_reward": round(env.last_reward, 2),
                "action": env.last_action, "event": env.last_event, "done": done, "stats": env.stats,
                "scores": scores, "score_kind": kind,
                "plan": None if done else self.plan(agent, env),
                "requests": [None if r is None else {"urgent": r["urgent"], "age": r["age"],
                                                     "deadline": URGENT_DEADLINE if r["urgent"] else ROUTINE_DEADLINE}
                             for r in env.requests],
            })
        return {"type": "frame", "step": max(e.steps for e in self.envs), "shift": self.shift + 1,
                "seed": self.seed + self.shift, "paused": paused, "fps": fps, "finished": self.finished,
                "drones": drones}


class ActRequest(BaseModel):
    observation: list[float]
    agent: str | None = None


def create_app(agents, scenario="standard", seed=10_000, fps=4, verbose=True):
    app = FastAPI(title="Rwanda Medical Drone API")
    by_name = {a.name.lower(): a for a in agents}

    def meta():
        return {"type": "meta", "grid_size": GRID_SIZE, "depot": DEPOT, "clinics": dict(zip(CLINIC_NAMES, CLINIC_POSITIONS)),
                "actions": ACTIONS, "observation_fields": OBS_FIELDS, "battery_max": BATTERY_MAX, "capacity": CAPACITY,
                "max_steps": MAX_STEPS, "scenario": scenario,
                "agents": [{"name": a.name, "label": a.label, "color": COLORS[a.algo]} for a in agents]}

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/meta")
    def get_meta():
        return meta()

    @app.post("/api/act")
    def act(request: ActRequest):
        if len(request.observation) != len(OBS_FIELDS):
            raise HTTPException(422, f"observation must have {len(OBS_FIELDS)} values: {OBS_FIELDS}")
        agent = by_name.get((request.agent or agents[0].name).lower())
        if agent is None or agent.net is None:
            raise HTTPException(404, f"unknown agent; choose from {[a.name for a in agents if a.net is not None]}")
        action, scores, kind = agent.act(None, request.observation)
        return {"agent": agent.name, "action": action, "action_name": ACTIONS[action], "scores": scores, "score_kind": kind}

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        await ws.accept()
        race = Race(agents, scenario, seed, verbose)
        state = {"paused": True, "fps": fps}

        async def listen():
            while True:
                message = await ws.receive_json()
                kind = message.get("type")
                if kind == "emergency" and 0 <= int(message.get("clinic", -1)) < len(CLINIC_NAMES):
                    race.emergency(int(message["clinic"]))
                elif kind == "pause":
                    state["paused"] = not state["paused"]
                elif kind == "start":
                    state["paused"] = False
                elif kind in ("reset", "next"):
                    race.next_shift()
                    state["paused"] = kind == "reset"
                elif kind == "speed":
                    state["fps"] = max(1, min(20, int(message.get("fps", fps))))
                    if verbose:
                        print(f"--- speed: {state['fps']} steps/s", flush=True)

        listener = asyncio.create_task(listen())
        await ws.send_json(meta())
        await ws.send_json(race.frame(**state))
        try:
            while not listener.done():
                await asyncio.sleep(1 / state["fps"])
                if not race.finished and not state["paused"]:
                    race.step()
                await ws.send_json(race.frame(**state))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            listener.cancel()

    return app
