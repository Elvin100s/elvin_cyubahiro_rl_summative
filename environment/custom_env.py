"""Rwanda medical drone delivery: a Gymnasium environment for the GCGO Healthcare mission."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

GRID_SIZE = 10
DEPOT = (6, 5)
CLINICS = {
    "Musanze": (3, 1),
    "Rubavu": (0, 3),
    "Nyagatare": (9, 1),
    "Kirehe": (9, 8),
    "Nyamagabe": (2, 8),
}
CLINIC_NAMES = list(CLINICS)
CLINIC_POSITIONS = list(CLINICS.values())

BATTERY_MAX = 50.0
FLY_COST = 1.0
GROUND_COST = 0.2
RECHARGE_RATE = 20.0
CAPACITY = 2
MAX_STEPS = 300

NEW_REQUEST_PROB = 0.04
URGENT_PROB = 0.35
URGENT_DEADLINE = 40
ROUTINE_DEADLINE = 120

ROUTINE_DELIVERY = 5.0
URGENT_DELIVERY = 10.0
URGENT_SPEED_BONUS = 10.0
URGENT_EXPIRED = -10.0
ROUTINE_EXPIRED = -3.0
URGENT_WAITING = -0.1
ROUTINE_WAITING = -0.02
INVALID_ACTION = -0.5
CRASH = -100.0
SHAPING_GAMMA = 0.99

ACTIONS = ["north", "south", "west", "east", "load", "deliver", "recharge"]
MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
LOAD, DELIVER, RECHARGE = 4, 5, 6

OBS_FIELDS = (
    ["drone_x", "drone_y", "battery", "packages", "time_left"]
    + [f"{name}_{field}" for name in CLINIC_NAMES for field in ("request", "urgent", "wait", "dx", "dy")]
    + ["depot_dx", "depot_dy"]
)

SCENARIOS = {
    "standard": {"request_prob": NEW_REQUEST_PROB, "urgent_prob": URGENT_PROB},
    "quiet": {"request_prob": 0.015, "urgent_prob": URGENT_PROB},
    "high_demand": {"request_prob": 0.10, "urgent_prob": URGENT_PROB},
    "urgent_surge": {"request_prob": NEW_REQUEST_PROB, "urgent_prob": 0.7},
}


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class DroneDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode=None, request_prob=NEW_REQUEST_PROB, urgent_prob=URGENT_PROB, shaping=0.0,
                 exploring_starts=0.0):
        """Training-only options: `shaping` adds potential-based shaping, `exploring_starts` is the share of
        shifts that start from a random position, battery and load. Evaluation always uses the defaults."""
        super().__init__()
        self.render_mode = render_mode
        self.request_prob = request_prob
        self.urgent_prob = urgent_prob
        self.shaping = shaping
        self.exploring_starts = exploring_starts
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(len(OBS_FIELDS),), dtype=np.float32)
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone = list(DEPOT)
        self.battery = BATTERY_MAX
        self.packages = 0
        self.steps = 0
        if self.exploring_starts and self.np_random.random() < self.exploring_starts:
            self.drone = [int(v) for v in self.np_random.integers(0, GRID_SIZE, size=2)]
            self.battery = float(self.np_random.uniform(min(BATTERY_MAX, _distance(self.drone, DEPOT) + 5), BATTERY_MAX))
            self.packages = int(self.np_random.integers(0, CAPACITY + 1))
        self.requests = [None] * len(CLINICS)
        for i in self.np_random.choice(len(CLINICS), size=self.np_random.integers(1, 3), replace=False):
            self.requests[i] = self._new_request()
        self.stats = {"delivered": 0, "urgent_delivered": 0, "expired": 0, "urgent_expired": 0, "crashed": False}
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_event = "Shift started"
        self.last_action = None
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        self.last_action = ACTIONS[action]
        self.steps += 1
        potential = self._potential()
        reward = self._apply_action(action) + self._update_requests()

        terminated = self.battery <= 0 and tuple(self.drone) != DEPOT
        if terminated:
            reward += CRASH + sum(self._expiry_penalty(r) for r in self.requests if r is not None)
            self.stats["crashed"] = True
            self.last_event = "Battery empty: crashed"
        self.battery = max(self.battery, 0.0)
        truncated = self.steps >= MAX_STEPS
        if self.shaping:
            reward += SHAPING_GAMMA * (0.0 if terminated else self._potential()) - potential

        self.last_reward = reward
        self.total_reward += reward
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _apply_action(self, action):
        """Anything done in the air (flying or hovering) costs FLY_COST; landed actions cost GROUND_COST."""
        at_depot = tuple(self.drone) == DEPOT
        if action in MOVES:
            self.battery -= FLY_COST
            x, y = self.drone[0] + MOVES[action][0], self.drone[1] + MOVES[action][1]
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                self.drone = [x, y]
                return 0.0
            return INVALID_ACTION

        if action == RECHARGE and at_depot:
            self.battery = min(BATTERY_MAX, self.battery + RECHARGE_RATE)
            self.last_event = "Recharging"
            return 0.0
        if action == LOAD and at_depot and self.packages < CAPACITY:
            self.battery -= GROUND_COST
            self.packages += 1
            self.last_event = "Loaded a package"
            return 0.0
        clinic = self._clinic_here()
        if action == DELIVER and clinic is not None and self.packages > 0 and self.requests[clinic] is not None:
            self.battery -= GROUND_COST
            return self._deliver(clinic)
        self.battery -= GROUND_COST if at_depot else FLY_COST
        return INVALID_ACTION

    def add_emergency(self, clinic):
        """Raise an urgent request at a clinic from outside the simulation (live demo)."""
        self.requests[clinic] = {"urgent": True, "age": 0}
        self.last_event = f"EMERGENCY raised at {CLINIC_NAMES[clinic]}"

    def _potential(self):
        """Minus the scaled number of steps to the next delivery (Ng et al., 1999), so the optimal policy is unchanged."""
        return -self.shaping * self.steps_to_next_delivery() if self.shaping else 0.0

    def steps_to_next_delivery(self):
        open_clinics = [pos for pos, request in zip(CLINIC_POSITIONS, self.requests) if request]
        if not open_clinics:
            return _distance(self.drone, DEPOT)
        if self.packages:
            return min(_distance(self.drone, pos) for pos in open_clinics)
        return _distance(self.drone, DEPOT) + 1 + min(_distance(DEPOT, pos) for pos in open_clinics)

    def _new_request(self):
        return {"urgent": bool(self.np_random.random() < self.urgent_prob), "age": 0}

    def _clinic_here(self):
        pos = tuple(self.drone)
        return CLINIC_POSITIONS.index(pos) if pos in CLINIC_POSITIONS else None

    @staticmethod
    def _deadline(request):
        return URGENT_DEADLINE if request["urgent"] else ROUTINE_DEADLINE

    @staticmethod
    def _expiry_penalty(request):
        return URGENT_EXPIRED if request["urgent"] else ROUTINE_EXPIRED

    def _deliver(self, clinic):
        request = self.requests[clinic]
        self.requests[clinic] = None
        self.packages -= 1
        self.stats["delivered"] += 1
        if request["urgent"]:
            self.stats["urgent_delivered"] += 1
            self.last_event = f"URGENT delivery to {CLINIC_NAMES[clinic]}"
            return URGENT_DELIVERY + URGENT_SPEED_BONUS * (1.0 - request["age"] / URGENT_DEADLINE)
        self.last_event = f"Delivery to {CLINIC_NAMES[clinic]}"
        return ROUTINE_DELIVERY

    def _update_requests(self):
        reward = 0.0
        arrivals, urgency = self.np_random.random((2, len(CLINICS)))
        for i, request in enumerate(self.requests):
            if request is None:
                if arrivals[i] < self.request_prob:
                    self.requests[i] = {"urgent": bool(urgency[i] < self.urgent_prob), "age": 0}
                continue
            request["age"] += 1
            if request["age"] >= self._deadline(request):
                reward += self._expiry_penalty(request)
                self.stats["expired"] += 1
                self.stats["urgent_expired"] += request["urgent"]
                self.requests[i] = None
            else:
                reward += URGENT_WAITING if request["urgent"] else ROUTINE_WAITING
        return reward

    def _get_obs(self):
        scale = GRID_SIZE - 1
        x, y = self.drone
        obs = [x / scale, y / scale, self.battery / BATTERY_MAX, self.packages / CAPACITY, 1.0 - self.steps / MAX_STEPS]
        for (cx, cy), request in zip(CLINIC_POSITIONS, self.requests):
            if request is None:
                obs += [0.0, 0.0, 0.0]
            else:
                obs += [1.0, float(request["urgent"]), request["age"] / self._deadline(request)]
            obs += [(cx - x) / scale, (cy - y) / scale]
        obs += [(DEPOT[0] - x) / scale, (DEPOT[1] - y) / scale]
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        return {"battery": self.battery, "packages": self.packages, "steps": self.steps, **self.stats}

    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from environment.rendering import Renderer
            self.renderer = Renderer(self.render_mode, self.metadata["render_fps"])
        return self.renderer.draw(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
