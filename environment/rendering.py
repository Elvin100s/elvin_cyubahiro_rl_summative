"""2D pygame view of the drone delivery environment."""
import math
import os

import numpy as np

from environment.custom_env import (
    BATTERY_MAX, CAPACITY, CLINIC_NAMES, CLINIC_POSITIONS, DEPOT, GRID_SIZE, MAX_STEPS, ROUTINE_DEADLINE,
    URGENT_DEADLINE,
)

CELL = 64
MAP = CELL * GRID_SIZE
PANEL = 360
SUBFRAMES = 6
POPUP_STEPS = 4
FLASH_STEPS = 3
ACTION_SHORT = ["N", "S", "W", "E", "Load", "Drop", "Chg"]

INK = (232, 238, 245)
MUTED = (140, 152, 168)
FAINT = (84, 94, 110)
PANEL_BG = (17, 22, 31)
CARD = (27, 34, 46)
GRASS = [(58, 112, 72), (63, 119, 76), (54, 105, 68)]
LAKE = (52, 110, 160)
LAKE_LIGHT = (84, 142, 190)
ROAD_EDGE = (120, 104, 72)
ROAD = (206, 186, 140)
TREE = (34, 80, 50)
DEPOT_COLOR = (66, 133, 244)
IDLE = (150, 160, 172)
ROUTINE = (245, 176, 65)
URGENT = (235, 77, 75)
OK = (72, 199, 116)
RECHARGE = (110, 170, 255)


def _center(pos):
    return pos[0] * CELL + CELL / 2, pos[1] * CELL + CELL / 2


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _battery_color(level):
    return OK if level > 0.5 else ROUTINE if level > 0.25 else URGENT


class Renderer:
    def __init__(self, render_mode, fps):
        if render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        self.mode, self.fps = render_mode, fps
        size = (MAP + PANEL, MAP)
        if render_mode == "human":
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Rwanda Medical Drone Delivery")
        else:
            self.screen = pygame.Surface(size)
        self.clock = pygame.time.Clock()
        self.font = {s: pygame.font.Font(None, s) for s in (16, 18, 20, 22, 26, 34)}
        self.background = self._draw_background()
        self.tints = {name: pygame.Surface((MAP, MAP)) for name in ("dawn", "dusk")}
        self.tints["dawn"].fill((255, 160, 80))
        self.tints["dusk"].fill((40, 50, 120))
        self.range_cache = {}
        self.frame = 0
        self.prev_drone = None
        self._reset_history()

    def draw(self, env):
        pg = self.pg
        current = tuple(env.drone)
        if self.prev_drone is None or env.steps == 0:
            self.prev_drone = current
        self._observe(env)
        if self.mode == "rgb_array":
            self._frame(env, current, 1.0)
            self.prev_drone = current
            return np.transpose(pg.surfarray.array3d(self.screen), (1, 0, 2))

        for i in range(1, SUBFRAMES + 1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    raise SystemExit
            t = i / SUBFRAMES
            pos = tuple(p + (c - p) * t for p, c in zip(self.prev_drone, current))
            self._frame(env, pos, t)
            pg.display.flip()
            self.clock.tick(self.fps * SUBFRAMES)
        if current != self.prev_drone:
            self.trail = (self.trail + [self.prev_drone])[-12:]
        self.prev_drone = current
        return None

    def close(self):
        self.pg.quit()

    # ------------------------------------------------------------------ history of the shift
    def _reset_history(self):
        self.trail, self.rewards, self.events, self.popups, self.flashes = [], [0.0], [], [], []
        self.seen_step, self.prev_requests, self.prev_stats = 0, None, None

    @staticmethod
    def _snapshot(env):
        return [None if r is None else r["urgent"] for r in env.requests]

    def _log(self, step, text, color):
        self.events = (self.events + [(step, text, color)])[-5:]

    def _observe(self, env):
        if env.steps == 0 or env.steps < self.seen_step or self.prev_stats is None:
            self._reset_history()
            self.prev_requests, self.prev_stats = self._snapshot(env), dict(env.stats)
            if env.steps == 0:
                return
        if env.steps == self.seen_step:
            return
        self.seen_step = env.steps
        self.rewards.append(env.total_reward)
        stats, requests, here = env.stats, self._snapshot(env), tuple(env.drone)

        if abs(env.last_reward) >= 1:
            self.popups.append((env.steps, _center(here), f"{env.last_reward:+.1f}",
                                OK if env.last_reward > 0 else URGENT))
        delivered_at = None
        if stats["delivered"] > self.prev_stats["delivered"]:
            delivered_at = env._clinic_here()
            self.flashes.append((env.steps, _center(here), OK))
            self._log(env.steps, env.last_event, URGENT if "URGENT" in env.last_event else OK)
        for i, (before, after) in enumerate(zip(self.prev_requests, requests)):
            name = CLINIC_NAMES[i]
            if before is not None and after is None and i != delivered_at:
                self.flashes.append((env.steps, _center(CLINIC_POSITIONS[i]), URGENT))
                self._log(env.steps, f"{'URGENT ' if before else ''}request expired at {name}", URGENT)
            elif before is None and after:
                self._log(env.steps, f"New URGENT request at {name}", ROUTINE)
        if env.last_action == "recharge" and tuple(env.drone) == DEPOT:
            text = f"Recharged to {env.battery:.0f}"
            if self.events and self.events[-1][1].startswith("Recharged"):
                self.events[-1] = (env.steps, text, RECHARGE)
            else:
                self._log(env.steps, text, RECHARGE)
        if stats["crashed"] and not self.prev_stats["crashed"]:
            self._log(env.steps, "CRASH: battery empty", URGENT)
        self.prev_requests, self.prev_stats = requests, dict(stats)

    # ------------------------------------------------------------------ static map
    def _draw_background(self):
        pg = self.pg
        surface = pg.Surface((MAP, MAP))
        rng = np.random.default_rng(7)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                pg.draw.rect(surface, GRASS[rng.integers(len(GRASS))], (x * CELL, y * CELL, CELL, CELL))

        hills = pg.Surface((MAP, MAP), pg.SRCALPHA)
        for _ in range(28):
            cx, cy, r = rng.uniform(0, MAP), rng.uniform(0, MAP), rng.uniform(40, 110)
            pg.draw.ellipse(hills, (20, 50, 30, 34), (cx - r + 10, cy - r * 0.7 + 10, 2 * r, 1.4 * r))
            pg.draw.ellipse(hills, (150, 200, 140, 30), (cx - r, cy - r * 0.7, 2 * r, 1.4 * r))
        surface.blit(hills, (0, 0))

        pg.draw.ellipse(surface, LAKE, (-CELL * 0.9, CELL * 2.6, CELL * 1.5, CELL * 5.2))
        pg.draw.ellipse(surface, LAKE_LIGHT, (-CELL * 0.9, CELL * 2.6, CELL * 1.5, CELL * 5.2), 3)
        self._text(surface, "Lake Kivu", (CELL * 0.05, CELL * 7.4), 16, (200, 225, 245))

        depot = _center(DEPOT)
        for i, pos in enumerate(CLINIC_POSITIONS):
            self._road(surface, depot, _center(pos), 0.18 if i % 2 else -0.18)

        for _ in range(80):
            x, y = rng.uniform(8, MAP - 8, size=2)
            near = [depot] + [_center(p) for p in CLINIC_POSITIONS]
            if x < CELL * 0.7 and 2.6 * CELL < y < 7.8 * CELL or any(math.dist((x, y), n) < 38 for n in near):
                continue
            pg.draw.circle(surface, TREE, (x, y), 5)
            pg.draw.circle(surface, (60, 120, 74), (x - 1.5, y - 1.5), 2)

        grid = pg.Surface((MAP, MAP), pg.SRCALPHA)
        for i in range(GRID_SIZE + 1):
            pg.draw.line(grid, (20, 50, 30, 60), (i * CELL, 0), (i * CELL, MAP))
            pg.draw.line(grid, (20, 50, 30, 60), (0, i * CELL), (MAP, i * CELL))
        surface.blit(grid, (0, 0))

        cx, cy = depot
        pg.draw.circle(surface, (40, 48, 60), (cx, cy), 27)
        pg.draw.circle(surface, DEPOT_COLOR, (cx, cy), 24)
        pg.draw.circle(surface, INK, (cx, cy), 24, 2)
        self._text(surface, "H", (cx, cy), 34, INK, center=True)
        self._pill(surface, "Kigali depot", (cx, cy + 38))
        for name, pos in zip(CLINIC_NAMES, CLINIC_POSITIONS):
            px, py = _center(pos)
            pg.draw.rect(surface, (60, 60, 60), (px - 13, py - 11, 28, 26), border_radius=4)
            pg.draw.rect(surface, (240, 242, 246), (px - 14, py - 13, 28, 26), border_radius=4)
            pg.draw.rect(surface, URGENT, (px - 3, py - 9, 6, 18))
            pg.draw.rect(surface, URGENT, (px - 9, py - 3, 18, 6))
            self._pill(surface, name, (px, py + 38))
        return surface

    def _road(self, surface, start, end, bend):
        (x0, y0), (x2, y2) = start, end
        length = math.dist(start, end)
        cx = (x0 + x2) / 2 - (y2 - y0) / length * bend * length
        cy = (y0 + y2) / 2 + (x2 - x0) / length * bend * length
        points = [((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x2,
                   (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y2) for t in np.linspace(0, 1, 30)]
        self.pg.draw.lines(surface, ROAD_EDGE, False, points, 7)
        self.pg.draw.lines(surface, ROAD, False, points, 4)

    def _pill(self, surface, text, center):
        image = self.font[18].render(text, True, INK)
        rect = image.get_rect(center=center).inflate(10, 4)
        pill = self.pg.Surface(rect.size, self.pg.SRCALPHA)
        self.pg.draw.rect(pill, (15, 20, 28, 170), pill.get_rect(), border_radius=6)
        surface.blit(pill, rect)
        surface.blit(image, image.get_rect(center=center))

    # ------------------------------------------------------------------ one animation frame
    def _frame(self, env, drone_pos, t):
        self.frame += 1
        self.screen.blit(self.background, (0, 0))
        self._time_tint(env)
        self._safe_range(env)
        self._clinics(env)
        self._flashes(env, t)
        for i, pos in enumerate(self.trail):
            self.pg.draw.circle(self.screen, (210, 235, 255), _center(pos), 2 + i // 4)
        self._drone(env, _center(drone_pos))
        self._popups(env, t)
        if env.stats["crashed"]:
            overlay = self.pg.Surface((MAP, MAP), self.pg.SRCALPHA)
            overlay.fill((180, 20, 20, 90))
            self.screen.blit(overlay, (0, 0))
            self._text(self.screen, "CRASHED: battery empty", (MAP / 2, MAP / 2), 34, INK, center=True)
        self._panel(env)

    def _time_tint(self, env):
        f = env.steps / MAX_STEPS
        for name, alpha in (("dawn", 45 * (1 - f / 0.2) if f < 0.2 else 0),
                            ("dusk", 70 * (f - 0.75) / 0.25 if f > 0.75 else 0)):
            if alpha > 0:
                self.tints[name].set_alpha(int(alpha))
                self.screen.blit(self.tints[name], (0, 0))

    def _safe_range(self, env):
        key = (tuple(env.drone), int(env.battery))
        if key not in self.range_cache:
            shade = self.pg.Surface((MAP, MAP), self.pg.SRCALPHA)
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    if _distance(env.drone, (x, y)) + _distance((x, y), DEPOT) > env.battery:
                        shade.fill((8, 12, 20, 95), (x * CELL, y * CELL, CELL, CELL))
            self.range_cache = {key: shade}
        self.screen.blit(self.range_cache[key], (0, 0))

    def _clinics(self, env):
        pg = self.pg
        pulse = 0.5 + 0.5 * math.sin(self.frame / 4)
        for pos, request in zip(CLINIC_POSITIONS, env.requests):
            px, py = _center(pos)
            if request is None:
                pg.draw.circle(self.screen, IDLE, (px, py), 22, 2)
                continue
            color = URGENT if request["urgent"] else ROUTINE
            pg.draw.circle(self.screen, color, (px, py), 22 + (5 * pulse if request["urgent"] else 0), 4)
            deadline = URGENT_DEADLINE if request["urgent"] else ROUTINE_DEADLINE
            self._bar(self.screen, (px - 24, py - 34, 48, 6), 1 - request["age"] / deadline, color)

    def _flashes(self, env, t):
        self.flashes = [f for f in self.flashes if env.steps - f[0] + t <= FLASH_STEPS]
        for born, (x, y), color in self.flashes:
            age = env.steps - born + t
            ring = self.pg.Surface((160, 160), self.pg.SRCALPHA)
            self.pg.draw.circle(ring, (*color, int(220 * (1 - age / FLASH_STEPS))), (80, 80), 20 + age * 18, 4)
            self.screen.blit(ring, (x - 80, y - 80))

    def _popups(self, env, t):
        self.popups = [p for p in self.popups if env.steps - p[0] + t <= POPUP_STEPS]
        for born, (x, y), text, color in self.popups:
            age = env.steps - born + t
            image = self.font[26].render(text, True, color)
            shadow = self.font[26].render(text, True, (10, 10, 10))
            alpha = int(255 * (1 - age / POPUP_STEPS))
            image.set_alpha(alpha)
            shadow.set_alpha(alpha)
            pos = (x + 18, y - 30 - age * 14)
            self.screen.blit(shadow, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(image, pos)

    def _drone(self, env, center):
        pg = self.pg
        x, y = center
        shadow = pg.Surface((44, 16), pg.SRCALPHA)
        pg.draw.ellipse(shadow, (0, 0, 0, 80), shadow.get_rect())
        self.screen.blit(shadow, (x - 16, y + 14))
        spin = self.frame * 0.6
        for k, (dx, dy) in enumerate(((-15, -15), (15, -15), (-15, 15), (15, 15))):
            rx, ry = x + dx, y + dy
            pg.draw.line(self.screen, (30, 30, 30), (x, y), (rx, ry), 3)
            pg.draw.circle(self.screen, (200, 210, 224), (rx, ry), 9, 1)
            for blade in (0, math.pi):
                angle = spin + k * 0.8 + blade
                pg.draw.line(self.screen, (240, 244, 250), (rx, ry),
                             (rx + 8 * math.cos(angle), ry + 8 * math.sin(angle)), 2)
        pg.draw.circle(self.screen, _battery_color(env.battery / BATTERY_MAX), (x, y), 10)
        pg.draw.circle(self.screen, (20, 20, 20), (x, y), 10, 2)
        for i in range(env.packages):
            pg.draw.rect(self.screen, (255, 196, 92), (x - 9 + i * 11, y + 14, 8, 8), border_radius=2)

    # ------------------------------------------------------------------ side panel
    def _panel(self, env):
        pg = self.pg
        left, width = MAP + 16, PANEL - 32
        pg.draw.rect(self.screen, PANEL_BG, (MAP, 0, PANEL, MAP))
        self._text(self.screen, "RWANDA MEDICAL DRONE", (left, 14), 26, INK)
        self._text(self.screen, self._fit(getattr(env, "label", None) or "GCGO: Healthcare", 18, width),
                   (left, 38), 18, MUTED)

        f = env.steps / MAX_STEPS
        self._pair(left, 60, "Step", f"{env.steps} / {MAX_STEPS}",
                   "Morning" if f < 0.33 else "Midday" if f < 0.66 else "Evening")
        self._bar(self.screen, (left, 80, width, 6), f, DEPOT_COLOR)
        level = env.battery / BATTERY_MAX
        self._pair(left, 92, "Battery", f"{env.battery:4.1f} / {BATTERY_MAX:.0f}", None, _battery_color(level))
        self._bar(self.screen, (left, 112, width, 6), level, _battery_color(level))
        self._pair(left, 124, "Packages", f"{env.packages} / {CAPACITY}", None)
        self._pair(left + width // 2, 124, "Reward", f"{env.total_reward:+.1f}", None,
                   OK if env.total_reward >= 0 else URGENT)
        self._pair(left, 144, "Delivered", f"{env.stats['delivered']} ({env.stats['urgent_delivered']} urg.)", None)
        self._pair(left + width // 2, 144, "Expired", f"{env.stats['expired']} ({env.stats['urgent_expired']} urg.)",
                   None, URGENT if env.stats["urgent_expired"] else INK)
        pg.draw.rect(self.screen, CARD, (left - 6, 166, width + 12, 26), border_radius=6)
        event = f"{(env.last_action or '-').upper()}  ·  {env.last_event}"
        self._text(self.screen, self._fit(event, 20, width), (left, 171), 20,
                   URGENT if "URGENT" in env.last_event or "crash" in env.last_event else INK)

        y = self._weights(env, left, 202, width)
        y = self._sparkline(env, left, y, width)
        y = self._event_log(left, y, width)
        self._requests(env, left, y, width)
        self._legend(left, MAP - 24)

    def _pair(self, x, y, label, value, note=None, color=INK):
        self._text(self.screen, label, (x, y), 20, MUTED)
        self._text(self.screen, value, (x + 70, y), 20, color)
        if note:
            image = self.font[18].render(note, True, MUTED)
            self.screen.blit(image, (MAP + PANEL - 16 - image.get_width(), y + 1))

    def _heading(self, text, x, y, width, note=None):
        self._text(self.screen, text, (x, y), 18, MUTED)
        if note:
            image = self.font[16].render(note, True, FAINT)
            self.screen.blit(image, (x + width - image.get_width(), y + 1))

    def _weights(self, env, left, y, width):
        pg = self.pg
        scores, kind = getattr(env, "scores", None), getattr(env, "score_kind", None)
        self._heading("WHAT IT'S WEIGHING", left, y, width,
                      None if scores is None else "relative Q-value" if kind == "Q-value" else "action probability")
        y += 18
        height, gap = 40, 6
        bar_w = (width - gap * 6) / 7
        if scores is None:
            self._text(self.screen, "baseline policy: no network scores", (left, y + 14), 18, FAINT)
            return y + height + 22
        valid = [s for s in scores if s is not None]
        lo, hi = min(valid), max(valid)
        top = scores.index(hi)
        for i, s in enumerate(scores):
            bx = left + i * (bar_w + gap)
            pg.draw.rect(self.screen, (36, 44, 58), (bx, y, bar_w, height), border_radius=3)
            if s is not None:
                frac = s if kind == "probability" else 0.12 + 0.88 * (s - lo) / ((hi - lo) or 1)
                bh = max(2, frac * height)
                pg.draw.rect(self.screen, INK if i == top else (96, 112, 135), (bx, y + height - bh, bar_w, bh),
                             border_radius=3)
            label = self.font[16].render(ACTION_SHORT[i], True, INK if i == top else MUTED if s is not None else FAINT)
            self.screen.blit(label, label.get_rect(center=(bx + bar_w / 2, y + height + 9)))
        return y + height + 22

    def _sparkline(self, env, left, y, width):
        pg = self.pg
        self._heading("REWARD OVER THE SHIFT", left, y, width, f"{env.total_reward:+.1f}")
        y += 18
        height = 42
        pg.draw.rect(self.screen, (24, 30, 41), (left, y, width, height), border_radius=4)
        lo, hi = min(min(self.rewards), -10), max(max(self.rewards), 10)
        to_y = lambda v: y + height - 4 - (v - lo) / (hi - lo) * (height - 8)
        pg.draw.line(self.screen, FAINT, (left, to_y(0)), (left + width, to_y(0)), 1)
        points = [(left + i / MAX_STEPS * width, to_y(v)) for i, v in enumerate(self.rewards)]
        if len(points) > 1:
            pg.draw.lines(self.screen, OK if self.rewards[-1] >= 0 else URGENT, False, points, 2)
        return y + height + 10

    def _event_log(self, left, y, width):
        self._heading("RECENT EVENTS", left, y, width)
        y += 18
        for step, text, color in reversed(self.events):
            self._text(self.screen, f"{step:>3}", (left, y), 18, FAINT)
            self._text(self.screen, self._fit(text, 18, width - 34), (left + 34, y), 18, color)
            y += 17
        return y + (5 - len(self.events)) * 17 + 8

    def _requests(self, env, left, y, width):
        self._heading("REQUESTS", left, y, width, "time left")
        y += 18
        for name, request in zip(CLINIC_NAMES, env.requests):
            self._text(self.screen, name, (left, y), 18, INK)
            if request is None:
                self._text(self.screen, "none", (left + 100, y), 18, FAINT)
            else:
                deadline = URGENT_DEADLINE if request["urgent"] else ROUTINE_DEADLINE
                color = URGENT if request["urgent"] else ROUTINE
                self._text(self.screen, "URGENT" if request["urgent"] else "routine", (left + 100, y), 18, color)
                self._bar(self.screen, (left + 170, y + 4, width - 170, 7), 1 - request["age"] / deadline, color)
            y += 18

    def _legend(self, left, y):
        pg = self.pg
        x = left
        for label, color in (("urgent", URGENT), ("routine", ROUTINE), ("none", IDLE)):
            pg.draw.circle(self.screen, color, (x + 5, y + 7), 5)
            self._text(self.screen, label, (x + 13, y), 16, MUTED)
            x += 70 if label != "none" else 52
        pg.draw.rect(self.screen, (60, 66, 78), (x, y + 2, 11, 11))
        self._text(self.screen, "beyond safe range", (x + 16, y), 16, MUTED)

    # ------------------------------------------------------------------ primitives
    def _bar(self, surface, rect, fraction, color):
        x, y, w, h = rect
        self.pg.draw.rect(surface, (55, 63, 77), rect, border_radius=3)
        self.pg.draw.rect(surface, color, (x, y, max(0, min(1, fraction)) * w, h), border_radius=3)

    def _fit(self, text, size, width):
        font = self.font[size]
        if font.size(text)[0] <= width:
            return text
        while text and font.size(text + "...")[0] > width:
            text = text[:-1]
        return text + "..."

    def _text(self, surface, text, pos, size, color, center=False):
        image = self.font[size].render(text, True, color)
        rect = image.get_rect(center=pos) if center else image.get_rect(topleft=pos)
        surface.blit(image, rect)
