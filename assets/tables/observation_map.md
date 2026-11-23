| Index | Field | Meaning | Encoding / range | Simulation rule (real-world source) | Updated | Path to the agent |
|---|---|---|---|---|---|---|
| 0–1 | `drone_x`, `drone_y` | Drone cell on the 10×10 map | cell ÷ 9, in [0, 1] | Changes by one cell on a valid move (GPS fix) | every step | obs vector → TorchRL `observation` → MLP input |
| 2 | `battery` | Remaining charge | battery ÷ 50, in [0, 1] | −1 per airborne step (flying or hovering), −0.2 per landed action, +20 per recharge at the depot, capped at 50 (battery management system) | every step | same |
| 3 | `packages` | Packages on board | packages ÷ 2, in {0, 0.5, 1} | +1 on load at the depot, −1 on delivery (payload sensor / manifest) | every step | same |
| 4 | `time_left` | Share of the shift remaining | 1 − step ÷ 300, in [0, 1] | Falls by 1/300 per step (clock) | every step | same |
| 5 + 5i | `<clinic>_request` | Clinic i has an open request | 0 or 1 | Free clinic opens a request with probability 4% per step; cleared on delivery or expiry (health-centre ordering system) | every step | same |
| 6 + 5i | `<clinic>_urgent` | That request is urgent | 0 or 1 (0 if none) | Drawn at creation, 35% urgent (order priority flag) | when a request opens | same |
| 7 + 5i | `<clinic>_wait` | How long it has waited | age ÷ deadline, in [0, 1) (0 if none) | Age +1 per step; expires at 40 (urgent) or 120 (routine) steps (order timestamp) | every step | same |
| 8–9 + 5i | `<clinic>_dx`, `<clinic>_dy` | Offset from drone to clinic i | (clinic − drone) ÷ 9, in [−1, 1] | Recomputed from the drone position (map + GPS) | every step | same |
| 30–31 | `depot_dx`, `depot_dy` | Offset from drone to the Kigali depot | (depot − drone) ÷ 9, in [−1, 1] | Recomputed from the drone position | every step | same |

Clinics in order i = 0…4: Musanze, Rubavu, Nyagatare, Kirehe, Nyamagabe. The same vector also drives the feasible-action mask (`training/common.py:feasible_actions`), so the agent needs no information beyond these 32 values.
