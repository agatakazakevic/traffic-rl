"""Traffic scenarios: random (Poisson) arrivals, congestion levels and disruptions.

A scenario describes one episode:
  * a base congestion level (nominal volume-to-capacity ratio, v/c),
  * a directional split (share of traffic on the North-South axis),
  * at most one disruption that starts at `disrupt_time`.

`write_route_file()` turns a scenario into a SUMO .rou.xml file. Surges and
direction flips are encoded directly as time-varying flows. Lane closures
cannot be expressed in a route file, so they are applied at run time by
`traffic_env.AdaptiveTrafficEnv` through TraCI.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np

# --------------------------------------------------------------------------
# Network facts (nets/single-4way): 4 approaches, 3 lanes each
#   lane 0: right + straight, lane 1: straight, lane 2: left (protected phase)
# --------------------------------------------------------------------------
APPROACHES = ["n", "e", "s", "w"]
# (right, straight, left) destination edge for each incoming approach
MOVES = {
    "n": ("t_w", "t_s", "t_e"),
    "e": ("t_n", "t_w", "t_s"),
    "s": ("t_e", "t_n", "t_w"),
    "w": ("t_s", "t_e", "t_n"),
}
TURN_SPLIT = (0.15, 0.70, 0.15)  # right, straight, left

# Nominal capacity of ONE approach under the default fixed-time plan
# (cycle 92 s: 30 s through + 10 s protected left per axis, 3 s yellows).
#   through/right: 2 lanes * 1800 veh/h * 30/92 = ~1170 veh/h for 85% of demand
#   left:          1 lane  * 1800 veh/h * 10/92 = ~ 195 veh/h for 15% of demand
# -> an approach saturates at roughly 1300 veh/h. This is an approximation;
#    check it empirically with scripts/check_env.py (fixed-time queues should
#    stay bounded below v/c = 1 and grow above it).
APPROACH_CAPACITY_VPH = 1300.0

CONGESTION_LEVELS = {  # nominal v/c
    "light": 0.30,
    "moderate": 0.55,
    "heavy": 0.80,
    "oversaturated": 1.05,
}

DISRUPTIONS = ("none", "surge", "flip", "lane_closure")
TRAIN_DISRUPTIONS = ("none", "surge", "flip")
TEST_DISRUPTIONS = ("lane_closure",)  # held out: never seen in training


@dataclass
class Scenario:
    vc: float = 0.55                 # base nominal v/c ratio
    ns_share: float = 0.5            # share of total demand on the N-S axis
    disruption: str = "none"         # one of DISRUPTIONS
    disrupt_time: int = 1200         # simulation second when the disruption starts
    disrupt_duration: int = 1200     # seconds (surge / lane closure); flips are permanent
    surge_factor: float = 1.6        # demand multiplier during a surge
    closed_approach: str = "n"       # lane closure: which approach loses a lane
    closed_lane: int = 1             # lane index closed (1 = straight-only lane)
    horizon: int = 3600              # episode length in simulated seconds
    seed: int = 0                    # seed used for SUMO's random arrivals

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def disrupt_end(self) -> Optional[int]:
        if self.disruption in ("surge", "lane_closure"):
            return min(self.horizon, self.disrupt_time + self.disrupt_duration)
        return None


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def sample_scenario(
    rng: np.random.Generator,
    split: str = "train",
    horizon: int = 3600,
    vc_range: tuple[float, float] = (0.3, 0.9),
    disruptions: Optional[tuple[str, ...]] = None,
) -> Scenario:
    """Draw a random scenario.

    split="train" uses TRAIN_DISRUPTIONS, split="test" uses TEST_DISRUPTIONS,
    unless `disruptions` is given explicitly.
    """
    if disruptions is None:
        disruptions = TRAIN_DISRUPTIONS if split == "train" else TEST_DISRUPTIONS
    disruption = str(rng.choice(disruptions))

    vc = float(rng.uniform(*vc_range))
    # flips need a clearly unbalanced start so that the flip is meaningful
    if disruption == "flip":
        ns_share = float(rng.uniform(0.65, 0.8))
        if rng.random() < 0.5:
            ns_share = 1 - ns_share  # E-W heavy before the flip instead
    else:
        ns_share = float(rng.uniform(0.4, 0.6))

    return Scenario(
        vc=vc,
        ns_share=ns_share,
        disruption=disruption,
        disrupt_time=int(rng.uniform(0.25, 0.6) * horizon),
        disrupt_duration=int(rng.uniform(0.2, 0.35) * horizon),
        surge_factor=float(rng.uniform(1.4, 1.9)),
        closed_approach=str(rng.choice(APPROACHES)),
        closed_lane=1,
        horizon=horizon,
        seed=int(rng.integers(0, 2**31 - 1)),
    )


# --------------------------------------------------------------------------
# Demand -> route file
# --------------------------------------------------------------------------
def _approach_vph(vc: float, ns_share: float) -> dict[str, float]:
    """Vehicles/hour entering from each approach."""
    total = vc * 4 * APPROACH_CAPACITY_VPH
    ns, ew = total * ns_share, total * (1 - ns_share)
    return {"n": ns / 2, "s": ns / 2, "e": ew / 2, "w": ew / 2}


def demand_segments(sc: Scenario) -> list[tuple[int, int, dict[str, float]]]:
    """Piecewise-constant demand: list of (begin, end, {approach: veh/h})."""
    base = _approach_vph(sc.vc, sc.ns_share)
    t0, T = sc.disrupt_time, sc.horizon
    if sc.disruption == "surge":
        t1 = sc.disrupt_end
        surged = {a: q * sc.surge_factor for a, q in base.items()}
        segs = [(0, t0, base), (t0, t1, surged)]
        if t1 < T:
            segs.append((t1, T, base))
        return segs
    if sc.disruption == "flip":
        flipped = _approach_vph(sc.vc, 1 - sc.ns_share)
        return [(0, t0, base), (t0, T, flipped)]
    # "none" and "lane_closure" keep demand constant
    return [(0, T, base)]


def write_route_file(sc: Scenario, path: str) -> str:
    """Write a SUMO route file with Poisson-like arrivals for scenario `sc`."""
    lines = ["<routes>",
             '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9"/>']
    for a, (r, s, l) in MOVES.items():
        for name, dest in zip(("r", "s", "l"), (r, s, l)):
            lines.append(f'  <route id="{a}_{name}" edges="{a}_t {dest}"/>')
    for k, (b, e, vph) in enumerate(demand_segments(sc)):
        if e <= b:
            continue
        for a in APPROACHES:
            for name, frac in zip(("r", "s", "l"), TURN_SPLIT):
                # probability = chance a vehicle departs in each simulated second
                # -> Bernoulli per second, i.e. approximately Poisson arrivals
                p = min(1.0, vph[a] * frac / 3600.0)
                if p <= 0:
                    continue
                lines.append(
                    f'  <flow id="f{k}_{a}_{name}" type="car" route="{a}_{name}" begin="{b}" end="{e}" '
                    f'probability="{p:.5f}" departLane="best" departSpeed="max"/>'
                )
    lines.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
