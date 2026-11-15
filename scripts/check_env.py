"""Sanity checks for the environment. Run from the repo root:

    python scripts/check_env.py            # all checks (~ a few minutes)
    python scripts/check_env.py --quick    # shorter episodes

1. Gymnasium / Stable-Baselines3 API check
2. Calibration: fixed-time controller at each congestion level, no disruption
   (queues should stay bounded for v/c < 1 and grow for oversaturated)
3. Each disruption under fixed-time control (the disruption should be visible)
4. Speed: agent steps per second -> estimate training time
Writes a plot to results/check_env.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from envs.metrics import summarize  # noqa: E402
from envs.scenarios import CONGESTION_LEVELS, Scenario  # noqa: E402
from envs.traffic_env import make_env  # noqa: E402


def run_episode(env, scenario, policy="fixed"):
    obs, info = env.reset(options={"scenario": scenario})
    done = False
    while not done:
        action = None if policy == "fixed" else env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
    return info["history"], info["scenario"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    H = 1800 if args.quick else 3600
    td = H // 3

    # 1. API check -------------------------------------------------------
    from stable_baselines3.common.env_checker import check_env

    env = make_env(split="train", seed=0, horizon=H)
    check_env(env, warn=True)
    print(f"[1] API OK | obs shape {env.observation_space.shape} | actions {env.action_space.n}")
    env.close()

    # 2. Calibration ------------------------------------------------------
    print("\n[2] Fixed-time calibration (no disruption)")
    fixed = make_env(fixed_ts=True, horizon=H)
    for name, vc in CONGESTION_LEVELS.items():
        hist, sc = run_episode(fixed, Scenario(vc=vc, horizon=H, seed=1))
        s = summarize(hist, sc)
        q = np.array([h["stopped"] for h in hist])
        first, last = q[: len(q) // 3].mean(), q[-len(q) // 3 :].mean()
        print(f"  {name:14s} v/c={vc:.2f}  mean queue {s['mean_queue']:6.1f}  "
              f"(first third {first:5.1f} -> last third {last:5.1f})  "
              f"mean wait {s['mean_wait_s']:6.1f}s  pending max {s['max_pending']:4d}  "
              f"throughput {s['throughput']:5d}  teleports {s['teleports']}")

    # 3. Disruptions --------------------------------------------------------
    print(f"\n[3] Disruptions under fixed-time control (v/c=0.55, disruption at t={td}s)")
    cases = {
        "none": Scenario(vc=0.55, disruption="none", disrupt_time=td, horizon=H, seed=2),
        "surge": Scenario(vc=0.55, disruption="surge", disrupt_time=td, disrupt_duration=H // 4,
                          surge_factor=1.6, horizon=H, seed=2),
        "flip": Scenario(vc=0.55, ns_share=0.75, disruption="flip", disrupt_time=td, horizon=H, seed=2),
        "lane_closure": Scenario(vc=0.55, disruption="lane_closure", disrupt_time=td,
                                 disrupt_duration=H // 4, closed_approach="n", horizon=H, seed=2),
    }
    curves = {}
    for name, sc in cases.items():
        hist, scd = run_episode(fixed, sc)
        s = summarize(hist, scd)
        curves[name] = hist
        rec = "n/a" if s["recovery_s"] is None else f"{s['recovery_s']:.0f}s"
        print(f"  {name:13s} queue before {s['queue_before']:5.1f} -> after {s['queue_after']:5.1f}  "
              f"recovery {rec:>6s}  throughput {s['throughput']:5d}  teleports {s['teleports']}")
    fixed.close()

    # 4. Speed --------------------------------------------------------------
    env = make_env(split="train", seed=1, horizon=H)
    env.reset()
    n, t0 = 300, time.time()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    sps = n / (time.time() - t0)
    env.close()
    print(f"\n[4] Speed: {sps:.0f} agent steps/s (= {sps*5:.0f} simulated s per real s)")
    print(f"    -> 1M training steps ~ {1e6/sps/3600:.1f} h of environment time on 1 core")
    if os.environ.get("LIBSUMO_AS_TRACI") != "1":
        print("    (tip: pip install libsumo and set LIBSUMO_AS_TRACI=1 for a ~10x speed-up)")

    # Plot ------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, hist in curves.items():
            t = np.array([h["t"] for h in hist]) / 60
            q = np.convolve([h["stopped"] for h in hist], np.ones(12) / 12, mode="same")
            ax.plot(t, q, label=name)
        ax.axvline(td / 60, color="k", ls="--", lw=1)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("stopped vehicles (1-min avg)")
        ax.set_title("Fixed-time control under each disruption")
        ax.legend()
        fig.tight_layout()
        fig.savefig("results/check_env.png", dpi=130)
        print("\nPlot saved to results/check_env.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
