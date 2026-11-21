"""Pilot: does memory help, and does it depend on what the controller can sense?

Trains PPO (no memory) and RecurrentPPO (LSTM memory) on random training
scenarios (surge / flip / none), for one sensing setting, then evaluates them
and the fixed-time plan on fixed test scenarios, including the unseen lane
closure. One sensing setting per process (libsumo allows one simulation per
process), so run the settings in parallel terminals:

    export LIBSUMO_AS_TRACI=1
    python scripts/pilot_memory.py --sensing full --steps 150000
    python scripts/pilot_memory.py --sensing 50   --steps 150000
    python scripts/pilot_memory.py --plot           # combine results -> runs/pilot/pilot.png

This is a quick, single-seed pilot to decide the project direction, not the
final experiment (that needs more training steps and several training seeds).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # one core per run; avoids thread contention when runs share a machine

from envs.metrics import summarize  # noqa: E402
from envs.scenarios import Scenario  # noqa: E402
from envs.traffic_env import make_env  # noqa: E402

OUT = os.path.join("runs", "pilot")
HORIZON = 3600


def test_scenarios(seeds=(100, 101, 102)) -> dict[str, list[Scenario]]:
    """Fixed evaluation set: heavy-ish traffic, disruption at 20 min."""
    common = dict(vc=0.7, disrupt_time=1200, disrupt_duration=900, horizon=HORIZON)
    return {
        "surge": [Scenario(disruption="surge", surge_factor=1.6, ns_share=0.5, seed=s, **common) for s in seeds],
        "flip": [Scenario(disruption="flip", ns_share=0.75, seed=s, **common) for s in seeds],
        # unseen in training; close a lane on the busier (N-S) approach so it bites
        "lane_closure": [Scenario(disruption="lane_closure", ns_share=0.65, closed_approach="n",
                                  seed=s, **common) for s in seeds],
    }


def run_eval(env, scenario, model=None):
    obs, info = env.reset(options={"scenario": scenario})
    state, start, done = None, True, False
    while not done:
        if model is None:
            action = None  # fixed-time
        else:
            action, state = model.predict(obs, state=state, episode_start=np.array([start]), deterministic=True)
            start = False
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return summarize(info["history"], info["scenario"])


def train_and_eval(sensing: str, steps: int, seed: int):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    rng = None if sensing == "full" else float(sensing)
    os.makedirs(OUT, exist_ok=True)
    common = dict(learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.99, ent_coef=0.01, seed=seed, verbose=0)
    results = {"sensing": sensing, "steps": steps, "seed": seed, "controllers": {}}

    controllers = {}
    for name, algo, policy in [("ppo", PPO, "MlpPolicy"), ("recurrent_ppo", RecurrentPPO, "MlpLstmPolicy")]:
        path = os.path.join(OUT, f"{name}_{sensing}.zip")
        if os.path.exists(path):  # reuse a finished run
            controllers[name] = algo.load(path)
            print(f"[{sensing}] loaded {path}", flush=True)
            continue
        env = make_env(split="train", seed=seed, horizon=HORIZON, sensing_range=rng)
        # small LSTM (64 units, default is 256): ~4x cheaper on CPU, plenty for a 29-dim input
        kwargs = dict(policy_kwargs=dict(lstm_hidden_size=64)) if algo is RecurrentPPO else {}
        model = algo(policy, env, **common, **kwargs)
        t0 = time.time()
        model.learn(total_timesteps=steps)
        mins = (time.time() - t0) / 60
        model.save(path)
        env.close()
        print(f"[{sensing}] trained {name} for {steps} steps in {mins:.1f} min -> {path}", flush=True)
        controllers[name] = model
        results.setdefault("train_minutes", {})[name] = round(mins, 1)

    # evaluation
    for name, model in [("fixed_time", None)] + list(controllers.items()):
        env = make_env(horizon=HORIZON, sensing_range=rng, fixed_ts=(model is None))
        per_dis = {}
        for dis, scs in test_scenarios().items():
            per_dis[dis] = [run_eval(env, sc, model) for sc in scs]
        env.close()
        results["controllers"][name] = per_dis
        line = "  ".join(
            f"{d}: wait {np.mean([r['mean_wait_s'] for r in rs]):5.1f}s queue {np.mean([r['mean_queue'] for r in rs]):5.1f}"
            for d, rs in per_dis.items()
        )
        print(f"[{sensing}] {name:14s} {line}", flush=True)

    with open(os.path.join(OUT, f"results_{sensing}.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)


def plot():
    import glob

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(OUT, "results_*.json")))
    if not files:
        raise SystemExit("no results yet")
    data = [json.load(open(f)) for f in files]
    dis = ["surge", "flip", "lane_closure"]
    ctrls = ["fixed_time", "ppo", "recurrent_ppo"]
    colors = {"fixed_time": "#9aa0a6", "ppo": "#4c78a8", "recurrent_ppo": "#f58518"}

    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 3.8), sharey=True, squeeze=False)
    rows = []
    for ax, d in zip(axes[0], data):
        x = np.arange(len(dis))
        for k, c in enumerate(ctrls):
            vals = [np.mean([r["mean_queue"] for r in d["controllers"][c][di]]) for di in dis]
            errs = [np.std([r["mean_queue"] for r in d["controllers"][c][di]]) for di in dis]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, yerr=errs, label=c.replace("_", " "), color=colors[c])
            rows += [(d["sensing"], c, di, v) for di, v in zip(dis, vals)]
        ax.set_xticks(x, ["surge", "flip", "lane closure\n(unseen)"])
        title = "full-lane sensing" if d["sensing"] == "full" else f"detector {d['sensing']} m"
        ax.set_title(title)
        ax.set_ylabel("mean queue (stopped vehicles)")
    axes[0][0].legend()
    fig.suptitle("Pilot: memory vs. no memory under different sensing (lower is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pilot.png"), dpi=130)
    print(f"saved {os.path.join(OUT, 'pilot.png')}")
    for r in rows:
        print(f"  {r[0]:>5s}  {r[1]:14s} {r[2]:13s} mean queue {r[3]:6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensing", default="full", help='"full" or a detector range in metres, e.g. 50')
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    plot() if a.plot else train_and_eval(a.sensing, a.steps, a.seed)
