"""Watch one episode in the SUMO GUI. Run from the repo root:

    python scripts/watch.py                                   # surge, fixed-time control
    python scripts/watch.py --disruption lane_closure --vc 0.8
    python scripts/watch.py --policy random --delay 50
    python scripts/watch.py --model runs/ppo.zip              # a trained SB3 model (PPO or RecurrentPPO)

Options (all optional):
    --disruption        none | surge | flip | lane_closure            (default: surge)
    --vc                congestion level, nominal v/c ratio:          (default: 0.6)
                        0.30 light, 0.55 moderate, 0.80 heavy, 1.05 oversaturated
    --policy            fixed (fixed-time plan) | random              (default: fixed)
                        ignored when --model is given
    --model             path to a trained Stable-Baselines3 .zip, PPO or RecurrentPPO
    --horizon           episode length in simulated seconds           (default: 1800)
    --disrupt-time      when the disruption starts, in seconds        (default: 300)
                        early on purpose, so you don't wait long to see it
    --disrupt-duration  how long a surge / lane closure lasts, in s   (default: 600)
                        flips are permanent
    --delay             playback speed: real ms per simulated second  (default: 100)
                        0 = as fast as possible
    --seed              SUMO random seed; same seed -> same traffic   (default: 0)

In the GUI: scroll to zoom into the junction, space to pause. Closing the
window ends the run. When the episode finishes, the terminal shows the
summary from envs.metrics.summarize().

Requirements: a display that sumo-gui can open a window on.
    Linux desktop, Windows: works as is.
    WSL2:     needs WSLg (Windows 11 / recent Windows 10); `echo $DISPLAY` must print something.
    macOS:    sumo-gui may need XQuartz (https://www.xquartz.org/).
    Headless servers / SSH without X forwarding: not supported.
libsumo has no GUI, so this script turns LIBSUMO_AS_TRACI off for its own
run (no need to unset it yourself); it runs slower than training.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# libsumo has no GUI: fall back to the (slower) TraCI socket connection.
# Must happen before sumo_rl / traci are imported.
os.environ.pop("LIBSUMO_AS_TRACI", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.metrics import summarize  # noqa: E402
from envs.scenarios import DISRUPTIONS, Scenario  # noqa: E402
from envs.traffic_env import make_env  # noqa: E402


def load_model(path: str):
    """Load a Stable-Baselines3 model, trying PPO first, then RecurrentPPO."""
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    for algo in (PPO, RecurrentPPO):
        try:
            return algo.load(path)
        except Exception:
            continue
    raise SystemExit(f"Could not load {path} as PPO or RecurrentPPO")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disruption", choices=DISRUPTIONS, default="surge")
    ap.add_argument("--vc", type=float, default=0.6, help="congestion level (nominal v/c), e.g. 0.3-1.05")
    ap.add_argument("--policy", choices=["fixed", "random"], default="fixed",
                    help="ignored when --model is given")
    ap.add_argument("--model", help="path to a trained SB3 model .zip")
    ap.add_argument("--horizon", type=int, default=1800, help="episode length in simulated seconds")
    ap.add_argument("--disrupt-time", type=int, default=300)
    ap.add_argument("--disrupt-duration", type=int, default=600)
    ap.add_argument("--delay", type=int, default=100, help="GUI delay in ms per simulated second")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenario = Scenario(
        vc=args.vc,
        ns_share=0.75 if args.disruption == "flip" else 0.5,
        disruption=args.disruption,
        disrupt_time=args.disrupt_time,
        disrupt_duration=args.disrupt_duration,
        horizon=args.horizon,
        seed=args.seed,
    )

    model = load_model(args.model) if args.model else None
    fixed = model is None and args.policy == "fixed"
    env = make_env(scenario=scenario, horizon=args.horizon, fixed_ts=fixed, use_gui=True)

    obs, info = env.reset()
    controller = "model " + args.model if model else args.policy
    print(f"Scenario: {info['scenario']}\nController: {controller}")
    if scenario.disruption == "lane_closure":
        print(f"Lane closure on approach '{scenario.closed_approach}' "
              f"from t={scenario.disrupt_time}s to t={scenario.disrupt_end}s")

    lstm_state, episode_start = None, True
    done, was_disrupted = False, False
    try:
        while not done:
            if model is not None:
                action, lstm_state = model.predict(obs, state=lstm_state,
                                                   episode_start=[episode_start], deterministic=True)
                episode_start = False
            elif fixed:
                action = None
            else:
                action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            time.sleep(args.delay * env.unwrapped.delta_time / 1000)
            if info["disrupted"] != was_disrupted:
                was_disrupted = info["disrupted"]
                state = "STARTED" if was_disrupted else "ended"
                print(f"t={info['sim_time']:6.0f}s  disruption {state}")
    except Exception as e:  # closing the GUI window ends the TraCI connection
        print(f"Simulation stopped: {e.__class__.__name__}")
        env.close()
        return

    print("\nEpisode summary:")
    for k, v in summarize(info["history"], info["scenario"]).items():
        print(f"  {k:14s} {v}")
    env.close()


if __name__ == "__main__":
    main()
