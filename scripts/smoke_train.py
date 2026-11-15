"""Tiny training run to confirm the env works with Stable-Baselines3.

    LIBSUMO_AS_TRACI=1 python scripts/smoke_train.py
Not a real experiment: just checks that PPO and RecurrentPPO run end to end.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from envs.traffic_env import make_env  # noqa: E402

for algo, policy in [(PPO, "MlpPolicy"), (RecurrentPPO, "MlpLstmPolicy")]:
    env = make_env(split="train", seed=0, horizon=900)
    model = algo(policy, env, n_steps=256, batch_size=64, verbose=0, seed=0)
    model.learn(total_timesteps=2048)
    print(f"{algo.__name__}: trained 2048 steps OK")
    env.close()
