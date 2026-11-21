"""Gymnasium environment for adaptive traffic signal control under disruptions.

Wraps sumo-rl's SumoEnvironment (single agent, one intersection) and adds:
  * a new random scenario every episode (congestion level, direction split,
    one disruption at a random time) -- see envs/scenarios.py
  * lane closures applied live through TraCI
  * a per-step history of traffic metrics, returned in `info` at the end of
    the episode, so every method is evaluated with the same numbers

Usage:
    from envs.traffic_env import make_env
    env = make_env(split="train", seed=0)          # random training scenarios
    obs, info = env.reset()
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())

    # evaluate on one fixed scenario
    from envs.scenarios import Scenario
    obs, info = env.reset(options={"scenario": Scenario(disruption="lane_closure")})
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

import gymnasium as gym
import numpy as np

from .scenarios import Scenario, sample_scenario, write_route_file

if "SUMO_HOME" not in os.environ:  # pip-installed eclipse-sumo
    try:
        import sumo

        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install SUMO (pip install eclipse-sumo) or set SUMO_HOME") from e

from sumo_rl import SumoEnvironment  # noqa: E402  (needs SUMO_HOME first)
from sumo_rl.environment.observations import DefaultObservationFunction  # noqa: E402

from .observations import make_detector_observation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_FILE = os.path.join(ROOT, "nets", "single-4way", "intersection.net.xml")


class AdaptiveTrafficEnv(gym.Wrapper):
    """One intersection, one disruption per episode.

    Args:
        split: "train" samples surge / flip / no disruption; "test" samples
            held-out disruptions (lane closure). Ignored if `scenario` is set.
        scenario: if given, every episode uses this exact scenario.
        horizon: episode length in simulated seconds.
        delta_time: seconds between agent decisions.
        reward_fn: any sumo-rl reward name ("diff-waiting-time", "queue",
            "pressure", "average-speed") or a callable.
        fixed_ts: True = ignore actions and run the default fixed-time plan
            (the fixed-time baseline).
        seed: seeds the scenario sampler (not SUMO itself; each scenario
            carries its own SUMO seed so episodes are reproducible).
        sensing_range: None = full-lane observation (sumo-rl default);
            a number R = only vehicles within R m of the stop line are
            visible (stop-line detectors), see envs/observations.py.
    """

    def __init__(
        self,
        split: str = "train",
        scenario: Optional[Scenario] = None,
        horizon: int = 3600,
        delta_time: int = 5,
        reward_fn="diff-waiting-time",
        fixed_ts: bool = False,
        use_gui: bool = False,
        seed: Optional[int] = None,
        vc_range: tuple[float, float] = (0.3, 0.9),
        sensing_range: Optional[float] = None,
    ):
        self.split = split
        self.sensing_range = sensing_range
        self.fixed_scenario = scenario
        self.horizon = horizon
        self.vc_range = vc_range
        self.rng = np.random.default_rng(seed)

        # each env instance gets its own route file -> safe for parallel envs
        self.work_dir = tempfile.mkdtemp(prefix="tsc_")
        self.route_file = os.path.join(self.work_dir, "routes.rou.xml")
        write_route_file(Scenario(horizon=horizon), self.route_file)

        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=self.route_file,
            single_agent=True,
            num_seconds=horizon,
            delta_time=delta_time,
            yellow_time=3,
            min_green=5,
            max_green=60,
            reward_fn=reward_fn,
            fixed_ts=fixed_ts,
            use_gui=use_gui,
            sumo_warnings=False,
            time_to_teleport=300,  # safety valve against gridlock; teleports are logged
            add_per_agent_info=False,
            additional_sumo_cmd="--no-step-log",  # hide the "Step #..." progress lines
            observation_class=(
                DefaultObservationFunction if sensing_range is None
                else make_detector_observation(sensing_range)
            ),
        )
        super().__init__(env)

        self.scenario: Optional[Scenario] = None
        self.history: list[dict] = []
        self._closure_on = False
        self._teleports = 0
        self._arrived = 0

        # sumo-rl advances SUMO several seconds per agent step; count arrivals
        # and teleports after every single simulation second
        base_env = self.env.unwrapped
        original_sumo_step = base_env._sumo_step

        def counting_sumo_step():
            original_sumo_step()
            sim = base_env.sumo.simulation
            self._arrived += sim.getArrivedNumber()
            self._teleports += sim.getStartingTeleportNumber()

        base_env._sumo_step = counting_sumo_step

    # ------------------------------------------------------------------
    @property
    def sumo(self):
        return self.env.unwrapped.sumo

    def _closed_lane_id(self) -> str:
        return f"{self.scenario.closed_approach}_t_{self.scenario.closed_lane}"

    def _update_disruption(self, t: float) -> None:
        sc = self.scenario
        if sc.disruption != "lane_closure":
            return
        should_be_closed = sc.disrupt_time <= t < sc.disrupt_end
        if should_be_closed and not self._closure_on:
            self.sumo.lane.setDisallowed(self._closed_lane_id(), ["passenger"])
            self._closure_on = True
        elif not should_be_closed and self._closure_on:
            self.sumo.lane.setDisallowed(self._closed_lane_id(), [])
            self._closure_on = False

    def _record(self, t: float, info: dict) -> None:
        sim = self.sumo.simulation
        self.history.append(
            {
                "t": t,
                "stopped": info.get("system_total_stopped", 0),          # queue proxy
                "waiting_time": info.get("system_total_waiting_time", 0.0),
                "mean_waiting_time": info.get("system_mean_waiting_time", 0.0),
                "mean_speed": info.get("system_mean_speed", 0.0),
                "pending": len(sim.getPendingVehicles()),  # cars that could not enter yet
                "arrived": self._arrived,                   # cumulative throughput
                "teleports": self._teleports,               # cumulative; should stay ~0
                "phase": int(self.sumo.trafficlight.getPhase(self.env.unwrapped.ts_ids[0])),
            }
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        if "scenario" in options:
            self.scenario = options["scenario"]
        elif self.fixed_scenario is not None:
            self.scenario = self.fixed_scenario
        else:
            self.scenario = sample_scenario(
                self.rng, split=self.split, horizon=self.horizon, vc_range=self.vc_range
            )

        write_route_file(self.scenario, self.route_file)
        obs, info = self.env.reset(seed=self.scenario.seed)

        self.history = []
        self._closure_on = False
        self._teleports = 0
        self._arrived = 0
        info = dict(info)
        info["scenario"] = self.scenario.to_dict()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        t = self.sumo.simulation.getTime()
        self._update_disruption(t)
        self._record(t, info)

        info = dict(info)
        info["sim_time"] = t
        info["disrupted"] = self._disruption_active(t)
        if terminated or truncated:
            info["scenario"] = self.scenario.to_dict()
            info["history"] = list(self.history)
        return obs, reward, terminated, truncated, info

    def _disruption_active(self, t: float) -> bool:
        sc = self.scenario
        if sc.disruption == "none":
            return False
        end = sc.disrupt_end if sc.disrupt_end is not None else sc.horizon
        return sc.disrupt_time <= t < end

    def close(self):
        try:
            self.env.close()
        finally:
            shutil.rmtree(self.work_dir, ignore_errors=True)


def make_env(**kwargs) -> AdaptiveTrafficEnv:
    """Factory used by training/evaluation scripts. Same kwargs as AdaptiveTrafficEnv."""
    return AdaptiveTrafficEnv(**kwargs)
