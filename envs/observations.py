"""Observation functions: what the controller can sense.

full (default, sumo-rl's DefaultObservationFunction)
    density and queue of each incoming lane over its whole length (~240 m).
    Unrealistic: real intersections rarely see the full queue.

detector (sensing_range=R metres)
    mimics stop-line detectors (inductive loops / short cameras): only
    vehicles within R metres of the stop line are visible. A long queue looks
    the same as a short one once it extends past the detector, so the
    controller has to *infer* what is happening upstream from how the readings
    change over time. This is where memory should matter.

Both variants return the same vector layout, so agents are directly comparable:
    [phase one-hot (4), min-green flag (1), occupancy per lane (12), halting per lane (12)]
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction

VEHICLE_SPACE_M = 7.5  # vehicle length (5 m) + min gap (2.5 m)


def make_detector_observation(sensing_range: float):
    """Return an ObservationFunction class that only sees `sensing_range` m upstream."""

    class DetectorObservation(ObservationFunction):
        RANGE = float(sensing_range)

        def __call__(self) -> np.ndarray:
            ts = self.ts
            phase = [1.0 if ts.green_phase == i else 0.0 for i in range(ts.num_green_phases)]
            min_green = [0.0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1.0]

            cap = max(1.0, self.RANGE / VEHICLE_SPACE_M)  # vehicles that fit in the detector zone
            occupancy, halting = [], []
            for lane in ts.lanes:
                length = ts.lanes_length[lane]
                n = h = 0
                for veh in ts.sumo.lane.getLastStepVehicleIDs(lane):
                    if length - ts.sumo.vehicle.getLanePosition(veh) <= self.RANGE:
                        n += 1
                        if ts.sumo.vehicle.getSpeed(veh) < 0.1:
                            h += 1
                occupancy.append(min(1.0, n / cap))
                halting.append(min(1.0, h / cap))
            return np.array(phase + min_green + occupancy + halting, dtype=np.float32)

        def observation_space(self) -> spaces.Box:
            n = self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)
            return spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32))

    DetectorObservation.__name__ = f"DetectorObservation{int(sensing_range)}m"
    return DetectorObservation
