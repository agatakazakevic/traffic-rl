"""Episode metrics computed from AdaptiveTrafficEnv's `info["history"]`.

Recovery time (proposed definition -- agree on it as a team before the main
experiments):
  baseline  = mean queue (stopped vehicles) in the `baseline_window` seconds
              before the disruption starts
  recovered = first time after the disruption *starts* at which the queue,
              smoothed over `smooth` seconds, is back within
              (1 + tol) * baseline + slack and stays there for `hold` seconds
  recovery_time = recovered - disrupt_time   (None if it never recovers)
For surges and lane closures the demand/capacity returns to normal at
`disrupt_end`, so their recovery includes clearing the extra queue.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


def _series(history: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([h["t"] for h in history], dtype=float)
    v = np.array([h[key] for h in history], dtype=float)
    return t, v


def recovery_time(
    history: list[dict],
    disrupt_time: float,
    key: str = "stopped",
    baseline_window: float = 600,
    smooth: float = 120,
    tol: float = 0.2,
    slack: float = 2.0,
    hold: float = 300,
) -> Optional[float]:
    t, v = _series(history, key)
    if len(t) < 2:
        return None
    dt = float(np.median(np.diff(t)))
    pre = v[(t >= disrupt_time - baseline_window) & (t < disrupt_time)]
    if len(pre) == 0:
        return None
    threshold = (1 + tol) * pre.mean() + slack

    k = max(1, int(round(smooth / dt)))
    smoothed = np.convolve(v, np.ones(k) / k, mode="same")
    hold_steps = max(1, int(round(hold / dt)))

    after = np.where(t >= disrupt_time)[0]
    ok = smoothed <= threshold
    for i in after:
        j = min(len(ok), i + hold_steps)
        if j - i == hold_steps and ok[i:j].all():
            return float(t[i] - disrupt_time)
    return None


def summarize(history: list[dict], scenario: dict) -> dict:
    """Headline numbers for one episode."""
    t, stopped = _series(history, "stopped")
    _, wait = _series(history, "mean_waiting_time")
    _, pending = _series(history, "pending")
    td = scenario["disrupt_time"]
    out = {
        "disruption": scenario["disruption"],
        "vc": round(scenario["vc"], 3),
        "mean_queue": float(stopped.mean()),
        "mean_wait_s": float(wait.mean()),
        "max_pending": int(pending.max()),
        "throughput": int(history[-1]["arrived"]),
        "teleports": int(history[-1]["teleports"]),
        "queue_before": float(stopped[(t >= td - 600) & (t < td)].mean()) if (t < td).any() else float("nan"),
        "queue_after": float(stopped[t >= td].mean()) if (t >= td).any() else float("nan"),
    }
    out["recovery_s"] = (
        recovery_time(history, td) if scenario["disruption"] != "none" else None
    )
    return out
