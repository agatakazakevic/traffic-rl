# Learning to Adapt: RL Traffic Signal Control under Sudden Disruptions

The environment for our CS.30707 project: one four-way intersection in SUMO. Every episode has random traffic plus one sudden disruption.

## Setup
> **Linux users:** on Linux, `pip` installs the CUDA build of PyTorch by default (PyTorch is a dependency of `stable-baselines3`), which pulls in several GB of NVIDIA libraries. This project trains fine on CPU, so install the CPU-only build of torch **before** `requirements.txt`, as shown below. Windows and macOS wheels from PyPI are already CPU-only, so skip that line there. If you already installed the CUDA build, see [Switching to CPU-only PyTorch](#switching-to-cpu-only-pytorch).

```bash
python3 -m venv .venv && source .venv/bin/activate   # isolated env (Windows: .venv\Scripts\activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only: CPU-only torch, avoids ~5 GB of CUDA libs
pip install -r requirements.txt
export LIBSUMO_AS_TRACI=1        # ~10x faster simulation (Windows: set LIBSUMO_AS_TRACI=1)
python scripts/check_env.py      # sanity checks + results/check_env.png (~3 min)
python scripts/smoke_train.py    # confirms PPO and RecurrentPPO train
python scripts/watch.py          # watch an episode in the SUMO GUI (see --help for options)
```
`SUMO_HOME` is set automatically from the pip-installed `eclipse-sumo`.

### Switching to CPU-only PyTorch
If `pip list | grep -iE 'nvidia|cuda'` shows packages, you have the CUDA build. Replace it (inside the activated venv):
```bash
pip uninstall -y torch triton $(pip list --format=freeze | grep -iE '^(nvidia-|cuda-)' | cut -d= -f1)
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect "...+cpu False"
pip cache purge                                                                # frees the cached CUDA wheels
```

## Watching an episode (SUMO GUI)
`scripts/watch.py` plays one episode in SUMO's viewer, so you can see the cars, the signals and the disruption. It works on any OS with a display; on WSL2 it needs WSLg. Details of all options and requirements are in the file itself.

```bash
python scripts/watch.py                                     # traffic surge, fixed-time control
python scripts/watch.py --disruption lane_closure --vc 0.8  # lane closure in heavy traffic
python scripts/watch.py --policy random                     # random phases (an untrained agent)
python scripts/watch.py --model runs/ppo.zip                # a trained PPO / RecurrentPPO model
```

## Using the environment
```python
from envs.traffic_env import make_env
from envs.scenarios import Scenario
from envs.metrics import summarize

env = make_env(split="train", seed=0)   # new random scenario every reset
obs, info = env.reset()                 # info["scenario"] describes the episode

# evaluation on one fixed scenario
obs, info = env.reset(options={"scenario": Scenario(vc=0.6, disruption="lane_closure", disrupt_time=1200)})
...
# at the end of an episode (truncated=True): info["history"] has per-step metrics
print(summarize(info["history"], info["scenario"]))

fixed_time = make_env(fixed_ts=True)    # fixed-time baseline: pass action=None to step()
```
With libsumo, only **one simulation per process** is allowed. For parallel training, use `SubprocVecEnv` (each env runs in its own process).

## The intersection (`nets/single-4way`)
- 4 approaches, each 250 m long with 3 lanes: lane 0 is right + straight, lane 1 is straight, lane 2 is left.
- 4 signal phases: N-S through, N-S protected left, E-W through, E-W protected left. Each has a 3 s yellow. The default fixed-time plan uses a 92 s cycle.
- Agent: picks the next green phase every 5 s, with a minimum green of 5 s and a maximum of 60 s.
- Observation (29 dims): one-hot of the current phase, a min-green flag, and occupancy and queue for each of the 12 incoming lanes. Two sensing modes with the same layout (`envs/observations.py`):
  - `make_env(sensing_range=None)` (default) gives **full-lane** sensing: the whole lane (~240 m) is visible. This is sumo-rl's default.
  - `make_env(sensing_range=50)` gives **stop-line detectors**: only vehicles within 50 m of the stop line are visible, as with real loop detectors. Long queues look the same as short ones once they pass the detector, so the controller has to infer them over time.
- Reward: `diff-waiting-time` by default. Change it with `make_env(reward_fn=...)`.

## Pilot: does memory help? (`scripts/pilot_memory.py`)
This trains PPO (no memory) and RecurrentPPO (LSTM memory) for one sensing mode. It then evaluates both, plus fixed-time, on surge, flip and the unseen lane closure (v/c 0.7, 3 seeds each). Run one sensing mode per terminal:
```bash
python scripts/pilot_memory.py --sensing full --steps 150000   # ~1 h on a laptop core
python scripts/pilot_memory.py --sensing 50   --steps 150000
python scripts/pilot_memory.py --plot                           # -> runs/pilot/pilot.png
```
It's a single-seed pilot to check the direction, not the final experiment.

## Traffic assumptions (`envs/scenarios.py`)
- **Arrivals:** vehicles arrive at random. Each flow has a fixed per-second departure probability, which is roughly Poisson.
- **Turning split:** 15% right, 70% straight, 15% left.
- **Congestion:** set by a nominal v/c ratio, relative to one approach's fixed-time capacity of about 1300 veh/h. Levels: light 0.30, moderate 0.55, heavy 0.80, oversaturated 1.05.
- **Direction split:** `ns_share` is the share of traffic on the N-S axis.

| Disruption | What happens | Split |
|---|---|---|
| `none` | constant demand | train |
| `surge` | demand × 1.4–1.9 for 12–21 min, then back | train |
| `flip` | heavy direction switches (e.g. 75/25 N-S → 25/75) | train |
| `lane_closure` | straight lane 1 of one approach closed for 12–21 min (TraCI) | **test only** |

Random scenarios: v/c is drawn from 0.3–0.9, and the disruption starts 15–36 min into a 60 min episode.

## Metrics (`envs/metrics.py`)
The per-step history includes:
- stopped vehicles (a proxy for queue length) and waiting time
- mean speed
- pending insertions (cars that can't enter the network)
- cumulative throughput and teleports
- the active phase

`summarize()` returns:
- mean queue and mean wait
- throughput
- queue before vs. after the disruption
- **recovery time**

The definition of recovery time is a proposal: agree on it as a team before running the main experiments.

## Results of `check_env.py` (fixed-time, our machine)
- Queues stay bounded up to v/c 0.80 and keep growing at 1.05, so the capacity calibration is reasonable.
- A surge clearly shows up in the queue (about 21 → 65 stopped vehicles) and clears roughly 18 min after it starts.
- A lane closure at v/c 0.55 has only a mild effect. Use higher v/c to make it bite.
- A flip barely affects fixed-time control, which is symmetric. It's meant to hurt *adaptive* controllers that have learned to favour one direction.
- Speed with libsumo is about 270–300 agent steps/s on one core, so 1M steps take about 1 h.
