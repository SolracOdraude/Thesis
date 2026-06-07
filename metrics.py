"""
metrics.py
==========
Shared logging infrastructure for QEggRoll, PPO, and DQN runs.

All three training scripts import RunLogger, log the same fields at each
eval checkpoint, and save a JSON file that can be loaded for plotting.

JSON structure
--------------
{
  "method": "QEggRoll",
  "env":    "CartPole-v1",
  "seed":   0,
  "config": { ...hyperparams... },
  "log": [
    {"env_steps": 0, "wall_time": 0.0, "eval_return": -inf, "train_return": null},
    ...
  ]
}
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class RunLogger:
    def __init__(self, method: str, env: str, seed: int, config: Dict[str, Any] = None):
        self.method = method
        self.env    = env
        self.seed   = seed
        self.config = config or {}
        self.log: List[Dict] = []
        self._t0 = time.time()

    def reset_clock(self):
        self._t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self._t0

    def record(self, env_steps: int, eval_return: float,
               train_return: Optional[float] = None, **extra):
        entry = {
            "env_steps":    env_steps,
            "wall_time":    self.elapsed(),
            "eval_return":  eval_return,
            "train_return": train_return,
        }
        entry.update(extra)
        self.log.append(entry)

    def print_row(self, epoch: int, env_steps: int, eval_return: float,
                  train_return: Optional[float] = None, **extra):
        parts = [
            f"epoch {epoch:5d}",
            f"steps {env_steps:9,}",
            f"t {self.elapsed():7.1f}s",
            f"eval {eval_return:9.2f}",
        ]
        if train_return is not None:
            parts.append(f"train {train_return:9.2f}")
        for k, v in extra.items():
            parts.append(f"{k} {v}")
        print("  " + " | ".join(parts))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "method": self.method,
            "env":    self.env,
            "seed":   self.seed,
            "config": self.config,
            "log":    self.log,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  -> saved to {path}")

    @staticmethod
    def load(path: str) -> Dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def default_path(method: str, env: str, seed: int, results_dir: str = "results", tag: str = "") -> str:
        safe_env = env.replace("/", "_")
        suffix = f"__{tag}" if tag else ""
        return os.path.join(results_dir, f"{method}__{safe_env}__seed{seed}{suffix}.json")
