"""
ppo_baseline.py
===============
PPO baseline using Rejax — matches the paper's methodology exactly.

Supports all 16 target environments via rejax.compat wrappers:
  gymnax  : CartPole-v1, Pendulum-v1
  brax    : brax/ant, brax/humanoid, brax/inverted_double_pendulum
  craftax : craftax/Craftax-Classic-Symbolic-AutoReset-v1,
            craftax/Craftax-Symbolic-AutoReset-v1
  jumanji : jumanji/Game2048-v1, jumanji/Knapsack-v1, jumanji/Snake-v1
  kinetix : kinetix/l/hard_pinball, kinetix/m/h17_thrustcontrol_left,
            kinetix/s/h1_thrust_over_ball
  navix   : navix/Navix-DoorKey-8x8-v0,
            navix/Navix-Dynamic-Obstacles-Random-6x6-v0,
            navix/Navix-FourRooms-v0

Training calls algo.train(rng) directly; the inner jax.lax.scan is
JIT-compiled by JAX automatically. The outer train() shell runs as Python
so init_state does not execute under a JAX tracer (required for jumanji).

Usage
-----
  python ppo_baseline.py --env CartPole-v1
  python ppo_baseline.py --env brax/ant --total_timesteps 5_000_000
  python ppo_baseline.py --env jumanji/Snake-v1 --save_results
  python ppo_baseline.py --env kinetix/l/hard_pinball --seed 1

Hyperparameters are loaded from hparams.PPO_HPARAMS when available;
CLI flags override individual fields.
"""

import os
# RTX 40-series int8 GEMM bug — same fix as experiments.py.
_xla = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm" not in _xla:
    _xla += " --xla_gpu_enable_triton_gemm=false"
os.environ["XLA_FLAGS"] = _xla.strip()

import time
import numpy as np
import jax
import jax.numpy as jnp
import gymnax
import rejax
import rejax.compat
from flax import linen as nn
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork
from rejax.evaluate import evaluate as rejax_evaluate
import tyro
from dataclasses import dataclass
from typing import Optional

from hparams import PPO_HPARAMS, EGGROLL_HPARAMS
from metrics import RunLogger


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    env:             str   = "CartPole-v1"
    seed:            int   = 0
    total_timesteps: Optional[int]   = None  # default derived from EGGROLL_HPARAMS
    eval_freq:       Optional[int]   = None  # env steps between evals (default 10×batch)
    save_results:    bool  = False
    results_dir:     str   = "results"
    run_tag:         str   = ""
    # PPO hyperparameter overrides (defaults loaded from PPO_HPARAMS)
    num_envs:        Optional[int]   = None
    num_steps:       Optional[int]   = None
    num_epochs:      Optional[int]   = None
    num_minibatches: Optional[int]   = None
    learning_rate:   Optional[float] = None
    gamma:           Optional[float] = None
    gae_lambda:      Optional[float] = None
    clip_eps:        Optional[float] = None
    vf_coef:         Optional[float] = None
    ent_coef:        Optional[float] = None
    max_grad_norm:   Optional[float] = None
    normalize_obs:   Optional[bool]  = None
    normalize_rew:   Optional[bool]  = None


# ─────────────────────────────────────────────────────────────────────────────
# Environment creation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _episode_length(env_params) -> int:
    """Extract episode length from env_params regardless of field name."""
    for field in ("max_steps_in_episode", "max_timesteps", "max_steps"):
        if hasattr(env_params, field):
            v = getattr(env_params, field)
            if v is not None:
                return int(v)
    return 1000


class _DiscreteActionWrapper:
    """
    Fixes rejax 0.1.2 jumanji bug: jumanji's DiscreteArray subclasses
    BoundedArray, so rejax's convert_spec matches BoundedArray first and
    returns Box instead of Discrete. This wrapper overrides action_space.

    Uses object.__setattr__/__getattribute__ throughout to prevent infinite
    recursion when deepcopy reconstructs the object before __init__ runs.
    """
    def __init__(self, env, n_actions: int):
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, '_space',
                           gymnax.environments.spaces.Discrete(num_categories=n_actions))

    def __getattr__(self, name):
        try:
            env = object.__getattribute__(self, '_env')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(env, name)

    def action_space(self, params):
        return object.__getattribute__(self, '_space')

    def reset(self, key, params=None):
        return object.__getattribute__(self, '_env').reset(key, params)

    def step(self, key, state, action, params=None):
        return object.__getattribute__(self, '_env').step(key, state, action, params)

    def observation_space(self, params):
        return object.__getattribute__(self, '_env').observation_space(params)


# ─────────────────────────────────────────────────────────────────────────────
# Environment creation
# ─────────────────────────────────────────────────────────────────────────────

def make_env(env_name: str):
    """
    Return (env, env_params, max_steps_per_episode) for any supported suite.
    All returned envs expose the gymnax interface:
      env.reset(key, params) -> (obs, state)
      env.step(key, state, action, params) -> (obs, state, reward, done, info)
      env.action_space(params) -> gymnax Space
    """
    cfg = EGGROLL_HPARAMS.get(env_name, {})
    suite = cfg.get("suite", "gymnax")

    if suite == "gymnax":
        env, env_params = gymnax.make(env_name)
        return env, env_params, _episode_length(env_params)

    elif suite in ("brax", "navix"):
        env, env_params = rejax.compat.create(env_name)
        return env, env_params, _episode_length(env_params)

    elif suite == "jumanji":
        import jumanji as _jumanji
        from jumanji.specs import DiscreteArray as _JumanjiDiscrete
        env, env_params = rejax.compat.create(env_name)
        # Fix rejax 0.1.2: DiscreteArray subclasses BoundedArray, so convert_spec
        # matches BoundedArray first and returns Box instead of Discrete.
        raw_spec = _jumanji.make(cfg["jumanji_name"]).action_spec
        if isinstance(raw_spec, _JumanjiDiscrete):
            env = _DiscreteActionWrapper(env, int(raw_spec.num_values))
        return env, env_params, _episode_length(env_params)

    elif suite == "craftax":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        try:
            env, env_params = gymnax.make(cfg["craftax_name"])
        except ValueError:
            env = CraftaxSymbolicEnv()
            env_params = env.default_params
        # Don't replace max_steps_in_episode — craftax EnvParams uses its own
        # field names. The QEggRoll cfg["max_steps"] is a training rollout param,
        # not an env property PPO needs to override.
        return env, env_params, _episode_length(env_params)

    elif suite == "kinetix":
        from kinetix.environment import make_kinetix_env, ActionType, ObservationType
        from kinetix.util import load_evaluation_levels
        level_path = cfg["kinetix_name"].removeprefix("kinetix/")
        levels, static_env_params = load_evaluation_levels([level_path])
        def _reset_fn(rng):
            return jax.tree.map(lambda x: x[0], levels)
        env = make_kinetix_env(
            ActionType.CONTINUOUS,
            ObservationType.SYMBOLIC_FLAT,
            reset_fn=_reset_fn,
            static_env_params=static_env_params,
        )
        env_params = env.default_params
        max_steps = int(env_params.max_timesteps)
        return env, env_params, max_steps

    else:
        raise ValueError(f"Unsupported suite for PPO: {suite!r}. "
                         f"Env={env_name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Build rejax.PPO
# ─────────────────────────────────────────────────────────────────────────────

def _build_ppo(env, env_params, hp: dict, total_timesteps: int,
               eval_freq: int, max_steps: int) -> rejax.PPO:
    """
    Construct a rejax.PPO instance directly without using PPO.create(), so
    we can supply a custom eval_callback that handles env_params that lack
    max_steps_in_episode (e.g. kinetix).
    """
    action_space = env.action_space(env_params)
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

    hidden = tuple([hp.get("layer_size", 256)] * hp.get("n_layers", 3))
    agent_kwargs = {"activation": nn.tanh, "hidden_layer_sizes": hidden}

    if discrete:
        actor = DiscretePolicy(int(action_space.n), **agent_kwargs)
    else:
        actor = GaussianPolicy(
            int(np.prod(action_space.shape)),
            (action_space.low, action_space.high),
            **agent_kwargs,
        )
    critic = VNetwork(**agent_kwargs)

    def eval_callback(algo, ts, rng):
        act = algo.make_act(ts)
        return rejax_evaluate(act, rng, env, env_params, 128, max_steps)

    return rejax.PPO(
        env=env,
        env_params=env_params,
        eval_callback=eval_callback,
        eval_freq=eval_freq,
        skip_initial_evaluation=hp.get("skip_initial_eval", False),
        total_timesteps=total_timesteps,
        learning_rate=hp["learning_rate"],
        gamma=hp["gamma"],
        max_grad_norm=hp["max_grad_norm"],
        normalize_rewards=hp.get("normalize_rew", False),
        reward_normalization_discount=hp.get("rew_norm_discount", 0.99),
        normalize_observations=hp.get("normalize_obs", True),
        num_envs=hp["num_envs"],
        num_steps=hp["num_steps"],
        num_minibatches=hp["num_minibatches"],
        actor=actor,
        critic=critic,
        num_epochs=hp["num_epochs"],
        gae_lambda=hp["gae_lambda"],
        clip_eps=hp["clip_eps"],
        vf_coef=hp["vf_coef"],
        ent_coef=hp["ent_coef"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_ppo(args: Args):
    # ── Hyperparameters ────────────────────────────────────────────────────────
    defaults = {
        "num_envs": 64, "num_steps": 128, "num_epochs": 4,
        "num_minibatches": 32, "learning_rate": 3e-4, "gamma": 0.99,
        "gae_lambda": 0.95, "clip_eps": 0.2, "vf_coef": 0.5, "ent_coef": 0.01,
        "max_grad_norm": 0.5, "normalize_obs": True, "normalize_rew": False,
        "rew_norm_discount": 0.99, "skip_initial_eval": False,
        "layer_size": 256, "n_layers": 3,
    }
    hp = {**defaults, **PPO_HPARAMS.get(args.env, {})}

    # CLI overrides
    for k in ("num_envs", "num_steps", "num_epochs", "num_minibatches",
              "learning_rate", "gamma", "gae_lambda", "clip_eps",
              "vf_coef", "ent_coef", "max_grad_norm", "normalize_obs",
              "normalize_rew"):
        v = getattr(args, k, None)
        if v is not None:
            hp[k] = v

    # ── Environment ────────────────────────────────────────────────────────────
    env, env_params, max_steps = make_env(args.env)

    # ── Schedule ───────────────────────────────────────────────────────────────
    batch_size = hp["num_envs"] * hp["num_steps"]

    # total_timesteps: CLI > default derived from QEggRoll epochs in EGGROLL_HPARAMS
    eggroll_cfg = EGGROLL_HPARAMS.get(args.env, {})
    default_total = eggroll_cfg.get("num_epochs", 500) * batch_size
    total_timesteps = args.total_timesteps or default_total

    # eval_freq: CLI > 10 × batch (logs ~100 checkpoints over the run)
    eval_freq = args.eval_freq or (10 * batch_size)

    # ── Info print ─────────────────────────────────────────────────────────────
    action_space = env.action_space(env_params)
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)
    try:
        obs_space_shape = env.observation_space(env_params).shape
    except Exception:
        obs, _ = env.reset(jax.random.key(0), env_params)
        obs_space_shape = obs.shape
    obs_dim = int(np.prod(obs_space_shape))
    act_dim = int(action_space.n) if discrete else int(np.prod(action_space.shape))

    print(f"\nPPO (Rejax) | {args.env}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  discrete={discrete}")
    print(f"  num_envs={hp['num_envs']}  num_steps={hp['num_steps']}"
          f"  num_epochs={hp['num_epochs']}  num_minibatches={hp['num_minibatches']}")
    print(f"  total_timesteps={total_timesteps:,}  eval_freq={eval_freq:,}")
    print(f"  lr={hp['learning_rate']}  gamma={hp['gamma']}"
          f"  gae_lambda={hp['gae_lambda']}  clip_eps={hp['clip_eps']}")
    print("Compiling...")

    # ── Build algo and train ───────────────────────────────────────────────────
    algo = _build_ppo(env, env_params, hp, total_timesteps, eval_freq, max_steps)
    rng = jax.random.key(args.seed)

    t0 = time.time()
    ts, (eval_lengths, eval_returns) = algo.train(rng)
    jax.block_until_ready(eval_returns)
    wall_total = time.time() - t0

    # ── Convert evaluation arrays to RunLogger entries ─────────────────────────
    # eval_returns: (num_evals, 128) — 128 parallel seeds per checkpoint
    # If skip_initial_evaluation=False, index 0 is at step 0 (before any training).
    num_evals = eval_returns.shape[0]
    skip = hp.get("skip_initial_eval", False)
    if skip:
        steps_at = [eval_freq * (i + 1) for i in range(num_evals)]
    else:
        steps_at = [0] + [eval_freq * i for i in range(1, num_evals)]

    hp_saved = {
        **hp,
        "seed": args.seed,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "max_steps_per_episode": max_steps,
    }
    logger = RunLogger("PPO", args.env, args.seed, hp_saved)
    returns_np = np.array(eval_returns)
    for i in range(num_evals):
        frac = steps_at[i] / max(steps_at[-1], 1)
        logger.log.append({
            "env_steps":       int(steps_at[i]),
            "wall_time":       wall_total * frac,
            "eval_return":     float(returns_np[i].mean()),
            "eval_return_std": float(returns_np[i].std()),
            "train_return":    None,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    means = returns_np.mean(axis=1)
    print(f"\n  wall={wall_total:.1f}s  checkpoints={num_evals}"
          f"  best={means.max():.2f}  final={means[-1]:.2f}")
    stride = max(1, num_evals // 8)
    for i in range(0, num_evals, stride):
        print(f"  step {steps_at[i]:>12,}  eval={means[i]:.2f}"
              f"  std={returns_np[i].std():.2f}")

    if args.save_results:
        path = RunLogger.default_path("PPO", args.env, args.seed,
                                      args.results_dir, args.run_tag)
        logger.save(path)

    return ts


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ppo(tyro.cli(Args))
