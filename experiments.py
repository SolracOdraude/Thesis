"""
experiments.py
==============
Reproduction of the EGGROLL paper's RL benchmark suite using QEggRoll (int8).

Environments (paper Tables 3–16):
  gymnax  : CartPole-v1, Pendulum-v1
  brax    : ant, humanoid, inverted_double_pendulum
  craftax : Craftax-Classic-Symbolic-AutoReset-v1, Craftax-Symbolic-AutoReset-v1
  jumanji : Game2048-v1, Knapsack-v1, Snake-v1
  kinetix : hard_pinball (l), h17_thrustcontrol_left (m), h1_thrust_over_ball (s)
  navix   : DoorKey-8x8-v0, Dynamic-Obstacles-Random-6x6-v0, FourRooms-8x8-v0

Install missing suites as needed:
  pip install gymnax brax craftax jumanji navix
  pip install git+https://github.com/FLAIROx/Kinetix.git  # RL kinetix (not PyPI)

Architectural differences from the paper's float EggRoll
---------------------------------------------------------
  1. Weights/activations: int8 fixed-point (FIXED_POINT=4) vs float32.
  2. Nonlinearity:  int8 symmetric clipping ±127 vs pqn = relu(layer_norm(x)).
     pqn zeroes negatives; our model preserves them — this is a genuine difference.
  3. Update rule: ±1 integer steps gated by threshold vs optax gradient step.
     As a result, paper hyperparameters learning_rate / lr_decay / optimizer do not
     apply to this implementation; see hparams.py for the mapping that was used.
  4. Continuous actions: paper uses MultivariateNormalDiag; we dequantize the int8
     output mean to float via tanh, with a fixed log_std hyperparameter (see
     IntMLPContinuous below).

Usage
-----
  python experiments.py --env CartPole-v1
  python experiments.py --env brax/ant --num_epochs 500
  python experiments.py --env jumanji/Snake-v1 --seed 1
"""

import os
# RTX 40-series (Ada Lovelace) int8 GEMM bug: Triton int8 kernels produce 2x values.
# Disabling Triton GEMM falls back to cuBLAS which handles int8 correctly.
_xla = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm" not in _xla:
    _xla += " --xla_gpu_enable_triton_gemm=false"
os.environ["XLA_FLAGS"] = _xla.strip()

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import tqdm
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

from main import (
    CommonInit, CommonParams, PARAM, MM_PARAM, EXCLUDED,
    recursive_scan_split, simple_es_tree_key,
    merge_inits, merge_frozen, call_submodule,
    Model, Parameter, MM, Linear, EGG_LN, clipped_add, IntMLP,
    QEggRoll, DTYPE, MAX, FIXED_POINT, FBIT,
)
from hparams import EGGROLL_HPARAMS
from metrics import RunLogger

# ── Optional environment suite imports ────────────────────────────────────────

try:
    import gymnax
    HAS_GYMNAX = True
except ImportError:
    HAS_GYMNAX = False

try:
    import brax.envs as brax_envs
    HAS_BRAX = True
except ImportError:
    HAS_BRAX = False

try:
    import craftax
    HAS_CRAFTAX = True
except ImportError:
    HAS_CRAFTAX = False

try:
    import jumanji
    HAS_JUMANJI = True
except ImportError:
    HAS_JUMANJI = False

try:
    import navix
    HAS_NAVIX = True
except ImportError:
    HAS_NAVIX = False

try:
    import kinetix
    HAS_KINETIX = True
except ImportError:
    HAS_KINETIX = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_DIM   = 256   # matches paper: 3 layers of 256 neurons
N_LAYER      = 3
LOG_STD_INIT = -0.5  # initial log std for continuous action policies


@dataclass
class Args:
    env:           str   = "CartPole-v1"
    seed:          int   = 0
    num_epochs:    int   = 1000
    log_every:     int   = 20
    eval_episodes: int   = 5
    save_results:  bool  = False
    results_dir:   str   = "results"
    run_tag:       str   = ""
    # Override hparams from the table (also used for ablation studies):
    pop_size:               Optional[int]   = None
    sigma_shift:            Optional[int]   = None
    rank:                   Optional[int]   = None
    n_parallel_evaluations: Optional[int]   = None
    hidden_dim:             Optional[int]   = None   # ablation: network width
    noise_size_exp:         Optional[int]   = None   # ablation: BIG_RAND_MATRIX size


# ─────────────────────────────────────────────────────────────────────────────
# Observation helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_obs(obs: jnp.ndarray, scale: int) -> jnp.ndarray:
    """Float or integer obs → int8 fixed-point."""
    return jnp.clip(
        jnp.round(obs.astype(jnp.float32) * scale).astype(jnp.int32),
        -MAX, MAX,
    ).astype(jnp.int8)


def flatten_obs(obs) -> jnp.ndarray:
    """Flatten any pytree observation (NamedTuple, dict, nested array) to 1-D float32."""
    leaves = jax.tree.leaves(obs)
    return jnp.concatenate([l.ravel().astype(jnp.float32) for l in leaves])


# ─────────────────────────────────────────────────────────────────────────────
# Continuous-action model
# ─────────────────────────────────────────────────────────────────────────────

class IntMLPContinuous(Model):
    """
    Same integer MLP as IntMLP but with a continuous action output.

    The final Linear head produces int8 logits that are dequantized to float
    and squashed through tanh to give action means in [-1, 1].  A fixed
    log_std scalar (stored as a frozen float parameter) completes the policy.

    Note: the tanh is applied *outside* the integer network — it is purely an
    output-to-environment boundary transformation, not an internal activation.
    """

    @classmethod
    def rand_init(cls, key, in_dim, out_dim, hidden_dim, n_layer, dtype, **_):
        keys = jax.random.split(key, n_layer + 2)
        blocks = {}
        for i in range(n_layer):
            blocks[f"ln{i}"]     = EGG_LN.rand_init(None, hidden_dim, dtype)
            blocks[f"linear{i}"] = Linear.rand_init(keys[i], hidden_dim, hidden_dim, dtype)
        merged = merge_inits(
            proj=Linear.rand_init(keys[-2], in_dim,    hidden_dim, dtype),
            head=Linear.rand_init(keys[-1], hidden_dim, out_dim,   dtype),
            **blocks,
        )
        return merge_frozen(merged, n_layer=n_layer)

    @classmethod
    def _forward(cls, common_params, x):
        n_layer = common_params.frozen_params["n_layer"]
        x = call_submodule(Linear, "proj", common_params, x)
        for i in range(n_layer):
            residual = x
            x = call_submodule(EGG_LN, f"ln{i}", common_params, x)
            x = call_submodule(Linear, f"linear{i}", common_params, x)
            x = clipped_add(x, residual)
        logits_int8 = call_submodule(Linear, "head", common_params, x)
        # Dequantize fixed-point int8 → float, squash to (-1, 1)
        return jnp.tanh(logits_int8.astype(jnp.float32) / (2 ** FIXED_POINT))


# ─────────────────────────────────────────────────────────────────────────────
# Action selection
# ─────────────────────────────────────────────────────────────────────────────

def select_action_discrete(logits, rng, deterministic):
    if deterministic:
        return jnp.argmax(logits)
    return jax.random.categorical(rng, logits.astype(jnp.float32))


def select_action_continuous(action_mean, log_std, rng, deterministic):
    if deterministic:
        return action_mean
    std = jnp.exp(log_std)
    return action_mean + std * jax.random.normal(rng, action_mean.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Per-suite rollout factory functions
# ─────────────────────────────────────────────────────────────────────────────

# ── gymnax (CartPole, Pendulum) ───────────────────────────────────────────────

def make_gymnax_rollout(env, env_params, cfg,
                        frozen_noiser_params, frozen_params, es_tree_key,
                        model_cls, log_std=None):
    obs_scale     = cfg["obs_scale"]
    deterministic = cfg["deterministic_policy"]
    action_type   = cfg["action_type"]

    def rollout(noiser_params_, params_, iterinfo_, rng_key):
        def step_fn(carry, _):
            obs, state, done, total_return, rng = carry
            obs_int8 = encode_obs(obs, obs_scale)
            out = model_cls.forward(
                QEggRoll, frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key, iterinfo_, obs_int8)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            if action_type == "discrete":
                action = select_action_discrete(out, act_rng, deterministic)
            else:
                action = select_action_continuous(out, log_std, act_rng, deterministic)
            next_obs, next_state, reward, next_done, _ = env.step(
                step_rng, state, action, env_params)
            total_return = total_return + reward * (1.0 - done.astype(jnp.float32))
            return (next_obs, next_state, next_done, total_return, rng), None

        rng_reset, rng_ep = jax.random.split(rng_key)
        obs, state = env.reset(rng_reset, env_params)
        (_, _, _, total_return, _), _ = jax.lax.scan(
            step_fn,
            (obs, state, jnp.array(False), jnp.array(0.0), rng_ep),
            None,
            length=env_params.max_steps_in_episode,
        )
        return total_return

    return rollout


# ── brax ──────────────────────────────────────────────────────────────────────

def make_brax_rollout(env, cfg,
                      frozen_noiser_params, frozen_params, es_tree_key,
                      log_std=None):
    obs_scale     = cfg["obs_scale"]
    deterministic = cfg["deterministic_policy"]
    max_steps     = cfg["max_steps"]

    def rollout(noiser_params_, params_, iterinfo_, rng_key):
        def step_fn(carry, _):
            state, total_return, done = carry
            obs_int8 = encode_obs(state.obs, obs_scale)
            action_mean = IntMLPContinuous.forward(
                QEggRoll, frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key, iterinfo_, obs_int8)
            # brax actions are typically in [-1, 1]; tanh already applied in forward
            action = action_mean   # deterministic by default for brax
            next_state = env.step(state, action)
            total_return = total_return + next_state.reward * (1.0 - done.astype(jnp.float32))
            return (next_state, total_return, next_state.done), None

        init_state = env.reset(rng_key)
        (_, total_return, _), _ = jax.lax.scan(
            step_fn,
            (init_state, jnp.array(0.0), jnp.array(0.0)),
            None,
            length=max_steps,
        )
        return total_return

    return rollout


# ── craftax (gymnax-compatible API) ──────────────────────────────────────────

def make_craftax_rollout(env, env_params, cfg,
                         frozen_noiser_params, frozen_params, es_tree_key):
    obs_scale = cfg["obs_scale"]
    max_steps = cfg["max_steps"]
    deterministic = cfg["deterministic_policy"]

    def rollout(noiser_params_, params_, iterinfo_, rng_key):
        def step_fn(carry, _):
            obs, state, done, total_return, rng = carry
            flat = flatten_obs(obs)
            obs_int8 = encode_obs(flat, obs_scale)
            logits = IntMLP.forward(
                QEggRoll, frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key, iterinfo_, obs_int8)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action = select_action_discrete(logits, act_rng, deterministic)
            next_obs, next_state, reward, next_done, _ = env.step(
                step_rng, state, action, env_params)
            total_return = total_return + reward * (1.0 - done.astype(jnp.float32))
            return (next_obs, next_state, next_done, total_return, rng), None

        rng_reset, rng_ep = jax.random.split(rng_key)
        obs, state = env.reset(rng_reset, env_params)
        (_, _, _, total_return, _), _ = jax.lax.scan(
            step_fn,
            (obs, state, jnp.array(False), jnp.array(0.0), rng_ep),
            None, length=max_steps,
        )
        return total_return

    return rollout


# ── jumanji ───────────────────────────────────────────────────────────────────

def make_jumanji_rollout(env, cfg,
                         frozen_noiser_params, frozen_params, es_tree_key):
    obs_scale = cfg["obs_scale"]
    max_steps = cfg["max_steps"]
    deterministic = cfg["deterministic_policy"]

    def rollout(noiser_params_, params_, iterinfo_, rng_key):
        def step_fn(carry, _):
            # In jumanji, observations are in the timestep, not the state.
            state, timestep, total_return, done, rng = carry
            flat = flatten_obs(timestep.observation)
            obs_int8 = encode_obs(flat, obs_scale)
            logits = IntMLP.forward(
                QEggRoll, frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key, iterinfo_, obs_int8)
            rng, act_rng = jax.random.split(rng)
            action = select_action_discrete(logits, act_rng, deterministic)
            next_state, next_timestep = env.step(state, action)
            reward = next_timestep.reward
            total_return = total_return + reward * (1.0 - done.astype(jnp.float32))
            return (next_state, next_timestep, total_return, timestep.last(), rng), None

        rng_reset, rng_ep = jax.random.split(rng_key)
        init_state, init_timestep = env.reset(rng_reset)
        (_, _, total_return, _, _), _ = jax.lax.scan(
            step_fn,
            (init_state, init_timestep, jnp.array(0.0), jnp.array(False), rng_ep),
            None, length=max_steps,
        )
        return total_return

    return rollout


# ── navix ─────────────────────────────────────────────────────────────────────

def make_navix_rollout(env, cfg,
                       frozen_noiser_params, frozen_params, es_tree_key):
    """
    Navix uses a timestep-based API: env.reset(key) → timestep,
    env.step(timestep, action) → timestep.
    The observation is at timestep.observation (typically an integer grid array).
    """
    obs_scale = cfg["obs_scale"]
    max_steps = cfg["max_steps"]
    deterministic = cfg["deterministic_policy"]

    def rollout(noiser_params_, params_, iterinfo_, rng_key):
        def step_fn(carry, _):
            timestep, total_return, rng = carry
            flat = flatten_obs(timestep.observation)
            obs_int8 = encode_obs(flat, obs_scale)
            logits = IntMLP.forward(
                QEggRoll, frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key, iterinfo_, obs_int8)
            rng, act_rng = jax.random.split(rng)
            action = select_action_discrete(logits, act_rng, deterministic)
            next_timestep = env.step(timestep, action)
            done = timestep.is_done()
            total_return = total_return + next_timestep.reward * (1.0 - done.astype(jnp.float32))
            return (next_timestep, total_return, rng), None

        init_timestep = env.reset(rng_key)
        (_, total_return, _), _ = jax.lax.scan(
            step_fn,
            (init_timestep, jnp.array(0.0), rng_key),
            None, length=max_steps,
        )
        return total_return

    return rollout


# ── kinetix ───────────────────────────────────────────────────────────────────

def make_kinetix_rollout(env, env_params, cfg,
                         frozen_noiser_params, frozen_params, es_tree_key,
                         log_std=None):
    """
    Kinetix has a gymnax-compatible API with continuous actions.
    obs is a flat float32 array; actions are continuous in [-1, 1].
    """
    obs_scale     = cfg["obs_scale"]
    max_steps     = cfg["max_steps"]
    deterministic = cfg["deterministic_policy"]

    def rollout(noiser_params_, params_, iterinfo_, rng_key):
        def step_fn(carry, _):
            obs, state, done, total_return, rng = carry
            obs_int8 = encode_obs(obs, obs_scale)
            action_mean = IntMLPContinuous.forward(
                QEggRoll, frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key, iterinfo_, obs_int8)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action = select_action_continuous(action_mean, log_std, act_rng, deterministic)
            next_obs, next_state, reward, next_done, _ = env.step(
                step_rng, state, action, env_params)
            total_return = total_return + reward * (1.0 - done.astype(jnp.float32))
            return (next_obs, next_state, next_done, total_return, rng), None

        rng_reset, rng_ep = jax.random.split(rng_key)
        obs, state = env.reset(rng_reset, env_params)
        (_, _, _, total_return, _), _ = jax.lax.scan(
            step_fn,
            (obs, state, jnp.array(False), jnp.array(0.0), rng_ep),
            None, length=max_steps,
        )
        return total_return

    return rollout


# ─────────────────────────────────────────────────────────────────────────────
# Environment setup: returns (rollout_fn, obs_dim, act_dim, eval_fn)
# ─────────────────────────────────────────────────────────────────────────────

def setup_environment(cfg, key, frozen_noiser_params, frozen_params, es_tree_key,
                      log_std=None):
    """
    Instantiate the environment and return a per-suite rollout factory.
    Returns: (perturbed_rollout_fn, eval_rollout_fn, obs_dim, act_dim)

    perturbed_rollout_fn(noiser_params, params, iterinfo, rng_key) → return (float32)
    eval_rollout_fn     (noiser_params, params, rng_key)           → return (float32)
    """
    suite = cfg["suite"]

    if suite == "gymnax":
        assert HAS_GYMNAX, "pip install gymnax"
        env, env_params = gymnax.make(cfg["gymnax_name"])
        env_params = env_params.replace(max_steps_in_episode=cfg["max_steps"])

        dummy_key = jax.random.key(0)
        obs, _ = env.reset(dummy_key, env_params)
        obs_dim = int(np.prod(obs.shape))
        act_dim = int(env.action_space(env_params).n) if cfg["action_type"] == "discrete" \
                  else int(np.prod(env.action_space(env_params).shape))

        model_cls = IntMLP if cfg["action_type"] == "discrete" else IntMLPContinuous
        fn = make_gymnax_rollout(env, env_params, cfg,
                                 frozen_noiser_params, frozen_params, es_tree_key,
                                 model_cls, log_std)
        eval_fn = lambda np_, p, rng: fn(np_, p, None, rng)

    elif suite == "brax":
        assert HAS_BRAX, "pip install brax"
        env = brax_envs.get_environment(cfg["brax_name"])

        dummy_state = env.reset(jax.random.key(0))
        obs_dim = int(dummy_state.obs.shape[-1])
        act_dim = int(env.action_size)

        fn = make_brax_rollout(env, cfg,
                               frozen_noiser_params, frozen_params, es_tree_key,
                               log_std)
        eval_fn = lambda np_, p, rng: fn(np_, p, None, rng)

    elif suite == "craftax":
        assert HAS_CRAFTAX, "pip install craftax"
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        try:
            env, env_params = gymnax.make(cfg["craftax_name"])
        except ValueError:
            env = CraftaxSymbolicEnv()
            env_params = env.default_params

        dummy_key = jax.random.key(0)
        obs, _ = env.reset(dummy_key, env_params)
        obs_dim = int(np.prod(flatten_obs(obs).shape))
        act_dim = int(env.action_space(env_params).n)

        fn = make_craftax_rollout(env, env_params, cfg,
                                  frozen_noiser_params, frozen_params, es_tree_key)
        eval_fn = lambda np_, p, rng: fn(np_, p, None, rng)

    elif suite == "jumanji":
        assert HAS_JUMANJI, "pip install jumanji"
        env = jumanji.make(cfg["jumanji_name"])

        dummy_key = jax.random.key(0)
        _, init_ts = env.reset(dummy_key)
        obs_dim = int(np.prod(flatten_obs(init_ts.observation).shape))
        act_spec = env.action_spec
        act_dim = int(act_spec.num_values) if hasattr(act_spec, "num_values") \
                  else int(np.prod(act_spec.shape))

        fn = make_jumanji_rollout(env, cfg,
                                  frozen_noiser_params, frozen_params, es_tree_key)
        eval_fn = lambda np_, p, rng: fn(np_, p, None, rng)

    elif suite == "navix":
        assert HAS_NAVIX, "pip install navix"
        env = navix.make(cfg["navix_name"])

        dummy_key = jax.random.key(0)
        init_ts = env.reset(dummy_key)
        obs_dim = int(np.prod(flatten_obs(init_ts.observation).shape))
        act_dim = int(env.action_space.n)

        fn = make_navix_rollout(env, cfg,
                                frozen_noiser_params, frozen_params, es_tree_key)
        eval_fn = lambda np_, p, rng: fn(np_, p, None, rng)

    elif suite == "kinetix":
        assert HAS_KINETIX, "pip install git+https://github.com/FLAIROx/Kinetix.git"
        from kinetix.environment import make_kinetix_env, ActionType, ObservationType
        from kinetix.util import load_evaluation_levels

        level_path = cfg["kinetix_name"].removeprefix("kinetix/")
        levels, static_env_params = load_evaluation_levels([level_path])

        def _kinetix_reset_fn(rng):
            return jax.tree.map(lambda x: x[0], levels)

        env = make_kinetix_env(
            ActionType.CONTINUOUS,
            ObservationType.SYMBOLIC_FLAT,
            reset_fn=_kinetix_reset_fn,
            static_env_params=static_env_params,
        )
        env_params = env.default_params

        dummy_key = jax.random.key(0)
        obs, _ = env.reset(dummy_key, env_params)
        obs_dim = int(np.prod(obs.shape))
        act_dim = int(np.prod(env.action_space(env_params).shape))

        fn = make_kinetix_rollout(env, env_params, cfg,
                                  frozen_noiser_params, frozen_params, es_tree_key,
                                  log_std)
        eval_fn = lambda np_, p, rng: fn(np_, p, None, rng)

    else:
        raise ValueError(f"Unknown suite: {suite!r}")

    return fn, eval_fn, obs_dim, act_dim


# ─────────────────────────────────────────────────────────────────────────────
# Fitness conversion (wraps QEggRoll, handles n_parallel_evaluations > 1)
# ─────────────────────────────────────────────────────────────────────────────

def convert_fitnesses(raw_returns, frozen_noiser_params, noiser_params):
    """raw_returns: (N,) float32 → (N//2,) int8 antithetic fitnesses."""
    scaled = (raw_returns * 10).astype(jnp.int32)
    return QEggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args: Args):
    if args.env not in EGGROLL_HPARAMS:
        raise ValueError(
            f"Unknown env {args.env!r}. Available: {list(EGGROLL_HPARAMS)}")

    cfg = dict(EGGROLL_HPARAMS[args.env])
    if args.pop_size               is not None: cfg["pop_size"]               = args.pop_size
    if args.sigma_shift            is not None: cfg["sigma_shift"]            = args.sigma_shift
    if args.rank                   is not None: cfg["rank"]                   = args.rank
    if args.n_parallel_evaluations is not None: cfg["n_parallel_evaluations"] = args.n_parallel_evaluations

    suite       = cfg["suite"]
    action_type = cfg["action_type"]
    N           = cfg["pop_size"]
    K           = cfg["n_parallel_evaluations"]
    hidden_dim  = args.hidden_dim if args.hidden_dim is not None else HIDDEN_DIM
    noise_size  = 2 ** (args.noise_size_exp if args.noise_size_exp is not None else 28)
    assert N % 2 == 0, "pop_size must be even (antithetic pairs)"

    print(f"\nEnvironment : {args.env}  ({suite})")
    print(f"Action type : {action_type}")
    print(f"pop_size={N}  rank={cfg['rank']}  sigma_shift={cfg['sigma_shift']}"
          f"  hidden_dim={hidden_dim}  n_parallel_evaluations={K}")

    key = jax.random.key(args.seed)
    model_key, es_key, rollout_key = jax.random.split(key, 3)

    # ── Stub init to discover obs_dim / act_dim ───────────────────────────────
    # We init a placeholder noiser and model with dummy dims, then rebuild properly
    # after learning the environment's dimensions.

    # First: dummy env probe to get dims (done inside setup_environment)
    # We need frozen_noiser_params for setup_environment but we don't have dims yet.
    # Solution: probe with a two-step init.

    # Step 1 – probe dims with a throwaway setup
    _probe_frozen, _probe_np = QEggRoll.init_noiser(
        {"w": jnp.zeros((4, 4), dtype=jnp.int8)},
        sigma_shift=cfg["sigma_shift"],
        update_threshold=2,
        dtype="int8", noise_seed=0, noise_reuse=1, rank=1,
        use_clt=False, fast_fitness=cfg["fast_fitness"],
        update_batch_size=2, noise_size=2**16,
    )
    _dummy_fp   = {"n_layer": N_LAYER}
    _dummy_p    = {"w": jnp.zeros((4, 4), dtype=jnp.int8)}
    _dummy_scan = {"w": ()}
    _dummy_key  = simple_es_tree_key(_dummy_p, key, _dummy_scan)

    _, _, obs_dim, act_dim = setup_environment(
        cfg, key, _probe_frozen, _dummy_fp, _dummy_key,
        log_std=jnp.full((1,), LOG_STD_INIT))

    print(f"obs_dim={obs_dim}  act_dim={act_dim}  hidden_dim={hidden_dim}  n_layer={N_LAYER}")

    # ── Model init ────────────────────────────────────────────────────────────
    model_cls = IntMLP if action_type == "discrete" else IntMLPContinuous
    frozen_params, params, scan_map, es_map = model_cls.rand_init(
        model_key,
        in_dim=obs_dim, out_dim=act_dim,
        hidden_dim=hidden_dim, n_layer=N_LAYER,
        dtype="int8",
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    num_params = jax.tree.reduce(
        lambda a, b: a + b, jax.tree.map(lambda x: x.size, params))
    print(f"Parameters  : {num_params:,}")

    # ── log_std for continuous actions ────────────────────────────────────────
    log_std = jnp.full((act_dim,), LOG_STD_INIT) if action_type == "continuous" else None

    # ── Noiser init ───────────────────────────────────────────────────────────
    update_batch_size = max(2, N // 16)
    frozen_noiser_params, noiser_params = QEggRoll.init_noiser(
        params,
        sigma_shift=cfg["sigma_shift"],
        update_threshold=2,
        dtype="int8",
        noise_seed=args.seed,
        noise_reuse=1,
        rank=cfg["rank"],
        use_clt=False,
        fast_fitness=cfg["fast_fitness"],
        update_batch_size=update_batch_size,
        noise_size=noise_size,
    )

    # ── Build rollout fns ─────────────────────────────────────────────────────
    perturbed_rollout, eval_rollout_fn, _, _ = setup_environment(
        cfg, key, frozen_noiser_params, frozen_params, es_tree_key, log_std)

    # Training rollout: vmap over N population members
    # If K > 1: inner vmap over K episode seeds per member
    if K == 1:
        v_train = jax.jit(jax.vmap(
            perturbed_rollout, in_axes=(None, None, 0, 0)))
        def get_raw_returns(np_, p, ii, rngs):
            return v_train(np_, p, ii, rngs)   # (N,)
    else:
        # rngs shape: (N, K)
        # inner vmap over K seeds with fixed iterinfo for that member
        v_train_inner = jax.vmap(
            lambda np_, p, ii, rng: perturbed_rollout(np_, p, ii, rng),
            in_axes=(None, None, None, 0))
        v_train = jax.jit(jax.vmap(
            lambda np_, p, ii, rngs: v_train_inner(np_, p, ii, rngs).mean(),
            in_axes=(None, None, 0, 0)))
        def get_raw_returns(np_, p, ii, rngs):
            return v_train(np_, p, ii, rngs)   # (N,)

    jit_update = jax.jit(
        lambda np_, p, f, ii:
            QEggRoll.do_updates(frozen_noiser_params, np_, p,
                                es_tree_key, f, ii, es_map))

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("Compiling...", end=" ", flush=True)
    dummy_ii  = (jnp.zeros(N, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32))
    dummy_rng = jax.random.split(rollout_key, N) if K == 1 \
                else jax.random.split(rollout_key, N * K).reshape(N, K)
    _ = jax.block_until_ready(get_raw_returns(noiser_params, params, dummy_ii, dummy_rng))
    dummy_fits = jnp.zeros(N // 2, dtype=DTYPE)
    _ = jax.block_until_ready(jit_update(noiser_params, params, dummy_fits, dummy_ii))
    print("done.\n")

    # ── Metrics setup ─────────────────────────────────────────────────────────
    run_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    run_cfg.update(
        hidden_dim=hidden_dim,
        noise_size_exp=args.noise_size_exp or 28,
        # Training schedule — stored so a JSON alone fully reproduces the run
        seed=args.seed,
        num_epochs=args.num_epochs,
        log_every=args.log_every,
        eval_episodes=args.eval_episodes,
    )
    logger = RunLogger("QEggRoll", args.env, args.seed, run_cfg)
    # env_steps per epoch: N population members × K episodes × max_steps
    # (conservative count — jax.lax.scan runs max_steps regardless of early done)
    steps_per_epoch = N * K * cfg["max_steps"]

    # ── Epoch loop ────────────────────────────────────────────────────────────
    rng       = rollout_key
    best_eval = -float("inf")
    cum_steps = 0
    logger.reset_clock()

    for epoch in tqdm.trange(args.num_epochs):
        rng, epoch_rng = jax.random.split(rng)
        if K == 1:
            thread_rngs = jax.random.split(epoch_rng, N)
        else:
            thread_rngs = jax.random.split(epoch_rng, N * K).reshape(N, K)

        iterinfo = (
            jnp.full(N, epoch, dtype=jnp.int32),
            jnp.arange(N, dtype=jnp.int32),
        )

        raw_returns = get_raw_returns(noiser_params, params, iterinfo, thread_rngs)
        fitnesses   = convert_fitnesses(raw_returns, frozen_noiser_params, noiser_params)
        noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)
        cum_steps += steps_per_epoch

        if epoch % args.log_every == 0 or epoch == args.num_epochs - 1:
            rng, eval_key = jax.random.split(rng)
            eval_rngs  = jax.random.split(eval_key, args.eval_episodes)
            eval_rets  = np.array([
                float(eval_rollout_fn(noiser_params, params, r)) for r in eval_rngs])
            mean_eval  = float(eval_rets.mean())
            best_eval  = max(best_eval, mean_eval)
            mean_train = float(raw_returns.mean())
            mean_fit   = float(jnp.mean(jnp.abs(fitnesses.astype(jnp.float32))))

            logger.record(cum_steps, mean_eval, mean_train,
                          fitness_signal=round(mean_fit, 4),
                          eval_std=round(float(eval_rets.std()), 2))
            logger.print_row(epoch, cum_steps, mean_eval, mean_train,
                             fit=f"{mean_fit:.3f}")

            # Save after every checkpoint so an OOM kill doesn't lose all data.
            if args.save_results:
                path = RunLogger.default_path("QEggRoll", args.env, args.seed, args.results_dir, args.run_tag)
                logger.save(path)

    print(f"\nDone. Best eval return: {best_eval:.2f}")
    return params, noiser_params


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_experiment(args)
