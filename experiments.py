"""
experiments.py
==============
QEggRoll (int8 ES) training loop for the thesis target environments:
  gymnax  : CartPole-v1, Pendulum-v1, MountainCar-v0, MountainCarContinuous-v0
  kinetix : h1_thrust_over_ball (s)

Install:
  pip install gymnax
  pip install git+https://github.com/FLAIROx/Kinetix.git

Architectural differences from the paper's float EggRoll
---------------------------------------------------------
  1. Weights/activations: int8 fixed-point (FIXED_POINT=4) vs float32.
  2. Nonlinearity:  integer ReLU after each EGG_LN: clip(x, 0, 127) vs pqn = relu(layer_norm(x)).
     Both zero negatives — architecturally equivalent to pqn.
  3. Update rule: ±1 integer steps gated by threshold vs optax gradient step.
     As a result, paper hyperparameters learning_rate / lr_decay / optimizer do not
     apply to this implementation; see hparams.py for the mapping that was used.
  4. Continuous actions: paper uses MultivariateNormalDiag; we dequantize the int8
     output mean to float via tanh, with a fixed log_std hyperparameter (see
     IntMLPContinuous below).

Usage
-----
  python experiments.py --env CartPole-v1
  python experiments.py --env Pendulum-v1 --seed 1
  python experiments.py --env kinetix/s/h1_thrust_over_ball
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
    Model, Parameter, MM, Linear, EGG_LN, IntMLP,
    QEggRoll, DTYPE, MAX, FIXED_POINT, FBIT,
)
from hparams import EGGROLL_HPARAMS
from metrics import RunLogger

try:
    import gymnax
    HAS_GYMNAX = True
except ImportError:
    HAS_GYMNAX = False

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
    action_scale:           Optional[float] = None   # output scale (e.g. 2.0 for Pendulum ±2)
    max_update_step:        Optional[int]   = None   # 1=pm1, 2=capped ±2
    update_threshold:       Optional[int]   = None   # noise gate: weight only moves if |Z| > threshold×sqrt(N/2)


# ─────────────────────────────────────────────────────────────────────────────
# Observation helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_obs(obs: jnp.ndarray, scale) -> jnp.ndarray:
    """Float or integer obs → int8 fixed-point.

    scale can be a scalar int or a per-dimension array/list. Using a
    per-dimension array is essential for environments where different
    observation dimensions have very different magnitudes (e.g. MountainCar:
    position ∈ [-1.2, 0.6] but velocity ∈ [-0.07, 0.07]).
    """
    s = jnp.asarray(scale, dtype=jnp.float32)
    return jnp.clip(
        jnp.round(obs.astype(jnp.float32) * s).astype(jnp.int32),
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
            x = call_submodule(EGG_LN, f"ln{i}", common_params, x)
            x = jnp.clip(x, 0, MAX).astype(DTYPE)   # integer ReLU (pqn: relu after layer norm)
            x = call_submodule(Linear, f"linear{i}", common_params, x)
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

# ── Reward shaping potentials ─────────────────────────────────────────────────
# Potential-based shaping: shaped_reward = env_reward + Φ(next_obs) − Φ(obs).
# Preserves the optimal-policy ordering (Ng et al. 1999) while giving a dense
SHAPING_FNS = {
    # Add potential-based shaping here if needed for specific environments.
    # MountainCar uses pure env rewards (-1/step) to keep the fitness
    # landscape unmodified; the ES distinguishes goal-reaching members
    # purely through the return difference (goal: -steps, non-goal: -200).
}


# ── gymnax (CartPole, Pendulum, MountainCar, …) ───────────────────────────────

def make_gymnax_rollout(env, env_params, cfg,
                        frozen_noiser_params, frozen_params, es_tree_key,
                        model_cls, log_std=None, shaping_fn=None):
    obs_scale     = cfg["obs_scale"]
    deterministic = cfg["deterministic_policy"]
    action_type   = cfg["action_type"]
    action_scale  = cfg.get("action_scale", 1.0)

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
                action = select_action_continuous(out, log_std, act_rng, deterministic) \
                         * action_scale
            next_obs, next_state, reward, next_done, _ = env.step(
                step_rng, state, action, env_params)
            if shaping_fn is not None:
                reward = reward + shaping_fn(next_obs) - shaping_fn(obs)
            total_return = total_return + reward * (1.0 - done.astype(jnp.float32))
            # Latch done: once the episode ends it must stay ended.  Without this,
            # gymnax's fixed-length scan can reset done to False when the car
            # leaves the goal region, unmasking rewards for a second "episode".
            return (next_obs, next_state, done | next_done, total_return, rng), None

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
    action_scale  = cfg.get("action_scale", 1.0)

    def rollout(noiser_params_, params_, iterinfo_, rng_key):
        def step_fn(carry, _):
            obs, state, done, total_return, rng = carry
            obs_int8 = encode_obs(obs, obs_scale)
            action_mean = IntMLPContinuous.forward(
                QEggRoll, frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key, iterinfo_, obs_int8)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action = select_action_continuous(action_mean, log_std, act_rng, deterministic) \
                     * action_scale
            next_obs, next_state, reward, next_done, _ = env.step(
                step_rng, state, action, env_params)
            total_return = total_return + reward * (1.0 - done.astype(jnp.float32))
            return (next_obs, next_state, done | next_done, total_return, rng), None

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

        model_cls  = IntMLP if cfg["action_type"] == "discrete" else IntMLPContinuous
        shaping_fn = SHAPING_FNS.get(cfg.get("gymnax_name", ""))
        # Training rollout: with reward shaping (if defined) for a dense fitness signal.
        fn = make_gymnax_rollout(env, env_params, cfg,
                                 frozen_noiser_params, frozen_params, es_tree_key,
                                 model_cls, log_std, shaping_fn=shaping_fn)
        # Eval rollout: always uses the true env reward so logged returns are comparable.
        eval_rollout = make_gymnax_rollout(env, env_params, cfg,
                                           frozen_noiser_params, frozen_params, es_tree_key,
                                           model_cls, log_std, shaping_fn=None)
        eval_fn = lambda np_, p, rng: eval_rollout(np_, p, None, rng)

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
    if args.action_scale           is not None: cfg["action_scale"]           = args.action_scale
    if args.max_update_step        is not None: cfg["max_update_step"]        = args.max_update_step
    if args.update_threshold       is not None: cfg["update_threshold"]       = args.update_threshold

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
          f"  hidden_dim={hidden_dim}  n_parallel_evaluations={K}"
          f"  action_scale={cfg.get('action_scale', 1.0)}"
          f"  max_update_step={cfg.get('max_update_step', 1)}")

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
        update_threshold=cfg.get("update_threshold", 2),
        dtype="int8",
        noise_seed=args.seed,
        noise_reuse=1,
        rank=cfg["rank"],
        use_clt=False,
        fast_fitness=cfg["fast_fitness"],
        update_batch_size=update_batch_size,
        noise_size=noise_size,
        max_update_step=cfg.get("max_update_step", 1),
    )

    # ── Build rollout fns ─────────────────────────────────────────────────────
    perturbed_rollout, eval_rollout_fn, _, _ = setup_environment(
        cfg, key, frozen_noiser_params, frozen_params, es_tree_key, log_std)

    # Eval rollout: vmap over eval_episodes seeds, single dispatch per checkpoint
    v_eval = jax.jit(jax.vmap(eval_rollout_fn, in_axes=(None, None, 0)))

    # Training rollout: single vmap over N*K members (flat, no nesting).
    # For K > 1 each member's ii is tiled K times so all K episodes use the
    # same perturbation but independent rng seeds; returns are averaged per member.
    v_train = jax.jit(jax.vmap(perturbed_rollout, in_axes=(None, None, 0, 0)))

    if K == 1:
        def get_raw_returns(np_, p, ii, rngs):
            return v_train(np_, p, ii, rngs)                          # (N,)
    else:
        def get_raw_returns(np_, p, ii, rngs):
            # ii: tuple of (N,) arrays → repeat each K times → (N*K,)
            ii_flat  = jax.tree.map(lambda x: jnp.repeat(x, K), ii)
            rngs_flat = rngs.reshape(N * K)                           # (N*K,)
            flat = v_train(np_, p, ii_flat, rngs_flat)                # (N*K,)
            return flat.reshape(N, K).mean(axis=1)                    # (N,)

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
    dummy_eval_rngs = jax.random.split(rollout_key, args.eval_episodes)
    _ = jax.block_until_ready(v_eval(noiser_params, params, dummy_eval_rngs))
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
            eval_rets  = np.array(v_eval(noiser_params, params, eval_rngs))
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
