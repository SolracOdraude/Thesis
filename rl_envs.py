"""
rl_envs.py
==========
Pure-JAX RL environment wrappers for the integer EGGROLL training loop.

Environments
------------
  CartPole-v1      in_dim= 4  out_dim=2  discrete actions
  Acrobot-v1       in_dim= 6  out_dim=3  discrete actions
  MountainCar-v0   in_dim= 2  out_dim=3  discrete actions  (hard, sparse reward)

Install
-------
  pip install stable-gymnax      # maintained gymnax fork, works with latest jax

Usage
-----
  python rl_envs.py                         # CartPole with default args
  python rl_envs.py --env acrobot           # Acrobot
  python rl_envs.py --env mountain_car      # MountainCar
  python rl_envs.py --env cartpole --num_epochs 500 --population_size 1024
"""

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import tqdm
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from functools import partial

import gymnax

# ── Import everything from the integer MLP training file ─────────────────────
# We reuse all model primitives and the QEggRoll noiser exactly as-is.
from main import (
    # Data structures
    CommonInit, CommonParams, PARAM, MM_PARAM, EXCLUDED,
    # Tree utilities
    recursive_scan_split, simple_es_tree_key,
    merge_inits, merge_frozen, call_submodule,
    # Model classes
    Model, Parameter, MM, Linear, EGG_LN, clipped_add, IntMLP,
    # Noiser
    QEggRoll,
    # Constants
    DTYPE, MAX, FIXED_POINT, FBIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RLArgs:
    # ── Environment ──────────────────────────────────────────────────────────
    env:             str   = "cartpole"   # cartpole | acrobot | mountain_car
    max_steps:       int   = 500          # episode length cap (env-specific)

    # ── Architecture ─────────────────────────────────────────────────────────
    hidden_dim:      int   = 64
    n_layer:         int   = 3

    # ── Training ─────────────────────────────────────────────────────────────
    seed:            int   = 0
    population_size: int   = 1024
    num_epochs:      int   = 500
    sigma_shift:     int   = 2
    update_threshold: int  = 2
    rank:            int   = 1
    use_clt:         bool  = False
    fast_fitness:    bool  = False
    noise_reuse:     int   = 1
    noise_seed:      int   = 42
    noise_size_exp:  int   = 22    # 2**22 = 4MB, safe for CPU dev

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every:       int   = 20
    eval_episodes:   int   = 10    # episodes averaged for eval return


# ─────────────────────────────────────────────────────────────────────────────
# Environment registry
# ─────────────────────────────────────────────────────────────────────────────

ENV_REGISTRY = {
    "cartpole": {
        "gymnax_name":       "CartPole-v1",
        "in_dim":            4,
        "out_dim":           2,
        "max_steps":         500,
        "obs_scale":         32,
        "success_threshold": 475,   # within 5 steps of perfect
        "description": "Balance a pole on a cart. Classic control, in_dim=4.",
    },
    "acrobot": {
        "gymnax_name":       "Acrobot-v1",
        "in_dim":            6,
        "out_dim":           3,
        "max_steps":         500,
        "obs_scale":         32,
        "success_threshold": -150,  # solved in under 150 steps
        "description": "Swing a two-link pendulum. in_dim=6, 3 discrete actions.",
    },
    "mountain_car": {
        "gymnax_name":       "MountainCar-v0",
        "in_dim":            2,
        "out_dim":           3,
        "max_steps":         200,
        "obs_scale":         64,
        "success_threshold": -110,  # reached the goal at all
        "description": "Drive a car up a hill. Sparse reward, hard exploration.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-point observation encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_obs(obs: jnp.ndarray, scale: int) -> jnp.ndarray:
    """
    Convert a float32 observation to int8 fixed-point.
    obs  : Array(in_dim,) float32  — raw environment observation
    scale: int                     — multiply before rounding (obs_scale in registry)

    The scale is chosen per-environment to use most of the [-127, 127] range
    without saturating on typical observation values.
    """
    return jnp.clip(
        jnp.round(obs * scale).astype(jnp.int32),
        -MAX, MAX
    ).astype(jnp.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Single-episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def make_rollout_fn(env, env_params, obs_scale,
                    frozen_noiser_params,
                    frozen_params, es_tree_key):
    """
    Returns a jit-friendly function that rolls out ONE episode for ONE
    population member.

    Signature:
        rollout(noiser_params, params, iterinfo, rng_key) -> total_return (float32)

    The episode uses jax.lax.scan over timesteps for full JIT compatibility.
    """

    def rollout(noiser_params_, params_, _iterinfo, rng_key):
        """
        noiser_params_ and params_ are passed explicitly so vmap can vary them
        and so that updated params are picked up on each call.
        """
        def step_fn(carry, _):
            obs, state, done, total_return, rng = carry

            obs_int8 = encode_obs(obs, obs_scale)

            logits = IntMLP.forward(
                QEggRoll,
                frozen_noiser_params, noiser_params_,
                frozen_params, params_, es_tree_key,
                None,          # iterinfo=None → eval mode, no perturbation
                obs_int8,
            )
            action = jnp.argmax(logits)

            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, reward, next_done, _ = env.step(
                step_rng, state, action, env_params)

            total_return = total_return + reward * (1.0 - done.astype(jnp.float32))

            return (next_obs, next_state, next_done, total_return, rng), None

        rng_reset, rng_ep = jax.random.split(rng_key)
        obs, state = env.reset(rng_reset, env_params)
        done = jnp.array(False)
        total_return = jnp.array(0.0, dtype=jnp.float32)

        (_, _, _, total_return, _), _ = jax.lax.scan(
            step_fn,
            (obs, state, done, total_return, rng_ep),
            None,
            length=env_params.max_steps_in_episode,
        )
        return total_return

    return rollout


def make_perturbed_rollout_fn(env, env_params, obs_scale,
                               frozen_noiser_params,
                               frozen_params, es_tree_key):
    """
    Returns a rollout function that DOES apply perturbations (training mode).
    noiser_params, params, and iterinfo are explicit arguments so vmap works.
    """

    def step_fn(carry, _):
        obs, state, done, total_return, rng, noiser_params_, params_, iterinfo_ = carry

        obs_int8 = encode_obs(obs, obs_scale)

        logits = IntMLP.forward(
            QEggRoll,
            frozen_noiser_params, noiser_params_,
            frozen_params, params_, es_tree_key,
            iterinfo_,
            obs_int8,
        )
        action = jnp.argmax(logits)

        rng, step_rng = jax.random.split(rng)
        next_obs, next_state, reward, next_done, _ = env.step(
            step_rng, state, action, env_params)

        total_return = total_return + reward * (1.0 - done.astype(jnp.float32))

        return (next_obs, next_state, next_done, total_return,
                rng, noiser_params_, params_, iterinfo_), None

    def perturbed_rollout(noiser_params_, params_, iterinfo_, rng_key):
        rng_reset, rng_ep = jax.random.split(rng_key)
        obs, state = env.reset(rng_reset, env_params)
        done = jnp.array(False)
        total_return = jnp.array(0.0, dtype=jnp.float32)

        (_, _, _, total_return, _, _, _, _), _ = jax.lax.scan(
            step_fn,
            (obs, state, done, total_return, rng_ep,
             noiser_params_, params_, iterinfo_),
            None,
            length=env_params.max_steps_in_episode,
        )
        return total_return

    return perturbed_rollout


# ─────────────────────────────────────────────────────────────────────────────
# Fitness conversion
# ─────────────────────────────────────────────────────────────────────────────

def rl_convert_fitnesses(raw_returns: jnp.ndarray,
                          frozen_noiser_params: dict,
                          noiser_params: dict) -> jnp.ndarray:
    """
    Wraps QEggRoll.convert_fitnesses for RL returns.

    raw_returns: Array(N,) float32  — total episodic return per population member
    Returns    : Array(N//2,) int8  — antithetic ±1 fitness per pair

    We cast returns to int32 (scaled by a constant to preserve ordering) before
    passing to convert_fitnesses. Since convert_fitnesses only uses sign(diff),
    the exact scale does not matter — only the relative ordering within each pair.
    """
    # Scale to int32 preserving order (multiply by 10 to keep fractional rewards)
    scaled = (raw_returns * 10).astype(jnp.int32)
    return QEggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, scaled)

# ─────────────────────────────────────────────────────────────────────────────
# Episode visualization
# ─────────────────────────────────────────────────────────────────────────────

def _gymnasium_episode(params, noiser_params, frozen_params, frozen_noiser_params,
                        es_tree_key, cfg, render=False):
    """Single greedy episode in gymnasium. Returns total return."""
    import gymnasium as gym
    env = gym.make(cfg["gymnax_name"], render_mode="human" if render else None)
    obs, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        obs_int8 = np.clip(
            np.round(obs * cfg["obs_scale"]), -MAX, MAX
        ).astype(np.int8)
        logits = IntMLP.forward(
            QEggRoll, frozen_noiser_params, noiser_params,
            frozen_params, params, es_tree_key, None,
            jnp.array(obs_int8),
        )
        obs, r, terminated, truncated, _ = env.step(int(jnp.argmax(logits)))
        total += r
        done = terminated or truncated
    env.close()
    return total


def evaluate_gymnasium(params, noiser_params, frozen_params, frozen_noiser_params,
                        es_tree_key, cfg, n_episodes):
    """Run n greedy episodes in gymnasium. Returns array of returns."""
    return np.array([
        _gymnasium_episode(params, noiser_params, frozen_params,
                           frozen_noiser_params, es_tree_key, cfg)
        for _ in range(n_episodes)
    ])


def render_episode(params, noiser_params, frozen_params, frozen_noiser_params,
                   es_tree_key, cfg):
    """Render one greedy episode in a gymnasium window."""
    ret = _gymnasium_episode(params, noiser_params, frozen_params,
                              frozen_noiser_params, es_tree_key, cfg, render=True)
    print(f"Episode return: {ret:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_rl(args: RLArgs):

    # ── Environment setup ─────────────────────────────────────────────────────
    if args.env not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown env '{args.env}'. Choose from: {list(ENV_REGISTRY.keys())}")

    cfg = ENV_REGISTRY[args.env]
    print(f"\nEnvironment : {cfg['gymnax_name']}")
    print(f"Description : {cfg['description']}")
    print(f"in_dim={cfg['in_dim']}  out_dim={cfg['out_dim']}  "
          f"max_steps={cfg['max_steps']}")

    env, env_params = gymnax.make(cfg["gymnax_name"])
    # Override max_steps if user specified a different value
    env_params = env_params.replace(max_steps_in_episode=cfg["max_steps"])
    obs_scale  = cfg["obs_scale"]

    in_dim  = cfg["in_dim"]
    out_dim = cfg["out_dim"]
    N       = args.population_size
    assert N % 2 == 0, "population_size must be even (antithetic pairs)"

    key = jax.random.key(args.seed)

    # ── Model init ────────────────────────────────────────────────────────────
    model_key, es_key, rollout_key = jax.random.split(key, 3)

    frozen_params, params, scan_map, es_map = IntMLP.rand_init(
        model_key,
        in_dim     = in_dim,
        out_dim    = out_dim,
        hidden_dim = args.hidden_dim,
        n_layer    = args.n_layer,
        dtype      = "int8",
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    num_params = jax.tree.reduce(
        lambda a, b: a + b, jax.tree.map(lambda x: x.size, params))
    print(f"Parameters  : {num_params:,}\n")

    # ── Noiser init ───────────────────────────────────────────────────────────
    update_batch_size = max(2, N // 16)
    frozen_noiser_params, noiser_params = QEggRoll.init_noiser(
        params,
        sigma_shift       = args.sigma_shift,
        update_threshold  = args.update_threshold,
        dtype             = "int8",
        noise_seed        = args.noise_seed,
        noise_reuse       = args.noise_reuse,
        rank              = args.rank,
        use_clt           = args.use_clt,
        fast_fitness      = args.fast_fitness,
        update_batch_size = update_batch_size,
        noise_size        = 2 ** args.noise_size_exp,
    )

    # ── Build rollout functions ───────────────────────────────────────────────
    perturbed_rollout = make_perturbed_rollout_fn(
        env, env_params, obs_scale,
        frozen_noiser_params, frozen_params, es_tree_key)

    # vmap training rollout over (iterinfo_per_thread, rng_per_thread)
    # Each population member gets its own iterinfo scalar and rng key.
    # noiser_params and params are shared (not perturbed at the vmap level —
    # the perturbation is applied inside _forward via iterinfo).
    v_train_rollout = jax.jit(jax.vmap(
        perturbed_rollout,
        in_axes=(None, None, 0, 0)   # noiser, params fixed; iterinfo, rng vary
    ))

    jit_update = jax.jit(
        lambda np_, p, f, ii:
            QEggRoll.do_updates(
                frozen_noiser_params, np_, p,
                es_tree_key, f, ii, es_map)
    )

    # ── Warmup / compile ─────────────────────────────────────────────────────
    print("Compiling... ", end="", flush=True)
    dummy_ii  = (jnp.zeros(N, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32))
    dummy_rng = jax.random.split(rollout_key, N)
    _ = jax.block_until_ready(
        v_train_rollout(noiser_params, params, dummy_ii, dummy_rng))
    dummy_fits = jnp.zeros(N // 2, dtype=DTYPE)
    _ = jax.block_until_ready(jit_update(noiser_params, params, dummy_fits, dummy_ii))
    print("done.\n")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    rng = rollout_key
    best_eval = -float("inf")

    for epoch in tqdm.trange(args.num_epochs):

        # One PRNG key per population member
        rng, epoch_rng = jax.random.split(rng)
        thread_rngs = jax.random.split(epoch_rng, N)

        iterinfo = (
            jnp.full(N, epoch, dtype=jnp.int32),
            jnp.arange(N, dtype=jnp.int32),
        )

        # ── Perturbed rollouts (training) ─────────────────────────────────
        raw_returns = v_train_rollout(
            noiser_params, params, iterinfo, thread_rngs)   # (N,) float32

        # ── Fitness conversion ────────────────────────────────────────────
        fitnesses = rl_convert_fitnesses(
            raw_returns, frozen_noiser_params, noiser_params) # (N//2,) int8

        # ── Parameter update ──────────────────────────────────────────────
        noiser_params, params = jit_update(
            noiser_params, params, fitnesses, iterinfo)

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % args.log_every == 0 or epoch == args.num_epochs - 1:
            eval_returns = evaluate_gymnasium(
                params, noiser_params, frozen_params, frozen_noiser_params,
                es_tree_key, cfg, n_episodes=args.eval_episodes)

            mean_eval    = float(eval_returns.mean())
            best_ep      = float(eval_returns.max())
            std_eval     = float(eval_returns.std())
            mean_train   = float(raw_returns.mean())
            mean_fitness = float(jnp.mean(jnp.abs(fitnesses.astype(jnp.float32))))
            success_rate = float(np.mean(eval_returns >= cfg["success_threshold"]))

            if mean_eval > best_eval:
                render_episode(params, noiser_params, frozen_params,
                               frozen_noiser_params, es_tree_key, cfg)

            best_eval = max(best_eval, mean_eval)

            tqdm.tqdm.write(
                f"  epoch {epoch:4d} | "
                f"train {mean_train:7.1f} | "
                f"eval {mean_eval:6.1f} ± {std_eval:.0f} | "
                f"best_ep {best_ep:6.1f} | "
                f"success {success_rate:.0%} | "
                f"|fitness| {mean_fitness:.3f} | "
                f"best_ever {best_eval:6.1f}"
            )


    print(f"\nDone. Best eval return: {best_eval:.1f}")
    return params, noiser_params


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(RLArgs)
    run_rl(args)