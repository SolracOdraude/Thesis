"""
ppo_baseline.py
===============
Self-contained JAX/gymnax PPO matching the architecture used in the EGGROLL paper:
  - 3-layer MLP, 256 hidden units, pqn = relu(layer_norm(x)) activation
  - Generalised Advantage Estimation (GAE)
  - Clipped surrogate objective with value-function and entropy terms
  - Running observation normalisation
  - Supports discrete and continuous (Gaussian) action spaces

The implementation is intentionally written in the same functional style as
main.py (plain JAX pytrees + optax) so it integrates naturally with the rest
of the codebase.  No Flax or Haiku dependency is required.

Usage
-----
  python ppo_baseline.py --env CartPole-v1
  python ppo_baseline.py --env Pendulum-v1 --num_updates 800
  python ppo_baseline.py --env CartPole-v1 --save_results

Hyperparameters are loaded from hparams.PPO_HPARAMS when the environment is
listed there; otherwise sensible defaults are used.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import gymnax
import tyro
import tqdm
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, NamedTuple

from hparams import PPO_HPARAMS, EGGROLL_HPARAMS
from metrics import RunLogger

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 256
N_LAYERS   = 3


@dataclass
class Args:
    env:          str   = "CartPole-v1"
    seed:         int   = 0
    num_updates:  int   = 1000      # number of PPO update iterations
    log_every:    int   = 20
    eval_episodes: int  = 10
    save_results: bool  = False
    results_dir:  str   = "results"
    # PPO overrides (loaded from PPO_HPARAMS by default)
    num_envs:          Optional[int]   = None
    num_steps:         Optional[int]   = None
    num_epochs:        Optional[int]   = None
    num_minibatches:   Optional[int]   = None
    learning_rate:     Optional[float] = None
    gamma:             Optional[float] = None
    gae_lambda:        Optional[float] = None
    clip_eps:          Optional[float] = None
    vf_coef:           Optional[float] = None
    ent_coef:          Optional[float] = None
    max_grad_norm:     Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Activation
# ─────────────────────────────────────────────────────────────────────────────

def layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def pqn(x):
    """pqn = relu(layer_norm(x))  — the activation used throughout the EGGROLL paper."""
    return jax.nn.relu(layer_norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Network (functional, plain JAX pytrees)
# ─────────────────────────────────────────────────────────────────────────────

def _fan_in_init(key, in_dim, out_dim, scale=1.0):
    return jax.random.normal(key, (in_dim, out_dim)) * jnp.sqrt(scale / in_dim)


def init_params(key, obs_dim: int, act_dim: int, continuous: bool):
    keys = jax.random.split(key, N_LAYERS + 3)
    dims = [obs_dim] + [HIDDEN_DIM] * N_LAYERS

    shared = [
        (_fan_in_init(keys[i], dims[i], dims[i+1], scale=2.0),
         jnp.zeros(dims[i+1]))
        for i in range(N_LAYERS)
    ]
    actor_W = _fan_in_init(keys[-3], HIDDEN_DIM, act_dim, scale=0.01)
    actor_b = jnp.zeros(act_dim)
    critic_W = _fan_in_init(keys[-2], HIDDEN_DIM, 1, scale=1.0)
    critic_b = jnp.zeros(1)

    params = {
        "shared": shared,
        "actor":  (actor_W, actor_b),
        "critic": (critic_W, critic_b),
    }
    if continuous:
        params["log_std"] = jnp.full((act_dim,), -0.5)
    return params


def _trunk(params, x):
    for W, b in params["shared"]:
        x = pqn(x @ W + b)
    return x


def actor_logits(params, obs):
    """Returns logits (discrete) or action mean (continuous)."""
    x = _trunk(params, obs)
    W, b = params["actor"]
    return x @ W + b


def critic_value(params, obs):
    x = _trunk(params, obs)
    W, b = params["critic"]
    return (x @ W + b).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Observation normalisation
# ─────────────────────────────────────────────────────────────────────────────

class ObsNorm(NamedTuple):
    mean:  jnp.ndarray
    var:   jnp.ndarray
    count: int


def init_obs_norm(obs_dim):
    return ObsNorm(jnp.zeros(obs_dim), jnp.ones(obs_dim), 0)


def update_obs_norm(norm: ObsNorm, obs_batch: jnp.ndarray) -> ObsNorm:
    flat = obs_batch.reshape(-1, obs_batch.shape[-1])
    n    = flat.shape[0]
    bm   = flat.mean(0)
    bv   = flat.var(0)
    total = norm.count + n
    delta = bm - norm.mean
    new_mean = norm.mean + delta * n / total
    new_var  = (norm.var * norm.count + bv * n +
                delta ** 2 * norm.count * n / total) / total
    return ObsNorm(new_mean, new_var, total)


def normalize_obs(norm: ObsNorm, obs: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    return (obs - norm.mean) / jnp.sqrt(norm.var + eps)


# ─────────────────────────────────────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """
    rewards, values, dones : (T, N)
    last_value             : (N,)
    Returns advantages and returns, both (T, N).
    """
    def scan_fn(carry, inputs):
        gae, next_val = carry
        reward, val, done = inputs
        delta = reward + gamma * next_val * (1.0 - done) - val
        gae   = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, val), gae

    _, adv_reversed = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(last_value), last_value),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = adv_reversed[::-1]
    returns    = advantages + values
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO loss
# ─────────────────────────────────────────────────────────────────────────────

def _discrete_log_prob_entropy(params, obs, actions):
    logits    = jax.vmap(partial(actor_logits, params))(obs)
    log_probs = jax.nn.log_softmax(logits)
    sel_lp    = log_probs[jnp.arange(len(actions)), actions]
    probs     = jax.nn.softmax(logits)
    entropy   = -jnp.sum(probs * log_probs, axis=-1).mean()
    return sel_lp, entropy


def _continuous_log_prob_entropy(params, obs, actions):
    means   = jax.vmap(partial(actor_logits, params))(obs)
    log_std = params["log_std"]
    std     = jnp.exp(log_std)
    log_p   = -0.5 * (((actions - means) / std) ** 2 + 2 * log_std
                      + jnp.log(2 * jnp.pi)).sum(-1)
    entropy = (0.5 * (1 + jnp.log(2 * jnp.pi)) + log_std).sum()
    return log_p, entropy


def ppo_loss(params, batch, clip_eps, vf_coef, ent_coef, continuous):
    obs, actions, old_log_probs, advantages, returns, old_values = batch

    # Normalise advantages within the minibatch
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if continuous:
        new_log_probs, entropy = _continuous_log_prob_entropy(params, obs, actions)
    else:
        new_log_probs, entropy = _discrete_log_prob_entropy(params, obs, actions)

    # Clipped actor loss
    ratio  = jnp.exp(new_log_probs - old_log_probs)
    pg     = -jnp.minimum(ratio * adv,
                           jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()

    # Clipped critic loss
    new_vals    = jax.vmap(partial(critic_value, params))(obs)
    v_clipped   = old_values + jnp.clip(new_vals - old_values, -clip_eps, clip_eps)
    vf          = jnp.maximum((new_vals - returns) ** 2,
                               (v_clipped - returns) ** 2).mean()

    loss = pg + vf_coef * vf - ent_coef * entropy
    return loss, (pg, vf, entropy)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

def make_rollout_fn(env, env_params, num_steps, num_envs, continuous):
    v_reset = jax.vmap(env.reset, in_axes=(0, None))
    v_step  = jax.vmap(env.step,  in_axes=(0, 0, 0, None))

    def rollout(params, obs_norm, carry, rng):
        """
        carry : (obs, state, done)  — gym carry across epochs
        Returns updated carry and stacked transitions.
        """
        obs0, state0, done0 = carry

        def step_fn(carry, _):
            obs, state, done, rng = carry
            norm_obs = normalize_obs(obs_norm, obs)

            logits = jax.vmap(partial(actor_logits, params))(norm_obs)
            vals   = jax.vmap(partial(critic_value, params))(norm_obs)

            rng, a_rng, s_rng = jax.random.split(rng, 3)
            a_rngs = jax.random.split(a_rng, num_envs)

            if continuous:
                std     = jnp.exp(params["log_std"])
                actions = logits + std * jax.random.normal(a_rng, logits.shape)
                std_bc  = jnp.broadcast_to(std, logits.shape)
                log_ps  = -0.5 * (((actions - logits) / std_bc) ** 2
                                  + 2 * jnp.log(std_bc)
                                  + jnp.log(2 * jnp.pi)).sum(-1)
            else:
                actions = jax.vmap(
                    lambda l, r: jax.random.categorical(r, l)
                )(logits, a_rngs)
                lp_all = jax.nn.log_softmax(logits)
                log_ps = lp_all[jnp.arange(num_envs), actions]

            s_rngs = jax.random.split(s_rng, num_envs)
            next_obs, next_state, reward, next_done, _ = v_step(
                s_rngs, state, actions, env_params)

            transition = (obs, actions, log_ps, reward, vals, done)
            return (next_obs, next_state, next_done, rng), transition

        (obs_T, state_T, done_T, rng), transitions = jax.lax.scan(
            step_fn, (obs0, state0, done0, rng), None, length=num_steps)

        # Bootstrap value for the last observation
        last_val = jax.vmap(partial(critic_value, params))(
            normalize_obs(obs_norm, obs_T))
        last_val = jnp.where(done_T, jnp.zeros_like(last_val), last_val)

        return (obs_T, state_T, done_T), transitions, last_val

    return rollout


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_ppo(args: Args):
    # ── Load hyperparameters ──────────────────────────────────────────────────
    hp = dict(PPO_HPARAMS.get(args.env, {}))
    cfg = {
        "num_envs":        args.num_envs        or hp.get("num_envs",        64),
        "num_steps":       args.num_steps        or hp.get("num_steps",       128),
        "num_epochs":      args.num_epochs       or hp.get("num_epochs",      4),
        "num_minibatches": args.num_minibatches  or hp.get("num_minibatches", 32),
        "learning_rate":   args.learning_rate    or hp.get("learning_rate",   3e-4),
        "gamma":           args.gamma            or hp.get("gamma",           0.99),
        "gae_lambda":      args.gae_lambda       or hp.get("gae_lambda",      0.95),
        "clip_eps":        args.clip_eps         or hp.get("clip_eps",        0.2),
        "vf_coef":         args.vf_coef          or hp.get("vf_coef",         0.5),
        "ent_coef":        args.ent_coef         or hp.get("ent_coef",        0.01),
        "max_grad_norm":   args.max_grad_norm    or hp.get("max_grad_norm",   0.5),
    }

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = EGGROLL_HPARAMS.get(args.env, {})
    env, env_params = gymnax.make(args.env)
    if env_cfg.get("max_steps"):
        env_params = env_params.replace(max_steps_in_episode=env_cfg["max_steps"])
    continuous = env_cfg.get("action_type", "discrete") == "continuous"

    key = jax.random.key(args.seed)
    reset_keys = jax.random.split(key, cfg["num_envs"])
    obs0, state0 = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
    obs_dim = int(np.prod(obs0.shape[1:]))
    act_dim = (int(np.prod(env.action_space(env_params).shape))
               if continuous
               else int(env.action_space(env_params).n))

    print(f"\nPPO | {args.env}  obs_dim={obs_dim}  act_dim={act_dim}"
          f"  continuous={continuous}")
    print(f"  num_envs={cfg['num_envs']}  num_steps={cfg['num_steps']}"
          f"  num_minibatches={cfg['num_minibatches']}  num_epochs={cfg['num_epochs']}")

    # ── Init ──────────────────────────────────────────────────────────────────
    key, p_key = jax.random.split(key)
    params   = init_params(p_key, obs_dim, act_dim, continuous)
    obs_norm = init_obs_norm(obs_dim)

    optimizer  = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["learning_rate"]),
    )
    opt_state = optimizer.init(params)

    rollout_fn = make_rollout_fn(
        env, env_params, cfg["num_steps"], cfg["num_envs"], continuous)

    # ── JIT-compiled update step ───────────────────────────────────────────────
    loss_fn = partial(ppo_loss,
                      clip_eps=cfg["clip_eps"], vf_coef=cfg["vf_coef"],
                      ent_coef=cfg["ent_coef"], continuous=continuous)

    @jax.jit
    def update(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_episode(params, obs_norm, rng_key):
        def step(carry, _):
            obs, state, done, total, rng = carry
            norm_obs = normalize_obs(obs_norm, obs)
            logits = actor_logits(params, norm_obs)
            action = jnp.argmax(logits) if not continuous else logits
            rng, s_rng = jax.random.split(rng)
            next_obs, next_state, reward, next_done, _ = env.step(
                s_rng, state, action, env_params)
            total = total + reward * (1.0 - done.astype(jnp.float32))
            return (next_obs, next_state, next_done, total, rng), None
        rng_r, rng_e = jax.random.split(rng_key)
        obs_e, state_e = env.reset(rng_r, env_params)
        (_, _, _, total, _), _ = jax.lax.scan(
            step, (obs_e, state_e, jnp.array(False), jnp.array(0.0), rng_e),
            None, length=env_params.max_steps_in_episode)
        return total

    # ── Training loop ─────────────────────────────────────────────────────────
    logger = RunLogger("PPO", args.env, args.seed, cfg)
    logger.reset_clock()

    B        = cfg["num_envs"] * cfg["num_steps"]   # batch size
    M        = cfg["num_minibatches"]
    mb_size  = B // M
    carry    = (obs0, state0, jnp.zeros(cfg["num_envs"], dtype=bool))
    cum_steps = 0

    for update_i in tqdm.trange(args.num_updates):
        key, r_key, e_key = jax.random.split(key, 3)

        # Collect rollout
        carry, (obs_t, act_t, lp_t, rew_t, val_t, done_t), last_val = \
            rollout_fn(params, obs_norm, carry, r_key)

        # Update obs normalisation
        obs_norm = update_obs_norm(obs_norm, obs_t)

        # GAE
        adv_t, ret_t = compute_gae(
            rew_t, val_t, done_t, last_val,
            cfg["gamma"], cfg["gae_lambda"])

        # Flatten (T*N,)
        def flat(x): return x.reshape(B, *x.shape[2:])
        batch_flat = tuple(map(flat, (obs_t, act_t, lp_t, adv_t, ret_t, val_t)))

        # PPO epochs with random minibatches
        for _ in range(cfg["num_epochs"]):
            key, shuf_key = jax.random.split(key)
            perm = jax.random.permutation(shuf_key, B)
            for mb_i in range(M):
                idx = perm[mb_i * mb_size : (mb_i + 1) * mb_size]
                mb  = tuple(x[idx] for x in batch_flat)
                params, opt_state, _ = update(params, opt_state, mb)

        cum_steps += B

        # Logging
        if update_i % args.log_every == 0 or update_i == args.num_updates - 1:
            eval_rngs = jax.random.split(e_key, args.eval_episodes)
            eval_rets = np.array([
                float(eval_episode(params, obs_norm, r)) for r in eval_rngs])
            mean_eval  = eval_rets.mean()
            mean_train = float(rew_t.sum(0).mean())

            logger.record(cum_steps, mean_eval, mean_train)
            logger.print_row(update_i, cum_steps, mean_eval, mean_train,
                             std=f"{eval_rets.std():.1f}")

    if args.save_results:
        path = RunLogger.default_path("PPO", args.env, args.seed, args.results_dir)
        logger.save(path)

    print(f"\nDone. Best eval: {max(e['eval_return'] for e in logger.log):.2f}")
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ppo(tyro.cli(Args))
