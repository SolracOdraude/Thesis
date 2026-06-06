"""
hparams.py
==========
Hyperparameters from the EGGROLL paper (Tables 3–16) for reproducing their RL experiments
using the QEggRoll (int8) implementation.

Mapping notes
-------------
The paper's hyperparameters are for the float EggRoll noiser, which uses an optax optimizer
(SGD/Adam) with a learning rate to apply gradient estimates. QEggRoll instead uses ±1 integer
steps gated by a threshold — the following fields from the paper therefore DO NOT apply and are
stored only for reference:
    learning_rate, lr_decay, optimizer

Fields that DO apply directly:
    pop_size            population size (same concept)
    rank                LoRA rank (same concept)
    n_parallel_evaluations  episodes averaged per population member
    deterministic_policy    use argmax vs. sample at eval

Fields that require approximate mapping:
    sigma → sigma_shift
        In float EggRoll, sigma controls the std of the Gaussian perturbation.
        In QEggRoll, the perturbation magnitude is ≈ 2^{-(FIXED_POINT + sigma_shift)}.
        Approximate mapping (FIXED_POINT=4):
            sigma=0.05  →  sigma_shift=4   (max perturbation ≈ 0.03 per weight)
            sigma=0.2   →  sigma_shift=2   (max perturbation ≈ 0.12 per weight)
            sigma=0.5   →  sigma_shift=1   (max perturbation ≈ 0.25 per weight)

    sigma_decay
        In float EggRoll, sigma is multiplied by sigma_decay each epoch.
        In QEggRoll, sigma_shift is an integer; continuous decay is not meaningful.
        We store sigma_decay for reference but do not apply it.

    rank_transform
        In float EggRoll, rank_transform replaces raw fitnesses with their rank ordering.
        QEggRoll's fast_fitness=True (sign of antithetic diff) is already a form of rank
        transformation. We set fast_fitness=True when rank_transform=True in the paper.

    activation (all environments use 'pqn' = relu(layer_norm(x)))
        Our analogue is EGG_LN (integer layer norm) + int8 symmetric clipping.
        These are NOT equivalent: pqn clips to [0, ∞) via relu; ours clips to [-127, 127].

obs_scale
    Not reported in the paper (their float model needs no scaling). Values here are
    chosen to map typical observation magnitudes into the int8 range [-127, 127].
"""

# sigma → sigma_shift lookup
SIGMA_TO_SIGMA_SHIFT = {0.05: 4, 0.2: 2, 0.5: 1}

EGGROLL_HPARAMS = {
    # ── gymnax ────────────────────────────────────────────────────────────────
    "CartPole-v1": {
        "suite":                  "gymnax",
        "gymnax_name":            "CartPole-v1",
        "action_type":            "discrete",
        "obs_scale":              32,
        "max_steps":              500,
        # Paper Table 3
        "pop_size":               2048,
        "rank":                   4,
        "sigma":                  0.2,
        "sigma_shift":            2,
        "sigma_decay":            0.999,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           True,     # rank_transform=False → still use sign
        # Reference only (float EggRoll)
        "_optimizer":             "sgd",
        "_learning_rate":         0.1,
        "_lr_decay":              0.9995,
    },
    "Pendulum-v1": {
        "suite":                  "gymnax",
        "gymnax_name":            "Pendulum-v1",
        "action_type":            "continuous",
        "action_dim":             1,
        "obs_scale":              16,
        "max_steps":              200,
        # Paper Table 4
        "pop_size":               4096,
        "rank":                   4,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            0.995,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           True,
        "_optimizer":             "adam",
        "_learning_rate":         0.01,
        "_lr_decay":              0.995,
    },

    # ── brax ──────────────────────────────────────────────────────────────────
    "brax/ant": {
        "suite":                  "brax",
        "brax_name":              "ant",
        "action_type":            "continuous",
        "obs_scale":              8,
        "max_steps":              1000,
        # Paper Table 5
        "pop_size":               2048,
        "rank":                   1,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            0.9995,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           True,
        "_optimizer":             "adam",
        "_learning_rate":         0.01,
        "_lr_decay":              0.9995,
    },
    "brax/humanoid": {
        "suite":                  "brax",
        "brax_name":              "humanoid",
        "action_type":            "continuous",
        "obs_scale":              4,
        "max_steps":              1000,
        # Paper Table 6
        "pop_size":               4096,
        "rank":                   1,
        "sigma":                  0.2,
        "sigma_shift":            2,
        "sigma_decay":            0.9995,
        "n_parallel_evaluations": 8,
        "deterministic_policy":   True,
        "fast_fitness":           False,    # rank_transform=True → use normalised fitness
        "_optimizer":             "adam",
        "_learning_rate":         0.1,
        "_lr_decay":              1.0,
    },
    "brax/inverted_double_pendulum": {
        "suite":                  "brax",
        "brax_name":              "inverted_double_pendulum",
        "action_type":            "continuous",
        "action_dim":             1,
        "obs_scale":              16,
        "max_steps":              1000,
        # Paper Table 7
        "pop_size":               2048,
        "rank":                   2,
        "sigma":                  0.5,
        "sigma_shift":            1,
        "sigma_decay":            0.995,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   True,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "adam",
        "_learning_rate":         0.1,
        "_lr_decay":              1.0,
    },

    # ── craftax ───────────────────────────────────────────────────────────────
    "craftax/Craftax-Classic-Symbolic-AutoReset-v1": {
        "suite":                  "craftax",
        "craftax_name":           "Craftax-Classic-Symbolic-AutoReset-v1",
        "action_type":            "discrete",
        "obs_scale":              1,
        "max_steps":              2500,
        # Paper Table 8
        "pop_size":               2048,
        "rank":                   1,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            1.0,
        "n_parallel_evaluations": 4,
        "deterministic_policy":   False,
        "fast_fitness":           True,
        "_optimizer":             "sgd",
        "_learning_rate":         0.01,
        "_lr_decay":              0.995,
    },
    "craftax/Craftax-Symbolic-AutoReset-v1": {
        "suite":                  "craftax",
        "craftax_name":           "Craftax-Symbolic-AutoReset-v1",
        "action_type":            "discrete",
        "obs_scale":              1,
        "max_steps":              2500,
        # Paper Table 9
        "pop_size":               512,
        "rank":                   4,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            0.999,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "sgd",
        "_learning_rate":         0.01,
        "_lr_decay":              0.999,
    },

    # ── jumanji ───────────────────────────────────────────────────────────────
    "jumanji/Game2048-v1": {
        "suite":                  "jumanji",
        "jumanji_name":           "Game2048-v1",
        "action_type":            "discrete",
        "obs_scale":              1,
        "max_steps":              1000,
        # Paper Table 10
        "pop_size":               1024,
        "rank":                   1,
        "sigma":                  0.5,
        "sigma_shift":            1,
        "sigma_decay":            0.9995,
        "n_parallel_evaluations": 4,
        "deterministic_policy":   False,
        "fast_fitness":           True,
        "_optimizer":             "adamw",
        "_learning_rate":         0.1,
        "_lr_decay":              1.0,
    },
    "jumanji/Knapsack-v1": {
        "suite":                  "jumanji",
        "jumanji_name":           "Knapsack-v1",
        "action_type":            "discrete",
        "obs_scale":              16,
        "max_steps":              500,
        # Paper Table 11
        "pop_size":               1024,
        "rank":                   4,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            1.0,
        "n_parallel_evaluations": 4,
        "deterministic_policy":   False,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "sgd",
        "_learning_rate":         0.1,
        "_lr_decay":              0.999,
    },
    "jumanji/Snake-v1": {
        "suite":                  "jumanji",
        "jumanji_name":           "Snake-v1",
        "action_type":            "discrete",
        "obs_scale":              1,
        "max_steps":              500,
        # Paper Table 12
        "pop_size":               4096,
        "rank":                   1,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            0.9995,
        "n_parallel_evaluations": 8,
        "deterministic_policy":   False,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "adam",
        "_learning_rate":         0.001,
        "_lr_decay":              0.9995,
    },

    # ── kinetix ───────────────────────────────────────────────────────────────
    "kinetix/l/hard_pinball": {
        "suite":                  "kinetix",
        "kinetix_name":           "kinetix/l/hard_pinball",
        "action_type":            "continuous",
        "obs_scale":              16,
        "max_steps":              1000,
        # Paper Table 13
        "pop_size":               2048,
        "rank":                   4,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            0.999,
        "n_parallel_evaluations": 8,
        "deterministic_policy":   True,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "sgd",
        "_learning_rate":         0.01,
        "_lr_decay":              0.995,
    },
    "kinetix/m/h17_thrustcontrol_left": {
        "suite":                  "kinetix",
        "kinetix_name":           "kinetix/m/h17_thrustcontrol_left",
        "action_type":            "continuous",
        "obs_scale":              16,
        "max_steps":              1000,
        # Paper Table 14
        "pop_size":               512,
        "rank":                   4,
        "sigma":                  0.5,
        "sigma_shift":            1,
        "sigma_decay":            1.0,
        "n_parallel_evaluations": 4,
        "deterministic_policy":   False,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "sgd",
        "_learning_rate":         0.1,
        "_lr_decay":              0.9995,
    },
    "kinetix/s/h1_thrust_over_ball": {
        "suite":                  "kinetix",
        "kinetix_name":           "kinetix/s/h1_thrust_over_ball",
        "action_type":            "continuous",
        "obs_scale":              16,
        "max_steps":              1000,
        # Paper Table 15
        "pop_size":               512,
        "rank":                   1,
        "sigma":                  0.5,
        "sigma_shift":            1,
        "sigma_decay":            0.9995,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "adamw",
        "_learning_rate":         0.1,
        "_lr_decay":              0.995,
    },

    # ── navix ─────────────────────────────────────────────────────────────────
    "navix/Navix-DoorKey-8x8-v0": {
        "suite":                  "navix",
        "navix_name":             "Navix-DoorKey-8x8-v0",
        "action_type":            "discrete",
        "obs_scale":              1,
        "max_steps":              500,
        # Paper Table 16
        "pop_size":               1024,
        "rank":                   1,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            1.0,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           True,
        "_optimizer":             "adamw",
        "_learning_rate":         0.01,
        "_lr_decay":              0.9995,
    },
    # Paper Table 17
    "navix/Navix-Dynamic-Obstacles-Random-6x6-v0": {
        "suite":                  "navix",
        "navix_name":             "Navix-Dynamic-Obstacles-6x6-Random-v0",
        "action_type":            "discrete",
        "obs_scale":              1,
        "max_steps":              500,
        "pop_size":               512,
        "rank":                   2,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            1.0,
        "n_parallel_evaluations": 4,
        "deterministic_policy":   False,
        "fast_fitness":           True,     # rank_transform=False
        "_optimizer":             "adam",
        "_learning_rate":         0.01,
        "_lr_decay":              0.999,
    },
    # Paper Table 18 (env name in paper: Navix-FourRooms-v0, no "8x8")
    "navix/Navix-FourRooms-v0": {
        "suite":                  "navix",
        "navix_name":             "Navix-FourRooms-v0",
        "action_type":            "discrete",
        "obs_scale":              1,
        "max_steps":              500,
        "pop_size":               2048,
        "rank":                   4,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "sigma_decay":            0.9995,
        "n_parallel_evaluations": 4,
        "deterministic_policy":   False,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "sgd",
        "_learning_rate":         0.01,
        "_lr_decay":              0.999,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PPO hyperparameters (paper Tables 19–20, for reference / baseline comparison)
# These are for the Rejax PPO implementation credited in the paper.
# Source: https://github.com/keraJLi/rejax
# ─────────────────────────────────────────────────────────────────────────────

PPO_HPARAMS = {
    "CartPole-v1": {
        "activation":          "pqn",
        "clip_eps":            0.2,
        "ent_coef":            0.0001,
        "gae_lambda":          0.9,
        "gamma":               0.995,
        "learning_rate":       0.0003,
        "max_grad_norm":       0.5,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            256,
        "num_epochs":          4,
        "num_minibatches":     32,
        "num_steps":           128,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.5,
    },
    "Pendulum-v1": {
        "activation":          "pqn",
        "clip_eps":            0.1,
        "ent_coef":            0.001,
        "gae_lambda":          0.95,
        "gamma":               0.999,
        "learning_rate":       0.0003,
        "max_grad_norm":       1,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            256,
        "num_epochs":          16,
        "num_minibatches":     16,
        "num_steps":           256,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             1,
    },
    "brax/ant": {
        "activation":          "pqn",
        "clip_eps":            0.2,
        "ent_coef":            0,
        "gae_lambda":          0.95,
        "gamma":               0.995,
        "learning_rate":       0.0003,
        "max_grad_norm":       0.5,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            64,
        "num_epochs":          8,
        "num_minibatches":     32,
        "num_steps":           128,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             1,
    },
    "brax/humanoid": {
        "activation":          "pqn",
        "clip_eps":            0.3,
        "ent_coef":            0.0001,
        "gae_lambda":          0.9,
        "gamma":               0.95,
        "learning_rate":       0.0001,
        "max_grad_norm":       2,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            256,
        "num_epochs":          4,
        "num_minibatches":     64,
        "num_steps":           64,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.75,
    },
    "brax/inverted_double_pendulum": {
        "activation":          "pqn",
        "clip_eps":            0.1,
        "ent_coef":            0.0001,
        "gae_lambda":          0.98,
        "gamma":               0.99,
        "learning_rate":       0.001,
        "max_grad_norm":       2,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            64,
        "num_epochs":          4,
        "num_minibatches":     64,
        "num_steps":           128,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             1,
    },
    "craftax/Craftax-Classic-Symbolic-AutoReset-v1": {
        "activation":          "pqn",
        "clip_eps":            0.2,
        "ent_coef":            0.0001,
        "gae_lambda":          0.98,
        "gamma":               0.95,
        "learning_rate":       0.001,
        "max_grad_norm":       2,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            128,
        "num_epochs":          4,
        "num_minibatches":     32,
        "num_steps":           128,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.5,
    },
    "craftax/Craftax-Symbolic-AutoReset-v1": {
        "activation":          "pqn",
        "clip_eps":            0.2,
        "ent_coef":            0,
        "gae_lambda":          0.9,
        "gamma":               0.95,
        "learning_rate":       0.0003,
        "max_grad_norm":       2,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            256,
        "num_epochs":          4,
        "num_minibatches":     32,
        "num_steps":           64,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.75,
    },
    "jumanji/Game2048-v1": {
        "activation":          "pqn",
        "clip_eps":            0.3,
        "ent_coef":            0.001,
        "gae_lambda":          0.9,
        "gamma":               0.99,
        "learning_rate":       0.0003,
        "max_grad_norm":       2,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            64,
        "num_epochs":          8,
        "num_minibatches":     16,
        "num_steps":           64,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.75,
    },
    # Tables for remaining environments (Knapsack, Snake, Kinetix, Navix) not
    # provided in the paper excerpt. Add here when available.
}

