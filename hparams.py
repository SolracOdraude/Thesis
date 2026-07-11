"""
hparams.py
==========
Hyperparameters for the three thesis target environments:
  CartPole-v1, Pendulum-v1, kinetix/s/h1_thrust_over_ball

QEggRoll parameters match the EggRoll paper (Tables 17, 18, 29).
PPO parameters from the paper's Tables 33/34 (Rejax implementation).

Mapping notes
-------------
Fields that DO apply directly to QEggRoll:
    pop_size, rank, n_parallel_evaluations, deterministic_policy

Fields that require approximate mapping:
    sigma -> sigma_shift
        sigma=0.05  ->  sigma_shift=4   (perturbation ~= 0.03 per weight)
        sigma=0.2   ->  sigma_shift=2   (perturbation ~= 0.12 per weight)
        sigma=0.5   ->  sigma_shift=1   (perturbation ~= 0.25 per weight)

Fields not applicable to QEggRoll (stored for reference only):
    learning_rate, lr_decay, optimizer, sigma_decay, rank_transform
"""

EGGROLL_HPARAMS = {
    "CartPole-v1": {
        "suite":                  "gymnax",
        "gymnax_name":            "CartPole-v1",
        "action_type":            "discrete",
        "obs_scale":              32,
        "max_steps":              500,
        # Paper Table 17
        "pop_size":               2048,
        "rank":                   4,
        "sigma":                  0.2,
        "sigma_shift":            2,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           True,
        "_optimizer":             "sgd",
        "_learning_rate":         0.1,
        "_lr_decay":              0.9995,
        "_sigma_decay":           0.999,
    },
    "Pendulum-v1": {
        "suite":                  "gymnax",
        "gymnax_name":            "Pendulum-v1",
        "action_type":            "continuous",
        "action_dim":             1,
        "obs_scale":              16,
        "max_steps":              200,
        "action_scale":           2.0,   # gymnax Pendulum expects torque in [-2, 2]
        # Paper Table 18
        "pop_size":               4096,
        "rank":                   4,
        "sigma":                  0.05,
        "sigma_shift":            4,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           True,
        "_optimizer":             "adam",
        "_learning_rate":         0.01,
        "_lr_decay":              0.995,
        "_sigma_decay":           0.995,
    },
    "kinetix/s/h1_thrust_over_ball": {
        "suite":                  "kinetix",
        "kinetix_name":           "kinetix/s/h1_thrust_over_ball",
        "action_type":            "continuous",
        "action_scale":           1.0,
        "obs_scale":              16,
        "max_steps":              1000,
        # Paper Table 29
        "pop_size":               512,
        "rank":                   1,
        "sigma":                  0.5,
        "sigma_shift":            1,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           False,    # rank_transform=True
        "_optimizer":             "adamw",
        "_learning_rate":         0.1,
        "_lr_decay":              0.995,
        "_sigma_decay":           0.9995,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PPO hyperparameters (paper Tables 33/34, Rejax implementation)
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
    "kinetix/s/h1_thrust_over_ball": {
        "activation":          "pqn",
        "clip_eps":            0.2,
        "ent_coef":            0.0001,
        "gae_lambda":          0.95,
        "gamma":               0.999,
        "learning_rate":       0.0001,
        "max_grad_norm":       0.5,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            64,
        "num_epochs":          16,
        "num_minibatches":     16,
        "num_steps":           64,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.5,
    },
}
