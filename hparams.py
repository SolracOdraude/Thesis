"""
hparams.py
==========
Hyperparameters for the five thesis target environments:
  CartPole-v1, Pendulum-v1, kinetix/s/h1_thrust_over_ball,
  MountainCar-v0, MountainCarContinuous-v0

QEggRoll parameters match the EggRoll paper (Tables 17, 18, 29) for the
original three environments; MountainCar parameters are derived from the
ablation results (pop=2048, rank=4 reaches full aggregate rank; sigma_shift
chosen per exploration difficulty).
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
        "max_steps":              256,
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
    # ── MountainCar-v0 ────────────────────────────────────────────────────────
    # Discrete, 3 actions (left / none / right), 200-step episodes.
    # Reward: -1 per step; episode ends early when position ≥ 0.45.
    # obs: [position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]].
    #
    # obs_scale design: both dimensions must be individually scaled, and the
    # velocity scale must NOT fill the full int8 range.
    #
    # Problem with scalar or equal-range scales: velocity at ×1800 fills int8
    # (±126 ticks), which dominates the projection layer and drives the ES to
    # converge on pure velocity-following before position-dependent timing can
    # be learned. Once output logits saturate at ±127, position information is
    # irrelevant regardless of how well it is encoded in the input.
    #
    # Fix — asymmetric scales that make position dominant:
    #   position: ×200 → clips at x=-0.635 (int8=-127); goal x=0.45 → +90
    #             The goal region [0.45, 0.6] maps to [+90, +120] — clearly visible.
    #   velocity: ×500 → ±35 int8.  Sign is clearly detectable (±35); magnitude
    #             contributes only 35//16=2 ticks per unit weight in the projection
    #             layer vs position's 120//16=7, making position 3.5× more salient.
    # sigma_shift=1 (σ≈0.5): largest perturbation; needed for hard-exploration task.
    "MountainCar-v0": {
        "suite":                  "gymnax",
        "gymnax_name":            "MountainCar-v0",
        "action_type":            "discrete",
        "obs_scale":              [200, 500],
        "max_steps":              200,
        "pop_size":               2048,
        "rank":                   4,
        "sigma":                  0.5,
        "sigma_shift":            1,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        # fast_fitness=False: normalize by RMS so the rare goal-reaching pairs
        # get high gradient weight; fast_fitness=True would let noise pairs (97%)
        # drown the goal signal.
        "fast_fitness":           False,
        # update_threshold=1 → threshold = 1×sqrt(N/2) = 32.
        # Lower than the default (2→64) so weights update from weaker signals;
        # safe here because fast_fitness=False already filters noise by magnitude.
        "update_threshold":       1,
    },
    # ── MountainCarContinuous-v0 ──────────────────────────────────────────────
    # Continuous, action ∈ [-1, 1], 999-step episodes.
    # Reward: -0.1 * action² per step + 100 on success (goal reached).
    # Same observation space as discrete variant → same per-dim obs_scale applies.
    # sigma_shift=2 (σ≈0.25): shaped reward provides gradient signal so smaller
    # perturbations suffice; mirrors CartPole's setting for continuous control.
    # pop=2048, 100 epochs → 2048 × 999 × 100 ≈ 200M steps.
    "MountainCarContinuous-v0": {
        "suite":                  "gymnax",
        "gymnax_name":            "MountainCarContinuous-v0",
        "action_type":            "continuous",
        "action_dim":             1,
        "obs_scale":              [200, 500],
        "max_steps":              999,
        "action_scale":           1.0,
        "pop_size":               2048,
        "rank":                   4,
        "sigma":                  0.25,
        "sigma_shift":            2,
        "n_parallel_evaluations": 1,
        "deterministic_policy":   False,
        "fast_fitness":           False,
        "update_threshold":       1,
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
    # batch_size = 256 × 200 = 51 200; minibatch = 51200 / 8 = 6400
    # High ent_coef (0.01) encourages broad exploration, critical for the
    # hard-exploration sparse-reward structure of MountainCar (-1/step).
    # gamma=0.999 provides long-horizon credit assignment over 200-step episodes.
    "MountainCar-v0": {
        "activation":          "pqn",
        "clip_eps":            0.2,
        "ent_coef":            0.01,
        "gae_lambda":          0.95,
        "gamma":               0.999,
        "learning_rate":       0.001,
        "max_grad_norm":       0.5,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            256,
        "num_epochs":          4,
        "num_minibatches":     8,
        "num_steps":           200,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.5,
    },
    # batch_size = 256 × 256 = 65 536; minibatch = 65536 / 8 = 8192
    # Continuous variant has a shaped reward (-0.1*a² + 100 on success), so
    # lower ent_coef is sufficient; action noise from GaussianPolicy handles
    # exploration.  gamma=0.999 for the 999-step episode horizon.
    "MountainCarContinuous-v0": {
        "activation":          "pqn",
        "clip_eps":            0.2,
        "ent_coef":            0.001,
        "gae_lambda":          0.95,
        "gamma":               0.999,
        "learning_rate":       0.0003,
        "max_grad_norm":       1.0,
        "layer_size":          256,
        "n_layers":            3,
        "normalize_obs":       True,
        "normalize_rew":       False,
        "num_envs":            256,
        "num_epochs":          4,
        "num_minibatches":     8,
        "num_steps":           256,
        "rew_norm_discount":   0.99,
        "skip_initial_eval":   False,
        "vf_coef":             0.5,
    },
}
