"""
dqn_baseline.py
===============
PyTorch DQN baseline (discrete environments only).

Architecture matches the paper: 3-layer MLP, 256 units, pqn activation
(relu(layer_norm(x))), same as the float EggRoll and PPO baselines.

This uses gymnasium (not gymnax) since PyTorch is CPU/CUDA-native.
Only discrete-action environments are supported.

Usage
-----
  python dqn_baseline.py --env CartPole-v1
  python dqn_baseline.py --env CartPole-v1 --save_results
  python dqn_baseline.py --env jumanji/Game2048-v1  # not supported, DQN discrete only

Note: for a fair wall-clock comparison with QEggRoll (JAX), note that DQN
runs single-environment steps sequentially while QEggRoll vmaps over N=512+
parallel environments. Wall-clock time reflects this structural difference.
"""

import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import tyro
from collections import deque
from dataclasses import dataclass
from typing import Optional

from metrics import RunLogger

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 256
N_LAYERS   = 3


@dataclass
class Args:
    env:             str   = "CartPole-v1"
    seed:            int   = 0
    total_timesteps: int   = 500_000
    log_every:       int   = 5_000     # log every N environment steps
    eval_episodes:   int   = 10
    save_results:    bool  = False
    results_dir:     str   = "results"
    run_tag:         str   = ""
    # DQN hyperparameters
    learning_rate:   float = 1e-4
    gamma:           float = 0.99
    buffer_size:     int   = 50_000
    batch_size:      int   = 128
    target_update:   int   = 1_000    # steps between target network syncs
    eps_start:       float = 1.0
    eps_end:         float = 0.05
    eps_decay:       int   = 50_000   # steps for linear epsilon decay
    min_replay:      int   = 5_000    # steps before first update
    train_freq:      int   = 4        # update every N steps


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class PQN(nn.Module):
    """pqn = relu(layer_norm(x)) — matches the EGGROLL paper's activation."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        return F.relu(self.ln(x))


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        dims = [obs_dim] + [HIDDEN_DIM] * N_LAYERS
        layers = []
        for i in range(N_LAYERS):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(PQN(dims[i+1]))
        layers.append(nn.Linear(HIDDEN_DIM, act_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Output layer: small init
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs),   dtype=torch.float32, device=device),
            torch.tensor(np.array(act),   dtype=torch.int64,   device=device),
            torch.tensor(np.array(rew),   dtype=torch.float32, device=device),
            torch.tensor(np.array(nobs),  dtype=torch.float32, device=device),
            torch.tensor(np.array(done),  dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_dqn(args: Args):
    # ── Validate environment is discrete ─────────────────────────────────────
    env_name = args.env.split("/")[-1]  # strip suite prefix for gymnasium
    probe_env = gym.make(env_name)
    if not isinstance(probe_env.action_space, gym.spaces.Discrete):
        raise ValueError(
            f"DQN requires discrete actions. {args.env} has "
            f"{type(probe_env.action_space).__name__} action space.")
    probe_env.close()

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDQN | {args.env}  device={device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Disable cuDNN non-deterministic algorithms so GPU runs are reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env  = gym.make(env_name)
    env.reset(seed=args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)

    q_net      = QNetwork(obs_dim, act_dim).to(device)
    target_net = QNetwork(obs_dim, act_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    buffer    = ReplayBuffer(args.buffer_size)

    print(f"  obs_dim={obs_dim}  act_dim={act_dim}"
          f"  params={sum(p.numel() for p in q_net.parameters()):,}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    logger = RunLogger("DQN", args.env, args.seed, vars(args))
    logger.reset_clock()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def select_action(obs_np, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        obs_t = torch.tensor(obs_np.ravel(), dtype=torch.float32, device=device)
        with torch.no_grad():
            return int(q_net(obs_t.unsqueeze(0)).argmax(dim=1).item())

    def evaluate():
        eval_env = gym.make(env_name)
        returns = []
        for ep in range(args.eval_episodes):
            obs, _ = eval_env.reset(seed=args.seed + ep)
            total, done = 0.0, False
            while not done:
                obs_t = torch.tensor(obs.ravel(), dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = int(q_net(obs_t.unsqueeze(0)).argmax(1).item())
                obs, r, term, trunc, _ = eval_env.step(action)
                total += r
                done = term or trunc
            returns.append(total)
        eval_env.close()
        return np.array(returns)

    def update_network():
        obs, act, rew, nobs, done = buffer.sample(args.batch_size, device)

        with torch.no_grad():
            next_q = target_net(nobs).max(1).values
            target = rew + args.gamma * next_q * (1.0 - done)

        current_q = q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()
        return loss.item()

    # ── Training loop ─────────────────────────────────────────────────────────
    obs, _   = env.reset(seed=args.seed)
    ep_return = 0.0
    last_log  = 0

    for step in range(args.total_timesteps):
        eps = args.eps_end + (args.eps_start - args.eps_end) * max(
            0.0, 1.0 - step / args.eps_decay)
        action    = select_action(obs, eps)
        next_obs, reward, term, trunc, _ = env.step(action)
        done_env  = term or trunc

        buffer.push(obs.ravel(), action, reward, next_obs.ravel(), float(done_env))
        ep_return += reward
        obs        = next_obs

        if done_env:
            obs, _    = env.reset()
            ep_return = 0.0

        if len(buffer) >= args.min_replay and step % args.train_freq == 0:
            update_network()

        if step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step - last_log >= args.log_every:
            eval_rets  = evaluate()
            mean_eval  = float(eval_rets.mean())
            logger.record(step, mean_eval, extra_eps=round(eps, 3))
            logger.print_row(0, step, mean_eval,
                             extra=f"eps={eps:.3f} std={eval_rets.std():.1f}")
            last_log = step

    env.close()

    if args.save_results:
        path = RunLogger.default_path("DQN", args.env, args.seed, args.results_dir, args.run_tag)
        logger.save(path)

    print(f"\nDone. Best eval: {max(e['eval_return'] for e in logger.log):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_dqn(tyro.cli(Args))
