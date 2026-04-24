"""
Deep Deterministic Policy Gradient (DDPG) for UAV Mobile Edge Computing.

Migrated from TensorFlow 1.15 to PyTorch.
Uses Actor-Critic architecture with soft target updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from UAV_environment import UAVEnv
from state_normalization import StateNormalization

# ─── Hyperparameters ────────────────────────────────────
MAX_EPISODES = 1000
LR_A = 0.001        # Actor learning rate
LR_C = 0.002        # Critic learning rate
GAMMA = 0.001        # Discount factor (kept from original)
TAU = 0.01           # Soft target update rate
VAR_MIN = 0.01       # Minimum exploration noise
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64


# ─── Actor Network ──────────────────────────────────────
class Actor(nn.Module):
    """Maps state → continuous action in [-1, 1]."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU6(),
            nn.Linear(400, 300),
            nn.ReLU6(),
            nn.Linear(300, 10),
            nn.ReLU(),
            nn.Linear(10, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


# ─── Critic Network ─────────────────────────────────────
class Critic(nn.Module):
    """Maps (state, action) → Q-value scalar."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # First layer: separate weights for state and action
        # (matches original TF architecture)
        self.w1_s = nn.Linear(state_dim, 400, bias=False)
        self.w1_a = nn.Linear(action_dim, 400, bias=False)
        self.b1 = nn.Parameter(torch.zeros(400))

        self.net = nn.Sequential(
            nn.ReLU6(),
            nn.Linear(400, 300),
            nn.ReLU6(),
            nn.Linear(300, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, state, action):
        x = self.w1_s(state) + self.w1_a(action) + self.b1
        return self.net(x)


# ─── Replay Buffer ──────────────────────────────────────
class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, state_dim, action_dim, capacity=MEMORY_CAPACITY):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def store(self, s, a, r, s_):
        idx = self.ptr % self.capacity
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
        )

    def is_ready(self):
        return self.size >= self.capacity


# ─── DDPG Agent ─────────────────────────────────────────
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.action_dim = action_dim
        self.action_bound = action_bound

        # Main networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)

        # Target networks (copies)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_C)

        # Replay buffer
        self.memory = ReplayBuffer(state_dim, action_dim)

    def choose_action(self, state):
        """Select action from actor network (no gradient needed)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        return action

    def learn(self):
        """One step of DDPG training."""
        # ── Soft update target networks ──
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # ── Sample batch ──
        bs, ba, br, bs_ = self.memory.sample(BATCH_SIZE)

        # ── Update Critic ──
        with torch.no_grad():
            next_actions = self.actor_target(bs_)
            target_q = br + GAMMA * self.critic_target(bs_, next_actions)

        current_q = self.critic(bs, ba)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Update Actor ──
        actor_loss = -self.critic(bs, self.actor(bs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def store_transition(self, s, a, r, s_):
        self.memory.store(s, a, r, s_)

    def _soft_update(self, source, target):
        """Polyak averaging: θ_target = τ·θ + (1-τ)·θ_target"""
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.copy_(TAU * src.data + (1 - TAU) * tgt.data)


# ─── Training Loop ──────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)

    env = UAVEnv()
    MAX_EP_STEPS = env.slot_num
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    ddpg = DDPG(s_dim, a_dim, a_bound)
    s_normal = StateNormalization()

    var = 0.01  # Exploration noise (matching original)
    t1 = time.time()
    ep_reward_list = []

    for i in range(MAX_EPISODES):
        s, _ = env.reset()  # Gymnasium returns (obs, info)
        ep_reward = 0
        j = 0

        while j < MAX_EP_STEPS:
            # Select action with exploration noise
            a = ddpg.choose_action(s_normal.state_normal(s))
            a = np.clip(np.random.normal(a, var), *a_bound)

            # Step environment
            (s_, comm_energy, comp_energy, prop_energy, latency,
             is_terminal, step_redo, offloading_ratio_change,
             reset_dist) = env.step(a)

            # Compute reward
            r = env.compute_reward(
                comm_energy, comp_energy, prop_energy, latency
            )

            if step_redo:
                continue
            if reset_dist:
                a[2] = -1
            if offloading_ratio_change:
                a[3] = -1

            # Store and learn
            ddpg.store_transition(
                s_normal.state_normal(s), a, r,
                s_normal.state_normal(s_)
            )

            if ddpg.memory.is_ready():
                var = max(var * 0.9997, VAR_MIN)
                ddpg.learn()

            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1 or is_terminal:
                print(f"Episode: {i:4d} | Steps: {j:2d} | "
                      f"Reward: {ep_reward:7.2f} | Explore: {var:.3f}")
                ep_reward_list.append(ep_reward)

                with open('output.txt', 'a') as f:
                    f.write("\n======== This episode is done ========")
                break

            j += 1

    # ─── Results ────────────────────────────────────────
    print(f"\nRunning time: {time.time() - t1:.1f}s")

    plt.figure(figsize=(10, 5))
    plt.plot(ep_reward_list, alpha=0.3, color='blue', label='Raw')
    # Moving average
    if len(ep_reward_list) > 20:
        moving_avg = np.convolve(
            ep_reward_list, np.ones(20) / 20, mode='valid'
        )
        plt.plot(range(19, len(ep_reward_list)), moving_avg,
                 color='red', linewidth=2, label='Moving Avg (20)')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDPG Training: UAV Mobile Edge Computing")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()