import sys

sys.path.append("/home/deepak/Desktop/RL-Adventure-2/")

from itertools import chain, combinations
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from common import SubprocVecEnv


def make_env(env_name):
    def _thunk():
        return gym.make(env_name)

    return _thunk


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


def test_env(model, transform, env, device):
    state = transform(env.reset())
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = transform(next_state)
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def ith_powerset(n, i):
    s = list(range(n))
    all_subsets = list(
        chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    )
    return all_subsets[i]


def get_transform(indices):
    def transform(x):
        # return np.array([x[idx] for idx in indices])
        return x  # return x[indices]  # np.array([x[idx] for idx in indices])

    return transform


def train_model(num_envs, env_name, indices):

    env_list = SubprocVecEnv([make_env(env_name) for _ in range(num_envs)])
    env = gym.make(env_name)
    hparams = Hparams(env_list)

    transform = get_transform(indices)

    # init model and optimizer
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    model = ActorCritic(
        num_inputs=hparams.num_inputs,
        num_outputs=hparams.num_outputs,
        hidden_size=hparams.hidden_size,
    ).to(device)

    optimizer = optim.Adam(lr=hparams.lr, params=model.parameters())

    # init env
    frame_idx = 0
    state = transform(env_list.reset())

    while frame_idx < hparams.max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(hparams.num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = env_list.step(action.cpu().numpy())
            next_state = transform(next_state)
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            state = next_state
            frame_idx += 1

            if frame_idx % 1000 == 0:
                # average reward over 10 episodes
                avg_reward = np.mean(
                    [test_env(model, transform, env, device) for _ in range(10)]
                )
                print(f"Frame={frame_idx} => avg_reward={avg_reward:.4f}")

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_reward = np.mean([test_env(model, transform, env, device) for _ in range(200)])
    return avg_reward


class Hparams:
    def __init__(self, envs):
        self.num_inputs = envs.observation_space.shape[0]
        self.num_outputs = envs.action_space.n
        self.hidden_size = 256
        self.lr = 3e-4
        self.num_steps = 5
        self.max_frames = 50000


if __name__ == "__main__":
    NUM_ENVS = 16
    ENV_NAME = "CartPole-v1"

    avg_reward = train_model(NUM_ENVS, ENV_NAME, indices=[0, 1, 2, 3])
    print(avg_reward)
