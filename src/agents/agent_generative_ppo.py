# src/agents/agent_generative_ppo.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class AgentGenerativePPO(BaseAgent):

    def __init__(self, name, log_directory=""):
        super().__init__(name=name, log_directory=log_directory)

        # PPO settings
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network (output size large enough for action space)
        self.policy = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # Memory
        self.memory = []
        self.episode_rewards = []
        self.current_reward = 0

        self.last_state = None
        self.last_action = None
        self.last_log_prob = None

    # -------------------------------------------------------
    # ACTION SELECTION
    # -------------------------------------------------------

    def request_action(self, payload):

        possible_actions = payload["possible_actions"]

        # Create simple numeric state representation
        state = np.zeros(100)
        state[: min(len(possible_actions), 100)] = 1
        state_tensor = torch.FloatTensor(state).to(self.device)

        logits = self.policy(state_tensor)
        probs = torch.softmax(logits[: len(possible_actions)], dim=0)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.last_state = state_tensor
        self.last_action = action
        self.last_log_prob = dist.log_prob(action)

        chosen_index = action.item()

        self.log(f"PPO chose action {chosen_index} ({possible_actions[chosen_index]})")

        return chosen_index

    # -------------------------------------------------------
    # STEP REWARD
    # -------------------------------------------------------

    def update_player_action(self, payload):

        if payload["player"] == self.name:

            # small reward for valid move
            reward = 0.05 if payload.get("valid", True) else -0.1
            self.current_reward += reward

            self.memory.append({
                "state": self.last_state,
                "action": self.last_action,
                "log_prob": self.last_log_prob,
                "reward": reward,
                "done": False
            })

    # -------------------------------------------------------
    # FINAL MATCH REWARD
    # -------------------------------------------------------

    def update_match_over(self, payload):

        # -----------------------------------
        # SAFE RANKING EXTRACTION
        # -----------------------------------

        final_reward = 0.0

        # Case 1: ranking exists
        if "ranking" in payload:
            ranking = payload["ranking"]

            if ranking[0] == self.name:
                final_reward = 1.0
            elif ranking[-1] == self.name:
                final_reward = -0.5
            else:
                final_reward = 0.2

        # Case 2: scores exist (more common in Chef’s Hat)
        elif "scores" in payload and "players" in payload:

            players = payload["players"]
            scores = payload["scores"]

            if self.name in players:
                my_index = players.index(self.name)
                my_score = scores[my_index]

                if my_score == max(scores):
                    final_reward = 1.0
                elif my_score == min(scores):
                    final_reward = -0.5
                else:
                    final_reward = 0.2

        # Case 3: unknown format → neutral reward
        else:
            print("Warning: Unknown match_over payload format")
            final_reward = 0.0

        # -----------------------------------
        # APPLY REWARD SAFELY
        # -----------------------------------

        self.current_reward += final_reward

        if len(self.memory) > 0:
            self.memory[-1]["reward"] += final_reward
            self.memory[-1]["done"] = True

        self.episode_rewards.append(self.current_reward)

        print(f"Episode reward: {self.current_reward}")

        self.current_reward = 0

        self.train_ppo()

        if len(self.episode_rewards) % 20 == 0:
            self.plot_rewards()
        # -------------------------------------------------------
        # PPO TRAINING
        # -------------------------------------------------------

    def train_ppo(self):

        if len(self.memory) == 0:
            return

        states = torch.stack([m["state"] for m in self.memory])
        actions = torch.stack([m["action"] for m in self.memory])
        old_log_probs = torch.stack([m["log_prob"] for m in self.memory]).detach()
        rewards = [m["reward"] for m in self.memory]
        dones = [m["done"] for m in self.memory]

        returns = self.compute_returns(rewards, dones)

        for _ in range(4):

            logits = self.policy(states)
            probs = torch.softmax(logits, dim=1)

            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()

            advantage = returns - returns.mean()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage

            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    # -------------------------------------------------------
    # RETURNS
    # -------------------------------------------------------

    def compute_returns(self, rewards, dones):

        returns = []
        R = 0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns).float().to(self.device)

    # -------------------------------------------------------
    # PLOT
    # -------------------------------------------------------

    def plot_rewards(self):
        plt.figure()
        plt.plot(self.episode_rewards)
        plt.title("Training Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()