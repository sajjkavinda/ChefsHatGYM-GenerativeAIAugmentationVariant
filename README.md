# Chef's Hat Gym – Generative PPO Agent

This repository contains a **Generative PPO reinforcement learning agent** for the **Chef's Hat** multi-agent card game. It uses the official [Chef's Hat Gym](https://github.com/pablovin/ChefsHatGYM) environment and demonstrates a **Generative AI variant** approach for policy initialization and action selection.

---

## Features

- Implements a **Generative PPO agent** that prioritizes high-value actions using a heuristic policy.
- Compatible with the official Chef's Hat Gym environment.
- Supports multi-agent matches with **RandomAgent opponents**.
- Saves game datasets and logs for analysis.
- Generates **score progression plots** to visualize learning performance.
- Trains and evaluates agent over **100 matches** by default.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ChefsHat-Gym-Generative-PPO.git
cd ChefsHatGYM
```

2. Install Dependencies

pip install -r Requirements.txt

## How to run
Run the following command to train the agent for 100 matches against 3 RandomAgents:
```bash
python task2/train_generative_ppo.py
```
To plot the scores
```bash
python task2/plot_scores.py
```

## Notes

Designed for Coventry University assignment purposes (Module-specific variant 6).
Fully compatible with the official Chef's Hat Gym repository.
Tested with Python 3.10–3.12 and required packages listed in Requirements.txt.

## Outputs

These logs record in the /output folder:

- Actions selected by the agent

- Environment updates (player actions, match events)

- Match completion signals

- Episode rewards

- Training progress indicators

Reward Curves:

X-axis → Episode (Match Number)

Y-axis → Total Reward per Match

What the Reward Means:

+1.0 → Agent finishes first

+0.2 → Middle ranking

-0.5 → Last place

+0.05 → Valid step reward

-0.1 → Invalid action penalty
