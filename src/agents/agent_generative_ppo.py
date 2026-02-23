# src/agents/agent_generative_ppo.py
import numpy as np
from .base_agent import BaseAgent

class AgentGenerativePPO(BaseAgent):
    """
    Generative PPO Agent for Chef's Hat.
    Uses simple generative policy to initialize actions,
    compatible with the official repo.
    """

    def __init__(self, name, log_directory=""):
        # Call BaseAgent with only the expected args
        super().__init__(name=name, log_directory=log_directory)
        self.memory = []  # optional memory for future PPO training

    def request_action(self, payload):
        possible_actions = payload["possible_actions"]

        def score_action(action_str):
            if action_str == "pass":
                return -1  # lowest score, so pass is chosen only if forced
            parts = action_str.split(";")
            return sum(int(p[1:]) for p in parts)

        scored_actions = [(score_action(a), idx) for idx, a in enumerate(possible_actions)]
        scored_actions.sort(reverse=True)

        # pick randomly among top 3
        top_choices = scored_actions[:3] if len(scored_actions) >= 3 else scored_actions
        chosen_score, chosen_index = top_choices[np.random.randint(len(top_choices))]

        self.log(f"GenerativePPO chose action index {chosen_index} ({possible_actions[chosen_index]})")
        return chosen_index