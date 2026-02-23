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

## Repository Structure

ChefsHatGYM/                                                                                                                                                                           
├── src/                                                                                                                                                                               
│   ├── agents/                                                                                                                                                                        
│   │   ├── agent_generative_ppo.py      # Generative PPO agent implementation                                                                                                         
│   │   ├── random_agent.py              # Random baseline agent                                                                                                                       
│   │   ├── agent_dqn.py                                                                                                                                                               
│   │   ├── agent_ppo.py                                                                                                                                                               
│   │   ├── agent_ppo_old.py                                                                                                                                                           
│   │   ├── base_agent.py                                                                                                                                                              
│   │   ├── base_agent_server.py                                                                                                                                                       
│   │   ├── larger_value.py                                                                                                                                                            
│   │   └── __init__.py                                                                                                                                                                
│   │                                                                                                                                                                                  
│   ├── rooms/                                                                                                                                                                         
│   │   ├── room.py                                                                                                                                                                    
│   │   ├── local_communicationn.py                                                                                                                                                    
│   │   └── __init__.py                                                                                                                                                                
│   │                                                                                                                                                                                  
│   └── ...                                                                                                                                                                            
│                                                                                                                                                                                      
├── task2/                                                                                                                                                                             
│   ├── train_generative_ppo.py           
│   └── ...                               
│
├── outputs/                               
│   └── Room_PPO_Gen_<timestamp>/          
│       ├── agents/                                                                  
│       │   └── PPO_Gen/                                                                                                                  
|               └── PLAYER_PPO_Gen.log/                                                                                              
│       │   └── Random0/                                                                                                                
|               └── PLAYER_Random0.log/                                                                                                    
│       │   └── Random1/                                                                                                                
|               └── PLAYER_Random1.log/                                                                                          
│       ├── dataset/                                                                                                        
│       │   └── game_dataset.pkl.csv                                                                                                        
│                                                                                                                                                              
├── LICENSE                                                                                                                                                                
├── README.md 
├── source_repo_README.md                                                                                                                                                          
├── Requirements.txt                                                                                                                                                  
├── setup.py                                                                                                                                                                      
├── pyproject.toml                                                                                                                                                              
├── docs/                                                                                                                                                                          
└── examples/                                                                                                                                                                      

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ChefsHat-Gym-Generative-PPO.git
cd ChefsHatGYM
```

2. Install Dependencies

pip install -r Requirements.txt
pip install matplotlib pandas numpy

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
