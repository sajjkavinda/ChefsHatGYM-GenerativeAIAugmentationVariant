import sys
import os
import asyncio

# add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rooms.room import Room
from agents.random_agent import RandomAgent
from agents.agent_generative_ppo import AgentGenerativePPO

def run_room(training=True, matches=100, output_folder="outputs"):
    room = Room(
        run_remote_room=False,
        room_name="Room_PPO_Gen",
        max_matches=matches,
        output_folder=output_folder,
        save_game_dataset=True,
        save_logs_game=False,
        save_logs_room=False
    )

    # Connect 3 random agents
    for i in range(3):
        room.connect_player(RandomAgent(name=f"Random{i}", log_directory=room.room_dir))

    # Connect Generative PPO agent
    agent = AgentGenerativePPO(name="PPO_Gen", log_directory=room.room_dir)
    room.connect_player(agent)

    # Run the room asynchronously
    asyncio.run(room.run())

    return room, agent

if __name__ == "__main__":
    run_room(training=True, matches=100, output_folder="outputs")