from pathlib import Path

import gymnasium as gym  # popravljeno
import numpy as np
from sb3_contrib import TQC
# from stable_baselines3 import PPO
import os
import circ_env


MODEL_PATH = Path(__file__).resolve().parent / "models" / "TQC02" / "280000"

env = gym.make('circ_env/AirHockey-v0', render_mode="human")

print(f"Loading model: {MODEL_PATH}")
model = TQC.load(str(MODEL_PATH), env=env)

# Reset the environment
vec_env = model.get_env()
obs = vec_env.reset()

EPISODES = 1000

for episode in range(EPISODES):

    obs = vec_env.reset()
    done = False

    while not done:

        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action.squeeze())  # popravljeno

        env.render()

# Close the environment
env.close()
vec_env.close()
