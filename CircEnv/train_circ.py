import gymnasium as gym
import numpy as np
#from sb3_contrib import TQC
from stable_baselines3 import PPO
import os
from pathlib import Path
import circ_env

env = gym.make('circ_env/Circle-v0')

# tensorboard logiranje
models_dir = Path("models/PPO_goal_push")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
logdir = "logs_goal_push"
if not os.path.exists(logdir):  
    os.makedirs(logdir)


def latest_checkpoint(models_dir):
    checkpoints = [
        path for path in models_dir.glob("*.zip")
        if path.stem.isdigit()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.stem))


# PPO01
checkpoint = latest_checkpoint(models_dir)
if checkpoint is not None:
    print(f"Resuming from {checkpoint}")
    model = PPO.load(str(checkpoint), env=env, tensorboard_log=logdir, verbose=1)
    start_steps = int(checkpoint.stem)
else:
    model = PPO('MlpPolicy',
                env=env,
                tensorboard_log=logdir,
                verbose=1,
                n_steps=2048,
                batch_size=64,
                gae_lambda=0.95,
                gamma=0.995,
                n_epochs=10,
                ent_coef=0.01,
                learning_rate=3e-4,
                clip_range=0.2,
                seed=2)
    start_steps = 0
     
# Reset the environment
obs = env.reset()

# iteracija skozi učenje in shranjevanje modela
TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,tb_log_name="PPO_goal_push")

    model.save(models_dir / str(start_steps + TIMESTEPS*iters))
