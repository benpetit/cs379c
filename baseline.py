import pacman
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# env = pacman.POPacman(20)
env = gym.make('MsPacman-v0')
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=0)

for k in range(300):
    print(f"Epoch {k}")
    model.learn(total_timesteps=1000)
    if k%10 == 0:
        for ep in range(3):
            obs = env.reset()
            total_reward = 0
            for i in range(1000):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            print(f"Episode {ep}: {total_reward}")
