import pacman
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# env = pacman.POPacman(20)
env = gym.make('MsPacman-v0')
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log="./logs/baseline_MDP")

eval_scores = []
for ep in range(10):
    obs = env.reset()
    total_reward = 0
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    eval_scores.append(total_reward)
print(f"Before learning: {np.mean(eval_scores)}")

for k in range(300):
    print(f"Epoch {k}")
    model.learn(total_timesteps=1000)
    if k%10 == 0:
        eval_scores = []
        for ep in range(10):
            obs = env.reset()
            total_reward = 0
            for i in range(1000):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            eval_scores.append(total_reward)
        print(f"Mean score: {np.mean(eval_scores)}")
