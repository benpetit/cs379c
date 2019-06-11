import pacman
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
# from stable_baselines.deepq import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack


# env = gym.make('MsPacman-v0')
env = make_atari_env('MsPacmanNoFrameskip-v0', num_env=1, seed=0)
# env = DummyVecEnv([lambda: env])
# env = VecFrameStack(env, n_stack=4)
model = PPO2(CnnPolicy, env, verbose=1, vf_coef=1, tensorboard_log="./logs/baseline_MDP")

eval_scores = []
for ep in range(100):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    eval_scores.append(total_reward)
print(f"Before learning: {np.mean(eval_scores)}")

model.learn(total_timesteps=100000)

eval_scores = []
for ep in range(100):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Total reward: {total_reward}")
    eval_scores.append(total_reward)
print(f"Mean score: {np.mean(eval_scores)}")
