import numpy as np
from time import sleep
from robotman import Manipulator2D
# from check_robotman import Manipulator2D

from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = Manipulator2D()

n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma = float(0.5) * np.ones(n_actions))

# Select DDPG algorithm from the stable-baseline library
model = DDPG(MlpPolicy, env=env, action_noise=action_noise, verbose=1)
print('loaded model')

print('learning')
model.learn(total_timesteps=1000) #3000000
print('finished learning')
#
model.save("ddpg_manipulator2D")

del model  # remove to demonstrate saving and loading

model = DDPG.load("ddpg_manipulator2D")

# Reset the simulation environment
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done:
        print('finish')
        break

env.render()