import numpy as np
from time import sleep
from robotman import Manipulator2D
# from check_robotman import Manipulator2D

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

env = Manipulator2D()
eval_env = Manipulator2D()
n_actions = env.action_space.shape[-1]
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma = float(0.5) * np.ones(n_actions))
eval_callback = EvalCallback(eval_env, best_model_save_path='/logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=False, render=False)

# Select DDPG algorithm from the stable-baseline library
model = SAC('MlpPolicy', env, verbose=1)
print('loaded model')
print('learning')
model.learn(total_timesteps=10000, callback=eval_callback) #3000000
print('finished learning')
#
model.save("sac_manipulator2D")

del model  # remove to demonstrate saving and loading

model = SAC.load("sac_manipulator2D")

# Reset the simulation environment
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print('finish')
        break
env.render()