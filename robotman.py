import gym
from gym import core, spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def sum_angle(a, b):
    c = a + b
    if c >= np.pi:
        c -= 2 * np.pi
    elif c < -np.pi:
        c += 2 * np.pi
    return c

class Manipulator2D(gym.Env):
    def __init__(self,arm1=1,arm2=1,dt=0.01, tol =0.1):
        #the location of the arm1, arm2, destination
        self.obs_high = np.array([1, 1, 2, 2, 2, 2],dtype=np.float64)  # x1, y1, x2, y2, xd, yd
        self.obs_low = -self.obs_high

        self.action_high = np.array([np.pi, np.pi],dtype=np.float64)
        self.action_low = -self.action_high

        #shape (Optional[Sequence[int]]) – The shape is inferred from the shape of low or high np.ndarray`s with `low and high scalars defaulting to a shape of (1,)
        #dtype – The dtype of the elements of the space. If this is an integer type, the Box is essentially a discrete space.
        #seed – Optionally, you can use this argument to seed the RNG that is used to sample from the space.
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float64)

        self.arm1 = arm1
        self.arm2 = arm2
        self.dt = dt #size of the the timestep
        self.tol = tol # threshold, the min distance to the goal

        self.reset()

        self.seed()

    def step(self,action):
        #numpy.clip(array, min, max)
    # array 내의 element들에 대해서
    # min 값 보다 작은 값들을 min값으로 바꿔주고
    # max 값 보다 큰 값들을 max값으로 바꿔주는 함수.
        #used to make the actions between the action value
        action = np.clip(action, self.action_low, self.action_high)

        self.theta1 += action[0] * self.dt
        self.theta21 += action[1] * self.dt
        self.theta2 = sum_angle(self.theta21, self.theta1)

        self.x1 = np.cos(self.theta1) * self.arm1
        self.y1 = np.sin(self.theta1) * self.arm1
        self.x2 = self.x1 + np.cos(self.theta2) * self.arm2
        self.y2 = self.y1 + np.sin(self.theta2) * self.arm2
        self.t += self.dt

        reward, done = self._get_reward(
            np.linalg.norm([self.xd-self.x2, self.yd-self.y2])
        )

        info ={}

        self.buffer.append(
            [
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.t,
                reward
            ]
        )

        return self._get_state(),reward, done, info

    def reset(self):
        self.theta1 = 0
        self.theta21 = 0
        self.theta2 = sum_angle(self.theta1, self.theta21)
        self.x1 = np.cos(self.theta1) * self.arm1
        self.y1 = np.sin(self.theta1) * self.arm1
        self.x2 = self.x1 + np.cos(self.theta2) * self.arm2
        self.y2 = self.y1 + np.sin(self.theta2) * self.arm2

        #get the random destination the length and the alpha
        #rd = np.random.uniform(low=1.5, high=1.99)
        #alphad = np.random.uniform(low=-np.pi, high=np.pi)
        rd =1.75
        alphad = np.pi/4

        self.xd = rd * np.cos(alphad)
        self.yd = rd * np.sin(alphad)

        self.done = False
        self.t=0
        self.buffer=[]

        return self._get_state()

    def _get_reward(self,l):
        done =False

        if 1 <self.tol:
            reward =1
            done =True

        else :
            reward = -l**2
        return reward, done

    def _get_state(self):
        return np.array([
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.xd,
            self.yd
        ])

    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def render(self,plot_reward = False):
        buffer = np.array(self.buffer)
        #plotting two gaphs
        #plot the first one
        plt.figure(1)
        #draw the destination
        plt.scatter([self.xd], [self.yd], c='r', marker='x', s=300)
        #plot the arm1 route
        plt.plot(buffer[:, 0], buffer[:, 1],c='g')
        plt.plot(buffer[:, 2] - 0.1*np.cos(self.theta2), buffer[:, 3] - 0.1 * np.sin(self.theta2), c='b')

        #drawing the arm on the plot
        plt.plot(
            [0, self.x1, self.x2 - 0.1*np.cos(self.theta2)],
            [0, self.y1, self.y2 - 0.1*np.sin(self.theta2)],
            marker='o',
            c='k'
        )
        #drawing the final gripper on the
        plt.plot(
            [
                self.x2 + 0.1*np.cos(self.theta2) - 0.1*np.sin(self.theta2),
                self.x2 - 0.1*np.cos(self.theta2) - 0.1*np.sin(self.theta2),
                self.x2 - 0.1*np.cos(self.theta2) + 0.1*np.sin(self.theta2),
                self.x2 + 0.1*np.cos(self.theta2) + 0.1*np.sin(self.theta2)
            ],
            [
                self.y2 + 0.1*np.sin(self.theta2) + 0.1*np.cos(self.theta2),
                self.y2 - 0.1*np.sin(self.theta2) + 0.1*np.cos(self.theta2),
                self.y2 - 0.1*np.sin(self.theta2) - 0.1*np.cos(self.theta2),
                self.y2 + 0.1*np.sin(self.theta2) - 0.1*np.cos(self.theta2)
            ],
            c='k'
        )
        plt.axis('square')
        plt.title('Trajectory')
        if plot_reward:
           # Plot the reward curve
           plt.figure(2)
           plt.plot(buffer[:, 4], buffer[:, 5])
           plt.title('Rewards')
        plt.show()



def test(env):
    env.reset()
    c= (env.xd**2 + env.yd**2 - env.arm1**2 - env.arm2**2) / env.arm1 /2
    s = np.sqrt(env.arm2**2 - c**2)
    theta21d = np.arctan2(s, c)
    theta1d = sum_angle(np.arctan2(env.yd, env.xd), - np.arctan2(s, env.arm1 + c))
    # Monitor the trajectories of the robotic arm for 10 seconds
    for t in np.arange(0, 10, env.dt):
        # Feedback control using the calculated degree value from the above
        action = [theta1d - env.theta1, theta21d - env.theta21]
        # Simulate for one timestep using the "step" method
        # Gets the changed state(observation), reward, terminal condition, and additional information
        next_state, reward, done, info = env.step(action)
        # Episode finished
        if done:
            break
    # Visualize the trajectories of the robotic arm
    env.render(plot_reward=True)

if __name__=='__main__':
    test(Manipulator2D(tol=0.01))







