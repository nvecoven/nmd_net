import gym
import numpy as np

class GymPendulum():
    def __init__(self, add_step = True):
        self.env = gym.make('Pendulum-v0')
        self.obs_dim = self.env.observation_space.shape[0]
        self.nofb_obs_dim = self.obs_dim
        if add_step:
            self.obs_dim += 1
        self.add_step = add_step
        self.act_dim = self.env.action_space.shape[0]
        self.nofb_obs_dim = self.obs_dim

    def take_step(self, actions):
        obs, reward, done, _ = self.env.step(actions)
        if self.add_step:
            self.step += 1e-3
            obs = np.concatenate([obs, [self.step]])
        return obs, reward, done, 0

    def reset(self, params = None):
        obs = self.env.reset()
        if self.add_step:
            self.step = 0
            obs = np.concatenate([obs, [self.step]])
        return obs

    def render(self):
        self.env.render()