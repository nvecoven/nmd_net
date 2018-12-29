import numpy as np
import pygame
import sys
import time

class IndependantBandits():
    def __init__(self, fixed_prob = False):
        self.first_render = True
        self.nbr_bandits = 2
        self.act_dim = 1
        self.obs_dim = 3
        self.nofb_obs_dim = 0
        self.fixed_prob = fixed_prob
        self.defaults_ps = [np.random.rand() for _ in range(self.nbr_bandits)]
        self.max_steps = 100
        print (self.defaults_ps)

    def take_step(self, actions):
        # actions = np.clip(actions, -1., 1.)
        if actions >= 0:
            prob = self.ps[0]
        else:
            prob = self.ps[1]
        if not prob == np.max(self.ps):
            self.bad_pulls_index.append(self.episode_step)
            self.bad_pulls += 1
        if np.random.rand() < prob:
            reward = 1.0
            self.rewards_list.append(self.episode_step)
        else:
            reward = 0.0
        self.regret += np.max(self.ps) - prob
        self.episode_step += 1
        done = False
        if self.episode_step >= self.max_steps:
            done = True
        self.state = [actions, reward, self.episode_step]
        self.dt = 0.1
        return self._get_obs(), reward, done, self.regret

    def _get_obs(self, params = None):
        previous_action, previous_reward, time = self.state
        return np.array([previous_action, previous_reward, time])

    def reset(self):
        self.regret = 0
        self.bad_pulls_index = []
        self.rewards_list = []
        self.bad_pulls = 0
        if self.fixed_prob:
            self.ps = self.defaults_ps
        else:
            if np.random.rand() > 0.5:
                self.ps = [0.25, 0.75]
            else:
                self.ps = [0.75, 0.25]
            # self.ps = []
            # for i in range(self.nbr_bandits):
            #     self.ps.append(np.random.rand())
        self.episode_step = 0
        self.state = np.array([0.0, 0.0, self.episode_step])
        return self._get_obs()

    def render(self):
        resolution = (640, 480)
        indices = self.bad_pulls_index
        rectangle_width = np.floor(resolution[0]/self.max_steps)
        if self.first_render:
            self.first_render = False
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(resolution)
            self.previous_time = time.time()
        middle_screen_height = (0, resolution[1] / 2)
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(self.ps), False, (0, 0, 0))
        bar_color = (255, 0, 0)
        bar_color2 = (0,255,0)
        self.screen.fill((255, 255, 255))
        self.screen.blit(textsurface, (int(resolution[0] / 2), 0))
        for p in indices:
            points = [(rectangle_width * p, middle_screen_height[1]),
                      (rectangle_width * (p + 1), middle_screen_height[1])]
            pygame.draw.lines(self.screen, bar_color, False, points, 5)
        for p2 in self.rewards_list:
            points = [(rectangle_width * p2, middle_screen_height[1] + 10),
                      (rectangle_width * (p2 + 1), middle_screen_height[1] + 10)]
            pygame.draw.lines(self.screen, bar_color2, False, points, 5)
        cur_time = time.time()
        pygame.display.update()
        while not (cur_time - self.previous_time >= self.dt):
            cur_time = time.time()
        self.previous_time = cur_time