import numpy as np
import pygame
import sys
import time

"""
Definition of the first benchmark's environment.
See ReinforcementLearning/Training/meta_rl_launcher for the parameters used.
"""

class MovingTarget():
    def __init__(self, fixed_offset = False, intra_variation = True, intra_variation_frequency = 0.1, margin = 1.0,
                 act_multiplicator = 2.5, spike_reward = False):
        self.first_render = True
        self.act_dim = 1
        self.obs_dim = 3
        self.spike_reward = spike_reward
        self.nofb_obs_dim = 1
        self.fixed_offset = fixed_offset
        self.intra_variation = intra_variation
        self.intra_variation_frequency = intra_variation_frequency
        self.max_bound = 10.0
        self.act_multiplicator = act_multiplicator
        self.max_steps = 1400
        self.margin = margin

    def test_show(self):
        self.reset()
        done = False
        while not done:
            _,_, done,_ = self.take_step(0.0)
            self.render()

    def take_step(self, actions):
        pos, previous_action, previous_reward, previous_time = self.state
        self.dt = 0.1
        self.episode_step += 1
        actions *= self.act_multiplicator
        goal = pos + self.offset
        actions = np.clip(actions, -2 * self.max_bound, 2 * self.max_bound)
        self.true_poses.append(pos)
        self.predicted_poses.append(actions)
        self.goal_poses.append(goal)
        if np.abs(goal-actions) < self.margin:
            new_pos = (np.random.rand()-0.5) * 1 * self.max_bound
            reward = 10.0
            if self.intra_variation and np.random.rand() < self.intra_variation_frequency:
                self.change += 1
                self.offset = (np.random.rand() - 0.5) * 2.0 * 10.0
        else:
            new_pos = pos
            reward = np.clip(-np.abs(goal - actions), -2 * self.max_bound, 2 * self.max_bound)
        done = False
        self.distance += np.abs(actions - goal)
        if self.episode_step >= self.max_steps:
            done = True
        self.state = [new_pos, actions, reward, self.episode_step]
        return self._get_obs(), reward, done, self.distance

    def _get_obs(self):
        position, previous_action, previous_reward, time = self.state
        return np.array([position, previous_action, previous_reward])

    def reset(self, params = None, reset_episode_step = True):
        self.change = 0
        self.true_poses = []
        self.predicted_poses = []
        self.goal_poses = []
        self.distance = 0.0
        self.initial_pose = (np.random.rand()-0.5) * 1 * self.max_bound
        if reset_episode_step:
            self.episode_step = 0
        if self.fixed_offset:
            self.offset = 0.0
        else:
            self.offset = (np.random.rand()-0.5) * 2.0 * 10.0
        self.state = [self.initial_pose, 0.0, 0.0, self.episode_step]
        if not params == None:
            if params[0] > 0:
                self.intra_variation = True
            else:
                self.intra_variation = False
        return self._get_obs()

    def render(self):
        resolution = (640, 480)
        rectangle_width = np.floor(resolution[0]/self.max_steps)
        if self.first_render:
            self.first_render = False
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(resolution)
            self.previous_time = time.time()
        self.screen.fill((255,255,255))
        middle_screen_height = int(resolution[1] / 2)
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(self.change), False, (0, 0, 0))
        self.screen.blit(textsurface, (int(resolution[0] / 2), 0))

        gt, pred, goal = self.true_poses[-1], self.predicted_poses[-1], self.goal_poses[-1]
        pixel = int(((gt + self.max_bound) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 255, 0), False, points, 5)
        pixel = int(((goal + self.max_bound) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 255), False, points, 5)
        pixel = int(((goal + self.max_bound +1.0) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points, 5)
        pixel = int(((goal + self.max_bound-1.0) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points, 5)
        pixel = int(((pred + self.max_bound) / (self.max_bound * 2)) * resolution[0])
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height - 100)]
        pygame.draw.lines(self.screen, (255, 0, 0), False, points, 5)


        cur_time = time.time()
        pygame.display.update()
        for event in pygame.event.get():
            pass
        while not (cur_time - self.previous_time >= self.dt):
            cur_time = time.time()
        self.previous_time = cur_time
