import numpy as np
import pygame
import sys
import time

class VaryingGaussian():
    def __init__(self, fixed_prob = False):
        self.first_render = True
        self.act_dim = 1
        self.obs_dim = 5
        self.fixed_prob = fixed_prob
        self.max_bound = 10.0
        self.max_steps = 100

    def take_step(self, actions):
        previous_pos, previous_change, previous_action, previous_reward, previous_time = self.state
        self.dt = 0.1
        self.episode_step += 1
        new_change = (np.random.rand() -0.5)
        new_position = np.clip(previous_pos + previous_change * self.move_multiplier, -self.max_bound, self.max_bound)
        actions *= 2
        actions = np.clip(actions, -self.max_bound - 5.0, self.max_bound + 5.0)
        reward = np.max([-np.abs(new_position + self.offset - actions), -2.0])
        self.true_poses.append(new_position)
        self.predicted_poses.append(actions)
        self.goal_poses.append(new_position + self.offset)
        done = False
        self.distance += np.abs(actions - new_position)
        if self.episode_step >= self.max_steps:
            done = True
        self.state = [new_position, new_change, actions, reward, self.episode_step]
        return self._get_obs(), reward, done, self.distance

    def _get_obs(self, params = None):
        previous_position, change, previous_action, previous_reward, time = self.state
        return np.array([previous_position, change, previous_action, previous_reward, time*1e-3])

    def reset(self):
        self.true_poses = []
        self.predicted_poses = []
        self.goal_poses = []
        self.distance = 0.0
        self.initial_pose = (np.random.rand()-0.5) * 2 * self.max_bound
        self.move_multiplier = 3.0
        self.episode_step = 0
        if self.fixed_prob:
            self.offset = 0.0
        else:
            self.offset = (np.random.rand()-0.5) * 2.0 * 5.0
        self.state = [self.initial_pose, 0.0, 0.0, 0.0, self.episode_step]
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

        gt, pred, goal = self.true_poses[-1], self.predicted_poses[-1], self.goal_poses[-1]
        pixel = int(((gt + self.max_bound) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 255, 0), False, points, 5)
        pixel = int(((goal + self.max_bound) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 255), False, points, 5)
        pixel = int(((goal + self.max_bound +2.0) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points, 5)
        pixel = int(((goal + self.max_bound-2.0) / (self.max_bound * 2)) * resolution[0]) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points, 5)
        pixel = int(((pred + self.max_bound) / (self.max_bound * 2)) * resolution[0])
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height - 100)]
        pygame.draw.lines(self.screen, (255, 0, 0), False, points, 5)


        cur_time = time.time()
        pygame.display.update()
        while not (cur_time - self.previous_time >= self.dt):
            cur_time = time.time()
        self.previous_time = cur_time