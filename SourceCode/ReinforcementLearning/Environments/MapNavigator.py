import numpy as np
import pygame
import sys
import time

class MapNavigator():
    def __init__(self, fixed_reference = True, varying_ref_freq = 0.05, split_reference = True, fixed_wind = False,
                 varying_repulsion = False):
        self.fixed_reference = fixed_reference
        self.varying_ref_freq = varying_ref_freq
        self.first_render = True
        self.act_dim = 2
        self.nofb_obs_dim = 3 # pos_x, pos_y, timestep
        self.obs_dim = self.nofb_obs_dim + 3 # actions and reward
        self.split_reference = split_reference
        self.fixed_wind = fixed_wind
        if self.split_reference:
            self.obs_dim += 2
            self.nofb_obs_dim += 2
        self.max_size = 1
        self.resolution = 640
        self.zone_size = 0.075
        self.varying_repulsion = varying_repulsion

        self.max_steps = 500

    def test_show(self):
        self.reset()
        done = False
        while not done:
            _,_, done,_ = self.take_step([0.0, 0.0])
            self.render()

    def take_step(self, actions):
        posx, posy, refx, refy, timestep, prev_dir, prev_move, prev_reward = self.state
        self.dt = 0.1
        self.episode_step += 1
        direction = actions[0]
        direction*= 1 * np.pi
        direction = np.clip(direction, -2*np.pi, 2*np.pi)
        move = actions[1]
        move = np.clip(move, 0.0, 1.0)
        distance_to_goal = np.sqrt(np.square(refy-posy) + np.square(refx-posx))
        self.distance += distance_to_goal
        if self.repulsion:
            reward = distance_to_goal
        else:
            reward = -distance_to_goal


        new_posx = posx + np.cos(direction + self.wind) * move * self.dt
        new_posy = posy + np.sin(direction + self.wind) * move * self.dt
        new_posx = np.clip(new_posx, -self.max_size*2, self.max_size*2)
        new_posy = np.clip(new_posy, -self.max_size*2, self.max_size*2)
        if np.random.rand() < self.varying_ref_freq and not self.fixed_reference:
            new_refx = (np.random.rand()-0.5) * 2 * 4 / 5 * self.max_size
            new_refy = (np.random.rand()-0.5) * 2 * 4 / 5 * self.max_size
            if self.varying_repulsion:
                self.repulsion = np.random.rand() < 0.5
        else:
            new_refx = refx
            new_refy = refy
        done = False
        if self.episode_step >= self.max_steps:
            done = True
        self.state = [new_posx, new_posy, new_refx, new_refy, self.episode_step, direction, move, reward]
        return self._get_obs(), reward, done, self.distance

    def _get_obs(self):
        posx, posy, refx, refy, timestep, prev_dir, prev_move, prev_reward = self.state
        if self.split_reference:
            return np.array([posx, posy, refx, refy, timestep*1e-1, prev_dir, prev_move, prev_reward])
        else:
            return np.array([refx-posx, refy-posy, timestep*1e-1, prev_dir, prev_move, prev_reward])

    def reset(self, params = None):
        self.change = 0
        self.distance = 0.0
        self.initial_refx = (np.random.rand() - 0.5) * 2 * 4 / 5 * self.max_size
        self.initial_refy = (np.random.rand() - 0.5) * 2 * 4 / 5 * self.max_size
        self.initial_posx = (np.random.rand() - 0.5) * 2 * self.max_size
        self.initial_posy = (np.random.rand() - 0.5) * 2 * self.max_size
        self.episode_step = 0
        self.repulsion = False
        if self.fixed_wind:
            self.wind = 0.0
        else:
            self.wind = (np.random.rand()-0.5) * 2.0 * np.pi/2
        if self.varying_repulsion:
            self.repulsion = np.random.rand() < 0.5

        self.state = [self.initial_posx, self.initial_posy, self.initial_refx,
                      self.initial_refy, 0, 0.0, 0.0, 0.0]
        if not params == None:
            pass
        return self._get_obs()

    def map_to_screen(self, x, y):
        point = (int((x+self.max_size) * self.resolution/(2*self.max_size)),
                 self.resolution - int((y+self.max_size) * self.resolution/(2*self.max_size)))
        return point

    def render(self):
        posx, posy, refx, refy, timestep, prev_action, prev_move, prev_reward = self.state
        resolution = (self.resolution, self.resolution)
        if self.first_render:
            self.first_render = False
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(resolution)
            self.previous_time = time.time()
        self.screen.fill((255,255,255))
        middle_screen_height = int(resolution[1] / 2)
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(prev_action), False, (0, 0, 0))
        self.screen.blit(textsurface, (int(resolution[0] / 2), 0))

        points = self.map_to_screen(refx, refy)
        if self.repulsion:
            pygame.draw.circle(self.screen, (0, 255, 255), points, 5, 5)
        else:
            pygame.draw.circle(self.screen, (0, 255, 0), points, 5, 5)

        points = self.map_to_screen(posx, posy)
        pygame.draw.circle(self.screen, (255, 0, 0), points, 5, 5)

        action_points = [self.map_to_screen(posx, posy),
                         self.map_to_screen(posx + np.cos(prev_action) * prev_move * 0.4,
                                            posy + np.sin(prev_action) * prev_move * 0.4)]
        pygame.draw.lines(self.screen, (0,0,255), False, action_points, 2)

        cur_time = time.time()
        pygame.display.update()
        for event in pygame.event.get():
            pass
        while not (cur_time - self.previous_time >= self.dt/10):
            cur_time = time.time()
        self.previous_time = cur_time