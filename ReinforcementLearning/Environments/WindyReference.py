import numpy as np
import pygame
import sys
import time

"""
Definition of the second benchmark's environment.
See ReinforcementLearning/Training/meta_rl_launcher for the parameters used.
"""

class WindyReference():
    def __init__(self, control_speed = False, intra_var_percentage = 0.0, fixed_position = False, fixed_reference = False, wind_half_cone = np.pi/4, wind_possible_dir = 2 * np.pi, wind_power = 0.0, max_steps = 2000):
        self.control_speed = control_speed
        self.fixed_position = fixed_position
        self.wind_half_cone = wind_half_cone
        self.wind_power = wind_power
        self.first_render = True
        self.intra_var_percentage = intra_var_percentage
        self.fixed_references = fixed_reference
        self.wind_possible_dir = wind_possible_dir
        np.random.seed(100)
        if not self.control_speed:
            self.act_dim = 1
        else:
            self.act_dim = 2

        self.nofb_obs_dim = 2     # pos_x, pos_y
        self.obs_dim = self.nofb_obs_dim + self.act_dim + 1 # actions and reward
        self.max_size = 1
        self.resolution = 640
        self.zone_size = 0.4
        self.default_speed = self.max_size/10.0
        self.max_steps = max_steps
        self.pixel_coord_ratio = 640 / (4*self.max_size)
        self.create_references()
        self.default_pos = self.create_pos()

    def random_range(self, range_max):
        return (np.random.rand()-0.5) * 2 * range_max

    def create_references(self):
        self.refs = []
        for r in range(1):
            created = False
            while not created:
                random_pos = [self.random_range(self.max_size), self.random_range(self.max_size)]
                distances = []
                for existing_ref in self.refs:
                    distances.append(self.euc_dist(random_pos, existing_ref))
                if True or distances == [] or np.min(distances) > 1*self.zone_size:
                    created = True
                    self.refs.append(random_pos)

    def create_pos(self):
        created = False
        while not created:
            pot_pos = [self.random_range(1.5 * self.max_size), self.random_range(1.5 * self.max_size)]
            distances = []
            for existing_ref in self.refs:
                distances.append(self.euc_dist(pot_pos, existing_ref))
            if np.min(distances) > 1.5*self.zone_size:
                created = True
        return pot_pos

    def euc_dist(self, ref1, ref2):
        return np.sqrt(np.square(ref1[0]-ref2[0]) + np.square(ref1[1]-ref2[1]))

    def test_show(self):
        self.reset()
        done = False
        while not done:
            _,_, done,_ = self.take_step([0.0, 0.0])
            self.render()

    def take_step(self, actions):
        pos, timestep, prev_action, prev_reward = self.state
        self.dt = 0.1
        self.episode_step += 1

        if self.episode_step in self.pos_switches[0]:
            self.create_references()
        if self.episode_step in self.pos_switches[1]:
            self.wind_main_direction = np.random.rand() * self.wind_possible_dir

        direction = actions[0]
        direction *= np.pi
        direction = self.angle_normalize(direction)
        if self.control_speed:
            move = actions[1]
            move = np.clip(move, 0.0, 5.0)
        else:
            move = 2.5

        distances_to_references = []
        for r in self.refs:
            distances_to_references.append(self.euc_dist(pos, r))
        self.distances_to_references = distances_to_references
        minimum_distance = np.min(distances_to_references)
        reward = -2
        if minimum_distance <= self.zone_size:
            reward = 100.0
            # ADDED FOR INTERNAL TEST
            if np.random.rand() < self.intra_var_percentage:
                self.create_references()
            if not self.fixed_position:
                new_pos = self.create_pos()
            else:
                new_pos = self.default_pos
            if not self.fixed_references:
                self.create_references()
            self.change += 1
        else:
            self.wind_direction = self.wind_main_direction + (np.random.rand() - 0.5) * 2 * self.wind_half_cone
            wind_move = [np.cos(self.wind_direction) * move * self.default_speed * self.wind_power,
                         np.sin(self.wind_direction) * move * self.default_speed * self.wind_power]
            action_movement = [np.cos(direction) * move * self.default_speed,
                               np.sin(direction) * move * self.default_speed]
            new_pos = [pos[0] + wind_move[0] + action_movement[0],
                       pos[1] + wind_move[1] + action_movement[1]]
        new_posx = new_pos[0]
        new_posy = new_pos[1]
        if new_posx > self.max_size * 2:
            new_posx -= 4 * self.max_size
        elif new_posx < -2 * self.max_size:
            new_posx += 4 * self.max_size
        if new_posy > self.max_size * 2:
            new_posy -= 4 * self.max_size
        elif new_posy < -2 * self.max_size:
            new_posy += 4 * self.max_size

        done = False
        if self.episode_step >= self.max_steps:
            done = True
        if not self.control_speed:
            self.state = [[new_posx, new_posy], self.episode_step, [direction], reward]
        else:
            self.state = [[new_posx, new_posy], self.episode_step, [direction, move], reward]
        return self._get_obs(), reward, done, self.change

    def angle_normalize(self, x):
        if x > 0:
            return x % (2*np.pi)
        else:
            return 2*np.pi - (np.abs(x) % (2 * np.pi))

    def _get_obs(self):
        pos, timestep, prev_action, prev_reward = self.state
        refs = np.array(self.refs)
        self.references_dif = np.concatenate([[el[0]-pos[0],el[1]-pos[1]] for el in refs])
        self.to_show_ref_dif = refs
        to_concat_obs = self.references_dif
        return np.concatenate([to_concat_obs, prev_action, [prev_reward]])

    def reset(self, params = None):
        self.pos_switches = params
        if params == None:
            self.pos_switches = [[],[]]
        self.switches = params
        self.change = 0
        self.episode_step = 0
        self.wind_main_direction = np.random.rand() * self.wind_possible_dir
        self.create_references()
        self.stopped = -1
        position = self.create_pos()
        if self.control_speed:
            self.state = [position, 0, [0.0, 0.0], 0.0]
        else:
            self.state = [position, 0, [0.0], 0.0]
        if not params == None:
            pass
        return self._get_obs()

    def map_to_screen(self, x, y):
        point = (int((x+2*self.max_size) * self.resolution/(4*self.max_size)),
                 self.resolution - int((y+2*self.max_size) * self.resolution/(4*self.max_size)))
        return point

    def render(self):
        pos, timestep, prev_action, prev_reward = self.state
        resolution = (self.resolution, self.resolution)
        if self.first_render:
            self.first_render = False
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(resolution)
            self.previous_time = time.time()
        self.screen.fill((255,255,255))
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(prev_action), False, (0, 0, 0))
        self.screen.blit(textsurface, (int(resolution[0] / 2), 0))
        textsurface2 = myfont.render(str(self.references_dif), False, (0, 0, 0))
        self.screen.blit(textsurface2, (int(resolution[0] / 2) - 200, 40))
        textsurface3 = myfont.render(str(self.change), False, (0, 0, 0))
        self.screen.blit(textsurface3, (int(resolution[0] / 2) - 100, 120))

        for r in self.refs:
            if r[-1]:
                color = (0,255,0)
            else:
                color = (255,0,0)
            points = self.map_to_screen(r[0],r[1])
            pygame.draw.circle(self.screen, color, points, 3, 3)
            pygame.draw.circle(self.screen, color, points, int(self.zone_size*self.pixel_coord_ratio), 3)
        points = self.map_to_screen(pos[0],pos[1])
        pygame.draw.circle(self.screen, (0,0,255), points, 3, 3)

        action_points = [self.map_to_screen(pos[0], pos[1]),
                         self.map_to_screen(pos[0] + np.cos(prev_action[0]) * 0.4,
                                            pos[1] + np.sin(prev_action[0]) * 0.4)]
        pygame.draw.lines(self.screen, (0,0,0), False, action_points, 2)

        action_points = [self.map_to_screen(pos[0], pos[1]),
                         self.map_to_screen(pos[0] + np.cos(self.wind_main_direction) * 0.4 * self.wind_power,
                                            pos[1] + np.sin(self.wind_main_direction) * 0.4 * self.wind_power)]
        pygame.draw.lines(self.screen, (0, 255, 255), False, action_points, 2)

        action_points = [self.map_to_screen(pos[0], pos[1]),
                         self.map_to_screen(pos[0] + np.cos(self.wind_main_direction-self.wind_half_cone) * 0.4 * self.wind_power,
                                            pos[1] + np.sin(self.wind_main_direction-self.wind_half_cone) * 0.4 * self.wind_power)]
        pygame.draw.lines(self.screen, (0, 120, 159), False, action_points, 2)

        action_points = [self.map_to_screen(pos[0], pos[1]),
                         self.map_to_screen(pos[0] + np.cos(self.wind_main_direction+self.wind_half_cone) * 0.4 * self.wind_power,
                                            pos[1] + np.sin(self.wind_main_direction+self.wind_half_cone) * 0.4 * self.wind_power)]
        pygame.draw.lines(self.screen, (0, 120, 159), False, action_points, 2)

        action_points = [self.map_to_screen(pos[0], pos[1]),
                         self.map_to_screen(pos[0] + np.cos(self.wind_direction) * 0.4 * self.wind_power,
                                            pos[1] + np.sin(self.wind_direction) * 0.4 * self.wind_power)]
        pygame.draw.lines(self.screen, (120, 120, 0), False, action_points, 2)

        cur_time = time.time()
        pygame.display.update()
        for event in pygame.event.get():
            pass
        while not (cur_time - self.previous_time >= 1.*self.dt):
            cur_time = time.time()
        self.previous_time = cur_time
