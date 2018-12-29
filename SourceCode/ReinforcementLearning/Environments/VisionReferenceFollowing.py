import numpy as np
import pygame
import sys
import time

#TODO : HERE TIMESTEP IS NOT AN OBSERVATION ... NEED TO CHANGE ?
# TODO : SHOULDN T WE USE IMAGES ? BUT THEN, DO WE NEED CONVOLUTION ?
class VisionReferenceFollowing():
    def __init__(self, hidden_pourcentage, hidden_offset, angle_variation, relative_input = False,
                 ref_speed = 1.0):
        self.first_render = True
        self.relative_input = relative_input
        self.hidden_pourcentage = hidden_pourcentage
        self.hidden_offset = hidden_offset
        self.angle_variation = angle_variation
        self.ref_speed = ref_speed
        self.act_dim = 2
        self.resolution = 640
        if not relative_input:
            self.nofb_obs_dim = 4
        else:
            self.nofb_obs_dim = 2
        self.obs_dim = self.nofb_obs_dim + self.act_dim + 1 # reward
        self.max_bound = 5.0
        self.max_steps = 2000
        self.act_multiplicator = np.pi

    def angle_normalize(self, x):
        if x > 0:
            return x % (2 * np.pi)
        else:
            return 2 * np.pi - (np.abs(x) % (2 * np.pi))

    def test_show(self):
        self.reset()
        done = False
        while not done:
            _,_, done,_ = self.take_step(0.0)
            self.render()

    def reset(self, params = None):
        self.invisible = False
        self.move_angle = (np.random.rand()-0.5) * 2 * self.angle_variation
        self.episode_step = 0
        self.state = [[-self.max_bound,0.0],[-self.max_bound,0.0], 0, [0.0,0.0],0]
        self.invisble_length = (2 * self.max_bound) * self.hidden_pourcentage
        self.halth_ilength = self.invisble_length/2
        self.offset_value = self.hidden_offset * self.invisble_length
        self.start_invisble = -self.halth_ilength + self.offset_value
        self.end_invisible = self.halth_ilength + self.offset_value
        if not params == None:
            pass

        return self._get_obs()

    def _get_obs(self):
        ref_pos, agent_pos, timestep, prev_actions, prev_reward = self.state
        if self.relative_input:
            to_concat_obs = [ref_pos[0]-agent_pos[0], ref_pos[1]-agent_pos[1]]
            if self.invisible:
                to_concat_obs = [-100, -100]
        else:
            to_concat_obs = [ref_pos[0],ref_pos[1],agent_pos[0],agent_pos[1]]
            if self.invisible:
                to_concat_obs = [-100, -100, agent_pos[0], agent_pos[1]]

        return np.concatenate([to_concat_obs, prev_actions, [prev_reward]])

    def euc_dist(self, ref1, ref2):
        return np.sqrt(np.square(ref1[0]-ref2[0]) + np.square(ref1[1]-ref2[1]))

    def take_step(self, actions):
        ref_pos, agent_pos, timestep, prev_actions, prev_reward = self.state
        self.dt = 0.1
        self.episode_step += 1
        direction = actions[0]
        intensity = actions[1]
        practical_intensity = intensity + (np.random.rand()-0.5) * 2 * (np.maximum(0.0,intensity-self.ref_speed) ** 3)# The bigger move, the higher noise,
        direction *= self.act_multiplicator
        direction = self.angle_normalize(direction)

        new_refx = ref_pos[0] + self.ref_speed * self.dt * np.cos(self.move_angle)
        new_refy = ref_pos[1] + self.ref_speed * self.dt * np.sin(self.move_angle)

        new_agx = agent_pos[0] + practical_intensity * self.dt * np.cos(direction)
        new_agy = agent_pos[1] + practical_intensity * self.dt * np.sin(direction)
        reward = -(self.euc_dist([new_refx, new_refy],[new_agx, new_agy])**3)


        if new_refx >= self.max_bound:
            new_refx = -5.0
            new_agx = -5.0
            new_refy = 0.0
            new_agy = 0.0

        self.invisible = False
        if self.start_invisble <= new_refx and new_refx <= self.end_invisible:
            self.invisible = True
            reward = 0

        done = False
        if self.episode_step >= self.max_steps:
            done = True

        self.state = [[new_refx, new_refy], [new_agx, new_agy], self.episode_step, [direction, intensity], reward]
        return self._get_obs(), reward, done, reward

    def map_to_screen(self, x, y):
        point = (int((x + 1.5 * self.max_bound) * self.resolution / (3 * self.max_bound)),
                 self.resolution - int((y + 1.5 * self.max_bound) * self.resolution / (3 * self.max_bound)))
        return point

    def render(self):
        ref_pos, agent_pos, timestep, prev_actions, prev_reward = self.state
        resolution = (self.resolution, self.resolution)
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
        textsurface = myfont.render(str(1.0), False, (0, 0, 0))
        self.screen.blit(textsurface, (int(resolution[0] / 2), 0))

        if not self.invisible:
            points = self.map_to_screen(ref_pos[0], ref_pos[1])
            pygame.draw.circle(self.screen, (255, 0, 0), points, 3, 3)
        else:
            points = self.map_to_screen(ref_pos[0], ref_pos[1])
            pygame.draw.circle(self.screen, (255, 255, 0), points, 3, 3)

        points = self.map_to_screen(agent_pos[0], agent_pos[1])
        pygame.draw.circle(self.screen, (0, 0, 255), points, 3, 3)


        cur_time = time.time()
        pygame.display.update()
        for event in pygame.event.get():
            pass
        while not (cur_time - self.previous_time >= self.dt):
            cur_time = time.time()
        self.previous_time = cur_time