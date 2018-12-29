import numpy as np
import pygame
import sys
import time

#TODO : Run during night and then test with **3 or ** 5 !!
class MyGymPendulum():
    def __init__(self, add_step = True, mass = 1.0, length = 1.0, add_speed = True, variable_mass = False, variable_reference = False):
        self.variable_mass = variable_mass
        self.max_mass = 100
        self.max_speed = 8
        self.max_torque = 2. * self.max_mass
        self.passed_mass = mass
        self.mass = mass
        self.length = length
        self.dt = 0.05
        self.act_dim = 1
        self.action_range = [-self.max_torque, self.max_torque]
        self.obs_dim = 4  # sin(theta), cos(theta), theta_dot
        self.nofb_obs_dim = 2
        # self.obs_dim = 1  # sin(theta), cos(theta), theta_dot
        if add_step:
            self.obs_dim += 1
            self.nofb_obs_dim += 1
        if add_speed:
            self.obs_dim += 1
            self.nofb_obs_dim += 1
        self.add_step = add_step
        self.add_speed = add_speed
        self.reference = 0
        self.variable_reference = variable_reference
        self.first_render = True

    def take_step(self, actions):
        th, th_dot, cur_step, old_action, old_reward = self.state

        g = 10.
        m = self.mass
        l = self.length
        dt = self.dt
        actions *= self.max_mass
        actions = np.clip(actions, -self.max_torque, self.max_torque)[0]
        self.previous_action = actions
        reward_th = th + self.reference
        reward = self.angle_normalize(reward_th)**2 + .1*th_dot**2 + 0.001*((actions/self.max_mass)**2)
        new_th_dot = th_dot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*actions)*dt
        new_th = th + new_th_dot * dt
        new_th_dot = np.clip(new_th_dot, -1.5 * self.max_speed * self.max_mass / self.mass, 1.5 * self.max_speed * self.max_mass / self.mass)
        cur_step += self.dt
        self.episode_step += 1
        self.state = (np.array([new_th, new_th_dot, cur_step, actions, reward]))
        done = self.episode_step >= 200
        # if np.abs(new_th_dot) >= 1.4 * self.max_speed * self.max_mass / self.mass:
        #     done = True
        #     reward = 2000
        return self._get_obs(), -reward, done, -reward

    def angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi))-np.pi)

    def _get_obs(self):
        theta, theta_dot, step, prev_action, prev_reward = self.state
        to_return = np.array([np.cos(theta), np.sin(theta)])
        # to_return = np.array([self.angle_normalize(theta)])
        if self.add_speed:
            to_return = np.concatenate([to_return, [theta_dot]])
        if self.add_step:
            to_return = np.concatenate([to_return, [step]])
        to_return = np.concatenate([to_return, [prev_action/self.max_mass, prev_reward]])
        return to_return

    def reset(self, params = None):
        if self.variable_reference:
            up = np.random.rand() < 0.5
            if up:
                self.reference = (np.random.rand() - 0.5) * 2 * np.pi / 4
            else:
                self.reference = (np.random.rand() - 0.5) * 2 * np.pi / 4 + np.pi
        else:
            self.reference = 0.0
        if self.variable_mass:
            self.mass = np.random.rand() * self.max_mass + 1
        if not params == None:
            self.reference = params[0]

        rand_init = np.random.rand(2)*2-1.0
        max_init_values = np.array([np.pi, 1.0])
        init_values = rand_init * max_init_values
        self.episode_step = 0
        self.state = np.concatenate([init_values, [0.], [0.], [0.]])
        return self._get_obs()

    def render(self):
        resolution = (640,480)
        th, th_speed, step, prev_action, prev_reward = self.state
        th = self.angle_normalize(th)
        if self.first_render:
            self.first_render = False
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(resolution)
            self.previous_time = time.time()
            self.rotating_arrow = pygame.image.load('./ReinforcementLearning/Environments/Assets/RightTurningArrow.png')
        middle_screen = (resolution[0]/2, resolution[1]/2)
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(self.mass), False, (0, 0, 0))
        bar_color = (255,0,0)
        drawing_width = 7
        drawing_length = 200 * self.length
        max_rotation_arrow = 120
        upper = middle_screen
        bottom = (middle_screen[0]+drawing_length*np.sin(th), middle_screen[1]-drawing_length*np.cos(th))
        bottom_ref = (middle_screen[0]+drawing_length*np.sin(-self.reference),
                      middle_screen[1]-drawing_length*np.cos(-self.reference))
        rectangle = [upper, bottom]
        rectangle2 = [upper, bottom_ref]
        self.screen.fill((255,255,255))
        self.screen.blit(textsurface, (int(resolution[0] / 2 - 50), 0))
        pygame.draw.lines(self.screen, bar_color, True, rectangle, drawing_width)
        pygame.draw.lines(self.screen, (0,255,0), True, rectangle2, drawing_width)
        cur_time = time.time()

        size = int(max_rotation_arrow * self.previous_action/self.max_torque)
        rotating_arrow = self.rotating_arrow
        if size < 0:
            size = -size
            rotating_arrow = pygame.transform.flip(rotating_arrow, True, False)
        rotating_arrow = pygame.transform.scale(rotating_arrow, (size, size))
        self.screen.blit(rotating_arrow, (middle_screen[0]-size/2, middle_screen[1]-size/2))
        pygame.display.update()
        for event in pygame.event.get():
            pass
        while not(cur_time-self.previous_time >= self.dt):
            cur_time = time.time()
        self.previous_time = cur_time


