import pygame
import random
import numpy as np
import math

class Player:

    polygon = [[0, -20], [-8, 8], [0,0], [8, 8]]
    linear_vel = 300
    rot_vel = math.pi / 2

    def __init__(self):
        self.x = random.randint(0, 1280)
        self.y = random.randint(0, 720)
        self.rot = 0

    def draw(self, screen):
        rotmat = [[math.cos(self.rot), -math.sin(self.rot)],
                  [math.sin(self.rot), math.cos(self.rot)]]

        points = np.matmul(self.polygon, rotmat) + [self.x,self.y]

        pygame.draw.polygon(screen, "yellow", points, 1)

    def update(self, dt):
        keys = pygame.key.get_pressed()

        dir_x = -math.sin(self.rot)
        dir_y = -math.cos(self.rot)

        if keys[pygame.K_w]:
            self.x += dir_x * self.linear_vel * dt
            self.y += dir_y * self.linear_vel * dt
        if keys[pygame.K_s]:
            self.x -= dir_x * self.linear_vel * dt
            self.y -= dir_y * self.linear_vel * dt

        if self.x > 1280:
            self.x = 1280
        if self.y > 720:
            self.y = 720
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0

        if keys[pygame.K_a]:
            self.rot += self.rot_vel * dt
        if keys[pygame.K_d]:
            self.rot -= self.rot_vel * dt

        if self.rot < 0:
            self.rot += math.pi * 2
        elif self.rot > math.pi * 2:
            self.rot -= math.pi * 2


class Sim:

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.dt = 0

        self.player = Player()

    def __events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True

    def __sim(self):

        self.player.update(self.dt)

    def __draw(self):
        self.screen.fill("black")

        self.player.draw(self.screen)

        pygame.display.flip()

    def run(self):
        while self.__events():
            self.__sim()

            self.__draw()

            self.dt = self.clock.tick(60) / 1000

