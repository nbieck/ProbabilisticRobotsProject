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

class Landmark:

    size = 5

    def __init__(self):
        self.x = random.randint(0, 1280)
        self.y = random.randint(0, 720)

    def draw(self, screen):
        pygame.draw.line(screen, (255, 255, 255), (self.x - self.size, self.y - self.size), (self.x + self.size, self.y + self.size))
        pygame.draw.line(screen, (255, 255, 255), (self.x - self.size, self.y + self.size), (self.x + self.size, self.y - self.size))

class SensorHit:

    size = 3

    def __init__(self, dist, heading):
        self.dist = dist
        self.heading = heading

    def draw(self, screen, x, y, base_rot):

        angle = self.heading + base_rot
        dir_x = math.cos(angle)
        dir_y = -math.sin(angle)

        offset_x = dir_x * self.dist
        offset_y = dir_y * self.dist

        p_x = x + offset_x
        p_y = y + offset_y

        pygame.draw.line(screen, (0, 255, 255), (x, y), (p_x, p_y))
        pygame.draw.line(screen, (255, 0, 0), (p_x - self.size, p_y - self.size), (p_x + self.size, p_y + self.size))
        pygame.draw.line(screen, (255, 0, 0), (p_x - self.size, p_y + self.size), (p_x + self.size, p_y - self.size))


class Sensor:

    range = 200
    sigma = math.sqrt(40.)

    def __init__(self, player):
        self.hits = []
        self.player = player

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 255), (self.player.x, self.player.y), self.range, 1)
        for hit in self.hits:
            hit.draw(screen, self.player.x, self.player.y, self.player.rot)

    def update(self, landmarks):
        self.hits.clear()

        player_pos = np.array([self.player.x, self.player.y])

        for lm in landmarks:
            if np.linalg.norm(player_pos - [lm.x, lm.y]) < self.range:
                offset = random.gauss(0, self.sigma)
                angle = random.uniform(0, math.pi * 2)

                noisy_pos = [lm.x, lm.y] + np.array([math.cos(angle), math.sin(angle)]) * offset

                diff = noisy_pos - player_pos

                self.hits.append(SensorHit(np.linalg.norm(diff), math.atan2(-diff[1], diff[0]) - self.player.rot))

class Sim:

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.dt = 0

        self.player = None
        self.landmarks = []
        self.sensor = None

    def __init(self):
        self.player = Player()
        self.sensor = Sensor(self.player)
        self.landmarks.clear()
        for _ in range(50):
            self.landmarks.append(Landmark())

    def __events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            return False
        if keys[pygame.K_r]:
            self.__init()
        
        return True

    def __sim(self):

        self.player.update(self.dt)
        self.sensor.update(self.landmarks)

    def __draw(self):
        self.screen.fill("black")

        for lm in self.landmarks:
            lm.draw(self.screen)
        self.sensor.draw(self.screen)
        self.player.draw(self.screen)

        pygame.display.flip()

    def run(self):
        self.__init()

        while self.__events():
            self.__sim()

            self.__draw()

            self.dt = self.clock.tick(60) / 1000

