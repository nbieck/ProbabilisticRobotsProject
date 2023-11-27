import pygame
import random
import numpy as np
import math
import icp


class Config:

    def __init__(self):

        self.vel = 300.
        self.angular_vel = math.pi
        self.sens_range = 200
        # noise takes std dev. set value using variance
        self.sens_noise = math.sqrt(40)
        self.landmarks = 50
        self.particles = 200
        # assumed noise in the movement model. As above, std dev
        self.vel_noise = math.sqrt(30)
        self.rot_vel_noise = math.sqrt(math.pi / 8)

class Player:

    polygon = [[0, -20], [-8, 8], [0,0], [8, 8]]

    def __init__(self, config):
        self.x = random.randint(0, 1280)
        self.y = random.randint(0, 720)
        self.rot = 0
        self.config = config

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
            self.x += dir_x * self.config.vel * dt
            self.y += dir_y * self.config.vel * dt
        if keys[pygame.K_s]:
            self.x -= dir_x * self.config.vel * dt
            self.y -= dir_y * self.config.vel * dt

        if self.x > 1280:
            self.x = 1280
        if self.y > 720:
            self.y = 720
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0

        if keys[pygame.K_a]:
            self.rot += self.config.angular_vel * dt
        if keys[pygame.K_d]:
            self.rot -= self.config.angular_vel * dt

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

    def __init__(self, player, config):
        self.hits = []
        self.player = player
        self.config = config

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 255), (self.player.x, self.player.y), self.config.sens_range, 1)
        for hit in self.hits:
            hit.draw(screen, self.player.x, self.player.y, self.player.rot)

    def update(self, landmarks):
        self.hits.clear()

        player_pos = np.array([self.player.x, self.player.y])

        for lm in landmarks:
            if np.linalg.norm(player_pos - [lm.x, lm.y]) < self.config.sens_range:
                offset = random.gauss(0, self.config.sens_noise)
                angle = random.uniform(0, math.pi * 2)

                noisy_pos = [lm.x, lm.y] + np.array([math.cos(angle), math.sin(angle)]) * offset

                diff = noisy_pos - player_pos

                self.hits.append(SensorHit(np.linalg.norm(diff), math.atan2(-diff[1], diff[0]) - self.player.rot))

class Particle:

    def make_rand():
        return Particle(np.array([random.uniform(0,1280), random.uniform(0,720)]), random.uniform(0, math.pi * 2))

    def copy(other):
        return Particle(np.copy(other.pos), other.rot)

    def __init__(self, pos, rot):
        self.pos = pos
        self.rot = rot

    def draw(self, screen):
        pygame.draw.circle(screen, (0,255,0), self.pos, 5, 1)
        pygame.draw.line(screen, (0,255,0), self.pos, self.pos + [5 * -math.sin(self.rot), -5 * math.cos(self.rot)])

class ParticleFilter:

    def __init__(self, config):
        self.config = config

        self.particles = []
        for _ in range(config.particles):
            self.particles.append(Particle.make_rand())

    def draw(self, screen):
        for particle in self.particles:
            particle.draw(screen)

    def prediction(self, dt):
        keys = pygame.key.get_pressed()
        for particle in self.particles:
            d = np.array([-math.sin(particle.rot), -math.cos(particle.rot)])

            vel_noise = random.gauss(0, self.config.vel_noise)
            rot_noise = random.gauss(0, self.config.rot_vel_noise)

            vel = self.config.vel + vel_noise
            rot_vel = self.config.angular_vel + rot_noise

            if keys[pygame.K_w]:
                particle.pos += d * vel * dt
            if keys[pygame.K_s]:
                particle.pos -= d * vel * dt

            if keys[pygame.K_a]:
                particle.rot += rot_vel * dt
            if keys[pygame.K_d]:
                particle.rot -= rot_vel * dt

            if particle.rot < 0:
                particle.rot += math.pi * 2
            elif particle.rot > math.pi * 2:
                particle.rot -= math.pi * 2

    def update(self, dt):
        self.prediction(dt)

        #correction

        #resampling

class Sim:

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.dt = 0

        self.player = None
        self.landmarks = []
        self.sensor = None
        self.config = Config()
        self.filter = None

    def __init(self):
        self.player = Player(self.config)
        self.sensor = Sensor(self.player, self.config)
        self.landmarks.clear()
        for _ in range(self.config.landmarks):
            self.landmarks.append(Landmark())
        self.filter = ParticleFilter(self.config)

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
        self.filter.update(self.dt)

    def __draw(self):
        self.screen.fill("black")

        for lm in self.landmarks:
            lm.draw(self.screen)
        self.sensor.draw(self.screen)
        self.filter.draw(self.screen)
        self.player.draw(self.screen)

        pygame.display.flip()

    def run(self):
        self.__init()

        while self.__events():
            self.__sim()

            self.__draw()

            self.dt = self.clock.tick(60) / 1000

