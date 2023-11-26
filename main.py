import pygame
from sim import Sim

def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Bayesian Filter")
    clock = pygame.time.Clock()

    sim = Sim(screen, clock)
    sim.run()


if __name__=="__main__":
    main()