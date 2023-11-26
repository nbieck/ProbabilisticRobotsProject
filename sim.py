import pygame

class Sim:

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock

    def __events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True

    def __sim(self):
        pass

    def __draw(self):
        self.screen.fill("black")

        pygame.display.flip()

    def run(self):
        while self.__events():
            self.__sim()

            self.__draw()

            self.clock.tick(60)

