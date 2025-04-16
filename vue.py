
import pygame
import os

class SnakeVue:
    def __init__(self, width, height, scale):
        self.width, self.height, self.scale = width, height, scale
        pygame.init()
        pygame.display.set_caption('Snake')
        self.game_window = pygame.display.set_mode((width*scale, height*scale))
        self.sheet = pygame.image.load("snake.png").convert()
        self.extractSprites()

    def extractSprites(self):
        self.images_body = []
        self.images_head = []
        self.images_queue = []
        for idx, pos in enumerate([(1,0), (0, 0), (0,1), (0,2), (1,2), (2,2)]):
            rect = pygame.Rect(64*pos[1], 64*pos[0], 64, 64)
            image = pygame.Surface(rect.size).convert()
            image.blit(self.sheet, (0, 0), rect)
            self.images_body.append(image)
        for idx, pos in enumerate([(0,3), (1, 4), (1,3), (0,4)]):
            rect = pygame.Rect(64*pos[1], 64*pos[0], 64, 64)
            image = pygame.Surface(rect.size).convert()
            image.blit(self.sheet, (0, 0), rect)
            self.images_head.append(image)
        for idx, pos in enumerate([(2,3), (3, 4), (3,3), (2,4)]):
            rect = pygame.Rect(64*pos[1], 64*pos[0], 64, 64)
            image = pygame.Surface(rect.size).convert()
            image.blit(self.sheet, (0, 0), rect)
            self.images_queue.append(image)
        rect = pygame.Rect(0, 64*3, 64, 64)
        self.image_fruit = pygame.Surface(rect.size).convert()
        self.image_fruit.blit(self.sheet, (0, 0), rect)
        colorkey = self.image_fruit.get_at((0,0))
        self.image_fruit.set_colorkey(colorkey, pygame.RLEACCEL)

    def displayGame(self, game):
        color = pygame.Color(0,0,0)
        pygame.draw.rect(self.game_window, color, pygame.Rect(0, 0, self.width*self.scale, self.height*self.scale))
        self.game_window.blit(self.image_fruit, (game.fruit[0]*self.scale, game.fruit[1]*self.scale))
        for idx, body in enumerate(game.serpent):
            coord = (body[0]*self.scale, body[1]*self.scale)
            if idx==0:
                self.game_window.blit(self.images_head[game.direction], coord)
                continue
            if idx==len(game.serpent)-1:
                if game.serpent[-2][0]==body[0]-1:
                    self.game_window.blit(self.images_queue[2], coord)
                elif game.serpent[-2][0]==body[0]+1:
                    self.game_window.blit(self.images_queue[3], coord)
                elif game.serpent[-2][1]==body[1]-1:
                    self.game_window.blit(self.images_queue[0], coord)
                else:
                    self.game_window.blit(self.images_queue[1], coord)
                continue
            if (game.serpent[idx+1][0]==game.serpent[idx][0]+1 and game.serpent[idx-1][1]==game.serpent[idx][1]-1) or (game.serpent[idx-1][0]==game.serpent[idx][0]+1 and game.serpent[idx+1][1]==game.serpent[idx][1]-1):
                self.game_window.blit(self.images_body[0], coord)
            elif (game.serpent[idx+1][1]==game.serpent[idx][1]+1 and game.serpent[idx-1][0]==game.serpent[idx][0]+1) or (game.serpent[idx-1][1]==game.serpent[idx][1]+1 and game.serpent[idx+1][0]==game.serpent[idx][0]+1):
                self.game_window.blit(self.images_body[1], coord)
            elif (game.serpent[idx+1][1]==game.serpent[idx][1]+1 and game.serpent[idx-1][0]==game.serpent[idx][0]-1) or (game.serpent[idx-1][1]==game.serpent[idx][1]+1 and game.serpent[idx+1][0]==game.serpent[idx][0]-1):
                self.game_window.blit(self.images_body[3], coord)
            elif (game.serpent[idx+1][0]==game.serpent[idx][0]-1 and game.serpent[idx-1][1]==game.serpent[idx][1]-1) or (game.serpent[idx-1][0]==game.serpent[idx][0]-1 and game.serpent[idx+1][1]==game.serpent[idx][1]-1):
                self.game_window.blit(self.images_body[5], coord)
            elif (game.serpent[idx+1][0]==game.serpent[idx][0]-1) or (game.serpent[idx-1][0]==game.serpent[idx][0]-1):
                self.game_window.blit(self.images_body[2], coord)
            else:
                self.game_window.blit(self.images_body[4], coord)
        pygame.display.set_caption(f'Score = {len(game.serpent)}')
        pygame.display.update()