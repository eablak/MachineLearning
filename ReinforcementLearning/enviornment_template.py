# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:53:03 2024

@author: esraablak
"""

# pygame template
import pygame
import random

# window size
WIDTH = 360
HEIGHT = 360
FPS = 30 # how fast game is

# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,255)
BLUE = (0,0,255)

class Player(pygame.sprite.Sprite):
    # sprite for player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH/2
        self.rect.bottom = HEIGHT - 1
        self.speed_x = 0
        
    def update(self):
        self.speed_x = 0
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_LEFT]:
            self.speed_x = -4
        elif keystate[pygame.K_RIGHT]:
            self.speed_x = 4
        else:
            self.speed_x = 0
        
        self.rect.x += self.speed_x 
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)

# initalize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Game")
clock = pygame.time.Clock()

# sprite
all_sprite = pygame.sprite.Group()
player = Player()
all_sprite.add(player)

# game loop
running = True
while running:
    # keep loop runing at the right speed
    clock.tick(FPS)
    
    # process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    # update
    all_sprite.update()
    
    # draw / render (show)
    screen.fill(GREEN)
    all_sprite.draw(screen)
    
    # after drawing flip display
    pygame.display.flip()
    
    
pygame.quit()