"""
player()
    init()
    update()
    
enemy()
    init()
    update()
    
dqlagent()
    init
    NN()
    remember()
    act()
    replay()
    Egreedy()

=> main class
env()
    init()
    step()
    reset()
    *run() 
    
RL GAME
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
GREEN = (0,255,0)
BLUE = (0,0,255)

class Player(pygame.sprite.Sprite): # sprite özelliklerini kullanabilmek için
    # sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self) # class içerisine yazdığın inheritance'ı sağlayabilmek için
        self.image = pygame.Surface((20,20)) # ekranda görünecek sprite'i oluştur
        self.image.fill(BLUE)
        self.rect = self.image.get_rect() # sprite'ını rectangle ile çerçevele. bu sana yeni methodalar sağlar
        self.rect.centerx = WIDTH/2
        self.rect.bottom = HEIGHT - 1
        self.y_speed = 5
        self.speedx = 0
        
    def update(self):
        self.speedx = 0
        keystate = pygame.key.get_pressed()
        
        if keystate[pygame.K_LEFT]:
            self.speedx = -4
        elif keystate[pygame.K_RIGHT]:
            self.speedx = 4
        else:
            self.speedx = 0
            
        self.rect.x += self.speedx
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
        
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)

# initalize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("RL Game")
clock = pygame.time.Clock()

# sprite
all_sprite = pygame.sprite.Group()
player = Player()
all_sprite.add(player)



# game loop
running = True
while running:
    # keep loop running at the right speed
    clock.tick(FPS)
    
    # process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False # sadece döngüden çıkarsın. initalize ettiğin game'i kapat
    
    # update
    all_sprite.update()
    
    # drow / render (show)
    screen.fill(GREEN)
    all_sprite.draw(screen)
    
    # after drawing / flip display
    pygame.display.flip() # boyadığını göster

pygame.quit()