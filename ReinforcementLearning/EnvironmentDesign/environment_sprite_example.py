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
    
SPRITE: ekranda hareket eden objeler. bu sprite'lara işlem yaptırmak istediğinde / method çalıştırdığında belli bir yerden sonra karmaşıklaşır bunun önüne geçmek için sprite group kullanarak beraber hareket ettirirsin. 
sprite_group(player1, player2, player3)
sprite_group.draw(screen)
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
        self.rect.center = (WIDTH/2, HEIGHT/2)
        self.y_speed = 5
        # self.x_speed = 3
        
    def update(self):
        self.rect.y += self.y_speed
        # self.rect.x += self.x_speed
        if self.rect.bottom > HEIGHT - 200: # oyunlarda y ekseni ters (160'tan sonra yukarı çık)
            self.y_speed = -5
        if self.rect.top < 0: # oyundan yukarı çıkma
            self.y_speed = 5
        if self.rect.right > WIDTH:
            self.x_speed = -3
        if self.rect.left < 0:
            self.x_speed = 3


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