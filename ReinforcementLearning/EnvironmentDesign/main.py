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

class player
class enemy
player-enemy -> collide
class dql agent
class env
"""

# pygame template

import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from keras.models import load_model
import pickle
import asyncio

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
        
        self.radius = 10 # rect'i hayali bir daire ile sarar
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius) # bu daireyi boya tam çarpışmaları gör
        
        self.rect.centerx = WIDTH/2
        self.rect.bottom = HEIGHT - 1
        
        self.y_speed = 5
        self.speedx = 0
        
    def update(self, action):
        self.speedx = 0
        keystate = pygame.key.get_pressed()
        
        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -4
        elif keystate[pygame.K_RIGHT] or action == 1:
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

class Enemy(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.Surface((10,10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        
        self.radius = 5
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)
        
        self.rect.x = random.randrange(0,WIDTH - self.rect.width)
        self.rect.y = random.randrange(2,6)
        
        self.speedx = 0
        self.speedy = 3
    
    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.top > HEIGHT + 10: # en aşağıya geldiyse yukarı çıkar
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(2,6)
            self.speedy = 3
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
            

class DQLAgent:
    def __init__(self):
        self.state_size = 4 # distance [(player-m1x), (player-m1y), (player-m2x), (player-m2y)]
        self.action_size = 3
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen=1000)
        self.model_path = "kayit.keras"
        self.model = self.build_model()
        
        
    def build_model(self):
        
        global model
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # q_values
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for (state, action, reward, next_state, done) in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward
            else:
                target = reward  + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)
            
    def load_saved_model(self):
        self.model = load_model("rl_game.keras")
        # self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

    def save_model(self):
        model.save("rl_game.keras")
    
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
class Env(pygame.sprite.Sprite):
    # gym env için 2 temel method reset(initilaze), step. Bunlara ek run yaz dql burada çalışacak
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()

        self.player = Player()
        self.m1 = Enemy()
        self.m2 = Enemy()

        self.all_sprite.add(self.player)
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)

        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()
        
    def findDistance(self, a, b):
        return (a-b)

    def step(self, action):
        state_list = []
        
        # update
        self.player.update(action) # önceden action yoktu dql için action verip player içindeki update methodunu modifiye ettim
        self.enemy.update()
        
        # get coordinate -> kordinatlar farkına bakıp (findDistance) action'u ona göre seçecek
        next_player_state = self.player.getCoordinates()
        next_m1_state = self.m1.getCoordinates()
        next_m2_state = self.m2.getCoordinates()
        
        # find distance
        state_list.append(self.findDistance(next_player_state[0], next_m1_state[0])) # player ve enemy x
        state_list.append(self.findDistance(next_player_state[1], next_m1_state[1])) # y
        state_list.append(self.findDistance(next_player_state[0], next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m2_state[1]))
        
        return [state_list]
    
    # reset
    def initialStates(self): # yeni oyun için ajan hariç her şeyi yenile. Environment'i resetle
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()

        self.player = Player()
        self.m1 = Enemy()
        self.m2 = Enemy()

        self.all_sprite.add(self.player)
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)

        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        
        state_list = []
        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()
        m2_state = self.m2.getCoordinates()
        
        state_list.append(self.findDistance(player_state[0], m1_state[0]))
        state_list.append(self.findDistance(player_state[1], m1_state[1]))
        state_list.append(self.findDistance(player_state[0], m2_state[0]))
        state_list.append(self.findDistance(player_state[1], m2_state[1]))
        
        return [state_list]
        
    
    def run(self): # game loop buraya taşınır
        
        state = self.initialStates()
        running = True
        batch_size = 24
        
        while running:
            self.reward = 2
            
            # keep loop running at the right speed
            clock.tick(FPS)
            
            # process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
            # update
            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward
        
            hits = pygame.sprite.spritecollide(self.player, self.enemy, False, pygame.sprite.collide_circle)  # collide
            if hits:
                self.reward = -150
                self.total_reward += self.reward
                self.done = True
                running = False
                print("Total reward: ",self.total_reward)
            
            # storage
            self.agent.remember(state, action, self.reward, next_state, self.done)
            
            # update state
            state = next_state
            
            # replay
            #self.agent.replay(batch_size)

            # save model
            # self.agent.save_model()

            # load model
            self.agent.load_saved_model()
            
            # epsilon greedy
            self.agent.adaptiveEGreedy()
            
            screen.fill(GREEN)
            self.all_sprite.draw(screen)
            
            # after drawing / flip display
            pygame.display.flip() # boyadığını göster

    pygame.quit()

env = Env()
liste = []
t = 0
episodes = 1
for episode in range(episodes):

    t += 1
    print("Episode: ",t)
    liste.append(env.total_reward)

    # initalize pygame and create window
    pygame.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("RL Game")
    clock = pygame.time.Clock()
    
    env.run()

    pygame.display.quit()