# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:18:03 2024

@author: esraablak
"""

import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    
    def __init__(self, env):
        # hyperparameter
        
        # build model
        self.state_size = env.observation_space.shape[0] # nn input layer !
        self.action_size = env.action_space.n # nn output layer !
        
        # replay
        self.gamma = 0.95 # future reward
        self.learning_rate = 0.01
        
        # adaptiveEGreedy
        self.epsilon = 1 # explore
        self.epsilon_decay = 0.995 # her iterasyonda epsilon_decay kadar explore'u azalt (exploitation'a kaymaya başla)
        self.epsilon_min = 0.01 # explore için min / treshold
        
        # remember
        self.memory = deque(maxlen=1000) # dolduğunda FIFO
        
        self.model = self.build_model() # agent / nn
    
    def build_model(self):
        
        # neural network for deep q learning
        
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation="tanh")) # hidden layer
        model.add(Dense(self.action_size, activation="linear")) # output layer
        # loss / optimizer
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # training / backward update / learning
        
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for (state, action, reward, next_state, done) in minibatch:
            if done: # oyun bittiği (env'daki başarısızlık durumu True) için  s' yok, alabileceğin tek şey reward
                target = reward
            else: # 
                # amax => birden fazla liste yan yana olduğu için önce hepsini birleştirir (flatted) sonra max'ı bulur indexini verir
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target # loss
            self.model.fit(state, train_target, verbose=0)
            
    
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    

if __name__ == "__main__":
    
    # initalize environment and agent
    env = gym.make("CartPole-v1")
    agent = DQLAgent(env)
    
    batch_size = 16
    episodes = 100
    for episode in range(episodes):
        
        # initalize environment
        state = env.reset()
        #state = state[0]
        state = np.reshape(state,[1,4])
        
        time = 0 # living penalty
        while True:
            
            # action
            action = agent.act(state) # select and action
            
            # process / step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1,4])
            
            
            # return değerleri depola (s,a,s',r) / remember / storage
            agent.remember(state, action, reward, next_state, done)
            
            # update state
            state = next_state
            
            # replay / geçmiş tecrübelerini kullan
            agent.replay(batch_size)
            
            # adjust epsilon
            agent.adaptiveEGreedy()
            
            
            time += 1
            
            if done:
                print("Episode: {}, time: {}".format(episode,time))
                break
            
# %% eğitilmiş modelini test et

import time

trained_model = agent
state = env.reset() # hiç görmediği yeni env aç
state = state[0]
state = np.reshape(state, [1,4])
time_t = 0

while True:
    
    env.render() # env'i göster
    
    action = trained_model.act(state)
    
    next_state, reward, done, info, _ = env.step(action)
    next_state = np.reshape(next_state, [1,4])
    
    state = next_state
    
    time_t += 1
    
    print(time_t)
    
    time.sleep(0.4)
    
    if done:
        break
print("Done")