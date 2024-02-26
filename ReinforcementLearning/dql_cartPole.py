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
    def __inti__(self, env):
        # parameter / hyperparameter
        pass
    
    def build_model(self):
        # neural network for deep q learning
        pass
    
    def remember(self, state, action, reward, next_state, done):
        # storage
        pass
    
    def act(self, state):
        # acting
        pass
    
    def replay(self, batch_size):
        # training
        pass

    def adaptiveEGreedy(self):
        pass

if __name__ == "__main__":
    
    # initilaze env and agent
    
    episodes = 100
    for episode in range(episodes):
        
        # initilaze environment
        
        while True:
            
            # act
            
            # step
            
            # remember
            
            # update state
            
            # replay
            
            # adjust epsilon
            
            if done:
                break