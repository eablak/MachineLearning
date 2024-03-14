# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:12:21 2024

@author: esraablak
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3").env

# Q-Table

# Hyperparameter

# Plotting Metrix

episode_number = 10000 # ajan kaç kez eğitilecek

for i in range(1,episode_number):
    
    # initalize environment
    
    # ajanı 1 episode için eğitmeye başla (yani ajan yanlış yerde müşteriyi bırakana kadar)
    while True:
        
        # expliot vs explore to find action -> epsilon
        
        # action process and take reward / observation
        
        # q-learning function
        
        # update q-table
        
        # update state
        
        # find wrong dropout
        
        if done:
            break