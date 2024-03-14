# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 08:02:12 2024

@author: esraablak
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3").env

# Q-Table
# column->action  row->state  / taxi için 6 action 500 state
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# Hyperparameter
alpha = 0.1 # learning rate
gamma = 0.9
epsilon = 0.1 # exploit vs explore (%10 explore)


# Plotting Metrix
reward_list = []
dropout_list = []


episode_number = 10000 # ajan kaç kez eğitilecek

for i in range(1,episode_number):
    
    # initalize environment
    state = env.reset()
    state = state[0] # yeni versiyonda hatayı önlemek için
    
    reward_count = 0
    dropouts = 0
    
    # ajanı 1 episode için eğitmeye başla (yani ajan yanlış yerde müşteriyi bırakana kadar)
    while True:
        
        # expliot vs explore to find action -> epsilon
        if random.uniform(0,1) < epsilon: #explore
            action = env.action_space.sample()
        else: # exploit (q-table'dan alıyosun -> bulunduğun state/columun'a gidip max değerli action'u alıyosun)
            action = np.argmax(q_table[state]) # return max values index
            
            
        # action process and take reward / observation
        next_state, reward, done, _, info = env.step(action)
        
        
        # q-learning function
        old_value = q_table[state,action] # old_value
        next_max = np.max(q_table[next_state]) # next_max
        
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # update q-table
        q_table[state,action] = next_value
        
        # update state
        state = next_state
        
        # find wrong dropout
        if reward == -10:
            dropouts += 1
        
        if done:
            break
        
        reward_count += reward
        
    if i % 10 == 0:
        dropout_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode {}, reward: {}, wrong dropout: {}".format(i,reward_count, dropouts))
        
# %% visualize

fig ,axs = plt.subplots(1,2)

axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.show()

# %% q-table

"""
Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
"""  

# taxi row, taxi column, passenger index, destionation
env.s = env.encode(0,0,3,4)
env.encode(4,4,4,3)