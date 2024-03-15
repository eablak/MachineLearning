# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:58:24 2024

@author: esraablak
"""
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DCQAgent:
    
    def __init__(self):
        
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.epsilon = 1
        self.epsilon_decay = 0.9903
        self.epsilon_min = 0.01
        
        self.gamma = 0.99
        self.learning_rate = 0.0001
        
        self.memory = deque(maxlen=4000)
        self.model = self.build_model()
        self.target_model = self.build_model()
    
    def build_model(self):
        
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # vectorized replay method
        if len(agent.memory) < batch_size:
            return
        
        # vectorized method for experience replay
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:,4] == False) # oyunun bitmediği örnekler
        y = np.copy(minibatch[:,2]) # reward'lar
        
        if len(not_done_indices[0] > 0):
            predict_sprime = self.model.predict(np.vstact(minibatch[:,3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:,3]))
            
            # ---
            y[not_done_indices] += np.multiply(self.gamma, predict_sprime_target[not_done_indices, np.argmax(predict_sprime[not_done_indices,:][0], axis=1)][0])
            
        actions = np.array(minibatch[:,1],dtype=int)
        y_target = self.nodel.predict(np.vstack(minibatch[:,0]))
        y_target[range(batch_size),actions] = y
        self.model.fit(np.vstack(minibatch[:,0]), y_target, epochs=1, verbose=0)
        
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())
    
    
if __name__ == "__main__":
    pass

    # initilaze gym env and agent
    env = gym.make("LunarLander-v2")
    agent = DCQAgent(env)
    state_number = env.observation_space.shape[0]

    batch_size = 32
    episodes = 10000
    for episode in episodes:
            
        # initilaze environment
        state = env.reset()
        state = state[0]
        state = np.reshape(state, [1,state_number])
        
        total_reward = 0
        for time in range(1000):
            
            # act
            action = agent.act(state)
            
            # step
            next_state, reward, done, info, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_number])
            
            # remember / storage
            agent.remember(state, action, reward, next_state, done)
            
            # update state
            state = next_state
            
            # replay
            agent.replay(batch_size)
            
            total_reward += reward
            if done:
                agent.targetModelUpdate()
                " her iterasyonda modeli update etmek yerine sadece bölüm sonunda update et. bir süre sonra öğrenmeye başladığında bölüm daha erken bitiyor öğrenme öğrene / üzerine koyarak gidiyor"
                break
            
        # epsilon greedy
        agent.adaptiveEGreedy()
        print("Episode: {}, Reward: {}".format(episode, reward))