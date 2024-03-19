# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:17:49 2024

@author: esraablak
"""

import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D
from collections import deque

NBACTIONS = 3
# resimleri 40x40'larala reshasape et ve agent'e 4 resim yolla
IMGHEIGHT = 40
IMGHWIDTH = 40
IMGHISTORY = 4

OBSERVEPERIOD = 250 # training'e başlamadan experience replay'de koşturulacak resim sayısı. daha sonra training
GAMMA = 0.975
BATCH_SIZE = 64

ExpReplay_CAPACITY = 2000

class Agent:
    
    def __init__(self):
        self.model = self.createModel()
        self.ExpReplay = deque()
        self.steps = 0  # training step. epsilon greedy'i update edebilmek için
        self.epsilon = 1.0
    
    def createModel(self):
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=4, strides=(2,2), input_shape=(IMGHEIGHT,IMGHWIDTH, IMGHISTORY), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=4, strides=(2,2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=3, strides=(1,1), padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dense(units=NBACTIONS, activation="linear"))
        
        model.compile(loss="mse", optimizer="adam")
        
        return model
    
    def FindBestAction(self, s):
        if random.random() < self.epsilon or self.steps < OBSERVEPERIOD: # explore
            return random.randint(0, NBACTIONS-1 )
        else:
            qvalue = self.model.predict(s) # exploit
            bestA = np.argmax(qvalue)
            return bestA
    
    # experience replay'i doldur + epsilon greedy update
    def CaptureSample(self, sample):
        self.ExpReplay.append(sample)
        if len(self.ExpReplay) > ExpReplay_CAPACITY:
            self.ExpReplay.popleft()
        
        self.steps += 1
        self.epsilon = 1.0
        # belirli bir training steps'i geçtiyse explore azalt
        if self.steps > OBSERVEPERIOD:
            self.epsilon = 0.75
            if self.steps > 7000:
                self.epsilon = 0.5
            if self.steps > 14000:
                self.epsilon = 0.25
            if self.steps > 30000:
                self.epsilon = 0.15
            if self.steps > 45000:
                self.epsilon = 0.1
            if self.steps > 70000:
                self.epsilon = 0.05
        
    # experience replay dolmaya başladığı anda training'e başla
    def Process(self):
        if self.steps > OBSERVEPERIOD:
            mini_batch = random.sample(self.ExpReplay, BATCH_SIZE)
            batchlen = len(mini_batch)
            
            inputs = np.zeros((BATCH_SIZE, IMGHEIGHT, IMGHWIDTH, IMGHISTORY)) # agent'in bulunduğu state değerleri
            targets = np.zeros((inputs.shape[0], NBACTIONS)) # agent'in yapacağı action değerleri
            
            Q_sa = 0
            
            for i in range(batchlen):
                # samples -> s, a, r, s'
                state_t = mini_batch[i][0]
                action_t = mini_batch[i][1]
                reward_t = mini_batch[i][2]
                state_t1 = mini_batch[i][3] # next state
            
                inputs[i:i+1] = state_t
                targets[i] = self.model.predict(state_t)
                Q_sa = self.model.predict(state_t1) # next_state predict sonuçları
                
                if state_t1 is None:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA*np.max(Q_sa) # bellman equation
            
            self.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)