# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:07:18 2024

@author: esraablak
"""

import DCQL_Agent
import DCQL_Pong

import numpy as np
import skimage as skimage
import warnings
warnings.filterwarnings("ignore")
import matplotlib .pyplot as plt

TOTALTrainTime = 100000

IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

def ProcessGameImage(RawImage):
    
    GreyImage = skimage.color.rgb2gray(RawImage)
    
    CroppedImage = GreyImage[0:400,0:400]
    
    ReducedImage = skimage.transform.resize(CroppedImage, (IMGHEIGHT, IMGWIDTH))
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range=(0,255))
    
    ReducedImage = ReducedImage / 128
    
    return ReducedImage

def TrainExperiment():
    
    
    TrainHistory = []
    
    TheGame = DCQL_Pong.PongGame()
    TheGame.InitialDispilay()
    
    TheAgent = DCQL_Agent.Agent()
    BestAction = 0
    
    [initialScore, initialScreenImage] = TheGame.playNextMove(BestAction)
    initialScreenImage = ProcessGameImage(initialScreenImage)
    
    GameState = np.stack((initialScreenImage,initialScreenImage,initialScreenImage,initialScreenImage), axis=2)
    GameState = GameState.reshape(1, GameState.shape[0],GameState.shape[1],GameState.shape[2])
    
    for i in range(TOTALTrainTime):
        BestAction = TheAgent.findBestAction(GameState)
        [returnScore, newScreenImage] = TheGame.playNextMove(BestAction)
        
        newGameImage = ProcessGameImage(newScreenImage)
        newGameImage = newGameImage.reshape(1, newGameImage.shape[0],newGameImage.shape[1],1)
        
        nextState = np.append(newGameImage, GameState[:,:,:,:3], axis=3)
        
        TheAgent.CaptureSample((GameState, BestAction, returnScore, nextState))
        TheAgent.Process()
        
        GameState = nextState
        
        if i % 250 == 0:
            print("train time: ", i, "game score: ", TheGame.GScore)
            TrainHistory.append(TheGame.GScore)
            
TrainExperiment()