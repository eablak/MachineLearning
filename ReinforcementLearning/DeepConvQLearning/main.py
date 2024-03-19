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
import matplotlib.pyplot as plt

TOTAL_TrainTime = 100000

IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

def ProcessGameImage(RawImage): # Image preprocessing
    # 420x400 olan image'leri eğitebilmek için 40x40'a çevir
    
    GreyImage = skimage.color.rgb2gray(RawImage) # renkliden siyaha çevir
    
    CroppedImage = GreyImage[0:400, 0:400] # önce 400x400'e çevir
    
    ReducedImage = skimage.transform.resize(CroppedImage, (IMGHEIGHT, IMGWIDTH))# 40x40 resize
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range=(0,255)) # rescale_insensity
    
    ReducedImage = ReducedImage / 128 # normalize
    
    return ReducedImage

def TrainExperiment():
    
    TrainHistory = []

    TheGame = DCQL_Pong.PongGame()
    TheGame.InitialDisplay()
    
    TheAgent = DCQL_Agent.Agent()
    
    # game ve agent arasındaki ilk bağlantı agent'in action yapmasıyla gerçekleşir o yüzden initial bi action yap
    BestAction = 0
    
    [initialScore, initialScreenImage] = TheGame.PlayNextMove(BestAction) # gelen initialScreenImage -> preprocessing
    initialGameImage = ProcessGameImage(initialScreenImage) # bir gameImage geliyo ama nn'u eğitirken ard arda 4 resim veriyoruz.
    
    GameState = np.stack((initialGameImage, initialGameImage, initialGameImage, initialGameImage), axis=2) # 4 tane arka arkaya birleştir ki initialize edebileceğin resimlerin olsun
    GameState = GameState.reshape(1,GameState.shape[0], GameState.shape[1], GameState.shape[2]) # 1x40x40x4 (keras) 40x40'lık 4 resim
    
    
    for i in range(TOTAL_TrainTime):
        
        BestAction = TheAgent.FindBestAction(GameState) # action bul
        [ReturnScore, NewScreenImage] = TheGame.PlayNextMove(BestAction) # action'u işle
        
        NewGameImage = ProcessGameImage(NewScreenImage) # gelen yeni screenImage'i preprocessing
        NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0], NewGameImage.shape[1], 1) # 1x40x40x1'lik resmini daha önceki 3 resimle birleştir
        
        # asıl train edeceğin yeni resimlerin (yani yeni gelen 1 + eski 3)
        NextState = np.append(NewGameImage, GameState[:,:,:,:3], axis=3) # axis=3 => 3.sünde birleştirme yap
        
        TheAgent.CaptureSample((GameState, BestAction, ReturnScore, NextState))
        
        TheAgent.Process()
        
        GameState = NextState
        
        if i % 250 == 0:
            print("Train time: ",i, "game score: ",TheGame.GScore)
            TrainHistory.append(TheGame.GScore)
        
TrainExperiment()