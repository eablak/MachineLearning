# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:07:18 2024

@author: esraablak
"""

import sys
sys.path.append("C:/Users/ESRA  ABLAK/Desktop/pongGame")

import DCQL_Pong
import DCQL_Agent
import numpy as np
import skimage as skimage
import warnings
warnings.filterwarnings("ignore")

TOTAL_TrainTime = 100000

IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

def ProcessGameImage(RawImage): # 420'e 400 olan resimleri 40'a 40'a çevir sonra da normalize et.
    
    GreyImage = skimage.color.rgb2gray(RawImage) # renkliden siyah-beyaza çevir
    
    CroppedImage = GreyImage[0:400,0:400]
    
    ReducedImage = skimage.transform.resize(CroppedImage,(IMGHEIGHT,IMGWIDTH)) # boyut azalt
    
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range = (0,255)) # yeniden ölçeklendir
    
    ReducedImage = ReducedImage / 128
    
    return ReducedImage

def TrainExperiment():
    
    TrainHistory = []
    
    TheGame = DCQL_Pong.PongGame()
    
    TheGame.InitialDisplay()
    
    TheAgent = DCQL_Agent.Agent()
    
    BestAction = 0
    
    [InitialScore, InitialScreenImage] = TheGame.PlayNextMove(BestAction)
    InitialGameImage = ProcessGameImage(InitialScreenImage) # gelen resmi boyutlandır
    
    GameState = np.stack((InitialGameImage,InitialGameImage,InitialGameImage,InitialGameImage),axis = 2) # resimleri 4'lü aldığın için birleştir
    # biz 40x40x4 elde ettik ama keras 1x40x40x4 istiyor
    GameState = GameState.reshape(1, GameState.shape[0],GameState.shape[1],GameState.shape[2])
    
    for i in range(TOTAL_TrainTime):
        
        BestAction = TheAgent.FindBestAct(GameState)
        [ReturnScore, NewScreenImage] = TheGame.PlayNextMove(BestAction)
        
        NewGameImage = ProcessGameImage(NewScreenImage)
        
        NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0],NewGameImage.shape[1],1)
        
        NextState = np.append(NewGameImage, GameState[:,:,:,:3], axis = 3) # 4'lü resim olacak ya 1x40x40x4 olanın 4'lük olan kısımından son 3'ünü al, axis 3 demek :,:,:,:3 yani 3.'sünde birleştirme yap demek
        
        TheAgent.CaptureSample((GameState,BestAction,ReturnScore,NextState)) # depola
        
        TheAgent.Process()
        
        GameState = NextState
        
        if i % 250 == 0:
            print("Train time: ",i, " game score: ",TheGame.GScore)
            TrainHistory.append(TheGame.GScore)
            
        
TrainExperiment() 