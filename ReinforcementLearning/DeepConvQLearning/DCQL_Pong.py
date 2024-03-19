# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 07:54:13 2024

@author: esraablak
"""

import pygame
import random

FPS = 60

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 420
GAME_HEIGHT = 400

PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15

BALL_WIDTH = 20
BALL_HEIGHT = 20

PADDLE_SPEED = 3
BALL_X_SPEED = 2
BALL_Y_SPEED = 2

WHITE = (255,255,255)
BLACK = (0,0,0)

screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))

def drawPaddle(switch, paddleYPos):
    
    if switch == "left":
        paddle = pygame.Rect(PADDLE_BUFFER, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    elif switch == "right":
        paddle = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    
    pygame.draw.rect(screen, WHITE, paddle)

def drawBall(ballXPos, ballYPos):
    
    ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)
    
def updatePaddle(switch, action, paddleYPos, ballYPos):
    
    dft = 7.5 
    
    # Agent
    if switch == "left":
        # gelen actiona göre hareket ettir
        if action == 1: # yukarı
            paddleYPos = paddleYPos - PADDLE_SPEED*dft
        if action == 2: # aşağı
            paddleYPos = paddleYPos + PADDLE_SPEED*dft
            
        # ama oyun çerçevesinden çıkmasın
        if paddleYPos < 0:
            paddleYPos = 0
        if paddleYPos > GAME_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGHT - PADDLE_HEIGHT
            
    elif switch == "right": # paddle y'si ile ball y'sini bir tut
        
        if paddleYPos + PADDLE_HEIGHT / 2 < ballYPos + BALL_HEIGHT / 2:
            paddleYPos = paddleYPos + PADDLE_SPEED*dft
        if paddleYPos + PADDLE_HEIGHT / 2 > ballYPos + BALL_HEIGHT / 2:
            paddleYPos = paddleYPos - PADDLE_SPEED*dft
            
        # ekrandan çıkmasın
        if paddleYPos < 0:
            paddleYPos = 0
        if paddleYPos > GAME_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGHT - PADDLE_HEIGHT
    
    return paddleYPos

def updateBall(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection, DeltaFrameTime):
    dft = 7.5
    
    # topun hareketi
    ballXPos = ballXPos + ballXDirection*BALL_X_SPEED*dft
    ballYPos = ballYPos + ballYDirection*BALL_Y_SPEED*dft
    
    score = -0.05 # living penalty
    
    # sol paddel / agent güncellemesi (agent'e çarptıysan top ona göre dönücek ve agent topu karşılayabilmiştir demek ona göre de score)
    if (ballXPos <= (PADDLE_BUFFER + PADDLE_WIDTH)) and ((ballYPos + BALL_HEIGHT) >= paddle1YPos) and (ballYPos <= (paddle1YPos + PADDLE_HEIGHT)) and (ballXDirection == -1):
        
        ballXDirection = 1 # çarpıysan topun yönünü değiş
        score = 10 
    
    elif (ballXPos <= 0): # agent topu karşılayamadıysa
        ballXDirection = 1
        score = -10
        
        return [score, ballXPos, ballYPos, ballXDirection, ballYDirection]
    
    # sağ padel / zaten her türlü karşılayacak -> score yok sadece topun yönünü değiş
    if ((ballXPos >= (WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER)) and ((ballYPos + BALL_HEIGHT)>= paddle2YPos) and (ballYPos <= (paddle2YPos + PADDLE_HEIGHT)) and (ballXDirection == 1)):
        
        ballXDirection = -1
        
    elif(ballXPos >= WINDOW_WIDTH - BALL_WIDTH): 
        
        ballXDirection = -1
        return [score, ballXPos,ballYPos, ballXDirection, ballYDirection]
    
    # yukarıya veya aşağıya çarparsa
    if ballYPos <= 0:
        
        ballYPos = 0
        ballYDirection = 1
        
    elif ballYPos >= GAME_HEIGHT - BALL_HEIGHT:
        
        ballYPos = GAME_HEIGHT - BALL_HEIGHT
        ballYDirection = -1
        
    return [score, ballXPos,ballYPos, ballXDirection, ballYDirection]
        

class PongGame:
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pong DCQL Env")
        
        self.paddle1YPos = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        
        self.ballXPos = WINDOW_WIDTH / 2
        self.ballYPos = random.randint(0,9) * (WINDOW_HEIGHT - BALL_HEIGHT) / 9 

        self.clock = pygame.time.Clock()
        
        self.GScore = 0.0
        
        self.ballXDirection = random.sample([-1,1], 1)[0]
        self.ballYDirection = random.sample([-1,1], 1)[0]
                
    
    
    
    def InitialDisplay(self): # env'i göster
        
        pygame.event.pump()
        
        screen.fill(BLACK)
        
        drawPaddle("left", self.paddle1YPos)
        drawPaddle("right", self.paddle2YPos)
        drawBall(self.ballXPos, self.ballYPos)
        
        pygame.display.flip() # oluşturduğun her şeyi window'a update et
        
        
    
    def PlayNextMove(self, action): # paddle ve ball hareket etmesi, hareket ettikten sonra tekrar görselleştirilmesi (display)
        
        DeltaFrameTime = self.clock.tick(FPS) # hareket => x=v.t

        pygame.event.pump() # operating system ile bizim yaptığımız oyunun eş zamanlı / senkronize çalışabilmesi için
        
        score = 0
        
        screen.fill(BLACK)
        
        self.paddle1YPos = updatePaddle("left", action, self.paddle1YPos, self.ballYPos)
        drawPaddle("left", self.paddle1YPos)
        
        self.paddle2YPos = updatePaddle("right", action, self.paddle2YPos, self.ballYPos)
        drawPaddle("right", self.paddle2YPos)
    
        [score, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection] = updateBall(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection, DeltaFrameTime)
    
    
        drawBall(self.ballXPos, self.ballYPos)
    
    
        if score > 0.5 or score < -0.5:
            self.GScore = self.GScore*0.9 + 0.1*score 
    
        # CNN için oyunun ekranını al
        ScreenImage = pygame.surfarray.array3d(pygame.display.get_surface())
        
        pygame.display.flip() # update display game
        
        # önemli olan env'den aldığın action'a göre score ve screenimage
        "dql için state'e göre action yaparsın ve s' verir. burada next_state => screenImage"
        return [score,ScreenImage]