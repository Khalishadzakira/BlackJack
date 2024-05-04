import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model

import cobafungsi as fb

Game = fb.Game()

cap = cv2.VideoCapture(0)
FrameCounter = 0
while(1):
    ret, image = cap.read()
    if not ret:
        break
    
    Game.runGame(image)
    
    # angka = Game.runGame(image)
    # if angka is not None :
    #     cv2.imshow("angka", angka)
    #     print(angka.shape)
        
    print("Game State: ", Game.GameState)
    print("Player Card: ", Game.PlayerCard)
    #print("Player Point: ", Game.PointPlayer)
    print("Dealer Card: ", Game.DealerCard)
    #print("Dealer Point: ", Game.PointDealer)
    
    cv2.imshow("asli", image)
    FrameCounter += 1
    if cv2.waitKey(25) == ord("q"):
        break  
    
cap.release()
cv2.destroyAllWindows()