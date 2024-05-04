import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model

x = 10

class Game:
    def __init__(self):
        self.GameState = "FirstDraw"
        self.GameTurn = "Player"
        self.PlayerCard = []
        self.DealerCard = []
        self.PointPlayer = 0
        self.PointDealer = 0
        
    def CheckState(self):
        if self.GameState == "FirstDraw":
            #player card draw --> dealer card draw
            if self.GameTurn == "Player":
                self.DrawCard(self)
            elif self.GameTurn == "Dealer":
                self.DealerDraw(self)
                
        elif self.GameState == "PlayerDecision":
            #milih hit/stand (baru itung lagi)
            pass  
        elif self.GameState == "DealerShow":
            #show card 2 --> jumlahin
                #if <= 16 --> ambil kartu
                #if >16 --> stand
            pass
        elif self.GameState == "CheckPoint":
            #jumlah total akhir keduanya --> bandingin win/lose
            pass
        
    def PlayerDraw(self):
        #function deteksi
        if cv2.waitKey(1) == ord("s"):
            #save kartu deteksi masukin ke player card
        
                
    
        #pembagian kartu, player, dealer
        #playerdraw --> deteksi --> save kartu --> hitung total
        #dealerdraw --> deteksi --> save kartu --> hitung total

        #player state : pilih hit/stand
            #if hit --> tambah kartu --> save
            #if stand --> buka kartu ke 2 --> dealerstate 
            
        #dealer state : 
            #hitung jumlah
                #if <= 16 --> ambil kartu
                #if >16 --> stand
            
        
        
        
        #player decission : hit, stand
        #hit --> masukin kartu player
        #stand --> ambil poin yang ada --> keluarin kartu2 dealer
        # dealer : <= 16 ambil kartu
        # dealer : > 16 stand
        #kalkulasi poin 
        #tentuin pemenang
        
        