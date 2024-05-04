from socketserver import ForkingMixIn
from tabnanny import verbose
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model

MODEL_KARTU = load_model("ModelKartu")
KELAS_KARTU = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q']
SORTER = ['0', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Peletakan text ditengah (perbandingan terhadap text itu sendiri)
def TextCenter(h, w, text, font, FontSize):
    TextSize = cv2.getTextSize(text, font, FontSize, 2)[0]
    TextX = int ((w-TextSize[0])/2)
    TextY = int ((h-TextSize[1])/2)
    return TextX, TextY

def EkstrakAngka(image):
    # ubah warna dan blur
    abu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(abu, (3,3), 0)
    # <100 = 0, >100 = 255
    ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # mencari contour dari tresh yang sudah ditentukan
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse = True)
    
    # mendeteksi bagian yang dianggap kartu saja
    if len (contours) > 0:    
        for i in range(len(contours)-1, -1, -1) :
            #print(cv2.contourArea(contours[i]))
            if cv2.contourArea(contours[i]) > 47000 or cv2.contourArea(contours[i]) < 42000:
                contours.pop(i)
    
    if len (contours) > 0:    
        # menentukan titik pusat kartu
        card = contours[0]
        perimeter = cv2.arcLength(card, True)
        polygon = cv2.approxPolyDP(card, 0.01 * perimeter, True)
        points = np.float64(polygon)

        x,y,w,h  = cv2.boundingRect(card)

        #mengambil bagian pojok kartu (bagian angka)
        flatten = flattener(image, points, w, h)
        angka = flatten[5:55, 5:35]
        angka = cv2.resize(angka, (0,0), fx=4, fy=4)

        angka = cv2.GaussianBlur(angka, (3,3), 0)
        _,angka = cv2.threshold(angka, 100, 255, cv2.THRESH_BINARY_INV)

        angka_contour,_ = cv2.findContours(angka, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        angka_contour = sorted(angka_contour, key=cv2.contourArea, reverse = True)

        if len (angka_contour) > 0:
            xa,ya,wa,ha = cv2.boundingRect(angka_contour[0])

            angka = angka[ya:ya+ha, xa:xa+wa]
            angka = cv2.resize(angka, (70,125), 0,0)
            angka = np.repeat(angka[:,:,np.newaxis],3,axis=2)
            
            return angka, card
        else:
            return None, None
        
    else:
        return None, None
        
def PrediksiAngka(image):
    prediksi = "0"
    if image is not None: 
        #image = cv2.resize(image, (70,125), 0,0)
        #image = image.reshape(1, 125, 70, 1)
        image = tf.keras.utils.img_to_array(image) # menjadikan gambar menjadi array untuk cnn
        image = np.expand_dims(image, axis=0)
        
        prediksi = MODEL_KARTU.predict(image, verbose=0)
        prediksi = prediksi.argmax()
        prediksi = KELAS_KARTU[prediksi] 
        
    return prediksi

def flattener(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8*h: # vertikal
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # horizontal
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] 
            temp_rect[1] = pts[0][0] 
            temp_rect[2] = pts[3][0] 
            temp_rect[3] = pts[2][0] 

        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] 
            temp_rect[1] = pts[3][0] 
            temp_rect[2] = pts[2][0] 
            temp_rect[3] = pts[1][0] 
        
    maxWidth = 200
    maxHeight = 300

    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp

class Game:
    #inisialisasi object pada kelas
    def __init__(self):
        self.GameState = "FirstDraw"
        self.PlayerCard = []
        self.DealerCard = []
        self.PointPlayer = 0
        self.PointDealer = 0

    def calculatePoint(self):
        PointPlayer = 0
        PointDealer = 0
        for i in self.PlayerCard:
            if i == "A": #aturan poin A
                if PointPlayer + 11 > 21:
                    PointPlayer += 1
                else:
                    PointPlayer += 11
            elif i == "J" or i == "Q" or i == "K": #aturan poin J, K, Q
                PointPlayer += 10
            else:
                PointPlayer += int(i) #penambahan poin kartu angka 
        for i in self.DealerCard:
            if i == "A":
                if PointDealer + 11 > 21:
                    PointDealer += 1
                else:
                    PointDealer += 11
            elif i == "J" or i == "Q" or i == "K":
                PointDealer += 10
            else:
                PointDealer += int(i)
        self.PointPlayer = PointPlayer
        self.PointDealer = PointDealer

    def runGame(self, image):
        #print(image.shape)
        angka, ContourKartu = EkstrakAngka(image)
        #return angka
        prediksi = PrediksiAngka(angka)
        font = cv2.FONT_HERSHEY_DUPLEX
        h, w, _= image.shape
        
        if ContourKartu is not None:
            cv2.drawContours(image,[ContourKartu], 0, (0,0,255), 2)
        
        cv2.putText(image, prediksi, (10, int (480/2)), font, 1, (0,0,255), 2)

        if self.GameState == "FirstDraw":
            if len(self.PlayerCard) < 2:
                #print("Press P to Set Player Card")
                x, y = TextCenter(h, w, "Press P to Set Player Card", 1, font)
                cv2.putText(image, "Press P to Set Player Card", (x, 70), font, 1, (255,0,0), 2)
                if cv2.waitKey(50) == ord("p"):
                    self.PlayerCard.append(prediksi)

            elif len(self.PlayerCard) >= 2:
                x, y = TextCenter(h, w, "Press P to Set Dealer Card", 1, font)
                cv2.putText(image, "Press P to Set Dealer Card", (x, 70), font, 1, (255,0,0), 2)
                if cv2.waitKey(50) == ord("p"):
                    self.DealerCard.append(prediksi)
                    self.GameState = "PlayerDecision"
                    

        elif self.GameState == "PlayerDecision":
            #print("Press H to Hit, Press S to Stand")
            x, y = TextCenter(h, w, "Press H to Hit, Press S to Stand", 1, font)
            cv2.putText(image, "Press H to Hit, Press S to Stand", (x, 70), font, 1, (255,0,0), 2)
            if cv2.waitKey(50) == ord("h"):
                self.PlayerCard.append(prediksi)
            elif cv2.waitKey(50) == ord("s"):
                self.GameState = "DealerShow"

        elif self.GameState == "DealerShow":
            #print("Press P to Set Card")
            x, y = TextCenter(h, w, "Press P to Set Card", 1, font)
            cv2.putText(image, "Press P to Set Card", (x, 70), font, 1, (255,0,0), 2)
            if cv2.waitKey(50) == ord("p"):
                self.DealerCard.append(prediksi)
                self.GameState = "DealerDecision"

        elif self.GameState == "DealerDecision":
            if self.PointDealer < 17 and len(self.DealerCard) <= 2 :
                #print("Press P to Set Card")
                x, y = TextCenter(h, w, "Press P to Set Card", 1, font)
                cv2.putText(image, "Press P to Set Card", (x, 70), font, 1, (255,0,0), 2)
                if cv2.waitKey(50) == ord("p"):
                    self.DealerCard.append(prediksi)

            else :
                self.GameState = "Result"

        elif self.GameState == "Result" :
            if self.PointDealer > 21 and self.PointPlayer > 21:
                #print("Draw")
                x, y = TextCenter(h, w, "Draw", 1, font)
                cv2.putText(image, "Draw", (x, y), font, 1, (255,0,0), 2)
            elif self.PointDealer > 21 :
                #print("Player Win")
                x, y = TextCenter(h, w, "Player Win", 1, font)
                cv2.putText(image, "Player Win", (x, y), font, 1, (255,0,0), 2)
            elif self.PointPlayer > 21 :
                #print("Dealer Win")
                x, y = TextCenter(h, w, "Dealer Win", 1, font)
                cv2.putText(image, "Dealer Win", (x, y), font, 1, (255,0,0), 2)
            elif self.PointPlayer > self.PointDealer:
                #print("Player Win")
                x, y = TextCenter(h, w, "Player Win", 1, font)
                cv2.putText(image, "Player Win", (x, y), font, 1, (255,0,0), 2)
            elif self.PointPlayer < self.PointDealer:
                #print("Dealer Win")
                x, y = TextCenter(h, w, "Dealer Win", 1, font)
                cv2.putText(image, "Dealer Win", (x, y), font, 1, (255,0,0), 2)
            else:
                #print("Draw")
                x, y = TextCenter(h, w, "Draw", 1, font)
                cv2.putText(image, "Draw", (x, y), font, 1, (255,0,0), 2)

        x, y = TextCenter(h, w, self.GameState, 1, font)
        cv2.putText(image, self.GameState, (x+10, 30), font, 1, (0,0,255), 2)
       
        self.PlayerCard.sort(key=SORTER.index)
        self.DealerCard.sort(key=SORTER.index)
        
        cv2.putText(image, "Player Card : ", (20,430), font, 0.5, (0,0,255), 2)
        cv2.putText(image, "Dealer Card : ", (480,430), font, 0.5, (0,0,255), 2)
        
        cv2.putText(image, "Player Points : " + str(self.PointPlayer), (20, 400), font, 0.5, (0,0,255), 2)
        cv2.putText(image, "Dealer Points : " +str(self.PointDealer), (480, 400), font, 0.5, (0,0,255), 2)
        
        for i in range(len(self.PlayerCard)):
            cv2.putText(image, self.PlayerCard[i], (20+(30*i),460), font, 0.5, (0,0,255), 2)
            
        for i in range(len(self.DealerCard)):
            cv2.putText(image, self.DealerCard[i], (580-(30*i),460), font, 0.5, (0,0,255), 2)
    
        # self.PlayerCard=sorted(self.PlayerCard,key=lambda x: SORTER.index(x))
        # self.DealerCard=sorted(self.DealerCard,key=lambda x: SORTER.index(x))

        self.calculatePoint()