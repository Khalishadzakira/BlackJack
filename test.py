    from fileinput import filename
    import cv2
    import os
    import datetime
    import time
    import numpy as np
    import copy
    from keras.models import load_model

    folder_dataset = "/Users/khalishadzakira/Documents/PCV/FP v2.0/Dataset_fix/"
    folder_angka = "Q"
    save_folder = folder_dataset + folder_angka
    os.chdir(save_folder)

    def flattener(image, pts, w, h):
        temp_rect = np.zeros((4,2), dtype = "float32")
        
        s = np.sum(pts, axis = 2)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        diff = np.diff(pts, axis = -1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        if w <= 0.8*h: # If card is vertically oriented
            temp_rect[0] = tl
            temp_rect[1] = tr
            temp_rect[2] = br
            temp_rect[3] = bl

        if w >= 1.2*h: # If card is horizontally oriented
            temp_rect[0] = bl
            temp_rect[1] = tl
            temp_rect[2] = tr
            temp_rect[3] = br

        # If the card is 'diamond' oriented, a different algorithm
        # has to be used to identify which point is top left, top right
        # bottom left, and bottom right.
        
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

    cap = cv2.VideoCapture(0)

    counter = 0 # Menghitung banyaknya frame

    while(1):
        ret, image = cap.read()
        
        if not ret:
            break
        
        # if counter % 10 == 0:
        # Mengubah gambar jadi abu, terus di blur, trus di threshold jadi item putih
        abu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(abu, (3,3), 0)
        ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        
        # Dicari pinggiran dari kartu dengan contour, terus contournya di sort dari paling besar jadi paling kecil
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse = True)
        
        # Contour terbesar diasumsikan sebagai kartunya
        if len (contours) > 0:
            
            card = contours[0]
            cv2.drawContours(image, contours, 0, (0,0,255), 2)
            
            # Mencari titik titik sudut kartu
            perimeter = cv2.arcLength(card, True)
            polygon = cv2.approxPolyDP(card, 0.01 * perimeter, True)
            points = np.float64(polygon) # titik titik sudut kartu
            
            x,y,w,h  = cv2.boundingRect(card) # membuat kotak sesuai dengan sudut2 kartu
            
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
            
            flatten = flattener(image, points, w, h)
            #cv2.imshow("flatten", flatten)
            
            angka = flatten[5:55, 5:35]
            angka = cv2.resize(angka, (0,0), fx=4, fy=4)
            
            angka = cv2.GaussianBlur(angka, (3,3), 0)
            _,angka = cv2.threshold(angka, 100, 255, cv2.THRESH_BINARY_INV)
            
            angka_contour,_ = cv2.findContours(angka, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #print(len(angka_contour))
            angka_contour = sorted(angka_contour, key=cv2.contourArea, reverse = True)
            
            if len (angka_contour) > 0:
                xa,ya,wa,ha = cv2.boundingRect(angka_contour[0])
                
                angka = angka[ya:ya+ha, xa:xa+wa]
                angka = cv2.resize(angka, (70,125), 0,0)
                
                #angka_warna = copy.deepcopy(angka)
                #angka_warna = cv2.cvtColor(angka, cv2.COLOR_GRAY2BGR)
                #print(angka_warna.shape)
                #cv2.drawContours(angka_warna, angka_contour, 0, (0,0,255), 2)
                
                cv2.imshow("angka", angka)
                
                if counter % 15 == 0:
                    filename = str(counter/15) + ".jpg"
                    cv2.imwrite(filename, angka)
                    
            
        # show gambar
        cv2.imshow("thresh", thresh)
        cv2.imshow("asli", image)
        
        counter += 1
        
        if cv2.waitKey(25) == ord("q"):
            break  
        
    cap.release()
    cv2.destroyAllWindows()
