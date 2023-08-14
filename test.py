#IMPORT REQUIRED PACKAGES
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math 
import time

#GET LIVE VIDEO FROM CAMERA
live=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)

#DEFINE REQUIRED VARIABLES
offset=20
imgSize=300
counter=0
folder="SourceCode/Data/Y"

#Classifier
classifier = Classifier("SourceCode/Model_TEST/model-bw.h5", "SourceCode/Model/labels.txt")

#Labels
labels=['A','B','C','D','E','F','G','H','I','K','L','M','O','P','Q','R','S','T','U','V','W','X','Y']

#START THE INFINITE LOOP
while True:
    
    #READ THE FRAMES FROM THE VIDEO
    rev,image=live.read()
    hands,image=detector.findHands(image)

    #CROP THE IMAGE INTO A GENERAL SHAPE
    if hands:
        hand = hands[0]
        #GET THE COORDINATES AND WIDTH,HEIGHT
        x,y,w,h=hand['bbox']

        #CREATE A WHITE IMAGE USING NUMPY
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=image[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropShape=imgCrop.shape


        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)

            #ADD THE CROPPED IMAGE TO THE WHITE IMAGE
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize
            prediction, index = classifier.getPrediction(image)
            print(prediction,index)
        else:
            k=imgSize/w
            hCal=math.ceil(k*h)

            #ADD THE CROPPED IMAGE TO THE WHITE IMAGE
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize

        #DISPLAY THE IMAGES
        cv2.imshow("Image",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image2",image)
    key=cv2.waitKey(1)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)