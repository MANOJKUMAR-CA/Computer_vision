import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller

cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = HandDetector(detectionCon=0.8,maxHands=2)

keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"]]

texts = ""

keyboard = Controller()

def drawAll(img,buttonList):
    for b in buttonList:
        x,y = b.pos
        w,h = b.size
        cvzone.cornerRect(img,(b.pos[0],b.pos[1],b.size[0],b.size[1]),l=30,rt=0)
        cv2.rectangle(img,b.pos,(x+w,y+h),(255,10,250),cv2.FILLED)
        cv2.putText(img,b.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
    return img


# def drawAll(img,buttonList):
#     imgNew = np.zeros_like(img,np.uint8)
#     for but in buttonList:
#         x , y = but.pos
#         cvzone.cornerRect(imgNew,(but.pos[0],but.pos[1],but.size[0],but.size[1]),l=30,rt=0)
#         cv2.rectangle(imgNew,but.pos,(x+but.size[0],y+but.size[1]),(255,0,250),cv2.FILLED)
#         cv2.putText(imgNew,but.text,(x+40,y+60),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
#     out = img.copy()
#     alpha = 0.5
#     mask = imgNew.astype(bool)
#     print(mask.shape)
#     out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha , 0)[mask]
#     return out

class Button():
    def __init__(self,pos,text,size=[85,85]):
        self.pos = pos
        self.text = text
        self.size = size
        

buttonList=[]
for i in range(len(keys)):
    for j,key in enumerate(keys[i]):
        buttonList.append(Button([100*j+50,100*i+50],key))


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    #lmList, bboxInfo = detector.findPosition(img)  #old method of hand tracking
    img = drawAll(img,buttonList)
    if hands:
        #hand1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] #21 Landmark points
        bbox1 = hand1["bbox"] #X,Y,W,H of box
        center1 = hand1["center"]
        htype1 = hand1["type"] #left or right
        fin1 = detector.fingersUp(hand1)

        if(len(hands)==2):
            #hand2
            hand2 = hands[1]
            lmList2 = hand2["lmList"] #21 Landmark points
            bbox2 = hand2["bbox"] #X,Y,W,H of box
            center2 = hand2["center"]
            htype2 = hand2["type"] #left or right
            fin2 = detector.fingersUp(hand2)

        if lmList1:
            for but in buttonList:
                x, y = but.pos
                w, h = but.size
                if x<lmList1[8][0] <x+w and y<lmList1[8][1]<y+h:
                    cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(175,0,175),cv2.FILLED)
                    cv2.putText(img,but.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                    l,_,_= detector.findDistance(lmList1[8][:2],lmList1[12][:2],img)  # Need to make the draw == false

                    #WHEN A BUTTON IS CLICKED
                    if l<30:
                        keyboard.press(but.text)
                        cv2.rectangle(img,but.pos,(x+w,y+h),(0,255,0),cv2.FILLED)
                        cv2.putText(img,but.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                        texts += but.text
                        sleep(1)

    cv2.rectangle(img,(50,350),(700,450),(175,0,175),cv2.FILLED)
    cv2.putText(img,texts,(60,430),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),5)
                        

    cv2.imshow("Image",img)
    cv2.waitKey(1)