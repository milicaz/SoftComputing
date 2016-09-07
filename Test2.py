# omogucava iscrtavanje slika i plotova unutar samog browsera

from imutils.object_detection import non_max_suppression
from imutils import paths

import cv2 # OpenCV biblioteka
import imutils
import numpy as np # NumPy biblioteka, "np" je sinonim koji se koristi dalje u kodu kada se koriste funkcije ove biblioteke
import matplotlib.pyplot as plt # biblioteka za plotovanje, tj. crtanje grafika, slika... "plt" je sinonim

import matplotlib.pylab as pylab# omogucava iscrtavanje slika i plotova unutar samog browsera

import cv2 # OpenCV biblioteka
import numpy as np # NumPy biblioteka, "np" je sinonim koji se koristi dalje u kodu kada se koriste funkcije ove biblioteke
import matplotlib.pyplot as plt # biblioteka za plotovanje, tj. crtanje grafika, slika... "plt" je sinonim

import matplotlib.pylab as pylab

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap=cv2.VideoCapture('/home/queen/video.avi')


count = 0

while(True):
        
    ret,frame=cap.read()

    frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    orig = frame.copy()

    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
      cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
      cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)



    cv2.imshow('frame', frame)


	count = count + len(pick)

    print count
        
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

cv2.destroyAllWindows()
