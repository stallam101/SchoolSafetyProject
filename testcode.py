import numpy as np
import cv2

gun_cascade = cv2.CascadeClassifier("cascade.xml")

img = cv2.imread('twopeople.jpg')
images=['blackshirt.jpg', 'twopeople.jpg','fire.jpg',
        'images(1).jpg','maninsuit.png','download.jpg','images.jpg',
        'manwithgun.jpg','personholdinggun2.jpg', 'sil.jpg',
        'verticalgun.jpg']
for v in images:
    image = cv2.imread(v)
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    guns = gun_cascade.detectMultiScale(grayimage, 1.01, 5)
    for (x,y,w,h) in guns:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
