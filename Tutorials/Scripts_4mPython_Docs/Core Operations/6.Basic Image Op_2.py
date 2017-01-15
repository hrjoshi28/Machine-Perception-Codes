"""
    Program to capture Region of Image , also displaying coordinates on mouse click

"""


import numpy as np
import cv2


###function to draw circle  on mouse click event and print co-ordinate of click .
##def draw_circle(event,x,y,flags,param):
##    if event == cv2.EVENT_FLAG_LBUTTON :
##        cv2.circle(img,(x,y),100,(255,0,0),-1)
##        print(x,y)



img=cv2.imread(r'C:\Users\hrjos\Pictures\Messi2.jpg',1)

#to print BGR value at 100,100
print(img[100,100])

#extracting ROI : rEGION OF IMAGE
roi=img[182:288,100:200]
ball=img[100:200,182:288]
img[155:255,375:481]=ball

while cv2.waitKey(1) & 0xFF != ord('q') :
    cv2.imshow('img',img)

cv2.destroyAllWindows()



""" Program which  draws a circle on double click event """ 
##import cv2
##import numpy as np
### mouse callback function
##def draw_circle(event,x,y,flags,param):
##    if event == cv2.EVENT_LBUTTONDBLCLK:           #left button double click
##        cv2.circle(img,(x,y),100,(255,0,0),-1)
##
### Create a black image, a window and bind the function to window
##img = np.zeros((512,512,3), np.uint8)
##cv2.namedWindow('image')
##cv2.setMouseCallback('image',draw_circle)
##while(1):
##    cv2.imshow('image',img)
##    if cv2.waitKey(20) & 0xFF == 27:
##        break
##cv2.destroyAllWindows()
