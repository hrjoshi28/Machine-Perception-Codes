##import numpy as np
##import cv2
##cap = cv2.VideoCapture('output.avi')
### Define the codec and create VideoWriter object
##fourcc = cv2.VideoWriter_fourcc(*'XVID')
##out = cv2.VideoWriter('ouput1.avi',fourcc, 20.0, (640,480))
##while(cap.isOpened()):
##    ret, frame = cap.read()
##    if ret==True:
##        #frame = cv2.flip(frame,0)
##        # write the flipped frame
##        out.write(frame)
##        cv2.imshow('frame',frame)
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##   
### Release everything if job is finished
##cap.release()
##out.release()
##cv2.destroyAllWindows()

import numpy as np
import cv2
cap1 = cv2.VideoCapture('C:/Users/hrjos/Videos/vid2.avi')
cap2 = cv2.VideoCapture('C:/Users/hrjos/Videos/vid1.avi')
t=0;
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('output3.avi',fourcc, 20.0, (640,480))
while(cap1.isOpened() or cap2.isOpened()):
    ret1 , frame1 = cap1.read()
    ret2 , frame2 = cap2.read()

    if ret1==True or ret2==True:
        #frame1 = cv2.flip(frame1,0)
        # write the flipped frame
        print("Hello")
        if t==0:
            out.write(frame1)
            cv2.imshow('frame',frame1)
        else :
            out.write(frame2)
            cv2.imshow('frame',frame2)
        
        t=(t+1)%2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
# Release everything if job is finished
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
