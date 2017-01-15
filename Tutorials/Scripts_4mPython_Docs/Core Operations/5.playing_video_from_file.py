import numpy as np
import cv2

cap = cv2.VideoCapture('C:/Users/hrjos/Videos/vid2.avi')

while(cap.isOpened()):  # check !
    # capture frame-by-frame
    ret, frame = cap.read()
    
    #check to avoid errors !!!
    if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)  #to process large aspect ratio... fit the screen
        # Display the resulting frame
        cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done release the capture
cap.release()
cv2.destroyAllWindows()

