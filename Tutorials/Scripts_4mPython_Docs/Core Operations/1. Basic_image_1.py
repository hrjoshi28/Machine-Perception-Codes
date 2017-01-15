"""
Below program loads an image in grayscale, displays it,
save the image if you press ‘s’ and exit, or simply exit without
saving if you press ESC key
"""

import cv2
img=cv2.imread(r'C:\Users\hrjos\Pictures\Messi.jpg',0) #0== cv2.IMREAD_GRAYSCALE
cv2.imshow('image',img)
num=cv2.waitKey(0)
#for 64bit machine::  cv2.waitKey(0) & 0xFF
#0 waits indefinitely till keystroke


if chr(num)=='s' :
    cv2.imwrite(r'C:\Users\hrjos\Pictures\Messi_graysacle.jpg',img)
cv2.destroyAllWindows()
    
"""
import numpy as np
import cv2
img = cv2.imread('messi5.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
cv2.imwrite('messigray.png',img)
cv2.destroyAllWindows()
"""
