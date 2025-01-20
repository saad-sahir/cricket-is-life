import numpy as np 
import cv2 
  
cap = cv2.VideoCapture('video.mp4') 
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=25, detectShadows=False)
  
while(1): 
    ret, frame = cap.read()        
  
    # applying on each frame 
    fgmask = fgbg.apply(frame)   
  
    cv2.imshow('frame', fgmask) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release() 
cv2.destroyAllWindows() 