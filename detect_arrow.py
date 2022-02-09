import cv2 as cv
import numpy as np
import math
import os
from matplotlib import pyplot as plt
def abcd(img,right_templates,left_templates):
    threshold = 0.55
    img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    right_res = 0
    right_loc = 0
    left_res = 0
    left_loc = 0
    img = cv.blur(img,(5,5))
    for template in right_templates:         
        x = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        if right_res <= np.max(x):
            w, h = template.shape[::-1]
            right_res = np.max(x)
            right_loc = np.asarray(np.unravel_index(x.argmax(), x.shape))
        else:
            right_res=right_res
            right_loc=right_loc

    for template in left_templates:         
        x = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        if left_res <= np.max(x):
            w, h = template.shape[::-1]
            left_res = np.max(x)
            left_loc = np.asarray(np.unravel_index(x.argmax(), x.shape))
        else:
            left_res=left_res
            left_loc=left_loc

    if right_res > left_res and right_res > threshold:
        # print("right")
        max_loc = right_loc
        return 1 
    elif left_res > right_res and left_res > threshold:
        # print("left")
        max_loc = left_loc
        return -1
    else:
        # print("Nothing")
        return 0
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # print(top_left,w,h)
    cropped_image = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv.imshow("cropped",cropped_image)
    ret,cropped_image = cv.threshold(cropped_image,127,255,cv.THRESH_BINARY)
    # kernel = np.ones((5,5),np.uint8)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    # erosion = cv.erode(cropped_image,kernel,iterations = 3)
    # dilation = cv.dilate(erosion,kernel,iterations = 3)
    # cv.imshow("cropped",dilation)
    # detect_corners(dilation)

def detect_corners(cropped_image):
    
    edges = cv.Canny(cropped_image, 100, 255) 
    cv.imshow("edges",edges)
    corners = cv.goodFeaturesToTrack(edges,5,0.01,1)
    if corners is None:
        print("Nothing")
        return None
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(edges,(x,y),4,255,-1)
    cv.circle(edges,(edges.shape[1]//2,edges.shape[0]//2),4,255,-1)
    cv.imshow("center",edges)
    right = 0
    left = 0
    for corner in corners:
        if corner[0][0]<=edges.shape[1]//2:
            left +=1
        if corner[0][0]>=edges.shape[1]//2:
            right +=1
    if right-2>left:
        print("right")
    elif left-2>right:
        print("left")
    else:
        print("Nothing")    

if __name__ == "__main__":
    every_frame = 10
    vid = cv.VideoCapture(0)
    right_templates = []
    left_templates = []
    for file in os.listdir("template"):
        if file.startswith("right"):
            print(file)
            right_templates.append(cv.imread(f"template/{file}",0))
    
    for file in os.listdir("template"):
        if file.startswith("left"):
            print(file)
            left_templates.append(cv.imread(f"template/{file}",0))
    scale_down = 0.6

    right_templates = [cv.resize(template, None, fx= scale_down, fy= scale_down, interpolation= cv.INTER_LINEAR) for template in right_templates]
    left_templates = [cv.resize(template, None, fx= scale_down, fy= scale_down, interpolation= cv.INTER_LINEAR) for template in left_templates]
    n_frames = 0
    direction = 0
    while(True):
        
        # Capture the video frame
        # by frame
    
        ret, frame = vid.read()
    
        # Display the resulting frame
        cv.imshow('frame', frame)
        
        n_frames+=1
        direction += abcd(frame,right_templates,left_templates)
        if n_frames%every_frame==0:
            print(direction)
            if direction>0.6*every_frame:
                print("right")
            elif direction<-0.6*every_frame:
                print("left")
            else:
                print("Nothing")
            direction = 0
            
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()