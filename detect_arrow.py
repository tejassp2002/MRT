import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
def abcd(img,templates):
    threshold = 0.55
    img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    w, h = templates[0].shape[::-1]
    res = []
    img = cv.blur(img,(5,5))
    for template in templates: 
        x = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        res.append((np.max(x), np.unravel_index(x.argmax(), x.shape)))

    max_val1, max_loc1 = res[0]
    max_val2, max_loc2 = res[1]
    if max_val1 > max_val2 and max_val1 > threshold:
        print("right")
        max_loc = max_loc1
    elif max_val2 > max_val1 and max_val2 > threshold:
        print("left")
        max_loc = max_loc2
    else:
        print("Nothing")
        return None
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # print(top_left,w,h)
    cropped_image = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    ret,cropped_image = cv.threshold(cropped_image,127,255,cv.THRESH_BINARY)
    # kernel = np.ones((5,5),np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    erosion = cv.erode(cropped_image,kernel,iterations = 3)
    dilation = cv.dilate(erosion,kernel,iterations = 3)
    cv.imshow("cropped",dilation)
    detect_corners(cropped_image)

def detect_corners(cropped_image):
    
    edges = cv.Canny(cropped_image, 100, 255) 
    cv.imshow("edges",edges)
    corners = cv.goodFeaturesToTrack(edges,5,0.01,1)
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

    vid = cv.VideoCapture(1)
    right_template = cv.imread('template/right.jpg',0)
    left_template = cv.imread('template/left.jpg',0)
    templates = [right_template,left_template]
    scale_down = 0.6

    templates = [cv.resize(template, None, fx= scale_down, fy= scale_down, interpolation= cv.INTER_LINEAR) for template in templates]
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        cv.imshow('frame', frame)
        abcd(frame,templates)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()