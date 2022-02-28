import cv2 as cv
import copy

#Using 4x4 dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
cap = cv.VideoCapture(0)

while(True):
    ret_val, image = cap.read()
    #ignore
    if not ret_val:
        break
    img_copy = copy.deepcopy(image)
    #convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = 127
    #converting image to black and white to make the process robust
    im_bw = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]
    #Parameters for the detectors
    parameters =  cv.aruco.DetectorParameters_create()
    #return values: corners, Tag ID array (nonetype), rejected candidates for tags 
    corners, ids, rejects = cv.aruco.detectMarkers(im_bw, dictionary, parameters=parameters)
    # TODO(Ashwin,Harsh): Use Camera Calibration
    #corners, ids, rejects = cv.aruco.detectMarkers(im_bw, dictionary, parameters=parameters,cameraMatrix=cameraMatrix) 
    #drawing markers
    img = cv.aruco.drawDetectedMarkers(img_copy, corners, ids)
    if len(corners) > 0:
        #print the coordinates (Can use the returned values)
        print("Detected: ", corners)
    #show image
    cv.imshow("out", img)
    if cv.waitKey(1) == 27:
        break
