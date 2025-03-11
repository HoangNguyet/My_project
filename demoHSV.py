import numpy as np
import cv2 as cv
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model  # type: ignore
import cv2
cap = cv.VideoCapture("static\videos\video1.mp4")
model = load_model("model.h5")

kernel_ci = np.array([[0,0,1,0,0],
                      [0,1,1,1,0],
                      [1,1,1,1,1],
                      [0,1,1,1,0],
                      [0,0,1,0,0]], dtype=np.uint8)

#hsv image

def returnHSV(img):
    blur = cv.GaussianBlur(img, (5,5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

#binary the img from hsv range
def binaryImg(img):
    image1 = img.copy()
    image2 = img.copy()
    image_blue = img.copy()
    
    hsv1 = returnHSV(image1)
    hsv2 = returnHSV(image2)
    hsvblue = returnHSV(image_blue)
    
    b_img1 = cv.inRange(hsv1, low_thresh1, high_thresh1)
    b_img2 = cv.inRange(hsv2, low_thresh2, high_thresh2)
    
    #binary resign img
    b_img_red = cv.bitwise_or(b_img1, b_img2)
    
    #binary blue sign img
    b_img_blue = cv.inRange(hsvblue, low_thresh3, high_thresh3)
    
    return b_img_red, b_img_blue

def findContour(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours
def boundaryBox(img, coutours):
    box = cv.boundingRect(coutours)
    sign = img[box[1]:(box[1] + box[3]), box[0]: (box[0] + box[2])]
    return img, sign, box

#preprocessing img

def preprocessingImageToClassifier(image = None, imageSize = 48, mu = 102.23982103497072, std = 72.11947698025735):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(imageSize, imageSize)
    image = (image - mu) / std
    image = image.reshape(1, imageSize, imageSize, 1)
    
    return image

def predict(sign):
    img = preprocessingImageToClassifier(sign, imageSize=48)
    return np.argmax(model.predict(img))

#finding the red sign
def findRedSign(frame):
    b_img_red, _ = binaryImg(frame)
    contours = findContour(b_img_red)
    for  c in contours:
        area = cv.contourArea(c)
        if(area > 1500):
            (a,b), r = cv.minEnclosingCircle(c)
            
            #checking the round shape or triangle shape of red sign
            if((area > 0.42 * np.pi* r*r)):
                img, sign, box = boundaryBox(frame, c)
                x,y,w,h = box
                
                #checking the distance of top and bottom, aspect ratio of triangle and round shape
                if((w/h > 0.7) and (w/h < 1.2) and ((y + h) < 0.6 *height) and y > height / 20):
                    label = labelToText[predict(sign)]
                    box = np.asarray(box)
                    rois.append(box)
                    labels.append(label)
                    
#finding the blue sign
def findingBlueSign(frame):
    _, b_img_blue = binaryImg(frame)
    contours_blue = findContour(b_img_blue)
    for c_blue in contours_blue:
        area_blue = cv.contourArea(c_blue)
        if(area_blue > 1200):
            (a,b), r = cv.minEnclosingCircle(c_blue)
            area_circle = np.pi * r * r
            
            #checking the round shape of blue sign
            if(area_blue > 0.7 * area_circle):
                _, sign, box = boundaryBox(frame, c_blue)
                x,y,w,h = box
                
                #checking the distance of top and bottom: aspect ratio
                if((w/h > 0.77) and (w/h < 1.2) and (y + h) < 0.6 *height):
                    label = labelToText[predict(sign)]
                    box = np.asarray(box)
                    rois.append(box)
                    labels.append(label)
                    
labelToText = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing vehicles over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Vehicles > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing vehicles > 3.5 tons'
}

#red
low_thresh1 = (165, 100, 40)
high_thresh1 = (179, 255,255)

low_thresh2 = (0, 160, 40)
high_thresh2 = (10, 255,255)

#blue
low_thresh3 = (100, 150, 40)
high_thresh3 = (130, 255,255)

isTracking  = 0
frame_count = 0
max_trackingFrame = 10

while(cap.isOpened()):
    ret, frame = cap.read()
    
    height = frame.shape[0]
    width = frame.shape[1]
    
    if not ret:
        print("Can't read video frame")
        break
    
    if isTracking == 0:
        #run detection code
        rois = []
        labels = []
        findRedSign(frame)
        findingBlueSign(frame)
        
        #re-create and initilize the tracker
        trackers = cv.legacy.MultiTracker_create()
        for roi in rois:
            trackers.add(cv.legacy.TrackerCSRT_create(), frame, roi)
        isTracking = 1
    else:
        if frame_count == max_trackingFrame:
            isTracking = 0
            frame_count = 0
        #update object location
        ret, objs = trackers.update(frame)
        if ret:
            label_count = 0
            for obj in objs: 
                
                #vẽ boundingbox và tên biển báo
                
                print(type(obj))
                p1 = (int(obj[0]), int(obj[1]))
                p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
                cv.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv.rectangle(frame, p1, (int(obj[0] + 2* obj[2]), int(obj[1] - 15)), (0, 255, 0), -1)
                cv.putText(frame, labels[label_count], (int(obj[0] + (obj[2]/2) - 5), int(obj[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  
                label_count = label_count + 1
        else:
            print("Tracking fail")
            isTracking = 0
        frame_count = frame_count + 1 
    print('rois = ', rois)
    cv.imshow('video', frame)
    if cv.waitKey(10) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()         
                    