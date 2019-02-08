
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure

import cv2

import numpy as np


plate = cv2.imread('imgs/positive00021.bmp')

img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#img = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
#equ = cv2.equalizeHist(img)
#img = np.pad(img, ((5, 5), (5, 5)), 'constant', constant_values=255)

img1 = img.copy()
#img2 = img.copy()


#lap = cv2.Laplacian(img1,cv2.CV_64F)
#lap = np.uint8(np.absolute(lap))
#median = cv2.medianBlur(lap,3)
#img = cv2.add(canny_edges,img)
#laplacian_edges = np.uint8(laplacian_edges)

#lap_edge = laplacian_edges.copy()

#ret, imgf = cv2.threshold(canny_edges, 0, 255, cv2.THRESH_OTSU)
#cv2.THRESH_BINARY+

#median = cv2.medianBlur(th2,3)
#img2 = imgf.copy()

#image, contours, hierarchy = cv2.findContours(imgf,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#im, lap_contours, hierarchy = cv2.findContours(imgf,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



#img = cv2.drawContours(img1, contours, -1, (0,255,0), 1)
#lap = cv2.drawContours(img1, lap_contours, -1, (0,255,0), 1)

#ret,imgf = cv2.threshold(img1, 130, 255, cv2.THRESH_BINARY)


#canny_edges = cv2.Canny(img,30,110)
#thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3)
#thresh1 = thresh.copy()

#kernel = np.ones((3,3),np.uint8)
#thresh = cv2.erode(thresh,kernel,iterations = 1)

# extract the Value component from the HSV color space and apply adaptive thresholding
# to reveal the characters on the license plate


T = threshold_local(img, 11, offset=1, method="gaussian")
thresh = (img > T).astype("uint8") * 255
#thresh = cv2.bitwise_not(thresh)


#thresh = cv2.bitwise_not(th2)
#median = cv2.medianBlur(th2,3)
#th2_eq = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)

# perform a connected components analysis and initialize the mask to store the locations
# of the character candidates
labels = measure.label(thresh, neighbors=8, background=0)
charCandidates = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
        continue

    # otherwise, construct the label mask to display only connected components for the
    # current label, then find contours in the label mask
    labelMask = np.zeros(thresh.shape, dtype="uint8")

    labelMask[labels == label] = 255
    #cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #nts = cnts[0] if imutils.is_cv2() else cnts[1]

    cv2.imshow('mask',labelMask)

    cv2.waitKey(0)

cv2.imshow('img2',img1)


#cv2.imshow('equ',equ)
#cv2.imshow('th2_5',th2)
#cv2.imshow('med',median)
cv2.imshow('thresh',thresh)
#cv2.imshow('non_erodethresh',thresh)
#cv2.imshow('lap',lap)

cv2.waitKey(0)

cv2.destroyAllWindows()

