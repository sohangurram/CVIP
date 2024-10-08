import numpy as np
import cv2
import sys
import math
from svm import *

SIZ = 20
bin_n = 8
NUMFARMECHANGE = 10

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

svm = cv2.ml.SVM_load('svm_data.dat')
cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print ("Webcam not connected. \n")
    sys.exit()

def face_detection(frame):
    gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_f, minSize=(120, 120))
    area_max = -1
    area_max_index = None
    for (x, y, w, h) in faces:
        if w * h > area_max:
            area_max = w * h
            area_max_index = (x, y, w, h)
    if area_max_index is None:
        return None
    return list(area_max_index)

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SIZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SIZ, SIZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def hog(img):
    xg = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    yg = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(xg, yg)
    bins = np.int32(bin_n*ang/(2*np.pi))
    cell_bin = bins[:8, :8], bins[:8, 8:16], bins[:8, 16:24], bins[:8, 24:], bins[8:16, :8], bins[8:16, 8:16], bins[8:16, 16:24], bins[8:16, 24:], bins[16:24, :8], bins[16:24, 8:16], bins[16:24, 16:24], bins[16:24, 24:], bins[24:, :8], bins[24:, 8:16], bins[24:, 16:24], bins[24:, 24:]
    cell_mag = mag[:8, :8], mag[:8, 8:16], mag[:8, 16:24], mag[:8, 24:], mag[8:16, :8], mag[8:16, 8:16], mag[8:16, 16:24], mag[8:16, 24:], mag[16:24, :8], mag[16:24, 8:16], mag[16:24, 16:24], mag[16:24, 24:], mag[24:, :8], mag[24:, 8:16], mag[24:, 16:24], mag[24:, 24:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(cell_bin, cell_mag)]
    histo = np.hstack(hists)
    histo = histo / math.sqrt(sum(hist ** 2))
    return histo

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
result = []
while(True):
    ret, frame = cap.read()

    # Smoothing image
    blur = cv2.GaussianBlur(frame, (5, 5), 3.0, 3.0)
    # Covert hsv
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Detect face
    face = face_detection(blur)
    skin_min = (0, 10, 60)
    skin_max = (20, 150, 255)

    rangeRes = cv2.inRange(hsv_frame, skin_min, skin_max)
    erode_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_e = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    rangeRes = cv2.erode(rangeRes, erode_e, iterations = 2)
    rangeRes = cv2.dilate(rangeRes, dilate_e, iterations = 2)

    if not (face is None):
        rangeRes[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = 0

    im2, contours, hierarchy = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    max_area_index = -1
    for i in range(len(contours)):
        contour_area = cv2.contourArea(contours[i])
        if contour_area > max_area:
            max_area = contour_area
            max_area_index = i
    if max_area >= 10000:
        x, y, w, h = cv2.boundingRect(contours[max_area_index])
        skin = cv2.bitwise_and(frame, frame, mask = rangeRes)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = skin[y:y + h, x:x + w]
        h_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        h_gray = cv2.resize(h_gray,(32, 32), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("cropped", h_gray)
        hist = hog(h_gray)
        feature_v = np.float32(hist).reshape(-1, 128)
        labels = svm.predict(feature_v)
        result.append(labels[1][0][0])
        #print ('-------------', labels[1][0][0])
    if (len(result) >= NUMFARMECHANGE):
        counts=[]
        counts.append(result.count(1))
        counts.append(result.count(2))
        counts.append(result.count(3))
        counts.append(result.count(4))
        counts.append(result.count(5))
        result = counts.index(max(counts))+1
        if max(counts) > 0.7 * NUMFARMECHANGE:
            print (counts,counts.index(max(counts))+1)
        else:
            print (None)
        result = []
    cv2.imshow('rangeRes', rangeRes)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
