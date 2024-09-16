import numpy as np
import cv2
import sys
import math

SIZ = 20
num_samples = []
numbs=1
tV=1
label_num=[]
bin_n = 8

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    s_skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, s_skew, -0.5*SIZ*s_skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SIZ, SIZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def hog(img):
    xg = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    yg = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(xg, yg)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:8, :8], bins[:8, 8:16], bins[:8, 16:24], bins[:8, 24:], bins[8:16, :8], bins[8:16, 8:16], bins[8:16, 16:24], bins[8:16, 24:], bins[16:24, :8], bins[16:24, 8:16], bins[16:24, 16:24], bins[16:24, 24:], bins[24:, :8], bins[24:, 8:16], bins[24:, 16:24], bins[24:, 24:]
    mag_cells = mag[:8, :8], mag[:8, 8:16], mag[:8, 16:24], mag[:8, 24:], mag[8:16, :8], mag[8:16, 8:16], mag[8:16, 16:24], mag[8:16, 24:], mag[16:24, :8], mag[16:24, 8:16], mag[16:24, 16:24], mag[16:24, 24:], mag[24:, :8], mag[24:, 8:16], mag[24:, 16:24], mag[24:, 24:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    histo = np.hstack(hists)
    histo = histo / math.sqrt(sum(hist ** 2))
    return histo

while (numbs<=1000):
    MIN_H_SKIN = (0, 10, 60)
    MAX_H_SKIN = (20, 150, 255)

    if tV*200+1 == numbs:
        tV= tV+1

    name  = 'images/1001_'+str(tV)+'_'+str(numbs)+'.png'
    print (name)
    res = cv2.imread(name,0)
    im2, contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_max = -1
    area_max_index = -1
    for i in range(len(contours)):
        a_contour = cv2.contourArea(contours[i])
        if a_contour > area_max:
            area_max = a_contour
            area_max_index = i
    if area_max != -1:
        x, y, w, h = cv2.boundingRect(contours[area_max_index])
        img_cropped = res[y:y+h, x:x+w]
        img_cropped = cv2.resize(img_cropped,(32, 32), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("cropped", img_cropped)
        cv2.waitKey(0)

    histo = hog(img_cropped)
    num_samples.append(histo)
    label_num.append(tV)
    numbs= numbs+1


num_samples = np.float32(num_samples)
label_num = np.array(label_num)

print ('samples',num_samples.size,label_num.size)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(num_samples, cv2.ml.ROW_SAMPLE,label_num)
svm.save('svm_data.dat')

