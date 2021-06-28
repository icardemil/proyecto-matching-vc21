import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("img/picadef1.png")
img2 = cv2.imread("img/picadef4.png")

#ORB Detector
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1,des2)

#Obtener las coordenadas del brute force matching
list_kp = []

for m in matches:
    img1_idx = m.queryIdx
    img2_idx = m.trainIdx

    #Coordenadas
    (x1,y1) = kp1[img1_idx].pt
    #(x2,y2) = kp2[img2_idx].pt
    list_kp.append((x1,y1))
    #list_kp.append((x2,y2))

#K-Means
list_kp = np.float32(np.vstack(list_kp))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

ret, label, center = cv2.kmeans(list_kp, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#aTX,aTY = center[1]
#print((aTX-25,aTY-25),(aTX-25,aTY+25),(aTX+25,aTY-25),(aTX+25,aTY+25))

A = list_kp[label.ravel() == 0]
#B = list_kp[label.ravel() == 1]

fig = plt.figure(figsize=(12, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

ax = plt.subplot(1, 2, 1)
plt.scatter(list_kp[:, 0], list_kp[:, 1], c='c')
plt.title("data")

ax = plt.subplot(1, 2, 2)
plt.scatter(A[:, 0], A[:, 1], c='b')
#plt.scatter(B[:, 0], B[:, 1], c='g')
plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
plt.title("clustered data and centroids (K = 1)")

plt.show()