import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("img/picadef1.png")
img2 = cv2.imread("img/picadef3.png")

#ORB Detector
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1,des2)

#Obtener las coordenadas del brute force matching
list_kp1 = []
list_kp2 = []

for m in matches:
    img1_idx = m.queryIdx
    img2_idx = m.trainIdx

    #Coordenadas
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt
    list_kp1.append((x1,y1))
    list_kp2.append((x2,y2))

#K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

#Primera imagen
list_kp1 = np.float32(np.vstack(list_kp1))
ret_1, label_1, center_1 = cv2.kmeans(list_kp1, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#Seguna imagen
list_kp2 = np.float32(np.vstack(list_kp2))
ret_2, label_2, center_2 = cv2.kmeans(list_kp2, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

A_1 = list_kp1[label_1.ravel() == 0]
A_2 = list_kp2[label_2.ravel() == 0]

fig = plt.figure(figsize=(12, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

ax = plt.subplot(1, 2, 2)
plt.scatter(A_2[:, 0], A_2[:, 1], c='b')
plt.scatter(center_2[:, 0], center_2[:, 1], s=100, c='m', marker='s')
plt.title("IMG 2")

ax = plt.subplot(1, 2, 1)
plt.scatter(A_1[:, 0], A_1[:, 1], c='b')
plt.scatter(center_1[:, 0], center_1[:, 1], s=100, c='m', marker='s')
plt.title("IMG 1")

plt.show()