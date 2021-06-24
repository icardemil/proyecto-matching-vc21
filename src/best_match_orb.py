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
matches = sorted(matches, key = lambda x:x.distance)

#Obtener las coordenadas del brute force matching
list_kp1 = []
list_kp2 = []
labels_dict = {1: 'red', 2: 'blue'}

for m in matches:
    img1_idx = m.queryIdx
    img2_idx = m.trainIdx

    #Coordenadas
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt
    list_kp1.append((x1,y1))
    list_kp2.append((x2,y2))

    plt.scatter(x1, y1, color="red", label='Imagen 1')
    plt.scatter(x2, y2, color="blue", label='Imagen 2')


plt.show()

""""
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

cv2.imshow("Detector ORB",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""