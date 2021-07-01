import cv2
import numpy as np

img1 = cv2.imread('img/picadef1.png')
img2 = cv2.imread('img/picadef4.png')

#Utilizando FAST para obtener los KP
fast = cv2.FastFeatureDetector_create()
kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)

#Obteniendo los descriptores con BRIEF
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
kpx, des1 = brief.compute(img1,kp1)
kpy, des2 = brief.compute(img2,kp2)

#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
result = cv2.drawMatches(img1, kpx, img2, kpy, matches[:30], None, flags=2)

#Mostrar resultado
cv2.imshow("BRIEF + FAST",result)
cv2.waitKey(0)
cv2.destroyAllWindows()