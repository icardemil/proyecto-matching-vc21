import cv2
import numpy as np

img1 = cv2.imread("../img/picadef2.png")
img2 = cv2.imread("../img/picadef4.png")

#ORB Detector
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1,des2)
"""
    print(len(matches))
    for m in matches:
        print(m.distance)
"""
matches = sorted(matches, key = lambda x:x.distance)
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

cv2.imshow("Detector ORB",result)
cv2.waitKey(0)
cv2.destroyAllWindows()