import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Recibe dos puntos (P,Q) y los centroides de cada uno para calcular la distancia
# y realizar la comparación para filtrar la lista de puntos con los que se realizará
# el matching.
def validarPuntos(p,q,c_1,c_2):
    p_x,p_y,p_i = p
    q_x,q_y,q_i = q
    c_x_p,c_y_p,c_i_p = c_1[0]
    c_x_q,c_y_q,c_i_q = c_2[0]
    if (math.dist([p_x,p_y],[c_x_p,c_y_p]) - math.dist([q_x,q_y],[c_x_q,c_y_q])) < 15:
        return True
    else:
        return False

img1 = cv2.imread("img/picadef1.png")
img2 = cv2.imread("img/picadef4.png")

#ORB Detector
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)
#print(kp1.__dir__())
#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1,des2)

#Obtener las coordenadas del brute force matching
list_kp1 = []
list_kp2 = []
for m in matches:
    img1_index = m.queryIdx
    img2_index = m.trainIdx
    #Coordenadas
    (x1,y1) = kp1[img1_index].pt
    (x2,y2) = kp2[img2_index].pt
    list_kp1.append((x1,y1,img1_index))
    list_kp2.append((x2,y2,img2_index))


#K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

#Primera imagen
list_kp1 = np.float32(np.vstack(list_kp1))
ret_1, label_1, center_1 = cv2.kmeans(list_kp1, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#Seguna imagen
list_kp2 = np.float32(np.vstack(list_kp2))
ret_2, label_2, center_2 = cv2.kmeans(list_kp2, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#Calcular las distancia
list_validos = []
for i in range(len(list_kp1)):
    if validarPuntos(list_kp1[i],list_kp2[i],center_1,center_2):
        list_validos.append(matches[i])
    
print(len(matches))
print(len(list_validos))

result = cv2.drawMatches(img1, kp1, img2, kp2, list_validos, None, flags=2)

cv2.imshow("Detector ORB",result)
cv2.waitKey(0)
cv2.destroyAllWindows()