import cv2
import matplotlib.pyplot as plt 

img1 = cv2.imread('../img/picadef1.png')
img2 = cv2.imread('../img/picadef4.png')

#FAST Detector
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(0)
fast.setThreshold(50)

#Obtención de los keypoints
kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)

#Dibujar los keypoints en las imagenes
cv2.drawKeypoints(img1, kp1, img1, color=(255,0,0))
cv2.drawKeypoints(img2, kp2, img2, color=(255,0,0))

#Poner las imagenes en una ventana
fig, ax = plt.subplots(nrows=1,ncols=2)
plt.suptitle("FAST Detector\nThreshold = 50")
ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax[1].axis('off')

#Mostrar resultados
plt.show()