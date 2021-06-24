import numpy as np
import cv2
import matplotlib.pyplot as plt

imagen_ori = cv2.imread("img/picadef1.png")
imagen_rgb = cv2.cvtColor(imagen_ori,cv2.COLOR_BGR2RGB)

vector_imagen = imagen_rgb.reshape((-1,3))

vector_imagen = np.float32(vector_imagen)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 3
attempts = 10
ret,label,center = cv2.kmeans(vector_imagen,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

res = center[label.flatten()]
result_image = res.reshape((imagen_rgb.shape))

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(imagen_rgb)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()
