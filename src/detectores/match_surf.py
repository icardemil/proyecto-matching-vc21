import cv2
# Descargar una versión que tenga el método de SURF 😞.
"""
    img1 = cv2.imread("../img/picadef1.png")

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(8000)
    kp1, dp1 = surf.detectAndCompute(gray, None)

    img1 = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=kp1, flags=4, color=(51,163,236))
    print(surf.descriptorSurf())

    cv2.imshow('Keypoints', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""