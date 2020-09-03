from skimage import io , img_as_float
from matplotlib import pyplot as plt
import numpy as np
import cv2

img = cv2.imread("d1.jpeg")

plt.hist(img.ravel(),256,[0,256])
plt.show()

img2 = img.reshape((-1,3))

img2 = np.float32(img2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 , 1.0)

k = 5

attempts = 10

ret ,label,center = cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)

res = center[label.flatten()] 
res2 = res.reshape((img.shape))

cv2.imwrite('segmented_mine.png',res2)   



