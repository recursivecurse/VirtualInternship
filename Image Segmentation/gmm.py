import cv2
from skimage import io
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM

img = cv2.imread("d2.jpeg")
img2 = img.reshape((-1,3))

plt.imshow(img)

gmm_model = GMM(n_components=2, covariance_type='tied').fit(img2)

gmm_labels = gmm_model.predict(img2)

original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0],original_shape[1])

cv2.imwrite("GMM_Segmented.jpeg",segmented)

