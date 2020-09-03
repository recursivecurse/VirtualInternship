import cv2
from skimage import io
from matplotlib import pyplot as plt

img = cv2.imread("mine2.png",1)
plt.imshow(img)

lab_image = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

l,a,b = cv2.split(lab_image)

equ = cv2.equalizeHist(l)

updated_lab_img1 = cv2.merge((equ,a,b))
hist_eq_image = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)


clahe = cv2.createCLAHE(clipLimit = 3.0,tileGridSize=(8,8))
clahe_image = clahe.apply(l)

updated_lab_img2 = cv2.merge((clahe_image,a,b))

CLAHE_img = cv2.cvtColor(updated_lab_img2,cv2.COLOR_LAB2BGR)


cv2.imshow("OriginalImage",img)
cv2.imshow("EqualisedImage",hist_eq_image)
cv2.imshow("Clahe Image",CLAHE_img)
cv2.imwrite('hist_eq_mine2.png',hist_eq_image)
cv2.imwrite('clahe_mine2.png',CLAHE_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
