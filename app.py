import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('camera4.jpg', 0)
img2 = img.copy()
template = cv.imread('camera4_part.jpg', 0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
methods2 = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']
methods3 = ['cv.TM_SQDIFF']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, 255, 16) #manage thickness here (16)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()


# Symperasma: apo olous tous algorithmous, o 2, 4, (5) kai 6 fernoun ta kalytera apotelesmata (oi NORMED)
# megalyteros paragontas to megethos ths eikonas (na einai idio - pixel matching) kai h katharothta
# 1 kai 2 deixnoun na leitourgoun kalytera gia to pazl, sygkrish aneksartita apto sxhma KAI megethos
