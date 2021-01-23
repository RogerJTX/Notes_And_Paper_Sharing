import cv2
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt

def get_imghist(img):
    # 判断图像是否为三通道；
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 无 Mask，256个bins,取值范围为[0,255]
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])


    return hist


def cal_equalhist(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    grathist = get_imghist(img)

    zerosumMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zerosumMoment[p] = grathist[0]
        else:
            zerosumMoment[p] = zerosumMoment[p - 1] + grathist[p]

    output_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zerosumMoment[p]) - 1
        if q >= 0:
            output_q[p] = math.floor(q)
        else:
            output_q[p] = 0

    equalhistimage = np.zeros(img.shape, np.uint8)   # 灰度直方图
    for i in range(h):
        for j in range(w):
            equalhistimage[i][j] = output_q[img[i][j]]

    # 第二种方法，opencv 库函数自带一种：
    # equalhistimage = cv2.equalizeHist(img)
    return equalhistimage

img = cv2.imread("007.jpg")
a = cal_equalhist(img)
print(a)
new_im = Image.fromarray(a)
# 显示图片
new_im.show()