from PIL import Image
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

img=Image.open("D:\\R-C.jpg")
img1=img.rotate(-30)
img2=img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

pic=cv2.imread("D:\\R-C.jpg")
image=cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)

M = np.float32([[1,0,0],[0,1,100]])
img3=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
M = np.float32([[1,0,100],[0,1,0]])
img4=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
M = np.float32([[1,0,100],[0,1,100]])
img5=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))

plt.figure(num="结果")
titles=["原图","旋转30°",'翻转','向下移动','向右移动','向右下移动']
images=[img,img1,img2,img3,img4,img5]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()