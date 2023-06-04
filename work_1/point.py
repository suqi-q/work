import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import log10
img=cv2.imread('work_1/think.jpg',0)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

height=img.shape[0]
width=img.shape[1]

result=np.zeros((height,width),np.uint8)
result1=np.zeros((height,width),np.uint8)

for i in range(height):
    for j in range(width):
        gray=-(img[i,j])+255
        result[i,j]=np.uint8(gray)
img1=result

for i in range(height):
    for j in range(width):
        if (img[i,j])+150>255:
            gray=255
        else:
            gray=(img[i,j])+150
        result1[i,j]=np.uint8(gray)
img2=result1

img=np.double(img)
result=img**0.5
result=np.uint8(result*255/np.max(result))
img3=result

img=np.double(img)
result=np.log10(img+1)
result=np.uint8(result*255/np.max(result))
img4=result

def PL(img,x1,x2,y1,y2):
    lut=np.zeros(256)
    for i in range(256):
        if i<x1:
            lut[i]=(y1/x1)*i
        elif i<x2:
            lut[i]=((y2-y1)/(x2-x1))*(i-x1)+y1
        else:
            lut[i]=((y2-255.0)/(x2-255.0))*(i-255.0)+255.0
    result=cv2.LUT(img,lut)
    result=np.uint8(result+0.5)
    return result

plt.figure(num="结果")
titles=["原图",'a=-1,b=0','b=150,a=1','r=0.5','log',"分段线性"]
img=cv2.imread('work_1/think.jpg',cv2.IMREAD_GRAYSCALE)
img_x1=100
img_x2=160
img_y1=30
img_y2=200
img5=PL(img,img_x1,img_x2,img_y1,img_y2)
images=[img,img1,img2,img3,img4,img5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
