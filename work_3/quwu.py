import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def zmMinFilterGray(src, r=7):
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):
    V1 = np.min(m, 2)
    Dark = zmMinFilterGray(V1, 7)
    V1 = guidedfilter(V1, Dark, r, eps)
    bins = 2000
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))
    return Y


if __name__ == '__main__':
    m1 = deHaze(cv2.imread("D:\\OIP-C (1).jpg") / 255.0) * 255
    m2 = deHaze(cv2.imread("D:\\tuxiang.png")/ 255.0) * 255


img1 =cv2.imread("D:\\OIP-C (1).jpg",flags=1)
img2 =cv2.imread("D:\\tuxiang.png",flags=1)
m1 = m1.astype(np.uint8)
m2 = m2.astype(np.uint8)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
m1 = cv2.cvtColor(m1,cv2.COLOR_BGR2RGB)
m2 = cv2.cvtColor(m2,cv2.COLOR_BGR2RGB)


plt.figure(num="去雾")
titles=["图1","去雾",'图2','去雾']
images=[img1,m1,img2,m2]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()