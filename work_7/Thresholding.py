import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

img = cv2.imread('work_7\Lena.png', 0)

#大津法
ret, thres_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#迭代法
bin_count, bins = np.histogram(img, bins=256)
bin_idx = np.argmax(bin_count)
left_idx = np.argmax(bin_count[:bin_idx])
right_idx = np.argmax(bin_count[bin_idx:]) + bin_idx

threshold = (bins[left_idx] + bins[right_idx]) / 2
prev_threshold = None
while prev_threshold is None or abs(threshold - prev_threshold) > 1e-5:
    thres_iter = (img >= threshold).astype(np.uint8) * 255
    prev_threshold = threshold
    threshold = (np.mean(img[thres_iter == 0]) + np.mean(img[thres_iter == 255])) / 2

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
ax1.imshow(img, cmap='gray')
ax1.set_title('原图')
ax2.imshow(thres_otsu, cmap='gray')
ax2.set_title('大津法')
ax3.imshow(thres_iter, cmap='gray')
ax3.set_title('迭代法')
plt.show()