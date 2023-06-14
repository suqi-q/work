import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def generate_noise(img, sigma):
    noise = np.random.randn(*img.shape) * sigma
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img, noise

def weiner_filter(img, noise, snr):
    N = np.fft.fft2(noise)
    N_abs = np.abs(N) ** 2
    Pn = np.sum(N_abs) / (N.shape[0] * N.shape[1])

    Rf = np.fft.ifft2(np.abs(np.fft.fft2(img)) ** 2)
    Rn = np.fft.ifft2(N_abs)

    H = (np.conj(np.fft.fft2(img)) / (np.abs(np.fft.fft2(img)) ** 2 + snr * Pn)) * np.fft.fft2(img)
    F = np.fft.ifft2(H * np.fft.fft2(noise))
    restored_img = img + np.real(F)

    return restored_img

def mmse_filter(img, noise, snr):
    N = np.fft.fft2(noise)
    N_abs = np.abs(N) ** 2
    Pn = np.sum(N_abs) / (N.shape[0] * N.shape[1])

    Rf = np.fft.ifft2(np.abs(np.fft.fft2(img)) ** 2)
    Rn = np.fft.ifft2(N_abs)

    H = ((np.abs(np.fft.fft2(img)) ** 2) / (np.abs(np.fft.fft2(img)) ** 2 + snr * Pn)) * np.conj(np.fft.fft2(img)) / (np.abs(np.fft.fft2(img)) ** 2 + snr * Pn + Pn / snr)
    F = np.fft.ifft2(H * np.fft.fft2(noise))
    restored_img = img + np.real(F)

    return restored_img

def optimal_filter(img, noise):
    N = np.fft.fft2(noise)
    N_abs = np.abs(N) ** 2
    Pn = np.sum(N_abs) / (N.shape[0] * N.shape[1])

    Rf = np.fft.ifft2(np.abs(np.fft.fft2(img)) ** 2)
    Rn = np.fft.ifft2(N_abs)

    H = np.conj(np.fft.fft2(img)) / (np.abs(np.fft.fft2(img)) ** 2 + Pn / np.abs(np.fft.fft2(img)) ** 2) * np.fft.fft2(img)
    F = np.fft.ifft2(H * np.fft.fft2(noise))
    restored_img = img + np.real(F)

    return restored_img

img = cv2.imread('work_4\Lena.png', 0)

sigma = 30
noisy_img, noise = generate_noise(img, sigma)

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('原图'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(noisy_img,cmap='gray')
plt.title('噪声图像'), plt.xticks([]), plt.yticks([])
plt.show()

restored_img_weiner_unknown = weiner_filter(img, noise, 0.1)
restored_img_mmse_unknown = mmse_filter(img, noise, 0.1)
restored_img_optimal_unknown = optimal_filter(img, noise)

plt.subplot(131),plt.imshow(restored_img_weiner_unknown,cmap='gray')
plt.title('Restored (Weiner, 信噪比未知)',fontsize=8), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(restored_img_mmse_unknown,cmap='gray')
plt.title('Restored (MMSE, 信噪比未知)',fontsize=8), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(restored_img_optimal_unknown,cmap='gray')
plt.title('Restored (最优的)',fontsize=8), plt.xticks([]), plt.yticks([])
plt.show()

snr = sigma**2 / np.var(noise)
restored_img_weiner_known = weiner_filter(img, noise, snr)
restored_img_mmse_known = mmse_filter(img, noise, snr)

plt.subplot(121),plt.imshow(restored_img_weiner_known,cmap='gray')
plt.title('Restored (Weiner, 信噪比已知)'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(restored_img_mmse_known,cmap='gray')
plt.title('Restored (MMSE,信噪比已知)'), plt.xticks([]), plt.yticks([])
plt.show()

Rf = np.fft.ifft2(np.abs(np.fft.fft2(img)) ** 2)
Rn = np.fft.ifft2(np.abs(np.fft.fft2(noise)) ** 2)
snr_map = np.abs(Rf) / np.abs(Rn)
snr_map /= np.max(snr_map)

restored_img_weiner_known_acf = weiner_filter(img, noise, snr_map)
restored_img_mmse_known_acf = mmse_filter(img, noise, snr_map)

plt.subplot(121),plt.imshow(restored_img_weiner_known_acf,cmap='gray')
plt.title('Restored (Weiner, 自函数已知)'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(restored_img_mmse_known_acf,cmap='gray')
plt.title('Restored (MMSE, 自函数已知)'), plt.xticks([]), plt.yticks([])
plt.show()