import cv2
import numpy as np
from scipy import signal
from skimage import metrics
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

img = cv2.imread("work_5\Lena.png", 0)
def dpcm_encode(img, qstep):

    rows, cols = img.shape
    pred_img = np.zeros((rows, cols))
    err_img = np.zeros((rows, cols))


    pred_img[0, :] = img[0, :]
    pred_img[:, 0] = img[:, 0]

    err_img[0, :] = img[0, :]
    err_img[:, 0] = img[:, 0]

    for i in range(1, rows):
        for j in range(1, cols):
            pred = (pred_img[i, j-1] + pred_img[i-1, j]) // 2
            err = img[i, j] - pred
            quant_err = np.round((err + 128) / qstep)
            rec_err = quant_err * qstep - 128
            pred_img[i, j] = pred
            err_img[i, j] = rec_err
    return err_img

def dpcm_decode(dpcm_img, qstep):
    rows, cols = dpcm_img.shape
    pred_img = np.zeros((rows, cols))
    rec_img = np.zeros((rows, cols))

    pred_img[0, :] = dpcm_img[0, :]
    pred_img[:, 0] = dpcm_img[:, 0]

    rec_img[0, :] = dpcm_img[0, :]
    rec_img[:, 0] = dpcm_img[:, 0]

    for i in range(1, rows):
        for j in range(1, cols):
            pred = (pred_img[i, j-1] + pred_img[i-1, j]) // 2

            rec_err = dpcm_img[i, j] * qstep

            rec = pred + rec_err

            pred_img[i, j] = pred
            rec_img[i, j] = rec

    return rec_img


qsteps = [1, 2, 4, 8]
num_steps = len(qsteps)

plt.figure(figsize=(10, 8))
rows = 2
cols = 3
plt.subplot(rows, cols, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

for i, qstep in enumerate(qsteps):
    dpcm_img = dpcm_encode(img, qstep)

    rec_img = dpcm_decode(dpcm_img, qstep)

    psnr_value = metrics.peak_signal_noise_ratio(img, rec_img, data_range=255)
    ssim_value = metrics.structural_similarity(img, rec_img, win_size=3, data_range=255)

    print('Quantization step: {}'.format(qstep))
    print('PSNR: {} dB'.format(psnr_value))
    print('SSIM: {}\n'.format(ssim_value))
    plt.subplot(rows, cols, i+2)
    plt.imshow(rec_img.astype(np.uint8), cmap='gray')
    plt.title('QStep={}'.format(qstep))

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show()

