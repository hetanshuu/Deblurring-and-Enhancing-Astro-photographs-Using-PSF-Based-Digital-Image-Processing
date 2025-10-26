import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_float, restoration, exposure
from numpy.fft import fft2, ifft2
from astropy.io import fits

# --- Load Image ---
with fits.open("/content/Hst3-6.fits") as hdul:
    img_data = hdul[0].data

astro = img_as_float(img_data)

# --- Define PSF (example: Gaussian 7x7) ---
def gaussian_psf(size=7, sigma=1.2):
    ax = np.linspace(-(size-1)/2, (size-1)/2, size)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    psf /= psf.sum()
    return psf

psf = gaussian_psf(size=7, sigma=1.2)

# --- Wiener filter ---
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = fft2(img)
    kernel_fft = fft2(kernel, s=img.shape)
    wiener = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = np.abs(ifft2(dummy * wiener))
    return result

astro_wiener = wiener_filter(astro, psf, K=0.0001)

# --- Contrast stretching to enhance faint regions ---
p2, p98 = np.percentile(astro_wiener, (2, 98))  # discard outliers
astro_stretched = exposure.rescale_intensity(astro_wiener, in_range=(p2, p98))

# --- Richardson-Lucy Deconvolution ---
deconv_rl = restoration.richardson_lucy(astro_stretched, psf, num_iter=30, clip=True)

# --- Optional: contrast stretch the final output again ---
p2, p98 = np.percentile(deconv_rl, (2, 98))
deconv_rl_stretched = exposure.rescale_intensity(deconv_rl, in_range=(p2, p98))

# --- Display Results ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(astro, cmap='gray')
plt.title("Original HST Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(astro_stretched, cmap='gray')
plt.title("After Wiener + Contrast Stretch")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(deconv_rl_stretched, cmap='gray')
plt.title("After RL Deconvolution + Stretch")
plt.axis('off')

plt.tight_layout()
plt.show()

# --- Save outputs ---
cv2.imwrite("Original.png", (astro * 255).astype(np.uint8))
cv2.imwrite("Wiener_Contrast.png", (astro_stretched * 255).astype(np.uint8))
cv2.imwrite("Wiener_RL_Contrast.png", (deconv_rl_stretched * 255).astype(np.uint8))