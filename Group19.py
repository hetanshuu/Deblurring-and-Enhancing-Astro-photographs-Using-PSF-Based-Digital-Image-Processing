import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_float, restoration, exposure
from numpy.fft import fft2, ifft2
from astropy.io import fits

def make_preview(img, max_dim=2048):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        return transform.resize(img, (int(h * scale), int(w * scale)), anti_aliasing=True)
    return img.copy()


# ========== Utility Functions ==========

def read_fits(path):
    with fits.open(path) as hdul:
        data = None
        header = None
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None:
                data = hdu.data
                header = hdu.header
                break

        if data is None:
            raise ValueError(f"No image data found in any HDU of {path}")

    if hasattr(data, 'filled'):
        data = data.filled(np.nan)

    data = np.asarray(data)

    if data.dtype == object:
        try:
            data = data.astype(np.float64)
        except Exception:
            flat = []
            for v in data.flat:
                try:
                    flat.append(float(v))
                except Exception:
                    flat.append(0.0)
            data = np.array(flat, dtype=np.float64).reshape(data.shape)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    data = data.astype(np.float64, copy=False)
    return img_as_float(data)


# --- Define PSF (example: Gaussian 7x7) ---
def gaussian_psf(size=7, sigma=1.5):
    ax = np.linspace(-(size-1)/2, (size-1)/2, size)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    psf /= psf.sum()
    return psf

psf = gaussian_psf(size=7, sigma=1.2)

# --- Wiener filter ---
def wiener_filter(img, kernel, K=0.0001):
    kernel /= np.sum(kernel)
    dummy = fft2(img)
    kernel_fft = fft2(kernel, s=img.shape)
    wiener = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = np.abs(ifft2(dummy * wiener))
    return result

def preprocess_channel(img, psf, K=0.0001, num_iter=30):
    """Apply Wiener filter + RL deconvolution + contrast stretch."""
    wiener_out = wiener_filter(img, psf, K)
    p2, p98 = np.percentile(wiener_out, (60, 98))
    stretched = exposure.rescale_intensity(wiener_out, in_range=(p2, p98))

    deconv = restoration.richardson_lucy(stretched, psf, num_iter=num_iter, clip=True)
    p2, p98 = np.percentile(deconv, (60, 98))
    final = exposure.rescale_intensity(deconv, in_range=(p2, p98))
    return final


# ========== Load FITS Images (update paths accordingly) ==========
path_B = "C:/Users/pnpar/Downloads/Image_Stacking_Data/6/icdm12050_438_drc.fits"  # Blue channel
path_G = "C:/Users/pnpar/Downloads/Image_Stacking_Data/6/icdm12060_555_drc.fits"  # Green channel
path_R = "C:/Users/pnpar/Downloads/Image_Stacking_Data/6/icdm12070_814_drc.fits"  # Red channel
img_B = read_fits(path_B)
img_G = read_fits(path_G)
img_R = read_fits(path_R)

# ========== PSF and Processing ==========
psf = gaussian_psf(size=7, sigma=1.5)

B_proc = preprocess_channel(img_B, psf)
G_proc = preprocess_channel(img_G, psf)
R_proc = preprocess_channel(img_R, psf)

# ========== Stack into RGB ==========
rgb = cv2.merge((B_proc, G_proc, R_proc))

# ========== Display ==========
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(make_preview(img_R), cmap='gray')
plt.title("Raw HST (F814W - Red)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(make_preview(img_G), cmap='gray')
plt.title("Raw HST (F555W - Green)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(make_preview(img_B), cmap='gray')
plt.title("Raw HST (F435W - Blue)")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(make_preview(R_proc), cmap='gray')
plt.title("Processed HST (F814W - Red)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(make_preview(G_proc), cmap='gray')
plt.title("Processed HST (F555W - Green)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(make_preview(B_proc), cmap='gray')
plt.title("Raw HST (F435W - Blue)")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(make_preview(rgb))
plt.title("True-Color HST Composite (Wiener + RL)")
plt.axis('off')
plt.show()

# ========== Save ==========
cv2.imwrite("HST_ColorComposite.png", cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
