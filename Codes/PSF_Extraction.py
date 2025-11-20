import os
import glob
import argparse
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com
from scipy.ndimage import fourier_shift
from scipy.fft import fftn, ifftn
from scipy.signal import fftconvolve, correlate
from skimage.restoration import richardson_lucy
from astropy.convolution import convolve_fft
from astropy.io import fits
from imageio import imread
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------

def read_image_any(path):
    """Read a FITS or common image file and return a 2D float array."""
    if path.lower().endswith(('.fits', '.fit', '.fz')):
        with fits.open(path) as h:
            # prefer primary or first non-empty extension
            data = h[0].data
            if data is None:
                for hdu in h:
                    if hdu.data is not None:
                        data = hdu.data
                        break
            if data is None:
                raise ValueError(f"No image data found in FITS: {path}")
            return np.array(data, dtype=float)
    else:
        arr = imread(path)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return np.array(arr, dtype=float)


def center_stamp_subpixel(stamp):
    """Center a stamp to its centroid using Fourier shift (subpixel).
    stamp: 2D array
    returns: recentered stamp
    """
    yc, xc = centroid_com(stamp)
    cy, cx = np.array(stamp.shape) / 2.0
    shift = (cy - yc, cx - xc)
    # Fourier shift
    F = fftn(stamp)
    fy = np.fft.fftfreq(stamp.shape[0])[:, None]
    fx = np.fft.fftfreq(stamp.shape[1])[None, :]
    phase = np.exp(-2j * np.pi * (shift[0] * fy + shift[1] * fx))
    recentered = np.real(ifftn(F * phase))
    return recentered


def crop_region(image, y0, y1, x0, x1, pad_value=0.0):
    """Crop a rectangular region from `image` safely with optional padding.

    Returns a 2D numpy array with shape (y1-y0, x1-x0). Handles edges by
    filling out-of-bounds areas with `pad_value`.
    """
    h, w = image.shape
    out_h = y1 - y0
    out_w = x1 - x0
    out = np.full((out_h, out_w), pad_value, dtype=float)
    y0_src = max(0, y0)
    y1_src = min(h, y1)
    x0_src = max(0, x0)
    x1_src = min(w, x1)
    y0_dst = y0_src - y0
    x0_dst = x0_src - x0
    out[y0_dst:y0_dst + (y1_src - y0_src), x0_dst:x0_dst + (x1_src - x0_src)] = image[y0_src:y1_src, x0_src:x1_src]
    return out


def aperture_photometry_annulus(image, positions, r_ap=5.0, r_in=8.0, r_out=12.0):
    """Aperture photometry with annulus-based local background subtraction.

    Parameters
    - image: 2D array
    - positions: sequence of (x, y) coordinates
    - r_ap: aperture radius
    - r_in, r_out: inner/outer annulus radii

    Returns
    - dict with 'table' (photutils result table) and 'net_flux' (numpy array)
    """
    apertures = CircularAperture(positions, r=r_ap)
    annuli = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    ap_tab = aperture_photometry(image, apertures)
    # compute annulus statistics for local background
    aperstats = ApertureStats(image, annuli)
    bkg_mean = aperstats.mean
    # compute overlapping areas for each aperture (handles edges)
    ap_area = apertures.area_overlap(image)
    total_bkg = np.array(bkg_mean) * np.array(ap_area)
    ap_tab['total_bkg'] = total_bkg
    # net flux after background subtraction
    ap_tab['net_flux'] = ap_tab['aperture_sum'] - ap_tab['total_bkg']
    return {'table': ap_tab, 'net_flux': np.array(ap_tab['net_flux'])}


def show_image_with_apertures(section, positions=None, aperture_r=5.0, annulus_r=(8.0,12.0), cmap='Greys'):
    """Display `section` with apertures and annuli overlay using LogNorm scaling.

    This mirrors the visualization style in the notebooks (Greys + LogNorm).
    """
    from matplotlib.colors import LogNorm
    plt.figure()
    plt.imshow(section, origin='lower', cmap=cmap, norm=LogNorm())
    if positions is not None and len(positions) > 0:
        apertures = CircularAperture(positions, r=aperture_r)
        apertures.plot(color='blue', lw=1.5, alpha=0.6)
        if annulus_r is not None:
            ann = CircularAnnulus(positions, r_in=annulus_r[0], r_out=annulus_r[1])
            ann.plot(color='green', lw=1.0, alpha=0.5)
    plt.colorbar()
    plt.show()


def fit_gaussian_2d(stamp):
    """Fit an astropy Gaussian2D to a stamp and return the fitted model and model image."""
    total = stamp.sum()
    if total <= 0:
        raise ValueError("Stamp has non-positive sum.")
    y, x = np.indices(stamp.shape)
    yc, xc = centroid_com(stamp)
    amp = stamp.max()
    x_var = np.sum((x - xc)**2 * stamp) / total
    y_var = np.sum((y - yc)**2 * stamp) / total
    x_std = np.sqrt(max(x_var, 0.5))
    y_std = np.sqrt(max(y_var, 0.5))

    g_init = models.Gaussian2D(amplitude=amp, x_mean=xc, y_mean=yc,
                               x_stddev=x_std, y_stddev=y_std, theta=0.0)
    fitter = fitting.LevMarLSQFitter()
    try:
        yy = y.ravel(); xx = x.ravel()
        stamp_flat = stamp.ravel()
        g_fit = fitter(g_init, xx, yy, stamp_flat)
    except Exception:
        g_fit = g_init
    model = g_fit(x, y)
    return g_fit, model


def shift_stamp_fft(stamp, shift_yx):
    """Shift stamp by (dy, dx) using Fourier shift. shift_yx: (dy, dx)."""
    F = fftn(stamp)
    fy = np.fft.fftfreq(stamp.shape[0])[:, None]
    fx = np.fft.fftfreq(stamp.shape[1])[None, :]
    ky = 2j * np.pi * (shift_yx[0] * fy)
    kx = 2j * np.pi * (shift_yx[1] * fx)
    Fshift = F * np.exp(-(ky + kx))
    return np.real(ifftn(Fshift))


def chi_square_residual(original, deconvolved, psf, noise_sigma=None):
    """Compute reduced chi-square between original and reconvolved(deconvolved).

    Parameters
    - original: 2D np.ndarray (background-subtracted)
    - deconvolved: 2D np.ndarray
    - psf: 2D np.ndarray (normalized)
    - noise_sigma: float or None (if None, estimate from MAD of residual)

    Returns
    - chi2_nu: float (reduced chi-square)
    """
    # reconvolution = forward model
    reconv = fftconvolve(deconvolved, psf, mode='same')

    # residual
    residual = original - reconv

    # estimate noise if not provided (robust MAD -> sigma)
    if noise_sigma is None:
        mad = np.median(np.abs(residual - np.median(residual)))
        noise_sigma = 1.4826 * mad

    noise_sigma = max(noise_sigma, 1e-8)
    chi2 = np.sum((residual / noise_sigma) ** 2)

    # degrees of freedom ~ number of pixels
    dof = original.size
    return chi2 / dof


def flux_conservation(original, deconvolved, mask=None):
    """Compute fractional absolute flux difference between original and deconvolved.

    Returns (ratio, F_orig, F_deconv) where ratio = |F_deconv - F_orig|/F_orig.
    """
    if mask is None:
        F_orig = np.sum(original)
        F_deconv = np.sum(deconvolved)
    else:
        F_orig = np.sum(original[mask])
        F_deconv = np.sum(deconvolved[mask])

    if F_orig == 0:
        return np.inf, F_orig, F_deconv
    ratio = np.abs(F_deconv - F_orig) / F_orig
    return ratio, F_orig, F_deconv


# -------------------------
# Main pipeline
# -------------------------

def build_average_psf(stars_folder, stamp_size=31, verbose=True, use_daofind=True,
                                  daofind_fwhm=2.5, daofind_thresh_sigma=5.0):
    """Load images from a folder, extract star stamps, center them, stack, and fit an average Gaussian PSF.

    Parameters
    ----------
    stars_folder : str
        Path to folder containing FITS or image files of isolated stars (each file may contain one star).
    stamp_size : int
        Size of square stamp to extract (odd recommended).
    use_daofind : bool
        If True, try to run DAOStarFinder on each image and use the best detected star; otherwise use brightest pixel.
    daofind_fwhm, daofind_thresh_sigma : float
        Parameters passed to DAOStarFinder (FWHM guess, detection threshold in sigma).

    Returns
    -------
    dict
        Contains 'stack_psf', 'gauss_psf', 'gauss_psf_from_params', 'fitted_params', 'kept_files'.
    """
    patterns = ['*.fits', '*.fit', '*.fz', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    files = []
    for patt in patterns:
        files += sorted(glob.glob(os.path.join(stars_folder, patt)))
    if len(files) == 0:
        raise FileNotFoundError(f"No star stamp images found in {stars_folder}")

    stamps = []
    fitted_params = []
    kept_files = []

    # Lazy import of DAOStarFinder to avoid hard dependency if not used
    if use_daofind:
        from photutils.detection import DAOStarFinder

    for f in files:
        try:
            img = read_image_any(f)
            # ensure 2D
            if img.ndim != 2:
                continue
            # Subtract simple local background from stamp-level image
            m, med, s = sigma_clipped_stats(img, sigma=3.0)
            img_bkgsub = img - med

            # Find star position
            if use_daofind:
                try:
                    daofind = DAOStarFinder(fwhm=daofind_fwhm, threshold=daofind_thresh_sigma * s)
                    sources = daofind(img_bkgsub)
                except Exception:
                    sources = None
                if sources is None or len(sources) == 0:
                    # fall back to brightest-pixel
                    yy, xx = np.unravel_index(np.nanargmax(img_bkgsub), img_bkgsub.shape)
                else:
                    # sort by flux and pick the brightest non-saturated candidate
                    sources.sort('flux')
                    brightest = sources[-1]
                    xx = brightest['xcentroid']; yy = brightest['ycentroid']
            else:
                yy, xx = np.unravel_index(np.nanargmax(img_bkgsub), img_bkgsub.shape)

            # Make a stamped cutout centered on (yy,xx)
            half = stamp_size // 2
            y0 = int(np.round(yy)) - half; y1 = int(np.round(yy)) + half + 1
            x0 = int(np.round(xx)) - half; x1 = int(np.round(xx)) + half + 1
            # handle edges with padding
            stamp = np.zeros((stamp_size, stamp_size), dtype=float)
            y0_src = max(0, y0); y1_src = min(img.shape[0], y1)
            x0_src = max(0, x0); x1_src = min(img.shape[1], x1)
            y0_dst = y0_src - y0; y1_dst = y0_dst + (y1_src - y0_src)
            x0_dst = x0_src - x0; x1_dst = x0_dst + (x1_src - x0_src)
            stamp[y0_dst:y1_dst, x0_dst:x1_dst] = img_bkgsub[y0_src:y1_src, x0_src:x1_src]

            # Clip negatives and subtract local median to reduce background bias
            m2, med2, s2 = sigma_clipped_stats(stamp, sigma=3.0)
            stamp = stamp - med2
            stamp[stamp < 0] = 0.0

            # center stamp with subpixel precision
            try:
                stamp_centered = center_stamp_subpixel(stamp)
            except Exception:
                stamp_centered = stamp

            total = stamp_centered.sum()
            if total <= 0:
                if verbose:
                    print(f"Skipping {os.path.basename(f)}: non-positive flux after bg-sub.")
                continue

            # normalize before stacking (so we capture shape not flux differences)
            stamp_norm = stamp_centered / total
            stamps.append(stamp_norm)
            kept_files.append(f)

            # fit gaussian to centered stamp and save params
            try:
                gfit, model = fit_gaussian_2d(stamp_centered)
                fitted_params.append({
                    'amplitude': float(gfit.amplitude.value),
                    'x_mean': float(gfit.x_mean.value),
                    'y_mean': float(gfit.y_mean.value),
                    'x_stddev': float(gfit.x_stddev.value),
                    'y_stddev': float(gfit.y_stddev.value),
                    'theta': float(getattr(gfit, 'theta', 0.0).value if hasattr(gfit, 'theta') else 0.0)
                })
            except Exception:
                # ignore fit failures for single stamps
                pass

            if verbose:
                print(f"Processed {os.path.basename(f)} (sum={total:.1f})")

        except Exception as e:
            if verbose:
                print(f"Skipping {os.path.basename(f)}: {e}")
            continue

    if len(stamps) == 0:
        raise RuntimeError("No usable star stamps found in folder.")

    # Stack stamps (mean) and fit Gaussian to stack
    stack = np.mean(np.stack(stamps), axis=0)
    gfit_stack, model_stack = fit_gaussian_2d(stack)
    gauss_psf = model_stack
    gauss_psf = gauss_psf / gauss_psf.sum()

    # Alternative: average parameters and synthesize gaussian PSF
    gauss_psf_from_params = None
    if len(fitted_params) > 0:
        keys = fitted_params[0].keys()
        avg_params = {k: np.mean([p[k] for p in fitted_params]) for k in keys}
        y, x = np.indices(stack.shape)
        g_avg = models.Gaussian2D(amplitude=avg_params['amplitude'],
                                  x_mean=avg_params['x_mean'],
                                  y_mean=avg_params['y_mean'],
                                  x_stddev=avg_params['x_stddev'],
                                  y_stddev=avg_params['y_stddev'],
                                  theta=avg_params.get('theta', 0.0))
        gauss_psf_from_params = g_avg(x, y)
        gauss_psf_from_params /= gauss_psf_from_params.sum()

    return {
        'stack_psf': stack,
        'gauss_psf': gauss_psf,
        'gauss_psf_from_params': gauss_psf_from_params,
        'fitted_params': fitted_params,
        'kept_files': kept_files
    }


def deconvolve_and_validate(test_image_path, psf_array, out_prefix='run', rl_iterations=30, npeaks=10):
    data = read_image_any(test_image_path)
    if data.ndim != 2:
        raise ValueError("Test image must be 2D.")
    img = data.astype(float)

    # background subtraction
    mean, med, std = sigma_clipped_stats(img, sigma=3.0)
    img_bkg = img - med
    img_bkg[img_bkg < 0] = 0.0

    # clean NaNs / infs which break FFT-based deconvolution routines
    finite_mask = np.isfinite(img_bkg)
    if not finite_mask.all():
        if finite_mask.any():
            fill_val = float(np.median(img_bkg[finite_mask]))
        else:
            fill_val = 0.0
        img_bkg_clean = img_bkg.copy()
        img_bkg_clean[~finite_mask] = fill_val
        print(f"Warning: input contained NaN/inf; filled with {fill_val:.4g} for deconvolution.")
    else:
        img_bkg_clean = img_bkg

    # Richardson-Lucy deconvolution
    # Attempt to align PSF to a bright star via cross-correlation (subpixel)
    psf_used = psf_array.copy()
    psf_shift = (0.0, 0.0)
    try:
        yi, xi = np.unravel_index(np.nanargmax(img_bkg_clean), img_bkg_clean.shape)
        # choose stamp size matching PSF
        ph, pw = psf_array.shape
        hy = ph // 2
        hx = pw // 2
        star_stamp = crop_region(img_bkg_clean, yi - hy, yi + hy + 1, xi - hx, xi + hx + 1, pad_value=0.0)
        # ensure same shape for correlation: if stamp or psf are different, pad smaller to match
        if star_stamp.shape != psf_array.shape:
            # pad to the larger shape
            maxh = max(star_stamp.shape[0], psf_array.shape[0])
            maxw = max(star_stamp.shape[1], psf_array.shape[1])
            # center both into new arrays
            def center_pad(arr, H, W):
                out = np.zeros((H, W), dtype=float)
                h, w = arr.shape
                y0 = (H - h)//2; x0 = (W - w)//2
                out[y0:y0+h, x0:x0+w] = arr
                return out
            star_pad = center_pad(star_stamp, maxh, maxw)
            psf_pad = center_pad(psf_array, maxh, maxw)
        else:
            star_pad = star_stamp
            psf_pad = psf_array

        corr = correlate(star_pad, psf_pad, mode='same')
        y0, x0 = np.unravel_index(np.nanargmax(corr), corr.shape)
        cy = corr.shape[0] // 2
        cx = corr.shape[1] // 2
        shift_y = cy - y0
        shift_x = cx - x0
        psf_shift = (float(shift_y), float(shift_x))
        # apply subpixel shift to PSF
        psf_used = shift_stamp_fft(psf_array, psf_shift)
        if True:
            print(f"Aligned PSF by shift (dy,dx)={psf_shift}")
    except Exception as e:
        if True:
            print(f"PSF alignment skipped/failed: {e}")

    # ensure PSF has no NaNs and is normalized
    psf_used = np.nan_to_num(psf_used, nan=0.0, posinf=0.0, neginf=0.0)
    psf_sum = psf_used.sum()
    if psf_sum <= 0:
        raise RuntimeError("PSF is empty or invalid after cleaning; cannot deconvolve.")
    psf_used = psf_used / psf_sum

    print(f"Running Richardson-Lucy with {rl_iterations} iterations...")
    # use a robust call that prefers direct convolution if FFT with NaNs could fail
    try:
        deconv = richardson_lucy(img_bkg_clean, psf_used, num_iter=rl_iterations, clip=False, method='direct')
    except TypeError:
        # older skimage versions don't accept 'method' kwarg
        deconv = richardson_lucy(img_bkg_clean, psf_used, num_iter=rl_iterations, clip=False)

    # Reconvolve to validate flux conservation (use aligned/clean PSF and cleaned original)
    reconv = convolve_fft(deconv, psf_used, allow_huge=True)
    residual = img_bkg_clean - reconv
    # compute goodness-of-fit and flux-conservation metrics
    try:
        chi2_nu = chi_square_residual(img_bkg_clean, deconv, psf_used)
    except Exception:
        chi2_nu = np.nan

    try:
        # flux conservation after reconvolution is a useful check
        flux_ratio_reconv, F_orig_reconv, F_reconv = flux_conservation(img_bkg_clean, reconv)
        flux_ratio_deconv, F_orig_deconv, F_deconv = flux_conservation(img_bkg_clean, deconv)
    except Exception:
        flux_ratio_reconv = flux_ratio_deconv = np.nan
        F_orig_reconv = F_reconv = F_orig_deconv = F_deconv = np.nan
    # Aperture photometry comparison on top npeaks
    flat = img_bkg_clean.copy()
    flat_copy = flat.copy()
    positions = []
    for i in range(npeaks):
        idx = np.nanargmax(flat_copy)
        yi, xi = np.unravel_index(idx, flat_copy.shape)
        positions.append((xi, yi))
        ry = 6; rx = 6
        y0 = max(0, yi-ry); y1 = min(flat_copy.shape[0], yi+ry+1)
        x0 = max(0, xi-rx); x1 = min(flat_copy.shape[1], xi+rx+1)
        flat_copy[y0:y1, x0:x1] = 0.0

    apertures = CircularAperture(positions, r=4.0)
    orig_flux_tab = aperture_photometry(img_bkg_clean, apertures)
    reconv_flux_tab = aperture_photometry(reconv, apertures)

    orig_fluxes = np.array(orig_flux_tab['aperture_sum'])
    reconv_fluxes = np.array(reconv_flux_tab['aperture_sum'])
    mask = orig_fluxes > 0
    ratios = np.ones_like(orig_fluxes)
    if mask.any():
        ratios[mask] = reconv_fluxes[mask] / orig_fluxes[mask]

    print("Median flux ratio (reconv / orig) for top peaks:", np.median(ratios[mask]) if mask.any() else np.nan)

    # Save quick preview
    plt.figure(figsize=(12,4))
    v1 = np.percentile(img_bkg_clean, 99)
    v2 = np.percentile(deconv, 99)
    v3 = np.percentile(reconv, 99)
    plt.subplot(1,3,1); plt.imshow(img_bkg_clean, origin='lower', cmap='gray', vmax=v1); plt.title('Original (bkg-sub)')
    plt.subplot(1,3,2); plt.imshow(deconv, origin='lower', cmap='gray', vmax=v2); plt.title('Deconvolved')
    plt.subplot(1,3,3); plt.imshow(reconv, origin='lower', cmap='gray', vmax=v3); plt.title('Reconvolved(deconv)')
    plt.tight_layout()
    preview_png = f"{out_prefix}_deconv_preview.png"
    plt.savefig(preview_png, dpi=150)
    print("Saved preview PNG:", preview_png)
    deconv_clean = np.nan_to_num(deconv, nan=0.0, posinf=0.0, neginf=0.0)
    reconv_clean = np.nan_to_num(reconv, nan=0.0, posinf=0.0, neginf=0.0)
    residual_clean = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    # Save as FITS
    fits.writeto(f"{out_prefix}_deconvolved.fits", deconv_clean, overwrite=True)
    fits.writeto(f"{out_prefix}_reconvolved.fits", reconv_clean, overwrite=True)
    fits.writeto(f"{out_prefix}_residual.fits", residual_clean, overwrite=True)

    print("Saved:")
    print(f"  {out_prefix}_deconvolved.fits")
    print(f"  {out_prefix}_reconvolved.fits")
    print(f"  {out_prefix}_residual.fits")
    plt.imsave(f"{out_prefix}_deconvolved.png", deconv_clean, cmap='gray')
    plt.imsave(f"{out_prefix}_reconvolved.png", reconv_clean, cmap='gray')
    plt.imsave(f"{out_prefix}_residual.png", residual_clean, cmap='gray')

    # report metrics
    print(f"Reduced chi-square (original vs reconv): {chi2_nu:.4g}")
    print(f"Flux ratio (reconv vs orig): {flux_ratio_reconv:.4g}  (F_orig={F_orig_reconv:.4g}, F_reconv={F_reconv:.4g})")
    print(f"Flux ratio (deconv vs orig): {flux_ratio_deconv:.4g}  (F_orig={F_orig_deconv:.4g}, F_deconv={F_deconv:.4g})")

    return {
        'img_bkg': img_bkg_clean,
        'deconv': deconv,
        'reconv': reconv,
        'positions': positions,
        'orig_fluxes': orig_fluxes,
        'reconv_fluxes': reconv_fluxes,
        'flux_ratios': ratios,
        'chi2_nu': chi2_nu,
        'flux_ratio_reconv': flux_ratio_reconv,
        'flux_ratio_deconv': flux_ratio_deconv,
        'F_orig_reconv': F_orig_reconv,
        'F_reconv': F_reconv,
        'F_orig_deconv': F_orig_deconv,
        'F_deconv': F_deconv
        , 'psf_shift': psf_shift,
        'psf_aligned': psf_used
    }


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="PSF Gaussian modeling + deconvolution pipeline")
    parser.add_argument('--stars_folder', type=str, required=True, help="Folder with star stamp images (fits/png/jpg)")
    parser.add_argument('--test_image', type=str, required=True, help="Path to test HST FITS (or any 2D image)")
    parser.add_argument('--stamp_size', type=int, default=31, help="Stamp size (odd integer)")
    parser.add_argument('--out_prefix', type=str, default='run', help="Prefix for saved outputs")
    parser.add_argument('--verbose', action='store_true', help="Verbose prints")
    parser.add_argument('--no_daofind', action='store_true', help="If set, use brightest-pixel instead of DAOStarFinder for star localization")
    parser.add_argument('--rl_iter', type=int, default=30, help="Richardson-Lucy iterations")
    args = parser.parse_args()

    use_daofind = not args.no_daofind
    print("Building average PSF from folder:", args.stars_folder)
    psf_result = build_average_psf(args.stars_folder, stamp_size=args.stamp_size,
                                               verbose=args.verbose, use_daofind=use_daofind)
    gauss_psf = psf_result['gauss_psf']
    stack_psf = psf_result['stack_psf']
    print("PSF sum (should be 1):", gauss_psf.sum())

    # Save PSFs
    fits.writeto(f"{args.out_prefix}_avg_gaussian_psf.fits", gauss_psf.astype(np.float32), overwrite=True)
    fits.writeto(f"{args.out_prefix}_stack_psf.fits", psf_result['stack_psf'].astype(np.float32), overwrite=True)
    if psf_result['gauss_psf_from_params'] is not None:
        fits.writeto(f"{args.out_prefix}_avg_gaussian_psf_from_params.fits",
                     psf_result['gauss_psf_from_params'].astype(np.float32), overwrite=True)

    print("Saved PSF fits.")

    # Deconvolution + validation
    report = deconvolve_and_validate(args.test_image, stack_psf, out_prefix=args.out_prefix, rl_iterations=args.rl_iter)

    # Save deconvolved and reconvolved images
    fits.writeto(f"{args.out_prefix}_deconv.fits", report['deconv'].astype(np.float32), overwrite=True)
    fits.writeto(f"{args.out_prefix}_reconv.fits", report['reconv'].astype(np.float32), overwrite=True)
    print("Saved deconv and reconv FITS.")

    ratios = report['flux_ratios']
    mask = report['orig_fluxes'] > 0
    if mask.any():
        print(f"Median (reconv/orig) for top peaks: {np.median(ratios[mask]):.4f}  std: {np.std(ratios[mask]):.4f}")
    else:
        print("No positive peaks found for flux comparison.")


if __name__ == '__main__':
    main()
