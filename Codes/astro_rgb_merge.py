#!/usr/bin/env python3
"""
astro_rgb_merge.py - compact, robust HST-style RGB merger

This script:
 - loads three FITS images (R,G,B)
 - crops to common shape
 - optionally aligns G,B to R using subpixel phase cross-correlation
 - applies optional photometric header scaling
 - computes simple data-driven channel weights
 - applies optional equalization (none/global/clahe)
 - applies asinh stretch and an optional color matrix
 - writes a 16-bit TIFF and an 8-bit PNG (percentile+gamma scaling)

Usage example:
  python astro_rgb_merge.py --r PATH_R.fits --g PATH_G.fits --b PATH_B.fits --png_simple
"""

import os
import numpy as np
from astropy.io import fits
import tifffile
from PIL import Image
from skimage import exposure
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift


def load_fits(path):
    data = fits.getdata(path)
    if data is None:
        raise ValueError(f"No image data in {path}")
    return np.array(data, dtype=float)


def crop_to_common(a, b, c):
    ny = min(a.shape[0], b.shape[0], c.shape[0])
    nx = min(a.shape[1], b.shape[1], c.shape[1])
    return a[:ny, :nx], b[:ny, :nx], c[:ny, :nx]


def photometric_scale_if_possible(arr, hdr):
    try:
        photflam = hdr.get('PHOTFLAM') or hdr.get('PHOTFLAM1') or hdr.get('PHOTFLAM0')
        exptime = hdr.get('EXPTIME') or hdr.get('EXPOSURE') or hdr.get('EXPOS')
        a = arr.astype(float)
        if photflam is not None and exptime is not None:
            return (a / exptime) * photflam
        if exptime is not None:
            return a / exptime
    except Exception:
        pass
    return arr.astype(float)


def robust_weight(img):
    arr = np.nan_to_num(img.astype(float))
    bg = np.percentile(arr, 5.0)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    noise = 1.4826 * mad if mad > 0 else np.std(arr)
    mask = arr > (bg + 3.0 * max(noise, 1e-12))
    if mask.sum() < 50:
        nonzero = arr[arr > bg]
        median_sig = np.median(nonzero) if nonzero.size else max(med, 1.0)
    else:
        median_sig = np.median(arr[mask])
    if median_sig <= 0:
        return 1.0
    return 1.0 / median_sig


def equalize_channel(arr, method='none'):
    data = np.nan_to_num(arr.astype(float))
    data -= data.min()
    if data.max() == 0:
        return np.zeros_like(data)
    if method == 'none':
        return data / data.max()
    if method == 'global':
        return exposure.equalize_hist(data)
    if method == 'clahe':
        scaled = data / data.max()
        return exposure.equalize_adapthist(scaled, clip_limit=0.01)
    raise ValueError('unknown equalize')


def asinh_stretch(arr, Q=8.0):
    a = np.nan_to_num(arr.astype(float))
    a -= a.min()
    if a.max() <= 0:
        return np.zeros_like(a)
    a /= np.percentile(a, 99.5)
    out = np.arcsinh(Q * a) / np.arcsinh(Q)
    return np.clip(out, 0.0, 1.0)


def apply_color_matrix(rgb, matrix=None):
    if matrix is None:
        matrix = np.array([[1.02, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.98]])
    flat = rgb.reshape(-1, 3)
    corrected = np.clip(flat.dot(matrix.T), 0.0, None)
    out = corrected.reshape(rgb.shape)
    if out.max() > 0:
        out /= out.max()
    return out


def align_to_ref(ref, img, upsample_factor=10):
    shift, error, diffphase = phase_cross_correlation(ref, img, upsample_factor=upsample_factor)
    shifted = nd_shift(img, shift=shift, order=3, mode='constant', cval=0.0)
    return shifted, tuple(shift)


def save_8bit_scaled(rgb, out_png, vmin_pct=0.5, vmax_pct=99.5, gamma=2.2):
    arr = np.nan_to_num(rgb.astype(float))
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr])
    vmin = np.percentile(arr, vmin_pct)
    vmax = np.percentile(arr, vmax_pct)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = arr.max() if arr.max() > vmin else vmin + 1e-6
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    img_disp = np.power(scaled, 1.0 / float(gamma))
    img8 = (img_disp * 255.0).round().astype(np.uint8)
    img8 = np.ascontiguousarray(img8)
    if img8.ndim != 3 or img8.shape[2] != 3:
        raise ValueError(f'unexpected RGB shape: {img8.shape}')
    Image.fromarray(img8, mode='RGB').save(out_png)


def build_hst_rgb(r_path, g_path, b_path,
                  out_tiff='out_RGB_16bit.tiff', out_png='out_preview.png',
                  do_photocal=True, equalize='clahe', align=True,
                  apply_color_corr=True, save_simple_png=True,
                  png_vmin=0.5, png_vmax=99.5, png_gamma=2.2):
    # load
    r = load_fits(r_path)
    g = load_fits(g_path)
    b = load_fits(b_path)
    r_hdr = fits.getheader(r_path)
    g_hdr = fits.getheader(g_path)
    b_hdr = fits.getheader(b_path)

    # crop
    r, g, b = crop_to_common(r, g, b)

    # photometric scale
    if do_photocal:
        r = photometric_scale_if_possible(r, r_hdr)
        g = photometric_scale_if_possible(g, g_hdr)
        b = photometric_scale_if_possible(b, b_hdr)

    # optional alignment: align G and B to R
    shifts = {'g': (0, 0), 'b': (0, 0)}
    if align:
        try:
            g, shifts['g'] = align_to_ref(r, g)
            b, shifts['b'] = align_to_ref(r, b)
            print(f'Aligned channels: g_shift={shifts["g"]}, b_shift={shifts["b"]}')
        except Exception as e:
            print(f'Channel alignment failed: {e}')

    # weights
    wR = robust_weight(r); wG = robust_weight(g); wB = robust_weight(b)
    print(f'weights R,G,B = {wR:.3e}, {wG:.3e}, {wB:.3e}')

    r_w = r * wR; g_w = g * wG; b_w = b * wB

    # equalize
    r_eq = equalize_channel(r_w, method=equalize)
    g_eq = equalize_channel(g_w, method=equalize)
    b_eq = equalize_channel(b_w, method=equalize)

    # asinh stretch
    r_s = asinh_stretch(r_eq); g_s = asinh_stretch(g_eq); b_s = asinh_stretch(b_eq)

    rgb = np.dstack([r_s, g_s, b_s])
    rgb = np.nan_to_num(rgb)
    rgb = np.clip(rgb, 0.0, 1.0)

    if apply_color_corr:
        rgb = apply_color_matrix(rgb)

    if rgb.max() > 0:
        rgb = rgb / rgb.max()

    # save 16-bit TIFF
    rgb16 = (rgb * 65535.0).round().astype(np.uint16)
    tifffile.imwrite(out_tiff, rgb16, photometric='rgb')
    print(f'Wrote {out_tiff}')

    # save 8-bit preview (simple scaled)
    try:
        save_8bit_scaled(rgb, out_png.replace('.png', '_simple.png') if save_simple_png else out_png,
                        vmin_pct=png_vmin, vmax_pct=png_vmax, gamma=png_gamma)
    except Exception:
        # fallback: write a standard 8-bit sRGB preview
        def linear_to_srgb(img_lin):
            a = 0.055
            srgb = np.where(img_lin <= 0.0031308,
                            12.92 * img_lin,
                            (1 + a) * np.power(img_lin, 1/2.4) - a)
            return np.clip(srgb, 0.0, 1.0)
        preview = linear_to_srgb(rgb)
        preview8 = (preview * 255.0).round().astype(np.uint8)
        Image.fromarray(preview8, mode='RGB').save(out_png)
        print(f'Wrote fallback preview {out_png}')

    return rgb16, (rgb * 255).round().astype(np.uint8)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--r', required=True)
    p.add_argument('--g', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--out_tiff', default='out_RGB_16bit.tiff')
    p.add_argument('--out_png', default='out_preview.png')
    p.add_argument('--no_photocal', action='store_true')
    p.add_argument('--equalize', choices=['none', 'global', 'clahe'], default='clahe')
    p.add_argument('--no_align', action='store_true')
    p.add_argument('--no_colorcorr', action='store_true')
    p.add_argument('--png_simple', action='store_true')
    p.add_argument('--png_vmin', type=float, default=0.5)
    p.add_argument('--png_vmax', type=float, default=99.5)
    p.add_argument('--png_gamma', type=float, default=2.2)
    args = p.parse_args()

    build_hst_rgb(args.r, args.g, args.b,
                  out_tiff=args.out_tiff, out_png=args.out_png,
                  do_photocal=not args.no_photocal,
                  equalize=args.equalize,
                  align=not args.no_align,
                  apply_color_corr=not args.no_colorcorr,
                  save_simple_png=args.png_simple,
                  png_vmin=args.png_vmin,
                  png_vmax=args.png_vmax,
                  png_gamma=args.png_gamma)


if __name__ == '__main__':
    main()
    
