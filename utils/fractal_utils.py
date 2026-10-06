#
# FUMA — Fractal complexity estimator (Module C).
# Differential box-counting (DBC, Sarkar & Chaudhuri 1992) of the grayscale image
# surface — the standard "box-counting fractal dimension" for grayscale images.
# It measures surface roughness: smooth open water -> D≈2, textured coral/rock ->
# D up to 3.
#
# IMPORTANT: the image is normalised to a fixed range ONCE (globally), then DBC is
# applied per window WITHOUT per-patch rescaling. Per-patch min-max normalisation
# was measured to destroy the water/coral contrast signal (correlation with local
# texture dropped to ~0), so we intentionally avoid it.
#

import numpy as np


def _box_count(binary, box_size):
    """Count boxes of side `box_size` containing at least one foreground pixel."""
    h, w = binary.shape
    n_rows = int(np.ceil(h / box_size))
    n_cols = int(np.ceil(w / box_size))
    pad_h = n_rows * box_size - h
    pad_w = n_cols * box_size - w
    padded = np.pad(binary, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    grid = padded.reshape(n_rows, box_size, n_cols, box_size)
    return int(grid.any(axis=(1, 3)).sum())


def estimate_binary_fractal_dimension(patch, scales=(2, 4, 8, 16, 32), binarize_thresh=None):
    """
    Plain (binary) box-counting: D = -log N(eps)/log eps, D in [0, 2].
    Kept for spec-compliance only; it binarises by the patch mean and discriminates
    underwater texture poorly — prefer `estimate_fractal_dimension` (DBC).
    """
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim == 3:
        patch = patch.mean(axis=2)
    if patch.ndim != 2 or patch.size < 4:
        return 0.0
    lo, hi = float(patch.min()), float(patch.max())
    if hi - lo < 1e-8:
        return 0.0
    norm = (patch - lo) / (hi - lo)
    thresh = binarize_thresh if binarize_thresh is not None else float(norm.mean())
    binary = (norm > thresh).astype(np.uint8)
    valid = [s for s in scales if s <= min(binary.shape)]
    if len(valid) < 2:
        return 0.0
    log_eps, log_n = [], []
    for s in valid:
        n = _box_count(binary, s)
        log_eps.append(np.log(s))
        log_n.append(np.log(max(n, 1.0)))
    slope = np.polyfit(log_eps, log_n, 1)[0]
    return float(np.clip(-slope, 0.0, 2.0))


def estimate_fractal_dimension(patch, scales=(2, 4, 8, 16, 32), normalize=False):
    """
    Differential box-counting fractal dimension of a 2D grayscale patch.

    The patch is treated as a 3D surface (x, y, intensity); the box height at grid
    size r is r/P (P = patch side), so a box of side r spans r/P of the fixed [0,1]
    intensity range. N_r is the number of such boxes covering the surface, and
    D = d(log N_r)/d(log(P/r)).

    Args:
        patch: 2D np.ndarray grayscale; 3D input is greyed by channel mean.
        scales: grid sizes (pixels) for the log-log fit.
        normalize: if True, min-max normalise this single patch to [0,1] first
            (convenient for standalone synthetic tests). For image maps pass
            False and feed a globally-normalised patch (see compute_fractal_map).

    Returns:
        D in [2, 3] (clamped). 2 = smooth surface, 3 = maximally rough.
    """
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim == 3:
        patch = patch.mean(axis=2)
    if patch.ndim != 2 or patch.size < 4:
        return 2.0

    if normalize:
        lo, hi = float(patch.min()), float(patch.max())
        if hi - lo < 1e-8:
            return 2.0
        patch = (patch - lo) / (hi - lo)

    h, w = patch.shape
    P = float(min(h, w))

    log_inv_r, log_n = [], []
    for r in scales:
        if r < 2 or r > P:
            continue
        n_rows = int(np.ceil(h / r))
        n_cols = int(np.ceil(w / r))
        pad_h = n_rows * r - h
        pad_w = n_cols * r - w
        p = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="edge")
        grid = p.reshape(n_rows, r, n_cols, r)
        mx = grid.max(axis=(1, 3))
        mn = grid.min(axis=(1, 3))
        box_height = r / P  # box of side r spans r/P of the [0,1] intensity range
        nr = (mx - mn) / box_height + 1.0
        N = float(nr.sum())
        if N <= 0:
            continue
        log_inv_r.append(np.log(P / r))
        log_n.append(np.log(N))

    if len(log_n) < 2:
        return 2.0
    slope = np.polyfit(log_inv_r, log_n, 1)[0]
    return float(np.clip(slope, 2.0, 3.0))


def normalize_complexity(D, d_min=2.0, d_max=3.0):
    """Affine map of DBC dimension D in [d_min, d_max] to C_f in [0, 1]."""
    D = np.asarray(D, dtype=np.float32)
    return np.clip((D - d_min) / (d_max - d_min), 0.0, 1.0)


def normalize_complexity_map(D_map, lo_pct=2.0, hi_pct=98.0):
    """
    Robust percentile normalisation of a whole D map to C_f in [0, 1].
    Uses robust percentiles (instead of the fixed [2,3] range) so the map spans
    [0,1] meaningfully for the actual scene content.
    """
    D_map = np.asarray(D_map, dtype=np.float32)
    lo = float(np.percentile(D_map, lo_pct))
    hi = float(np.percentile(D_map, hi_pct))
    if hi - lo < 1e-8:
        return np.zeros_like(D_map)
    return np.clip((D_map - lo) / (hi - lo), 0.0, 1.0)


def compute_fractal_map(image, patch_size=16, scales=(2, 4, 8, 16), stride=8):
    """
    Sliding-window DBC fractal dimension over a grayscale image.

    The image is robust-normalised to [0,1] ONCE (globally), then DBC is applied per
    window without per-patch rescaling. Returns a float32 D map of the same spatial
    size as `image` (D in [2,3]). Overlapping windows are averaged; border pixels
    without a full window default to 2.0 (smooth).
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image.mean(axis=2)
    lo = float(np.percentile(image, 1.0))
    hi = float(np.percentile(image, 99.0))
    if hi - lo < 1e-8:
        return np.full(image.shape[:2], 2.0, dtype=np.float32)
    g = (image - lo) / (hi - lo)
    h, w = g.shape

    D_map = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    half = patch_size // 2
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            patch = g[y - half:y + half, x - half:x + half]
            D_map[y - half:y + half, x - half:x + half] += estimate_fractal_dimension(patch, scales)
            counts[y - half:y + half, x - half:x + half] += 1
    valid = counts > 0
    D_map[valid] /= counts[valid]
    D_map[~valid] = 2.0
    return D_map
