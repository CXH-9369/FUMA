#!/usr/bin/env python
# FUMA geometry diagnostics — no ground-truth mesh/depth needed.
#
# Quantifies how "surface-like" the core1 (surface core) Gaussians are, so the
# manifold (config F) can be compared against a no-manifold baseline (config G):
#   s_min_mean/median  smallest scale axis (lower = flatter surface patches)
#   anisotropy         s_min / s_max (lower = more disc-like, less volumetric)
#   norm_consist       mean |n_i . n_j| over kNN (higher = smoother normal field)
#   norm_sfm_align     mean |n_i . n_sfm| (higher = normals stay close to SfM)
#   flat_w_alpha1      alpha_core1-weighted mean smallest scale
#
# The surface normal is the smallest-scale axis of each Gaussian's covariance
# (2DGS/SuGaR convention), derived from the saved rotation quaternion + scales.
#
# Usage:  python eval_geometry.py <path/to/point_cloud.ply> [...] [-k 8]

import argparse
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def quat_to_rot(q):
    """q = [N,4] = (w, x, y, z), matching utils.general_utils.build_rotation."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.zeros((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_ply(path):
    v = PlyData.read(path).elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    sfm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64)
    scale_log = np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    # opacity_core1 is stored as the inverse-sigmoid logit
    alpha1 = 1.0 / (1.0 + np.exp(-v["opacity_core1"].astype(np.float64)))
    return xyz, sfm, scale_log, rot, alpha1


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def metrics(path, k=8):
    xyz, sfm, scale_log, rot, alpha1 = load_ply(path)
    scales = np.exp(scale_log)                       # [N,3]
    s_min = scales.min(axis=1)
    s_max = scales.max(axis=1)
    R = quat_to_rot(rot)                             # [N,3,3]
    min_axis = scales.argmin(axis=1)
    normal = _unit(R[np.arange(len(xyz)), :, min_axis])   # [N,3]
    sfm_n = _unit(sfm)

    # normal consistency over kNN
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    idx = idx[:, 1:]                                 # drop self
    cos = np.abs(np.einsum("ij,ikj->ik", normal, normal[idx]))  # [N,k]
    consistency = float(cos.mean())

    align = float(np.abs((normal * sfm_n).sum(axis=1)).mean())

    w = alpha1 / (alpha1.sum() + 1e-12)              # alpha1-weighted flatness
    flat_w = float((w * s_min).sum())

    return dict(
        n=int(len(xyz)),
        s_min_mean=float(s_min.mean()),
        s_min_median=float(np.median(s_min)),
        anisotropy=float((s_min / s_max.clip(1e-12)).mean()),
        norm_consist=consistency,
        norm_sfm_align=align,
        flat_w_alpha1=flat_w,
    )


def main():
    ap = argparse.ArgumentParser(description="FUMA surface-geometry diagnostics")
    ap.add_argument("ply", nargs="+", help="one or more final point_cloud.ply paths")
    ap.add_argument("-k", type=int, default=8, help="kNN neighbours for normal consistency")
    args = ap.parse_args()

    rows = []
    for p in args.ply:
        m = metrics(p, k=args.k)
        m["path"] = p
        rows.append(m)

    cols = [("path", "{:<70}"), ("n", "{:>8}"), ("s_min_mean", "{:>10}"),
            ("s_min_median", "{:>9}"), ("anisotropy", "{:>10}"),
            ("norm_consist", "{:>9}"), ("norm_sfm_align", "{:>11}"), ("flat_w_alpha1", "{:>10}")]
    print(" ".join(h.format(c) for c, h in cols))
    print("-" * (70 + 8 + 10 + 9 + 10 + 9 + 11 + 10 + 7))
    for r in rows:
        print(" ".join(h.format(r[c]) for c, h in cols))


if __name__ == "__main__":
    main()
