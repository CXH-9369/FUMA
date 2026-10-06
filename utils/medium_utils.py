#
# FUMA — Medium degradation estimator (Module D).
# Per-Gaussian medium degradation score M in [0,1] (0 = clear, 1 = strongly
# degraded). Underwater degradation grows with distance (scattering/absorption)
# and shows up as red-channel attenuation, so we combine:
#   - depth to the nearest camera centre (robust-normalised),
#   - colour attenuation (blue/green dominance over red) from the SH DC.
# The full C/G/A/V decomposition (contrast, gradient, colour, view consistency)
# from the spec can be layered on later; these two capture the dominant cues.
#

import torch

C0 = 0.28209479177387814  # SH DC coefficient scale (matches sh_utils.C0)


def estimate_medium_quality(xyz, sh_dc, camera_centers, scene_extent,
                            w_depth=0.6, w_color=0.4, depth_pct=90.0):
    """
    Args:
        xyz: [N,3] Gaussian means.
        sh_dc: [N,3] SH DC coefficients (features_dc[:, :3, 0]).
        camera_centers: [K,3] camera centres.
        scene_extent: scalar scene extent (for the depth saturation scale).
        w_depth, w_color: blending weights of the two cues.
        depth_pct: robust percentile used to normalise depth.

    Returns:
        M: [N,1] medium degradation score in [0,1].
    """
    xyz = torch.as_tensor(xyz, dtype=torch.float32)
    sh_dc = torch.as_tensor(sh_dc, dtype=torch.float32)
    camera_centers = torch.as_tensor(camera_centers, dtype=torch.float32, device=xyz.device)

    if xyz.shape[0] == 0:
        return torch.zeros((0, 1), device=xyz.device)

    # depth to nearest camera centre, robust-normalised
    d = torch.cdist(xyz, camera_centers).min(dim=1).values  # [N]
    d_hi = torch.quantile(d, depth_pct / 100.0).clamp_min(1e-8)
    depth_norm = (d / d_hi).clamp(0.0, 1.0)

    # colour attenuation: (g+b)/2 - r; red attenuates fastest underwater
    rgb = sh_dc * C0 + 0.5  # recover RGB from SH DC (matches sh_utils.SH2RGB)
    red_atten = 0.5 * (rgb[:, 1] + rgb[:, 2]) - rgb[:, 0]
    lo, hi = red_atten.min(), red_atten.max()
    color_norm = ((red_atten - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)

    M = (w_depth * depth_norm + w_color * color_norm).clamp(0.0, 1.0)
    return M.unsqueeze(-1)  # [N,1]
