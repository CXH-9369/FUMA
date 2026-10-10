# FUMA hyperparameters.
# Module-level constants; wire into argparse if CLI tunability is needed.

# Dual-core soft partition: w1 = sigmoid(split_sharpness * side_score)
split_sharpness = 10.0

# Learning rates for the dual-core parameters.
side_score_lr = 0.005
split_dir_lr = 0.0005

# Asymmetric dual-core opacity init (Step 7): core1 = "object/direct" core (higher),
# core2 = "medium/scatter" core (lower). Chosen so the blended opacity stays 0.1.
core1_init_opacity = 0.15
core2_init_opacity = 0.05

# Single-opacity init used when the dual-core is disabled (original 3DGS value).
baseline_init_opacity = 0.1

# Fractal-guided densification (Step 6).
# Adaptive gradient threshold: T_i = T_0 * (1 - fractal_lambda * C_f_i)
fractal_lambda = 0.2
# Adaptive child scale on split: s' = s * (1 - fractal_gamma * C_f_i)
fractal_gamma = 0.15
# Image pre-processing for the DBC fractal maps.
fractal_downsample = 2     # downsample factor before computing fractal maps
fractal_patch_size = 16    # local window for DBC
fractal_camera_stride = 1  # use every Nth training camera (1 = all)

# Fractal-aware pruning (Step 8 tuning): complexity-modulated opacity threshold.
# Prune when alpha < min_opacity * (1 - fractal_prune_beta * C_f). Complex regions
# (high C_f) get a lower threshold -> retained more; smooth regions prune normally.
# Replaces the earlier retention score R = w1*a + w2*C_f + w3*M, whose C_f/M terms
# dominated the 0.005 threshold and disabled pruning (2x Gaussian bloat, no gain).
fractal_prune_beta = 0.5

# Medium-adaptive opacity regularization (Step 7).
# L_medium = medium_lambda * mean(M * (a1-a2)^2)  — couple the two cores in
# degraded regions; L_structure = -structure_lambda * mean(C_f * (a1-a2)^2) —
# amplify separation in complex/clear regions.
medium_lambda = 0.1
structure_lambda = 0.05

# Medium degradation score M = w_depth * depth_norm + w_color * color_attenuation.
medium_w_depth = 0.6
medium_w_color = 0.4


# ---------------------------------------------------------------------------
# Step 8 — Ablation switches.
#
# Each FUMA mechanism can be toggled independently; False restores the original
# 3DGS behaviour. The five ablation configs follow the spec's Section 10:
#   A: 3DGS baseline
#   B: + Dual-Core            (dual opacity + structure loss to activate it)
#   C: + Fractal Growth       (fractal-guided densification + fractal-aware pruning)
#   D: + Medium Regularization(dual-core substrate + medium reg, no structure loss)
#   E: Full FUMA
# ---------------------------------------------------------------------------
use_dual_core = True        # dual-core opacity representation (core1/core2 + side_score)
use_fractal_growth = True   # fractal-guided densification + fractal-aware pruning
use_medium_reg = True       # medium-adaptive opacity regularization loss
use_structure_loss = True   # structure loss (activates the dual-core separation)


_ABLATIONS = {
    'A': dict(use_dual_core=False, use_fractal_growth=False, use_medium_reg=False, use_structure_loss=False),
    'B': dict(use_dual_core=True,  use_fractal_growth=False, use_medium_reg=False, use_structure_loss=True),
    'C': dict(use_dual_core=False, use_fractal_growth=True,  use_medium_reg=False, use_structure_loss=False),
    'D': dict(use_dual_core=True,  use_fractal_growth=False, use_medium_reg=True,  use_structure_loss=False),
    'E': dict(use_dual_core=True,  use_fractal_growth=True,  use_medium_reg=True,  use_structure_loss=True),
}


def apply_ablation(name):
    """Set the module-level switches for an ablation config.

    Args:
        name: one of 'A'..'E', or None/'' to keep the current (default) switches.

    Returns:
        dict of the active switch values (for logging).
    """
    global use_dual_core, use_fractal_growth, use_medium_reg, use_structure_loss
    if name in _ABLATIONS:
        cfg = _ABLATIONS[name]
        use_dual_core = cfg['use_dual_core']
        use_fractal_growth = cfg['use_fractal_growth']
        use_medium_reg = cfg['use_medium_reg']
        use_structure_loss = cfg['use_structure_loss']
    return dict(use_dual_core=use_dual_core,
                use_fractal_growth=use_fractal_growth,
                use_medium_reg=use_medium_reg,
                use_structure_loss=use_structure_loss)
