# FUMA hyperparameters.
# Module-level constants for now; wire into argparse if CLI tunability is needed.

# Dual-core soft partition: w1 = sigmoid(split_sharpness * side_score)
split_sharpness = 10.0

# Learning rates for the dual-core parameters.
side_score_lr = 0.005
split_dir_lr = 0.0005

# Asymmetric dual-core opacity init (Step 7): core1 = "object/direct" core (higher),
# core2 = "medium/scatter" core (lower). Chosen so the blended opacity stays 0.1.
core1_init_opacity = 0.15
core2_init_opacity = 0.05

# Fractal-guided densification (Step 6).
# Adaptive gradient threshold: T_i = T_0 * (1 - fractal_lambda * C_f_i)
fractal_lambda = 0.5
# Adaptive child scale on split: s' = s * (1 - fractal_gamma * C_f_i)
fractal_gamma = 0.5
# Image pre-processing for the DBC fractal maps.
fractal_downsample = 2     # downsample factor before computing fractal maps
fractal_patch_size = 16    # local window for DBC
fractal_camera_stride = 1  # use every Nth training camera (1 = all)

# Fractal-aware pruning retention score R = w1*a + w2*C_f + w3*M
prune_w_opacity = 0.5
prune_w_complexity = 0.3
prune_w_gradient = 0.2
prune_retention_thresh = 0.005

# Medium-adaptive opacity regularization (Step 7).
# L_medium = medium_lambda * mean(M * (a1-a2)^2)  — couple the two cores in
# degraded regions; L_structure = -structure_lambda * mean(C_f * (a1-a2)^2) —
# amplify separation in complex/clear regions.
medium_lambda = 0.1
structure_lambda = 0.05

# Medium degradation score M = w_depth * depth_norm + w_color * color_attenuation.
medium_w_depth = 0.6
medium_w_color = 0.4
