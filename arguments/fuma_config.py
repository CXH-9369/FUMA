# FUMA hyperparameters.
# Module-level constants for now; wire into argparse if CLI tunability is needed.

# Dual-core soft partition: w1 = sigmoid(split_sharpness * side_score)
split_sharpness = 10.0

# Learning rates for the dual-core parameters.
side_score_lr = 0.005
split_dir_lr = 0.0005
