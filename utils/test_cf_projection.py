import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import torch

parser = ArgumentParser()
ModelParams(parser)
PipelineParams(parser)
OptimizationParams(parser)
args = parser.parse_args(['-s', '/home/cxh/FUMA/SeathruNeRF_dataset/Panama',
                          '-m', '/tmp/fuma_frac_test'])

safe_state(False)
os.makedirs(args.model_path, exist_ok=True)
gaussians = GaussianModel(args.sh_degree)
scene = Scene(args, gaussians)
gaussians.training_setup(args)

t = time.time()
cf = gaussians.compute_fractal_complexity(scene.getTrainCameras())
elapsed = time.time() - t

cf = cf.squeeze(-1)
print(f"\ncompute_fractal_complexity: {elapsed:.1f}s  over {len(scene.getTrainCameras())} cameras")
print(f"N = {cf.shape[0]}")
print(f"C_f: min={cf.min().item():.4f} mean={cf.mean().item():.4f} max={cf.max().item():.4f}")
print(f"fraction C_f > 0.05: {(cf > 0.05).float().mean().item():.3f}")
print(f"fraction C_f > 0.20: {(cf > 0.20).float().mean().item():.3f}")
print(f"fraction C_f == 0 (unseen): {(cf == 0).float().mean().item():.3f}")
