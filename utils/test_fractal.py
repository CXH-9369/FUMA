import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractal_utils import (estimate_fractal_dimension, compute_fractal_map,
                           normalize_complexity, normalize_complexity_map)

print("=== Synthetic sanity checks (DBC: D in [2,3], normalize=True for single patches) ===")
rng = np.random.RandomState(0)
uniform = np.full((32, 32), 0.5, dtype=np.float32)
cb = (np.indices((32, 32)).sum(axis=0) % 2).astype(np.float32)
noise = rng.rand(32, 32).astype(np.float32)
grad = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
edge = np.zeros((32, 32), dtype=np.float32); edge[:, 16:] = 1.0
for name, p in [("uniform", uniform), ("smooth gradient", grad), ("sharp edge", edge),
                ("checkerboard", cb), ("random noise", noise)]:
    print(f"  {name:16s} D = {estimate_fractal_dimension(p, normalize=True):.3f}")

print("\n=== Real underwater image (Panama MTN_1544) ===")
from PIL import Image
img = np.asarray(Image.open("/home/cxh/FUMA/SeathruNeRF_dataset/Panama/images/MTN_1544.png").convert("L"), dtype=np.float32)
small = img[::2, ::2]
print(f"half-res shape: {small.shape}")

t = time.time()
D_map = compute_fractal_map(small, patch_size=16, scales=(2, 4, 8, 16), stride=8)
print(f"DBC map in {time.time()-t:.2f}s  D: min={D_map.min():.3f} mean={D_map.mean():.3f} max={D_map.max():.3f}")

C_f = normalize_complexity_map(D_map)          # robust percentile -> [0,1]
C_f_theory = normalize_complexity(D_map)        # fixed [2,3]
print(f"C_f (percentile norm): min={C_f.min():.3f} mean={C_f.mean():.3f} max={C_f.max():.3f}")
print(f"C_f (fixed [2,3]):      min={C_f_theory.min():.3f} mean={C_f_theory.mean():.3f} max={C_f_theory.max():.3f}")

# correlate D map with an independent local-std texture proxy on identical centers
half, stride = 8, 8
Ds, stds = [], []
for y in range(half, small.shape[0]-half, stride):
    for x in range(half, small.shape[1]-half, stride):
        p = small[y-half:y+half, x-half:x+half]
        Ds.append(D_map[y, x])
        stds.append(p.std())
Ds = np.array(Ds); stds = np.array(stds)
Dn = (Ds - Ds.mean())/(Ds.std()+1e-8); Sn = (stds - stds.mean())/(stds.std()+1e-8)
print(f"Pearson r (DBC map vs local-std, same centers): {float((Dn*Sn).mean()):.3f}")

def jet(gray):
    gray = np.clip(gray, 0, 1)
    r = np.clip(1.5 - np.abs(4*gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*gray - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)
vis = np.concatenate([np.stack([small/255.0]*3, -1), jet(C_f)], axis=1)
Image.fromarray((vis*255).astype(np.uint8)).save("/home/cxh/FUMA/gaussian-splatting/fractal_dimension_debug.png")
print("saved fractal_dimension_debug.png (left=grayscale, right=C_f heatmap)")
