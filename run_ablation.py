#!/usr/bin/env python
# FUMA Step 8 — ablation runner.
#
# Trains the five ablation configs (A-E), renders the test views, and reports
# PSNR / SSIM / LPIPS plus the final Gaussian count and training wall-time.
#
# Usage:
#   python run_ablation.py -s <scene> -o <output_root> --iterations 30000 \
#       --configs A B C D E
#
# Stages can be skipped to resume a partially-finished run:
#   --skip_train / --skip_render / --skip_metrics

import argparse
import json
import os
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = "/home/cxh/anaconda3/envs/FUMA/bin/python"

CONFIGS = ['A', 'B', 'C', 'D', 'E']
CONFIG_DESC = {
    'A': '3DGS baseline',
    'B': '+ Dual-Core',
    'C': '+ Fractal Growth',
    'D': '+ Medium Regularization',
    'E': 'Full FUMA',
}


def run(cmd, log_path):
    """Run a command, appending stdout/stderr to log_path. Returns (code, seconds)."""
    with open(log_path, 'a') as f:
        f.write("\n" + "$ " + " ".join(cmd) + "\n")
        f.flush()
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
    return proc.returncode, dt


def count_points(ply_path):
    """Number of vertices in a saved point_cloud.ply (read ASCII header only)."""
    try:
        with open(ply_path, 'rb') as f:
            for raw in f:
                line = raw.decode('ascii', errors='replace').strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line.startswith("end_header"):
                    break
    except OSError:
        pass
    return -1


def main():
    ap = argparse.ArgumentParser(description="FUMA Step 8 ablation runner")
    ap.add_argument('-s', '--source', required=True, help="path to the scene (dataset)")
    ap.add_argument('-o', '--output', required=True, help="output root dir")
    ap.add_argument('--iterations', type=int, default=30000)
    ap.add_argument('--configs', nargs='+', default=CONFIGS, choices=CONFIGS)
    ap.add_argument('--skip_train', action='store_true')
    ap.add_argument('--skip_render', action='store_true')
    ap.add_argument('--skip_metrics', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = {}

    for cfg in args.configs:
        model_dir = os.path.join(args.output, cfg)
        log_path = os.path.join(args.output, f"train_{cfg}.log")
        results[cfg] = {'desc': CONFIG_DESC[cfg], 'train_time': 0.0, 'n_points': -1}

        if not args.skip_train:
            code, dt = run([
                PY, 'train.py',
                '-s', args.source,
                '-m', model_dir,
                '--eval',
                '--fuma_ablation', cfg,
                '--iterations', str(args.iterations),
                '--quiet',
            ], log_path)
            results[cfg]['train_time'] = dt
            if code != 0:
                print(f"[{cfg}] training FAILED (exit {code}) — see {log_path}")
                continue

        ply_path = os.path.join(model_dir, 'point_cloud',
                                f'iteration_{args.iterations}', 'point_cloud.ply')
        results[cfg]['n_points'] = count_points(ply_path)

        if not args.skip_render:
            code, _ = run([
                PY, 'render.py',
                '-s', args.source,
                '-m', model_dir,
                '--iteration', str(args.iterations),
                '--eval', '--skip_train', '--quiet',
            ], log_path)
            if code != 0:
                print(f"[{cfg}] rendering FAILED (exit {code}) — see {log_path}")

    if not args.skip_metrics:
        code, _ = run([
            PY, 'metrics.py',
            '-m', *[os.path.join(args.output, c) for c in args.configs],
        ], os.path.join(args.output, "metrics.log"))

    # Aggregate
    print("\n" + "=" * 78)
    print(f"Ablation on {os.path.basename(args.source.rstrip('/'))} "
          f"({args.iterations} iters)")
    print("=" * 78)
    print(f"{'cfg':<4}{'config':<26}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>9}"
          f"{'#Gauss':>9}{'train(min)':>12}")
    for cfg in args.configs:
        r = results[cfg]
        m = {}
        rj = os.path.join(args.output, cfg, 'results.json')
        if os.path.exists(rj):
            with open(rj) as f:
                data = json.load(f)
            # results.json is keyed by method dir -> {PSNR, SSIM, LPIPS}
            for method, metrics in data.items():
                m.update(metrics)
        psnr = m.get('PSNR', float('nan'))
        ssim = m.get('SSIM', float('nan'))
        lpips = m.get('LPIPS', float('nan'))
        print(f"{cfg:<4}{r['desc']:<26}{psnr:>8.3f}{ssim:>8.4f}{lpips:>9.4f}"
              f"{r['n_points']:>9}{r['train_time']/60:>12.1f}")

    # Persist a JSON aggregate
    agg = []
    for cfg in args.configs:
        r = results[cfg]
        rj = os.path.join(args.output, cfg, 'results.json')
        m = {}
        if os.path.exists(rj):
            with open(rj) as f:
                data = json.load(f)
            for method, metrics in data.items():
                m.update(metrics)
        agg.append({'config': cfg, 'desc': r['desc'],
                    'PSNR': m.get('PSNR'), 'SSIM': m.get('SSIM'), 'LPIPS': m.get('LPIPS'),
                    'n_gaussians': r['n_points'], 'train_sec': r['train_time']})
    with open(os.path.join(args.output, 'ablation_summary.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved {os.path.join(args.output, 'ablation_summary.json')}")


if __name__ == '__main__':
    main()
