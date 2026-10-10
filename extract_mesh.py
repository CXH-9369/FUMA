#!/usr/bin/env python
# FUMA mesh extraction — reconstruct a surface mesh from the core1 (surface core)
# Gaussians + their manifold-smoothed normals, via open3d Poisson reconstruction.
#
# No ground-truth mesh/depth needed: the surface is the set of Gaussians whose
# core1 (surface) opacity is high, and the normal is the smallest-scale axis of
# each Gaussian's covariance (2DGS/SuGaR convention), smoothed by the manifold
# regularizer. Normals are oriented toward the training-camera centroid so Poisson
# gets a consistent outward field.
#
# Usage:
#   python extract_mesh.py -m <model_dir> --iteration 30000 \
#       --alpha1_threshold 0.05 --depth 9 --output <out.ply>
#
# Also saves the oriented surface point cloud (<model_dir>/mesh_<iter>_pcd.ply).

import os
import argparse

import numpy as np
import torch
import open3d as o3d

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state


def main():
    parser = argparse.ArgumentParser(description="FUMA surface-mesh extraction")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--alpha1_threshold", type=float, default=0.05,
                        help="keep Gaussians with sigmoid(opacity_core1) >= this")
    parser.add_argument("--depth", type=int, default=9, help="Poisson octree depth")
    parser.add_argument("--voxel_size", type=float, default=0.0,
                        help="optional voxel downsample (0 = off)")
    parser.add_argument("--density_quantile", type=float, default=0.05,
                        help="crop Poisson vertices below this density quantile")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # --- surface geometry (same frame as the PLY / cameras) ---
    xyz = gaussians.get_xyz.detach().cpu().numpy()                    # [N,3]
    alpha1 = torch.sigmoid(gaussians._opacity_core1).detach().cpu().numpy().squeeze(-1)  # [N]
    normal = gaussians.get_surface_normal().detach().cpu().numpy()    # [N,3]

    cam_centers = torch.stack([c.camera_center for c in scene.getTrainCameras()]).cpu().numpy()
    centroid = cam_centers.mean(0)                                    # [3]

    mask = alpha1 >= args.alpha1_threshold
    pts = xyz[mask].astype(np.float64)
    nrm = normal[mask].astype(np.float64)
    print(f"[surface] kept {mask.sum()}/{len(xyz)} Gaussians "
          f"(alpha1 >= {args.alpha1_threshold})")

    # orient normals toward the camera centroid (outward to the viewer)
    flip = (nrm * (centroid[None, :] - pts)).sum(1) < 0.0
    nrm[flip] *= -1.0
    print(f"[orient] flipped {int(flip.sum())} normals toward camera centroid")

    # --- open3d pipeline ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm)

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)
        print(f"[downsample] {len(pcd.points)} points after voxel "
              f"size {args.voxel_size}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth)
    print(f"[poisson] {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # crop low-density (spurious) vertices
    if args.density_quantile > 0:
        dens = np.asarray(densities)
        thresh = np.quantile(dens, args.density_quantile)
        mesh.remove_vertices_by_mask(dens < thresh)
        print(f"[crop] kept verts with density >= {thresh:.3g} "
              f"(quantile {args.density_quantile})")

    mesh.compute_vertex_normals()

    out_mesh = getattr(args, "output", None) or os.path.join(dataset.model_path,
                                                             f"mesh_{args.iteration}.ply")
    out_pcd = os.path.join(dataset.model_path, f"mesh_{args.iteration}_pcd.ply")
    o3d.io.write_triangle_mesh(out_mesh, mesh)
    o3d.io.write_point_cloud(out_pcd, pcd)
    print(f"saved mesh -> {out_mesh}")
    print(f"saved oriented surface pcd -> {out_pcd}")


if __name__ == "__main__":
    main()
