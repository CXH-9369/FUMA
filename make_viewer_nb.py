#!/usr/bin/env python
# FUMA 交互式 mesh 查看器（VSCode 内嵌版）——生成：
#   1) view_mesh.ipynb        plotly notebook：在 VSCode 里直接运行、内嵌交互旋转/缩放，不依赖外部浏览器
#   2) mesh_F_plotly.html     自包含 plotly 网页（备用，用浏览器打开）
#
# 与 make_viewer.py（Three.js）的区别：plotly 走 VSCode 自带的 notebook 渲染器，
# 不受 Simple Browser / 外部浏览器 WebGL 限制，适合"在 VSCode 里直接看"。
#
# Usage:  python make_viewer_nb.py <mesh.ply> [--tris 30000] [--outdir viewer]

import argparse
import json
import os

import numpy as np
import open3d as o3d

NOTEBOOK_SRC = '''import numpy as np
import open3d as o3d
import plotly.graph_objects as go

MESH = "{mesh}"
N_TRI = {tris}   # 三角面数量，越大越精细越卡

m = o3d.io.read_triangle_mesh(MESH)
if len(m.triangles) > N_TRI:
    m = m.simplify_quadric_decimation(N_TRI)
m.compute_vertex_normals()

v = np.asarray(m.vertices)
t = np.asarray(m.triangles)
n = np.asarray(m.vertex_normals)
print(f"{{len(v)}} verts / {{len(t)}} tris")

# 法向 -> RGB（等效 MeshNormalMaterial：表面朝向可视化）
rgb = ((n + 1.0) / 2.0 * 255).astype(np.uint8)
colors = [f"rgb({{int(r)}},{{int(g)}},{{int(b)}})" for r, g, b in rgb]

fig = go.Figure(data=[
    go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=t[:, 0], j=t[:, 1], k=t[:, 2],
        vertexcolor=colors,
        flatshading=True,
        opacity=1.0,
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.15, roughness=0.5),
    )
])
fig.update_layout(
    scene=dict(aspectmode="data", bgcolor="#11151a"),
    scene_camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.2, y=0.6, z=1.4)),
    margin=dict(l=0, r=0, t=0, b=0),
    height=820,
    showlegend=False,
)
fig.show()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--tris", type=int, default=30000)
    ap.add_argument("--outdir", default="viewer")
    ap.add_argument("--tag", default="F", help="output name tag (view_mesh_<tag>.ipynb, mesh_<tag>_plotly.html)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mesh_abs = os.path.abspath(args.mesh)

    # 1) notebook
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# FUMA mesh 交互查看器\n",
                "在 VSCode 里：选择 **FUMA** 内核 → 点右上角 **Run All**（或 Shift+Enter 逐格运行）。\n",
                "左键拖拽=旋转，右键=平移，滚轮=缩放。颜色=表面法向。\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [NOTEBOOK_SRC.format(mesh=mesh_abs, tris=args.tris)]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (FUMA)", "language": "python", "name": "fuma"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    nb_path = os.path.join(args.outdir, f"view_mesh_{args.tag}.ipynb")
    with open(nb_path, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", nb_path)

    # 2) plotly HTML 备用
    import plotly.graph_objects as go
    m = o3d.io.read_triangle_mesh(args.mesh)
    if len(m.triangles) > args.tris:
        m = m.simplify_quadric_decimation(args.tris)
    m.compute_vertex_normals()
    v = np.asarray(m.vertices); t = np.asarray(m.triangles)
    n = np.asarray(m.vertex_normals)
    rgb = ((n + 1.0) / 2.0 * 255).astype(np.uint8)
    colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]
    fig = go.Figure(data=[go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                    i=t[:, 0], j=t[:, 1], k=t[:, 2],
                                    vertexcolor=colors, flatshading=True,
                                    lighting=dict(ambient=0.55, diffuse=0.75, specular=0.15, roughness=0.5))])
    fig.update_layout(scene=dict(aspectmode="data", bgcolor="#11151a"),
                      scene_camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.2, y=0.6, z=1.4)),
                      margin=dict(l=0, r=0, t=0, b=0), height=820, showlegend=False)
    html_path = os.path.join(args.outdir, f"mesh_{args.tag}_plotly.html")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print("wrote", html_path, f"({os.path.getsize(html_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
