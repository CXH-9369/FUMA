#!/usr/bin/env python
# FUMA mesh viewer generator — turn a reconstructed mesh (.ply/.obj/.glb) into a
# single fully self-contained interactive HTML file (Three.js + OrbitControls).
#
# The mesh is exported to GLB, base64-embedded into the HTML, AND Three.js r128
# (classic UMD build, no ES modules / no import-map / no CDN) is inlined too, so
# the result opens:
#   - by double-clicking (file://)
#   - via any static server (e.g. python -m http.server)
#   - in VSCode's Simple Browser / Live Server
# No network, no CORS, no external files required. Works in any browser.
#
# The inlined library files live next to this script in viewer_lib/:
#   three.min.js, OrbitControls.js, GLTFLoader.js  (three@0.128.0, examples/js)
#
# Usage:
#   python make_viewer.py <mesh.ply> -o viewer.html [--simplify 200000]
#
#   --simplify N   quadric-decimate the mesh to ~N triangles first (default: none)

import argparse
import base64
import os

import numpy as np
import open3d as o3d

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FUMA Mesh Viewer</title>
<style>
  html, body { margin:0; padding:0; overflow:hidden; background:#11151a; height:100%; font-family:system-ui,sans-serif; }
  #info { position:absolute; top:12px; left:12px; color:#e6edf3; background:rgba(0,0,0,.55); padding:9px 13px; border-radius:7px; font-size:13px; line-height:1.6; pointer-events:none; white-space:pre-line; }
  #status { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#ffd166; background:rgba(0,0,0,.72); padding:14px 20px; border-radius:8px; font-size:15px; line-height:1.7; text-align:center; max-width:82%; }
  #toolbar { position:absolute; bottom:16px; left:50%; transform:translateX(-50%); display:flex; gap:8px; }
  button { background:rgba(255,255,255,.12); color:#e6edf3; border:1px solid rgba(255,255,255,.28); padding:7px 14px; border-radius:6px; cursor:pointer; font-size:13px; }
  button:hover { background:rgba(255,255,255,.24); }
  button.on { background:#2f81f7; border-color:#2f81f7; }
</style>
<script>%%THREE_JS%%</script>
<script>%%ORBIT_JS%%</script>
<script>%%GLTF_JS%%</script>
</head>
<body>
<div id="info"></div>
<div id="status">正在加载三维模型…</div>
<div id="toolbar">
  <button id="bSolid">实体</button>
  <button id="bNormal" class="on">法向</button>
  <button id="bWire">线框</button>
</div>
<script>
(function(){
  var statusEl = document.getElementById('status');
  function fail(msg){ if (statusEl) statusEl.innerHTML = '⚠️ ' + msg; }
  try {
    var GLB_B64 = %%B64%%;
    var bin = atob(GLB_B64);
    var buf = (function(){ var b = new Uint8Array(bin.length); for (var i=0;i<bin.length;i++) b[i]=bin.charCodeAt(i); return b.buffer; })();

    var renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputEncoding = THREE.sRGBEncoding;
    document.body.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x11151a);
    var camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.0001, 1000);

    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    var key = new THREE.DirectionalLight(0xffffff, 1.8); key.position.set(2,3,4); scene.add(key);
    var fill = new THREE.DirectionalLight(0x88bbff, 0.5); fill.position.set(-3,-1,-2); scene.add(fill);

    var mesh = null;
    var nVerts = %%NVERTS%%, nTris = %%NTRIS%%;

    new THREE.GLTFLoader().parse(buf, '', function(gltf){
      mesh = gltf.scene;
      mesh.traverse(function(o){ if (o.isMesh) o.geometry.computeVertexNormals(); });
      var box = new THREE.Box3().setFromObject(mesh);
      var c = box.getCenter(new THREE.Vector3());
      var s = box.getSize(new THREE.Vector3());
      mesh.position.sub(c);
      var d = Math.max(s.x, s.y, s.z) || 1;
      camera.position.set(d*0.6, d*0.45, d*1.2);
      camera.near = d/2000; camera.far = d*200; camera.updateProjectionMatrix();
      controls.target.set(0,0,0); controls.update();
      setMat('normal');
      if (statusEl) statusEl.style.display = 'none';
      document.getElementById('info').textContent =
        'FUMA mesh — ' + nVerts.toLocaleString() + ' verts / ' + nTris.toLocaleString() + ' tris\\n' +
        '拖拽=旋转  右键=平移  滚轮=缩放';
    }, function(err){ fail('模型解析失败：' + err); });

    function setMat(mode) {
      if (!mesh) return;
      mesh.traverse(function(o){ if (o.isMesh) {
        if (o.material) o.material.dispose();
        o.material = mode === 'normal' ? new THREE.MeshNormalMaterial()
                   : mode === 'wire'   ? new THREE.MeshStandardMaterial({ color:0x2a2a2a, wireframe:true })
                   : new THREE.MeshStandardMaterial({ color:0x9ab3c4, roughness:0.62, metalness:0.06 });
      }});
      ['bSolid','bNormal','bWire'].forEach(function(id){ document.getElementById(id).classList.remove('on'); });
      document.getElementById(mode==='normal'?'bNormal':mode==='wire'?'bWire':'bSolid').classList.add('on');
    }
    document.getElementById('bSolid').onclick = function(){ setMat('solid'); };
    document.getElementById('bNormal').onclick = function(){ setMat('normal'); };
    document.getElementById('bWire').onclick = function(){ setMat('wire'); };

    addEventListener('resize', function(){ camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
    (function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); })();
  } catch(e) {
    fail('渲染初始化失败：' + e.message);
  }
})();
</script>
</body>
</html>
"""

_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_lib")


def _read_lib(name):
    p = os.path.join(_LIB_DIR, name)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description="Generate a self-contained interactive mesh viewer HTML")
    ap.add_argument("mesh", help="input mesh (.ply/.obj/.glb/...)")
    ap.add_argument("-o", "--output", default="viewer.html")
    ap.add_argument("--simplify", type=int, default=0,
                    help="quadric-decimate to ~N triangles (0 = keep all)")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    print(f"loaded: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    if args.simplify > 0 and len(mesh.triangles) > args.simplify:
        target = min(args.simplify, len(mesh.triangles))
        mesh = mesh.simplify_quadric_decimation(target)
        print(f"simplified to {len(mesh.triangles)} tris")

    mesh.compute_vertex_normals()
    nv, nt = len(mesh.vertices), len(mesh.triangles)

    # export via trimesh so the GLB uses float32 (open3d writes float64 -> 2x size)
    import tempfile
    import trimesh
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float32),
                         faces=np.asarray(mesh.triangles, dtype=np.int32),
                         vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
                         process=False)
    with tempfile.TemporaryDirectory() as td:
        glb = os.path.join(td, "mesh.glb")
        tm.export(glb)
        data = open(glb, "rb").read()
    assert data[:4] == b"glTF", "GLB export failed (bad magic)"

    b64 = base64.b64encode(data).decode("ascii")
    # chunk into lines so the HTML stays editor-friendly
    lines = [b64[i:i + 120] for i in range(0, len(b64), 120)]
    b64_js = "[\n" + ",\n".join('"%s"' % ln for ln in lines) + "\n].join('')"

    html = (TEMPLATE
            .replace("%%THREE_JS%%", _read_lib("three.min.js"))
            .replace("%%ORBIT_JS%%", _read_lib("OrbitControls.js"))
            .replace("%%GLTF_JS%%", _read_lib("GLTFLoader.js"))
            .replace("%%B64%%", b64_js)
            .replace("%%NVERTS%%", str(nv))
            .replace("%%NTRIS%%", str(nt)))

    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB) "
          f"— open it in a browser / VSCode preview")


if __name__ == "__main__":
    main()
