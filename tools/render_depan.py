"""Render tampak depan model (tekstur saja, tanpa cahaya) tepat di bingkai piksel foto input.

Dipakai untuk menyejajarkan foto tajam ke tekstur model secara presisi (lihat sejajar_foto.py).
Pakai: blender -b -P tools/render_depan.py -- <model.glb> <pose-tajam.json> <keluaran.png>
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, POSE_JSON, OUT = argv[:3]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
sc = bpy.context.scene
meshes = [o for o in sc.objects if o.type == "MESH"]
vs = []
for o in meshes:
    vs += [o.matrix_world @ v.co for v in o.data.vertices]
vs = np.array([v[:] for v in vs])
zmin, zmax = vs[:, 2].min(), vs[:, 2].max()
xmid = (vs[:, 0].min() + vs[:, 0].max()) / 2

data = json.load(open(POSE_JSON))
W, H = data["w"], data["h"]
x0, y0, x1, y1 = data["bbox"]
s = (zmax - zmin) / (y1 - y0)
cx = (x0 + x1) / 2

# material jadi emisi tekstur dasar (warna asli tanpa bayangan)
for o in meshes:
    for mat in o.data.materials:
        nt = mat.node_tree
        tex = next(n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "metal" not in n.image.name.lower())
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        emi = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
        nt.links.new(emi.outputs[0], out.inputs["Surface"])

sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.film_transparent = True
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = max(W, H) * s
# pusat gambar (W/2, H/2) dalam koordinat model
pusat = Vector((xmid + (W / 2 - cx) * s, 0, zmin + (y1 - H / 2) * s))
cam.location = pusat + Vector((0, -5, 0))
cam.rotation_euler = (Vector((0, 1, 0))).to_track_quat("-Z", "Y").to_euler()
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("render depan", OUT, W, H)
