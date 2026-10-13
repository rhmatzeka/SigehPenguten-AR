"""Render GLB dari 4 sudut untuk dicek (Blender headless); gabungkan dengan gabung_gambar.py.

Pakai: blender -b -P tools/render_cek.py -- <model.glb> <keluaran.png> [frame,frame,...] [depan,kanan,...]
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
frames = [int(f) for f in argv[2].split(",")] if len(argv) > 2 else [None]
sudut_pilih = argv[3].split(",") if len(argv) > 3 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
sc = bpy.context.scene
if frames[0] is not None:
    sc.frame_set(frames[0])
bpy.context.view_layer.update()

meshes = [o for o in sc.objects if o.type == "MESH"]
deps = bpy.context.evaluated_depsgraph_get()
pts = []
for o in meshes:
    ev = o.evaluated_get(deps)
    m = ev.to_mesh()
    pts += [ev.matrix_world @ v.co for v in list(m.vertices)[:: max(1, len(m.vertices) // 5000)]]
    ev.to_mesh_clear()
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
pusat, ukuran = (lo + hi) / 2, max(hi - lo)
print("bbox", tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi))

sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x, sc.render.resolution_y = 600, 900
sc.render.film_transparent = False
sc.world = bpy.data.worlds.new("w")
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.38, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 1.2

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type = "ORTHO"
cam.data.ortho_scale = ukuran * 1.15
sc.collection.objects.link(cam)
sc.camera = cam
lamp = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
lamp.data.energy = 3
lamp.rotation_euler = (math.radians(50), 0, math.radians(30))
sc.collection.objects.link(lamp)

# 4 sudut: depan (-Y), samping kanan (+X), belakang, samping kiri
sudut = [("depan", 0), ("kanan", 90), ("belakang", 180), ("kiri", 270)]
if sudut_pilih:
    sudut = [s for s in sudut if s[0] in sudut_pilih]
if frames[0] is not None:
    # animasi: bingkai tetap di sekitar poster + lantai supaya terlihat menapak atau melayang
    pusat, ukuran = Vector((0, 0, 0.55)), 1.15
    bpy.ops.mesh.primitive_plane_add(size=1.2)
    lantai = bpy.context.active_object
    mat = bpy.data.materials.new("lantai")
    mat.diffuse_color = (0.85, 0.75, 0.6, 1)
    lantai.data.materials.append(mat)
for fr in frames:
    if fr is not None:
        sc.frame_set(fr)
    for nama, deg in sudut:
        a = math.radians(deg)
        arah = Vector((math.sin(a), -math.cos(a), 0))
        cam.location = pusat + arah * ukuran * 3
        cam.rotation_euler = (-arah).to_track_quat("-Z", "Y").to_euler()
        cam.data.ortho_scale = ukuran * 1.15
        akhir = f"-f{fr:04d}-{nama}.png" if fr is not None else f"-{nama}.png"
        sc.render.filepath = out.replace(".png", akhir)
        bpy.ops.render.render(write_still=True)
