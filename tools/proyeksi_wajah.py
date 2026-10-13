"""Tempel foto input generator ke tekstur model (bake Cycles): wajah + seluruh permukaan yang menghadap depan.

Pakai: blender -b -P tools/proyeksi_wajah.py -- <model.glb> <foto-input.png> <pose-input.json> <keluaran.glb> [masker.png]
Model hasil image-to-3D menghadap -Y dan sejajar dengan foto input (tampak depan), jadi titik (x, z)
model dapat dipetakan langsung ke piksel foto. Hanya area wajah yang menghadap depan yang diganti,
dengan tepi dibaurkan ke tekstur asli.
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, FOTO, POSE_JSON, OUT = argv[:4]
MASKER = argv[4] if len(argv) > 4 else None  # PNG putih = penari (sudah dikikis tepinya), hitam = latar

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in bpy.context.scene.objects:
    o.select_set(o in meshes)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
obj.parent = None
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(bpy.context.scene.objects):
    if o.type != "MESH":
        bpy.data.objects.remove(o)

vs = np.array([v.co[:] for v in obj.data.vertices])
zmin, zmax = vs[:, 2].min(), vs[:, 2].max()
xmid = (vs[:, 0].min() + vs[:, 0].max()) / 2

data = json.load(open(POSE_JSON))
W, H = data["w"], data["h"]
x0, y0, x1, y1 = data["bbox"]
lm = np.array(data["lm"])[:, :2]
s = (zmax - zmin) / (y1 - y0)          # meter per piksel
cx = (x0 + x1) / 2


def ke_model(px, py):
    return (px - cx) * s + xmid, (y1 - py) * s + zmin


hx, hz = ke_model(*lm[0])               # hidung
ex1, _ = ke_model(*lm[7])
ex2, _ = ke_model(*lm[8])
R = abs(ex1 - ex2) * 0.85               # radius area wajah
print(f"wajah di x={hx:.3f} z={hz:.3f}, radius {R:.3f}")

# titik yang tertutup bagian lain bila dilihat dari depan (-Y) tidak boleh menerima proyeksi
bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
arah = Vector((0, -1, 0))
attr = obj.data.color_attributes.new("terlihat", "FLOAT_COLOR", "POINT")
tertutup = 0
for i, vert in enumerate(obj.data.vertices):
    hit = bvh.ray_cast(vert.co + arah * 0.004, arah)
    lihat = 0.0 if hit[0] is not None else 1.0
    tertutup += lihat == 0.0
    attr.data[i].color = (lihat, lihat, lihat, 1)
print("titik tertutup dari depan:", tertutup, "/", len(obj.data.vertices))

mat = obj.data.materials[0]
nt = mat.node_tree
tex_node = next(n for n in nt.nodes if n.type == "TEX_IMAGE")
img = tex_node.image
asli = img.copy()                       # salinan untuk dibaca selama bake
asli.name = "tekstur_asli"
foto = bpy.data.images.load(FOTO)

# material sementara: campur tekstur asli dan foto proyeksi dengan masker wajah
bake_mat = bpy.data.materials.new("bake_wajah")
bake_mat.use_nodes = True
n = bake_mat.node_tree.nodes
L = bake_mat.node_tree.links
n.clear()
out = n.new("ShaderNodeOutputMaterial")
emi = n.new("ShaderNodeEmission")
L.new(emi.outputs[0], out.inputs["Surface"])
uvmap = n.new("ShaderNodeUVMap")
t_asli = n.new("ShaderNodeTexImage"); t_asli.image = asli
L.new(uvmap.outputs[0], t_asli.inputs[0])
geo = n.new("ShaderNodeNewGeometry")
sep = n.new("ShaderNodeSeparateXYZ")
L.new(geo.outputs["Position"], sep.inputs[0])


def math(op, a, b=None):
    m = n.new("ShaderNodeMath"); m.operation = op
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            L.new(v, m.inputs[i])
    return m.outputs[0]


# u = ((x - xmid)/s + cx) / W ; v = 1 - ((y1 - (z - zmin)/s) / H)
u = math("DIVIDE", math("ADD", math("DIVIDE", math("SUBTRACT", sep.outputs["X"], xmid), s), cx), W)
v = math("SUBTRACT", 1.0, math("DIVIDE", math("SUBTRACT", y1, math("DIVIDE", math("SUBTRACT", sep.outputs["Z"], zmin), s)), H))
comb = n.new("ShaderNodeCombineXYZ")
L.new(u, comb.inputs["X"]); L.new(v, comb.inputs["Y"])
t_foto = n.new("ShaderNodeTexImage"); t_foto.image = foto; t_foto.extension = "EXTEND"
L.new(comb.outputs[0], t_foto.inputs[0])

# masker: jarak (x,z) ke hidung -> 1 di dalam 0.55R, 0 di luar R; dikali seberapa menghadap depan
dx = math("SUBTRACT", sep.outputs["X"], hx)
dz = math("SUBTRACT", sep.outputs["Z"], hz)
jarak = math("SQRT", math("ADD", math("MULTIPLY", dx, dx), math("MULTIPLY", dz, dz)))
mr = n.new("ShaderNodeMapRange")
L.new(jarak, mr.inputs["Value"])
mr.inputs["From Min"].default_value = 0.55 * R
mr.inputs["From Max"].default_value = R
mr.inputs["To Min"].default_value = 1.0
mr.inputs["To Max"].default_value = 0.0
mr.interpolation_type = "SMOOTHSTEP"
sepn = n.new("ShaderNodeSeparateXYZ")
L.new(geo.outputs["Normal"], sepn.inputs[0])
hadap = n.new("ShaderNodeMapRange")       # normal.y -1 (depan) -> 1, -0.35 -> 0
L.new(sepn.outputs["Y"], hadap.inputs["Value"])
hadap.inputs["From Min"].default_value = -0.35
hadap.inputs["From Max"].default_value = -0.75
hadap.inputs["To Min"].default_value = 0.0
hadap.inputs["To Max"].default_value = 1.0
hadap.clamp = True
# permukaan tubuh lain: menghadap depan + tidak tertutup + di dalam siluet foto
hadap_badan = n.new("ShaderNodeMapRange")
L.new(sepn.outputs["Y"], hadap_badan.inputs["Value"])
hadap_badan.inputs["From Min"].default_value = -0.3
hadap_badan.inputs["From Max"].default_value = -0.65
hadap_badan.inputs["To Min"].default_value = 0.0
hadap_badan.inputs["To Max"].default_value = 1.0
hadap_badan.clamp = True
lihat = n.new("ShaderNodeAttribute"); lihat.attribute_name = "terlihat"
badan = math("MULTIPLY", hadap_badan.outputs[0], lihat.outputs["Fac"])
if MASKER:
    t_mask = n.new("ShaderNodeTexImage"); t_mask.image = bpy.data.images.load(MASKER)
    t_mask.image.colorspace_settings.name = "Non-Color"; t_mask.extension = "EXTEND"
    L.new(comb.outputs[0], t_mask.inputs[0])
    badan = math("MULTIPLY", badan, t_mask.outputs["Color"])
wajah = math("MULTIPLY", math("MULTIPLY", mr.outputs[0], hadap.outputs[0]), lihat.outputs["Fac"])
masker = math("MAXIMUM", wajah, badan)
mix = n.new("ShaderNodeMix"); mix.data_type = "RGBA"
L.new(masker, mix.inputs["Factor"])
L.new(t_asli.outputs["Color"], mix.inputs[6])
L.new(t_foto.outputs["Color"], mix.inputs[7])
L.new(mix.outputs[2], emi.inputs["Color"])
target = n.new("ShaderNodeTexImage"); target.image = img
n.active = target
for nd in n:
    nd.select = False
target.select = True

obj.data.materials[0] = bake_mat
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 4
sc.render.bake.margin = 4
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.bake(type="EMIT", use_clear=False)
print("bake selesai")

img.pack()
obj.data.materials[0] = mat
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_image_format="JPEG", export_jpeg_quality=92)
print("tersimpan", OUT)
