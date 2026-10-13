"""Pasang tulang otomatis ke model hasil image-to-3D, lalu pasang gerakan mocap MediaPipe.

Pakai: blender -b -P tools/rig_animasi.py -- <model.glb> <pose-input.json> <gerak.npz> <keluaran.glb> [maks_frame]

- Letak sendi diambil dari landmark MediaPipe pada gambar input generator (tampak depan),
  dipetakan ke bounding box model; kedalaman = tengah irisan mesh di titik itu.
- Bobot kulit: auto weights pada proxy hasil voxel remesh (mesh AI sering bolong), lalu
  ditransfer ke mesh asli. Rok tapis dibuat lebih banyak ikut pinggul supaya tidak melar.
- Gerakan: tiap tulang diputar supaya arahnya sama dengan arah ruas di data mocap.
"""
import json
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.kdtree import KDTree

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, POSE_JSON, GERAK, OUT = argv[:4]
MAKS_FRAME = int(argv[4]) if len(argv) > 4 else None

# indeks landmark MediaPipe
NOSE, EAR_L, EAR_R = 0, 7, 8
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
PK_L, PK_R, IX_L, IX_R = 17, 18, 19, 20
HP_L, HP_R, KN_L, KN_R, AN_L, AN_R, FT_L, FT_R = 23, 24, 25, 26, 27, 28, 31, 32


def siapkan_model():
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
    # buang objek kosong sisa impor
    for o in list(bpy.context.scene.objects):
        if o.type != "MESH":
            bpy.data.objects.remove(o)
    # kaki di z=0, tengah di x=y=0
    vs = np.array([v.co[:] for v in obj.data.vertices])
    geser = Vector((-(vs[:, 0].min() + vs[:, 0].max()) / 2, -(vs[:, 1].min() + vs[:, 1].max()) / 2, -vs[:, 2].min()))
    obj.data.transform(Matrix.Translation(geser))
    obj.name = "Penari"
    # glTF tanpa metallicFactor dianggap logam penuh (gelap); jadikan kain/kulit biasa
    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            for nd in mat.node_tree.nodes:
                if nd.type == "BSDF_PRINCIPLED":
                    nd.inputs["Metallic"].default_value = 0.0
                    nd.inputs["Roughness"].default_value = 0.8
    print("mesh", len(obj.data.vertices), "verts", len(obj.data.polygons), "faces", "tinggi", round(vs[:, 2].max() - vs[:, 2].min(), 3))
    return obj


def sendi_dari_foto(obj):
    data = json.load(open(POSE_JSON))
    x0, y0, x1, y1 = data["bbox"]
    lm = np.array(data["lm"])[:, :2]
    vs = np.array([v.co[:] for v in obj.data.vertices])
    tinggi = vs[:, 2].max()
    s = tinggi / (y1 - y0)
    cx = (x0 + x1) / 2
    xz = np.stack([(lm[:, 0] - cx) * s, (y1 - lm[:, 1]) * s], axis=1)

    def kedalaman(x, z, depan=False, r=0.03):
        m = (np.abs(vs[:, 0] - x) < r) & (np.abs(vs[:, 2] - z) < r)
        if m.sum() < 5:
            return 0.0
        y = vs[m, 1]
        return float(y.min()) if depan else float((y.min() + y.max()) / 2)

    # hidung ada di permukaan depan wajah (model menghadap -Y), sendi lain di tengah irisan
    return {i: Vector((xz[i, 0], kedalaman(*xz[i], depan=(i == NOSE)), xz[i, 1])) for i in range(len(xz))}


def tengah(*ps):
    return sum(ps, Vector()) / len(ps)


def struktur_tulang(J):
    """Daftar (nama, induk, kepala, ekor, terhubung)."""
    hip = tengah(J[HP_L], J[HP_R])
    sh = tengah(J[SH_L], J[SH_R])
    ear = tengah(J[EAR_L], J[EAR_R])
    atas = Vector((0, 0, 1))
    tangan_l = tengah(J[PK_L], J[IX_L])
    tangan_r = tengah(J[PK_R], J[IX_R])
    kaki_maju = Vector((0, -0.07, -0.01))
    return [
        ("hips", None, hip, hip.lerp(sh, 0.35), False),
        ("spine", "hips", hip.lerp(sh, 0.35), hip.lerp(sh, 0.7), True),
        ("chest", "spine", hip.lerp(sh, 0.7), sh, True),
        ("neck", "chest", sh, ear, True),
        ("head", "neck", ear, ear + atas * 0.15, True),
        ("shoulder.L", "chest", sh, J[SH_L], False),
        ("upper_arm.L", "shoulder.L", J[SH_L], J[EL_L], True),
        ("forearm.L", "upper_arm.L", J[EL_L], J[WR_L], True),
        ("hand.L", "forearm.L", J[WR_L], tangan_l, True),
        ("shoulder.R", "chest", sh, J[SH_R], False),
        ("upper_arm.R", "shoulder.R", J[SH_R], J[EL_R], True),
        ("forearm.R", "upper_arm.R", J[EL_R], J[WR_R], True),
        ("hand.R", "forearm.R", J[WR_R], tangan_r, True),
        ("thigh.L", "hips", J[HP_L], J[KN_L], False),
        ("shin.L", "thigh.L", J[KN_L], J[AN_L], True),
        ("foot.L", "shin.L", J[AN_L], J[AN_L] + kaki_maju, True),
        ("thigh.R", "hips", J[HP_R], J[KN_R], False),
        ("shin.R", "thigh.R", J[KN_R], J[AN_R], True),
        ("foot.R", "shin.R", J[AN_R], J[AN_R] + kaki_maju, True),
    ]


def buat_armature(daftar):
    arm_data = bpy.data.armatures.new("Rig")
    arm = bpy.data.objects.new("Rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for nama, induk, kepala, ekor, sambung in daftar:
        eb = arm_data.edit_bones.new(nama)
        eb.head, eb.tail = kepala, ekor
        eb.roll = 0
        if induk:
            eb.parent = arm_data.edit_bones[induk]
            eb.use_connect = sambung
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def pasang_bobot(obj, arm):
    # proxy rapat (watertight) supaya bone heat tidak gagal
    proxy = obj.copy()
    proxy.data = obj.data.copy()
    proxy.name = "Proxy"
    bpy.context.scene.collection.objects.link(proxy)
    proxy.data.remesh_voxel_size = 0.012
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    bpy.ops.object.voxel_remesh()
    dec = proxy.modifiers.new("dec", "DECIMATE")
    dec.ratio = min(1.0, 40000 / max(1, len(proxy.data.polygons)))
    bpy.ops.object.modifier_apply(modifier="dec")
    print("proxy", len(proxy.data.polygons), "faces")

    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    kosong = sum(1 for v in proxy.data.vertices if not v.groups)
    print("proxy tanpa bobot:", kosong, "/", len(proxy.data.vertices))

    # transfer bobot ke mesh asli
    for b in arm.data.bones:
        obj.vertex_groups.new(name=b.name)
    dt = obj.modifiers.new("dt", "DATA_TRANSFER")
    dt.object = proxy
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier="dt")
    bpy.data.objects.remove(proxy)

    isi_bobot_kosong(obj)
    rapikan_rok(obj, arm)

    obj.parent = arm
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm


def isi_bobot_kosong(obj):
    """Titik tanpa bobot (mis. ujung kuku tanggai yang hilang di proxy) menyalin bobot titik berbobot terdekat."""
    vs = obj.data.vertices
    berbobot = [v.index for v in vs if sum(g.weight for g in v.groups) > 0.01]
    kosong = [v.index for v in vs if sum(g.weight for g in v.groups) <= 0.01]
    kd = KDTree(len(berbobot))
    for i in berbobot:
        kd.insert(vs[i].co, i)
    kd.balance()
    for i in kosong:
        _, j, _ = kd.find(vs[i].co)
        for g in vs[j].groups:
            obj.vertex_groups[g.group].add([i], g.weight, "REPLACE")
    print("titik tanpa bobot diisi:", len(kosong))


def rapikan_rok(obj, arm):
    """Rok: campur bobot paha/betis dengan pinggul makin ke atas, supaya kain tidak terbelah."""
    bones = arm.data.bones
    z_pinggul = bones["hips"].head_local.z
    z_lutut = (bones["shin.L"].head_local.z + bones["shin.R"].head_local.z) / 2
    vg = {g.name: g for g in obj.vertex_groups}
    idx = {g.index: g.name for g in obj.vertex_groups}
    kaki = {"thigh.L", "thigh.R", "shin.L", "shin.R"}
    for v in obj.data.vertices:
        if v.co.z > z_pinggul or v.co.z < 0.02:
            continue
        w = {idx[g.group]: g.weight for g in v.groups}
        w_kaki = sum(w.get(n, 0) for n in kaki)
        if w_kaki <= 0:
            continue
        # 0 di pinggul -> 0.5 di lutut ke bawah: setengah bobot kaki dipindah ke pinggul
        t = min(1.0, max(0.0, (z_pinggul - v.co.z) / max(1e-6, z_pinggul - z_lutut)))
        pindah = (1 - 0.5 * t) * 0.6
        for n in kaki:
            if n in w:
                vg[n].add([v.index], w[n] * (1 - pindah), "REPLACE")
        vg["hips"].add([v.index], w.get("hips", 0) + w_kaki * pindah, "REPLACE")


# ---------- gerakan ----------

def frame_orient(x, atas):
    """Matriks 3x3 dari sumbu lateral x dan petunjuk atas."""
    x = x.normalized()
    z = (atas - x * atas.dot(x)).normalized()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed()


def pasang_gerak(arm, J_rest):
    d = np.load(GERAK)
    pos = d["pos"]
    fps = float(d["fps"])
    if MAKS_FRAME:
        pos = pos[:MAKS_FRAME]
    n = len(pos)
    sc = bpy.context.scene
    sc.render.fps = int(round(fps))
    sc.frame_start, sc.frame_end = 0, n - 1

    bones = arm.data.bones
    pb = arm.pose.bones

    # skala: panjang kaki model vs mocap (median)
    def panjang_kaki(P):
        return (np.linalg.norm(P[..., HP_L, :] - P[..., KN_L, :], axis=-1) + np.linalg.norm(P[..., KN_L, :] - P[..., AN_L, :], axis=-1)
                + np.linalg.norm(P[..., HP_R, :] - P[..., KN_R, :], axis=-1) + np.linalg.norm(P[..., KN_R, :] - P[..., AN_R, :], axis=-1)) / 2
    kaki_model = ((J_rest[HP_L] - J_rest[KN_L]).length + (J_rest[KN_L] - J_rest[AN_L]).length
                  + (J_rest[HP_R] - J_rest[KN_R]).length + (J_rest[KN_R] - J_rest[AN_R]).length) / 2
    # skala dari tinggi pinggul saat berdiri (persentil 95), lebih stabil daripada panjang kaki di balik rok
    hip_mocap = float(np.percentile((pos[:, HP_L, 2] + pos[:, HP_R, 2]) / 2, 95))
    skala = bones["hips"].head_local.z / hip_mocap
    print(f"skala mocap->model {skala:.3f} (kaki: {kaki_model / float(np.median(panjang_kaki(pos))):.3f}), pinggul model {bones['hips'].head_local.z:.3f}")

    hip_rest = bones["hips"].head_local.copy()
    # orientasi istirahat (rest) dari sendi model
    sh_r = tengah(J_rest[SH_L], J_rest[SH_R])
    hp_r = tengah(J_rest[HP_L], J_rest[HP_R])
    ear_r = tengah(J_rest[EAR_L], J_rest[EAR_R])
    F_rest = {
        "hips": frame_orient(J_rest[HP_L] - J_rest[HP_R], sh_r - hp_r),
        "chest": frame_orient(J_rest[SH_L] - J_rest[SH_R], sh_r - hp_r),
        "head": frame_orient(J_rest[EAR_L] - J_rest[EAR_R], (ear_r - J_rest[NOSE]).cross(J_rest[EAR_L] - J_rest[EAR_R]) * -1),
    }
    ruas = {  # tulang -> (titik awal, titik akhir) di mocap
        "neck": (lambda P: tengah(P[SH_L], P[SH_R]), lambda P: tengah(P[EAR_L], P[EAR_R])),
        "shoulder.L": (lambda P: tengah(P[SH_L], P[SH_R]), lambda P: P[SH_L]),
        "shoulder.R": (lambda P: tengah(P[SH_L], P[SH_R]), lambda P: P[SH_R]),
        "upper_arm.L": (lambda P: P[SH_L], lambda P: P[EL_L]),
        "forearm.L": (lambda P: P[EL_L], lambda P: P[WR_L]),
        "hand.L": (lambda P: P[WR_L], lambda P: tengah(P[PK_L], P[IX_L])),
        "upper_arm.R": (lambda P: P[SH_R], lambda P: P[EL_R]),
        "forearm.R": (lambda P: P[EL_R], lambda P: P[WR_R]),
        "hand.R": (lambda P: P[WR_R], lambda P: tengah(P[PK_R], P[IX_R])),
        "thigh.L": (lambda P: P[HP_L], lambda P: P[KN_L]),
        "shin.L": (lambda P: P[KN_L], lambda P: P[AN_L]),
        "foot.L": (lambda P: P[AN_L], lambda P: P[FT_L]),
        "thigh.R": (lambda P: P[HP_R], lambda P: P[KN_R]),
        "shin.R": (lambda P: P[KN_R], lambda P: P[AN_R]),
        "foot.R": (lambda P: P[AN_R], lambda P: P[FT_R]),
    }
    urutan = [b.name for b in bones]  # induk selalu sebelum anak (urutan pembuatan)
    for b in pb:
        b.rotation_mode = "QUATERNION"

    for f in range(n):
        P = {i: Vector(pos[f, i]) for i in range(33)}
        hp = tengah(P[HP_L], P[HP_R])
        sh = tengah(P[SH_L], P[SH_R])
        ear = tengah(P[EAR_L], P[EAR_R])
        F_now = {
            "hips": frame_orient(P[HP_L] - P[HP_R], sh - hp),
            "chest": frame_orient(P[SH_L] - P[SH_R], sh - hp),
            "head": frame_orient(P[EAR_L] - P[EAR_R], (ear - P[NOSE]).cross(P[EAR_L] - P[EAR_R]) * -1),
        }
        M = {}  # matriks pose ruang armature
        for nama in urutan:
            bone = bones[nama]
            rest = bone.matrix_local
            if bone.parent:
                rel = bone.parent.matrix_local.inverted() @ rest
                bawa = M[bone.parent.name] @ rel
            else:
                bawa = rest.copy()
            R_bawa = bawa.to_3x3()
            if nama in F_rest:
                R = F_now[nama] @ F_rest[nama].transposed() @ rest.to_3x3()
            elif nama == "spine":
                qa = (F_now["hips"] @ F_rest["hips"].transposed()).to_quaternion()
                qb = (F_now["chest"] @ F_rest["chest"].transposed()).to_quaternion()
                R = qa.slerp(qb, 0.5).to_matrix() @ rest.to_3x3()
            else:
                a, b = ruas[nama]
                arah = b(P) - a(P)
                y_bawa = R_bawa.col[1]
                R = y_bawa.rotation_difference(arah.normalized()).to_matrix() @ R_bawa
            lokasi = bawa.translation
            if nama == "hips":
                lokasi = Vector((hip_rest.x, hip_rest.y, hp.z * skala))
            M[nama] = Matrix.Translation(lokasi) @ R.to_4x4()
            if bone.parent:
                basis = rel.inverted() @ M[bone.parent.name].inverted() @ M[nama]
            else:
                basis = rest.inverted() @ M[nama]
            p = pb[nama]
            p.rotation_quaternion = basis.to_quaternion()
            p.keyframe_insert("rotation_quaternion", frame=f)
            if nama == "hips":
                p.location = basis.translation
                p.keyframe_insert("location", frame=f)
    arm.animation_data.action.name = "Tari"
    print("keyframe", n, "frame")


def main():
    obj = siapkan_model()
    J = sendi_dari_foto(obj)
    daftar = struktur_tulang(J)
    arm = buat_armature(daftar)
    pasang_bobot(obj, arm)
    pasang_gerak(arm, J)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_animations=True,
                              export_force_sampling=True, export_optimize_animation_size=True,
                              export_frame_step=2, export_image_format="JPEG", export_jpeg_quality=85)
    print("tersimpan", OUT)


main()
