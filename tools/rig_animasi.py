"""Pasang tulang otomatis ke model hasil image-to-3D, lalu pasang gerakan mocap MediaPipe.

Pakai: blender -b -P tools/rig_animasi.py -- <model.glb> <pose-input.json> <gerak.npz> <keluaran.glb> [maks_frame]

- Letak sendi diambil dari landmark MediaPipe pada gambar input generator (tampak depan),
  dipetakan ke bounding box model; kedalaman = tengah irisan mesh di titik itu.
- Bobot kulit: auto weights pada proxy hasil voxel remesh (mesh AI sering bolong), lalu
  ditransfer ke mesh asli. Rok tapis dibuat lebih banyak ikut pinggul supaya tidak melar.
- Gerakan: tiap tulang diputar supaya arahnya sama dengan arah ruas di data mocap.
"""
import json
import math
import sys

import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tangan import ganti_tangan  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, POSE_JSON, GERAK, OUT = argv[:4]
MAKS_FRAME = int(argv[4]) if len(argv) > 4 and argv[4] != "-" else None
MAKS_SEGITIGA = 70000

# indeks landmark MediaPipe
NOSE, EAR_L, EAR_R = 0, 7, 8
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
PK_L, PK_R, IX_L, IX_R = 17, 18, 19, 20
HP_L, HP_R, KN_L, KN_R, AN_L, AN_R, FT_L, FT_R = 23, 24, 25, 26, 27, 28, 31, 32
LENGAN_UJUNG = {EL_L, EL_R, WR_L, WR_R, PK_L, PK_R, IX_L, IX_R}
JARI_LENGAN = 0.028        # perkiraan jari-jari lengan berlengan renda (model setinggi ~1)
TEBAL_LENGAN_MAKS = 0.08   # irisan lebih tebal dari ini berarti lengan menempel badan/siger
# titik yang lebih jauh dari ruas lengan daripada ini tidak boleh ikut lengan (siger/dada yang tersentuh tangan)
LENGAN_R0, LENGAN_R1 = 0.06, 0.085
LENGAN_BELAKANG = 0.03     # titik sejauh ini di belakang lengan yang menempel di depan badan = badan
GELANG_R = 0.035           # gelang emas di pergelangan tetap ikut lengan


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
    # glTF memecah titik di sambungan UV -> satukan lagi lalu haluskan normal supaya tidak bersegi-segi
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.to_mesh(obj.data)
    bm.free()
    # warna titik bawaan generator & atribut bantu proyeksi tidak dipakai di web, hanya menambah ukuran
    for nama in [a.name for a in obj.data.color_attributes]:
        obj.data.color_attributes.remove(obj.data.color_attributes[nama])
    # model detail (TRELLIS.2 ~200rb segitiga) dikecilkan supaya ringan di HP
    if len(obj.data.polygons) > MAKS_SEGITIGA:
        dec = obj.modifiers.new("dec", "DECIMATE")
        dec.ratio = MAKS_SEGITIGA / len(obj.data.polygons)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="dec")
    for p in obj.data.polygons:
        p.use_smooth = True
    if hasattr(obj.data, "set_sharp_from_angle"):
        obj.data.set_sharp_from_angle(angle=math.radians(50))
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

    def kedalaman(i, x, z, r=0.03):
        m = (np.abs(vs[:, 0] - x) < r) & (np.abs(vs[:, 2] - z) < r)
        if m.sum() < 5:
            return 0.0
        y = vs[m, 1]
        if i == NOSE:
            return float(y.min())
        # lengan bawah/tangan di depan badan: irisan ikut menembus badan di belakangnya,
        # jadi sendi ditaruh sedikit di belakang permukaan depan (bukan di tengah badan)
        if i in LENGAN_UJUNG and y.max() - y.min() > TEBAL_LENGAN_MAKS:
            menempel.add(i)
            return float(y.min()) + JARI_LENGAN
        return float((y.min() + y.max()) / 2)

    # hidung ada di permukaan depan wajah (model menghadap -Y), sendi lain di tengah irisan
    menempel = set()
    J = {i: Vector((xz[i, 0], kedalaman(i, *xz[i]), xz[i, 1])) for i in range(len(xz))}
    pisahkan_lengan.tempel = [J[i] for i in menempel]
    print("sendi lengan menempel badan:", sorted(menempel))
    return J


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


def pasang_bobot(obj, arm, J):
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
    pisahkan_lengan(obj, arm)
    putus_jembatan(obj, arm)
    rapikan_rok(obj, arm)
    # tangan hasil generator cacat (jari menyatu) -> ganti tangan buatan dengan tanggai emas
    kulit = tuple(c ** 2.2 for c in (0.88, 0.70, 0.62))  # sRGB -> linear
    ganti_tangan(obj, arm, J, (IX_L, IX_R), (PK_L, PK_R), kulit)

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


def jarak_ruas(p, a, b):
    ab = b - a
    t = min(1.0, max(0.0, (p - a).dot(ab) / max(1e-9, ab.length_squared)))
    return (p - (a + ab * t)).length


def warna_emas(obj):
    """Per titik: True bila warna teksturnya emas (siger), False untuk renda putih, kulit, pita merah, rompi."""
    me = obj.data
    hasil = np.zeros(len(me.vertices), bool)
    gambar = {}
    for i, mat in enumerate(me.materials):
        for nd in (mat.node_tree.nodes if mat and mat.use_nodes else []):
            # tekstur warna bisa langsung ke Base Color atau lewat node Mix (campuran warna titik hasil proyeksi)
            if nd.type == "TEX_IMAGE" and nd.image and any(l.to_socket.name == "Base Color" or l.to_node.type == "MIX"
                                                         for l in nd.outputs[0].links):
                img = nd.image
                gambar[i] = (np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], -1)[..., :3], img.size)
    if not gambar or not me.uv_layers.active:
        return hasil
    uv = me.uv_layers.active.data
    jumlah = np.zeros((len(me.vertices), 3))
    n = np.zeros(len(me.vertices))
    for poly in me.polygons:
        if poly.material_index not in gambar:
            continue
        px, (w, h) = gambar[poly.material_index]
        for li in poly.loop_indices:
            u, v = uv[li].uv
            c = px[int(min(h - 1, max(0, v * h))), int(min(w - 1, max(0, u * w)))]
            vi = me.loops[li].vertex_index
            jumlah[vi] += c
            n[vi] += 1
    rata = jumlah / np.maximum(n, 1)[:, None]
    mx, mn = rata.max(1), rata.min(1)
    jenuh = (mx - mn) / np.maximum(mx, 1e-6)
    r, g, b = rata.T
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    # emas siger: kuning-jingga, cukup terang & jenuh (kulit lebih merah, renda putih tidak jenuh)
    return (n > 0) & (jenuh > 0.25) & (mx > 0.3) & (hue > 30) & (hue < 62)


def pisahkan_lengan(obj, arm):
    """Pose foto: tangan menyentuh siger / di depan dada, jadi bone heat ikut membobot siger & dada ke lengan
    (melar saat lengan bergerak). Bobot lengan pada titik yang jauh dari ruas lengan dipindah ke kepala/dada."""
    bones = arm.data.bones
    z_leher = bones["neck"].head_local.z
    z_kepala = bones["head"].head_local.z
    vg = {g.name: g for g in obj.vertex_groups}
    idx = {g.index: g.name for g in obj.vertex_groups}
    dipindah = 0
    emas = warna_emas(obj)
    tempel = getattr(pisahkan_lengan, "tempel", [])

    def sendi_tempel(p):
        return any((p - q).length < 1e-4 for q in tempel)

    for sisi in "LR":
        rantai = [f"upper_arm.{sisi}", f"forearm.{sisi}", f"hand.{sisi}"]
        ruas = []
        for n in rantai:
            b = bones[n]
            ekor = b.tail_local
            if n.startswith("hand"):
                ekor = b.head_local + (b.tail_local - b.head_local) * 1.6  # jari & kuku tanggai
            ruas.append((b.head_local.copy(), ekor.copy()))
        W = bones[f"hand.{sisi}"].head_local
        fore = bones[f"forearm.{sisi}"]
        arah = (fore.tail_local - fore.head_local).normalized()
        Lp = 0.4 * fore.length  # perkiraan panjang telapak, sama seperti tangan.py
        for v in obj.data.vertices:
            w = {idx[g.group]: g.weight for g in v.groups}
            w_lengan = sum(w.get(n, 0) for n in rantai)
            if w_lengan <= 0:
                continue
            d = min(jarak_ruas(v.co, a, b) for a, b in ruas)
            f = min(1.0, max(0.0, (d - LENGAN_R0) / (LENGAN_R1 - LENGAN_R0)))
            # siger (emas, bukan renda putih/kulit) yang tersentuh tangan terangkat ikut kepala
            # (jari tangan asli yang bertumpuk dengan siger tidak dibedakan: tangannya diganti tangan buatan)
            if v.co.z > z_kepala - 0.02 and emas[v.index] and jarak_ruas(v.co, *ruas[1]) > GELANG_R:
                f = 1.0
            # zona tangan asli (di depan pergelangan) dekat siger: tangan asli nanti dihapus (tangan.py, bobot >= 0.5)
            # dan diganti tangan buatan; titik bobot campuran di sini jadi lembaran melar, jadi ikut kepala penuh
            if v.co.z > z_kepala - 0.02 and (v.co - W).dot(arah) > 0.004 and (v.co - W).length < 4 * Lp:
                if w.get(f"hand.{sisi}", 0) + w.get(f"forearm.{sisi}", 0) < 0.5:
                    f = 1.0
            # lengan bawah/tangan menempel di depan badan: yang di belakangnya adalah badan (selendang, kalung)
            for a, b in ruas[1:]:
                if sendi_tempel(a) or sendi_tempel(b):
                    ab = b - a
                    t = min(1.0, max(0.0, (v.co - a).dot(ab) / max(1e-9, ab.length_squared)))
                    if v.co.y - (a + ab * t).y > LENGAN_BELAKANG:
                        f = 1.0
            if f <= 0:
                continue
            tujuan = "head" if v.co.z > z_kepala - 0.02 else ("neck" if v.co.z > z_leher + 0.02 else "chest")
            for n in rantai:
                if n in w:
                    vg[n].add([v.index], w[n] * (1 - f), "REPLACE")
            vg[tujuan].add([v.index], w.get(tujuan, 0) + w_lengan * f, "REPLACE")
            dipindah += 1
    print("pisahkan lengan: titik dipindah", dipindah)


def putus_jembatan(obj, arm):
    """Mesh generator menyatu di tempat tangan menyentuh siger/badan; segitiga yang menghubungkan titik milik
    lengan bawah/tangan dengan titik milik kepala/badan akan melar saat lengan bergerak, jadi dibuang."""
    gi = {g.name: g.index for g in obj.vertex_groups}
    badan = {gi[n] for n in ("head", "neck", "chest", "spine")}
    lengan = {gi[n] for s in "LR" for n in (f"forearm.{s}", f"hand.{s}")}
    # di samping siger lengan atas yang terangkat juga bukan bagian kepala
    lengan_siger = lengan | {gi["upper_arm.L"], gi["upper_arm.R"]}
    z_siger = arm.data.bones["head"].head_local.z - 0.02
    # di area siger bobot campuran kepala+lengan ditegaskan ke yang dominan, supaya batasnya tajam dan terpotong
    nama = {g.index: g.name for g in obj.vertex_groups}
    for v in obj.data.vertices:
        if v.co.z <= z_siger:
            continue
        b_ = sum(g.weight for g in v.groups if g.group in badan)
        l_ = sum(g.weight for g in v.groups if g.group in lengan_siger)
        if b_ > 0.1 and l_ > 0.1:
            kalah = lengan_siger if b_ > l_ else badan
            for g in list(v.groups):
                if g.group in kalah:
                    obj.vertex_groups[nama[g.group]].remove([v.index])
    wb, wl = [], []
    for v in obj.data.vertices:
        wb.append(sum(g.weight for g in v.groups if g.group in badan))
        wl.append(sum(g.weight for g in v.groups if g.group in (lengan_siger if v.co.z > z_siger else lengan)))
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # bobot lengan berubah tajam dalam satu segitiga yang juga menyentuh kepala/badan = jembatan
    # (siku tidak kena: di sana pasangannya lengan atas, bukan badan)
    buang = [f for f in bm.faces
             if max(wl[v.index] for v in f.verts) - min(wl[v.index] for v in f.verts) > 0.35
             and max(wb[v.index] for v in f.verts) > 0.35]
    bmesh.ops.delete(bm, geom=buang, context="FACES_ONLY")
    bm.to_mesh(obj.data)
    bm.free()
    print("putus jembatan lengan-badan:", len(buang), "segitiga")


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
        # 80% bobot kaki pindah ke pinggul di atas, 48% setinggi lutut ke bawah
        t = min(1.0, max(0.0, (z_pinggul - v.co.z) / max(1e-6, z_pinggul - z_lutut)))
        # rok jatuh dari pinggul (tidak ikut betis yang terlipat saat bersimpuh); yang tembus lantai
        # digerai rata di lantai oleh shader web (index.html, "rok tergerai")
        pindah = (1 - 0.13 * t) * 0.92
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


def batasi_miring(v, maks_derajat):
    """Putar vektor `v` mendekati sumbu atas bila kemiringannya melebihi `maks_derajat`."""
    atas = Vector((0, 0, 1))
    sudut = v.angle(atas)
    batas = math.radians(maks_derajat)
    if sudut <= batas:
        return v
    poros = atas.cross(v).normalized()
    return Matrix.Rotation(batas, 3, poros) @ atas * v.length


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
            # panggul tetap tegak (hanya menoleh): duduk bersimpuh tidak memiringkan panggul,
            # bungkukan (sembah) cukup di badan atas, dibatasi 45 derajat supaya tidak rebah
            "hips": frame_orient(P[HP_L] - P[HP_R], Vector((0, 0, 1))),
            "chest": frame_orient(P[SH_L] - P[SH_R], batasi_miring(sh - hp, 45)),
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
            if nama in ("head", "neck"):
                # telinga & hidung tertutup siger sehingga titik MediaPipe meleset; leher dan kepala ikut dada
                R = R_bawa
            elif nama in F_rest:
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
                if nama.startswith("hand."):
                    # titik jari MediaPipe di video kecil & berisik: redam putaran pergelangan
                    R = R_bawa.to_quaternion().slerp(R.to_quaternion(), 0.6).to_matrix()
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


def tempel_lantai(arm, obj):
    """Tiap frame: geser seluruh rig supaya lutut/pergelangan/ujung kaki terendah tepat di lantai z=0.
    Patokannya tulang, bukan ujung rok: rok yang tembus lantai digerai rata oleh shader web."""
    sc = bpy.context.scene
    pb = arm.pose.bones
    # (tulang, ujung, tebal): tebal = jarak permukaan kulit/kain ke sendi
    titik = [("shin.L", "head", 0.014), ("shin.R", "head", 0.014),
             ("foot.L", "head", 0.02), ("foot.R", "head", 0.02),
             ("foot.L", "tail", 0.006), ("foot.R", "tail", 0.006)]
    terendah = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        mw = arm.matrix_world
        terendah.append(min((mw @ getattr(pb[t], u)).z - tebal for t, u, tebal in titik))
    t = np.array(terendah)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); k /= k.sum()
    t = np.convolve(np.pad(t, 6, mode="edge"), k, mode="valid")
    for i, f in enumerate(range(sc.frame_start, sc.frame_end + 1)):
        arm.location.z = -t[i]
        arm.keyframe_insert("location", index=2, frame=f)
    print(f"tempel lantai (tulang): koreksi {t.min():.3f}..{t.max():.3f}")


def main():
    obj = siapkan_model()
    J = sendi_dari_foto(obj)
    daftar = struktur_tulang(J)
    arm = buat_armature(daftar)
    pasang_bobot(obj, arm, J)
    pasang_gerak(arm, J)
    tempel_lantai(arm, obj)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_animations=True,
                              export_force_sampling=True, export_optimize_animation_size=True,
                              export_frame_step=2, export_image_format="JPEG", export_jpeg_quality=85,
                              export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=7,
                              export_draco_position_quantization=14, export_draco_normal_quantization=10,
                              export_draco_texcoord_quantization=12)
    print("tersimpan", OUT)


main()
