"""Ganti lengan hasil generator 3D dengan lengan buatan utuh (lengan baju renda putih).

Di foto sumber tangan kanan menyentuh siger dan lengan bawah kiri melintang di depan perut, sehingga
generator membuat lengan dan baju menjadi satu permukaan. Bagaimanapun bobot kulitnya diatur, saat lengan
bergerak kain di sekitarnya sobek, melar, atau lengan tampak putus. Jadi:

1. Titik lengan asli (bobot lengan >= 0.5) dibuang; sisa bobot lengan pada badan dipindah ke badan sehingga
   badan kaku dan tidak ada serpihan yang ikut tertarik lengan. Lubang bekas lengan di badan ditambal.
2. Lengan baru dibangun sepanjang tulang: lengan baju renda dari dalam bahu sampai pergelangan, kerut renda di
   siku, gelang emas di lengan atas. Satu potongan utuh; sendi siku dibobot bertahap sehingga menekuk halus.

Dipanggil dari rig_animasi.py sesudah ganti_tangan (tangan buatan & gelang kano tidak disentuh).
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from tangan import _cincin, material

N = 16  # titik per keliling lengan


def _jarak(p, a, b):
    ab = b - a
    t = min(1.0, max(0.0, (p - a).dot(ab) / max(1e-12, ab.length_squared)))
    return (p - (a + ab * t)).length


def _tabung_uv(bm, uv, titik, jari2, bobot, dl, ulang_v):
    """Tabung terbuka sepanjang `titik` dengan UV (u keliling, v sepanjang) dan bobot tulang per cincin."""
    acuan = None
    cincin = []
    panjang = [0.0]
    for a, b in zip(titik, titik[1:]):
        panjang.append(panjang[-1] + (b - a).length)
    for i, (p, r) in enumerate(zip(titik, jari2)):
        arah = ((titik[min(i + 1, len(titik) - 1)] - titik[max(i - 1, 0)])).normalized()
        if acuan is None:
            acuan = arah.orthogonal().normalized()
        x = (acuan - arah * acuan.dot(arah)).normalized()   # sumbu keliling diteruskan supaya tidak terpuntir
        acuan = x
        z = arah.cross(x)
        ring = []
        for k in range(N):
            a = 2 * math.pi * k / N
            v = bm.verts.new(p + (x * math.cos(a) + z * math.sin(a)) * r)
            for g, w in bobot[i].items():
                v[dl][g] = w
            ring.append(v)
        cincin.append(ring)
    for i, (a, b) in enumerate(zip(cincin, cincin[1:])):
        for k in range(N):
            k2 = (k + 1) % N
            f = bm.faces.new((a[k], a[k2], b[k2], b[k]))
            for loop, (kk, ii) in zip(f.loops, ((k, i), (k + 1, i), (k + 1, i + 1), (k, i + 1))):
                loop[uv].uv = (kk / N * 2, panjang[ii] * ulang_v)
    return cincin


def ganti_lengan(obj, arm):
    me = obj.data
    bones = arm.data.bones
    gi = {g.name: g.index for g in obj.vertex_groups}
    nama_g = {g.index: g.name for g in obj.vertex_groups}
    buatan_mat = {i for i, m in enumerate(me.materials) if m and m.name.startswith(("kulit_tangan", "emas_tanggai"))}
    buatan = set()
    for p in me.polygons:
        if p.material_index in buatan_mat:
            buatan.update(p.vertices)
    rantai = {s: [gi[f"upper_arm.{s}"], gi[f"forearm.{s}"], gi[f"hand.{s}"]] for s in "LR"}
    semua_lengan = rantai["L"] + rantai["R"]
    seg = {}
    for s in "LR":
        ua, fa = bones[f"upper_arm.{s}"], bones[f"forearm.{s}"]
        seg[s] = (ua.head_local.copy(), ua.tail_local.copy(), fa.tail_local.copy())
    z_kepala = bones["head"].head_local.z - 0.02
    z_leher = bones["neck"].head_local.z + 0.02

    hapus, sampel = [], {s: ([], []) for s in "LR"}
    pindah = 0
    for v in me.vertices:
        if v.index in buatan:
            continue
        w = {g.group: g.weight for g in v.groups}
        wl = sum(w.get(i, 0) for i in semua_lengan)
        if wl >= 0.5:
            hapus.append(v.index)
            s = max("LR", key=lambda x: sum(w.get(i, 0) for i in rantai[x]))
            S, E, W = seg[s]
            if w.get(rantai[s][0], 0) >= w.get(rantai[s][1], 0):
                sampel[s][0].append(_jarak(v.co, S, E))
            else:
                sampel[s][1].append(_jarak(v.co, E, W))
        elif wl > 0:
            tujuan = "head" if v.co.z > z_kepala else ("neck" if v.co.z > z_leher else "chest")
            for i in semua_lengan:
                if i in w:
                    obj.vertex_groups[nama_g[i]].remove([v.index])
            obj.vertex_groups[tujuan].add([v.index], w.get(gi[tujuan], 0) + wl, "REPLACE")
            pindah += 1

    # sisa lengan baju asli yang menggembung (bobotnya badan) menggantung di tempat lengan dulu: buang titik di
    # sisi luar lengan atas (menjauh dari badan); sisi dalam yang menempel badan dibiarkan
    sudah = set(hapus)
    g_kepala = gi["head"]
    luar = 0
    for v in me.vertices:
        if v.index in buatan or v.index in sudah:
            continue
        if any(g.group == g_kepala and g.weight > 0.5 for g in v.groups):
            continue
        for s, arah_x in (("L", 1), ("R", -1)):
            S, E, _ = seg[s]
            ab = E - S
            t = min(1.0, max(0.0, (v.co - S).dot(ab) / ab.length_squared))
            off = v.co - (S + ab * t)
            if off.length < 0.075 and off.x * arah_x > 0.006 and t > 0.08:
                hapus.append(v.index)
                luar += 1
                break
    print(f"sisa lengan baju di sisi luar dibuang: {luar} titik")

    # ukuran lengan dari titik lengan asli (persentil bawah: renda baju yang ikut terbobot lengan lebih jauh)
    ukuran = {}
    for s in "LR":
        ra = float(np.percentile(sampel[s][0], 45)) if len(sampel[s][0]) > 50 else 0.03
        rf = float(np.percentile(sampel[s][1], 45)) if len(sampel[s][1]) > 50 else 0.024
        # badan model ramping: lengan baju dibatasi supaya tidak tampak gemuk
        ukuran[s] = (min(max(ra, 0.02), 0.025), min(max(rf, 0.016), 0.019))
    print(f"lengan asli dibuang: {len(hapus)} titik, bobot lengan dipindah ke badan: {pindah} titik, "
          f"jari-jari (atas, bawah) L {ukuran['L'][0]:.3f},{ukuran['L'][1]:.3f} R {ukuran['R'][0]:.3f},{ukuran['R'][1]:.3f}")

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in hapus], context="VERTS")
    # serpihan lepas kecil (bukan tangan buatan)
    dilihat, sisa = set(), []
    for v0 in bm.verts:
        if v0 in dilihat:
            continue
        pulau, tumpuk = [], [v0]
        dilihat.add(v0)
        while tumpuk:
            v = tumpuk.pop()
            pulau.append(v)
            for e in v.link_edges:
                o = e.other_vert(v)
                if o not in dilihat:
                    dilihat.add(o)
                    tumpuk.append(o)
        if len(pulau) < 150 and not any(f.material_index in buatan_mat for v in pulau for f in v.link_faces):
            sisa += pulau
    bmesh.ops.delete(bm, geom=sisa, context="VERTS")
    # tambal lubang bekas lengan di badan (hanya tepi dekat lengan; tepi lain seperti ujung rok dibiarkan)
    dekat = [e for e in bm.edges if e.is_boundary and all(
        min(_jarak(v.co, a, b) for S, E, W in seg.values() for a, b in ((S, E), (E, W))) < 0.09 for v in e.verts)]
    hasil = bmesh.ops.holes_fill(bm, edges=dekat, sides=0)
    tambal = bmesh.ops.triangulate(bm, faces=hasil["faces"])["faces"]
    me.materials.append(material("renda_badan", (0.93, 0.9, 0.86), 0.0, 0.7))
    for f in tambal:
        f.material_index = len(me.materials) - 1
        f.smooth = True
    bm.to_mesh(me)
    bm.free()
    print(f"serpihan dibuang: {len(sisa)} titik, tambalan badan: {len(tambal)} segitiga")

    # lengan baru
    mat_renda = material("renda_lengan", (0.95, 0.93, 0.9), 0.0, 0.75)
    mat_gelang = material("emas_gelang", (0.86, 0.62, 0.22), 1.0, 0.28)
    for s in "LR":
        S, E, W = seg[s]
        ra, rf = ukuran[s]
        da, df = (E - S).normalized(), (W - E).normalized()
        La, Lf = (E - S).length, (W - E).length
        bm = bmesh.new()
        dl = bm.verts.layers.deform.verify()
        uv = bm.loops.layers.uv.new("UVMap")
        g_bahu, g_atas, g_bawah = gi[f"shoulder.{s}"], gi[f"upper_arm.{s}"], gi[f"forearm.{s}"]
        titik, jari2, bobot = [], [], []

        def tambah(p, r, w_bawah, w_bahu=0.0):
            titik.append(p)
            jari2.append(r)
            b = {g_atas: max(0.0, 1 - w_bawah - w_bahu)}
            if w_bawah:
                b[g_bawah] = w_bawah
            if w_bahu:
                b[g_bahu] = w_bahu
            bobot.append(b)

        tambah(S - da * 1.3 * ra, ra * 0.85, 0, 0.35)      # pangkal di dalam bahu (sambungan tersembunyi)
        tambah(S, ra * 1.08, 0, 0.2)
        for k in (0.2, 0.4, 0.6, 0.78):
            tambah(S + da * La * k, ra * (1.05 - 0.1 * k), 0)
        tambah(E - da * ra * 0.9, ra * 0.95, 0.15)
        tambah(E + (df - da) * ra * 0.15, ra * 0.95, 0.5)   # siku: setengah-setengah supaya menekuk halus
        tambah(E + df * ra * 0.9, (ra + rf) / 2, 0.85)
        # ujung lengan baju mengecil sampai pas pergelangan (tertutup gelang kano), tidak menganga seperti corong
        for k in (0.3, 0.55, 0.78, 0.92, 1.0):
            tambah(E + df * Lf * k, rf * (1 - 0.3 * k), 1.0)
        _tabung_uv(bm, uv, titik, jari2, bobot, dl, 12)
        # kerut renda di siku: corong pendek melebar, ikut lengan bawah
        corong = [E + df * ra * 0.3, E + df * (ra * 0.3 + 0.018), E + df * (ra * 0.3 + 0.03)]
        _tabung_uv(bm, uv, corong, [ra * 0.98, ra * 1.18, ra * 1.32],
                   [{g_bawah: 0.8, g_atas: 0.2}, {g_bawah: 1.0}, {g_bawah: 1.0}], dl, 20)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me_l = bpy.data.meshes.new(f"lengan_{s}")
        bm.to_mesh(me_l)
        bm.free()
        # gelang burung (emas) di lengan atas
        bg = bmesh.new()
        _cincin(bg, S + da * La * 0.5, da, ra * 1.06, ra * 0.16)
        me_g = bpy.data.meshes.new(f"gelang_atas_{s}")
        bg.to_mesh(me_g)
        bg.free()
        for me_x, mat, kelompok in ((me_l, mat_renda, None), (me_g, mat_gelang, g_atas)):
            o = bpy.data.objects.new(me_x.name, me_x)
            bpy.context.scene.collection.objects.link(o)
            me_x.materials.append(mat)
            for p in me_x.polygons:
                p.use_smooth = True
            for g in obj.vertex_groups:  # urutan sama dengan badan: indeks bobot di bmesh = indeks badan
                o.vertex_groups.new(name=g.name)
            if kelompok is not None:
                o.vertex_groups[f"upper_arm.{s}"].add(list(range(len(me_x.vertices))), 1.0, "REPLACE")
            bpy.ops.object.select_all(action="DESELECT")
            o.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.join()
    print("lengan buatan terpasang")
