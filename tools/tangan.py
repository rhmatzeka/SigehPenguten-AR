"""Ganti tangan hasil generator (jari menyatu, kuku tanggai lebur) dengan tangan buatan:
telapak, 5 jari lentik, dan kuku tanggai emas melengkung di 8 jari.

Dipanggil dari rig_animasi.py setelah bobot kulit terpasang. Semua titik tangan baru 100% ikut tulang
hand.L / hand.R sehingga tidak melar.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

N_KELILING = 10


def _tabung(bm, titik, jari2, sumbu_x, sumbu_z, skala_z=1.0, tutup=True):
    """Tabung sepanjang polyline `titik` dengan jari-jari per titik; penampang elips (x, z*skala_z)."""
    cincin = []
    for i, (p, r) in enumerate(zip(titik, jari2)):
        if i < len(titik) - 1:
            arah = (titik[i + 1] - p).normalized()
        else:
            arah = (p - titik[i - 1]).normalized()
        x = (sumbu_x - arah * sumbu_x.dot(arah)).normalized()
        z = arah.cross(x).normalized()
        ring = []
        for k in range(N_KELILING):
            a = 2 * math.pi * k / N_KELILING
            ring.append(bm.verts.new(p + x * math.cos(a) * r + z * math.sin(a) * r * skala_z))
        cincin.append(ring)
    for a, b in zip(cincin, cincin[1:]):
        for k in range(N_KELILING):
            bm.faces.new((a[k], a[(k + 1) % N_KELILING], b[(k + 1) % N_KELILING], b[k]))
    if tutup:
        for ring, ujung in ((cincin[0], titik[0]), (cincin[-1], titik[-1])):
            c = bm.verts.new(ujung)
            for k in range(N_KELILING):
                f = (ring[k], ring[(k + 1) % N_KELILING], c)
                bm.faces.new(f if ring is cincin[-1] else f[::-1])
    return cincin


def _kurva(awal, arah, tekuk_sumbu, panjang, n, tekuk_total):
    """Titik-titik sepanjang garis yang melengkung perlahan (total `tekuk_total` derajat)."""
    pts = [awal.copy()]
    d = arah.normalized()
    langkah = panjang / n
    rot = Matrix.Rotation(math.radians(tekuk_total / n), 3, tekuk_sumbu)
    for _ in range(n):
        pts.append(pts[-1] + d * langkah)
        d = rot @ d
    return pts


def bangun_tangan(Lp, kanan_ke_telunjuk, jari_wrist):
    """Mesh tangan di ruang lokal: Y = arah jari, X = sisi telunjuk, Z = punggung tangan.
    Kembalikan (bmesh kulit, bmesh emas)."""
    X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    kulit, emas = bmesh.new(), bmesh.new()
    lebar = 0.85 * Lp
    tebal = 0.30 * Lp
    # telapak: tabung pipih dari pergelangan ke pangkal jari
    pts = [Y * (-0.05 * Lp), Y * (0.35 * Lp), Y * (0.75 * Lp), Y * Lp]
    r = [jari_wrist * 0.85, lebar * 0.5, lebar * 0.52, lebar * 0.45]
    _tabung(kulit, pts, r, X, Z, skala_z=tebal / lebar)
    # 4 jari: telunjuk (sisi +X) .. kelingking
    panjang = [0.78, 0.86, 0.8, 0.64]
    for i, pj in enumerate(panjang):
        u = 0.36 - 0.24 * i                       # posisi melintang di ujung telapak
        pangkal = Y * (0.95 * Lp) + X * (u * lebar)
        arah = (Y + X * (u * 0.35)).normalized()   # sedikit mekar
        L = pj * Lp
        ruas = _kurva(pangkal, arah, X, L, 4, -18)  # lentik ke arah punggung tangan
        rj = [0.125 * Lp, 0.118 * Lp, 0.108 * Lp, 0.098 * Lp, 0.09 * Lp]
        _tabung(kulit, ruas, rj, X, Z)
        # kuku tanggai: kerucut emas panjang dari ujung jari, melengkung keluar
        d_ujung = (ruas[-1] - ruas[-2]).normalized()
        tg = _kurva(ruas[-2].lerp(ruas[-1], 0.4), d_ujung, X, 1.15 * Lp, 8, -34)
        rt = [0.1 * Lp * (1 - k / 8) ** 0.9 + 0.006 * Lp for k in range(9)]
        _tabung(emas, tg, rt, X, Z)
    # ibu jari: dari sisi telunjuk dekat pergelangan, mengarah keluar
    pangkal = Y * (0.25 * Lp) + X * (0.45 * lebar)
    arah = (Y * 0.8 + X * 0.6).normalized()
    ruas = _kurva(pangkal, arah, Z, 0.62 * Lp, 3, -10)
    _tabung(kulit, ruas, [0.13 * Lp, 0.11 * Lp, 0.095 * Lp, 0.08 * Lp], X, Z)
    # (gelang dihapus: manset emas lengan baju dari model sudah menutup sambungan)
    return kulit, emas


def material(nama, warna, logam, kasar):
    m = bpy.data.materials.new(nama)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*warna, 1)
    bsdf.inputs["Metallic"].default_value = logam
    bsdf.inputs["Roughness"].default_value = kasar
    return m


def ganti_tangan(obj, arm, J, IX, PK, warna_kulit):
    bones = arm.data.bones
    mat_kulit = material("kulit_tangan", warna_kulit, 0.0, 0.55)
    mat_emas = material("emas_tanggai", (0.86, 0.62, 0.22), 1.0, 0.28)
    for sisi, ix, pk in (("L", IX[0], PK[0]), ("R", IX[1], PK[1])):
        b = bones[f"hand.{sisi}"]
        W = b.head_local.copy()
        Y = (b.tail_local - b.head_local).normalized()
        # panjang telapak dari lengan bawah (titik jari di foto sering memendek karena menghadap kamera)
        lengan = [(bones[f"forearm.{x}"].tail_local - bones[f"forearm.{x}"].head_local).length for x in "LR"]
        Lp = 0.4 * sum(lengan) / 2
        X = (J[ix] - J[pk])
        X = (X - Y * X.dot(Y)).normalized()
        Z = X.cross(Y).normalized()
        fore = bones[f"forearm.{sisi}"]
        arah_lengan = (fore.tail_local - fore.head_local).normalized()

        # buang tangan lama: titik di depan pergelangan, dekat, dan memang berbobot lengan/tangan
        gi = {g.name: g.index for g in obj.vertex_groups}
        milik = {gi[f"hand.{sisi}"], gi[f"forearm.{sisi}"]}
        hapus, jari_cuff = [], []
        for v in obj.data.vertices:
            rel = v.co - W
            maju = rel.dot(arah_lengan)
            if rel.length > 4.0 * Lp:
                continue
            w_milik = sum(g.weight for g in v.groups if g.group in milik)
            if w_milik < 0.5:
                continue
            if maju > 0.004:
                hapus.append(v.index)
            elif maju > -0.02:
                jari_cuff.append((rel - arah_lengan * maju).length)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in hapus], context="VERTS")
        # pergelangan sebenarnya = tepi bolong bekas potongan di ujung lengan model
        tepi = {v for e in bm.edges if e.is_boundary for v in e.verts if (v.co - W).length < 3.0 * Lp}
        if len(tepi) >= 6:
            C = sum((v.co for v in tepi), Vector()) / len(tepi)
            jari_wrist = sum(((v.co - C) - arah_lengan * (v.co - C).dot(arah_lengan)).length for v in tepi) / len(tepi)
        else:
            C = W
            jari_wrist = 0.35 * Lp
        jari_wrist = min(max(jari_wrist, 0.22 * Lp), 0.55 * Lp)
        # serpihan lepas sisa kuku lama di sekitar tangan
        bm.verts.ensure_lookup_table()
        sisa, dilihat = [], set()
        for v0 in bm.verts:
            if v0 in dilihat or (v0.co - C).length > 5.0 * Lp:
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
            if len(pulau) < 300:
                sisa += pulau
        if sisa:
            bmesh.ops.delete(bm, geom=sisa, context="VERTS")
        bm.to_mesh(obj.data)
        bm.free()
        # arah tangan: kebanyakan mengikuti lengan bawah, sedikit dari titik jari foto
        Y = (arah_lengan * 0.75 + Y * 0.25).normalized()
        X = (X - Y * X.dot(Y)).normalized()
        Z = X.cross(Y).normalized()
        W = C
        print(f"tangan {sisi}: hapus {len(hapus)} titik, telapak {Lp:.3f}, jari2 pergelangan {jari_wrist:.3f}, tepi {len(tepi)} titik")

        kulit, emas = bangun_tangan(Lp, X, jari_wrist)
        M = Matrix((X, Y, Z)).transposed().to_4x4()
        M.translation = W
        for nama, bmx, mat in (("kulit", kulit, mat_kulit), ("emas", emas, mat_emas)):
            bmx.transform(M)
            bmesh.ops.recalc_face_normals(bmx, faces=bmx.faces)
            me = bpy.data.meshes.new(f"tangan_{nama}_{sisi}")
            bmx.to_mesh(me)
            bmx.free()
            o = bpy.data.objects.new(me.name, me)
            bpy.context.scene.collection.objects.link(o)
            me.materials.append(mat)
            for p in me.polygons:
                p.use_smooth = True
            vg = o.vertex_groups.new(name=f"hand.{sisi}")
            vg.add(list(range(len(me.vertices))), 1.0, "REPLACE")
            sub = o.modifiers.new("halus", "SUBSURF")
            sub.levels = 1
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.modifier_apply(modifier="halus")
            # gabung ke tubuh (UV kosong tidak masalah, material tangan tidak memakai tekstur)
            bpy.ops.object.select_all(action="DESELECT")
            o.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.join()
