"""Ganti tangan hasil generator (jari menyatu, kuku tanggai lebur) dengan tangan buatan:
telapak, 5 jari lentik, dan kuku tanggai emas melengkung di 8 jari.

Dipanggil dari rig_animasi.py setelah bobot kulit terpasang. Semua titik tangan baru 100% ikut tulang
hand.L / hand.R sehingga tidak melar.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

N_KELILING = 12
MUNDUR = 0.4  # lengan dipotong sekian kali panjang telapak sebelum ujung lengan bawah asli


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


def _cincin(bm, pusat, sumbu, R, r, n=28, m=8):
    """Torus (gelang) mengelilingi `sumbu` di `pusat`: jari-jari besar R, tebal r."""
    sumbu = sumbu.normalized()
    u = sumbu.orthogonal().normalized()
    v = sumbu.cross(u)
    ring = []
    for i in range(n):
        a = 2 * math.pi * i / n
        arah = u * math.cos(a) + v * math.sin(a)
        ring.append([bm.verts.new(pusat + arah * (R + r * math.cos(b)) + sumbu * (r * math.sin(b)))
                     for b in (2 * math.pi * j / m for j in range(m))])
    for i in range(n):
        for j in range(m):
            bm.faces.new((ring[i][j], ring[(i + 1) % n][j], ring[(i + 1) % n][(j + 1) % m], ring[i][(j + 1) % m]))


def bangun_tangan(Lp, kanan_ke_telunjuk, jari_wrist):
    """Mesh tangan di ruang lokal: Y = arah jari, X = sisi telunjuk, Z = punggung tangan.
    Sikap tangan tari Sigeh: jari rapat dan lentik ke belakang, ibu jari sedikit dilipat ke telapak,
    kuku tanggai emas ramping melengkung (gelang kano dipasang terpisah di ganti_tangan, ikut lengan bawah).
    Kembalikan (bmesh kulit, bmesh emas)."""
    X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    kulit, emas = bmesh.new(), bmesh.new()
    lebar = 0.80 * Lp
    tebal = 0.34 * Lp
    # telapak: dari pergelangan (bulat) melebar pipih ke pangkal jari, sedikit melengkung ke punggung
    pts = [Y * (-0.12 * Lp), Y * (0.15 * Lp), Y * (0.45 * Lp), Y * (0.75 * Lp) + Z * (0.03 * Lp), Y * (0.98 * Lp) + Z * (0.05 * Lp)]
    r = [jari_wrist * 0.95, jari_wrist * 0.9 + lebar * 0.1, lebar * 0.48, lebar * 0.52, lebar * 0.47]
    _tabung(kulit, pts, r, X, Z, skala_z=tebal / lebar)
    # 4 jari rapat (telunjuk sisi +X .. kelingking), sejajar dengan sedikit kipas, lentik ke punggung tangan
    panjang = [0.70, 0.78, 0.73, 0.56]
    for i, pj in enumerate(panjang):
        u = 0.30 - 0.20 * i                        # posisi melintang di ujung telapak
        pangkal = Y * (0.92 * Lp) + X * (u * lebar) + Z * (0.05 * Lp)
        arah = (Y + X * (u * 0.12)).normalized()    # nyaris sejajar: jari rapat, bukan mekar seperti garpu
        L = pj * Lp
        ruas = _kurva(pangkal, arah, X, L, 5, -32)  # lentik ke arah punggung tangan
        rj = [0.105 * Lp, 0.1 * Lp, 0.093 * Lp, 0.085 * Lp, 0.077 * Lp, 0.065 * Lp]
        _tabung(kulit, ruas, rj, X, Z, skala_z=0.82)
        # kuku tanggai: kerucut emas ramping dari ruas terakhir, melanjutkan lengkung jari
        d_ujung = (ruas[-1] - ruas[-2]).normalized()
        n = 10
        tg = _kurva(ruas[-3].lerp(ruas[-2], 0.5), d_ujung, X, 1.05 * Lp, n, -38)
        rt = [0.085 * Lp * (1 - k / n) ** 1.1 + 0.004 * Lp for k in range(n + 1)]
        _tabung(emas, tg, rt, X, Z, skala_z=0.85)
    # ibu jari: dari sisi telunjuk dekat pergelangan, dilipat ke depan telapak (bukan mencuat ke samping)
    pangkal = Y * (0.22 * Lp) + X * (0.40 * lebar) - Z * (0.04 * Lp)
    arah = (Y * 0.85 + X * 0.30 - Z * 0.35).normalized()
    ruas = _kurva(pangkal, arah, Z, 0.55 * Lp, 4, 16)
    _tabung(kulit, ruas, [0.12 * Lp, 0.11 * Lp, 0.098 * Lp, 0.088 * Lp, 0.075 * Lp], X, Z, skala_z=0.85)
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
    pergelangan = {}
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

        # buang tangan lama. Sendi pergelangan dari foto (W) bisa meleset beberapa cm, jadi patokannya ujung lengan
        # bawah yang sebenarnya: titik lengan bawah dekat sumbu yang paling jauh searah lengan. Lengan dipotong rata
        # sedikit sebelum ujung itu (gelang asli yang kasar ikut terbuang; gelang kano emas baru menutupnya).
        gi = {g.name: g.index for g in obj.vertex_groups}
        hid, fid = gi[f"hand.{sisi}"], gi[f"forearm.{sisi}"]
        milik = {hid, fid}
        data = []
        for v in obj.data.vertices:
            rel = v.co - W
            if rel.length > 4.0 * Lp:
                continue
            wh = sum(g.weight for g in v.groups if g.group == hid)
            wf = sum(g.weight for g in v.groups if g.group == fid)
            if wh + wf < 0.5:
                continue
            maju = rel.dot(arah_lengan)
            data.append((v.index, maju, wh, wf, (rel - arah_lengan * maju).length))
        ujung = max((m for _, m, wh, wf, r in data if wf >= wh and r < 1.2 * Lp), default=0.0)
        potong = ujung - MUNDUR * Lp
        hapus = [i for i, m, wh, wf, r in data if wh >= 0.4 or m > potong]
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in hapus], context="VERTS")
        # pergelangan = tepi bolong bekas potongan rata itu (hanya tepi milik lengan ini di dekat bidang potong,
        # bukan tepi bekas potongan siger/badan yang tersentuh tangan, lihat putus_jembatan di rig_animasi.py)
        dl = bm.verts.layers.deform.active
        tepi = {v for e in bm.edges if e.is_boundary for v in e.verts
                if (v.co - W).length < 4.0 * Lp and (v.co - W).dot(arah_lengan) > potong - 0.25 * Lp
                and (dl is None or sum(w for g, w in v[dl].items() if g in milik) >= 0.5)}
        if len(tepi) >= 6:
            C = sum((v.co for v in tepi), Vector()) / len(tepi)
            jari_wrist = sum(((v.co - C) - arah_lengan * (v.co - C).dot(arah_lengan)).length for v in tepi) / len(tepi)
        else:
            C = W + arah_lengan * potong
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
        pergelangan[sisi] = C.copy()
        print(f"tangan {sisi}: ujung lengan {ujung:+.3f}, hapus {len(hapus)} titik, telapak {Lp:.3f}, jari2 pergelangan {jari_wrist:.3f}, tepi {len(tepi)} titik")

        kulit, emas = bangun_tangan(Lp, X, jari_wrist)
        M = Matrix((X, Y, Z)).transposed().to_4x4()
        M.translation = W
        # gelang kano: dua gelang emas memeluk pergelangan, searah lengan bawah dan ikut tulangnya
        # (kalau ikut tulang tangan, gelang berayun keluar dari lengan saat pergelangan menekuk)
        gelang = bmesh.new()
        _cincin(gelang, C - arah_lengan * (0.02 * Lp), arah_lengan, jari_wrist + 0.05 * Lp, 0.07 * Lp)
        _cincin(gelang, C - arah_lengan * (0.22 * Lp), arah_lengan, jari_wrist * 1.05 + 0.05 * Lp, 0.055 * Lp)
        for nama, bmx, mat, tulang, ubah in (("kulit", kulit, mat_kulit, f"hand.{sisi}", True),
                                             ("emas", emas, mat_emas, f"hand.{sisi}", True),
                                             ("gelang", gelang, mat_emas, f"forearm.{sisi}", False)):
            if ubah:  # tangan dibangun di ruang lokal; gelang sudah di ruang model
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
            vg = o.vertex_groups.new(name=tulang)
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
    # tulang tangan berputar di sendi pergelangan hasil tebakan foto; geser sendi itu ke pusat potongan lengan
    # supaya tangan buatan dan gelang tetap menempel di ujung lengan saat pergelangan menekuk
    aktif = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for sisi, C in pergelangan.items():
        tangan, lengan = eb[f"hand.{sisi}"], eb[f"forearm.{sisi}"]
        arah = tangan.tail - tangan.head
        geser = (C - tangan.head).length
        lengan.tail = C
        tangan.head = C
        tangan.tail = C + arah
        print(f"sendi pergelangan {sisi} digeser {geser:.3f}")
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.select_set(False)
    bpy.context.view_layer.objects.active = aktif

