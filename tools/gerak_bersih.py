"""Susun ulang gerak mocap menjadi gerak tari yang bersih dan wajar.

Pakai: .venv/bin/python -I tools/gerak_bersih.py work/mocap/gerak.npz work/mocap/gerak-bersih.npz

Data MediaPipe dari video kelompok (resolusi kecil, kaki tertutup kain tapis) berisik: badan miring-miring,
kaki melayang, penari naik-turun sendiri, tangan patah-patah. Di sini tiap bagian tubuh disusun ulang:

- Kaki: pose bersih. Berdiri tegak (telapak rata di lantai) atau duduk bersimpuh (lutut & punggung kaki di
  lantai), peralihan halus. Penari tidak lagi naik-turun sendiri.
- Panggul: tetap di tengah, hanya menoleh sedikit (dibatasi), tinggi mengikuti berdiri/duduk.
- Badan: arah badan dari video dihaluskan dan dibatasi (miring samping maks 8 derajat); membungkuk saat
  menaruh/mengambil tepak.
- Tangan: dari video, dihaluskan, lengan atas tidak diangkat terlalu tinggi (lengan baju tidak melar).
  Saat membawa tepak kedua tangan dikunci di depan badan (IK), lalu turun ke lantai saat menaruh.
- Urutan (sesuai video): berdiri membawa tepak -> turun bersimpuh -> menaruh tepak (tangan di lantai
  17,2-18,9 dtk) -> menari bersimpuh -> 3 detik terakhir mengambil tepak lagi lalu berdiri, sehingga
  ulangan tarian tersambung mulus ke awal.
Ukuran tubuh dan tinggi pinggul berdiri sama dengan data asal, jadi skala di rig_animasi.py tidak berubah.
"""
import sys

import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rapikan_pose import haluskan  # noqa: E402

# jadwal (detik); AKHIR dihitung dari panjang data
DUDUK_MULAI, DUDUK_SELESAI = 15.6, 17.2      # turun ke bersimpuh sambil memegang tepak
TARUH_MULAI, TARUH_SAMPAI = 17.2, 18.4       # tangan membawa tepak ke lantai
LEPAS, LEPAS_SELESAI = 18.9, 19.8            # tepak dilepas, tangan kembali ke gerak video
AMBIL_MULAI, AMBIL_SAMPAI = 3.0, 2.3         # detik sebelum akhir: tangan menuju tepak di lantai
ANGKAT_MULAI, ANGKAT_SAMPAI = 2.0, 1.1       # detik sebelum akhir: tepak diangkat ke depan badan
BERDIRI_MULAI = 1.5                          # detik sebelum akhir: bangkit berdiri

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HP, R_HP = 23, 24


def langkah(t, a, b):
    """0 sebelum a, 1 sesudah b, smoothstep di antaranya."""
    k = np.clip((t - a) / (b - a), 0, 1)
    return k * k * (3 - 2 * k)


def rot_z(v, sudut):
    c, s = np.cos(sudut), np.sin(sudut)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def satuan(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def ik(bahu, target, a, b, kutub):
    """Siku untuk lengan dua ruas (panjang a, b) dari bahu ke target, menekuk ke arah `kutub`."""
    d_vec = target - bahu
    d = min(np.linalg.norm(d_vec), (a + b) * 0.995)
    u = d_vec / (np.linalg.norm(d_vec) + 1e-9)
    target = bahu + u * d
    x = (a * a - b * b + d * d) / (2 * d)
    h = np.sqrt(max(a * a - x * x, 0))
    v = satuan(kutub - u * kutub.dot(u))
    return bahu + u * x + v * h, target


def main(src, out):
    d = np.load(src)
    P = d["pos"].astype(float)
    fps = float(d["fps"])
    n = len(P)
    t = np.arange(n) / fps
    akhir = t[-1]
    atas = np.array([0, 0, 1.0])

    def ukur(a, b):
        return float(np.median(np.linalg.norm(P[:, a] - P[:, b], axis=1)))

    bahu_w, pinggul_w = ukur(L_SH, R_SH), ukur(L_HP, R_HP)
    lengan_a = (ukur(L_SH, L_EL) + ukur(R_SH, R_EL)) / 2
    lengan_b = (ukur(L_EL, L_WR) + ukur(R_EL, R_WR)) / 2
    paha = (ukur(23, 25) + ukur(24, 26)) / 2
    betis = (ukur(25, 27) + ukur(26, 28)) / 2
    hipc = (P[:, L_HP] + P[:, R_HP]) / 2
    shc = (P[:, L_SH] + P[:, R_SH]) / 2
    badan = float(np.median(np.linalg.norm(shc - hipc, axis=1)))
    h_berdiri = float(np.percentile(hipc[:, 2], 95))
    h_lutut = 0.03                                   # tinggi sendi lutut di lantai saat bersimpuh
    h_duduk = h_lutut + 0.6 * paha
    tinggi_pergelangan = 0.07
    print(f"badan {badan:.2f} paha {paha:.2f} betis {betis:.2f} lengan {lengan_a:.2f}/{lengan_b:.2f} "
          f"pinggul berdiri {h_berdiri:.2f} duduk {h_duduk:.2f}")

    # --- bobot waktu ---
    duduk = langkah(t, DUDUK_MULAI, DUDUK_SELESAI) * (1 - langkah(t, akhir - BERDIRI_MULAI, akhir))
    ambil = langkah(t, akhir - AMBIL_MULAI, akhir - AMBIL_SAMPAI)
    # pegang: 1 = lengan mengikuti target tepak, 0 = lengan dari video
    pegang = np.maximum(1 - langkah(t, LEPAS, LEPAS_SELESAI), ambil)
    # 0 = tangan di depan badan, 1 = tangan di lantai (menaruh lalu, di akhir, mengambil tepak)
    lantai_k = np.where(t < LEPAS + 1, langkah(t, TARUH_MULAI, TARUH_SAMPAI),
                        ambil * (1 - langkah(t, akhir - ANGKAT_MULAI, akhir - ANGKAT_SAMPAI)))
    # ujung tarian kembali ke nilai awal supaya ulangan tersambung
    ke_awal = langkah(t, akhir - 1.5, akhir)

    # --- arah hadap & badan dari video, dihaluskan dan dibatasi ---
    def yaw(a, b):
        v = P[:, a] - P[:, b]
        return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))

    y_hip, y_sh = yaw(L_HP, R_HP), yaw(L_SH, R_SH)
    pusat = np.median(y_hip)
    y_hip = haluskan((y_hip - pusat)[:, None], fps * 0.5)[:, 0]
    putar_dada = haluskan((y_sh - pusat)[:, None], fps * 0.35)[:, 0] - y_hip   # puntiran dada terhadap panggul
    batas_hadap = np.where(pegang > 0.5, np.radians(6), np.radians(22))
    y_hip = np.clip(y_hip, -batas_hadap, batas_hadap)
    putar_dada = np.clip(putar_dada, -np.radians(15), np.radians(15)) * (1 - 0.6 * pegang)

    # arah badan (pinggul -> bahu) dalam ruang lokal penari
    v_badan = np.stack([rot_z(v, -y) for v, y in zip(shc - hipc, y_hip)])
    v_badan = haluskan(v_badan, fps * 0.3)
    maju_lokal, lat_lokal = np.array([0, -1.0, 0]), np.array([1.0, 0, 0])
    bungkuk = np.degrees(np.arctan2(v_badan @ maju_lokal, v_badan[:, 2]))
    miring = np.degrees(np.arctan2(v_badan @ lat_lokal, v_badan[:, 2]))
    bungkuk = np.clip(bungkuk, -4, 40)
    miring = np.clip(miring, -8, 8)
    tenang = np.clip(pegang + (1 - duduk) * 0.5, 0, 1)    # saat membawa tepak badan lebih tenang
    bungkuk = bungkuk * (1 - tenang) + np.clip(bungkuk, -2, 6) * tenang
    miring = miring * (1 - 0.7 * tenang)
    bungkuk = np.maximum(bungkuk, 38 * lantai_k)          # membungkuk saat tangan ke lantai
    # napas: goyang badan sangat kecil supaya tidak kaku saat berdiri diam
    bungkuk = bungkuk + 1.2 * np.sin(2 * np.pi * t / 3.2)
    for arr in (y_hip, putar_dada, bungkuk, miring):
        arr[:] = arr * (1 - ke_awal) + arr[0] * ke_awal

    # --- arah lengan dari video (lokal), dihaluskan ---
    def arah_lokal(a, b):
        v = np.stack([rot_z(x, -y) for x, y in zip(P[:, b] - P[:, a], y_hip)])
        return satuan(haluskan(satuan(v), fps * 0.15))

    lengan_video = {}
    for sisi, (sh, el, wr, pk, ix) in (("L", (11, 13, 15, 17, 19)), ("R", (12, 14, 16, 18, 20))):
        ua = arah_lokal(sh, el)
        # lengan atas tidak diangkat lebih dari 15 derajat di atas datar (lengan baju renda tidak melar)
        maks = np.sin(np.radians(15))
        lebih = ua[:, 2] > maks
        dat = satuan(ua[:, :2])
        ua[lebih] = np.concatenate([dat[lebih] * np.cos(np.radians(15)), np.full((lebih.sum(), 1), maks)], axis=1)
        fa = arah_lokal(el, wr)
        tg = np.stack([rot_z(x, -y) for x, y in zip((P[:, pk] + P[:, ix]) / 2 - P[:, wr], y_hip)])
        tg = satuan(haluskan(satuan(tg), fps * 0.2))
        lengan_video[sisi] = (ua, fa, tg)

    Q = np.zeros_like(P)
    for f in range(n):
        yw = y_hip[f]
        lat = rot_z(lat_lokal, yw)
        maju = rot_z(maju_lokal, yw)
        h = h_berdiri * (1 - duduk[f]) + h_duduk * duduk[f]
        hip = np.array([0, 0, h])
        Q[f, L_HP] = hip + lat * pinggul_w / 2
        Q[f, R_HP] = hip - lat * pinggul_w / 2

        # kaki: berdiri vs bersimpuh
        for s, (hp, kn, an, he, ft) in ((1, (23, 25, 27, 29, 31)), (-1, (24, 26, 28, 30, 32))):
            ph = Q[f, hp]
            # berdiri: telapak di bawah pinggul, lutut sedikit menekuk ke depan bila kaki kepanjangan
            ang_b = np.array([ph[0], ph[1], tinggi_pergelangan]) - lat * s * 0.01
            lutut_b, ang_b = ik(ph, ang_b, paha, betis, maju)
            tumit_b = ang_b - maju * 0.05 + np.array([0, 0, 0.02 - tinggi_pergelangan])
            jari_b = ang_b + maju * 0.14 + lat * s * 0.025 + np.array([0, 0, 0.012 - tinggi_pergelangan])
            # bersimpuh: lutut di depan menyentuh lantai, betis terlipat ke belakang, punggung kaki di lantai
            lutut_d = np.array([*(ph + paha * 0.8 * maju + lat * s * 0.015)[:2], h_lutut])
            ang_d = lutut_d + betis * satuan(-maju + np.array([0, 0, -0.03])) - lat * s * 0.01
            tumit_d = ang_d + 0.06 * satuan(-0.5 * maju + 0.85 * atas)
            jari_d = ang_d + 0.13 * satuan(-0.95 * maju - 0.25 * atas)
            w = duduk[f]
            for i, b_, d_ in ((kn, lutut_b, lutut_d), (an, ang_b, ang_d), (he, tumit_b, tumit_d), (ft, jari_b, jari_d)):
                Q[f, i] = (1 - w) * b_ + w * d_

        # badan
        bk, mr = np.radians(bungkuk[f]), np.radians(miring[f])
        arah = satuan(np.array([np.sin(mr), -np.sin(bk), np.cos(bk) * np.cos(mr)]))
        arah = rot_z(arah, yw)
        sh = hip + arah * badan
        lat_dada = rot_z(lat_lokal, yw + putar_dada[f])
        lat_dada = satuan(lat_dada - arah * lat_dada.dot(arah))
        maju_dada = np.cross(lat_dada, arah)
        Q[f, L_SH] = sh + lat_dada * bahu_w / 2
        Q[f, R_SH] = sh - lat_dada * bahu_w / 2
        # kepala (rig memakai arah dada untuk kepala; titik ini hanya pelengkap)
        telinga = sh + arah * 0.2
        Q[f, 7], Q[f, 8] = telinga + lat_dada * 0.07, telinga - lat_dada * 0.07
        Q[f, 0] = telinga + maju_dada * 0.1
        for i, k in ((1, 0.03), (2, 0.035), (3, 0.04), (4, -0.03), (5, -0.035), (6, -0.04)):
            Q[f, i] = Q[f, 0] + lat_dada * k + arah * 0.03
        Q[f, 9], Q[f, 10] = Q[f, 0] - arah * 0.06 + lat_dada * 0.025, Q[f, 0] - arah * 0.06 - lat_dada * 0.025

        # lengan
        # tepak setinggi ulu hati (bunga di puncak tepak sejajar dagu, wajah tidak tertutup)
        tinggi_tepak = h + 0.16 + 0.015 * np.sin(2 * np.pi * t[f] / 3.2)
        for sisi, s, (shi, eli, wri, pki, ixi, thi) in (("L", 1, (11, 13, 15, 17, 19, 21)), ("R", -1, (12, 14, 16, 18, 20, 22))):
            bahu = Q[f, shi]
            ua, fa, tg = (rot_z(x[f], yw) for x in lengan_video[sisi])
            # target tangan saat membawa tepak (depan badan) dan saat di lantai (depan lutut)
            pegang_pos = hip + maju * 0.3 + lat * s * 0.19 + np.array([0, 0, tinggi_tepak - h])
            lantai_pos = np.array([0, 0, 0]) + maju * 0.5 + lat * s * 0.17 + np.array([0, 0, 0.13])
            target = (1 - lantai_k[f]) * pegang_pos + lantai_k[f] * lantai_pos
            kutub = lat * s * 0.6 - atas * 0.8 - maju * 0.2
            siku_o, wr_o = ik(bahu, target, lengan_a, lengan_b, kutub)
            ua_o, fa_o = satuan(siku_o - bahu), satuan(wr_o - siku_o)
            tg_o = satuan(maju * 0.8 - lat * s * 0.55 - atas * 0.15)
            w = pegang[f]
            ua_ = satuan((1 - w) * ua + w * ua_o)
            fa_ = satuan((1 - w) * fa + w * fa_o)
            tg_ = satuan((1 - w) * tg + w * tg_o)
            siku = bahu + ua_ * lengan_a
            wr = siku + fa_ * lengan_b
            Q[f, eli], Q[f, wri] = siku, wr
            Q[f, pki] = wr + tg_ * 0.08 - lat_dada * s * 0.012
            Q[f, ixi] = wr + tg_ * 0.08 + lat_dada * s * 0.012
            Q[f, thi] = wr + tg_ * 0.05 + atas * 0.02

    Q = haluskan(Q, 1.5)
    np.savez(out, pos=Q, fps=fps)
    hp = (Q[:, L_HP, 2] + Q[:, R_HP, 2]) / 2
    kaki = Q[:, [27, 28, 29, 30, 31, 32], 2].min(axis=1)
    print(f"{n} frame, pinggul {hp.min():.2f}-{hp.max():.2f}, titik kaki terendah {kaki.min():.3f}..{kaki.max():.3f}")
    print(f"selisih frame awal-akhir (ulang): {np.abs(Q[0] - Q[-1]).max():.3f} m")


if __name__ == "__main__":
    main(*sys.argv[1:])
