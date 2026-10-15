"""Rapikan pose MediaPipe: isi frame kosong, haluskan, ubah ke sumbu Blender, tempelkan ke lantai.

Pakai: .venv/bin/python -I tools/rapikan_pose.py <pose.npz> <keluaran.npz> [detik_awal] [detik_akhir]
Sumbu MediaPipe world: x kanan gambar, y ke bawah, z menjauh dari kamera.
Sumbu Blender: X kanan, Y menjauh dari penonton (model menghadap -Y), Z atas.
"""
import sys

import numpy as np

# indeks titik yang menyentuh lantai saat berdiri/berlutut: lutut, pergelangan kaki, tumit, ujung kaki
TITIK_LANTAI = [25, 26, 27, 28, 29, 30, 31, 32]


def haluskan(a, sigma):
    """Rata-rata bergerak Gaussian sepanjang waktu (sumbu 0), tepi dipantulkan."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([a[r:0:-1], a, a[-2:-r - 2:-1]], axis=0)
    flat = pad.reshape(len(pad), -1)
    hasil = np.stack([np.convolve(flat[:, i], k, mode="valid") for i in range(flat.shape[1])], axis=1)
    return hasil.reshape(a.shape)


def isi_kosong(a):
    """Interpolasi linear untuk frame tanpa deteksi (NaN)."""
    t = np.arange(len(a))
    ok = ~np.isnan(a[:, 0, 0])
    out = a.copy()
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            out[:, j, k] = np.interp(t, t[ok], a[ok, j, k])
    return out


def betulkan_hadap(P, fps):
    """MediaPipe kadang salah menebak kedalaman dan mengira penari membelakangi kamera (di video ia tetap
    menghadap depan), sehingga badan berputar ~180 derajat sesaat. Frame itu (plus sedikit tepi) diganti
    peralihan dari pose terakhir yang benar ke pose benar berikutnya."""
    def hadap(a, b):
        v = P[:, a] - P[:, b]
        return np.degrees(np.arctan2(v[:, 1], v[:, 0]))

    def selisih(sudut):
        return np.abs((sudut - np.median(sudut) + 180) % 360 - 180)

    salah = (selisih(hadap(23, 24)) > 75) | (selisih(hadap(11, 12)) > 75)
    tepi = int(fps * 0.3)  # tepi rentang salah sering masih goyah (kaki terdeteksi di atas pinggul)
    salah = np.convolve(salah, np.ones(2 * tepi + 1), mode="same") > 0
    if not salah.any():
        return P
    Q = P.copy()
    t = np.arange(len(P))
    baik = ~salah
    for j in range(P.shape[1]):
        for k in range(P.shape[2]):
            Q[salah, j, k] = np.interp(t[salah], t[baik], P[baik, j, k])
    awal = np.flatnonzero(np.diff(np.r_[0, salah.astype(int)]) == 1)
    akhir = np.flatnonzero(np.diff(np.r_[salah.astype(int), 0]) == -1)
    print("hadap terbalik dibetulkan:", ", ".join(f"{a / fps:.1f}-{b / fps:.1f} dtk" for a, b in zip(awal, akhir)))
    return Q


def kaki_simpuh(P, fps):
    """Saat penari duduk, ganti kaki hasil mocap (tertutup rok, sering tertekuk aneh) dengan pose duduk
    bersimpuh yang benar: lutut di depan, betis terlipat ke belakang di lantai, punggung kaki menempel lantai.
    Badan atas tetap dari video. Peralihan berdiri <-> duduk dibuat halus."""
    hip = (P[:, 23] + P[:, 24]) / 2
    tinggi = haluskan(hip[:, 2].copy(), fps * 0.5)
    # 0 = berdiri (pinggul >= 0.66 m), 1 = duduk (pinggul <= 0.5 m)
    w = np.clip((0.66 - tinggi) / 0.16, 0, 1)
    w = np.maximum.accumulate(w)                    # sekali duduk, tetap duduk (sesuai video)
    n_akhir = int(fps * 1.2)                        # akhir loop: berdiri lagi supaya nyambung ke awal
    w[-n_akhir:] *= np.linspace(1, 0, n_akhir)
    w = w * w * (3 - 2 * w)                         # smoothstep
    atas = np.array([0, 0, 1.0])
    Q = P.copy()
    for f in range(len(P)):
        x = P[f, 23] - P[f, 24]
        x[2] = 0
        x /= np.linalg.norm(x) + 1e-9
        maju = np.cross(x, atas)                    # arah hadap penari
        for sisi, (hp, kn, an, he, ft) in ((1, (23, 25, 27, 29, 31)), (-1, (24, 26, 28, 30, 32))):
            paha = np.linalg.norm(P[f, kn] - P[f, hp])
            betis = np.linalg.norm(P[f, an] - P[f, kn])
            lutut = P[f, hp] + paha * (0.8 * maju - 0.6 * atas) + x * sisi * 0.015
            pergelangan = lutut + betis * (-0.985 * maju - 0.17 * atas) - x * sisi * 0.01
            tumit = pergelangan + 0.06 * (-0.5 * maju + 0.85 * atas)
            jari = pergelangan + 0.13 * (-0.95 * maju - 0.3 * atas)
            for i, t in ((kn, lutut), (an, pergelangan), (he, tumit), (ft, jari)):
                Q[f, i] = (1 - w[f]) * P[f, i] + w[f] * t
    print(f"kaki simpuh: duduk mulai detik {np.argmax(w > 0.5) / fps:.1f}")
    return Q


def main(src, out, awal=None, akhir=None):
    d = np.load(src)
    fps = float(d["fps"])
    w = d["world"]
    a = int(float(awal) * fps) if awal else 0
    b = int(float(akhir) * fps) if akhir else len(w)
    w = isi_kosong(w[a:b])

    # haluskan getaran (sigma ~0.1 dtk)
    w = haluskan(w, fps * 0.1)

    blender = np.stack([w[..., 0], w[..., 2], -w[..., 1]], axis=-1)
    blender = betulkan_hadap(blender, fps)

    # pinggul tetap di tengah poster (x,y), tinggi diatur supaya titik terendah menyentuh lantai
    terendah = blender[:, TITIK_LANTAI, 2].min(axis=1)
    terendah = haluskan(terendah, fps * 0.2)
    blender[..., 2] -= terendah[:, None]
    blender = kaki_simpuh(blender, fps)
    np.savez(out, pos=blender, fps=fps)
    tinggi_pinggul = (blender[:, 23, 2] + blender[:, 24, 2]) / 2
    print(f"{len(blender)} frame ({len(blender)/fps:.1f} dtk), tinggi pinggul {tinggi_pinggul.min():.2f}-{tinggi_pinggul.max():.2f} m")


if __name__ == "__main__":
    main(*sys.argv[1:])
