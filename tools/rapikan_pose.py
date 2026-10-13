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


def main(src, out, awal=None, akhir=None):
    d = np.load(src)
    fps = float(d["fps"])
    w = d["world"]
    a = int(float(awal) * fps) if awal else 0
    b = int(float(akhir) * fps) if akhir else len(w)
    w = isi_kosong(w[a:b])

    # haluskan getaran (sigma ~0.07 dtk)
    w = haluskan(w, fps * 0.07)

    blender = np.stack([w[..., 0], w[..., 2], -w[..., 1]], axis=-1)

    # pinggul tetap di tengah poster (x,y), tinggi diatur supaya titik terendah menyentuh lantai
    terendah = blender[:, TITIK_LANTAI, 2].min(axis=1)
    terendah = haluskan(terendah, fps * 0.2)
    blender[..., 2] -= terendah[:, None]
    np.savez(out, pos=blender, fps=fps)
    tinggi_pinggul = (blender[:, 23, 2] + blender[:, 24, 2]) / 2
    print(f"{len(blender)} frame ({len(blender)/fps:.1f} dtk), tinggi pinggul {tinggi_pinggul.min():.2f}-{tinggi_pinggul.max():.2f} m")


if __name__ == "__main__":
    main(*sys.argv[1:])
