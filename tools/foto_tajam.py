"""Buat versi tajam dari foto input generator: foto asli resolusi penuh di-warp ke bingkai foto input.

Generator (TRELLIS dsb.) memotong & mengecilkan foto ke ~518 px; untuk proyeksi tekstur kita pakai foto asli
yang disejajarkan ke bingkai yang sama tetapi FAKTOR kali lebih besar.

Pakai: .venv/bin/python -I tools/foto_tajam.py <input.png> <pose-input.json> <foto-asli.jpg> <pose-foto.json> \
           <keluaran.png> <keluaran.json> [faktor]
"""
import json
import sys

import cv2
import numpy as np

inp_png, inp_json, asli, asli_json, out_png, out_json = sys.argv[1:7]
F = int(sys.argv[7]) if len(sys.argv) > 7 else 4

pi = json.load(open(inp_json))
pa = json.load(open(asli_json))
li, la = np.array(pi["lm"]), np.array(pa["lm"])
ok = (li[:, 2] > 0.5) & (la[:, 2] > 0.5)
A, B = li[ok, :2], la[ok, :2]
# skala seragam + geser: asli = s * input + t (kuadrat terkecil)
ma, mb = A.mean(0), B.mean(0)
s = float(((A - ma) * (B - mb)).sum() / ((A - ma) ** 2).sum())
t = mb - s * ma
galat = np.linalg.norm(s * A + t - B, axis=1)
print(f"skala {s:.3f}, geser {t.round(1)}, galat median {np.median(galat):.1f}px dari {ok.sum()} titik")

img = cv2.imread(asli)
W, H = pi["w"] * F, pi["h"] * F
# piksel keluaran (x, y) -> input (x/F, y/F) -> asli (s*x/F + t)
M = np.array([[s / F, 0, t[0]], [0, s / F, t[1]]], np.float32)
hasil = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(out_png, hasil)
json.dump({"w": W, "h": H, "bbox": [v * F for v in pi["bbox"]], "lm": [[x * F, y * F, v] for x, y, v in pi["lm"]]},
          open(out_json, "w"))
print("tersimpan", out_png, (W, H))
