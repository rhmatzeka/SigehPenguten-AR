"""Siger kecil untuk puncak tepak: ambil siger dari potongan foto penari, belahan kanan (tidak tertutup tangan)
dicerminkan ke kiri, dipotong tepat di bawah pita siger.

Pakai: .venv/bin/python -I tools/siger_tepak.py <penari.png dari potong_penari.py> assets/siger-tepak.webp <cek.png>
"""
import sys
import numpy as np
from PIL import Image
src, out, cek = sys.argv[1:4]
im = np.array(Image.open(src).convert("RGBA")).astype(float)
# cari sumbu tengah siger dari permata merah besar di tengah (sekitar x 258, y 150 pada potongan)
cx = 258
h = 228
reg = im[:h].copy()
r, g, b, a = [reg[..., i] for i in range(4)]
mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
v = mx / 255; s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
gelap = v < 0.32                                  # rambut di bawah pita
kulit = (r > g) & (g > b) & (s > 0.18) & (s < 0.5) & (v > 0.6) & ((r - g) > 25) & ((g - b) < 60)
bawah = np.zeros((h, 1), bool)  # tanpa filter warna: batas bawah ditentukan oleh h
alpha = np.where(bawah & (gelap | kulit), 0, a)
reg[..., 3] = alpha
# belahan kanan (tanpa tangan) dicerminkan ke kiri
kanan = reg[:, cx:]
lebar = min(kanan.shape[1], 150)
kanan = kanan[:, :lebar]
siger = np.concatenate([kanan[:, ::-1], kanan[:, 1:]], axis=1)
img = Image.fromarray(siger.astype(np.uint8), "RGBA")
bbox = img.getbbox(); img = img.crop(bbox)
# perhalus tepi alpha
img.save(out, quality=90, method=6)
print("ukuran", img.size, "bbox", bbox)
c = Image.new("RGB", img.size, (40, 120, 200)); c.paste(img, mask=img.split()[3]); c.resize((img.width * 3, img.height * 3)).save(cek)
