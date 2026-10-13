"""Gabungkan beberapa gambar berdampingan. Pakai: python tools/gabung_gambar.py <keluaran> <gambar...>"""
import sys

from PIL import Image

gambar = [Image.open(p).convert("RGB") for p in sys.argv[2:]]
h = max(g.height for g in gambar)
hasil = Image.new("RGB", (sum(g.width for g in gambar), h), "white")
x = 0
for g in gambar:
    hasil.paste(g, (x, 0))
    x += g.width
hasil.save(sys.argv[1])
