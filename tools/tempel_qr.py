"""Tempel kode QR ke poster (pojok kanan bawah, sejajar kotak Makna).

Pakai: .venv/bin/python -I tools/tempel_qr.py <poster.jpg> <keluaran-tanpa-ekstensi>
Poster diperbesar SKALA kali lalu QR digambar langsung di resolusi akhir supaya tajam.
"""
import sys

import segno
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# QR langsung ke Vercel. Poster lama yang sudah dicetak memakai rhmatzeka.github.io/tari-sigeh-penguten/
# (dialihkan ke GitHub Pages), jadi tetap jalan.
URL = "https://tarisigehpenguten.vercel.app/"
SKALA = 2
TINTA = "#2b1a0a"
FONT = "work/fonts/Poppins-SemiBold.ttf"

# Koordinat di poster asli 1137x1280: kotak Makna x 147-567, y 882-1165.
# Kotak QR dicerminkan terhadap kartu krem (x 92-1040) dan setinggi kotak Makna.
KOTAK_KANAN, KOTAK_ATAS, KOTAK_BAWAH = 985, 882, 1165
MODUL = 10  # piksel per modul QR di resolusi akhir


def main(src, out):
    poster = Image.open(src).convert("RGB")
    w, h = poster.size
    poster = poster.resize((w * SKALA, h * SKALA), Image.LANCZOS)
    poster = poster.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    d = ImageDraw.Draw(poster)

    qr = segno.make(URL, error="h", micro=False)
    sisi = len(qr.matrix) * MODUL
    pad = 4 * MODUL  # zona sepi 4 modul
    kanan, atas, bawah = KOTAK_KANAN * SKALA, KOTAK_ATAS * SKALA, KOTAK_BAWAH * SKALA
    kiri = kanan - (sisi + 2 * pad)
    d.rounded_rectangle((kiri, atas, kanan, bawah), radius=12 * SKALA,
                        fill="#ffffff", outline=TINTA, width=2 * SKALA)

    qx, qy = kiri + pad, atas + pad
    for r, baris in enumerate(qr.matrix):
        for c, nyala in enumerate(baris):
            if nyala:
                x, y = qx + c * MODUL, qy + r * MODUL
                d.rectangle((x, y, x + MODUL - 1, y + MODUL - 1), fill=TINTA)

    font = ImageFont.truetype(FONT, 17 * SKALA)
    teks = "Scan untuk melihat AR"
    tw = d.textlength(teks, font=font)
    ruang_atas = qy + sisi
    _, t_atas, _, t_bawah = d.textbbox((0, 0), teks, font=font)
    ty = ruang_atas + (bawah - ruang_atas - (t_bawah - t_atas)) / 2 - t_atas - 4 * SKALA
    d.text(((kiri + kanan) / 2 - tw / 2, ty), teks, fill=TINTA, font=font)

    poster.save(out + ".png", optimize=True)
    poster.save(out + ".pdf", resolution=150)
    print(out, poster.size, "QR versi", qr.version)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
