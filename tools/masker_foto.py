"""Masker siluet dari foto input generator (latar hitam atau transparan), tepi dikikis supaya proyeksi tidak bocor.

Pakai: .venv/bin/python -I tools/masker_foto.py <foto-input.png> <masker.png> [kikis_px]
"""
import sys

import cv2
import numpy as np

im = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
if im.ndim == 3 and im.shape[2] == 4:
    fg = (im[:, :, 3] > 128).astype(np.uint8)
else:
    fg = (im[:, :, :3].max(axis=2) > 20).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
fg = cv2.erode(fg, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
fg = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), k / 2)
cv2.imwrite(sys.argv[2], np.clip(fg * 255, 0, 255).astype(np.uint8))
print("masker", sys.argv[2], fg.shape)
