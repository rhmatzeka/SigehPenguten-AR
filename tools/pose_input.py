"""Deteksi sendi MediaPipe + bbox siluet pada foto input generator (latar hitam/transparan).

Pakai: .venv/bin/python -I tools/pose_input.py <input.png> <model.task> <keluaran.json>
"""
import json
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

src, model, out = sys.argv[1:4]
im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
if im.ndim == 3 and im.shape[2] == 4:
    fg = (im[:, :, 3] > 128).astype(np.uint8)
    im = im[:, :, :3]
else:
    fg = (im.max(axis=2) > 20).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
big = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
ys, xs = np.where(big)
bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
rgb = cv2.cvtColor(np.where(big[..., None], im, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
opsi = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.IMAGE)
with vision.PoseLandmarker.create_from_options(opsi) as d:
    r = d.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
h, w = big.shape
lm = [[p.x * w, p.y * h, p.visibility] for p in r.pose_landmarks[0]]
json.dump({"w": w, "h": h, "bbox": bb, "lm": lm}, open(out, "w"))
print("bbox", bb, "rasio", round((bb[2] - bb[0]) / (bb[3] - bb[1]), 3))
