"""Sejajarkan foto tajam ke render tampak depan model memakai titik wajah & tubuh (MediaPipe).

(1) badan: similarity dari titik pose (bahu, siku, pergelangan, pinggul) render vs foto
(2) wajah: affine dari 468 titik wajah render vs foto, dibaurkan di lingkaran sekitar wajah
Pakai: .venv/bin/python -I tools/sejajar_foto.py <render-depan.png> <foto-tajam.png> <pose.task> <face.task> <keluaran.png>
"""
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

render_p, foto_p, pose_task, face_task, out_p = sys.argv[1:6]
ren = cv2.imread(render_p, cv2.IMREAD_UNCHANGED)
alpha = ren[:, :, 3:4] / 255.0
ren = (ren[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)  # latar putih
foto = cv2.imread(foto_p)
H, W = foto.shape[:2]


def mpimg(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def titik_pose(img):
    o = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=pose_task), running_mode=vision.RunningMode.IMAGE)
    with vision.PoseLandmarker.create_from_options(o) as d:
        r = d.detect(mpimg(img))
    return np.array([[p.x * W, p.y * H, p.visibility] for p in r.pose_landmarks[0]])


def titik_wajah(img, pusat, r):
    # potong sekitar wajah supaya detektor wajah bekerja di resolusi cukup
    x0, y0 = int(max(0, pusat[0] - r)), int(max(0, pusat[1] - r))
    pot = img[y0:int(pusat[1] + r), x0:int(pusat[0] + r)]
    o = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=face_task), running_mode=vision.RunningMode.IMAGE, num_faces=1)
    with vision.FaceLandmarker.create_from_options(o) as d:
        res = d.detect(mpimg(pot))
    if not res.face_landmarks:
        return None
    h, w = pot.shape[:2]
    return np.array([[p.x * w + x0, p.y * h + y0] for p in res.face_landmarks[0]])


pr, pf = titik_pose(ren), titik_pose(foto)
idx = [11, 12, 13, 14, 15, 16, 23, 24]
ok = [i for i in idx if pr[i, 2] > 0.5 and pf[i, 2] > 0.5]
M1, _ = cv2.estimateAffinePartial2D(pr[ok, :2], pf[ok, :2])  # render -> foto
galat1 = np.linalg.norm(pr[ok, :2] @ M1[:, :2].T + M1[:, 2] - pf[ok, :2], axis=1)
print(f"badan: {len(ok)} titik, galat sisa median {np.median(galat1):.1f}px, M={M1.round(3).tolist()}")
# foto_sejajar(x) = foto(M1 * x)
foto1 = cv2.warpAffine(foto, M1, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

hidung = pr[0, :2]
r = abs(pr[7, 0] - pr[8, 0]) * 1.4
fr = titik_wajah(ren, hidung, r)
ff = titik_wajah(foto1, hidung, r)
if fr is None or ff is None:
    print("wajah tidak terdeteksi di", "render" if fr is None else "foto", "- pakai penyejajaran badan saja")
    hasil = foto1
else:
    M2, inl = cv2.estimateAffine2D(fr, ff, ransacReprojThreshold=3)
    galat2 = np.linalg.norm(fr @ M2[:, :2].T + M2[:, 2] - ff, axis=1)
    sebelum = np.linalg.norm(fr - ff, axis=1)
    print(f"wajah: galat sebelum {np.median(sebelum):.1f}px -> sesudah {np.median(galat2):.1f}px")
    foto2 = cv2.warpAffine(foto1, M2, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    pusat = fr.mean(0)
    rw = np.ptp(fr[:, 0]) * 0.75
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - pusat[0], (yy - pusat[1]) / 1.2)
    bobot = np.clip((1.4 * rw - d) / (0.5 * rw), 0, 1)[..., None]
    hasil = (foto2 * bobot + foto1 * (1 - bobot)).astype(np.uint8)
cv2.imwrite(out_p, hasil)
print("tersimpan", out_p)
