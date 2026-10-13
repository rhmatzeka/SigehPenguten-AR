"""Video -> titik pose 3D per frame (MediaPipe Pose Landmarker, CPU).

Pakai: .venv/bin/python -I tools/ekstrak_pose.py <video.mp4> <model.task> <keluaran.npz> [cek.jpg]
Menyimpan world landmarks (meter, pusat di pinggul) 33x3, visibility, dan landmark 2D.
"""
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


def main(video, model, out, cek=None):
    opsi = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    world, img, vis = [], [], []
    contoh = []
    with vision.PoseLandmarker.create_from_options(opsi) as det:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
            if res.pose_world_landmarks:
                w = res.pose_world_landmarks[0]
                n = res.pose_landmarks[0]
                world.append([[p.x, p.y, p.z] for p in w])
                img.append([[p.x, p.y] for p in n])
                vis.append([p.visibility for p in w])
            else:
                world.append(np.full((33, 3), np.nan))
                img.append(np.full((33, 2), np.nan))
                vis.append(np.zeros(33))
            if cek and i % int(fps * 2) == 0:
                h, wd = frame.shape[:2]
                for x, y in img[-1]:
                    if not np.isnan(x):
                        cv2.circle(frame, (int(x * wd), int(y * h)), 6, (0, 255, 0), -1)
                contoh.append(cv2.resize(frame, (240, 360)))
            i += 1
    np.savez(out, world=np.array(world, float), img=np.array(img, float), vis=np.array(vis, float), fps=fps)
    hilang = int(np.isnan(np.array(world)[:, 0, 0]).sum())
    print(f"{i} frame, fps {fps}, tanpa deteksi {hilang}")
    if cek and contoh:
        cv2.imwrite(cek, np.hstack(contoh))


if __name__ == "__main__":
    main(*sys.argv[1:])
