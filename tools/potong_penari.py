"""Potong penari dari latar abu-abu foto (GrabCut) -> PNG transparan (dipakai layar awal & siger tepak).

Pakai: .venv/bin/python -I tools/potong_penari.py work/v2/foto-penari.jpg <keluaran.png>
"""
import cv2, numpy as np, sys
src, out = sys.argv[1], sys.argv[2]
im = cv2.imread(src); h, w = im.shape[:2]
# latar abu-abu seragam: tandai piksel mirip warna sudut sebagai "pasti latar"
bg = np.median(np.concatenate([im[:20,:20].reshape(-1,3), im[:20,-20:].reshape(-1,3), im[-20:,:20].reshape(-1,3), im[-20:,-20:].reshape(-1,3)]),0)
d = np.linalg.norm(im.astype(float)-bg,axis=2)
sat = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[...,1]
mask = np.full((h,w), cv2.GC_PR_FGD, np.uint8)
mask[(d<18)&(sat<30)] = cv2.GC_PR_BGD
mask[(d<8)] = cv2.GC_BGD
mask[:3,:]=mask[-3:,:]=cv2.GC_BGD; mask[:,:3]=mask[:,-3:]=cv2.GC_BGD
b=np.zeros((1,65),float); f=np.zeros((1,65),float)
cv2.grabCut(im, mask, None, b, f, 6, cv2.GC_INIT_WITH_MASK)
a = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
# buang pulau kecil, ambil komponen terbesar
n, lab, st, _ = cv2.connectedComponentsWithStats(a)
keep = 1+np.argmax(st[1:,cv2.CC_STAT_AREA]); a = np.where(lab==keep,255,0).astype(np.uint8)
a = cv2.GaussianBlur(a,(3,3),0)
rgba = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA); rgba[...,3]=a
x,y,ww,hh = cv2.boundingRect((a>10).astype(np.uint8))
rgba = rgba[max(0,y-8):y+hh+8, max(0,x-8):x+ww+8]
cv2.imwrite(out, rgba)
print(rgba.shape)
