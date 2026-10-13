"""Foto -> GLB bertekstur lewat Space HuggingFace TRELLIS.2 (lebih detail dari TRELLIS lama).

Pakai: .venv/bin/python -I tools/gen_trellis2.py <foto> <keluaran.glb> [resolusi] [seed]
"""
import shutil, sys
from gradio_client import Client, handle_file
from huggingface_hub import get_token

foto, out = sys.argv[1], sys.argv[2]
resolusi = sys.argv[3] if len(sys.argv) > 3 else "1024"
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
c = Client("microsoft/TRELLIS.2", token=get_token(), verbose=True, httpx_kwargs={"timeout": 300})
c.predict(api_name="/start_session")
prep = c.predict(input=handle_file(foto), api_name="/preprocess_image")
path = prep["path"] if isinstance(prep, dict) else prep
shutil.copy(path, out.replace(".glb", "-input.png"))
print("praproses:", path)
c.predict(image=handle_file(path), seed=seed, resolution=resolusi, api_name="/image_to_3d")
print("bentuk selesai")
hasil = c.predict(decimation_target=200000, texture_size=2048, api_name="/extract_glb")
print("hasil:", hasil)
shutil.copy(hasil[1] or hasil[0], out)
print("tersimpan", out)
