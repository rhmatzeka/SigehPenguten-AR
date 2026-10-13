"""Foto -> GLB bertekstur lewat Space HuggingFace TRELLIS (gratis, ZeroGPU)."""
import shutil, sys
from gradio_client import Client, handle_file
from huggingface_hub import get_token

foto, out = sys.argv[1], sys.argv[2]
c = Client("trellis-community/TRELLIS", token=get_token(), verbose=True, httpx_kwargs={"timeout": 180})
c.predict(api_name="/start_session")
prep = c.predict(image=handle_file(foto), api_name="/preprocess_image")
print("praproses:", prep)
hasil = c.predict(
    image=handle_file(prep["path"] if isinstance(prep, dict) else prep),
    multiimages=[], seed=0,
    ss_guidance_strength=7.5, ss_sampling_steps=12,
    slat_guidance_strength=3.0, slat_sampling_steps=12,
    multiimage_algo="stochastic", mesh_simplify=0.9, texture_size=2048,
    api_name="/generate_and_extract_glb",
)
print("hasil:", hasil)
glb = hasil[2] if hasil[2] else hasil[1]
shutil.copy(glb, out)
shutil.copy(prep["path"] if isinstance(prep, dict) else prep, out.replace(".glb", "-input.png"))
print("tersimpan", out)
