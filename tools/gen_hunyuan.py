"""Foto -> GLB bertekstur lewat Space HuggingFace Hunyuan3D-2 (pembanding TRELLIS)."""
import shutil, sys
from gradio_client import Client, handle_file
from huggingface_hub import get_token

foto, out = sys.argv[1], sys.argv[2]
c = Client("tencent/Hunyuan3D-2", token=get_token(), verbose=True, httpx_kwargs={"timeout": 180})
hasil = c.predict(
    caption=None, image=handle_file(foto),
    mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
    steps=30, guidance_scale=5.0, seed=1234, octree_resolution=256,
    check_box_rembg=True, num_chunks=8000, randomize_seed=False,
    api_name="/generation_all",
)
print("hasil:", hasil[:2], hasil[3])
shutil.copy(hasil[1], out)
print("tersimpan", out)
