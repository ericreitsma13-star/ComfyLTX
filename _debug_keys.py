import sys, os

sys.path.insert(0, "/opt/ComfyUI")
os.environ["PYTHONPATH"] = "/opt/ComfyUI"
import main as _
import comfy.utils

sd = comfy.utils.load_torch_file(
    "/root/comfy-models/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
)
keys = list(sd.keys())
print(f"Total keys: {len(keys)}")
for k in keys:
    print(f"  {k}: {sd[k].shape}")
