import json, requests

API = "http://127.0.0.1:8188/prompt"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

# Decode a zero latent -> should produce blank gray if VAE is correct
prompt = {
    "1": node("VAELoader",
        vae_name="wan_2.1_vae.safetensors"),
    "2": node("EmptyHunyuanLatentVideo",
        width=256, height=256, length=16, batch_size=1),
    "3": node("VAEDecode",
        samples=("2", 0),
        vae=("1", 0)),
    "4": node("SaveImage",
        images=("3", 0),
        filename_prefix="vae_zero"),
}

r = requests.post(API, json={"prompt": prompt})
print(r.status_code, r.text[:500])
