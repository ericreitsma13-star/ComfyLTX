# Memory-Efficient Config for RTX 4090 Mobile (16GB VRAM, 64GB RAM)

## The Problem
Current pipeline uses `ltx-2.3-22b-distilled-1.1.safetensors` (43 GB) as checkpoint — impossible on 16GB VRAM. Even the Q6_K GGUF (20 GB) is too big. Models accumulate in VRAM across stages with no cleanup.

## The Fix: Three Changes

### 1. Use smaller quants for everything
| Component | Use this instead of | Savings |
|---|---|---|
| **UNet** | `LTX-2.3-22B-distilled-1.1-Q3_K_M.gguf` (14 GB) | Skip 43 GB checkpoint entirely |
| **Text Encoder** | `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` (7 GB) | 1.8 GB less than fp4 mixed safetensors |
| **Video VAE** | Already small (1.4 GB) | ✓ |
| **Z-Image (ref gen)** | **SKIP on this machine** — use SDXL pipeline instead | Saves 12 GB |
| **LLM** | `supergemma4-26b...gguf` (16 GB) but load/unload per stage | Only in VRAM when prompt gen runs |

### 2. Explicit memory cleanup between stages
Add POST to `/free` endpoint after every prompt:
```python
def free_memory():
    req = urllib.request.Request(
        "http://127.0.0.1:8188/free",
        data=json.dumps({"free_memory": True}).encode(),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=30)
```

### 3. Split workflow into 3 separate GPU stages
```
Stage 1: LLM prompt gen → /free → 
Stage 2: Z-Image ref images (×N, /free between each) → /free → 
Stage 3: LTX video gen (×N per scene, /free between each) → /free →
Stage 4: FFmpeg assembly (no GPU)
```

## VRAM Budget

### Stage 2: Z-Image (references)
| Model | VRAM |
|---|---|
| Z-Image Q6_K GGUF UNet | ~4 GB |
| Qwen 4B text encoder | ~6 GB (or ~2 GB with GGUF CLIP) |
| ae VAE | ~0.3 GB |
| Activations | ~2 GB |
| **Total** | **~12 GB** (or ~8 GB with GGUF CLIP) ✅ |

### Stage 3: LTX Video (tightest)
| Model | VRAM |
|---|---|
| Q4_K_M GGUF UNet | ~8-9 GB (mmap + dynamic offloading) |
| Gemma Q4_XL GGUF text encoder on CPU | ~0 GB (text encode on CPU) |
| Video VAE | ~0.5 GB |
| Audio VAE | ~0.3 GB |
| Activations + intermediates | ~3 GB |
| **Total** | **~14 GB** — fits with dynamic offloading ✅ |

Note: ComfyUI already handles "dynamic VRAM loading" — it stages models in RAM and keeps only active layers in VRAM. The Q3_K_M GGUF is memory-mapped so weights stay on disk/RAM until needed by a GPU operation.

## Pipeline Script Changes Needed

In each pipeline script that calls `queue()`:

```python
def queue(prompt_wf):
    req = urllib.request.Request(
        f"{COMFY}/prompt",
        data=json.dumps({"prompt": prompt_wf}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["prompt_id"]

def free_vram():
    """Unload ALL models from VRAM. Call between stages."""
    req = urllib.request.Request(
        f"{COMFY}/free",
        data=json.dumps({"free_memory": True}).encode(),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=30)

# Usage pattern:
for scene in scenes:
    pid = queue(workflow_for_scene(scene))
    wait_for_completion(pid)
    free_vram()  # ← critical! clears VRAM for next scene
```

## Workflow Model Changes
In the workflow JSON, replace `CheckpointLoaderSimple` with:
- `UnetLoaderGGUF`: `LTX-2.3-22B-distilled-1.1-Q3_K_M.gguf`
- `DualCLIPLoaderGGUF` or `LTXAVTextEncoderLoader`: `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf`
- Keep existing VAELoader + LTXVAudioVAELoader

## Verifying it works
Watch `nvidia-smi` during runs:
```bash
watch -n 1 nvidia-smi
```
VRAM should peak at ~15-16 GB and drop to near 0 after each `/free` call.
