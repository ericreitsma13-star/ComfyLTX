import sys, os, torch

sys.path = ["/opt/ComfyUI"] + sys.path
os.environ["PYTHONPATH"] = "/opt/ComfyUI"

# Force import of the full comfy module structure first
import main as _  # noqa: F401
import comfy.text_encoders.llama  # noqa: F401

from comfy.text_encoders.lt import ltxav_te

clip_target_cls = ltxav_te()
model = clip_target_cls(device="cpu")
print(f"text_embedding_projection: {model.text_embedding_projection}")
print(f"  weight shape: {model.text_embedding_projection.weight.shape}")
print(f"  in_features: {model.text_embedding_projection.in_features}")
print(f"  out_features: {model.text_embedding_projection.out_features}")

L = model.gemma3_12b.num_layers
print(f"\nGemma3_12B num_layers: {L}")

D = 3840
out = torch.randn(1, 20, D * (L + 1))
print(f"\nInput to text_embedding_projection: {out.shape}")
print(f"Weight shape: {model.text_embedding_projection.weight.shape}")
print(f"Weight.T shape: {model.text_embedding_projection.weight.T.shape}")

try:
    result = torch.nn.functional.linear(out, model.text_embedding_projection.weight)
    print(f"F.linear result shape: {result.shape}")
except Exception as e:
    print(f"F.linear ERROR: {e}")

# Now try the actual encode_token_weights flow with movedim
out2 = torch.randn(1, L + 1, 20, D)
print(f"\nIntermediate out shape: {out2.shape}")
out3 = out2.movedim(1, -1)
print(f"After movedim(1, -1): {out3.shape}")
out3 = out3.reshape(out3.shape[0], out3.shape[1], -1)
print(f"After reshape: {out3.shape}")
try:
    result = torch.nn.functional.linear(out3, model.text_embedding_projection.weight)
    print(f"F.linear result shape: {result.shape}")
except Exception as e:
    print(f"F.linear ERROR: {e}")
