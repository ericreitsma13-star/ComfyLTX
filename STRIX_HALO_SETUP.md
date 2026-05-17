# Strix Halo SDXL Setup

**Hardware:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, RDNA 3.5)  
**System RAM:** 93GB (unified, GPU sees ~86GB)  
**ROCm:** 7.2.1 | **PyTorch:** 2.9.1+rocm7.11 | **Diffusers:** 0.38.0

## Fixes Applied (May 15)

### 1. HSA_OVERRIDE_GFX_VERSION=11.5.1
Matches actual gfx1151 hardware. Was `11.0.0` (RDNA 3) which caused
suboptimal memory management. Set permanently in `/etc/environment`.

### 2. amdgpu.noreplay=1 (kernel param)
Prevents GPU driver workqueue hang that caused full machine power-off.
Without it, the KFD driver tries to replay failed GPU commands, blocking
the workqueue for 10+ seconds (`kfd_process_wq_release hogged CPU`).
Added to GRUB_CMDLINE_LINUX_DEFAULT in `/etc/default/grub`. Reboot required.

### 3. torchvision ABI patch
torchvision (ROCm 7.13) vs torch (ROCm 7.11) mismatch caused:
`RuntimeError: operator torchvision::nms does not exist`
Patched `_meta_registrations.py` with try/except around register_fake.

## Crash Pattern
Machine powers off with no clean shutdown when running batch SDXL.
Log signature: `kfd_process_wq_release [amdgpu] hogged CPU for >10000us`
Trigger: Rapid GPU memory alloc/dealloc cycles, VRAM >90%.

## Tested SDXL Settings (Stable)
- 1024x1024, 20 steps, CFG 7.0, bfloat16
- Attention slicing + VAE tiling enabled
- ~30-40s per image
- gc.collect() + torch.cuda.empty_cache() between gens
