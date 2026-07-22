# Wan 2.2 Phase 0 — setup & metrics log

Date: 2026-07-20  
Hardware target: RTX 3060 12GB

## Installed (complete)

| Item | Path | Size |
|------|------|------|
| Wan 2.2 I2V high (distill fp8) | `ComfyUI/models/diffusion_models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui_1030.safetensors` | 15.0 GB |
| Wan 2.2 I2V low (distill fp8) | `ComfyUI/models/diffusion_models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | 15.0 GB |
| Wan VAE | `ComfyUI/models/vae/wan_2.1_vae.safetensors` | 242 MB |
| umt5 XXL fp8 | `ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.3 GB |
| `ComfyUI-WanMoeKSampler` | custom_nodes | installed |
| `ComfyUI-VideoHelperSuite` | custom_nodes | installed |
| Graph API workflows | `ai_diffusion/workflows/wan22_*_native_api.json` + AppData workflows | shipped |
| Built-in **Animate (Wan)** workspace | plugin | Phase 1+2 shipped |

## Smoke-test protocol (after Krita + Comfy restart)

1. Open workspace menu → **Animate (Wan)**.
2. Confirm High/Low noise dropdowns list the Wan models.
3. **I2V**: start layer = still, 640×640, 33 frames → Generate Animation → Apply to Timeline.
4. **FLF2V**: set end layer → Generate.
5. Optional: **Stitch bridge** after a clip; Comic start/end panel pickers if Comic page has panels.

## Results (fill after first local run)

| Test | Res | Frames | Peak VRAM | System RAM | Duration | Notes (color / blur / OOM) |
|------|-----|--------|-----------|------------|----------|----------------------------|
| I2V | 640 | 33 | | | | ready to run — models on disk |
| I2V | 768 | 33 | | | | |
| I2V | 1024 | 33 | | | | |
| FLF2V | 640 | 33 | | | | |
| FLF2V | 640 | 49 | | | | |

## Expected on 3060 12GB

- 640×640 @ 33 frames with distilled 4-step: viable starting point
- 1024×1024 @ 81 frames: likely OOM — drop length first
- Soft cap ~81 frames (~5s); same long-load / websocket lessons as Krea 2

## Phase 1 / 2 delivered

- Workspace **Animate (Wan)**: I2V + Start/End (native dual KSampler path)
- Comic panel start/end pickers
- **Stitch bridge** (FLF2V from last frame → end)
