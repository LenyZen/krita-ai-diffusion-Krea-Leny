# MiniMax H3 — Video workspace (Phase 1)

Native **Reference→Video (R2V)** from Krita via ComfyUI ≥ **0.30** nodes
(`MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`).

Workspace menu: **Video · MiniMax H3**

Official overview: https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui  
Docs: https://docs.comfy.org/tutorials/video/minimax/minimax-h3  
Weights: https://huggingface.co/Comfy-Org/MiniMax-H3

## What Phase 1 includes

- Up to **6 reference layers** → `<Picture 1>`…`<Picture 6>`
- Prompt + duration + megapixel / aspect presets
- **Turbo** toggle (5-step pack LoRAs when present)
- Frames return to Krita → **Apply to Timeline** (same path as Wan Animate)
- Optional audio: **Find audio…** picks any local file (desktop / other drive); uploaded to ComfyUI on Generate as `<Audio 1>`

Not in Phase 1 (planned later): Spectrum aggressor, RTX VSR, realtime TAE preview, MP4 auto-export with audio mux.

## Required models

| File | Folder |
|------|--------|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (or Dasiwa REF2VA hybrid) | `models/diffusion_models/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

## Turbo LoRAs (optional, recommended)

Place under `models/loras/` or `models/loras/Minimax H3/`:

| File | Source |
|------|--------|
| `minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo-SLA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) |
| `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors` | [drbaph/MiniMax-H3-Turbo-Lora-ComfyUI](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI) |

## Server notes

- Your Krita-managed Comfy at `E:\ComfyforKrita\ComfyUI` must be **≥ 0.30** with H3 weights, **or** point Connection at an install that already has them (e.g. `F:\ComfyUI_V116`).
- Start at **0.4 MP / ~5s** for iteration; raise size after the pipeline is stable.
- Prompt tip: write cut-based scenes; if you omit `<Picture N>` tags, the plugin prepends them from the ref layers you set.

## vs Animate (Wan)

| | Wan Animate | MiniMax H3 |
|--|-------------|------------|
| Best for | I2V / FLF2V motion | Multi-ref identity + native audio conditioning |
| Speed path | lightx2v 4-step | turbo LoRAs + 5-step |
| Keep both | Yes — until H3 wins your movie work | Phase 1 alongside Wan |
