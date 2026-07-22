# Wan 2.2 Graph workflows (Phase 0)

Native-node API graphs for smoke-testing Wan 2.2 I2V / FLF2V in ComfyUI before using the built-in **Animate (Wan)** workspace (which also supports **T2V**).

Official Comfy overview: https://comfyanonymous.github.io/ComfyUI_examples/wan22/

## Shared text encoder + VAE

| File | Folder | Download |
|------|--------|----------|
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors |
| `wan_2.1_vae.safetensors` | `vae/` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors |

## I2V — distilled lightx2v 4-step (12GB-friendly)

Repo: https://huggingface.co/lightx2v/Wan2.2-Distill-Models

| File | Folder | Download |
|------|--------|----------|
| `wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | `diffusion_models/` | https://huggingface.co/lightx2v/Wan2.2-Distill-Models/resolve/main/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors |
| `wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | `diffusion_models/` | https://huggingface.co/lightx2v/Wan2.2-Distill-Models/resolve/main/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors |

## I2V — official Comfy fp8 (non-distilled)

Repo: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged

| File | Download |
|------|----------|
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors |

## T2V — official Comfy fp8

| File | Folder | Download |
|------|--------|----------|
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors |

### Optional T2V 4-step LoRAs → `loras/`

| File | Download |
|------|----------|
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors |

Do not mix I2V and T2V UNETs in one run.

## Custom nodes (optional Graph / MoE)

- `ComfyUI-WanMoeKSampler` — https://github.com/stduhpf/ComfyUI-WanMoeKSampler
- `ComfyUI-VideoHelperSuite` — https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

The built-in Animate workspace uses **native** Comfy Wan nodes only (no MoE required).

## How to smoke-test

1. Load `wan22_i2v_native_api.json` or `wan22_flf2v_native_api.json` in ComfyUI (API format / Convert).
2. Point LoadImage nodes at a still; set length 33, size 640×640 on 12GB.
3. For Krita Graph: open the graph in Comfy while Krita is connected, then **Save Workflow to File** from the Graph workspace so ETN parameter widgets appear.
4. Prefer the docker menu **Animate (Wan)** for day-to-day use — it builds the same dual-sampler path and applies frames via **Generate Animation** semantics.

## 12GB notes

Start at **640×640 / 33 frames**. Raise length or resolution only after a successful short clip. Soft cap remains **81 frames (~5s)**.
