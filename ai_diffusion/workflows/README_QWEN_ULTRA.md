# Qwen · Ultra (Image Edit 2.1)

Krita workspace that mirrors the **QWEN-IMAGE-EDIT-21_ULTRA_WORKFLOW** graph.

## Modes

| Mode | What it does |
|------|----------------|
| **Multi-Edit** | Instruction edit with canvas + up to 6 reference layers (`<image1>`…) |
| **Text to Image** | Pure T2I (Ultra default: `res_2s` / `beta`, 50 steps) |
| **Inpaint** | Uses the active Krita selection as a noise mask |
| **Outpaint** | Pads the canvas (L/T/R/B) then continues the scene |

## Required models (ComfyUI)

Place under your shared `F:\models` (or Comfy `models/`) paths:

- **Diffusion** `diffusion_models/`: `qwen_image_2.1_int8_convrot.safetensors`
- **Text encoder** `text_encoders/` or `clip/`: `qwen3vl_8b_int8_convrot.safetensors`
- **VAE** `vae/`: `texture_fix_vae_for_qwen_image_2.1_bf16.safetensors`

## ComfyUI version

Prefers **`TextEncodeQwenImage21`** (ComfyUI **≥ 0.38**). Falls back to `TextEncodeQwenImageEdit` / `EditPlus` on older servers.

## Optional LoRAs (from the Ultra JSON)

Examples that appear in the source workflow:

- `bfs_head_v1.1_qwen_2.1.safetensors` / `bfs_body_swap_v1.0_qwen_2.1.safetensors`
- `qwen-image-2.1-lightning-…`, style LoRAs
- `qwen-image-2.1-outpaint-v2.safetensors` (Outpaint)
- Object remover / mover bbox LoRAs (Inpaint helpers)

Add them via **Add LoRA** in the Qwen · Ultra panel.

## Usage

1. Connect to a Comfy server that has the models + Qwen 2.1 nodes.
2. Workspace selector → **Qwen · Ultra**.
3. Pick a mode / preset, write a prompt, set refs, **Generate**.
4. Apply the result from **History** when the job finishes.
