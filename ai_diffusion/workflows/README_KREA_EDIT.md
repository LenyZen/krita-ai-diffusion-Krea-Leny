# Krea Edit workspace

Identity-preserving instruction edit via **comfyui-krea2edit** (Identity Edit LoRA).

## Links (nodes + models)

| What | Link |
|------|------|
| Krea Edit nodes | https://github.com/lbouaraba/comfyui-krea2edit |
| Identity Edit LoRA (v1.2) | https://huggingface.co/conradlocke/krea2-identity-edit |
| Inpaint Crop & Stitch | https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch |
| Krea 2 base models (UNET / TE / VAE / Turbo LoRA) | https://huggingface.co/Comfy-Org/Krea-2 |
| Reference Comfy workflow (Inpaint&Edit) | https://civitai.com/models/2769318/krea2-inpaintandedit |
| Optional RTX 50xx attention fix | https://github.com/lonecatone23/comfyUI-BlackwellAttentionFix |

### Install nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lbouaraba/comfyui-krea2edit.git
git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git
# optional on RTX 5080/5090 if xformers crashes Identity Edit:
# git clone https://github.com/lonecatone23/comfyUI-BlackwellAttentionFix.git
```

### Download models

```
→ models/loras/Krea2/   (or models/loras/)
  krea2_identity_edit_v1_2.safetensors
  from https://huggingface.co/conradlocke/krea2-identity-edit

→ models/diffusion_models/   (same as Generate)
  krea2_turbo_fp8_scaled.safetensors  and/or  krea2_raw_fp8_scaled.safetensors
  https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors
  https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_fp8_scaled.safetensors

→ models/text_encoders/
  https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors

→ models/vae/
  https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors

→ models/loras/krea/sliders/   (create folder; drop slider LoRAs here)
```

Restart ComfyUI and reconnect from Krita after installing.

## Full canvas vs local Inpaint Edit

| Krita selection | Path |
|-----------------|------|
| None (or whole canvas) | Full-canvas Identity Edit |
| Partial selection | **Inpaint Edit**: crop → Identity Edit on 1024×1024 → soft stitch |
| Selection **and** active Sliders | Forced **full-canvas** (see Sliders note below) |

Local path matches the CivitAI *Krea2 Inpaint&Edit* pattern:

1. `InpaintCropImproved` — mask + context factor → forced 1024×1024 crop  
2. Same Identity Edit stack (`Krea2EditModelPatch` + `Krea2EditGroundedEncode` + KSampler)  
3. `InpaintStitchImproved` — blend only the masked region; **unmasked pixels stay untouched**

The paint/selection mask is **not** wired to `ref_boost_mask` (that input is a reference-fidelity ROI; boosting the area you want to change fights the edit).

### Settings (Advanced → Inpaint crop)

- **Use selection for local Inpaint Edit** — on by default  
- **Mask expand** — default `8`  
- **Mask blend** — default `32` (soft seam)  
- **Context factor** — default `1.5` (extra surround for identity/scene)

## Refs (image identity)

Identity Edit accepts **two images only**: the canvas/crop (A) and one secondary ref (B).

### Role sheets (recommended for multiple cues)

On the **Refs** tab you can assign separate layers: Face / Hair / Clothing / Pose.

When more than one is set, the plugin **collages them into a single sheet** and sends that as image B.

You can still use **Subject** (one person photo) or a handmade **Full Ref sheet**.

## Sliders (`loras/krea/sliders`)

Edit responds especially well to “slider” LoRAs on the **full image**.

1. Create `models/loras/krea/sliders/`
2. Drop slider `.safetensors` files there; reconnect ComfyUI
3. Advanced → **Sliders** → **Add Slider** (dropdown is filtered to that folder)

Bars: **center = 0 (off)**, right up to **+15**, left down to **−15**.

### Important: Sliders + selection

Sliders work great when the **entire canvas** is the ref/edit target. With a **marquee / selection**, they often push the crop path into rebuilding a **smaller whole-frame image** instead of a clean local inpaint.

**Practical rule:** use Sliders on full-canvas Edit; **Deselect** (or zero all slider bars) when you need selection Inpaint. If both are active, the plugin **forces full-canvas Edit** and shows that in the status line.

## Edit field starters

Character / Scene / Lighting / Other each have a **Pick a starting point…** dropdown (~10 examples). Selecting one fills the text box so you can rewrite it.

## Requirements checklist

- [ ] Nodes: [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)
- [ ] Nodes: [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) (selection Inpaint)
- [ ] LoRA: [krea2_identity_edit_v1_2](https://huggingface.co/conradlocke/krea2-identity-edit)
- [ ] Krea 2 Turbo or RAW + TE + VAE from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)
- [ ] Optional sliders folder: `models/loras/krea/sliders/`
