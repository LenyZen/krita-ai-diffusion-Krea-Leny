"""Wan 2.2 Animate workspace — T2V, I2V, and start/end (FLF2V) video from Krita."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, QUuid, pyqtSignal

from .. import eventloop
from ..backend.api import LoraInput, WanVideoInput, WanVideoMode
from ..backend import workflow as workflow_module
from ..backend.workflow import prepare_wan_video
from ..image import Bounds, Image, multiple_of
from ..localization import translate as _
from .jobs import JobKind, JobParams, JobState
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from .model import DocumentModel

log = logging.getLogger(__name__)

MAX_ANIMATE_LORAS = 3


@dataclass
class AnimateLora:
    name: str = ""
    strength: float = 1.0

    def to_dict(self):
        return {"name": self.name, "strength": self.strength}

    @staticmethod
    def from_dict(data: dict):
        return AnimateLora(data.get("name", ""), float(data.get("strength", 1.0)))

    def as_input(self) -> LoraInput | None:
        if not self.name:
            return None
        return LoraInput(self.name, self.strength)


# Preferred distilled Wan 2.2 I2V filenames (lightx2v 4-step baked).
DEFAULT_HIGH_NOISE = (
    "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui_1030.safetensors"
)
DEFAULT_LOW_NOISE = (
    "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors"
)
# Official Comfy Wan 2.2 T2V 14B fp8 (pair with optional lightx2v T2V LoRAs for 4-step).
DEFAULT_T2V_HIGH_NOISE = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
DEFAULT_T2V_LOW_NOISE = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
DEFAULT_TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
DEFAULT_VAE = "wan_2.1_vae.safetensors"

# Frame presets: (label_key, frames) — frames must be 4n+1
LENGTH_PRESETS = [
    ("~2s (33 frames)", 33),
    ("~3s (49 frames)", 49),
    ("~4s (65 frames)", 65),
    ("~5s (81 frames)", 81),
]

RESOLUTION_PRESETS = [
    ("640×640 (12GB-friendly)", 640, 640),
    ("768×768", 768, 768),
    ("832×480 (16:9)", 832, 480),
    ("480×832 (9:16)", 480, 832),
    ("1024×1024 (heavy)", 1024, 1024),
]

# Common Comfy KSampler names that work well with Wan 2.2 (plus fallbacks).
WAN_SAMPLERS = [
    "euler",
    "euler_ancestral",
    "lcm",
    "uni_pc",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "ddim",
    "heun",
    "dpm_2",
]

WAN_SCHEDULERS = [
    "simple",
    "normal",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
]

# label, settings dict applied by apply_sampling_preset
SAMPLING_PRESETS: list[tuple[str, dict]] = [
    (
        "Distilled I2V (lcm / 4-step)",
        {
            "sampler": "lcm",
            "scheduler": "sgm_uniform",
            "steps": 4,
            "split_step": 2,
            "cfg": 1.0,
            "shift": 8.0,
        },
    ),
    (
        "T2V + lightx2v (euler / 4-step)",
        {
            "sampler": "euler",
            "scheduler": "simple",
            "steps": 4,
            "split_step": 2,
            "cfg": 1.0,
            "shift": 5.0,
        },
    ),
    (
        "T2V / I2V quality (euler / 20-step)",
        {
            "sampler": "euler",
            "scheduler": "simple",
            "steps": 20,
            "split_step": 10,
            "cfg": 1.0,
            "shift": 5.0,
        },
    ),
    (
        "Experimental (uni_pc / 8-step)",
        {
            "sampler": "uni_pc",
            "scheduler": "simple",
            "steps": 8,
            "split_step": 4,
            "cfg": 1.0,
            "shift": 5.0,
        },
    ),
]


class AnimateMode(Enum):
    i2v = 0
    flf2v = 1
    t2v = 2


def _normalize_length(length: int) -> int:
    length = max(1, int(length))
    if (length - 1) % 4 != 0:
        length = ((length - 1) // 4) * 4 + 1
    return length


def _pick_model(names: list[str], preferred: str, *needles: str) -> str:
    if preferred and preferred in names:
        return preferred
    lower_pref = preferred.lower()
    for n in names:
        if n.lower() == lower_pref:
            return n
    scored: list[tuple[int, str]] = []
    for n in names:
        nl = n.lower()
        score = sum(1 for needle in needles if needle in nl)
        if score:
            scored.append((score, n))
    if scored:
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored[0][1]
    return preferred


def _family_pref(preferred: str, family: str, default: str, names: list[str]) -> str:
    """Prefer a checkpoint matching family (i2v/t2v); else fall back to default."""
    pref = (preferred or "").strip()
    if pref and family in pref.lower():
        return pref
    if any(family in n.lower() for n in names):
        return default
    return pref or default


class AnimateWorkspace(QObject, ObservableProperties):
    """First-class Wan 2.2 T2V / I2V / FLF2V workspace."""

    mode = Property(AnimateMode.i2v, persist=True)
    prompt = Property("", persist=True)
    negative_prompt = Property("", persist=True)
    start_layer_id = Property("", persist=True)
    end_layer_id = Property("", persist=True)
    width = Property(640, persist=True)
    height = Property(640, persist=True)
    length = Property(33, persist=True)
    high_noise_model = Property("", persist=True)
    low_noise_model = Property("", persist=True)
    text_encoder = Property(DEFAULT_TEXT_ENCODER, persist=True)
    vae = Property(DEFAULT_VAE, persist=True)
    steps = Property(4, persist=True)
    split_step = Property(2, persist=True)
    cfg = Property(1.0, persist=True)
    sampler = Property("lcm", persist=True)
    scheduler = Property("sgm_uniform", persist=True)
    shift = Property(8.0, persist=True)
    # Phase 2: comic panel indices (-1 = none)
    comic_start_panel = Property(-1, persist=True)
    comic_end_panel = Property(-1, persist=True)
    # Stitch: generate bridge clip from last frame of previous result to end
    stitch_bridge = Property(False, persist=True)

    mode_changed = pyqtSignal(AnimateMode)
    prompt_changed = pyqtSignal(str)
    negative_prompt_changed = pyqtSignal(str)
    start_layer_id_changed = pyqtSignal(str)
    end_layer_id_changed = pyqtSignal(str)
    width_changed = pyqtSignal(int)
    height_changed = pyqtSignal(int)
    length_changed = pyqtSignal(int)
    high_noise_model_changed = pyqtSignal(str)
    low_noise_model_changed = pyqtSignal(str)
    text_encoder_changed = pyqtSignal(str)
    vae_changed = pyqtSignal(str)
    steps_changed = pyqtSignal(int)
    split_step_changed = pyqtSignal(int)
    cfg_changed = pyqtSignal(float)
    sampler_changed = pyqtSignal(str)
    scheduler_changed = pyqtSignal(str)
    shift_changed = pyqtSignal(float)
    comic_start_panel_changed = pyqtSignal(int)
    comic_end_panel_changed = pyqtSignal(int)
    stitch_bridge_changed = pyqtSignal(bool)
    loras_changed = pyqtSignal()
    modified = pyqtSignal(QObject, str)
    status_changed = pyqtSignal(str)

    def __init__(self, model: DocumentModel):
        super().__init__()
        self._model = model
        self._last_clip_frames: list[Image] = []
        self._status = ""
        self._loras: list[AnimateLora] = []

    @property
    def status(self) -> str:
        return self._status

    def _set_status(self, text: str):
        self._status = text
        self.status_changed.emit(text)

    def idle_hint(self) -> str:
        """Shown when nothing has run yet — where you are + first step."""
        if self.mode is AnimateMode.t2v:
            return _(
                "Text → Video — write a prompt, then Generate (needs wan2.2_t2v_*). "
                "Tip: apply sampling preset “T2V + lightx2v (euler / 4-step)” if using those LoRAs."
            )
        if self.mode is AnimateMode.flf2v:
            return _(
                "Start/End mode — set Start and End layers (or comic panels), add a motion prompt, "
                "then Generate. Distilled I2V models usually want the lcm / 4-step preset."
            )
        return _(
            "Image → Video — set a Start layer (or canvas), add a motion prompt, then Generate. "
            "Distilled lightx2v I2V → lcm preset; official fp8 → try euler / 20-step."
        )

    def _msg_generating(self, mode_label: str, length: int) -> str:
        return _(
            "Generating {mode} — {n} frames… Progress updates below. You can cancel from the queue."
        ).format(mode=mode_label, n=length)

    def _msg_ready(self, n: int, *, kind: str = "clip") -> str:
        # kind: clip | extended
        if kind == "extended":
            head = _("Extended — {n} frames in memory (not on the Timeline yet).").format(n=n)
        else:
            head = _("Ready — {n} frames in memory (not on the Timeline yet).").format(n=n)
        return head + " " + _(
            "Next: Apply to Timeline · Extend from last (continue) · "
            "Stitch (bridge to End layer) · or Generate a new clip."
        )

    def _msg_applied(self, n: int) -> str:
        return _(
            "Applied {n} frames to the Timeline (play with the Animation docker). "
            "Next: Extend from last to continue · Stitch to an End layer · "
            "Generate for a new clip · or File → Render Animation to export video."
        ).format(n=n)

    def _msg_stitched(self, n: int) -> str:
        return _(
            "Stitched and applied {n} frames to the Timeline. "
            "Next: Extend from last · Generate a new clip · "
            "or File → Render Animation to export video."
        ).format(n=n)

    def _msg_failed(self, what: str) -> str:
        return _(
            "{what} failed. Check Connection / server.log, that the right Wan models are selected, "
            "then try again."
        ).format(what=what)

    @property
    def loras(self) -> list[AnimateLora]:
        return self._loras

    def load_loras(self, data: list[dict]):
        self._loras = [AnimateLora.from_dict(d) for d in data if d.get("name")][:MAX_ANIMATE_LORAS]
        self.loras_changed.emit()

    def set_loras(self, loras: list[AnimateLora]):
        self._loras = [l for l in loras if l.name][:MAX_ANIMATE_LORAS]
        self.loras_changed.emit()
        self.modified.emit(self, "loras")

    def add_lora(self) -> bool:
        if len(self._loras) >= MAX_ANIMATE_LORAS:
            return False
        self._loras.append(AnimateLora())
        self.loras_changed.emit()
        self.modified.emit(self, "loras")
        return True

    def remove_lora(self, index: int):
        if 0 <= index < len(self._loras):
            del self._loras[index]
            self.loras_changed.emit()
            self.modified.emit(self, "loras")

    def update_lora(self, index: int, name: str, strength: float):
        if 0 <= index < len(self._loras):
            self._loras[index].name = name
            self._loras[index].strength = strength
            self.modified.emit(self, "loras")

    def loras_as_input(self) -> list[LoraInput]:
        out: list[LoraInput] = []
        for lora in self._loras:
            if inp := lora.as_input():
                out.append(inp)
        return out

    @property
    def can_generate(self) -> bool:
        try:
            client = self._model._connection.client_if_connected
            if client is None:
                return False
            return bool(self.resolve_models(client)[0] and self.resolve_models(client)[1])
        except Exception:
            return False

    def list_unet_models(self) -> list[str]:
        client = self._model._connection.client_if_connected
        if client is None:
            return []
        names: set[str] = set()
        for filename, info in client.models.checkpoints.items():
            from ..backend.resources import Arch

            if info.arch is Arch.wan22 or "wan" in filename.lower():
                names.add(filename)
        try:
            for n in client.models.node_inputs.options("UNETLoader", "unet_name"):
                nl = n.lower()
                if "wan" in nl or "i2v" in nl or "t2v" in nl:
                    names.add(n)
        except Exception:
            pass
        return sorted(names)

    def list_text_encoders(self) -> list[str]:
        client = self._model._connection.client_if_connected
        if client is None:
            return [DEFAULT_TEXT_ENCODER]
        try:
            opts = client.models.node_inputs.options("CLIPLoader", "clip_name")
            return sorted(opts) if opts else [DEFAULT_TEXT_ENCODER]
        except Exception:
            return [DEFAULT_TEXT_ENCODER]

    def list_vaes(self) -> list[str]:
        client = self._model._connection.client_if_connected
        if client is None:
            return [DEFAULT_VAE]
        vaes = list(client.models.vae) if client.models.vae else []
        return sorted(vaes) if vaes else [DEFAULT_VAE]

    def list_samplers(self) -> list[str]:
        """Prefer live ComfyUI sampler list; always keep curated Wan-friendly names."""
        names = list(WAN_SAMPLERS)
        client = self._model._connection.client_if_connected
        if client is not None:
            try:
                opts = client.models.node_inputs.options("KSamplerAdvanced", "sampler_name")
                if opts:
                    # Curated first, then any extras from the server
                    extra = [n for n in opts if n not in names]
                    names = [n for n in names if n in opts] + extra
            except Exception:
                pass
        if self.sampler and self.sampler not in names:
            names.insert(0, self.sampler)
        return names

    def list_schedulers(self) -> list[str]:
        names = list(WAN_SCHEDULERS)
        client = self._model._connection.client_if_connected
        if client is not None:
            try:
                opts = client.models.node_inputs.options("KSamplerAdvanced", "scheduler")
                if opts:
                    extra = [n for n in opts if n not in names]
                    names = [n for n in names if n in opts] + extra
            except Exception:
                pass
        if self.scheduler and self.scheduler not in names:
            names.insert(0, self.scheduler)
        return names

    def apply_sampling_preset(self, preset_key: str) -> bool:
        """Apply a named SAMPLING_PRESETS entry. Returns True if applied."""
        for label, settings in SAMPLING_PRESETS:
            if label == preset_key:
                self.sampler = str(settings["sampler"])
                self.scheduler = str(settings["scheduler"])
                self.steps = int(settings["steps"])
                self.split_step = int(settings["split_step"])
                self.cfg = float(settings["cfg"])
                self.shift = float(settings["shift"])
                self._set_status(
                    _("Sampling preset: {name} — {sampler} / {scheduler}, {steps} steps, shift {shift}.").format(
                        name=label,
                        sampler=self.sampler,
                        scheduler=self.scheduler,
                        steps=self.steps,
                        shift=self.shift,
                    )
                    + " "
                    + _("Adjust freely, then Generate.")
                )
                return True
        return False

    def resolve_models(
        self, client, mode: AnimateMode | None = None
    ) -> tuple[str, str, str, str]:
        unets = self.list_unet_models()
        # Also accept any UNETLoader name if list is empty of wan-tagged
        if not unets:
            try:
                unets = list(client.models.node_inputs.options("UNETLoader", "unet_name"))
            except Exception:
                unets = []
        use_mode = mode if mode is not None else self.mode
        if use_mode is AnimateMode.t2v:
            family = "t2v"
            def_high = DEFAULT_T2V_HIGH_NOISE
            def_low = DEFAULT_T2V_LOW_NOISE
        else:
            family = "i2v"
            def_high = DEFAULT_HIGH_NOISE
            def_low = DEFAULT_LOW_NOISE
        high = _pick_model(
            unets,
            _family_pref(self.high_noise_model, family, def_high, unets),
            "high",
            "noise",
            family,
            "wan",
        )
        low = _pick_model(
            unets,
            _family_pref(self.low_noise_model, family, def_low, unets),
            "low",
            "noise",
            family,
            "wan",
        )
        # Avoid picking the same file for both
        if high and low and high == low:
            for n in unets:
                if n != high and "low" in n.lower() and family in n.lower():
                    low = n
                    break
        tes = self.list_text_encoders()
        te = _pick_model(tes, self.text_encoder or DEFAULT_TEXT_ENCODER, "umt5")
        vaes = self.list_vaes()
        vae = _pick_model(vaes, self.vae or DEFAULT_VAE, "wan")
        return high, low, te, vae

    def _layer_image(self, layer_id: str) -> Image | None:
        if not layer_id:
            return None
        try:
            layer = self._model.layers.find(QUuid(layer_id))
            if layer is None:
                return None
            bounds = Bounds(0, 0, *self._model.document.extent)
            return layer.get_pixels(bounds)
        except Exception as e:
            log.warning("Animate: failed to read layer %s: %s", layer_id, e)
            return None

    def _comic_panel_image(self, index: int) -> Image | None:
        comic = self._model.comic
        if index < 0:
            return None
        return comic.panel_image(index)

    def resolve_start_image(self) -> Image | None:
        if self.comic_start_panel >= 0:
            img = self._comic_panel_image(self.comic_start_panel)
            if img is not None:
                return img
        if self.start_layer_id:
            img = self._layer_image(self.start_layer_id)
            if img is not None:
                return img
        # Fallback: current document projection
        try:
            bounds = Bounds(0, 0, *self._model.document.extent)
            return self._model.document.get_image(bounds)
        except Exception:
            return None

    def resolve_end_image(self) -> Image | None:
        if self.comic_end_panel >= 0:
            img = self._comic_panel_image(self.comic_end_panel)
            if img is not None:
                return img
        if self.end_layer_id:
            return self._layer_image(self.end_layer_id)
        return None

    def generate(self):
        eventloop.run(self._report_errors(self._generate()))

    def generate_stitch(self):
        """Generate a bridge clip from the last frame of the previous clip to the end image."""
        eventloop.run(self._report_errors(self._generate_stitch()))

    def generate_extend(self):
        """I2V from the last frame of the previous clip (prompt-only continue)."""
        eventloop.run(self._report_errors(self._generate_extend()))

    async def _report_errors(self, coro):
        try:
            await coro
        except Exception as e:
            log.exception("Animate generation failed")
            self._model.report_error(str(e))
            self._set_status(_("Error: {msg}").format(msg=str(e)) + " " + _("Fix the issue, then try again."))

    async def _generate(self):
        model = self._model
        client = model._connection.client
        high, low, te, vae = self.resolve_models(client)
        if not high or not low:
            if self.mode is AnimateMode.t2v:
                raise ValueError(
                    _(
                        "Wan 2.2 T2V high/low noise models not found. Download wan2.2_t2v_* into "
                        "ComfyUI/models/diffusion_models (same umt5 + wan_2.1_vae as I2V)."
                    )
                )
            raise ValueError(
                _(
                    "Wan 2.2 high/low noise models not found. Download distilled I2V models into "
                    "ComfyUI/models/diffusion_models (see Animate workspace tip)."
                )
            )

        if self.mode is AnimateMode.t2v:
            if not self.prompt.strip():
                raise ValueError(_("Text → Video needs a motion / scene prompt."))
            mode = WanVideoMode.t2v
            start = None
            end = None
        else:
            start = self.resolve_start_image()
            if start is None:
                raise ValueError(_("Set a start layer (or comic panel) for the clip."))
            mode = WanVideoMode.flf2v if self.mode is AnimateMode.flf2v else WanVideoMode.i2v
            end = None
            if mode is WanVideoMode.flf2v:
                end = self.resolve_end_image()
                if end is None:
                    raise ValueError(_("Start/End mode needs an end layer (or comic panel)."))

        width = multiple_of(max(16, self.width), 16)
        height = multiple_of(max(16, self.height), 16)
        length = _normalize_length(self.length)

        params = WanVideoInput(
            mode=mode,
            positive=self.prompt.strip(),
            negative=self.negative_prompt.strip(),
            width=width,
            height=height,
            length=length,
            high_noise_model=high,
            low_noise_model=low,
            text_encoder=te,
            vae=vae,
            steps=max(2, int(self.steps)),
            split_step=max(1, int(self.split_step)),
            cfg=float(self.cfg),
            sampler=self.sampler or "lcm",
            scheduler=self.scheduler or "sgm_uniform",
            shift=float(self.shift),
            end_image=end,
            loras=self.loras_as_input(),
        )
        seed = model.seed if model.fixed_seed else workflow_module.generate_seed()
        workflow_input = prepare_wan_video(params, start, seed)

        mode_name = {WanVideoMode.t2v: "T2V", WanVideoMode.flf2v: "FLF2V"}.get(mode, "I2V")
        job_name = f"[Wan {mode_name}] {self.prompt.strip()[:80] or 'clip'}"
        job_params = JobParams(Bounds(0, 0, width, height), job_name)
        job_params.metadata["workspace"] = "animate"
        job_params.metadata["wan_mode"] = mode.name
        job_params.metadata["prompt"] = self.prompt
        job_params.seed = seed

        self._set_status(self._msg_generating(mode_name, length))
        model.clear_error()
        job = model.jobs.add(JobKind.animation, job_params)
        await model._enqueue_job(job, workflow_input)

        while job.state not in (JobState.finished, JobState.cancelled):
            await asyncio.sleep(0.05)

        if job.state is not JobState.finished or len(job.results) == 0:
            if not model.error:
                model.report_error(_("Wan video generation failed."))
            self._set_status(self._msg_failed(_("Generation")))
            return

        self._last_clip_frames = list(job.results)
        self._set_status(self._msg_ready(len(job.results)))

        if self.stitch_bridge and mode is WanVideoMode.i2v and end is not None:
            await self._run_bridge(self._last_clip_frames[-1], end, high, low, te, vae, seed + 1)

    async def _generate_stitch(self):
        if not self._last_clip_frames:
            raise ValueError(_("Generate a clip first, then stitch a bridge to the end frame."))
        end = self.resolve_end_image()
        if end is None:
            raise ValueError(_("Set an end layer / comic panel for the stitch bridge."))
        client = self._model._connection.client
        # Stitch is always FLF2V — resolve I2V-family models even if UI mode is T2V
        high, low, te, vae = self.resolve_models(client, AnimateMode.i2v)
        seed = (
            self._model.seed if self._model.fixed_seed else workflow_module.generate_seed()
        )
        await self._run_bridge(self._last_clip_frames[-1], end, high, low, te, vae, seed)

    def _ensure_last_clip(self) -> list[Image]:
        """Prefer in-memory clip; else take the latest finished animation job."""
        if self._last_clip_frames:
            return self._last_clip_frames
        model = self._model
        for job in reversed(list(model.jobs)):
            if job.kind is JobKind.animation and job.state is JobState.finished and job.results:
                self._last_clip_frames = list(job.results)
                return self._last_clip_frames
        if model.jobs.selection:
            job = model.jobs.find(model.jobs.selection[0][0])
            if job and job.kind is JobKind.animation and job.results:
                self._last_clip_frames = list(job.results)
                return self._last_clip_frames
        return []

    async def _generate_extend(self):
        """Continue the last clip with I2V from its final frame + current prompt/length."""
        if not self._ensure_last_clip():
            raise ValueError(_("Generate a clip first, then Extend from last."))
        model = self._model
        client = model._connection.client
        high, low, te, vae = self.resolve_models(client, AnimateMode.i2v)
        if not high or not low:
            raise ValueError(
                _("Wan 2.2 high/low noise models not found. Check Animate model dropdowns.")
            )

        start = self._last_clip_frames[-1]
        width = multiple_of(max(16, self.width), 16)
        height = multiple_of(max(16, self.height), 16)
        length = _normalize_length(self.length)
        seed = model.seed if model.fixed_seed else workflow_module.generate_seed()

        params = WanVideoInput(
            mode=WanVideoMode.i2v,
            positive=self.prompt.strip() or "continue the motion smoothly",
            negative=self.negative_prompt.strip(),
            width=width,
            height=height,
            length=length,
            high_noise_model=high,
            low_noise_model=low,
            text_encoder=te,
            vae=vae,
            steps=max(2, int(self.steps)),
            split_step=max(1, int(self.split_step)),
            cfg=float(self.cfg),
            sampler=self.sampler or "lcm",
            scheduler=self.scheduler or "sgm_uniform",
            shift=float(self.shift),
            end_image=None,
            loras=self.loras_as_input(),
        )
        workflow_input = prepare_wan_video(params, start, seed)
        job_params = JobParams(Bounds(0, 0, width, height), "[Wan Extend] continue")
        job_params.metadata["workspace"] = "animate"
        job_params.metadata["wan_mode"] = "extend"
        job_params.metadata["prompt"] = self.prompt
        job_params.seed = seed

        self._set_status(self._msg_generating(_("Extend (I2V)"), length))
        model.clear_error()
        job = model.jobs.add(JobKind.animation, job_params)
        await model._enqueue_job(job, workflow_input)
        while job.state not in (JobState.finished, JobState.cancelled):
            await asyncio.sleep(0.05)
        if job.state is not JobState.finished or not job.results:
            if not model.error:
                model.report_error(_("Wan extend failed."))
            self._set_status(self._msg_failed(_("Extend")))
            return

        # Drop first frame (duplicate of previous last frame) and append
        extension = list(job.results[1:]) if len(job.results) > 1 else list(job.results)
        self._last_clip_frames = self._last_clip_frames + extension
        self._set_status(self._msg_ready(len(self._last_clip_frames), kind="extended"))

    async def _run_bridge(
        self,
        start: Image,
        end: Image,
        high: str,
        low: str,
        te: str,
        vae: str,
        seed: int,
    ):
        """FLF2V bridge between two frames; concatenates onto last clip and imports."""
        model = self._model
        width = multiple_of(max(16, self.width), 16)
        height = multiple_of(max(16, self.height), 16)
        length = _normalize_length(min(self.length, 49))
        params = WanVideoInput(
            mode=WanVideoMode.flf2v,
            positive=self.prompt.strip() or "smooth transition between the two frames",
            negative=self.negative_prompt.strip(),
            width=width,
            height=height,
            length=length,
            high_noise_model=high,
            low_noise_model=low,
            text_encoder=te,
            vae=vae,
            steps=max(2, int(self.steps)),
            split_step=max(1, int(self.split_step)),
            cfg=float(self.cfg),
            sampler=self.sampler or "lcm",
            scheduler=self.scheduler or "sgm_uniform",
            shift=float(self.shift),
            end_image=end,
            loras=self.loras_as_input(),
        )
        workflow_input = prepare_wan_video(params, start, seed)
        job_params = JobParams(Bounds(0, 0, width, height), "[Wan Bridge] stitch")
        job_params.metadata["workspace"] = "animate"
        job_params.metadata["wan_mode"] = "bridge"
        self._set_status(self._msg_generating(_("Stitch (FLF2V)"), length))
        job = model.jobs.add(JobKind.animation, job_params)
        await model._enqueue_job(job, workflow_input)
        while job.state not in (JobState.finished, JobState.cancelled):
            await asyncio.sleep(0.05)
        if job.state is not JobState.finished or not job.results:
            self._set_status(self._msg_failed(_("Stitch")))
            return

        # Drop first bridge frame (duplicate of start) and append
        bridge = list(job.results[1:]) if len(job.results) > 1 else list(job.results)
        combined = self._last_clip_frames + bridge
        self._last_clip_frames = combined
        self._import_frames(combined, "[Wan Stitched]")
        self._set_status(self._msg_stitched(len(combined)))

    def apply_last_clip(self):
        if not self._last_clip_frames:
            # Fall back to selected animation job
            model = self._model
            if model.jobs.selection:
                job = model.jobs.find(model.jobs.selection[0][0])
                if job and job.kind is JobKind.animation and job.results:
                    model.apply_animation(job)
                    self._last_clip_frames = list(job.results)
                    self._set_status(self._msg_applied(len(job.results)))
                    return
            raise ValueError(_("No clip to apply. Generate a clip first, then Apply to Timeline."))
        n = len(self._last_clip_frames)
        self._import_frames(self._last_clip_frames, f"[Wan] {self.prompt.strip()[:80] or 'clip'}")
        self._set_status(self._msg_applied(n))

    def _import_frames(self, frames: list[Image], layer_name: str):
        model = self._model
        with TemporaryDirectory(prefix="wan_animate") as temp_dir:
            paths = []
            for i, image in enumerate(frames):
                filename = Path(temp_dir) / f"{i:03}.png"
                image.save(filename)
                paths.append(filename)
            model.document.import_animation(paths, model.document.playback_time_range[0])

        async def _rename():
            model.layers.active.name = layer_name

        eventloop.run(_rename())
