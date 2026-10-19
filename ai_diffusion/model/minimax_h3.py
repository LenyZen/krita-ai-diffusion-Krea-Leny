"""MiniMax H3 Reference→Video workspace — fast multi-ref clips with optional audio."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
import shutil
import subprocess
import textwrap

from PyQt5.QtCore import QObject, QUrl, QUuid, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog

from .. import eventloop
from ..backend.api import LoraInput, MiniMaxH3Input
from ..backend import workflow as workflow_module
from ..backend.workflow import prepare_minimax_h3
from ..image import Bounds, Image, multiple_of
from ..localization import translate as _
from ..settings import settings
from ..util import user_data_dir
from .jobs import JobKind, JobParams, JobState
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from .model import DocumentModel

log = logging.getLogger(__name__)

MAX_REF_IMAGES = 6
MAX_H3_LORAS = 6


@dataclass
class H3Lora:
    """Style / character LoRA applied to the H3 REF2VA UNET."""

    name: str = ""
    strength: float = 1.0

    def to_dict(self):
        return {"name": self.name, "strength": self.strength}

    @staticmethod
    def from_dict(data: dict):
        return H3Lora(data.get("name", ""), float(data.get("strength", 1.0)))

    def as_input(self) -> LoraInput | None:
        if not self.name:
            return None
        return LoraInput(self.name, self.strength)

# Preferred filenames (Comfy-Org MiniMax-H3 + common turbo packs)
DEFAULT_UNET_CANDIDATES = (
    "DasiwaMinimaxH3_dasiwaREF2VAHybridV1.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va",
)
DEFAULT_TE_CANDIDATES = (
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "qwen3vl_32b_minimax_h3_int8_convrot",
    "qwen3vl_32b_h3",
    "qwen3vl_32b_minimax_h3",
    "minimax_h3",
)
DEFAULT_VIDEO_VAE_CANDIDATES = (
    "minimax_h3_video_vae_int8_convrot.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_video_vae",
)
DEFAULT_AUDIO_VAE_CANDIDATES = (
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_audio_vae",
)
DEFAULT_TURBO_LORAS = (
    "Minimax H3/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors",
    "Minimax H3/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
    "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors",
    "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
)

# (label, megapixels) — resolutions aligned to 32 from the extreme-fast workflow notes
MEGAPIXEL_PRESETS = [
    ("0.4 MP (draft)", 0.4),
    ("0.6 MP", 0.6),
    ("0.8 MP", 0.8),
    ("0.98 MP (~1344×768)", 0.98),
    ("1.2 MP", 1.2),
    ("1.5 MP", 1.5),
]

ASPECT_PRESETS = [
    ("16:9", 16, 9),
    ("9:16", 9, 16),
    ("1:1", 1, 1),
]

DURATION_PRESETS = [
    ("~3s", 3.0),
    ("~5s", 5.0),
    ("~8s", 8.0),
    ("~10s", 10.0),
    ("~15s", 15.0),
]


def size_from_megapixels(mp: float, aspect_w: int, aspect_h: int) -> tuple[int, int]:
    """Pick width×height near `mp` megapixels, multiple of 32, matching aspect."""
    mp = max(0.1, float(mp))
    ar = aspect_w / max(1, aspect_h)
    # pixels = w * h = w * (w / ar) = w² / ar  →  w = sqrt(pixels * ar)
    pixels = mp * 1_000_000
    width = int(round(math.sqrt(pixels * ar)))
    height = int(round(width / ar))
    width = multiple_of(max(32, width), 32)
    height = multiple_of(max(32, height), 32)
    return width, height


def length_from_duration(seconds: float, fps: int = 24) -> int:
    """Frame count at fps for the given duration (H3 trained ~124–362)."""
    fps = max(1, int(fps))
    frames = int(round(float(seconds) * fps))
    return max(24, min(3600, frames))


def _pick_name(available: list[str], *needles_groups: tuple[str, ...]) -> str:
    """Prefer candidates in order: for each needle, return the first available match."""
    lower_map = {n: n.lower().replace("\\", "/") for n in available}
    for group in needles_groups:
        for needle in group:
            if not needle:
                continue
            key = needle.lower().replace("\\", "/")
            for name, low in lower_map.items():
                if key in low or Path(key).name in low:
                    return name
    return ""


def _find_turbo_loras(available: list[str]) -> list[LoraInput]:
    found: list[LoraInput] = []
    lower_map = {n: n.lower().replace("\\", "/") for n in available}
    used: set[str] = set()
    for preferred in DEFAULT_TURBO_LORAS:
        key = preferred.lower().replace("\\", "/")
        for name, low in lower_map.items():
            if name in used:
                continue
            if key in low or Path(key).name in low:
                found.append(LoraInput(name, 1.0))
                used.add(name)
                break
        if len(found) >= 2:
            break
    # Prefer one SLA + one turbo_v4 if both classes exist
    return found[:2]


class MiniMaxH3Workspace(QObject, ObservableProperties):
    """Reference→Video clips via MiniMax H3 (native Comfy nodes)."""

    prompt = Property("", persist=True)
    megapixels = Property(0.4, persist=True)
    aspect_w = Property(16, persist=True)
    aspect_h = Property(9, persist=True)
    duration_sec = Property(5.0, persist=True)
    fps = Property(24, persist=True)
    steps = Property(5, persist=True)
    cfg = Property(1.0, persist=True)
    sampler = Property("res_multistep", persist=True)
    scheduler = Property("simple", persist=True)
    shift_video = Property(12.0, persist=True)
    shift_audio = Property(3.0, persist=True)
    turbo = Property(True, persist=True)
    unet = Property("", persist=True)
    text_encoder = Property("", persist=True)
    video_vae = Property("", persist=True)
    audio_vae = Property("", persist=True)
    # Layer UUIDs for up to 6 reference stills (empty = unused)
    ref_layer_0 = Property("", persist=True)
    ref_layer_1 = Property("", persist=True)
    ref_layer_2 = Property("", persist=True)
    ref_layer_3 = Property("", persist=True)
    ref_layer_4 = Property("", persist=True)
    ref_layer_5 = Property("", persist=True)
    # Optional absolute path to a local audio file (uploaded to Comfy input on generate)
    audio_path = Property("", persist=True)

    prompt_changed = pyqtSignal(str)
    megapixels_changed = pyqtSignal(float)
    aspect_w_changed = pyqtSignal(int)
    aspect_h_changed = pyqtSignal(int)
    duration_sec_changed = pyqtSignal(float)
    fps_changed = pyqtSignal(int)
    steps_changed = pyqtSignal(int)
    cfg_changed = pyqtSignal(float)
    sampler_changed = pyqtSignal(str)
    scheduler_changed = pyqtSignal(str)
    shift_video_changed = pyqtSignal(float)
    shift_audio_changed = pyqtSignal(float)
    turbo_changed = pyqtSignal(bool)
    unet_changed = pyqtSignal(str)
    text_encoder_changed = pyqtSignal(str)
    video_vae_changed = pyqtSignal(str)
    audio_vae_changed = pyqtSignal(str)
    ref_layer_0_changed = pyqtSignal(str)
    ref_layer_1_changed = pyqtSignal(str)
    ref_layer_2_changed = pyqtSignal(str)
    ref_layer_3_changed = pyqtSignal(str)
    ref_layer_4_changed = pyqtSignal(str)
    ref_layer_5_changed = pyqtSignal(str)
    audio_path_changed = pyqtSignal(str)
    modified = pyqtSignal(QObject, str)
    status_changed = pyqtSignal(str)
    loras_changed = pyqtSignal()

    def __init__(self, model: DocumentModel):
        super().__init__()
        self._model = model
        self._last_clip_frames: list[Image] = []
        self._loras: list[H3Lora] = []
        self._last_audio_path = ""
        self._last_video_path = ""
        self._status = ""

    @property
    def loras(self) -> list[H3Lora]:
        return self._loras

    def load_loras(self, data: list[dict]):
        self._loras = [H3Lora.from_dict(d) for d in data if d.get("name")][:MAX_H3_LORAS]
        self.loras_changed.emit()

    def set_loras(self, loras: list[H3Lora]):
        self._loras = [l for l in loras if l.name][:MAX_H3_LORAS]
        self.loras_changed.emit()
        self.modified.emit(self, "loras")

    def add_lora(self) -> bool:
        if len(self._loras) >= MAX_H3_LORAS:
            return False
        self._loras.append(H3Lora())
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

    @property
    def last_audio_path(self) -> str:
        return self._last_audio_path

    @property
    def last_video_path(self) -> str:
        return self._last_video_path

    def open_last_audio(self):
        path = (self._last_audio_path or "").strip()
        if not path or not Path(path).is_file():
            self._model.report_error(
                _(
                    "No H3 audio file yet. Generate a clip first — soundtrack is saved "
                    "beside the plugin data (Krita Timeline is frames-only)."
                )
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def export_mp4(self):
        """Mux last frames + soundtrack into one MP4 (or copy Comfy mux if already downloaded)."""
        eventloop.run(self._report_errors(self._export_mp4()))

    async def _export_mp4(self):
        dest_hint = Path.home() / "Videos" / "MiniMaxH3"
        dest_hint.mkdir(parents=True, exist_ok=True)
        suggested = dest_hint / f"h3_clip_{len(self._last_clip_frames) or 0}f.mp4"
        existing = (self._last_video_path or "").strip()
        if existing and Path(existing).is_file():
            suggested = dest_hint / Path(existing).name

        path, _filter = QFileDialog.getSaveFileName(
            None,
            _("Export MiniMax H3 video"),
            str(suggested),
            _("MP4 video") + " (*.mp4);;" + _("All files") + " (*)",
        )
        if not path:
            return
        dest = Path(path)
        if dest.suffix.lower() != ".mp4":
            dest = dest.with_suffix(".mp4")

        # Prefer already-muxed Comfy/VHS download
        if existing and Path(existing).is_file():
            shutil.copy2(existing, dest)
            self._last_video_path = str(dest)
            self._set_status(_("Exported video: {path}").format(path=str(dest)))
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(dest.parent)))
            return

        frames = self._ensure_last_clip()
        audio = Path(self._last_audio_path) if self._last_audio_path else None
        if not frames:
            raise ValueError(_("No H3 frames to export — generate a clip first."))
        if audio is None or not audio.is_file():
            raise ValueError(
                _(
                    "No soundtrack file found. Generate again so the MP3 is saved, "
                    "then Export MP4."
                )
            )

        self._set_status(_("Muxing frames + soundtrack to MP4…"))
        out = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _mux_frames_audio_mp4(frames, audio, dest, fps=max(1, int(self.fps))),
        )
        self._last_video_path = str(out)
        self._set_status(_("Exported video: {path}").format(path=str(out)))
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out.parent)))


    @property
    def status(self) -> str:
        return self._status

    def _set_status(self, text: str):
        self._status = text
        self.status_changed.emit(text)

    def idle_hint(self) -> str:
        return _(
            "MiniMax H3 Reference→Video — pick 1–6 reference layers, write a prompt "
            "(use <Picture 1>… tags or leave them out and we’ll add them), then Generate. "
            "Turbo = 5-step LoRA pack. Needs ComfyUI ≥0.30 + H3 models."
        )

    def apply_preset_turbo(self):
        self.turbo = True
        self.steps = 5
        self.cfg = 1.0
        self.sampler = "res_multistep"
        self.scheduler = "simple"
        self.shift_video = 12.0
        self.shift_audio = 3.0
        self.megapixels = 0.4
        self.duration_sec = 5.0

    def apply_preset_quality(self):
        self.turbo = False
        self.steps = 20
        self.cfg = 1.0
        self.sampler = "res_multistep"
        self.scheduler = "simple"
        self.megapixels = 0.98
        self.duration_sec = 5.0

    def resolve_models(self, client) -> tuple[str, str, str, str]:
        # H3 UNETs live under diffusion_models. ETN often reports base_model=unknown for
        # them, which drops them from client.models.checkpoints — always ask UNETLoader.
        unet_opts: list[str] = []
        try:
            unet_opts = list(client.models.node_inputs.options("UNETLoader", "unet_name") or [])
        except Exception:
            unet_opts = []
        if not unet_opts:
            unet_opts = list(client.models.checkpoints.keys())

        vaes = list(client.models.vae) if client.models.vae else []
        try:
            if not vaes:
                vaes = list(client.models.node_inputs.options("VAELoader", "vae_name") or [])
        except Exception:
            pass

        te_opts: list[str] = []
        try:
            te_opts = list(client.models.node_inputs.options("CLIPLoader", "clip_name") or [])
        except Exception:
            te_opts = []
        if not te_opts:
            te_opts = list(client.models.checkpoints.keys())

        unet = self.unet if self.unet in unet_opts else ""
        if not unet:
            unet = _pick_name(
                unet_opts,
                DEFAULT_UNET_CANDIDATES,
                ("dasiwa", "ref2va", "minimax_h3", "minimax"),
            )

        te = self.text_encoder if self.text_encoder in te_opts else ""
        if not te:
            te = _pick_name(te_opts, DEFAULT_TE_CANDIDATES, ("qwen3vl_32b", "minimax", "h3"))

        video_vae = self.video_vae if self.video_vae in vaes else ""
        if not video_vae:
            video_vae = _pick_name(
                vaes, DEFAULT_VIDEO_VAE_CANDIDATES, ("minimax_h3_video", "h3_video")
            )

        audio_vae = self.audio_vae if self.audio_vae in vaes else ""
        if not audio_vae:
            audio_vae = _pick_name(
                vaes, DEFAULT_AUDIO_VAE_CANDIDATES, ("minimax_h3_audio", "h3_audio")
            )

        return unet, te, video_vae, audio_vae

    def _ref_layer_ids(self) -> list[str]:
        return [
            self.ref_layer_0,
            self.ref_layer_1,
            self.ref_layer_2,
            self.ref_layer_3,
            self.ref_layer_4,
            self.ref_layer_5,
        ]

    def resolve_ref_images(self) -> list[Image]:
        model = self._model
        images: list[Image] = []
        bounds = Bounds(0, 0, *model.document.extent)
        for layer_id in self._ref_layer_ids():
            if not layer_id:
                continue
            try:
                layer = model.layers.find(QUuid(layer_id))
            except Exception:
                layer = None
            if layer is None:
                continue
            try:
                images.append(layer.get_pixels(bounds))
            except Exception as e:
                log.warning("H3: failed to read layer %s: %s", layer_id, e)
                continue
            if len(images) >= MAX_REF_IMAGES:
                break
        return images

    def generate(self):
        eventloop.run(self._report_errors(self._generate()))

    async def _report_errors(self, coro):
        try:
            await coro
        except Exception as e:
            log.exception("MiniMax H3 generation failed")
            self._model.report_error(str(e))
            self._set_status(_("Error: {msg}").format(msg=str(e)))

    async def _generate(self):
        model = self._model
        ok, msg = model.document.check_color_mode()
        if not ok and msg:
            raise ValueError(msg)

        client = model._connection.client
        # Require native H3 node
        if "MiniMaxH3ReferenceToVideo" not in client.models.node_inputs:
            raise ValueError(
                _(
                    "MiniMax H3 nodes not found. Update ComfyUI to 0.30+ "
                    "(needs MiniMaxH3ReferenceToVideo)."
                )
            )

        unet, te, video_vae, audio_vae = self.resolve_models(client)
        if not unet or not te or not video_vae or not audio_vae:
            missing = []
            if not unet:
                missing.append("REF2VA UNET (diffusion_models)")
            if not te:
                missing.append("minimax CLIP / text encoder")
            if not video_vae:
                missing.append("minimax_h3 video VAE")
            if not audio_vae:
                missing.append("minimax_h3 audio VAE")
            raise ValueError(
                _("Missing MiniMax H3 models: {list}").format(list=", ".join(missing))
                + "\n"
                + _("See workflows/README_MINIMAX_H3.md for download links.")
            )

        refs = self.resolve_ref_images()
        if not refs and not self.prompt.strip():
            raise ValueError(_("Add at least one reference layer or a prompt."))

        width, height = size_from_megapixels(self.megapixels, self.aspect_w, self.aspect_h)
        length = length_from_duration(self.duration_sec, self.fps)

        loras = self._collect_loras(client)

        audio_filename = ""
        audio_src = (self.audio_path or "").strip()
        if audio_src:
            audio_file = Path(audio_src)
            if not audio_file.is_file():
                raise ValueError(_("Audio file not found") + f":\n{audio_src}")
            self._set_status(_("Uploading audio to ComfyUI…"))
            if not hasattr(client, "upload_input_file"):
                raise ValueError(
                    _("Audio upload needs a local ComfyUI connection (not available on this server).")
                )
            audio_filename = await client.upload_input_file(audio_file)
            log.info("H3 audio staged as %s", audio_filename)

        params = MiniMaxH3Input(
            prompt=self.prompt.strip(),
            width=width,
            height=height,
            length=length,
            fps=int(self.fps),
            unet=unet,
            text_encoder=te,
            video_vae=video_vae,
            audio_vae=audio_vae,
            steps=max(1, int(self.steps)),
            cfg=float(self.cfg),
            sampler=self.sampler or "res_multistep",
            scheduler=self.scheduler or "simple",
            shift_video=float(self.shift_video),
            shift_audio=float(self.shift_audio),
            ref_images=refs,
            audio_filename=audio_filename,
            loras=loras,
        )
        seed = model.seed if model.fixed_seed else workflow_module.generate_seed()
        workflow_input = prepare_minimax_h3(params, seed)

        job_name = f"[H3 R2V] {self.prompt.strip()[:80] or 'clip'}"
        job_params = JobParams(Bounds(0, 0, width, height), job_name)
        job_params.metadata["workspace"] = "minimax_h3"
        job_params.metadata["prompt"] = self.prompt
        job_params.seed = seed

        self._set_status(
            _("Generating MiniMax H3 — {n} frames @ {w}×{h}…").format(
                n=length, w=width, h=height
            )
        )
        model.clear_error()
        job = model.jobs.add(JobKind.animation, job_params)
        await model._enqueue_job(job, workflow_input)

        while job.state not in (JobState.finished, JobState.cancelled):
            await asyncio.sleep(0.05)

        if job.state is not JobState.finished or len(job.results) == 0:
            if not model.error:
                model.report_error(_("MiniMax H3 generation failed."))
            self._set_status(_("Generation failed — check server.log / Connection."))
            return

        self._last_clip_frames = list(job.results)
        audio_note = ""
        video_note = ""
        dest_dir = Path(user_data_dir) / "minimax_h3"
        try:
            saved = await self._pull_generated_audio(client, job.id)
            if saved:
                self._last_audio_path = str(saved)
                audio_note = " " + _("Soundtrack: {name}").format(name=saved.name)
        except Exception as e:
            log.warning("H3: could not download generated audio: %s", e)
            audio_note = " " + _("(soundtrack download failed)")

        try:
            if hasattr(client, "download_history_video"):
                dest_dir.mkdir(parents=True, exist_ok=True)
                vid = await client.download_history_video(job.id, dest_dir)
                if vid:
                    self._last_video_path = str(vid)
                    video_note = " " + _("MP4 ready — use Export MP4.")
        except Exception as e:
            log.warning("H3: could not download muxed video: %s", e)

        self._set_status(
            _("Ready — {n} frames in memory.").format(n=len(job.results))
            + audio_note
            + video_note
            + " "
            + _(
                "Apply to Timeline = frames only. Export MP4 merges frames + soundtrack."
            )
        )

    def _collect_loras(self, client) -> list[LoraInput]:
        """Turbo pack first (if enabled), then user LoRAs — skip duplicates."""
        result: list[LoraInput] = []
        used: set[str] = set()
        lora_names = list(client.models.loras) if client.models.loras else []
        if self.turbo:
            for lora in _find_turbo_loras(lora_names):
                key = lora.name.lower().replace("\\", "/")
                if key in used:
                    continue
                result.append(lora)
                used.add(key)
            if self.turbo and not result:
                log.warning(
                    "H3 turbo enabled but no turbo LoRAs found — continuing with user LoRAs only"
                )
        for h3l in self._loras:
            inp = h3l.as_input()
            if not inp:
                continue
            key = inp.name.lower().replace("\\", "/")
            if key in used:
                continue
            # Resolve to a server-visible name when possible
            match = next(
                (
                    n
                    for n in lora_names
                    if n.lower().replace("\\", "/") == key
                    or Path(n).name.lower() == Path(inp.name).name.lower()
                ),
                None,
            )
            if match:
                result.append(LoraInput(match, inp.strength))
                used.add(match.lower().replace("\\", "/"))
            else:
                result.append(inp)
                used.add(key)
        return result

    async def _pull_generated_audio(self, client, job_id: str) -> Path | None:
        """Download SaveAudioMP3 output from Comfy history into plugin user data."""
        if not hasattr(client, "download_history_audio"):
            return None
        dest_dir = Path(user_data_dir) / "minimax_h3"
        dest_dir.mkdir(parents=True, exist_ok=True)
        return await client.download_history_audio(job_id, dest_dir)

    def apply_last_clip(self):
        frames = self._ensure_last_clip()
        if not frames:
            self._model.report_error(_("No H3 clip to apply — generate first."))
            return
        model = self._model
        with TemporaryDirectory(prefix="minimax_h3") as temp_dir:
            paths = []
            for i, image in enumerate(frames):
                filename = Path(temp_dir) / f"{i:03}.png"
                image.save(filename)
                paths.append(filename)
            model.document.import_animation(paths, model.document.playback_time_range[0])
        self._set_status(
            _("Applied {n} frames to the Timeline.").format(n=len(frames))
            + " "
            + _(
                "Krita Animation has no generated soundtrack — use Open soundtrack… "
                "for the H3 audio file, or attach audio via Animation docker."
            )
        )

    def _ensure_last_clip(self) -> list[Image]:
        if self._last_clip_frames:
            return self._last_clip_frames
        model = self._model
        for job in reversed(list(model.jobs)):
            if (
                job.kind is JobKind.animation
                and job.state is JobState.finished
                and job.results
                and job.params.metadata.get("workspace") == "minimax_h3"
            ):
                self._last_clip_frames = list(job.results)
                return self._last_clip_frames
        return []


def _find_mux_python() -> Path | None:
    """Prefer ComfyUI venv python (has PyAV) for local frame+audio mux."""
    candidates: list[Path] = []
    server = Path(getattr(settings, "server_path", "") or "")
    if server:
        candidates.extend(
            [
                server / "venv" / "Scripts" / "python.exe",
                server / "python_embeded" / "python.exe",
                server.parent / "venv" / "Scripts" / "python.exe",
            ]
        )
    candidates.extend(
        [
            Path(r"F:\ComfyUI_V116\ComfyUI\venv\Scripts\python.exe"),
            Path(r"E:\ComfyforKrita\ComfyUI\venv\Scripts\python.exe"),
        ]
    )
    which = shutil.which("python")
    if which:
        candidates.append(Path(which))
    for p in candidates:
        if p.is_file():
            try:
                r = subprocess.run(
                    [str(p), "-c", "import av"],
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                if r.returncode == 0:
                    return p
            except Exception:
                continue
    return None


def _mux_frames_audio_mp4(
    frames: list[Image], audio_path: Path, dest: Path, fps: int = 24
) -> Path:
    """Write PNG sequence + mux with soundtrack via Comfy's PyAV (or ffmpeg if present)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(fps))

    with TemporaryDirectory(prefix="h3_mux_") as temp:
        temp_path = Path(temp)
        pattern = temp_path / "frame_%05d.png"
        for i, image in enumerate(frames):
            image.save(temp_path / f"frame_{i:05d}.png")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(pattern),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(dest),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and dest.is_file():
                return dest
            log.warning("ffmpeg mux failed: %s", r.stderr[-500:] if r.stderr else r.returncode)

        py = _find_mux_python()
        if not py:
            raise RuntimeError(
                _(
                    "Could not mux MP4: install ffmpeg on PATH, or keep ComfyUI’s venv "
                    "(PyAV) available. Soundtrack is still at: {audio}"
                ).format(audio=str(audio_path))
            )

        script = textwrap.dedent(
            f"""
            import sys
            from pathlib import Path
            import av
            from fractions import Fraction

            frames_dir = Path(r"{temp_path}")
            audio_path = Path(r"{audio_path}")
            dest = Path(r"{dest}")
            fps = {fps}

            paths = sorted(frames_dir.glob("frame_*.png"))
            if not paths:
                raise SystemExit("no frames")

            container = av.open(str(dest), mode="w")
            stream = container.add_stream("libx264", rate=fps)
            stream.pix_fmt = "yuv420p"
            first = av.open(str(paths[0]))
            f0 = next(first.decode(video=0))
            stream.width = f0.width
            stream.height = f0.height
            first.close()

            for p in paths:
                packet_img = av.open(str(p))
                for frame in packet_img.decode(video=0):
                    frame = frame.reformat(format="yuv420p")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                packet_img.close()
            for packet in stream.encode():
                container.mux(packet)

            a_in = av.open(str(audio_path))
            a_stream_in = a_in.streams.audio[0]
            a_out = container.add_stream("aac", rate=a_stream_in.rate)
            for packet in a_in.demux(a_stream_in):
                for frame in packet.decode():
                    frame.pts = None
                    for out_packet in a_out.encode(frame):
                        container.mux(out_packet)
            for out_packet in a_out.encode(None):
                container.mux(out_packet)
            a_in.close()
            container.close()
            print("OK", dest)
            """
        )
        r = subprocess.run(
            [str(py), "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0 or not dest.is_file():
            raise RuntimeError(
                _("MP4 mux failed")
                + f":\n{(r.stderr or r.stdout or '')[-800:]}"
            )
        return dest
