"""SeedVR2 image enhance workspace — diffusion upscale via ComfyUI-SeedVR2."""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, pyqtSignal

from .. import eventloop
from ..backend.api import SeedVRInput
from ..backend import workflow as workflow_module
from ..backend.workflow import estimate_seedvr_extent, prepare_seedvr
from ..image import Bounds, Extent
from ..localization import translate as _
from ..util import ensure
from .jobs import JobKind, JobParams
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from ..backend.client import Client
    from .model import DocumentModel

log = logging.getLogger(__name__)

DEFAULT_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE = "ema_vae_fp16.safetensors"

# Curated DiT / VAE names when the server list is empty (first-run download).
FALLBACK_DIT_MODELS = [
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_3b-Q4_K_M.gguf",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b-Q4_K_M.gguf",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
]
FALLBACK_VAE_MODELS = ["ema_vae_fp16.safetensors"]

COLOR_CORRECTION_OPTIONS = [
    "lab",
    "wavelet",
    "wavelet_adaptive",
    "hsv",
    "adain",
    "none",
]
ATTENTION_OPTIONS = ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"]
RESOLUTION_PRESETS = [
    ("720p (shortest edge)", 720),
    ("1080p (shortest edge)", 1080),
    ("1440p (shortest edge)", 1440),
    ("4K (shortest edge)", 2160),
]


class SeedVRWorkspace(QObject, ObservableProperties):
    """SeedVR2 single-image enhance / upscale (separate from ESRGAN Upscale)."""

    dit_model = Property(DEFAULT_DIT, persist=True)
    vae_model = Property(DEFAULT_VAE, persist=True)
    device = Property("cuda:0", persist=True)
    dit_offload_device = Property("cpu", persist=True)
    vae_offload_device = Property("none", persist=True)
    blocks_to_swap = Property(16, persist=True)
    swap_io_components = Property(True, persist=True)
    cache_model = Property(False, persist=True)
    attention_mode = Property("sdpa", persist=True)
    encode_tiled = Property(True, persist=True)
    decode_tiled = Property(True, persist=True)
    encode_tile_size = Property(512, persist=True)
    encode_tile_overlap = Property(64, persist=True)
    decode_tile_size = Property(512, persist=True)
    decode_tile_overlap = Property(64, persist=True)
    resolution = Property(1080, persist=True)
    max_resolution = Property(0, persist=True)
    color_correction = Property("lab", persist=True)
    input_noise_scale = Property(0.0, persist=True)
    latent_noise_scale = Property(0.0, persist=True)
    tensor_offload_device = Property("cpu", persist=True)
    enable_debug = Property(False, persist=True)
    use_active_layer = Property(False, persist=True)
    can_generate = Property(True)

    dit_model_changed = pyqtSignal(str)
    vae_model_changed = pyqtSignal(str)
    device_changed = pyqtSignal(str)
    dit_offload_device_changed = pyqtSignal(str)
    vae_offload_device_changed = pyqtSignal(str)
    blocks_to_swap_changed = pyqtSignal(int)
    swap_io_components_changed = pyqtSignal(bool)
    cache_model_changed = pyqtSignal(bool)
    attention_mode_changed = pyqtSignal(str)
    encode_tiled_changed = pyqtSignal(bool)
    decode_tiled_changed = pyqtSignal(bool)
    encode_tile_size_changed = pyqtSignal(int)
    encode_tile_overlap_changed = pyqtSignal(int)
    decode_tile_size_changed = pyqtSignal(int)
    decode_tile_overlap_changed = pyqtSignal(int)
    resolution_changed = pyqtSignal(int)
    max_resolution_changed = pyqtSignal(int)
    color_correction_changed = pyqtSignal(str)
    input_noise_scale_changed = pyqtSignal(float)
    latent_noise_scale_changed = pyqtSignal(float)
    tensor_offload_device_changed = pyqtSignal(str)
    enable_debug_changed = pyqtSignal(bool)
    use_active_layer_changed = pyqtSignal(bool)
    can_generate_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    target_extent_changed = pyqtSignal(Extent)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: DocumentModel):
        super().__init__()
        self._model = weakref.ref(model)
        self._in_progress = False
        self._status = ""

    @property
    def status(self) -> str:
        return self._status

    def _set_status(self, text: str):
        self._status = text
        self.status_changed.emit(text)

    def set_in_progress(self, in_progress: bool):
        self._in_progress = in_progress
        self.can_generate = not in_progress

    @staticmethod
    def pack_installed(client: Client) -> bool:
        nodes = client.models.node_inputs
        return (
            "SeedVR2VideoUpscaler" in nodes
            and "SeedVR2LoadDiTModel" in nodes
            and "SeedVR2LoadVAEModel" in nodes
        )

    def list_dit_models(self) -> list[str]:
        model = ensure(self._model())
        client = model._connection.client_if_connected
        if client is None or not self.pack_installed(client):
            return list(FALLBACK_DIT_MODELS)
        try:
            opts = client.models.node_inputs.options("SeedVR2LoadDiTModel", "model")
            return sorted(opts) if opts else list(FALLBACK_DIT_MODELS)
        except Exception:
            return list(FALLBACK_DIT_MODELS)

    def list_vae_models(self) -> list[str]:
        model = ensure(self._model())
        client = model._connection.client_if_connected
        if client is None or not self.pack_installed(client):
            return list(FALLBACK_VAE_MODELS)
        try:
            opts = client.models.node_inputs.options("SeedVR2LoadVAEModel", "model")
            return sorted(opts) if opts else list(FALLBACK_VAE_MODELS)
        except Exception:
            return list(FALLBACK_VAE_MODELS)

    def list_devices(self) -> list[str]:
        model = ensure(self._model())
        client = model._connection.client_if_connected
        if client is None:
            return ["cuda:0", "cpu"]
        try:
            opts = client.models.node_inputs.options("SeedVR2LoadDiTModel", "device")
            return list(opts) if opts else ["cuda:0"]
        except Exception:
            return ["cuda:0", "cpu"]

    def estimate_target(self) -> Extent:
        model = ensure(self._model())
        src = model.document.extent
        return estimate_seedvr_extent(src, self.resolution, self.max_resolution)

    def apply_preset_12gb(self):
        """Balanced defaults for ~12GB VRAM."""
        self.dit_model = DEFAULT_DIT
        self.vae_model = DEFAULT_VAE
        self.blocks_to_swap = 16
        self.swap_io_components = True
        self.dit_offload_device = "cpu"
        self.vae_offload_device = "none"
        self.encode_tiled = True
        self.decode_tiled = True
        self.encode_tile_size = 512
        self.decode_tile_size = 512
        self.resolution = 1080
        self.attention_mode = "sdpa"
        self.color_correction = "lab"
        self.tensor_offload_device = "cpu"
        self._set_status(_("Applied 12GB Balanced preset"))

    def apply_preset_quality(self):
        """Higher quality — needs more VRAM (often 16GB+)."""
        models = self.list_dit_models()
        preferred = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
        self.dit_model = preferred if preferred in models else DEFAULT_DIT
        self.blocks_to_swap = 8
        self.swap_io_components = False
        self.dit_offload_device = "cpu"
        self.encode_tiled = False
        self.decode_tiled = False
        self.resolution = 1440
        self._set_status(_("Applied Quality preset"))

    def apply_preset_low_vram(self):
        """GGUF + aggressive BlockSwap / tiling for ≤8–10GB."""
        models = self.list_dit_models()
        preferred = "seedvr2_ema_3b-Q4_K_M.gguf"
        self.dit_model = preferred if preferred in models else DEFAULT_DIT
        self.blocks_to_swap = 28
        self.swap_io_components = True
        self.dit_offload_device = "cpu"
        self.vae_offload_device = "cpu"
        self.encode_tiled = True
        self.decode_tiled = True
        self.encode_tile_size = 384
        self.decode_tile_size = 384
        self.resolution = 720
        self.max_resolution = 1536
        self._set_status(_("Applied Low VRAM preset"))

    def _to_input(self, seed: int) -> SeedVRInput:
        return SeedVRInput(
            dit_model=self.dit_model or DEFAULT_DIT,
            vae_model=self.vae_model or DEFAULT_VAE,
            device=self.device or "cuda:0",
            dit_offload_device=self.dit_offload_device,
            vae_offload_device=self.vae_offload_device,
            blocks_to_swap=self.blocks_to_swap,
            swap_io_components=self.swap_io_components,
            cache_model=self.cache_model,
            attention_mode=self.attention_mode,
            encode_tiled=self.encode_tiled,
            decode_tiled=self.decode_tiled,
            encode_tile_size=self.encode_tile_size,
            encode_tile_overlap=self.encode_tile_overlap,
            decode_tile_size=self.decode_tile_size,
            decode_tile_overlap=self.decode_tile_overlap,
            seed=seed,
            resolution=self.resolution,
            max_resolution=self.max_resolution,
            batch_size=1,
            color_correction=self.color_correction,
            input_noise_scale=self.input_noise_scale,
            latent_noise_scale=self.latent_noise_scale,
            tensor_offload_device=self.tensor_offload_device,
            enable_debug=self.enable_debug,
        )

    def enhance(self):
        eventloop.run(self._report_errors(self._enhance()))

    async def _report_errors(self, coro):
        try:
            await coro
        except Exception as e:
            log.exception("SeedVR enhance failed")
            model = ensure(self._model())
            model.report_error(str(e))
            self.set_in_progress(False)
            self._set_status(_("Error: {msg}").format(msg=str(e)))

    async def _enhance(self):
        model = ensure(self._model())
        ok, msg = model.document.check_color_mode()
        if not ok and msg:
            model.report_error(msg)
            return
        client = model._connection.client
        if not self.pack_installed(client):
            model.report_error(
                _("SeedVR2 custom nodes not found.")
                + " "
                + _(
                    "Install ComfyUI-SeedVR2_VideoUpscaler "
                    "(https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) "
                    "and restart ComfyUI."
                )
            )
            return

        if self.use_active_layer:
            layer = model.layers.active
            bounds = layer.compute_bounds()
            if bounds.is_zero:
                model.report_error(_("The active layer is empty."))
                return
            image = layer.get_pixels(bounds)
        else:
            extent = model.document.extent
            bounds = Bounds(0, 0, *extent)
            image = model.document.get_image(bounds)

        seed = model.seed if model.fixed_seed else workflow_module.generate_seed()
        params = self._to_input(seed)
        target = estimate_seedvr_extent(image.extent, params.resolution, params.max_resolution)
        workflow_input = prepare_seedvr(params, image)

        job_params = JobParams(
            Bounds(0, 0, *target),
            f"[SeedVR] {target.width}×{target.height}",
        )
        job_params.metadata["workspace"] = "seedvr"
        job_params.metadata["seedvr_dit"] = params.dit_model
        job_params.seed = seed

        self._set_status(
            _("SeedVR enhancing → {w}×{h}…").format(w=target.width, h=target.height)
        )
        model.clear_error()
        self.set_in_progress(True)
        self.target_extent_changed.emit(target)
        job = model.jobs.add(JobKind.upscaling, job_params)
        # Resize canvas to estimated target (same pattern as Upscale workspace)
        model.document.resize(target)
        await model._enqueue_job(job, workflow_input)
