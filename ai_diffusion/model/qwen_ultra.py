"""Qwen Image Edit 2.1 Ultra workspace — multi-edit / t2i / inpaint / outpaint."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, QUuid, pyqtSignal

from .. import eventloop
from ..backend.api import LoraInput, QwenUltraInput, QwenUltraMode
from ..backend import workflow as workflow_module
from ..backend.workflow import prepare_qwen_ultra
from ..document import SelectionModifiers
from ..image import Bounds, Extent, Image, multiple_of
from ..localization import translate as _
from .jobs import JobKind, JobParams
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from .model import DocumentModel

log = logging.getLogger(__name__)

MAX_REF_IMAGES = 6
MAX_QWEN_LORAS = 6

DEFAULT_UNET_CANDIDATES = (
    "qwen_image_2.1_int8_convrot.safetensors",
    "qwen_image_2.1",
    "qwen-image-2.1",
)
DEFAULT_TE_CANDIDATES = (
    "qwen3vl_8b_int8_convrot.safetensors",
    "qwen3vl_8b",
    "qwen3vl",
)
DEFAULT_VAE_CANDIDATES = (
    "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
    "texture_fix_vae_for_qwen_image_2.1",
    "qwen_image_vae",
    "qwen_image",
)

MODE_LABELS = {
    QwenUltraMode.multi_edit: "Multi-Edit",
    QwenUltraMode.t2i: "Text to Image",
    QwenUltraMode.inpaint: "Inpaint",
    QwenUltraMode.outpaint: "Outpaint",
}

SAMPLER_PRESETS = [
    ("Edit (euler / simple)", "euler", "simple", 25, 1.0),
    ("T2I (res_2s / beta)", "res_2s", "beta", 50, 1.0),
]

MEGAPIXEL_PRESETS = [
    ("0.5 MP", 0.5),
    ("1.0 MP (Ultra default)", 1.0),
    ("1.5 MP", 1.5),
    ("2.0 MP", 2.0),
]

T2I_SIZE_PRESETS = [
    ("1024×1024", 1024, 1024),
    ("1280×720", 1280, 720),
    ("720×1280", 720, 1280),
    ("1920×1080", 1920, 1080),
    ("1080×1920", 1080, 1920),
]


@dataclass
class QwenUltraLora:
    name: str = ""
    strength: float = 1.0

    def to_dict(self):
        return {"name": self.name, "strength": self.strength}

    @staticmethod
    def from_dict(data: dict):
        return QwenUltraLora(data.get("name", ""), float(data.get("strength", 1.0)))

    def as_input(self) -> LoraInput | None:
        if not self.name:
            return None
        return LoraInput(self.name, self.strength)


def _pick_name(available: list[str], *needles_groups: tuple[str, ...]) -> str:
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


class QwenUltraWorkspace(QObject, ObservableProperties):
    """Qwen Image Edit 2.1 Ultra — settings mirror the Aitrepreneur Ultra workflow."""

    mode = Property(QwenUltraMode.multi_edit, persist=True)
    prompt = Property("", persist=True)
    negative = Property("", persist=True)
    megapixels = Property(1.0, persist=True)
    width = Property(1024, persist=True)
    height = Property(1024, persist=True)
    resolution = Property(1024, persist=True)
    steps = Property(25, persist=True)
    cfg = Property(1.0, persist=True)
    sampler = Property("euler", persist=True)
    scheduler = Property("simple", persist=True)
    denoise = Property(1.0, persist=True)
    unet = Property("", persist=True)
    text_encoder = Property("", persist=True)
    vae = Property("", persist=True)
    # Layer UUIDs for refs (image_1..6). Empty = unused. For multi_edit, image_1 can be canvas.
    use_canvas_as_image1 = Property(True, persist=True)
    ref_layer_0 = Property("", persist=True)
    ref_layer_1 = Property("", persist=True)
    ref_layer_2 = Property("", persist=True)
    ref_layer_3 = Property("", persist=True)
    ref_layer_4 = Property("", persist=True)
    ref_layer_5 = Property("", persist=True)
    pad_left = Property(0, persist=True)
    pad_top = Property(0, persist=True)
    pad_right = Property(256, persist=True)
    pad_bottom = Property(0, persist=True)

    mode_changed = pyqtSignal(QwenUltraMode)
    prompt_changed = pyqtSignal(str)
    negative_changed = pyqtSignal(str)
    megapixels_changed = pyqtSignal(float)
    width_changed = pyqtSignal(int)
    height_changed = pyqtSignal(int)
    resolution_changed = pyqtSignal(int)
    steps_changed = pyqtSignal(int)
    cfg_changed = pyqtSignal(float)
    sampler_changed = pyqtSignal(str)
    scheduler_changed = pyqtSignal(str)
    denoise_changed = pyqtSignal(float)
    unet_changed = pyqtSignal(str)
    text_encoder_changed = pyqtSignal(str)
    vae_changed = pyqtSignal(str)
    use_canvas_as_image1_changed = pyqtSignal(bool)
    ref_layer_0_changed = pyqtSignal(str)
    ref_layer_1_changed = pyqtSignal(str)
    ref_layer_2_changed = pyqtSignal(str)
    ref_layer_3_changed = pyqtSignal(str)
    ref_layer_4_changed = pyqtSignal(str)
    ref_layer_5_changed = pyqtSignal(str)
    pad_left_changed = pyqtSignal(int)
    pad_top_changed = pyqtSignal(int)
    pad_right_changed = pyqtSignal(int)
    pad_bottom_changed = pyqtSignal(int)
    modified = pyqtSignal(QObject, str)
    status_changed = pyqtSignal(str)
    loras_changed = pyqtSignal()

    def __init__(self, model: DocumentModel):
        super().__init__()
        self._model = model
        self._loras: list[QwenUltraLora] = []
        self._status = ""

    @property
    def status(self) -> str:
        return self._status

    def _set_status(self, text: str):
        self._status = text
        self.status_changed.emit(text)

    def idle_hint(self) -> str:
        return _(
            "Qwen Image Edit 2.1 Ultra — Multi-Edit with refs, Text-to-Image, "
            "Inpaint (selection), or Outpaint. Needs ComfyUI ≥0.38 + qwen_image_2.1 models."
        )

    @property
    def loras(self) -> list[QwenUltraLora]:
        return self._loras

    def load_loras(self, data: list[dict]):
        self._loras = [QwenUltraLora.from_dict(d) for d in data if d.get("name")][:MAX_QWEN_LORAS]
        self.loras_changed.emit()

    def add_lora(self) -> bool:
        if len(self._loras) >= MAX_QWEN_LORAS:
            return False
        self._loras.append(QwenUltraLora())
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

    def apply_preset_edit(self):
        self.mode = QwenUltraMode.multi_edit
        self.steps = 25
        self.cfg = 1.0
        self.sampler = "euler"
        self.scheduler = "simple"
        self.denoise = 1.0
        self.megapixels = 1.0

    def apply_preset_t2i(self):
        self.mode = QwenUltraMode.t2i
        self.steps = 50
        self.cfg = 1.0
        self.sampler = "res_2s"
        self.scheduler = "beta"
        self.denoise = 1.0
        self.width = 1024
        self.height = 1024
        self.resolution = 1024

    def apply_preset_outpaint(self):
        self.mode = QwenUltraMode.outpaint
        self.steps = 25
        self.cfg = 1.0
        self.sampler = "euler"
        self.scheduler = "simple"
        self.prompt = (
            "Outpaint the image: replace the solid gray areas with a seamless "
            "continuation of the scene, keeping the existing picture unchanged."
        )
        if self.pad_right == 0 and self.pad_left == 0 and self.pad_top == 0 and self.pad_bottom == 0:
            self.pad_right = 256

    def resolve_models(self, client) -> tuple[str, str, str]:
        unet_opts: list[str] = []
        try:
            unet_opts = list(client.models.node_inputs.options("UNETLoader", "unet_name") or [])
        except Exception:
            unet_opts = []
        if not unet_opts:
            unet_opts = list(client.models.checkpoints.keys())

        te_opts: list[str] = []
        try:
            te_opts = list(client.models.node_inputs.options("CLIPLoader", "clip_name") or [])
        except Exception:
            te_opts = []

        vaes = list(client.models.vae) if client.models.vae else []
        try:
            if not vaes:
                vaes = list(client.models.node_inputs.options("VAELoader", "vae_name") or [])
        except Exception:
            pass

        unet = self.unet if self.unet in unet_opts else ""
        if not unet:
            unet = _pick_name(unet_opts, DEFAULT_UNET_CANDIDATES, ("qwen_image_2.1", "qwen_image"))

        te = self.text_encoder if self.text_encoder in te_opts else ""
        if not te:
            te = _pick_name(te_opts, DEFAULT_TE_CANDIDATES, ("qwen3vl", "qwen"))

        vae = self.vae if self.vae in vaes else ""
        if not vae:
            vae = _pick_name(vaes, DEFAULT_VAE_CANDIDATES, ("texture_fix", "qwen_image", "qwen"))

        return unet, te, vae

    def _ref_layer_ids(self) -> list[str]:
        return [
            self.ref_layer_0,
            self.ref_layer_1,
            self.ref_layer_2,
            self.ref_layer_3,
            self.ref_layer_4,
            self.ref_layer_5,
        ]

    def resolve_images(self) -> tuple[list[Image], Image | None]:
        """Return (images for encode, optional inpaint mask)."""
        model = self._model
        extent = model.document.extent
        bounds = Bounds(0, 0, *extent)
        images: list[Image] = []
        mask: Image | None = None

        if self.mode is QwenUltraMode.t2i:
            return [], None

        if self.use_canvas_as_image1 or self.mode in (QwenUltraMode.inpaint, QwenUltraMode.outpaint):
            try:
                images.append(model.document.get_image(bounds))
            except Exception as e:
                log.warning("Qwen Ultra: failed to capture canvas: %s", e)

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
                log.warning("Qwen Ultra: failed to read layer %s: %s", layer_id, e)
                continue
            if len(images) >= MAX_REF_IMAGES:
                break

        if self.mode is QwenUltraMode.inpaint:
            try:
                mask_obj, _orig = model.document.create_mask_from_selection(
                    SelectionModifiers(pad_offset_px=8, multiple=8)
                )
                if mask_obj is not None:
                    mask = mask_obj.to_image(Extent(*extent))
            except Exception as e:
                log.warning("Qwen Ultra: no selection mask: %s", e)

        return images[:11], mask

    def _collect_loras(self, client) -> list[LoraInput]:
        result: list[LoraInput] = []
        lora_names = list(client.models.loras) if client.models.loras else []
        used: set[str] = set()
        for ql in self._loras:
            inp = ql.as_input()
            if not inp:
                continue
            key = inp.name.lower().replace("\\", "/")
            if key in used:
                continue
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

    def generate(self):
        eventloop.run(self._report_errors(self._generate()))

    async def _report_errors(self, coro):
        try:
            await coro
        except Exception as e:
            log.exception("Qwen Ultra generation failed")
            self._model.report_error(str(e))
            self._set_status(_("Error: {msg}").format(msg=str(e)))

    async def _generate(self):
        model = self._model
        ok, msg = model.document.check_color_mode()
        if not ok and msg:
            raise ValueError(msg)

        client = model._connection.client
        nodes = client.models.node_inputs
        if (
            "TextEncodeQwenImage21" not in nodes
            and "TextEncodeQwenImageEditPlus" not in nodes
            and "TextEncodeQwenImageEdit" not in nodes
        ):
            raise ValueError(
                _(
                    "Qwen Image Edit nodes not found. Update ComfyUI to 0.38+ "
                    "(needs TextEncodeQwenImage21) or install Qwen Edit nodes."
                )
            )

        unet, te, vae = self.resolve_models(client)
        if not unet or not te or not vae:
            missing = []
            if not unet:
                missing.append("qwen_image_2.1 diffusion (diffusion_models)")
            if not te:
                missing.append("qwen3vl text encoder")
            if not vae:
                missing.append("texture_fix / qwen VAE")
            raise ValueError(
                _("Missing Qwen Ultra models: {list}").format(list=", ".join(missing))
            )

        images, mask = self.resolve_images()
        if self.mode is not QwenUltraMode.t2i and not images:
            raise ValueError(
                _("Add a canvas image and/or reference layers before generating.")
            )
        if self.mode is QwenUltraMode.inpaint and mask is None:
            raise ValueError(_("Inpaint mode needs a selection on the canvas."))
        if self.mode is QwenUltraMode.t2i and not self.prompt.strip():
            raise ValueError(_("Text-to-Image needs a prompt."))

        width = multiple_of(max(32, int(self.width)), 32)
        height = multiple_of(max(32, int(self.height)), 32)
        if images and self.mode is not QwenUltraMode.t2i:
            width = images[0].width
            height = images[0].height

        params = QwenUltraInput(
            mode=self.mode,
            prompt=self.prompt.strip(),
            negative=self.negative.strip(),
            width=width,
            height=height,
            megapixels=float(self.megapixels),
            resolution=int(self.resolution),
            unet=unet,
            text_encoder=te,
            vae=vae,
            steps=max(1, int(self.steps)),
            cfg=float(self.cfg),
            sampler=self.sampler or "euler",
            scheduler=self.scheduler or "simple",
            denoise=float(self.denoise),
            images=images,
            mask=mask,
            pad_left=int(self.pad_left),
            pad_top=int(self.pad_top),
            pad_right=int(self.pad_right),
            pad_bottom=int(self.pad_bottom),
            loras=self._collect_loras(client),
        )
        seed = model.seed if model.fixed_seed else workflow_module.generate_seed()
        workflow_input = prepare_qwen_ultra(params, seed)

        mode_name = MODE_LABELS.get(self.mode, "Qwen")
        job_name = f"[Qwen Ultra · {mode_name}] {self.prompt.strip()[:80] or 'edit'}"
        job_params = JobParams(Bounds(0, 0, width, height), job_name)
        job_params.metadata["workspace"] = "qwen_ultra"
        job_params.metadata["prompt"] = self.prompt
        job_params.seed = seed

        self._set_status(_("Generating Qwen Ultra · {mode}…").format(mode=mode_name))
        model.clear_error()
        job = model.jobs.add(JobKind.diffusion, job_params)
        await model._enqueue_job(job, workflow_input)
        self._set_status(
            _("Queued — {mode}. Apply from History when done.").format(mode=mode_name)
        )
