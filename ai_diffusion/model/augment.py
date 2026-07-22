from __future__ import annotations

import asyncio
import uuid
import weakref
from dataclasses import dataclass

from PyQt5.QtCore import QObject, QUuid, pyqtSignal
from PyQt5.QtGui import QColor, QImage, qAlpha, qBlue, qGreen, qRed, qRgba

from .. import eventloop, util
from ..backend import workflow
from ..backend.api import ConditioningInput, FillMode, InpaintMode, InpaintParams, LoraInput, WorkflowKind
from ..backend.resources import Arch
from ..backend.resolution import compute_augment_context_bounds, compute_relative_bounds
from ..files import FileLibrary
from ..image import Bounds, Extent, Image, ImageCollection, Mask
from ..localization import translate as _
from ..settings import settings
from ..util import client_logger as log, unique
from .jobs import JobKind, JobParams, JobRegion, JobState
from ..layer import Layer, LayerType
from .properties import ObservableProperties, Property

OVERLAY_GROUP_NAME = "[AI Augment]"
MAX_REGIONS = 10
MIN_REGION_SIZE = 64

OVERLAY_COLORS = [
    (255, 80, 80),
    (80, 160, 255),
    (80, 220, 120),
    (255, 180, 60),
    (200, 100, 255),
    (60, 220, 220),
    (255, 120, 180),
    (180, 255, 80),
    (255, 140, 60),
    (140, 140, 255),
]


@dataclass
class AnnotationRegion:
    id: str
    bounds: Bounds
    description: str
    color_index: int = 0
    lora_name: str = ""
    lora_strength: float = 1.0

    def to_dict(self):
        result = {
            "id": self.id,
            "bounds": [self.bounds.x, self.bounds.y, self.bounds.width, self.bounds.height],
            "description": self.description,
            "color_index": self.color_index,
        }
        if self.lora_name:
            result["lora_name"] = self.lora_name
            result["lora_strength"] = self.lora_strength
        return result

    @staticmethod
    def from_dict(data: dict):
        b = data["bounds"]
        return AnnotationRegion(
            data["id"],
            Bounds(b[0], b[1], b[2], b[3]),
            data.get("description", ""),
            data.get("color_index", 0),
            data.get("lora_name", ""),
            float(data.get("lora_strength", 1.0)),
        )

    def loras(self) -> list[LoraInput]:
        if not self.lora_name:
            return []
        return [LoraInput(self.lora_name, self.lora_strength)]


@dataclass
class AugmentSkippedRegion:
    number: int
    description: str
    reason: str


def format_skipped_regions_warning(skipped: list[AugmentSkippedRegion]) -> str:
    lines = [_("The following regions will be skipped:")]
    for entry in skipped:
        desc = entry.description.strip() or _("(no description)")
        if len(desc) > 48:
            desc = desc[:45] + "..."
        lines.append(f"• {_('Region {n}').format(n=entry.number)}: {desc} — {entry.reason}")
    return "\n".join(lines)


def plan_augment_inpaint_regions(
    regions: list[AnnotationRegion],
    extent: Extent,
    global_prompt: str,
    min_coverage: float = 0.005,
) -> tuple[
    list[tuple[Mask, Bounds, str, JobRegion, list[LoraInput]]], list[AugmentSkippedRegion]
]:
    """Build per-region inpaint masks and prompts (one inpaint pass per region)."""
    canvas = Bounds(0, 0, *extent)
    entries: list[tuple[Mask, Bounds, str, JobRegion, list[LoraInput], int]] = []
    skipped: list[AugmentSkippedRegion] = []

    for i, ann in enumerate(regions):
        number = i + 1
        if not ann.description.strip():
            continue
        if ann.bounds.area == 0:
            skipped.append(
                AugmentSkippedRegion(number, ann.description, _("selection has zero area"))
            )
            continue
        coverage = Bounds.intersection(canvas, ann.bounds).area / canvas.area
        if coverage < min_coverage:
            skipped.append(
                AugmentSkippedRegion(
                    number,
                    ann.description,
                    _("selection is too small relative to the canvas"),
                )
            )
            continue
        prompt = (
            workflow.merge_prompt(global_prompt, ann.description.strip())
            if global_prompt.strip()
            else ann.description.strip()
        )
        mask = Mask.rectangle(ann.bounds, canvas)
        job_region = JobRegion(ann.id, ann.description, ann.bounds)
        entries.append((mask, ann.bounds, prompt, job_region, ann.loras(), number))

    # Later regions take priority over overlapping areas (same as ETN region stacking).
    accumulated: Image | None = None
    for i in range(len(entries) - 1, -1, -1):
        mask, bounds, prompt, job_region, region_loras, number = entries[i]
        region_mask = mask.to_image(extent)
        if accumulated is not None:
            region_mask = Image.mask_subtract(region_mask, accumulated)
        if region_mask.average() < min_coverage:
            skipped.append(
                AugmentSkippedRegion(
                    number,
                    job_region.prompt,
                    _("mostly covered by higher-numbered regions"),
                )
            )
            entries.pop(i)
            continue
        content_bounds = Image.mask_content_bounds(region_mask)
        if content_bounds is None:
            skipped.append(
                AugmentSkippedRegion(
                    number,
                    job_region.prompt,
                    _("no visible area remains after overlap"),
                )
            )
            entries.pop(i)
            continue
        if min(content_bounds.width, content_bounds.height) < MIN_REGION_SIZE:
            skipped.append(
                AugmentSkippedRegion(
                    number,
                    job_region.prompt,
                    _("unobstructed area smaller than {size}×{size} px").format(
                        size=MIN_REGION_SIZE
                    ),
                )
            )
            entries.pop(i)
            continue
        accumulated = (
            Image.copy(region_mask)
            if accumulated is None
            else Image.mask_add(accumulated, region_mask)
        )
        entries[i] = (
            Mask(canvas, region_mask._qimage),
            content_bounds,
            prompt,
            job_region,
            region_loras,
            number,
        )

    result = [
        (mask, bounds, prompt, job_region, region_loras)
        for mask, bounds, prompt, job_region, region_loras, _ in entries
    ]
    return result, skipped


def process_augment_inpaint_regions(
    regions: list[AnnotationRegion],
    extent: Extent,
    global_prompt: str,
    min_coverage: float = 0.005,
) -> list[tuple[Mask, Bounds, str, JobRegion, list[LoraInput]]]:
    entries, _ = plan_augment_inpaint_regions(regions, extent, global_prompt, min_coverage)
    return entries


def _lora_trigger_suffix(loras: list[LoraInput]) -> str:
    files = FileLibrary.instance()
    triggers: list[str] = []
    for lora in loras:
        if file := files.loras.find(lora.name):
            if text := file.meta("lora_triggers", "").strip():
                triggers.append(text)
    return (", " + ", ".join(triggers)) if triggers else ""


def _augment_denoise_edge(strength: float) -> float:
    """How much denoise mask grow/feather to use (ramps after ~0.52)."""
    return max(0.0, min(1.0, (strength - 0.52) / 0.38))


def _augment_composite_edge(strength: float) -> float:
    """Compositing blend only when the edit is strong enough to need a soft paste edge."""
    if strength < 0.58:
        return 0.0
    return max(0.0, min(1.0, (strength - 0.58) / 0.27))


def _smoothstep_edge_fade(x: int, y: int, width: int, height: int, margin: int) -> float:
    """1.0 in the interior; smooth ramp toward the patch border in the outer `margin` pixels."""
    if margin <= 0:
        return 1.0
    dist = min(x, y, width - 1 - x, height - 1 - y)
    if dist >= margin:
        return 1.0
    t = dist / margin
    return t * t * (3.0 - 2.0 * t)


def _augment_paste_edge_px(strength: float, min_dim: int) -> int:
    """Extra client-side edge blend (on top of Comfy alpha) for strong edits."""
    if strength < 0.58:
        return 0
    px = round(12 + (strength - 0.58) * 48)
    return min(px, max(8, min_dim // 4))


def adapt_augment_mask_params(
    bounds: Bounds, strength: float, context: Bounds | None = None
) -> tuple[int, int, int]:
    """Compute grow/feather/blend for augment.

    Below ~0.58 strength: no compositing blend (avoids a visible box when the edit is subtle).
    Denoise grow/feather ramps with strength so differential diffusion handles soft edges first.
    composite_mask = denoise_mask -> erode(blend/2) -> blur(blend)  [only when strength >= 0.58]
    """
    min_dim = min(bounds.width, bounds.height)
    if settings.selection_feather == 0 or strength <= 0:
        return 0, 0, 0

    denoise_t = _augment_denoise_edge(strength)
    composite_t = _augment_composite_edge(strength)
    if denoise_t <= 0 and composite_t <= 0:
        return 0, 0, 0

    size_factor = bounds.extent.diagonal
    cap_dim = min(context.extent.width, context.extent.height) if context else min_dim

    base_feather = max(
        int((settings.selection_feather / 100) * size_factor),
        settings.selection_min_transition,
    )
    feather = round(base_feather * denoise_t) if denoise_t > 0 else 0
    grow = settings.selection_grow_offset + feather // 2 if feather > 0 else 0

    if strength >= 0.55 and denoise_t > 0:
        grow += round((strength - 0.55) * 22 * denoise_t)

    max_edge = max(10, min(cap_dim // 4, 88))
    feather = min(feather, max_edge * 2)
    grow = min(grow, settings.selection_grow_offset + feather // 2 + round(4 + strength * 18))

    blend = 0
    if composite_t > 0:
        base_blend = min(settings.selection_blend, grow + feather // 2)
        max_blend = max(12, min(min_dim - 8, cap_dim // 2))
        blend = round(base_blend * composite_t)
        blend = max(blend, round(14 * composite_t))
        if strength >= 0.65:
            blend = max(blend, round(18 + (strength - 0.65) * 70))
        blend = min(blend, max_blend)

    return grow, feather, blend


def _fit_augment_patch(patch: Image, expected: Extent) -> Image:
    if patch.extent == expected:
        return patch
    if patch.extent.width >= expected.width and patch.extent.height >= expected.height:
        ox = (patch.extent.width - expected.width) // 2
        oy = (patch.extent.height - expected.height) // 2
        return Image.crop(patch, Bounds(ox, oy, expected.width, expected.height))
    if patch.extent.width > 0 and patch.extent.height > 0:
        return Image.scale(patch, expected)
    return patch


def composite_augment_region(
    base: Image, region_result: Image, paste_bounds: Bounds, strength: float = 1.0
) -> Image:
    """Paste patch with strength-aware edge handling (no interior blur)."""
    composite = Image.copy(base)
    patch = _fit_augment_patch(region_result, paste_bounds.extent)
    x0, y0 = paste_bounds.offset
    w, h = paste_bounds.width, paste_bounds.height

    if not patch.is_rgba:
        composite.draw_image(patch, (x0, y0))
        return composite

    base_crop = Image.crop(base, paste_bounds)
    out = composite._qimage
    pq = patch._qimage.convertToFormat(QImage.Format.Format_ARGB32)
    bq = base_crop._qimage.convertToFormat(QImage.Format.Format_ARGB32)

    edge_px = _augment_paste_edge_px(strength, min(w, h))
    low_strength = strength < 0.58
    threshold = max(10, int(36 * (0.58 - strength))) if low_strength else 0

    for y in range(h):
        for x in range(w):
            pc = pq.pixel(x, y)
            pa = qAlpha(pc) / 255.0
            if pa < 0.02:
                continue

            bc = bq.pixel(x, y)
            if low_strength:
                diff = (
                    abs(qRed(pc) - qRed(bc))
                    + abs(qGreen(pc) - qGreen(bc))
                    + abs(qBlue(pc) - qBlue(bc))
                )
                if diff < threshold:
                    continue

            edge_fade = _smoothstep_edge_fade(x, y, w, h, edge_px)
            a = pa * edge_fade
            if a < 0.02:
                continue

            cx, cy = x0 + x, y0 + y
            dc = out.pixel(cx, cy)
            inv = 1.0 - a
            nr = int(qRed(pc) * a + qRed(dc) * inv)
            ng = int(qGreen(pc) * a + qGreen(dc) * inv)
            nb = int(qBlue(pc) * a + qBlue(dc) * inv)
            out.setPixel(cx, cy, qRgba(nr, ng, nb, 255))

    return composite


def augment_inpaint_params(
    region_bounds: Bounds, arch, strength: float, context: Bounds | None = None
) -> InpaintParams:
    """Inpaint mask processing for augment: soft edges scaled by strength and region size."""
    min_dim = min(region_bounds.width, region_bounds.height)
    inpaint = workflow.detect_inpaint(
        InpaintMode.add_object, region_bounds, arch, ConditioningInput("", ""), strength
    )
    grow, feather, blend = adapt_augment_mask_params(region_bounds, strength, context)
    inpaint.grow = grow
    inpaint.feather = feather
    inpaint.blend = blend
    if min_dim < 64:
        inpaint.fill = FillMode.neutral
    return inpaint.clamped()


def augment_context_mask(
    mask: Mask, extent: Extent, context_bounds: Bounds, paste_bounds: Bounds
) -> Mask:
    """Build a context-sized mask aligned to the user's annotation rectangle."""
    rel = paste_bounds.relative_to(context_bounds)
    context_mask_img = Image.crop(mask.to_image(extent), context_bounds)
    # If overlap trimming changed the mask shape, use the actual mask pixels.
    if trimmed := Image.mask_content_bounds(context_mask_img):
        if (
            trimmed.x != rel.x
            or trimmed.y != rel.y
            or trimmed.width != rel.width
            or trimmed.height != rel.height
        ):
            return Mask(Bounds(0, 0, *context_bounds.extent), context_mask_img._qimage)
    return Mask.rectangle(rel, Bounds(0, 0, *context_bounds.extent))


def _overlay_fill(bounds: Bounds, color: tuple[int, int, int], alpha: int = 90) -> Image:
    img = Image.create(bounds.extent)
    qcolor = QColor(*color)
    qcolor.setAlpha(alpha)
    img._qimage.fill(qcolor.rgba())
    return img


class AugmentWorkspace(QObject, ObservableProperties):
    """Mark rectangular regions on a source layer and re-render with per-area descriptions."""

    source_layer_id = Property("", persist=True)
    global_prompt = Property("", persist=True)
    strength = Property(0.5, persist=True)
    active_region_id = Property("", persist=True)
    can_generate = Property(True)

    source_layer_id_changed = pyqtSignal(str)
    global_prompt_changed = pyqtSignal(str)
    strength_changed = pyqtSignal(float)
    active_region_id_changed = pyqtSignal(str)
    can_generate_changed = pyqtSignal(bool)
    regions_changed = pyqtSignal()
    modified = pyqtSignal(QObject, str)

    def __init__(self, model):
        super().__init__()
        self._model = weakref.ref(model)
        self._regions: list[AnnotationRegion] = []
        self._overlay_layer_ids: dict[str, str] = {}
        self._overlay_group_id = ""
        self._in_progress = False
        self._regions_remaining = 0

    @property
    def regions(self) -> list[AnnotationRegion]:
        return self._regions

    def set_regions(self, regions: list[AnnotationRegion]):
        self._regions = regions[:MAX_REGIONS]
        self.regions_changed.emit()
        self.modified.emit(self, "regions")
        self.update_overlay()

    def load_regions(self, data: list[dict]):
        self._regions = [AnnotationRegion.from_dict(d) for d in data][:MAX_REGIONS]
        self.regions_changed.emit()

    def source_layer(self) -> Layer | None:
        model = self._model()
        if model is None or not self.source_layer_id:
            return None
        try:
            return model.layers.find(QUuid(self.source_layer_id))
        except Exception:
            return None

    def set_source_layer(self, layer: Layer | None):
        self.source_layer_id = layer.id_string if layer else ""
        self.modified.emit(self, "source_layer_id")

    def active_region(self) -> AnnotationRegion | None:
        if not self.active_region_id:
            return None
        return next((r for r in self._regions if r.id == self.active_region_id), None)

    def add_region_from_selection(self):
        model = util.ensure(self._model())
        selection = model.document.selection_bounds
        if selection is None or selection.area == 0:
            model.report_error(_("Draw a rectangular selection on the canvas first."))
            return
        if len(self._regions) >= MAX_REGIONS:
            model.report_error(_("Maximum of {n} regions reached.").format(n=MAX_REGIONS))
            return

        region = AnnotationRegion(
            str(uuid.uuid4()),
            selection,
            "",
            len(self._regions) % len(OVERLAY_COLORS),
        )
        self._regions.append(region)
        self.active_region_id = region.id
        self.regions_changed.emit()
        self.modified.emit(self, "regions")
        self.update_overlay()

    def remove_region(self, region_id: str):
        self._regions = [r for r in self._regions if r.id != region_id]
        self._overlay_layer_ids.pop(region_id, None)
        if self.active_region_id == region_id:
            self.active_region_id = self._regions[-1].id if self._regions else ""
        self.regions_changed.emit()
        self.modified.emit(self, "regions")
        self.update_overlay()

    def update_region_description(self, region_id: str, text: str):
        for region in self._regions:
            if region.id == region_id:
                region.description = text
                self.modified.emit(self, "regions")
                return

    def update_region_lora(self, region_id: str, lora_name: str, lora_strength: float):
        for region in self._regions:
            if region.id == region_id:
                region.lora_name = lora_name
                region.lora_strength = lora_strength
                self.modified.emit(self, "regions")
                return

    def _find_overlay_group(self, layers) -> Layer | None:
        if self._overlay_group_id:
            layer = layers.find(QUuid(self._overlay_group_id))
            if layer is not None:
                return layer
        for layer in layers.all:
            if layer.name == OVERLAY_GROUP_NAME and layer.type is LayerType.group:
                self._overlay_group_id = layer.id_string
                return layer
        return None

    def update_overlay(self):
        model = self._model()
        if model is None:
            return
        layers = model.layers
        group = self._find_overlay_group(layers)
        if group is None and len(self._regions) > 0:
            group = layers.create_group(OVERLAY_GROUP_NAME)
            group.is_locked = True
            self._overlay_group_id = group.id_string

        if group is None:
            return

        active_ids = {r.id for r in self._regions}
        for region_id, layer_id in list(self._overlay_layer_ids.items()):
            if region_id not in active_ids:
                if layer := layers.find(QUuid(layer_id)):
                    layer.remove()
                del self._overlay_layer_ids[region_id]

        for i, region in enumerate(self._regions):
            color = OVERLAY_COLORS[region.color_index % len(OVERLAY_COLORS)]
            fill = _overlay_fill(region.bounds, color)
            layer_id = self._overlay_layer_ids.get(region.id)
            layer = layers.find(QUuid(layer_id)) if layer_id else None
            if layer is None:
                layer = layers.create(
                    f"Region {i + 1}",
                    fill,
                    region.bounds,
                    make_active=False,
                    parent=group,
                )
                layer.is_locked = True
                self._overlay_layer_ids[region.id] = layer.id_string
            else:
                layer.name = f"Region {i + 1}"
                layer.write_pixels(fill, region.bounds)

        group.is_visible = True

    def generate(self):
        model = util.ensure(self._model())
        ok, msg = model.document.check_color_mode()
        if not ok and msg:
            model.report_error(msg)
            return

        layer = self.source_layer()
        if layer is None:
            model.report_error(_("Select a source layer to augment."))
            return

        bounds = layer.compute_bounds()
        if bounds.is_zero:
            model.report_error(_("The source layer is empty."))
            return

        described = [r for r in self._regions if r.description.strip()]
        if len(described) == 0:
            model.report_error(_("Add at least one region with a description."))
            return

        extent = model.document.extent
        region_list, skipped = plan_augment_inpaint_regions(
            self._regions, extent, self.global_prompt
        )
        from .model import Error, ErrorKind

        if len(region_list) == 0:
            if skipped:
                model.report_error(
                    Error(ErrorKind.warning, format_skipped_regions_warning(skipped))
                )
            else:
                model.report_error(
                    _(
                        "No valid regions to process. Each region needs a description, "
                        "at least {size}×{size} px of unobstructed area, and minimal overlap "
                        "with higher-numbered regions."
                    ).format(size=MIN_REGION_SIZE)
                )
            return
        if skipped:
            model.report_error(Error(ErrorKind.warning, format_skipped_regions_warning(skipped)))
            log.info("Augment: %d region(s) will be skipped", len(skipped))
        else:
            model.clear_error()

        self.set_in_progress(True)
        from .model import _report_errors

        eventloop.run(_report_errors(model, self._generate_all(model, layer, region_list)))

    async def _generate_all(self, model, layer, region_list):
        client = model._connection.client
        extent = model.document.extent
        canvas_bounds = Bounds(0, 0, *extent)
        try:
            image = layer.get_pixels(canvas_bounds)
            if len(region_list) == 0:
                return

            self._regions_remaining = len(region_list)
            arch = model.arch
            seed = model.seed if model.fixed_seed else workflow.generate_seed()
            current = image
            last_job = None

            for mask, region_bounds, prompt, job_region, region_loras in region_list:
                # Use the user's annotation for placement unless overlap trimmed the mask.
                paste_bounds = (
                    region_bounds
                    if region_bounds != job_region.bounds
                    else job_region.bounds
                )
                prompt_with_triggers = prompt + _lora_trigger_suffix(region_loras)
                conditioning = ConditioningInput(positive=prompt_with_triggers, negative="")
                conditioning.language = model.prompt_translation_language
                conditioning, loras, prompt_meta = workflow.prepare_prompts(
                    conditioning, model.style, seed, arch, InpaintMode.add_object, None
                )
                loras = unique(region_loras + loras, key=lambda l: l.name)

                context_bounds = compute_augment_context_bounds(extent, paste_bounds)
                context_image = Image.crop(current, context_bounds)
                context_mask = augment_context_mask(mask, extent, context_bounds, paste_bounds)
                _, relative_region = compute_relative_bounds(context_bounds, paste_bounds)

                inpaint = augment_inpaint_params(
                    relative_region, arch, self.strength, context_bounds
                )
                log.info(
                    "Augment region %s: context %dx%d, paste %dx%d at (%d,%d)",
                    job_region.prompt[:40],
                    context_bounds.width,
                    context_bounds.height,
                    paste_bounds.width,
                    paste_bounds.height,
                    paste_bounds.x,
                    paste_bounds.y,
                )
                workflow_input = workflow.prepare(
                    WorkflowKind.refine_region,
                    context_image,
                    conditioning,
                    model.style,
                    seed,
                    client.models,
                    FileLibrary.instance(),
                    model._performance_settings(client),
                    mask=context_mask,
                    strength=self.strength,
                    loras=loras,
                    inpaint=inpaint,
                    krea2_augment_encode=arch is Arch.krea2,
                )
                job_name = prompt_meta.get("prompt_eval", f"[Augment] {prompt[:60]}")
                job_params = JobParams(canvas_bounds, job_name, regions=[job_region])
                job_params.set_style(model.style, util.ensure(workflow_input.models).checkpoint)
                job_params.metadata.update(prompt_meta)
                job_params.metadata["strength"] = self.strength
                if arch is Arch.krea2:
                    job_params.metadata["color_match"] = workflow.krea2_augment_color_match(
                        self.strength
                    )
                if region_loras:
                    job_params.metadata["region_lora"] = {
                        "name": region_loras[0].name,
                        "strength": region_loras[0].strength,
                    }
                job = model.jobs.add(JobKind.augment, job_params)
                last_job = job
                await model._enqueue_job(job, workflow_input)

                while job.state not in (JobState.finished, JobState.cancelled):
                    await asyncio.sleep(0.05)
                if job.state is not JobState.finished or len(job.results) == 0:
                    if not model.error:
                        model.report_error(_("Augment failed while processing a region."))
                    return
                current = composite_augment_region(
                    current, job.results[0], paste_bounds, self.strength
                )

            assert last_job is not None
            # refine_region returns a cropped patch; preview/apply need the full canvas.
            model.jobs.set_results(last_job, ImageCollection([current]))
            model.show_preview(last_job.id, 0)
        except Exception as e:
            model.report_error(util.log_error(e))
        finally:
            self._regions_remaining = 0
            self.set_in_progress(False)

    def set_in_progress(self, in_progress: bool):
        self._in_progress = in_progress
        self.can_generate = not in_progress
