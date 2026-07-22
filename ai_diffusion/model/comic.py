"""Comic Book workspace: layout templates, per-panel generation, page assembly."""

from __future__ import annotations

import asyncio
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum

from PyQt5.QtCore import QObject, QRectF, QUuid, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen

from .. import eventloop, util
from ..backend import workflow
from ..backend.api import ConditioningInput, ControlInput, LoraInput, WorkflowKind
from ..backend.resources import Arch, ControlMode
from ..files import FileLibrary
from ..image import Bounds, Extent, Image, ImageCollection, multiple_of
from ..localization import translate as _
from ..settings import GenerationFinishedAction, settings
from ..util import client_logger as log, ensure, trim_text, unique
from .jobs import JobKind, JobParams, JobState
from .properties import ObservableProperties, Property

PAGE_WIDTH = 1024
PAGE_HEIGHT = 1536
GUTTER = 16
MARGIN = 24
PAGE_BG = QColor(245, 240, 230)  # cream paper
GUTTER_COLOR = QColor(20, 20, 20)
MAX_CHARACTERS = 8
MAX_SCENES = 8
MAX_PAGE_LORAS = 3
MAX_FACE_REFS_PER_PANEL = 2
# Denoise when regenerating a panel over a locked scene plate (keep set, change action).
SCENE_PLATE_STRENGTH = 0.72
SCENE_PLATE_STRENGTH_DRAFT = 0.58

# Avoid the word "comic" — with vision models it often means "multi-panel page".
COMIC_STYLE = (
    "bold inked illustration, clear cel shading, dynamic cinematic framing, "
    "one unbroken full-bleed scene"
)
COMIC_NEGATIVE = (
    "photorealistic, blurry, low quality, watermark, signature, UI chrome, "
    "multiple panels, comic page layout, gutters, split screen, diptych, triptych, "
    "two images side by side, collage, mirrored duplicate, storyboard grid, "
    "sequential panels, comic strip, inset panel, panel borders inside the image, "
    "Picture 1 Picture 2, stick figure, skeleton pose overlay, neon bones, "
    "text balloons, speech bubbles"
)
# Krea2EncodeRebalance labels each vision ref as "Picture N". Two+ pictures
# strongly bias the model into side-by-side collages (and CFG=1 ignores negatives).
KREA2_COMIC_MAX_VISION_REFS = 1
KREA2_SINGLE_FRAME_GUARD = (
    "Output exactly one single full-frame image of this scene only. "
    "Never place two pictures side by side. Never make a collage, diptych, "
    "split screen, or duplicate of the same scene."
)


class ComicLayout(Enum):
    # Values 0–3 kept stable for document persistence
    grid_6 = 0
    grid_8 = 1
    splash_2 = 2
    splash_3 = 3
    grid_4 = 4
    wide_5 = 5
    mosaic_4 = 6
    strip_3 = 7
    nine_grid = 8
    tall_left_4 = 9
    banner_3 = 10
    inverted_splash_3 = 11
    stack_4 = 12
    offset_6 = 13


class ComicGenerationMode(Enum):
    final = 0
    draft = 1
    refine = 2


class ComicPanelStyle(Enum):
    """Page-level panel frame chrome (stable ints for document persistence)."""

    classic = 0
    thin_black = 1
    bold_black = 2
    double_black = 3
    red_keyline = 4
    white_keyline = 5
    rounded = 6
    rounded_bold = 7


class ComicCameraPreset(Enum):
    """Shot-size / angle presets injected into panel camera notes."""

    custom = 0
    establishing = 1
    wide = 2
    medium = 3
    close_up = 4
    extreme_close = 5
    over_shoulder = 6
    low_angle = 7
    high_angle = 8
    top_down = 9


CAMERA_PRESET_TEXT: dict[ComicCameraPreset, str] = {
    ComicCameraPreset.custom: "",
    ComicCameraPreset.establishing: (
        "wide establishing shot showing the full environment and where figures stand in space"
    ),
    ComicCameraPreset.wide: "wide shot with full bodies and clear surroundings",
    ComicCameraPreset.medium: "medium shot from the waist up, clear face and gesture",
    ComicCameraPreset.close_up: "close-up on the face and upper shoulders, emotional detail",
    ComicCameraPreset.extreme_close: (
        "extreme close-up on eyes or a critical detail, shallow depth of field"
    ),
    ComicCameraPreset.over_shoulder: (
        "over-the-shoulder shot looking past one figure toward another"
    ),
    ComicCameraPreset.low_angle: "low-angle heroic shot looking upward at the subject",
    ComicCameraPreset.high_angle: "high-angle shot looking down on the subject",
    ComicCameraPreset.top_down: "top-down bird's-eye view of the scene",
}


def camera_preset_prompt(preset: ComicCameraPreset) -> str:
    return CAMERA_PRESET_TEXT.get(preset, "")


@dataclass(frozen=True)
class _FrameLook:
    color: QColor
    width: float
    radius: float = 0.0  # 0 = sharp corners
    double: bool = False
    double_gap: float = 5.0


def frame_look(style: ComicPanelStyle) -> _FrameLook:
    match style:
        case ComicPanelStyle.thin_black:
            return _FrameLook(GUTTER_COLOR, 1.5)
        case ComicPanelStyle.bold_black:
            return _FrameLook(GUTTER_COLOR, 5.0)
        case ComicPanelStyle.double_black:
            return _FrameLook(GUTTER_COLOR, 2.5, double=True, double_gap=5.0)
        case ComicPanelStyle.red_keyline:
            return _FrameLook(QColor(180, 24, 24), 3.0)
        case ComicPanelStyle.white_keyline:
            return _FrameLook(QColor(250, 250, 250), 3.0)
        case ComicPanelStyle.rounded:
            return _FrameLook(GUTTER_COLOR, 3.0, radius=22.0)
        case ComicPanelStyle.rounded_bold:
            return _FrameLook(GUTTER_COLOR, 5.0, radius=32.0)
        case _:
            return _FrameLook(GUTTER_COLOR, 3.0)


def panel_frame_path(slot: Bounds, style: ComicPanelStyle, inset: float = 0.0) -> QPainterPath:
    look = frame_look(style)
    x = slot.x + inset
    y = slot.y + inset
    w = max(1.0, slot.width - 2 * inset)
    h = max(1.0, slot.height - 2 * inset)
    rect = QRectF(x, y, w, h)
    path = QPainterPath()
    if look.radius > 0:
        # Keep radius proportional so tiny panels don't become pills.
        r = min(look.radius, min(w, h) * 0.35)
        path.addRoundedRect(rect, r, r)
    else:
        path.addRect(rect)
    return path


@dataclass
class PanelSlot:
    index: int
    bounds: Bounds
    role: str = "panel"  # "panel" | "splash" | "inset"


@dataclass
class CharacterSheet:
    id: str
    name: str = ""
    synopsis: str = ""
    face_layer_id: str = ""
    lora_name: str = ""
    lora_strength: float = 1.0

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "synopsis": self.synopsis,
            "face_layer_id": self.face_layer_id,
            "lora_name": self.lora_name,
            "lora_strength": self.lora_strength,
        }

    @staticmethod
    def from_dict(data: dict):
        return CharacterSheet(
            data.get("id") or str(uuid.uuid4()),
            data.get("name", ""),
            data.get("synopsis", ""),
            data.get("face_layer_id", ""),
            data.get("lora_name", ""),
            float(data.get("lora_strength", 1.0)),
        )

    def loras(self) -> list[LoraInput]:
        if not self.lora_name:
            return []
        return [LoraInput(self.lora_name, self.lora_strength)]


@dataclass
class PageLora:
    name: str = ""
    strength: float = 1.0

    def to_dict(self):
        return {"name": self.name, "strength": self.strength}

    @staticmethod
    def from_dict(data: dict):
        return PageLora(data.get("name", ""), float(data.get("strength", 1.0)))

    def as_input(self) -> LoraInput | None:
        if not self.name:
            return None
        return LoraInput(self.name, self.strength)


@dataclass
class ScenePlate:
    """Reusable stage: prose bible + optional locked background paint layer."""

    id: str
    name: str = ""
    description: str = ""
    layer_id: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "layer_id": self.layer_id,
        }

    @staticmethod
    def from_dict(data: dict):
        return ScenePlate(
            data.get("id") or str(uuid.uuid4()),
            data.get("name", ""),
            data.get("description", ""),
            data.get("layer_id", "") or "",
        )


@dataclass
class PanelBeat:
    """Structured shot notes + optional polished final prompt for one panel."""

    location: str = ""
    camera: str = ""
    action: str = ""
    final: str = ""
    same_location: bool = False
    scene_id: str = ""
    camera_preset: ComicCameraPreset = ComicCameraPreset.custom
    character_ids: list[str] = field(default_factory=list)
    locked: bool = False
    layer_id: str = ""
    # Legacy single-field beat (migrated into action/final on load)
    text: str = ""

    def to_dict(self):
        return {
            "location": self.location,
            "camera": self.camera,
            "action": self.action,
            "final": self.final,
            "same_location": self.same_location,
            "scene_id": self.scene_id,
            "camera_preset": self.camera_preset.value,
            "character_ids": list(self.character_ids),
            "locked": self.locked,
            "layer_id": self.layer_id,
            # Keep text mirrored to final for older readers
            "text": self.final or self.action or self.text,
        }

    @staticmethod
    def from_dict(data: dict):
        location = data.get("location", "") or ""
        camera = data.get("camera", "") or ""
        action = data.get("action", "") or ""
        final = data.get("final", "") or ""
        legacy = data.get("text", "") or ""
        # Older documents only had `text` — treat it as action + final.
        if legacy and not (location or camera or action or final):
            action = legacy
            final = legacy
        elif legacy and not final:
            final = legacy
        preset_val = data.get("camera_preset", ComicCameraPreset.custom.value)
        try:
            preset = ComicCameraPreset(int(preset_val))
        except (TypeError, ValueError):
            preset = ComicCameraPreset.custom
        return PanelBeat(
            location=location,
            camera=camera,
            action=action,
            final=final,
            same_location=bool(data.get("same_location", False)),
            scene_id=data.get("scene_id", "") or "",
            camera_preset=preset,
            character_ids=list(data.get("character_ids", [])),
            locked=bool(data.get("locked", False)),
            layer_id=data.get("layer_id", "") or "",
            text=legacy,
        )

    def has_notes(self) -> bool:
        return bool(
            self.final.strip()
            or self.location.strip()
            or self.camera.strip()
            or self.action.strip()
            or self.text.strip()
            or self.scene_id
            or self.camera_preset is not ComicCameraPreset.custom
        )


def stitch_beat_fields(panel: PanelBeat, location: str = "", camera: str = "") -> str:
    """Join structured notes into a compact scene string (no LLM)."""
    parts: list[str] = []
    loc = location.strip() or panel.location.strip()
    if loc:
        parts.append(f"Location: {loc}")
    elif panel.same_location:
        parts.append("Location: same setting as the previous panel")
    cam = camera.strip() or panel.camera.strip()
    if not cam:
        cam = camera_preset_prompt(panel.camera_preset)
    if cam:
        parts.append(f"Camera: {cam}")
    if panel.action.strip():
        parts.append(f"Action: {panel.action.strip()}")
    elif panel.text.strip() and not panel.final.strip():
        parts.append(f"Action: {panel.text.strip()}")
    return ". ".join(parts)


def panel_scene_text(panel: PanelBeat, location: str = "", camera: str = "") -> str:
    """Text used for generation: polished final if present, else stitched notes."""
    if panel.final.strip():
        return panel.final.strip()
    stitched = stitch_beat_fields(panel, location, camera)
    if stitched:
        return stitched
    return panel.text.strip()


def _grid_slots(
    page: Extent, cols: int, rows: int, margin: int = MARGIN, gutter: int = GUTTER
) -> list[PanelSlot]:
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    cell_w = (inner_w - gutter * (cols - 1)) // cols
    cell_h = (inner_h - gutter * (rows - 1)) // rows
    slots: list[PanelSlot] = []
    i = 0
    for r in range(rows):
        for c in range(cols):
            x = margin + c * (cell_w + gutter)
            y = margin + r * (cell_h + gutter)
            slots.append(PanelSlot(i, Bounds(x, y, cell_w, cell_h), "panel"))
            i += 1
    return slots


def _splash_slots(page: Extent, inset_count: int) -> list[PanelSlot]:
    """Large splash on top, inset_count panels along the bottom."""
    margin, gutter = MARGIN, GUTTER
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    inset_h = max(220, (inner_h - gutter) // 3)
    splash_h = inner_h - gutter - inset_h
    slots = [PanelSlot(0, Bounds(margin, margin, inner_w, splash_h), "splash")]
    cell_w = (inner_w - gutter * (inset_count - 1)) // inset_count
    y = margin + splash_h + gutter
    for i in range(inset_count):
        x = margin + i * (cell_w + gutter)
        slots.append(PanelSlot(i + 1, Bounds(x, y, cell_w, inset_h), "panel"))
    return slots


def _wide_five_slots(page: Extent) -> list[PanelSlot]:
    """Large wide top beat, then two mid panels, then two bottom panels (5 total)."""
    margin, gutter = MARGIN, GUTTER
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    # Proportions: wide ~38%, mid row ~30%, bottom row ~32%
    wide_h = max(280, int(inner_h * 0.38))
    remain = inner_h - wide_h - 2 * gutter
    mid_h = remain // 2
    bot_h = remain - mid_h
    half_w = (inner_w - gutter) // 2
    y0 = margin
    y1 = margin + wide_h + gutter
    y2 = y1 + mid_h + gutter
    return [
        PanelSlot(0, Bounds(margin, y0, inner_w, wide_h), "splash"),
        PanelSlot(1, Bounds(margin, y1, half_w, mid_h), "panel"),
        PanelSlot(2, Bounds(margin + half_w + gutter, y1, half_w, mid_h), "panel"),
        PanelSlot(3, Bounds(margin, y2, half_w, bot_h), "panel"),
        PanelSlot(4, Bounds(margin + half_w + gutter, y2, half_w, bot_h), "panel"),
    ]


def _mosaic_slots(page: Extent) -> list[PanelSlot]:
    """Full-bleed background splash with 3 smaller panels scattered on top."""
    margin = MARGIN
    # Slightly inset background so page edges still show cream gutter
    back = Bounds(margin, margin, page.width - 2 * margin, page.height - 2 * margin)
    # Haphazard inset sizes (~28–34% of page)
    w1, h1 = int(page.width * 0.42), int(page.height * 0.22)
    w2, h2 = int(page.width * 0.36), int(page.height * 0.26)
    w3, h3 = int(page.width * 0.40), int(page.height * 0.24)
    # Positions chosen to feel scattered without fully covering the splash
    x1, y1 = int(page.width * 0.48), int(page.height * 0.10)
    x2, y2 = int(page.width * 0.08), int(page.height * 0.38)
    x3, y3 = int(page.width * 0.50), int(page.height * 0.68)
    return [
        PanelSlot(0, back, "splash"),
        PanelSlot(1, Bounds(x1, y1, w1, h1), "inset"),
        PanelSlot(2, Bounds(x2, y2, w2, h2), "inset"),
        PanelSlot(3, Bounds(x3, y3, w3, h3), "inset"),
    ]


def _strip_slots(page: Extent, count: int) -> list[PanelSlot]:
    """Stacked full-width cinematic strips."""
    margin, gutter = MARGIN, GUTTER
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    cell_h = (inner_h - gutter * (count - 1)) // count
    slots: list[PanelSlot] = []
    for i in range(count):
        y = margin + i * (cell_h + gutter)
        role = "splash" if i == 0 and count <= 3 else "panel"
        slots.append(PanelSlot(i, Bounds(margin, y, inner_w, cell_h), role))
    return slots


def _tall_left_slots(page: Extent) -> list[PanelSlot]:
    """Tall establishing panel on the left, three stacked beats on the right."""
    margin, gutter = MARGIN, GUTTER
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    left_w = int(inner_w * 0.55)
    right_w = inner_w - left_w - gutter
    right_h = (inner_h - 2 * gutter) // 3
    # Last right panel absorbs rounding remainder
    right_h3 = inner_h - 2 * (right_h + gutter)
    x_right = margin + left_w + gutter
    return [
        PanelSlot(0, Bounds(margin, margin, left_w, inner_h), "splash"),
        PanelSlot(1, Bounds(x_right, margin, right_w, right_h), "panel"),
        PanelSlot(2, Bounds(x_right, margin + right_h + gutter, right_w, right_h), "panel"),
        PanelSlot(
            3,
            Bounds(x_right, margin + 2 * (right_h + gutter), right_w, right_h3),
            "panel",
        ),
    ]


def _banner_three_slots(page: Extent) -> list[PanelSlot]:
    """Wide banner on top, two large equal panels below."""
    margin, gutter = MARGIN, GUTTER
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    banner_h = max(200, int(inner_h * 0.28))
    lower_h = inner_h - banner_h - gutter
    half_w = (inner_w - gutter) // 2
    y1 = margin + banner_h + gutter
    return [
        PanelSlot(0, Bounds(margin, margin, inner_w, banner_h), "splash"),
        PanelSlot(1, Bounds(margin, y1, half_w, lower_h), "panel"),
        PanelSlot(2, Bounds(margin + half_w + gutter, y1, half_w, lower_h), "panel"),
    ]


def _inverted_splash_slots(page: Extent, inset_count: int = 3) -> list[PanelSlot]:
    """Small insets across the top, large splash dominating the bottom."""
    margin, gutter = MARGIN, GUTTER
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    inset_h = max(200, (inner_h - gutter) // 4)
    splash_h = inner_h - gutter - inset_h
    cell_w = (inner_w - gutter * (inset_count - 1)) // inset_count
    slots: list[PanelSlot] = []
    for i in range(inset_count):
        x = margin + i * (cell_w + gutter)
        slots.append(PanelSlot(i, Bounds(x, margin, cell_w, inset_h), "panel"))
    slots.append(
        PanelSlot(
            inset_count,
            Bounds(margin, margin + inset_h + gutter, inner_w, splash_h),
            "splash",
        )
    )
    return slots


def _offset_six_slots(page: Extent) -> list[PanelSlot]:
    """Asymmetric 6-panel brick layout (wide/narrow alternating each row)."""
    margin, gutter = MARGIN, GUTTER
    inner_w = page.width - 2 * margin
    inner_h = page.height - 2 * margin
    row_h = (inner_h - 2 * gutter) // 3
    wide_w = int(inner_w * 0.62)
    narrow_w = inner_w - wide_w - gutter
    y0 = margin
    y1 = margin + row_h + gutter
    y2 = margin + 2 * (row_h + gutter)
    row_h3 = page.height - margin - y2
    return [
        PanelSlot(0, Bounds(margin, y0, wide_w, row_h), "panel"),
        PanelSlot(1, Bounds(margin + wide_w + gutter, y0, narrow_w, row_h), "panel"),
        PanelSlot(2, Bounds(margin, y1, narrow_w, row_h), "panel"),
        PanelSlot(3, Bounds(margin + narrow_w + gutter, y1, wide_w, row_h), "panel"),
        PanelSlot(4, Bounds(margin, y2, wide_w, row_h3), "panel"),
        PanelSlot(5, Bounds(margin + wide_w + gutter, y2, narrow_w, row_h3), "panel"),
    ]


def layout_slots(layout: ComicLayout, page: Extent | None = None) -> list[PanelSlot]:
    page = page or Extent(PAGE_WIDTH, PAGE_HEIGHT)
    match layout:
        case ComicLayout.grid_4:
            return _grid_slots(page, 2, 2)
        case ComicLayout.grid_6:
            return _grid_slots(page, 2, 3)
        case ComicLayout.grid_8:
            return _grid_slots(page, 2, 4)
        case ComicLayout.nine_grid:
            return _grid_slots(page, 3, 3)
        case ComicLayout.splash_2:
            return _splash_slots(page, 2)
        case ComicLayout.splash_3:
            return _splash_slots(page, 3)
        case ComicLayout.wide_5:
            return _wide_five_slots(page)
        case ComicLayout.mosaic_4:
            return _mosaic_slots(page)
        case ComicLayout.strip_3:
            return _strip_slots(page, 3)
        case ComicLayout.stack_4:
            return _strip_slots(page, 4)
        case ComicLayout.tall_left_4:
            return _tall_left_slots(page)
        case ComicLayout.banner_3:
            return _banner_three_slots(page)
        case ComicLayout.inverted_splash_3:
            return _inverted_splash_slots(page, 3)
        case ComicLayout.offset_6:
            return _offset_six_slots(page)
    return _grid_slots(page, 2, 3)


def slot_label(slot: PanelSlot) -> str:
    """User-facing label for a panel beat editor."""
    match slot.role:
        case "splash":
            return _("Splash") if slot.index == 0 else _("Splash {n}").format(n=slot.index + 1)
        case "inset":
            return _("Inset {n}").format(n=slot.index)
        case _:
            return _("Panel {n}").format(n=slot.index + 1)


def layout_panel_count(layout: ComicLayout) -> int:
    return len(layout_slots(layout))


def create_page_canvas(page: Extent | None = None) -> Image:
    page = page or Extent(PAGE_WIDTH, PAGE_HEIGHT)
    return Image.create(page, fill=PAGE_BG)


def paste_panel(
    page: Image,
    panel: Image,
    slot: Bounds,
    style: ComicPanelStyle = ComicPanelStyle.classic,
) -> Image:
    """Paste panel art into the page slot, clipped and framed per panel style."""
    scaled = Image.scale(panel, slot.extent)
    look = frame_look(style)
    painter = QPainter(page._qimage)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    clip = panel_frame_path(slot, style, inset=0.5)
    painter.setClipPath(clip)
    painter.drawImage(slot.x, slot.y, scaled._qimage)
    painter.setClipping(False)

    painter.setBrush(Qt.BrushStyle.NoBrush)
    outer = panel_frame_path(slot, style, inset=1.0)
    painter.setPen(QPen(look.color, look.width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
    painter.drawPath(outer)
    if look.double:
        inner_inset = look.width + look.double_gap
        inner = panel_frame_path(slot, style, inset=inner_inset)
        painter.setPen(
            QPen(look.color, max(1.0, look.width * 0.75), Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap)
        )
        painter.drawPath(inner)
    painter.end()
    return page


def generation_extent_for_slot(slot: Bounds, draft: bool = False) -> Extent:
    """Snap panel size up to a model-friendly multiple of 64."""
    min_side = 512 if not draft else 384
    target_short = 768 if draft else None
    w = max(min_side, multiple_of(slot.width, 64))
    h = max(min_side, multiple_of(slot.height, 64))
    if draft and target_short is not None:
        # Shrink so the short side is around 768 for faster drafts
        short = min(w, h)
        if short > target_short:
            scale = target_short / short
            w = max(min_side, multiple_of(int(w * scale), 64))
            h = max(min_side, multiple_of(int(h * scale), 64))
    slot_aspect = slot.width / max(slot.height, 1)
    gen_aspect = w / h
    if abs(gen_aspect - slot_aspect) > 0.15:
        if slot_aspect > gen_aspect:
            w = multiple_of(int(h * slot_aspect), 64)
        else:
            h = multiple_of(int(w / slot_aspect), 64)
    return Extent(w, h).at_least(min_side)


def _lora_trigger_suffix(loras: list[LoraInput]) -> str:
    files = FileLibrary.instance()
    triggers: list[str] = []
    for lora in loras:
        if file := files.loras.find(lora.name):
            if text := file.meta("lora_triggers", "").strip():
                triggers.append(text)
    return (", " + ", ".join(triggers)) if triggers else ""


def build_panel_prompt(
    synopsis: str,
    beat: str,
    panel_index: int,
    panel_count: int,
    role: str,
    title: str = "",
    characters: list[CharacterSheet] | None = None,
) -> str:
    # Guard first — with CFG=1 (Krea2 turbo) negatives are unused.
    parts = [KREA2_SINGLE_FRAME_GUARD, COMIC_STYLE]
    if role == "splash":
        parts.append("dramatic establishing shot, large heroic composition")
    elif role == "inset":
        parts.append("focused secondary moment, tight framing")
    else:
        parts.append("one clear scene filling the whole frame")
    if title.strip():
        parts.append(f'Story titled "{title.strip()}"')
    if synopsis.strip():
        parts.append(f"Story context: {synopsis.strip()}")
    if characters:
        for ch in characters:
            name = ch.name.strip() or _("Unnamed character")
            body = ch.synopsis.strip()
            if body:
                parts.append(f"Character {name}: {body}")
            else:
                parts.append(f"Character {name} appears")
    beat_text = beat.strip() or "continue the story visually"
    # Avoid "panel X of N" / "comic page" wording — models turn that into multi-panel art.
    _ = panel_count, panel_index
    parts.append(f"Scene: {beat_text}")
    parts.append(
        "Depict only this one moment. Keep character identity, costume and art style consistent. "
        "No panel borders, gutters, page layouts, or speech balloons."
    )
    return ". ".join(parts)


def build_finalize_instruction(
    synopsis: str,
    title: str,
    panel: PanelBeat,
    location: str,
    characters: list[CharacterSheet],
    camera: str = "",
) -> str:
    """Instruction for Qwen-VL: rewrite structured notes into a Krea-ready scene prompt."""
    loc = location.strip() or (
        "same setting as the previous panel" if panel.same_location else "(not specified)"
    )
    cam = (
        camera.strip()
        or panel.camera.strip()
        or camera_preset_prompt(panel.camera_preset)
        or "(not specified)"
    )
    act = panel.action.strip() or panel.text.strip() or "(not specified)"
    cast_lines: list[str] = []
    for ch in characters:
        name = ch.name.strip() or "Character"
        body = ch.synopsis.strip()
        cast_lines.append(f"- {name}: {body}" if body else f"- {name}")
    cast_block = "\n".join(cast_lines) if cast_lines else "- (no cast selected)"
    title_line = title.strip() or "(untitled)"
    syn = synopsis.strip() or "(none)"
    return (
        "IGNORE the attached image completely. Do not describe it.\n"
        "You are writing a SINGLE-FRAME illustration prompt for the Krea image model.\n"
        "Combine the notes below into ONE rich, vivid scene description "
        "(3–6 sentences). Include lighting, materials, atmosphere, clothing details, "
        "spatial layout, and camera framing. Krea works best with long, concrete detail.\n"
        "Rules:\n"
        "- Output ONLY the scene description prose (no bullet lists, no labels, no preamble).\n"
        "- Exactly one unbroken scene filling the whole frame.\n"
        "- Do NOT mention comic panels, gutters, page layouts, speech balloons, "
        "storyboards, split screens, or multiple images.\n"
        "- Preserve character identity and costume from the cast notes.\n\n"
        f"Story title: {title_line}\n"
        f"Page synopsis: {syn}\n"
        f"Cast:\n{cast_block}\n\n"
        f"Location / stage: {loc}\n"
        f"Camera: {cam}\n"
        f"Action: {act}\n"
    )


def local_finalize_beat(
    synopsis: str,
    title: str,
    panel: PanelBeat,
    location: str,
    characters: list[CharacterSheet],
    camera: str = "",
) -> str:
    """Fallback polish without Qwen — denser stitch than raw fields."""
    parts: list[str] = []
    if title.strip():
        parts.append(f'In the story "{title.strip()}"')
    loc = location.strip() or panel.location.strip()
    if loc:
        parts.append(f"set in {loc}")
    elif panel.same_location:
        parts.append("in the same location as the previous moment")
    cam = camera.strip() or panel.camera.strip() or camera_preset_prompt(panel.camera_preset)
    if cam:
        parts.append(f"seen from {cam}")
    if characters:
        names = ", ".join(ch.name.strip() or "a character" for ch in characters)
        parts.append(f"featuring {names}")
        for ch in characters:
            if ch.synopsis.strip():
                parts.append(f"{ch.name.strip() or 'Character'}: {ch.synopsis.strip()}")
    act = panel.action.strip() or panel.text.strip()
    if act:
        parts.append(act)
    if synopsis.strip():
        parts.append(f"Story context: {synopsis.strip()}")
    parts.append(
        "Highly detailed single-frame illustration with clear lighting, "
        "materials, and atmosphere; one unbroken composition"
    )
    return ". ".join(parts) + "."


def _limit_krea2_vision_refs(controls: list[ControlInput]) -> list[ControlInput]:
    """Keep a single vision image for Krea2 comic jobs.

    Style refs are stripped entirely — moodboards / sample pages often make the
    model copy a multi-panel layout into every panel. Priority among the rest:
    face → refine/continuity reference.
    """
    # Never feed Style as a vision image in comic (confirmed to cause diptychs).
    controls = [c for c in controls if c.mode is not ControlMode.style]
    if len(controls) <= KREA2_COMIC_MAX_VISION_REFS:
        return controls

    def _rank(c: ControlInput) -> int:
        if c.mode is ControlMode.face:
            return 0
        if c.mode is ControlMode.reference:
            return 1
        return 2

    ordered = sorted(controls, key=_rank)
    keep = ordered[:KREA2_COMIC_MAX_VISION_REFS]
    dropped = ordered[KREA2_COMIC_MAX_VISION_REFS:]
    if dropped:
        log.info(
            "Comic Krea2: using 1 vision ref (%s); skipped %s",
            keep[0].mode.name if keep else "none",
            ", ".join(c.mode.name for c in dropped),
        )
    return keep


WORKING_GROUP_PREFIX = "[Comic Working]"


class ComicWorkspace(QObject, ObservableProperties):
    """Design a comic page from a layout, generate each panel, assemble the page."""

    layout = Property(ComicLayout.grid_6, persist=True)
    panel_style = Property(ComicPanelStyle.classic, persist=True)
    title = Property("", persist=True)
    synopsis = Property("", persist=True)
    style_layer_id = Property("", persist=True)
    can_generate = Property(True)
    progress_text = Property("")
    can_refine = Property(False)
    can_assemble = Property(False)

    layout_changed = pyqtSignal(ComicLayout)
    panel_style_changed = pyqtSignal(ComicPanelStyle)
    title_changed = pyqtSignal(str)
    synopsis_changed = pyqtSignal(str)
    style_layer_id_changed = pyqtSignal(str)
    can_generate_changed = pyqtSignal(bool)
    progress_text_changed = pyqtSignal(str)
    can_refine_changed = pyqtSignal(bool)
    can_assemble_changed = pyqtSignal(bool)
    panels_changed = pyqtSignal()
    panel_images_changed = pyqtSignal()
    characters_changed = pyqtSignal()
    scenes_changed = pyqtSignal()
    page_loras_changed = pyqtSignal()
    modified = pyqtSignal(QObject, str)

    def __init__(self, model):
        super().__init__()
        self._model = weakref.ref(model)
        self._panels: list[PanelBeat] = [PanelBeat() for _ in range(layout_panel_count(self.layout))]
        self._panel_images: list[Image | None] = [None] * len(self._panels)
        self._characters: list[CharacterSheet] = []
        self._scenes: list[ScenePlate] = []
        self._page_loras: list[PageLora] = []
        self._in_progress = False
        self._panels_remaining = 0
        self._single_panel_job = False
        self._working_group_id = ""
        self._last_panel_images: list[Image] = []
        self._last_slots: list[PanelSlot] = []
        self._draft_panel_images: list[Image] = []
        self._draft_job_id: str = ""
        self._finalize_index: int | None = None
        self.layout_changed.connect(self._sync_panel_count)
        self.panel_style_changed.connect(self._on_panel_style_changed)

    @property
    def panels(self) -> list[PanelBeat]:
        return self._panels

    @property
    def characters(self) -> list[CharacterSheet]:
        return self._characters

    @property
    def scenes(self) -> list[ScenePlate]:
        return self._scenes

    @property
    def page_loras(self) -> list[PageLora]:
        return self._page_loras

    def set_panels(self, panels: list[PanelBeat]):
        self._panels = panels
        self.panels_changed.emit()
        self.modified.emit(self, "panels")

    def load_panels(self, data: list[dict]):
        self._panels = [PanelBeat.from_dict(d) for d in data]
        self._sync_panel_count()
        # Restore images from linked working layers when possible
        for i, panel in enumerate(self._panels):
            if panel.layer_id:
                img = self._layer_image(panel.layer_id)
                if img is not None:
                    self._panel_images[i] = img
        self.panel_images_changed.emit()
        self.refresh_can_assemble()
        self.panels_changed.emit()

    def load_characters(self, data: list[dict]):
        self._characters = [CharacterSheet.from_dict(d) for d in data][:MAX_CHARACTERS]
        self._prune_panel_casts()
        self.characters_changed.emit()

    def load_scenes(self, data: list[dict]):
        self._scenes = [ScenePlate.from_dict(d) for d in data][:MAX_SCENES]
        self._prune_panel_scenes()
        self.scenes_changed.emit()

    def load_page_loras(self, data: list[dict]):
        self._page_loras = [PageLora.from_dict(d) for d in data if d.get("name")][:MAX_PAGE_LORAS]
        self.page_loras_changed.emit()

    def update_panel_text(self, index: int, text: str):
        """Legacy helper — writes into action (+ final mirror for older callers)."""
        if 0 <= index < len(self._panels):
            self._panels[index].action = text
            self._panels[index].text = text
            self.modified.emit(self, "panels")

    def update_panel_fields(
        self,
        index: int,
        *,
        location: str | None = None,
        camera: str | None = None,
        action: str | None = None,
        final: str | None = None,
        same_location: bool | None = None,
        scene_id: str | None = None,
        camera_preset: ComicCameraPreset | None = None,
    ):
        if not (0 <= index < len(self._panels)):
            return
        panel = self._panels[index]
        if location is not None:
            panel.location = location
        if camera is not None:
            panel.camera = camera
        if action is not None:
            panel.action = action
            panel.text = action  # keep legacy field in sync
        if final is not None:
            panel.final = final
            panel.text = final or panel.action
        if same_location is not None:
            panel.same_location = same_location
        if scene_id is not None:
            known = {s.id for s in self._scenes}
            panel.scene_id = scene_id if scene_id in known else ""
        if camera_preset is not None:
            panel.camera_preset = camera_preset
        self.modified.emit(self, "panels")

    def update_panel_cast(self, index: int, character_ids: list[str]):
        if 0 <= index < len(self._panels):
            known = {c.id for c in self._characters}
            self._panels[index].character_ids = [cid for cid in character_ids if cid in known]
            self.modified.emit(self, "panels")

    def scene_by_id(self, scene_id: str) -> ScenePlate | None:
        if not scene_id:
            return None
        for scene in self._scenes:
            if scene.id == scene_id:
                return scene
        return None

    def _scene_id_for_panel(self, index: int) -> str:
        if not (0 <= index < len(self._panels)):
            return ""
        panel = self._panels[index]
        if panel.scene_id and self.scene_by_id(panel.scene_id):
            return panel.scene_id
        if panel.same_location:
            for j in range(index - 1, -1, -1):
                prev = self._panels[j]
                if prev.scene_id and self.scene_by_id(prev.scene_id):
                    return prev.scene_id
        return ""

    def location_for_panel(self, index: int) -> str:
        """Resolve location from scene plate bible, then panel notes / same-as-previous."""
        if not (0 <= index < len(self._panels)):
            return ""
        panel = self._panels[index]
        scene = self.scene_by_id(self._scene_id_for_panel(index))
        scene_desc = scene.description.strip() if scene else ""
        extra = panel.location.strip()
        if scene_desc:
            if extra and extra != scene_desc:
                return f"{scene_desc}. {extra}"
            return scene_desc
        if panel.same_location:
            for j in range(index - 1, -1, -1):
                prev = self._panels[j]
                prev_scene = self.scene_by_id(prev.scene_id)
                base = (
                    prev_scene.description.strip()
                    if prev_scene and prev_scene.description.strip()
                    else prev.location.strip()
                )
                if base:
                    if extra and extra != base:
                        return f"{base}. {extra}"
                    return base
        return extra

    def camera_for_panel(self, index: int) -> str:
        if not (0 <= index < len(self._panels)):
            return ""
        panel = self._panels[index]
        preset = camera_preset_prompt(panel.camera_preset)
        custom = panel.camera.strip()
        if preset and custom and custom.lower() not in preset.lower():
            return f"{preset}. {custom}"
        return custom or preset

    def scene_plate_image_for_panel(self, index: int) -> Image | None:
        scene = self.scene_by_id(self._scene_id_for_panel(index))
        if scene is None or not scene.layer_id:
            return None
        return self._layer_image(scene.layer_id)

    def scene_text_for_panel(self, index: int) -> str:
        if not (0 <= index < len(self._panels)):
            return ""
        return panel_scene_text(
            self._panels[index],
            self.location_for_panel(index),
            self.camera_for_panel(index),
        )

    def add_character(self) -> CharacterSheet | None:
        if len(self._characters) >= MAX_CHARACTERS:
            return None
        sheet = CharacterSheet(str(uuid.uuid4()), name=_("Character {n}").format(
            n=len(self._characters) + 1
        ))
        self._characters.append(sheet)
        self.characters_changed.emit()
        self.modified.emit(self, "characters")
        return sheet

    def remove_character(self, character_id: str):
        self._characters = [c for c in self._characters if c.id != character_id]
        self._prune_panel_casts()
        self.characters_changed.emit()
        self.modified.emit(self, "characters")

    def add_scene(self) -> ScenePlate | None:
        if len(self._scenes) >= MAX_SCENES:
            return None
        scene = ScenePlate(
            str(uuid.uuid4()),
            name=_("Scene {n}").format(n=len(self._scenes) + 1),
        )
        self._scenes.append(scene)
        self.scenes_changed.emit()
        self.modified.emit(self, "scenes")
        return scene

    def remove_scene(self, scene_id: str):
        self._scenes = [s for s in self._scenes if s.id != scene_id]
        self._prune_panel_scenes()
        self.scenes_changed.emit()
        self.modified.emit(self, "scenes")

    def update_scene(
        self,
        scene_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        layer_id: str | None = None,
    ):
        for scene in self._scenes:
            if scene.id != scene_id:
                continue
            if name is not None:
                scene.name = name
            if description is not None:
                scene.description = description
            if layer_id is not None:
                scene.layer_id = layer_id
            self.modified.emit(self, "scenes")
            return

    def _prune_panel_scenes(self):
        known = {s.id for s in self._scenes}
        for panel in self._panels:
            if panel.scene_id and panel.scene_id not in known:
                panel.scene_id = ""

    def update_character(
        self,
        character_id: str,
        *,
        name: str | None = None,
        synopsis: str | None = None,
        face_layer_id: str | None = None,
        lora_name: str | None = None,
        lora_strength: float | None = None,
    ):
        for ch in self._characters:
            if ch.id != character_id:
                continue
            if name is not None:
                ch.name = name
            if synopsis is not None:
                ch.synopsis = synopsis
            if face_layer_id is not None:
                ch.face_layer_id = face_layer_id
            if lora_name is not None:
                ch.lora_name = lora_name
            if lora_strength is not None:
                ch.lora_strength = lora_strength
            self.modified.emit(self, "characters")
            return

    def set_page_loras(self, loras: list[PageLora]):
        self._page_loras = [l for l in loras if l.name][:MAX_PAGE_LORAS]
        self.page_loras_changed.emit()
        self.modified.emit(self, "page_loras")

    def add_page_lora(self) -> bool:
        if len(self._page_loras) >= MAX_PAGE_LORAS:
            return False
        self._page_loras.append(PageLora())
        self.page_loras_changed.emit()
        self.modified.emit(self, "page_loras")
        return True

    def remove_page_lora(self, index: int):
        if 0 <= index < len(self._page_loras):
            del self._page_loras[index]
            self.page_loras_changed.emit()
            self.modified.emit(self, "page_loras")

    def update_page_lora(self, index: int, name: str, strength: float):
        if 0 <= index < len(self._page_loras):
            self._page_loras[index].name = name
            self._page_loras[index].strength = strength
            self.modified.emit(self, "page_loras")

    def _prune_panel_casts(self):
        known = {c.id for c in self._characters}
        for panel in self._panels:
            panel.character_ids = [cid for cid in panel.character_ids if cid in known]

    def _sync_panel_count(self, *_args):
        needed = layout_panel_count(self.layout)
        if len(self._panels) < needed:
            self._panels.extend(PanelBeat() for _ in range(needed - len(self._panels)))
        elif len(self._panels) > needed:
            self._panels = self._panels[:needed]
        while len(self._panel_images) < needed:
            self._panel_images.append(None)
        if len(self._panel_images) > needed:
            self._panel_images = self._panel_images[:needed]
        self.panels_changed.emit()
        self.panel_images_changed.emit()
        self.refresh_can_assemble()
        self.modified.emit(self, "panels")

    def set_in_progress(self, in_progress: bool):
        self._in_progress = in_progress
        self.can_generate = not in_progress
        if not in_progress:
            self.progress_text = ""
        self.refresh_can_refine()
        self.refresh_can_assemble()

    def refresh_can_refine(self):
        if self._in_progress:
            self.can_refine = False
            return
        if self._draft_panel_images:
            self.can_refine = True
            return
        model = self._model()
        if model is None:
            self.can_refine = False
            return
        for job in model.jobs:
            if (
                job.kind is JobKind.comic
                and job.state is JobState.finished
                and job.params.metadata.get("comic_draft")
                and job.params.metadata.get("comic_slots")
            ):
                self.can_refine = True
                return
        self.can_refine = False

    def refresh_can_assemble(self):
        if self._in_progress or not self._panels:
            self.can_assemble = False
            return
        self._ensure_panel_images()
        self.can_assemble = all(
            p.locked and img is not None
            for p, img in zip(self._panels, self._panel_images)
        )

    def _compose_page(
        self,
        panel_images: list[Image],
        slots: list[PanelSlot] | None = None,
        page_extent: Extent | None = None,
    ) -> Image:
        page_extent = page_extent or Extent(PAGE_WIDTH, PAGE_HEIGHT)
        slots = slots or layout_slots(self.layout, page_extent)
        page = create_page_canvas(page_extent)
        style = self.panel_style
        for slot, img in zip(slots, panel_images):
            page = paste_panel(page, img, slot.bounds, style)
        return page

    def _on_panel_style_changed(self, *_args):
        """Re-frame existing panel art when the page keyline style changes (no diffusion)."""
        if self._in_progress:
            return
        model = self._model()
        if model is None:
            return
        self._ensure_panel_images()
        slots = layout_slots(self.layout)
        images = [img for img in self._panel_images[: len(slots)] if img is not None]
        if not images:
            return
        # Compose using available panels in slot order (gaps stay as page background).
        page = create_page_canvas()
        style = self.panel_style
        for slot in slots:
            img = self.panel_image(slot.index)
            if img is not None:
                paste_panel(page, img, slot.bounds, style)
        if model._layer and not model._layer.was_removed:
            model._layer.write_pixels(page, Bounds(0, 0, PAGE_WIDTH, PAGE_HEIGHT))
            model._layer.name = f"[Preview] Comic ({self.panel_style.name.replace('_', ' ')})"
        # Refresh the latest finished page job so Apply uses the new frames.
        for job in reversed(list(model.jobs)):
            if (
                job.kind is JobKind.comic
                and job.state is JobState.finished
                and job.params.metadata.get("comic_slots")
                and len(job.results) > 0
            ):
                panels = (
                    list(job.results[1:])
                    if len(job.results) > 1
                    else list(self._last_panel_images)
                )
                if panels:
                    recomposed = self._compose_page(panels, slots)
                    model.jobs.set_results(job, ImageCollection([recomposed] + panels))
                    job.params.metadata["comic_panel_style"] = self.panel_style.name
                    model.show_preview(job.id, 0)
                break

    def panel_image(self, index: int) -> Image | None:
        if 0 <= index < len(self._panel_images):
            return self._panel_images[index]
        return None

    def _set_panel_image(self, index: int, image: Image | None):
        self._ensure_panel_images()
        if 0 <= index < len(self._panel_images):
            self._panel_images[index] = image
            self.panel_images_changed.emit()
            self.refresh_can_assemble()

    def _ensure_panel_images(self):
        n = len(self._panels)
        while len(self._panel_images) < n:
            self._panel_images.append(None)
        if len(self._panel_images) > n:
            self._panel_images = self._panel_images[:n]

    def _previous_locked_image(self, index: int) -> Image | None:
        for i in range(index - 1, -1, -1):
            if i < len(self._panels) and self._panels[i].locked:
                img = self.panel_image(i)
                if img is not None:
                    return img
        return None

    def lock_panel(self, index: int):
        if not (0 <= index < len(self._panels)):
            return
        if self.panel_image(index) is None:
            model = self._model()
            if model:
                model.report_error(_("Generate this panel before locking it."))
            return
        self._panels[index].locked = True
        self.sync_panel_to_layer(index)
        self.panel_images_changed.emit()
        self.refresh_can_assemble()
        self.modified.emit(self, "panels")

    def unlock_panel(self, index: int):
        if 0 <= index < len(self._panels):
            self._panels[index].locked = False
            self.panel_images_changed.emit()
            self.refresh_can_assemble()
            self.modified.emit(self, "panels")

    def clear_panel(self, index: int):
        if not (0 <= index < len(self._panels)):
            return
        self._panels[index].locked = False
        self._set_panel_image(index, None)
        self.refresh_can_assemble()
        self.modified.emit(self, "panels")

    def _find_working_group(self):
        model = ensure(self._model())
        if self._working_group_id:
            try:
                group = model.layers.find(QUuid(self._working_group_id))
                if group is not None:
                    return group
            except Exception:
                pass
        title = self.title.strip() or "Page"
        name = f"{WORKING_GROUP_PREFIX} {title}"
        for layer in model.layers.all:
            if layer.name == name or layer.name.startswith(WORKING_GROUP_PREFIX):
                self._working_group_id = layer.id_string
                return layer
        group = model.layers.create_group(name)
        self._working_group_id = group.id_string
        return group

    def sync_panel_to_layer(self, index: int):
        model = ensure(self._model())
        image = self.panel_image(index)
        if image is None or not (0 <= index < len(self._panels)):
            return
        slots = layout_slots(self.layout)
        if index >= len(slots):
            return
        slot = slots[index]
        panel = Image.scale(image, slot.bounds.extent)
        group = self._find_working_group()
        layer_name = f"Panel {index + 1}"
        layer = None
        lid = self._panels[index].layer_id
        if lid:
            try:
                layer = model.layers.find(QUuid(lid))
            except Exception:
                layer = None
        if layer is None:
            for child in group.child_layers:
                if child.name == layer_name:
                    layer = child
                    break
        if layer is not None:
            replacement = model.layers.update_layer_image(layer, panel, slot.bounds)
            self._panels[index].layer_id = replacement.id_string
        else:
            layer = model.layers.create(
                layer_name, panel, slot.bounds, make_active=False, parent=group
            )
            self._panels[index].layer_id = layer.id_string
        self.modified.emit(self, "panels")

    def reload_panel_from_layer(self, index: int):
        if not (0 <= index < len(self._panels)):
            return
        lid = self._panels[index].layer_id
        img = self._layer_image(lid)
        if img is None:
            # Try by name under working group
            model = self._model()
            if model is None:
                return
            group = None
            if self._working_group_id:
                try:
                    group = model.layers.find(QUuid(self._working_group_id))
                except Exception:
                    group = None
            if group is not None:
                for child in group.child_layers:
                    if child.name == f"Panel {index + 1}":
                        self._panels[index].layer_id = child.id_string
                        img = self._layer_image(child.id_string)
                        break
        if img is None:
            model = ensure(self._model())
            model.report_error(_("No working layer found for this panel."))
            return
        self._set_panel_image(index, img)
        self.panels_changed.emit()
        self.modified.emit(self, "panels")

    def reload_all_from_layers(self):
        for i in range(len(self._panels)):
            if self._panels[i].layer_id or self._panels[i].locked:
                try:
                    self.reload_panel_from_layer(i)
                except Exception:
                    pass

    def _layer_image(self, layer_id: str) -> Image | None:
        if not layer_id:
            return None
        model = self._model()
        if model is None:
            return None
        try:
            layer = model.layers.find(QUuid(layer_id))
        except Exception:
            return None
        if layer is None:
            return None
        bounds = layer.compute_bounds()
        if bounds.is_zero:
            return None
        return layer.get_pixels(bounds)

    def _characters_for_panel(self, panel: PanelBeat) -> list[CharacterSheet]:
        by_id = {c.id: c for c in self._characters}
        return [by_id[cid] for cid in panel.character_ids if cid in by_id]

    def _collect_loras(self, cast: list[CharacterSheet]) -> list[LoraInput]:
        result: list[LoraInput] = []
        for pl in self._page_loras:
            if inp := pl.as_input():
                result.append(inp)
        for ch in cast:
            result.extend(ch.loras())
        return unique(result, key=lambda l: l.name)

    def _resolve_draft_panels(self, model, slot_count: int) -> list[Image] | None:
        if self._draft_panel_images and len(self._draft_panel_images) >= slot_count:
            return self._draft_panel_images[:slot_count]
        # Prefer selected history job tagged as draft
        for item in model.jobs.selection:
            job = model.jobs.find(item.job)
            if (
                job
                and job.kind is JobKind.comic
                and job.params.metadata.get("comic_draft")
                and len(job.results) > 1
            ):
                panels = list(job.results[1:])
                if len(panels) >= slot_count:
                    return panels[:slot_count]
        if self._draft_job_id:
            job = model.jobs.find(self._draft_job_id)
            if job and len(job.results) > 1:
                panels = list(job.results[1:])
                if len(panels) >= slot_count:
                    return panels[:slot_count]
        return None

    def draft_page(self):
        self.generate_page(ComicGenerationMode.draft)

    def refine_page(self):
        self.generate_page(ComicGenerationMode.refine)

    def generate_remaining(self):
        self.generate_page(ComicGenerationMode.final, skip_locked=True)

    def generate_panel(self, index: int):
        model = ensure(self._model())
        ok, msg = model.document.check_color_mode()
        if not ok and msg:
            model.report_error(msg)
            return
        if not (0 <= index < len(self._panels)):
            return
        panel = self._panels[index]
        if panel.locked:
            model.report_error(_("Unlock this panel before regenerating it."))
            return
        if not panel.has_notes() and not self.synopsis.strip():
            model.report_error(
                _(
                    "Add location/camera/action notes (or Finalize a description) "
                    "for this panel, or a page synopsis, before generating."
                )
            )
            return
        model.clear_error()
        self.set_in_progress(True)
        from .model import _report_errors

        eventloop.run(_report_errors(model, self._generate_single(model, index)))

    def finalize_panel(self, index: int):
        """Polish structured beat notes into a Krea-ready final description via Qwen."""
        model = ensure(self._model())
        if not (0 <= index < len(self._panels)):
            return
        if self._in_progress or self._finalize_index is not None:
            model.report_error(_("Wait for the current comic job to finish."))
            return
        panel = self._panels[index]
        if not (
            panel.location.strip()
            or panel.camera.strip()
            or panel.action.strip()
            or panel.same_location
            or panel.scene_id
            or panel.camera_preset is not ComicCameraPreset.custom
            or panel.text.strip()
        ):
            model.report_error(
                _("Fill in scene, camera, or action before finalizing this panel.")
            )
            return

        cast = self._characters_for_panel(panel)
        location = self.location_for_panel(index)
        camera = self.camera_for_panel(index)
        client = model._connection.client
        from .model import DescribeWorkspace, _report_errors

        vram_gb = client.device_info.vram
        qwen_model = DescribeWorkspace.find_qwenvl_model(client, vram_gb)
        if qwen_model is None:
            # Offline fallback — still useful without Qwen installed.
            panel.final = local_finalize_beat(
                self.synopsis, self.title, panel, location, cast, camera
            )
            panel.text = panel.final
            self.modified.emit(self, "panels")
            self.panels_changed.emit()
            self.progress_text = _(
                "Finalized locally (install ComfyUI-QwenVL for richer polish)"
            )
            log.warning("Comic finalize: Qwen-VL unavailable, used local polish")
            return

        instruction = build_finalize_instruction(
            self.synopsis, self.title, panel, location, cast, camera
        )
        # Tiny blank image — instruction tells Qwen to ignore it (text-only rewrite).
        blank = Image.create(Extent(64, 64), fill=PAGE_BG)
        quantization = DescribeWorkspace._qwenvl_quantization(qwen_model, vram_gb)
        try:
            workflow_input = workflow.prepare_describe(
                blank, "caption", qwen_model, "qwenvl", instruction, quantization
            )
            job_params = JobParams(
                Bounds(0, 0, 64, 64),
                f"[Comic Finalize] panel {index + 1}",
            )
            job_params.metadata["workspace"] = "comic"
            job_params.metadata["comic_finalize_panel"] = index
            job = model.jobs.add(JobKind.describe, job_params)
        except Exception as e:
            model.report_error(util.log_error(e))
            return

        model.clear_error()
        self._finalize_index = index
        self.progress_text = _("Finalizing panel {i}…").format(i=index + 1)
        self.can_generate = False
        eventloop.run(_report_errors(model, model._enqueue_job(job, workflow_input)))

    def handle_finalize_output(self, job, output) -> bool:
        """Apply Qwen finalize text to the panel. Returns True if this was a comic finalize job."""
        idx = job.params.metadata.get("comic_finalize_panel")
        if idx is None:
            return False
        from ..backend.client import TextOutput

        index = int(idx)
        text = ""
        if isinstance(output, TextOutput) and output.text:
            text = output.text.strip()
        if text and 0 <= index < len(self._panels):
            # Strip accidental wrappers some VLMs add.
            if text.lower().startswith("scene description:"):
                text = text.split(":", 1)[1].strip()
            self._panels[index].final = text
            self._panels[index].text = text
            self.modified.emit(self, "panels")
            self.panels_changed.emit()
            log.info("Comic finalize panel %d: %s", index + 1, text[:100])
        self._finalize_index = None
        self.progress_text = ""
        if not self._in_progress:
            self.can_generate = True
        return True

    def finish_finalize_job(self):
        self._finalize_index = None
        self.progress_text = ""
        if not self._in_progress:
            self.can_generate = True

    def assemble_page(self):
        model = ensure(self._model())
        self.reload_all_from_layers()
        self._ensure_panel_images()
        slots = layout_slots(self.layout)
        locked_images: list[Image | None] = []
        for i, slot in enumerate(slots):
            panel = self._panels[i] if i < len(self._panels) else PanelBeat()
            img = self.panel_image(i)
            if panel.locked and img is not None:
                locked_images.append(img)
            else:
                locked_images.append(None)
        if not any(img is not None for img in locked_images):
            model.report_error(_("Lock at least one generated panel before assembling."))
            return
        if any(img is None for img in locked_images):
            model.report_error(
                _("Assemble Page needs every panel locked. Generate and lock remaining panels, "
                  "or use Generate Remaining.")
            )
            return

        page_extent = Extent(PAGE_WIDTH, PAGE_HEIGHT)
        panel_images = [img for img in locked_images if img is not None]
        assert len(panel_images) == len(slots)
        page = self._compose_page(panel_images, slots, page_extent)

        job_name = f"[Comic] {self.title.strip() or 'Page'}"
        job_params = JobParams(Bounds(0, 0, *page_extent), job_name)
        job_params.metadata["workspace"] = "comic"
        job_params.metadata["comic_layout"] = self.layout.name
        job_params.metadata["comic_panel_style"] = self.panel_style.name
        job_params.metadata["comic_mode"] = "assemble"
        job_params.metadata["comic_slots"] = [
            {
                "index": s.index,
                "bounds": [s.bounds.x, s.bounds.y, s.bounds.width, s.bounds.height],
                "role": s.role,
            }
            for s in slots
        ]
        job = model.jobs.add(JobKind.comic, job_params)
        job.id = str(uuid.uuid4())
        job.state = JobState.finished
        model.jobs.set_results(job, ImageCollection([page] + panel_images))
        model.jobs.job_finished.emit(job)
        model.show_preview(job.id, 0)
        self._last_panel_images = panel_images
        self._last_slots = slots
        action = settings.generation_finished_action
        if action is GenerationFinishedAction.preview:
            model.jobs.select(job.id, 0)
        elif action is GenerationFinishedAction.apply:
            self.apply_page(job.id, 0)
        self.progress_text = _("Page assembled")

    def generate_page(
        self, mode: ComicGenerationMode = ComicGenerationMode.final, skip_locked: bool = False
    ):
        model = ensure(self._model())
        ok, msg = model.document.check_color_mode()
        if not ok and msg:
            model.report_error(msg)
            return
        if not any(p.has_notes() for p in self._panels) and not self.synopsis.strip():
            model.report_error(
                _(
                    "Add a page synopsis or location/camera/action notes "
                    "for at least one panel before generating."
                )
            )
            return
        if mode is ComicGenerationMode.refine:
            slots = layout_slots(self.layout)
            # Locked panel images can substitute for draft refs
            has_locked = any(
                p.locked and self.panel_image(i) is not None
                for i, p in enumerate(self._panels)
            )
            if self._resolve_draft_panels(model, len(slots)) is None and not has_locked:
                model.report_error(
                    _("Create a Draft Page first (or select a draft in history) before refining.")
                )
                return

        # When locked panels exist, skip them unless this is a full draft (re-draft everything)
        if mode is not ComicGenerationMode.draft:
            skip_locked = skip_locked or any(p.locked for p in self._panels)

        model.clear_error()
        self.set_in_progress(True)
        from .model import _report_errors

        eventloop.run(_report_errors(model, self._generate_all(model, mode, skip_locked)))

    async def _generate_single(self, model, index: int):
        self._single_panel_job = True
        try:
            slots = layout_slots(self.layout)
            if index >= len(slots):
                return
            slot = slots[index]
            seed = model.seed if model.fixed_seed else workflow.generate_seed()
            self._panels_remaining = 1
            self.progress_text = _("Generating panel {i}…").format(i=index + 1)
            panel_img, _job = await self._generate_one_panel(
                model,
                slot,
                seed,
                is_draft=False,
                is_refine=False,
                draft_ref=None,
                continuity_ref=self._previous_locked_image(index),
                keep_job=False,
            )
            if panel_img is None:
                return
            self._set_panel_image(index, panel_img)
            # Preview just this panel on canvas
            page = create_page_canvas()
            for i, s in enumerate(slots):
                img = self.panel_image(i)
                if img is not None:
                    paste_panel(page, img, s.bounds, self.panel_style)
            job_params = JobParams(Bounds(0, 0, PAGE_WIDTH, PAGE_HEIGHT), f"[Comic] panel {index + 1}")
            job_params.metadata["workspace"] = "comic"
            job_params.metadata["comic_panel_style"] = self.panel_style.name
            job_params.metadata["comic_single_panel"] = index
            job = model.jobs.add(JobKind.comic, job_params)
            job.id = str(uuid.uuid4())
            job.state = JobState.finished
            model.jobs.set_results(job, ImageCollection([page]))
            # Do not emit job_finished — avoid history spam; just preview
            if model._layer and not model._layer.was_removed:
                model._layer.write_pixels(page, Bounds(0, 0, PAGE_WIDTH, PAGE_HEIGHT))
                model._layer.name = f"[Preview] Comic panel {index + 1}"
            else:
                model._layer = model.layers.create(
                    f"[Preview] Comic panel {index + 1}",
                    page,
                    Bounds(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                    make_active=False,
                )
                model._layer.is_locked = True
            # Remove the temporary job from queue
            model.jobs.remove(job)
            self.progress_text = _("Panel {i} ready — lock it when happy").format(i=index + 1)
            self.panel_images_changed.emit()
        except Exception as e:
            model.report_error(util.log_error(e))
        finally:
            self._single_panel_job = False
            self._panels_remaining = 0
            self.set_in_progress(False)

    async def _generate_one_panel(
        self,
        model,
        slot: PanelSlot,
        seed: int,
        *,
        is_draft: bool,
        is_refine: bool,
        draft_ref: Image | None,
        continuity_ref: Image | None,
        keep_job: bool,
        mode_name: str = "final",
    ) -> tuple[Image | None, object | None]:
        client = model._connection.client
        page_extent = Extent(PAGE_WIDTH, PAGE_HEIGHT)
        slots = layout_slots(self.layout, page_extent)
        panel = self._panels[slot.index] if slot.index < len(self._panels) else PanelBeat()
        cast = self._characters_for_panel(panel)
        arch = model.arch

        scene = panel_scene_text(
            panel,
            self.location_for_panel(slot.index),
            self.camera_for_panel(slot.index),
        )
        prompt = build_panel_prompt(
            self.synopsis,
            scene,
            slot.index,
            len(slots),
            slot.role,
            self.title,
            cast,
        )
        # Scene plate = locked set. Prefer img2img over the plate (not vision ref —
        # vision refs of full environments often become diptychs on Krea2).
        plate_img = None if (is_refine and draft_ref is not None) else self.scene_plate_image_for_panel(
            slot.index
        )
        if plate_img is not None:
            prompt += (
                " Keep the environment, architecture, props and lighting from the background"
                " plate. Integrate the characters and action naturally into that set."
            )
        log.info("Comic panel %d/%d: %s", slot.index + 1, len(slots), prompt[:80])

        controls: list[ControlInput] = []
        # Krea2 comic: at most one face image (extra faces become Picture 2 → diptychs).
        max_faces = 1 if arch is Arch.krea2 else MAX_FACE_REFS_PER_PANEL
        face_added = 0
        for ch in cast:
            if face_added >= max_faces:
                break
            face_img = self._layer_image(ch.face_layer_id)
            if face_img is not None:
                controls.append(ControlInput(ControlMode.face, face_img, 1.1))
                face_added += 1
        # Do not attach Style as a vision ref — sample pages / moodboards make
        # Krea2 (and similar VLMs) paint multi-panel layouts into every panel.
        # Art direction stays via style LoRAs + prompt; Face/continuity still work.
        if self.style_layer_id:
            log.info("Comic: Style layer not used as vision ref (use Face/LoRAs for look)")
        if is_refine and draft_ref is not None:
            controls.append(ControlInput(ControlMode.reference, draft_ref, 0.65))
        elif plate_img is None and continuity_ref is not None and arch is Arch.krea2:
            # Skip previous-panel continuity when a scene plate drives the canvas.
            controls.append(ControlInput(ControlMode.reference, continuity_ref, 0.55))
        if arch is Arch.krea2:
            controls = _limit_krea2_vision_refs(controls)
            if controls:
                prompt += (
                    " If a reference picture is attached, use it only for character identity — "
                    "do not copy any multi-panel layout from it."
                )
        elif len(controls) > 4:
            controls = controls[:4]

        cast_loras = self._collect_loras(cast)
        prompt_with_triggers = prompt + _lora_trigger_suffix(cast_loras)
        gen_extent = generation_extent_for_slot(slot.bounds, draft=is_draft)
        log.info(
            "Comic panel %d vision_refs=%d extent=%sx%s arch=%s plate=%s",
            slot.index + 1,
            len(controls),
            gen_extent.width,
            gen_extent.height,
            arch.name,
            plate_img is not None,
        )
        conditioning = ConditioningInput(
            positive=prompt_with_triggers, negative=COMIC_NEGATIVE, control=controls
        )
        conditioning.language = model.prompt_translation_language
        conditioning, style_loras, prompt_meta = workflow.prepare_prompts(
            conditioning, model.style, seed, arch, None, None, is_live=is_draft
        )
        loras = unique(cast_loras + style_loras, key=lambda l: l.name)
        if plate_img is not None:
            plate_scaled = Image.scale(plate_img, gen_extent)
            strength = SCENE_PLATE_STRENGTH_DRAFT if is_draft else SCENE_PLATE_STRENGTH
            workflow_input = workflow.prepare(
                WorkflowKind.refine,
                plate_scaled,
                conditioning,
                model.style,
                seed,
                client.models,
                FileLibrary.instance(),
                model._performance_settings(client),
                strength=strength,
                loras=loras,
                is_live=is_draft,
                layer_count=1,
            )
        else:
            workflow_input = workflow.prepare(
                WorkflowKind.generate,
                gen_extent,
                conditioning,
                model.style,
                seed,
                client.models,
                FileLibrary.instance(),
                model._performance_settings(client),
                strength=1.0,
                loras=loras,
                is_live=is_draft,
                layer_count=1,
            )
        # Performance batch sizing can request 2+ images; comic needs exactly one frame.
        workflow_input.batch_count = 1
        if workflow_input.images is not None:
            workflow_input.images.layer_count = 1

        prefix = "[Comic Draft]" if is_draft else "[Comic]"
        job_name = (
            f"{prefix} {self.title.strip() or 'Page'} — "
            f"panel {slot.index + 1}/{len(slots)}"
        )
        job_params = JobParams(Bounds(0, 0, *page_extent), job_name)
        job_params.set_style(model.style, ensure(workflow_input.models).checkpoint)
        job_params.metadata.update(prompt_meta)
        job_params.metadata["workspace"] = "comic"
        job_params.metadata["comic_panel"] = slot.index
        job_params.metadata["comic_layout"] = self.layout.name
        job_params.metadata["comic_panel_style"] = self.panel_style.name
        job_params.metadata["comic_mode"] = mode_name
        if not keep_job:
            job_params.metadata["comic_single_panel"] = slot.index
        job_params.metadata["loras"] = [{"name": l.name, "weight": l.strength} for l in loras]
        job = model.jobs.add(JobKind.comic, job_params)
        await model._enqueue_job(job, workflow_input)

        while job.state not in (JobState.finished, JobState.cancelled):
            await asyncio.sleep(0.05)
        if job.state is not JobState.finished or len(job.results) == 0:
            if not model.error:
                model.report_error(_("Comic generation failed on a panel."))
            return None, None

        if len(job.results) > 1:
            log.warning(
                "Comic panel returned %d images; using the first only (batch forced to 1)",
                len(job.results),
            )
        panel_img = job.results[0]
        if not keep_job:
            try:
                model.jobs.remove(job)
            except Exception:
                pass
            return panel_img, None
        return panel_img, job

    async def _generate_all(
        self, model, mode: ComicGenerationMode, skip_locked: bool = False
    ):
        page_extent = Extent(PAGE_WIDTH, PAGE_HEIGHT)
        slots = layout_slots(self.layout, page_extent)
        page = create_page_canvas(page_extent)
        panel_images: list[Image] = []
        last_job = None
        seed = model.seed if model.fixed_seed else workflow.generate_seed()
        is_draft = mode is ComicGenerationMode.draft
        is_refine = mode is ComicGenerationMode.refine
        draft_panels = self._resolve_draft_panels(model, len(slots)) if is_refine else None
        self._ensure_panel_images()

        # Count jobs we will actually run
        to_generate = [
            s
            for s in slots
            if not (
                skip_locked
                and s.index < len(self._panels)
                and self._panels[s.index].locked
                and self.panel_image(s.index) is not None
            )
        ]

        try:
            self._panels_remaining = max(1, len(to_generate))
            for slot in slots:
                panel = self._panels[slot.index] if slot.index < len(self._panels) else PanelBeat()
                existing = self.panel_image(slot.index)
                if (
                    skip_locked
                    and panel.locked
                    and existing is not None
                ):
                    panel_images.append(existing)
                    page = paste_panel(page, existing, slot.bounds, self.panel_style)
                    continue

                if is_draft:
                    self.progress_text = _("Drafting panel {i} of {n}…").format(
                        i=slot.index + 1, n=len(slots)
                    )
                elif is_refine:
                    self.progress_text = _("Refining panel {i} of {n}…").format(
                        i=slot.index + 1, n=len(slots)
                    )
                else:
                    self.progress_text = _("Generating panel {i} of {n}…").format(
                        i=slot.index + 1, n=len(slots)
                    )

                draft_ref = None
                if is_refine:
                    if draft_panels is not None and slot.index < len(draft_panels):
                        draft_ref = draft_panels[slot.index]
                    elif existing is not None:
                        draft_ref = existing

                continuity = self._previous_locked_image(slot.index)
                # Fall back to previous image in this run if no locked prior
                if continuity is None and panel_images:
                    continuity = panel_images[-1]

                is_last = bool(to_generate) and slot.index == to_generate[-1].index
                panel_img, job = await self._generate_one_panel(
                    model,
                    slot,
                    seed,
                    is_draft=is_draft,
                    is_refine=is_refine,
                    draft_ref=draft_ref,
                    continuity_ref=continuity if not (is_refine and draft_ref) else None,
                    keep_job=is_last,
                    mode_name=mode.name,
                )
                if panel_img is None:
                    return
                if is_last and job is not None:
                    last_job = job

                panel_images.append(panel_img)
                self._set_panel_image(slot.index, panel_img)
                page = paste_panel(page, panel_img, slot.bounds, self.panel_style)
                seed = (seed + 1) & 0xFFFFFFFFFFFFFFFF

            if last_job is None:
                # All panels were locked — assemble without a diffusion job
                self._last_panel_images = panel_images
                self._last_slots = slots
                job_params = JobParams(
                    Bounds(0, 0, *page_extent),
                    f"[Comic] {self.title.strip() or 'Page'}",
                )
                job_params.metadata["workspace"] = "comic"
                job_params.metadata["comic_layout"] = self.layout.name
                job_params.metadata["comic_panel_style"] = self.panel_style.name
                job_params.metadata["comic_slots"] = [
                    {
                        "index": s.index,
                        "bounds": [s.bounds.x, s.bounds.y, s.bounds.width, s.bounds.height],
                        "role": s.role,
                    }
                    for s in slots
                ]
                last_job = model.jobs.add(JobKind.comic, job_params)
                last_job.id = str(uuid.uuid4())
                last_job.state = JobState.finished

            assert last_job is not None and last_job.id
            self._last_panel_images = panel_images
            self._last_slots = slots
            last_job.params.metadata["comic_panel_style"] = self.panel_style.name
            last_job.params.metadata["comic_slots"] = [
                {
                    "index": s.index,
                    "bounds": [s.bounds.x, s.bounds.y, s.bounds.width, s.bounds.height],
                    "role": s.role,
                }
                for s in slots
            ]
            if is_draft:
                last_job.params.metadata["comic_draft"] = True
                last_job.params.name = f"[Comic Draft] {self.title.strip() or 'Page'}"
                self._draft_panel_images = list(panel_images)
                self._draft_job_id = last_job.id
            else:
                last_job.params.metadata.pop("comic_draft", None)
                last_job.params.name = f"[Comic] {self.title.strip() or 'Page'}"
                if is_refine:
                    self._draft_panel_images = []
                    self._draft_job_id = ""

            model.jobs.set_results(last_job, ImageCollection([page] + panel_images))
            model.jobs.job_finished.emit(last_job)
            model.show_preview(last_job.id, 0)
            action = settings.generation_finished_action
            if action is GenerationFinishedAction.preview:
                model.jobs.select(last_job.id, 0)
            elif action is GenerationFinishedAction.apply:
                self.apply_page(last_job.id, 0)
            self.progress_text = _("Draft complete") if is_draft else _("Page complete")
        except Exception as e:
            model.report_error(util.log_error(e))
        finally:
            self._panels_remaining = 0
            self.set_in_progress(False)
            self.refresh_can_refine()
            self.refresh_can_assemble()

    def apply_page(self, job_id: str, index: int = 0):
        """Apply a finished comic job as a layer group: full page + panel layers."""
        model = ensure(self._model())
        job = model.jobs.find(job_id)
        if job is None or job.kind is not JobKind.comic or len(job.results) == 0:
            model.report_error(_("No comic page result to apply."))
            return

        page = job.results[0]
        page_extent = page.extent
        page_bounds = Bounds(0, 0, *page_extent)

        slots_data = job.params.metadata.get("comic_slots") or []
        if slots_data:
            slots = [
                PanelSlot(
                    int(s["index"]),
                    Bounds(*s["bounds"]),
                    s.get("role", "panel"),
                )
                for s in slots_data
            ]
        else:
            slots = self._last_slots or layout_slots(self.layout, page_extent)

        panel_images = list(job.results[1:]) if len(job.results) > 1 else self._last_panel_images

        title = self.title.strip() or trim_text(job.params.name, 40)
        draft = bool(job.params.metadata.get("comic_draft"))
        group_name = f"[Comic Draft] {title}" if draft else f"[Comic] {title}"
        group = model.layers.create_group(group_name)
        model.layers.create(
            "Page",
            page,
            page_bounds,
            make_active=False,
            parent=group,
        )
        for i, slot in enumerate(slots):
            if i >= len(panel_images):
                break
            panel = Image.scale(panel_images[i], slot.bounds.extent)
            model.layers.create(
                f"Panel {slot.index + 1}",
                panel,
                slot.bounds,
                make_active=False,
                parent=group,
            )

        if model._layer:
            model._layer.remove()
            model._layer = None
        model.jobs.selection = []
        model.jobs.notify_used(job_id, index)
