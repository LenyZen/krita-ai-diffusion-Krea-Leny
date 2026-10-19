"""Krea Edit workspace — identity-preserving instruction edit via comfyui-krea2edit."""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, pyqtSignal

from .. import eventloop
from ..backend.api import CheckpointInput, KreaEditInput, LoraInput
from ..backend import workflow as workflow_module
from ..backend.resources import Arch
from ..backend.workflow import (
    estimate_krea_edit_extent,
    find_identity_edit_lora,
    prepare_krea_edit,
)
from ..document import SelectionModifiers
from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..style import Styles
from ..util import ensure
from .jobs import JobKind, JobParams
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from ..backend.client import Client
    from .model import DocumentModel

log = logging.getLogger(__name__)

PRESET_TURBO = "turbo"
PRESET_RAW = "raw"
MAX_EDIT_LORAS = 8
MAX_EDIT_SLIDERS = 8
# LoRAs whose path contains this (case-insensitive) show in the Sliders UI.
# Create: ComfyUI/models/loras/krea/sliders/  (or via extra_model_paths)
SLIDERS_PATH_FRAGMENT = "krea/sliders"
ROLE_SHEET_ORDER = ("face", "hair", "clothing", "pose")
ROLE_SHEET_LABELS = {
    "face": "face / facial identity",
    "hair": "hair",
    "clothing": "clothing / outfit",
    "pose": "pose / body language",
}
_COLLAGE_CELL = 768

# Official identity-edit graph defaults; Raw trades speed for fidelity.
PRESET_SETTINGS = {
    PRESET_TURBO: dict(steps=10, cfg_scale=1.0, sampler="euler", scheduler="simple"),
    PRESET_RAW: dict(steps=24, cfg_scale=3.0, sampler="euler", scheduler="simple"),
}

# Placeholder / tip copy — kept in the model so UI and assembly stay aligned.
PLACEHOLDER_CHARACTER = (
    "Change her outfit to a red raincoat and keep the same body proportions.\n"
    "Preserve the exact facial identity — same eyes, nose, mouth, and hair."
)
PLACEHOLDER_SCENE = (
    "Replace the background with a rainy Tokyo street at night.\n"
    "Keep the subject in the same place and scale in the frame."
)
PLACEHOLDER_LIGHTING = (
    "Relight with soft warm golden-hour sunlight from camera-left.\n"
    "Add gentle rim light on the hair; keep shadows natural."
)
PLACEHOLDER_OTHER = (
    "Optional extras — mood, camera, props, weather details…\n"
    "e.g. Add subtle motion blur on the falling rain."
)
PLACEHOLDER_REF_NOTES = (
    "Optional: how to read the refs — e.g.\n"
    "Use the Face sheet for identity and the Hair sheet for the new style."
)

# Starter prompts for the Edit tab dropdowns — (short label, full instruction text).
# Selecting one fills the text box so the user can edit from a working example.
FIELD_PRESETS: dict[str, list[tuple[str, str]]] = {
    "character": [
        (
            "Outfit swap — raincoat",
            "Change her outfit to a red raincoat and keep the same body proportions.\n"
            "Preserve the exact facial identity — same eyes, nose, mouth, and hair.",
        ),
        (
            "Outfit swap — formal suit",
            "Change the outfit to a tailored black formal suit with a white shirt.\n"
            "Keep the same face, hair, body proportions, and pose.",
        ),
        (
            "Hair restyle",
            "Change the hair to long wavy auburn hair past the shoulders.\n"
            "Preserve the exact facial identity and keep the same outfit and pose.",
        ),
        (
            "Expression change",
            "Change the expression to a soft genuine smile looking at camera.\n"
            "Keep the same face structure, hair, outfit, and body pose.",
        ),
        (
            "Age down slightly",
            "Make the subject look a few years younger while keeping the same person.\n"
            "Preserve facial identity, hair style, outfit, and pose.",
        ),
        (
            "Age up slightly",
            "Make the subject look a few years older with subtle mature features.\n"
            "Preserve who they are — same face structure, hair, outfit, and pose.",
        ),
        (
            "Add accessory",
            "Add thin round wire-frame glasses and keep everything else the same.\n"
            "Preserve exact facial identity, hair, outfit, and pose.",
        ),
        (
            "Pose adjust",
            "Adjust the pose so the subject stands slightly turned three-quarter to camera-left.\n"
            "Keep the same face, hair, outfit, and body proportions.",
        ),
        (
            "Makeup / look",
            "Apply natural soft glam makeup with a warm lip tint.\n"
            "Preserve exact facial identity and keep hair, outfit, and pose unchanged.",
        ),
        (
            "Clothing color only",
            "Change only the clothing color to deep forest green; keep the same garment shape.\n"
            "Preserve face, hair, pose, and body proportions exactly.",
        ),
    ],
    "scene": [
        (
            "Tokyo rain night",
            "Replace the background with a rainy Tokyo street at night.\n"
            "Keep the subject in the same place and scale in the frame.",
        ),
        (
            "Studio seamless",
            "Replace the background with a clean light-gray photography studio seamless.\n"
            "Keep the subject centered at the same scale with no environment props.",
        ),
        (
            "Forest path",
            "Replace the background with a misty forest path in soft daylight.\n"
            "Keep the subject in the same place and scale in the frame.",
        ),
        (
            "Beach sunset",
            "Replace the background with a calm beach at sunset, soft waves and warm sky.\n"
            "Keep the subject standing in the same place and scale.",
        ),
        (
            "Cozy interior",
            "Replace the background with a cozy warm living room with soft lamps and bookshelves.\n"
            "Keep the subject in the same place and scale in the frame.",
        ),
        (
            "City rooftop",
            "Replace the background with a city rooftop at dusk, skyline soft in the distance.\n"
            "Keep the subject in the same place and scale.",
        ),
        (
            "Café window",
            "Replace the background with a sunlit café interior by a large window.\n"
            "Keep the subject seated/standing in the same place and scale in the frame.",
        ),
        (
            "Remove clutter",
            "Simplify the background: remove distracting clutter and keep a clean soft bokeh.\n"
            "Do not move the subject; keep the same place and scale.",
        ),
        (
            "Snow outdoor",
            "Replace the background with a quiet snowy outdoor street, soft falling snow.\n"
            "Keep the subject in the same place and scale in the frame.",
        ),
        (
            "Neon alley",
            "Replace the background with a narrow neon-lit alley, wet pavement reflections.\n"
            "Keep the subject in the same place and scale.",
        ),
    ],
    "lighting": [
        (
            "Golden hour",
            "Relight with soft warm golden-hour sunlight from camera-left.\n"
            "Add gentle rim light on the hair; keep shadows natural.",
        ),
        (
            "Soft overcast",
            "Relight with soft even overcast daylight, low contrast shadows.\n"
            "Keep skin tones natural and avoid hard specular highlights.",
        ),
        (
            "Studio softbox",
            "Relight with a large softbox key light from camera-right and a subtle fill.\n"
            "Keep a clean portrait look with controlled soft shadows.",
        ),
        (
            "Dramatic Rembrandt",
            "Relight with dramatic Rembrandt lighting: strong key from upper camera-left.\n"
            "Deepen shadows on the far cheek; keep the face readable.",
        ),
        (
            "Neon night",
            "Relight with cool cyan and magenta neon bounce as if from night signs.\n"
            "Keep face identity clear; avoid crushing shadows to pure black.",
        ),
        (
            "Window light",
            "Relight as soft window light from camera-left, gentle falloff across the face.\n"
            "Keep natural skin and a calm indoor mood.",
        ),
        (
            "Backlit rim",
            "Relight with strong backlight creating a bright rim on hair and shoulders.\n"
            "Add soft front fill so the face stays clear and not silhouetted.",
        ),
        (
            "Candle / warm low",
            "Relight with warm low candle-like light from below-front, soft and intimate.\n"
            "Keep exposure on the face; avoid muddy underexposure.",
        ),
        (
            "High key bright",
            "Relight to a bright high-key look with minimal shadows and airy highlights.\n"
            "Keep skin natural — not blown out or plastic.",
        ),
        (
            "Moody low key",
            "Relight to a low-key moody look with a narrow key light and deep soft shadows.\n"
            "Keep eyes and facial identity clearly lit.",
        ),
    ],
    "other": [
        (
            "Shallow DOF",
            "Add shallow depth of field with soft creamy background bokeh.\n"
            "Keep the subject sharp; do not change identity or outfit.",
        ),
        (
            "Film still mood",
            "Give it a cinematic film-still mood with subtle grain and gentle color grade.\n"
            "Keep the same composition, face, and clothing.",
        ),
        (
            "Anamorphic feel",
            "Add a subtle anamorphic cinematic feel with soft horizontal highlights.\n"
            "Do not change the subject identity or framing.",
        ),
        (
            "Motion blur rain",
            "Add subtle motion blur on falling rain or moving background elements only.\n"
            "Keep the subject sharp and unchanged.",
        ),
        (
            "Camera slightly lower",
            "Shift the camera feel slightly lower (subtle hero angle) without restaging the pose.\n"
            "Preserve face identity and outfit exactly.",
        ),
        (
            "Closer crop feel",
            "Tighten the framing slightly toward a medium close-up while keeping the same pose.\n"
            "Preserve identity; do not invent a new background.",
        ),
        (
            "Cleaner skin",
            "Gently clean minor skin blemishes while keeping natural skin texture.\n"
            "Do not change face shape, age, or identity.",
        ),
        (
            "Color grade teal-orange",
            "Apply a subtle teal-and-orange cinematic color grade.\n"
            "Keep skin tones believable; do not change content or identity.",
        ),
        (
            "Add light fog",
            "Add a light atmospheric haze/fog in the mid-distance only.\n"
            "Keep the subject clear and sharp in the foreground.",
        ),
        (
            "Remove watermark / text",
            "Remove any watermark, logo, or stray text in the frame and reconstruct cleanly behind it.\n"
            "Do not change the subject or overall composition.",
        ),
    ],
}


def assemble_edit_instruction(
    character: str = "",
    scene: str = "",
    lighting: str = "",
    other: str = "",
    legacy_prompt: str = "",
    ref_guidance: str = "",
) -> str:
    """Build one grounded edit instruction from directed fields + ref guidance.

    Empty sections are skipped. A legacy single-box prompt is used only when all
    directed fields are empty (backward compatible with older document state).
    """
    parts: list[str] = []
    if (ref_guidance or "").strip():
        parts.append(ref_guidance.strip())
    for text in (character, scene, lighting, other):
        cleaned = (text or "").strip()
        if cleaned:
            parts.append(cleaned)
    if parts:
        return " ".join(parts)
    return (legacy_prompt or "").strip()


def _panel_position_labels(count: int) -> list[str]:
    """Human-readable panel positions for a 1 / 2 / 2×2 collage."""
    if count <= 1:
        return ["full frame"]
    if count == 2:
        return ["left panel", "right panel"]
    # 3–4 in reading order: TL, TR, BL, BR
    names = ["top-left panel", "top-right panel", "bottom-left panel", "bottom-right panel"]
    return names[:count]


def collage_role_sheets(panels: list[tuple[str, Image]], cell: int = _COLLAGE_CELL) -> Image:
    """Stitch role sheet images into one secondary ref (Identity Edit max = 2 images)."""
    if not panels:
        raise ValueError("collage_role_sheets requires at least one panel")
    if len(panels) == 1:
        return panels[0][1]

    n = len(panels)
    cols = 2
    rows = 1 if n <= 2 else 2
    # Dark letterbox so empty cells / margins don't look like white content.
    canvas = Image.create(Extent(cols * cell, rows * cell), fill=0xFF1A1A1A)
    for i, (_role, img) in enumerate(panels):
        col, row = i % cols, i // cols
        fitted = Image.scale_to_fit(img, Extent(cell, cell))
        ox = col * cell + (cell - fitted.width) // 2
        oy = row * cell + (cell - fitted.height) // 2
        canvas.draw_image(fitted, (ox, oy))
    return canvas


def assemble_ref_guidance(
    *,
    has_subject: bool,
    has_sheet: bool,
    role_panels: list[str] | None = None,
    take_face: bool = True,
    take_hair: bool = False,
    take_clothing: bool = False,
    take_pose: bool = False,
    notes: str = "",
) -> str:
    """Prompt text telling Identity Edit what to pull from the 2nd reference."""
    roles = [r for r in (role_panels or []) if r]
    if not has_subject and not has_sheet and not roles:
        return ""

    parts: list[str] = []
    if roles:
        positions = _panel_position_labels(len(roles))
        layout = ", ".join(
            f"{positions[i]} = {ROLE_SHEET_LABELS.get(r, r)}" for i, r in enumerate(roles)
        )
        takes = [ROLE_SHEET_LABELS.get(r, r) for r in roles]
        parts.append(
            f"The secondary reference is a character sheet collage. Panel layout: {layout}. "
            f"From those panels, use: {', '.join(takes)}. "
            f"Keep the canvas composition unless the edit instruction says otherwise."
        )
    else:
        source = "reference sheet" if has_sheet else "subject reference"
        takes: list[str] = []
        if take_face:
            takes.append("face / facial identity")
        if take_hair:
            takes.append("hair")
        if take_clothing:
            takes.append("clothing / outfit")
        if take_pose:
            takes.append("pose / body language")
        if not takes:
            takes.append("overall identity and appearance")
        parts.append(
            f"From the {source}, use: {', '.join(takes)}. "
            f"Keep the canvas composition unless the edit instruction says otherwise."
        )
    if (notes or "").strip():
        parts.append(notes.strip())
    return " ".join(parts)


def is_slider_lora_path(path: str) -> bool:
    """True if a LoRA id lives under loras/.../krea/sliders/."""
    key = (path or "").replace("\\", "/").lower()
    return SLIDERS_PATH_FRAGMENT in key


@dataclass
class KreaEditLora:
    """Optional style/character LoRA stacked on top of the Identity Edit adapter."""

    name: str = ""
    strength: float = 1.0

    def to_dict(self):
        return {"name": self.name, "strength": self.strength}

    @staticmethod
    def from_dict(data: dict):
        return KreaEditLora(data.get("name", ""), float(data.get("strength", 1.0)))

    def as_input(self) -> LoraInput | None:
        if not self.name:
            return None
        return LoraInput(self.name, self.strength)


class KreaEditWorkspace(QObject, ObservableProperties):
    """Instruction-based edit that preserves identity (Krea2 Identity Edit LoRA)."""

    # Directed edit fields (preferred). `prompt` kept for older documents.
    character = Property("", persist=True)
    scene = Property("", persist=True)
    lighting = Property("", persist=True)
    other = Property("", persist=True)
    prompt = Property("", persist=True)  # legacy single instruction / mirror of assembled
    # Balanced defaults — ref_boost 4 reconstructs the source too hard for most edits.
    ref_boost = Property(2.0, persist=True)
    grounding_px = Property(512, persist=True)
    preset = Property(PRESET_TURBO, persist=True)
    use_active_layer = Property(False, persist=True)
    # Image refs (preferred over Face LoRA). Role sheets collage → full sheet → subject.
    subject_layer_id = Property("", persist=True)
    sheet_layer_id = Property("", persist=True)  # one full collage layer
    sheet_face_layer_id = Property("", persist=True)
    sheet_hair_layer_id = Property("", persist=True)
    sheet_clothing_layer_id = Property("", persist=True)
    sheet_pose_layer_id = Property("", persist=True)
    ref_layer_id = Property("", persist=True)  # resolved 2nd ref (compat / job wiring)
    ref_take_face = Property(True, persist=True)
    ref_take_hair = Property(False, persist=True)
    ref_take_clothing = Property(False, persist=True)
    ref_take_pose = Property(False, persist=True)
    ref_notes = Property("", persist=True)
    # Inpaint crop/stitch (Krita selection → local Identity Edit)
    use_selection_inpaint = Property(True, persist=True)  # auto when selection exists
    mask_expand_pixels = Property(8, persist=True)
    mask_blend_pixels = Property(32, persist=True)
    context_factor = Property(1.5, persist=True)
    # Role-scoped LoRAs — Advanced only; image refs are the primary identity path
    face_lora = Property("", persist=True)
    face_lora_strength = Property(1.0, persist=True)
    body_lora = Property("", persist=True)
    body_lora_strength = Property(1.0, persist=True)
    style_lora = Property("", persist=True)
    style_lora_strength = Property(1.0, persist=True)
    can_generate = Property(True)

    character_changed = pyqtSignal(str)
    scene_changed = pyqtSignal(str)
    lighting_changed = pyqtSignal(str)
    other_changed = pyqtSignal(str)
    prompt_changed = pyqtSignal(str)
    ref_boost_changed = pyqtSignal(float)
    grounding_px_changed = pyqtSignal(int)
    preset_changed = pyqtSignal(str)
    use_active_layer_changed = pyqtSignal(bool)
    subject_layer_id_changed = pyqtSignal(str)
    sheet_layer_id_changed = pyqtSignal(str)
    sheet_face_layer_id_changed = pyqtSignal(str)
    sheet_hair_layer_id_changed = pyqtSignal(str)
    sheet_clothing_layer_id_changed = pyqtSignal(str)
    sheet_pose_layer_id_changed = pyqtSignal(str)
    ref_layer_id_changed = pyqtSignal(str)
    ref_take_face_changed = pyqtSignal(bool)
    ref_take_hair_changed = pyqtSignal(bool)
    ref_take_clothing_changed = pyqtSignal(bool)
    ref_take_pose_changed = pyqtSignal(bool)
    ref_notes_changed = pyqtSignal(str)
    use_selection_inpaint_changed = pyqtSignal(bool)
    mask_expand_pixels_changed = pyqtSignal(int)
    mask_blend_pixels_changed = pyqtSignal(int)
    context_factor_changed = pyqtSignal(float)
    face_lora_changed = pyqtSignal(str)
    face_lora_strength_changed = pyqtSignal(float)
    body_lora_changed = pyqtSignal(str)
    body_lora_strength_changed = pyqtSignal(float)
    style_lora_changed = pyqtSignal(str)
    style_lora_strength_changed = pyqtSignal(float)
    can_generate_changed = pyqtSignal(bool)
    loras_changed = pyqtSignal()
    sliders_changed = pyqtSignal()
    status_changed = pyqtSignal(str)
    target_extent_changed = pyqtSignal(Extent)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: DocumentModel):
        super().__init__()
        self._model = weakref.ref(model)
        self._in_progress = False
        self._status = ""
        self._loras: list[KreaEditLora] = []
        self._sliders: list[KreaEditLora] = []

    @property
    def status(self) -> str:
        return self._status

    @property
    def loras(self) -> list[KreaEditLora]:
        return list(self._loras)

    @property
    def sliders(self) -> list[KreaEditLora]:
        return list(self._sliders)

    def role_sheet_entries(self) -> list[tuple[str, str]]:
        """Non-empty (role, layer_id) pairs in collage order."""
        mapping = {
            "face": self.sheet_face_layer_id,
            "hair": self.sheet_hair_layer_id,
            "clothing": self.sheet_clothing_layer_id,
            "pose": self.sheet_pose_layer_id,
        }
        out: list[tuple[str, str]] = []
        for role in ROLE_SHEET_ORDER:
            lid = (mapping.get(role) or "").strip()
            if lid:
                out.append((role, lid))
        return out

    def assembled_instruction(self) -> str:
        return assemble_edit_instruction(
            self.character,
            self.scene,
            self.lighting,
            self.other,
            self.prompt,
            self.assembled_ref_guidance(),
        )

    def assembled_ref_guidance(self) -> str:
        roles = [r for r, _ in self.role_sheet_entries()]
        sheet = bool((self.sheet_layer_id or "").strip()) and not roles
        subject = bool((self.subject_layer_id or self.ref_layer_id or "").strip()) and not roles and not sheet
        return assemble_ref_guidance(
            has_subject=subject,
            has_sheet=sheet,
            role_panels=roles,
            take_face=bool(self.ref_take_face),
            take_hair=bool(self.ref_take_hair),
            take_clothing=bool(self.ref_take_clothing),
            take_pose=bool(self.ref_take_pose),
            notes=self.ref_notes or "",
        )

    def resolve_secondary_ref_id(self) -> str:
        """Role sheets / full sheet win over subject; keeps ref_layer_id in sync."""
        roles = self.role_sheet_entries()
        sheet = (self.sheet_layer_id or "").strip()
        subject = (self.subject_layer_id or "").strip()
        legacy = (self.ref_layer_id or "").strip()
        if roles:
            # First role layer stands in for "has a 2nd ref" checks; collage built at generate.
            chosen = roles[0][1]
        else:
            chosen = sheet or subject or legacy
        if chosen != self.ref_layer_id:
            self.ref_layer_id = chosen
        return chosen

    def migrate_legacy_prompt(self):
        """Move a saved single-box prompt into Other when directed fields are empty."""
        if (self.prompt or "").strip() and not any(
            [
                (self.character or "").strip(),
                (self.scene or "").strip(),
                (self.lighting or "").strip(),
                (self.other or "").strip(),
            ]
        ):
            self.other = self.prompt
        # Old default (ref 4 / ground 768) over-copies the source; nudge once to Balanced.
        if float(self.ref_boost) == 4.0 and int(self.grounding_px) == 768:
            self.ref_boost = 2.0
            self.grounding_px = 512
        # Migrate legacy single 2nd-ref slot → subject
        if (self.ref_layer_id or "").strip() and not (self.subject_layer_id or "").strip():
            if not (self.sheet_layer_id or "").strip() and not self.role_sheet_entries():
                self.subject_layer_id = self.ref_layer_id

    def _set_status(self, text: str):
        self._status = text
        self.status_changed.emit(text)

    def set_in_progress(self, in_progress: bool):
        self._in_progress = in_progress
        self.can_generate = not in_progress

    def add_lora(self) -> bool:
        if len(self._loras) >= MAX_EDIT_LORAS:
            return False
        self._loras.append(KreaEditLora())
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

    def set_loras(self, loras: list[KreaEditLora]):
        self._loras = [l for l in loras if l.name][:MAX_EDIT_LORAS]
        self.loras_changed.emit()
        self.modified.emit(self, "loras")

    def load_loras(self, data: list[dict]):
        self._loras = [KreaEditLora.from_dict(d) for d in data][:MAX_EDIT_LORAS]
        self.loras_changed.emit()

    def add_slider(self, name: str = "", strength: float = 0.0) -> bool:
        if len(self._sliders) >= MAX_EDIT_SLIDERS:
            return False
        # Default 0 = bar centered (off) until the user slides.
        self._sliders.append(KreaEditLora(name=name, strength=strength))
        self.sliders_changed.emit()
        self.modified.emit(self, "sliders")
        return True

    def remove_slider(self, index: int):
        if 0 <= index < len(self._sliders):
            del self._sliders[index]
            self.sliders_changed.emit()
            self.modified.emit(self, "sliders")

    def update_slider(self, index: int, name: str, strength: float):
        if 0 <= index < len(self._sliders):
            self._sliders[index].name = name
            self._sliders[index].strength = strength
            self.modified.emit(self, "sliders")

    def set_sliders(self, sliders: list[KreaEditLora]):
        self._sliders = [l for l in sliders if l.name][:MAX_EDIT_SLIDERS]
        self.sliders_changed.emit()
        self.modified.emit(self, "sliders")

    def load_sliders(self, data: list[dict]):
        self._sliders = [KreaEditLora.from_dict(d) for d in data][:MAX_EDIT_SLIDERS]
        self.sliders_changed.emit()

    def _extra_lora_inputs(self) -> list[LoraInput]:
        result: list[LoraInput] = []
        # Sliders first — they tend to steer the edit subtly and stack with style LoRAs.
        for entry in self._sliders:
            if not entry.name or abs(float(entry.strength)) < 1e-6:
                continue  # bar at center (0) = off
            if inp := entry.as_input():
                result.append(inp)
        for entry in self._loras:
            if inp := entry.as_input():
                result.append(inp)
        return result

    def _role_lora_inputs(self) -> dict[str, LoraInput | None]:
        def make(name: str, strength: float) -> LoraInput | None:
            name = (name or "").strip()
            return LoraInput(name, float(strength)) if name else None

        return {
            "face": make(self.face_lora, self.face_lora_strength),
            "body": make(self.body_lora, self.body_lora_strength),
            "style": make(self.style_lora, self.style_lora_strength),
        }

    def _layer_pixels(self, model: DocumentModel, layer_id: str) -> Image:
        ref_layer = next((l for l in model.layers.images if l.id_string == layer_id), None)
        if ref_layer is None:
            raise RuntimeError(_("Reference layer not found."))
        rb = ref_layer.compute_bounds()
        if rb.is_zero:
            raise RuntimeError(_("The reference layer is empty."))
        return ref_layer.get_pixels(rb)

    def build_secondary_ref(self, model: DocumentModel) -> tuple[Image | None, str]:
        """Build image_b: role-sheet collage, or full sheet, or subject.

        Returns (image_or_none, ref_kind) where ref_kind is
        'roles' | 'sheet' | 'subject' | ''.
        """
        roles = self.role_sheet_entries()
        if roles:
            panels: list[tuple[str, Image]] = []
            for role, lid in roles:
                panels.append((role, self._layer_pixels(model, lid)))
            return collage_role_sheets(panels), "roles"

        secondary_id = self.resolve_secondary_ref_id()
        if not secondary_id:
            return None, ""

        pixels = self._layer_pixels(model, secondary_id)
        # Fit onto document-sized canvas (same behavior as before)
        ref_b = Image.create(model.document.extent)
        # Center the crop content; bounds were relative to document so re-get with offset
        ref_layer = next(l for l in model.layers.images if l.id_string == secondary_id)
        rb = ref_layer.compute_bounds()
        ref_b.draw_image(pixels, (rb.x, rb.y))
        sheet = (self.sheet_layer_id or "").strip()
        kind = "sheet" if sheet == secondary_id else "subject"
        return ref_b, kind

    @staticmethod
    def pack_installed(client: Client) -> bool:
        nodes = client.models.node_inputs
        return "Krea2EditModelPatch" in nodes and "Krea2EditGroundedEncode" in nodes

    @staticmethod
    def crop_stitch_installed(client: Client) -> bool:
        nodes = client.models.node_inputs
        return "InpaintCropImproved" in nodes and "InpaintStitchImproved" in nodes

    def estimate_target(self) -> Extent:
        model = ensure(self._model())
        return estimate_krea_edit_extent(model.document.extent)

    def apply_preset_turbo(self):
        self.preset = PRESET_TURBO
        self._set_status(_("Turbo preset — ~10 steps, CFG 1"))

    def apply_preset_raw(self):
        self.preset = PRESET_RAW
        self._set_status(_("Raw preset — ~24 steps, CFG 3"))

    def apply_fidelity_locked(self):
        """Max likeness — use when identity drifts; edits will be subtler."""
        self.ref_boost = 4.0
        self.grounding_px = 768
        self._set_status(_("Fidelity: Locked (ref 4 / ground 768)"))

    def apply_fidelity_balanced(self):
        """Default for instruction edits that should clearly change the image."""
        self.ref_boost = 2.0
        self.grounding_px = 512
        self._set_status(_("Fidelity: Balanced (ref 2 / ground 512)"))

    def apply_fidelity_creative(self):
        """Looser lock — stronger prompt adherence, more identity drift."""
        self.ref_boost = 1.0
        self.grounding_px = 384
        self._set_status(_("Fidelity: Creative (ref 1 / ground 384)"))

    def _resolve_style_key(self) -> str:
        if self.preset == PRESET_RAW:
            keys = (
                "built-in/krea2-raw-native.json",
                "krea2-raw-native.json",
                "built-in/krea2-raw.json",
                "krea2-raw.json",
            )
        else:
            keys = ("built-in/krea2-turbo.json", "krea2-turbo.json")
        for key in keys:
            if Styles.list().find(key):
                return key
        # Any Krea 2 style as last resort
        for style in Styles.list():
            if style.architecture is Arch.krea2:
                return style.filename
        return keys[0]

    def _resolve_checkpoint(self, client: Client) -> CheckpointInput:
        style = Styles.list().find(self._resolve_style_key())
        if style is None:
            raise RuntimeError(
                _("No Krea 2 style found. Install a Krea 2 style (Turbo or RAW) first.")
            )
        ckpt = style.get_models(client.models.checkpoints)
        ckpt.loras = []  # Identity Edit + extras applied separately (model-only)
        ckpt.version = Arch.krea2
        if not ckpt.checkpoint or ckpt.checkpoint == "not-found":
            raise RuntimeError(
                _("Krea 2 checkpoint not found on the server for style '{name}'.").format(
                    name=style.name
                )
            )
        return ckpt

    def generate(self):
        eventloop.run(self._report_errors(self._generate()))

    async def _report_errors(self, coro):
        try:
            await coro
        except Exception as e:
            log.exception("Krea Edit failed")
            model = ensure(self._model())
            model.report_error(str(e))
            self.set_in_progress(False)
            self._set_status(_("Error: {msg}").format(msg=str(e)))

    async def _generate(self):
        model = ensure(self._model())
        ok, msg = model.document.check_color_mode()
        if not ok and msg:
            model.report_error(msg)
            return

        client = model._connection.client
        if not self.pack_installed(client):
            model.report_error(
                _("Krea2Edit custom nodes not found.")
                + " "
                + _(
                    "Install comfyui-krea2edit "
                    "(https://github.com/lbouaraba/comfyui-krea2edit) "
                    "and restart ComfyUI."
                )
            )
            return

        self.migrate_legacy_prompt()
        instruction = self.assembled_instruction()
        secondary_id = self.resolve_secondary_ref_id()
        has_roles = bool(self.role_sheet_entries())
        if not instruction:
            if secondary_id or has_roles:
                # Refs alone are enough — guidance text will carry the intent.
                instruction = self.assembled_ref_guidance() or (
                    "Match the subject from the reference image onto the canvas person."
                )
            else:
                model.report_error(
                    _("Write at least one edit — character, scene, lighting, or other.")
                )
                return
        # Keep legacy prompt in sync for older tooling / document history labels
        self.prompt = instruction

        lora = find_identity_edit_lora(client.models.loras)
        if not lora:
            model.report_error(
                _("Identity Edit LoRA not found.")
                + " "
                + _(
                    "Download krea2_identity_edit_v1_2.safetensors and place it in "
                    "ComfyUI/models/loras (e.g. loras/Krea2/), then reconnect."
                )
            )
            return

        extra_loras = self._extra_lora_inputs()
        role = self._role_lora_inputs()
        to_check = list(extra_loras)
        for slot in role.values():
            if slot:
                to_check.append(slot)
        available = {p.replace("\\", "/") for p in client.models.loras}
        for entry in to_check:
            key = entry.name.replace("\\", "/")
            if key not in available and entry.name not in client.models.loras:
                model.report_error(
                    _("LoRA not found on the server: {name}").format(name=entry.name)
                )
                return

        if self.use_active_layer:
            layer = model.layers.active
            bounds = layer.compute_bounds()
            if bounds.is_zero:
                model.report_error(_("The active layer is empty."))
                return
            image = layer.get_pixels(bounds)
            # Pad onto full canvas so output aligns with the document
            canvas = Image.create(model.document.extent)
            canvas.draw_image(image, (bounds.x, bounds.y))
            image = canvas
        else:
            # Ensure flatten/apply is visible to pixelData before capture
            try:
                model.document._doc.refreshProjection()
            except Exception:
                pass
            extent = model.document.extent
            bounds = Bounds(0, 0, *extent)
            image = model.document.get_image(bounds)

        # Selection → local Inpaint Edit (crop 1024 → Identity Edit → stitch)
        inpaint_mask: Image | None = None
        use_inpaint = False
        if self.use_selection_inpaint:
            mask_obj, _sel_bounds = model.document.create_mask_from_selection(
                SelectionModifiers(multiple=8)
            )
            if mask_obj is not None:
                if not self.crop_stitch_installed(client):
                    model.report_error(
                        _("Inpaint Crop & Stitch nodes not found.")
                        + " "
                        + _(
                            "Install ComfyUI-Inpaint-CropAndStitch "
                            "(https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) "
                            "into custom_nodes and restart ComfyUI."
                        )
                    )
                    return
                inpaint_mask = mask_obj.to_image(model.document.extent)
                use_inpaint = True

        # Sliders steer the whole Identity Edit pass; on a selection crop they often
        # rebuild a miniature full frame instead of a clean local inpaint.
        active_sliders = [
            s for s in self._sliders if s.name and abs(float(s.strength)) >= 1e-6
        ]
        slider_forced_full = False
        if use_inpaint and active_sliders:
            log.info(
                "Krea Edit: %d active slider(s) with selection → forcing full-canvas "
                "(sliders unreliable on inpaint crop)",
                len(active_sliders),
            )
            use_inpaint = False
            inpaint_mask = None
            slider_forced_full = True

        try:
            ref_b, ref_kind = self.build_secondary_ref(model)
        except RuntimeError as e:
            model.report_error(str(e))
            return

        # Image refs are the strong identity path — don't also run Face-LoRA replace mode
        # unless the user explicitly set a Face LoRA.
        seed = model.seed if model.fixed_seed else workflow_module.generate_seed()
        settings = PRESET_SETTINGS.get(self.preset, PRESET_SETTINGS[PRESET_TURBO])
        # Subject/sheet as image_b: bias toward likeness on the last ref
        ref_boost = float(self.ref_boost)
        grounding_px = int(self.grounding_px)
        if use_inpaint:
            # Match Inpaint&Edit v2 defaults for local crops (stronger identity lock)
            if ref_boost < 3.5:
                ref_boost = max(ref_boost, 4.0)
            if grounding_px == 0 or grounding_px < 512:
                grounding_px = 768
        elif ref_b is not None and ref_boost < 2.5 and not (self.face_lora or "").strip():
            ref_boost = max(ref_boost, 3.0)

        params = KreaEditInput(
            prompt=instruction,
            negative="",
            identity_lora=lora,
            lora_strength=1.0,
            face_lora=(role["face"].name if role["face"] else ""),
            face_lora_strength=(role["face"].strength if role["face"] else 1.0),
            body_lora=(role["body"].name if role["body"] else ""),
            body_lora_strength=(role["body"].strength if role["body"] else 1.0),
            style_lora=(role["style"].name if role["style"] else ""),
            style_lora_strength=(role["style"].strength if role["style"] else 1.0),
            extra_loras=extra_loras,
            ref_boost=ref_boost,
            ref_boost_a=1.0,
            grounding_px=grounding_px,
            fit_mode="fit",
            sampler=settings["sampler"],
            scheduler=settings["scheduler"],
            steps=int(settings["steps"]),
            cfg_scale=float(settings["cfg_scale"]),
            seed=seed,
            use_inpaint=use_inpaint,
            mask_expand_pixels=int(self.mask_expand_pixels),
            mask_blend_pixels=int(self.mask_blend_pixels),
            context_factor=float(self.context_factor),
        )
        checkpoint = self._resolve_checkpoint(client)
        workflow_input = prepare_krea_edit(params, image, ref_b, checkpoint, mask=inpaint_mask)
        target = workflow_input.extent.target
        edit_size = workflow_input.extent.desired

        mode_tag = "Inpaint" if use_inpaint else "Edit"
        job_params = JobParams(
            Bounds(0, 0, *target),
            f"[Krea {mode_tag}] {instruction[:80]}",
        )
        job_params.metadata["workspace"] = "krea_edit"
        job_params.metadata["prompt"] = instruction
        job_params.metadata["ref_boost"] = ref_boost
        job_params.metadata["grounding_px"] = grounding_px
        job_params.metadata["edit_mode"] = "inpaint" if use_inpaint else "full"
        if use_inpaint:
            job_params.metadata["mask_expand"] = int(self.mask_expand_pixels)
            job_params.metadata["mask_blend"] = int(self.mask_blend_pixels)
            job_params.metadata["context_factor"] = float(self.context_factor)
        if ref_kind:
            job_params.metadata["ref_kind"] = ref_kind
            if ref_kind == "roles":
                job_params.metadata["ref_roles"] = [r for r, _ in self.role_sheet_entries()]
            elif secondary_id:
                job_params.metadata["ref_layer"] = secondary_id
        if self._sliders:
            job_params.metadata["sliders"] = [
                {"name": l.name, "strength": l.strength} for l in self._sliders if l.name
            ]
        if self._loras:
            job_params.metadata["loras"] = [l.name for l in self._loras if l.name]
        if role["face"]:
            job_params.metadata["face_lora"] = role["face"].name
        if role["body"]:
            job_params.metadata["body_lora"] = role["body"].name
        if role["style"]:
            job_params.metadata["style_lora"] = role["style"].name
        job_params.seed = seed

        log.info(
            "Krea Edit [%s]: %dx%d → edit %dx%d, ref_boost=%.2f grounding=%d steps=%d cfg=%.1f "
            "seed=%s ref=%s roles=%s sliders=%d face=%s body=%s style=%s | %s",
            "inpaint" if use_inpaint else "full",
            image.extent.width,
            image.extent.height,
            edit_size.width,
            edit_size.height,
            ref_boost,
            grounding_px,
            int(settings["steps"]),
            float(settings["cfg_scale"]),
            seed,
            (ref_kind or "-"),
            [r for r, _ in self.role_sheet_entries()] or "-",
            sum(1 for l in self._sliders if l.name),
            (role["face"].name if role["face"] else "-"),
            (role["body"].name if role["body"] else "-"),
            (role["style"].name if role["style"] else "-"),
            instruction[:200],
        )
        if use_inpaint:
            self._set_status(
                _("Krea Inpaint Edit… crop {sz} (ref {rb:g}, ground {g})").format(
                    sz=int(params.inpaint_target_size),
                    rb=ref_boost,
                    g=grounding_px,
                )
            )
        elif slider_forced_full:
            self._set_status(
                _(
                    "Sliders + selection → full-canvas Edit "
                    "(sliders need the whole image; Deselect for Inpaint). "
                    "{w}×{h}, ref {rb:g}"
                ).format(w=edit_size.width, h=edit_size.height, rb=ref_boost)
            )
        else:
            self._set_status(
                _("Krea Edit… {w}×{h} (ref {rb:g}, ground {g})").format(
                    w=edit_size.width,
                    h=edit_size.height,
                    rb=ref_boost,
                    g=grounding_px,
                )
            )
        model.clear_error()
        self.set_in_progress(True)
        self.target_extent_changed.emit(
            Extent(params.inpaint_target_size, params.inpaint_target_size)
            if use_inpaint
            else estimate_krea_edit_extent(image.extent)
        )
        job = model.jobs.add(JobKind.diffusion, job_params)
        await model._enqueue_job(job, workflow_input)
