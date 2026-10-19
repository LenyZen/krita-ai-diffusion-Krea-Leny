"""Krea Edit workspace UI — identity-preserving instruction edit."""

from __future__ import annotations

from PyQt5.QtCore import QMetaObject, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..files import FileFilter
from ..localization import translate as _
from ..model.jobs import JobKind
from ..model.krea_edit import (
    FIELD_PRESETS,
    MAX_EDIT_LORAS,
    MAX_EDIT_SLIDERS,
    PLACEHOLDER_CHARACTER,
    PLACEHOLDER_LIGHTING,
    PLACEHOLDER_OTHER,
    PLACEHOLDER_REF_NOTES,
    PLACEHOLDER_SCENE,
    PRESET_RAW,
    PRESET_TURBO,
    SLIDERS_PATH_FRAGMENT,
    KreaEditLora,
    KreaEditWorkspace,
)
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind, bind_toggle
from ..model.root import root
from . import theme
from .generation import ProgressBar
from .widget import ErrorBox, GenerateButton, WorkspaceSelectWidget


class KreaEditLoraRow(QWidget):
    """One editable LoRA picker row for Krea Edit."""

    def __init__(self, index: int, lora: KreaEditLora, parent=None):
        super().__init__(parent)
        self.index = index
        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        completer = QCompleter(self._loras)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        self._combo = QComboBox(self)
        self._combo.setEditable(True)
        self._combo.setModel(self._loras)
        self._combo.setCompleter(completer)
        self._combo.setMaxVisibleItems(16)
        self._combo.setMinimumWidth(120)
        self._combo.setToolTip(_("Style / character LoRA stacked after Identity Edit"))
        layout.addWidget(self._combo, 1)

        self._strength = QDoubleSpinBox(self)
        self._strength.setRange(0.05, 3.0)
        self._strength.setSingleStep(0.05)
        self._strength.setDecimals(2)
        self._strength.setValue(lora.strength)
        self._strength.setPrefix(_("Strength") + ": ")
        layout.addWidget(self._strength)

        self._clear = QToolButton(self)
        self._clear.setIcon(theme.icon("discard"))
        self._clear.setToolTip(_("Remove this LoRA"))
        layout.addWidget(self._clear)

        if line_edit := self._combo.lineEdit():
            line_edit.setPlaceholderText(_("(none)"))
        self._set_lora(lora.name, lora.strength)

    def _set_lora(self, name: str, strength: float):
        self._updating = True
        self._strength.setValue(strength)
        if name:
            idx = self._combo.findData(name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            else:
                self._combo.setEditText(name)
        else:
            self._combo.setCurrentIndex(-1)
            self._combo.setEditText("")
        self._updating = False

    def lora_name(self) -> str:
        data = self._combo.currentData()
        if data:
            return str(data)
        text = self._combo.currentText().strip()
        if not text:
            return ""
        if file := root.files.loras.find(text):
            return file.id
        return text

    def lora_strength(self) -> float:
        return self._strength.value()


class KreaEditRoleLoraRow(QWidget):
    """Fixed Face / Body / Style LoRA slot with label, picker, strength, clear."""

    changed = None  # set in __init__ via pyqt isn't needed; parent connects widgets

    def __init__(self, label: str, tip: str, parent=None):
        super().__init__(parent)
        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        name_label = QLabel(label, self)
        name_label.setMinimumWidth(44)
        name_label.setToolTip(tip)
        layout.addWidget(name_label)

        completer = QCompleter(self._loras)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        self._combo = QComboBox(self)
        self._combo.setEditable(True)
        self._combo.setModel(self._loras)
        self._combo.setCompleter(completer)
        self._combo.setMaxVisibleItems(16)
        self._combo.setMinimumWidth(120)
        self._combo.setToolTip(tip)
        layout.addWidget(self._combo, 1)

        self._strength = QDoubleSpinBox(self)
        self._strength.setRange(0.05, 3.0)
        self._strength.setSingleStep(0.05)
        self._strength.setDecimals(2)
        self._strength.setValue(1.0)
        self._strength.setPrefix(_("Strength") + ": ")
        layout.addWidget(self._strength)

        self._clear = QToolButton(self)
        self._clear.setIcon(theme.icon("discard"))
        self._clear.setToolTip(_("Clear this LoRA"))
        self._clear.clicked.connect(self.clear)
        layout.addWidget(self._clear)

        if line_edit := self._combo.lineEdit():
            line_edit.setPlaceholderText(_("(none)"))

    def set_value(self, name: str, strength: float):
        self._updating = True
        self._strength.setValue(strength if strength else 1.0)
        if name:
            idx = self._combo.findData(name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            else:
                self._combo.setEditText(name)
        else:
            self._combo.setCurrentIndex(-1)
            self._combo.setEditText("")
        self._updating = False

    def clear(self):
        self.set_value("", 1.0)

    def lora_name(self) -> str:
        data = self._combo.currentData()
        if data:
            return str(data)
        text = self._combo.currentText().strip()
        if not text:
            return ""
        if file := root.files.loras.find(text):
            return file.id
        return text

    def lora_strength(self) -> float:
        return self._strength.value()

    @property
    def updating(self) -> bool:
        return self._updating


# QSlider uses tenths so −15.0 … +15.0 maps to −150 … +150 (center = 0).
_SLIDER_STRENGTH_MIN = -15.0
_SLIDER_STRENGTH_MAX = 15.0
_SLIDER_STRENGTH_SCALE = 10  # 1 unit on the bar = 0.1 strength


class KreaEditSliderRow(QWidget):
    """One LoRA from loras/krea/sliders with a −15…0…+15 bar (center = off)."""

    def __init__(self, index: int, lora: KreaEditLora, parent=None):
        super().__init__(parent)
        self.index = index
        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True
        self._loras.path_contains = SLIDERS_PATH_FRAGMENT
        self._updating = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 2, 0, 4)
        root_layout.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)

        completer = QCompleter(self._loras)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        self._combo = QComboBox(self)
        self._combo.setEditable(True)
        self._combo.setModel(self._loras)
        self._combo.setCompleter(completer)
        self._combo.setMaxVisibleItems(16)
        self._combo.setMinimumWidth(120)
        self._combo.setToolTip(_("Slider LoRA from loras/krea/sliders"))
        top.addWidget(self._combo, 1)

        self._clear = QToolButton(self)
        self._clear.setIcon(theme.icon("discard"))
        self._clear.setToolTip(_("Remove this slider"))
        top.addWidget(self._clear)
        root_layout.addLayout(top)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)

        self._value_label = QLabel("0.0", self)
        self._value_label.setMinimumWidth(40)
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._value_label.setStyleSheet(f"color: {theme.grey};")
        bar.addWidget(self._value_label)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(int(_SLIDER_STRENGTH_MIN * _SLIDER_STRENGTH_SCALE))
        self._slider.setMaximum(int(_SLIDER_STRENGTH_MAX * _SLIDER_STRENGTH_SCALE))
        self._slider.setValue(0)
        self._slider.setSingleStep(1)  # 0.1
        self._slider.setPageStep(10)  # 1.0
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(50)  # marks at −15, −10, −5, 0, 5, 10, 15
        self._slider.setToolTip(
            _("Center = 0 (off). Left = negative (−15), right = positive (+15).")
        )
        # Keep `_strength` alias so parent can connect valueChanged the same way.
        self._strength = self._slider
        self._slider.valueChanged.connect(self._on_slider_moved)
        bar.addWidget(self._slider, 1)
        root_layout.addLayout(bar)

        if line_edit := self._combo.lineEdit():
            line_edit.setPlaceholderText(_("Pick a slider…"))
        self._set_lora(lora.name, lora.strength)

    def _format_value(self, strength: float) -> str:
        if abs(strength) < 0.05:
            return "0.0"
        return f"{strength:+.1f}"

    def _on_slider_moved(self, raw: int):
        if self._updating:
            return
        self._value_label.setText(self._format_value(raw / _SLIDER_STRENGTH_SCALE))

    def _set_lora(self, name: str, strength: float):
        self._updating = True
        clamped = max(_SLIDER_STRENGTH_MIN, min(_SLIDER_STRENGTH_MAX, float(strength)))
        raw = int(round(clamped * _SLIDER_STRENGTH_SCALE))
        self._slider.setValue(raw)
        self._value_label.setText(self._format_value(raw / _SLIDER_STRENGTH_SCALE))
        if name:
            idx = self._combo.findData(name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            else:
                self._combo.setEditText(name)
        else:
            self._combo.setCurrentIndex(-1)
            self._combo.setEditText("")
        self._updating = False

    def lora_name(self) -> str:
        data = self._combo.currentData()
        if data:
            return str(data)
        text = self._combo.currentText().strip()
        if not text:
            return ""
        if file := root.files.loras.find(text):
            return file.id
        return text

    def lora_strength(self) -> float:
        return self._slider.value() / _SLIDER_STRENGTH_SCALE


def _make_section(
    parent: QWidget,
    title: str,
    tip: str,
    placeholder: str,
    min_height: int = 64,
    field_presets: list[tuple[str, str]] | None = None,
) -> tuple[QGroupBox, QPlainTextEdit, QComboBox | None]:
    box = QGroupBox(title, parent)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(6, 8, 6, 6)
    layout.setSpacing(4)
    help_label = QLabel(tip, box)
    help_label.setWordWrap(True)
    help_label.setStyleSheet(f"color: {theme.grey};")
    layout.addWidget(help_label)

    preset_combo: QComboBox | None = None
    if field_presets:
        preset_combo = QComboBox(box)
        preset_combo.setToolTip(_("Pick a starting prompt, then edit it below."))
        preset_combo.addItem(_("Pick a starting point…"), "")
        for label, text in field_presets:
            preset_combo.addItem(_(label), text)
        layout.addWidget(preset_combo)

    edit = QPlainTextEdit(box)
    edit.setPlaceholderText(placeholder)
    edit.setMinimumHeight(min_height)
    layout.addWidget(edit)
    return box, edit, preset_combo


class KreaEditWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._lora_rows: list[KreaEditLoraRow] = []
        self._slider_rows: list[KreaEditSliderRow] = []
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 4, 0)
        outer.setSpacing(4)

        self.workspace_select = WorkspaceSelectWidget(self)
        header = QHBoxLayout()
        header.addWidget(self.workspace_select)
        header.addStretch(1)
        outer.addLayout(header)

        tip = QLabel(
            _(
                "Write the edit on Edit. Refs hold identity (Subject / role sheets). "
                "Krita selection = local Inpaint Edit (crop → stitch); no selection = full canvas."
            ),
            self,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.grey};")
        outer.addWidget(tip)

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)

        # --- Edit tab ---
        edit_scroll = QScrollArea(self.tabs)
        edit_scroll.setWidgetResizable(True)
        edit_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        edit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        edit_body = QWidget(edit_scroll)
        layout = QVBoxLayout(edit_body)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        presets = QHBoxLayout()
        self.preset_turbo = QPushButton(_("Turbo"), edit_body)
        self.preset_raw = QPushButton(_("Raw"), edit_body)
        self.preset_turbo.setToolTip(_("Fast: 10 steps, CFG 1 — matches the official edit graph"))
        self.preset_raw.setToolTip(_("Higher quality: 24 steps, CFG 3 — uses RAW checkpoint"))
        presets.addWidget(self.preset_turbo)
        presets.addWidget(self.preset_raw)
        layout.addLayout(presets)

        fidelity = QHBoxLayout()
        self.fidelity_locked = QPushButton(_("Locked"), edit_body)
        self.fidelity_balanced = QPushButton(_("Balanced"), edit_body)
        self.fidelity_creative = QPushButton(_("Creative"), edit_body)
        self.fidelity_locked.setToolTip(_("Ref 4 / Ground 768 — max likeness, subtle edits"))
        self.fidelity_balanced.setToolTip(_("Ref 2 / Ground 512 — recommended for clear edits"))
        self.fidelity_creative.setToolTip(_("Ref 1 / Ground 384 — strongest prompt adherence"))
        fidelity.addWidget(self.fidelity_locked)
        fidelity.addWidget(self.fidelity_balanced)
        fidelity.addWidget(self.fidelity_creative)
        layout.addLayout(fidelity)

        character_box, self.character_edit, self.character_preset = _make_section(
            edit_body,
            _("Character — body & clothing"),
            _(
                "Outfit, pose, expression, body. Always say what to keep (face / identity). "
                "Use the dropdown for a starter, then edit the text."
            ),
            _(PLACEHOLDER_CHARACTER),
            min_height=72,
            field_presets=FIELD_PRESETS["character"],
        )
        layout.addWidget(character_box)

        scene_box, self.scene_edit, self.scene_preset = _make_section(
            edit_body,
            _("Scene — background & setting"),
            _(
                "Location, props, weather. Say whether the subject stays put in frame. "
                "Dropdown = starter prompt you can rewrite."
            ),
            _(PLACEHOLDER_SCENE),
            min_height=64,
            field_presets=FIELD_PRESETS["scene"],
        )
        layout.addWidget(scene_box)

        lighting_box, self.lighting_edit, self.lighting_preset = _make_section(
            edit_body,
            _("Lighting"),
            _(
                "Light quality, direction, color, and shadow mood. "
                "Pick a starter from the dropdown, then tune the wording."
            ),
            _(PLACEHOLDER_LIGHTING),
            min_height=56,
            field_presets=FIELD_PRESETS["lighting"],
        )
        layout.addWidget(lighting_box)

        other_box, self.other_edit, self.other_preset = _make_section(
            edit_body,
            _("Other"),
            _(
                "Anything that doesn’t fit above — camera, mood, small extras. "
                "Dropdown starters show the kind of detail that works here."
            ),
            _(PLACEHOLDER_OTHER),
            min_height=48,
            field_presets=FIELD_PRESETS["other"],
        )
        layout.addWidget(other_box)

        src_box = QGroupBox(_("Source"), edit_body)
        src_form = QFormLayout(src_box)
        self.layer_check = QCheckBox(_("Use active layer only"), src_box)
        self.layer_check.setToolTip(_("Off = full canvas composite (recommended)"))
        src_form.addRow(self.layer_check)

        refs_hint = QLabel(_("Subject / role sheets → Refs tab"), src_box)
        refs_hint.setStyleSheet(f"color: {theme.grey};")
        src_form.addRow(_("Identity"), refs_hint)

        self.target_label = QLabel("", src_box)
        self.target_label.setStyleSheet(f"color: {theme.grey};")
        src_form.addRow(_("Edit size"), self.target_label)
        layout.addWidget(src_box)

        opts_box = QGroupBox(_("Options"), edit_body)
        opts_form = QFormLayout(opts_box)
        self.ref_boost_spin = QDoubleSpinBox(opts_box)
        self.ref_boost_spin.setRange(0.0, 20.0)
        self.ref_boost_spin.setSingleStep(0.5)
        self.ref_boost_spin.setDecimals(2)
        self.ref_boost_spin.setToolTip(
            _(
                "Reference fidelity. 4 = near-copy of source; 2 = balanced edits; "
                "1 = follow the prompt more loosely."
            )
        )
        opts_form.addRow(_("Ref boost"), self.ref_boost_spin)

        self.grounding_spin = QSpinBox(opts_box)
        self.grounding_spin.setRange(0, 2048)
        self.grounding_spin.setSingleStep(64)
        self.grounding_spin.setSpecialValueText(_("Native"))
        self.grounding_spin.setToolTip(
            _("Vision size for reading the image+prompt. Lower = stronger edits (try 384–512).")
        )
        opts_form.addRow(_("Grounding px"), self.grounding_spin)
        layout.addWidget(opts_box)

        self.status_label = QLabel("", edit_body)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        edit_scroll.setWidget(edit_body)
        self.tabs.addTab(edit_scroll, _("Edit"))

        # --- Refs tab (primary identity path) ---
        ref_scroll = QScrollArea(self.tabs)
        ref_scroll.setWidgetResizable(True)
        ref_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        ref_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ref_body = QWidget(ref_scroll)
        ref_layout = QVBoxLayout(ref_body)
        ref_layout.setContentsMargins(0, 4, 0, 0)
        ref_layout.setSpacing(6)

        ref_tip = QLabel(
            _(
                "Identity Edit accepts one secondary image. Pick a Subject, and/or separate "
                "Face / Hair / Clothing / Pose sheet layers — we collage role sheets into "
                "one ref automatically. A single full Ref sheet still works if you prefer "
                "to arrange panels yourself (used only when no role sheets are set)."
            ),
            ref_body,
        )
        ref_tip.setWordWrap(True)
        ref_tip.setStyleSheet(f"color: {theme.grey};")
        ref_layout.addWidget(ref_tip)

        layers_box = QGroupBox(_("Subject"), ref_body)
        layers_form = QFormLayout(layers_box)
        self.subject_combo = QComboBox(layers_box)
        self.subject_combo.setToolTip(
            _("Clear face/body photo of the character to place onto the canvas.")
        )
        layers_form.addRow(_("Subject"), self.subject_combo)
        ref_layout.addWidget(layers_box)

        roles_box = QGroupBox(_("Role sheets"), ref_body)
        roles_form = QFormLayout(roles_box)
        roles_tip = QLabel(
            _(
                "One layer per cue. Face + Hair + Clothing + Pose are stitched into a "
                "grid and sent as the secondary ref."
            ),
            roles_box,
        )
        roles_tip.setWordWrap(True)
        roles_tip.setStyleSheet(f"color: {theme.grey};")
        roles_form.addRow(roles_tip)
        self.sheet_face_combo = QComboBox(roles_box)
        self.sheet_hair_combo = QComboBox(roles_box)
        self.sheet_clothing_combo = QComboBox(roles_box)
        self.sheet_pose_combo = QComboBox(roles_box)
        self.sheet_face_combo.setToolTip(_("Layer with the face / identity close-up"))
        self.sheet_hair_combo.setToolTip(_("Layer showing the hair style to take"))
        self.sheet_clothing_combo.setToolTip(_("Layer showing outfit / clothing"))
        self.sheet_pose_combo.setToolTip(_("Layer showing pose / body language"))
        roles_form.addRow(_("Face"), self.sheet_face_combo)
        roles_form.addRow(_("Hair"), self.sheet_hair_combo)
        roles_form.addRow(_("Clothing"), self.sheet_clothing_combo)
        roles_form.addRow(_("Pose"), self.sheet_pose_combo)
        ref_layout.addWidget(roles_box)

        full_box = QGroupBox(_("Full ref sheet (optional)"), ref_body)
        full_form = QFormLayout(full_box)
        self.sheet_combo = QComboBox(full_box)
        self.sheet_combo.setToolTip(
            _(
                "Optional single collage layer. Used only when no Face/Hair/Clothing/Pose "
                "role sheets are set. Arrange panels yourself on one layer."
            )
        )
        full_form.addRow(_("Ref sheet"), self.sheet_combo)
        ref_layout.addWidget(full_box)

        take_box = QGroupBox(_("Take from Subject / full sheet"), ref_body)
        take_layout = QVBoxLayout(take_box)
        take_tip = QLabel(
            _(
                "When using Subject or a full Ref sheet (not role sheets), tell the model "
                "which parts to use."
            ),
            take_box,
        )
        take_tip.setWordWrap(True)
        take_tip.setStyleSheet(f"color: {theme.grey};")
        take_layout.addWidget(take_tip)
        take_row = QHBoxLayout()
        self.take_face = QCheckBox(_("Face"), take_box)
        self.take_hair = QCheckBox(_("Hair"), take_box)
        self.take_clothing = QCheckBox(_("Clothing"), take_box)
        self.take_pose = QCheckBox(_("Pose"), take_box)
        for cb in (self.take_face, self.take_hair, self.take_clothing, self.take_pose):
            take_row.addWidget(cb)
        take_row.addStretch(1)
        take_layout.addLayout(take_row)
        ref_layout.addWidget(take_box)

        notes_box, self.ref_notes_edit, _notes_preset = _make_section(
            ref_body,
            _("Ref notes"),
            _("Optional details — which panel, what to ignore, etc."),
            _(PLACEHOLDER_REF_NOTES),
            min_height=56,
        )
        ref_layout.addWidget(notes_box)
        ref_layout.addStretch(1)
        ref_scroll.setWidget(ref_body)
        self.tabs.addTab(ref_scroll, _("Refs"))

        # --- Advanced: sliders + role LoRAs + extras ---
        adv_scroll = QScrollArea(self.tabs)
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        adv_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        adv_body = QWidget(adv_scroll)
        adv_layout = QVBoxLayout(adv_body)
        adv_layout.setContentsMargins(0, 4, 0, 0)
        adv_layout.setSpacing(6)

        adv_tip = QLabel(
            _(
                "Optional. Prefer Refs for identity. Selection Inpaint = local crop only. "
                "Sliders work best on the full canvas (see Sliders tip). "
                "Identity Edit LoRA is applied automatically — do not add it again."
            ),
            adv_body,
        )
        adv_tip.setWordWrap(True)
        adv_tip.setStyleSheet(f"color: {theme.grey};")
        adv_layout.addWidget(adv_tip)

        inpaint_box = QGroupBox(_("Inpaint crop"), adv_body)
        inpaint_form = QFormLayout(inpaint_box)
        inpaint_tip = QLabel(
            _(
                "When a Krita selection exists, crop to 1024×1024 with context, run "
                "Identity Edit on the crop, then soft-stitch back. Unmasked pixels stay "
                "untouched. Needs ComfyUI-Inpaint-CropAndStitch. "
                "Tip: clear Sliders (or Deselect) for reliable local inpaint."
            ),
            inpaint_box,
        )
        inpaint_tip.setWordWrap(True)
        inpaint_tip.setStyleSheet(f"color: {theme.grey};")
        inpaint_form.addRow(inpaint_tip)
        self.selection_inpaint_check = QCheckBox(
            _("Use selection for local Inpaint Edit"), inpaint_box
        )
        self.selection_inpaint_check.setToolTip(
            _("Off = always full-canvas edit, even with a selection.")
        )
        inpaint_form.addRow(self.selection_inpaint_check)
        self.mask_expand_spin = QSpinBox(inpaint_box)
        self.mask_expand_spin.setRange(0, 256)
        self.mask_expand_spin.setToolTip(_("Expand mask before crop (workflow default 8)"))
        inpaint_form.addRow(_("Mask expand"), self.mask_expand_spin)
        self.mask_blend_spin = QSpinBox(inpaint_box)
        self.mask_blend_spin.setRange(0, 64)
        self.mask_blend_spin.setToolTip(_("Soft stitch seam in pixels (workflow default 32)"))
        inpaint_form.addRow(_("Mask blend"), self.mask_blend_spin)
        self.context_factor_spin = QDoubleSpinBox(inpaint_box)
        self.context_factor_spin.setRange(1.0, 4.0)
        self.context_factor_spin.setSingleStep(0.1)
        self.context_factor_spin.setDecimals(2)
        self.context_factor_spin.setToolTip(
            _("Grow crop around the mask for scene/identity context (default 1.5)")
        )
        inpaint_form.addRow(_("Context factor"), self.context_factor_spin)
        adv_layout.addWidget(inpaint_box)

        sliders_box = QGroupBox(_("Sliders"), adv_body)
        sliders_layout = QVBoxLayout(sliders_box)
        sliders_tip = QLabel(
            _(
                "Put slider LoRAs in ComfyUI/models/loras/krea/sliders/, then reconnect. "
                "Bars: center = 0 (off), right +15, left −15. "
                "IMPORTANT: Sliders work best on the FULL image. With a marquee/selection "
                "they often rebuild a smaller whole-frame edit instead of a clean local "
                "inpaint — Deselect (or zero the bars) when you need selection Inpaint."
            ),
            sliders_box,
        )
        sliders_tip.setWordWrap(True)
        sliders_tip.setStyleSheet(f"color: {theme.grey};")
        sliders_layout.addWidget(sliders_tip)
        self.slider_list = QVBoxLayout()
        self.slider_list.setContentsMargins(0, 0, 0, 0)
        self.slider_list.setSpacing(4)
        sliders_layout.addLayout(self.slider_list)
        self.add_slider_button = QPushButton(_("Add Slider"), sliders_box)
        self.add_slider_button.clicked.connect(self._add_slider)
        sliders_layout.addWidget(self.add_slider_button)
        adv_layout.addWidget(sliders_box)

        role_box = QGroupBox(_("Role LoRAs"), adv_body)
        role_layout = QVBoxLayout(role_box)
        self.face_lora_row = KreaEditRoleLoraRow(
            _("Face"),
            _("Usually weaker than a Subject / sheet ref for face swaps."),
            role_box,
        )
        self.body_lora_row = KreaEditRoleLoraRow(
            _("Body"),
            _("Body / physique LoRA — shape and anatomy"),
            role_box,
        )
        self.style_lora_row = KreaEditRoleLoraRow(
            _("Style"),
            _("Rendering / aesthetic LoRA"),
            role_box,
        )
        role_layout.addWidget(self.face_lora_row)
        role_layout.addWidget(self.body_lora_row)
        role_layout.addWidget(self.style_lora_row)
        adv_layout.addWidget(role_box)

        extra_box = QGroupBox(_("Additional LoRAs"), adv_body)
        extra_layout = QVBoxLayout(extra_box)
        self.lora_list = QVBoxLayout()
        self.lora_list.setContentsMargins(0, 0, 0, 0)
        self.lora_list.setSpacing(4)
        extra_layout.addLayout(self.lora_list)
        self.add_lora_button = QPushButton(_("Add LoRA"), extra_box)
        self.add_lora_button.clicked.connect(self._add_lora)
        extra_layout.addWidget(self.add_lora_button)
        adv_layout.addWidget(extra_box)
        adv_layout.addStretch(1)
        adv_scroll.setWidget(adv_body)
        self.tabs.addTab(adv_scroll, _("Advanced"))

        outer.addWidget(self.tabs, 1)

        self.progress_bar = ProgressBar(self)
        outer.addWidget(self.progress_bar)
        self.error_box = ErrorBox(self)
        outer.addWidget(self.error_box)

        self.generate_button = GenerateButton(JobKind.diffusion, self)
        self.generate_button.operation = _("Edit")
        self.generate_button.clicked.connect(self._generate)
        outer.addWidget(self.generate_button)

        self.preset_turbo.clicked.connect(lambda: self._krea().apply_preset_turbo())
        self.preset_raw.clicked.connect(lambda: self._krea().apply_preset_raw())
        self.fidelity_locked.clicked.connect(lambda: self._krea().apply_fidelity_locked())
        self.fidelity_balanced.clicked.connect(lambda: self._krea().apply_fidelity_balanced())
        self.fidelity_creative.clicked.connect(lambda: self._krea().apply_fidelity_creative())
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        self.sheet_combo.currentIndexChanged.connect(self._on_sheet_changed)
        self.sheet_face_combo.currentIndexChanged.connect(
            lambda: self._on_role_sheet_changed("sheet_face_layer_id", self.sheet_face_combo)
        )
        self.sheet_hair_combo.currentIndexChanged.connect(
            lambda: self._on_role_sheet_changed("sheet_hair_layer_id", self.sheet_hair_combo)
        )
        self.sheet_clothing_combo.currentIndexChanged.connect(
            lambda: self._on_role_sheet_changed(
                "sheet_clothing_layer_id", self.sheet_clothing_combo
            )
        )
        self.sheet_pose_combo.currentIndexChanged.connect(
            lambda: self._on_role_sheet_changed("sheet_pose_layer_id", self.sheet_pose_combo)
        )
        self.take_face.toggled.connect(lambda v: setattr(self._krea(), "ref_take_face", v))
        self.take_hair.toggled.connect(lambda v: setattr(self._krea(), "ref_take_hair", v))
        self.take_clothing.toggled.connect(lambda v: setattr(self._krea(), "ref_take_clothing", v))
        self.take_pose.toggled.connect(lambda v: setattr(self._krea(), "ref_take_pose", v))
        self.ref_notes_edit.textChanged.connect(
            lambda: self._set_field("ref_notes", self.ref_notes_edit)
        )
        self.character_edit.textChanged.connect(
            lambda: self._set_field("character", self.character_edit)
        )
        self.scene_edit.textChanged.connect(lambda: self._set_field("scene", self.scene_edit))
        self.lighting_edit.textChanged.connect(
            lambda: self._set_field("lighting", self.lighting_edit)
        )
        self.other_edit.textChanged.connect(lambda: self._set_field("other", self.other_edit))
        self.character_preset.currentIndexChanged.connect(
            lambda: self._apply_field_preset(self.character_preset, "character", self.character_edit)
        )
        self.scene_preset.currentIndexChanged.connect(
            lambda: self._apply_field_preset(self.scene_preset, "scene", self.scene_edit)
        )
        self.lighting_preset.currentIndexChanged.connect(
            lambda: self._apply_field_preset(self.lighting_preset, "lighting", self.lighting_edit)
        )
        self.other_preset.currentIndexChanged.connect(
            lambda: self._apply_field_preset(self.other_preset, "other", self.other_edit)
        )
        for row in (self.face_lora_row, self.body_lora_row, self.style_lora_row):
            row._combo.currentIndexChanged.connect(self._sync_role_loras_to_model)
            row._combo.editTextChanged.connect(lambda _t: self._sync_role_loras_to_model())
            row._strength.valueChanged.connect(lambda _v: self._sync_role_loras_to_model())
            row._clear.clicked.connect(self._sync_role_loras_to_model)

        self._update_target_label()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            s = model.krea_edit
            s.migrate_legacy_prompt()
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                bind(s, "ref_boost", self.ref_boost_spin, "value"),
                bind(s, "grounding_px", self.grounding_spin, "value"),
                bind(s, "can_generate", self.generate_button, "enabled", Bind.one_way),
                bind_toggle(s, "use_active_layer", self.layer_check),
                bind_toggle(s, "use_selection_inpaint", self.selection_inpaint_check),
                bind(s, "mask_expand_pixels", self.mask_expand_spin, "value"),
                bind(s, "mask_blend_pixels", self.mask_blend_spin, "value"),
                bind(s, "context_factor", self.context_factor_spin, "value"),
                bind_toggle(s, "ref_take_face", self.take_face),
                bind_toggle(s, "ref_take_hair", self.take_hair),
                bind_toggle(s, "ref_take_clothing", self.take_clothing),
                bind_toggle(s, "ref_take_pose", self.take_pose),
                s.status_changed.connect(self.status_label.setText),
                s.target_extent_changed.connect(lambda _: self._update_target_label()),
                s.character_changed.connect(
                    lambda t: self._sync_field(self.character_edit, t)
                ),
                s.scene_changed.connect(lambda t: self._sync_field(self.scene_edit, t)),
                s.lighting_changed.connect(lambda t: self._sync_field(self.lighting_edit, t)),
                s.other_changed.connect(lambda t: self._sync_field(self.other_edit, t)),
                s.ref_notes_changed.connect(lambda t: self._sync_field(self.ref_notes_edit, t)),
                s.subject_layer_id_changed.connect(self._sync_subject_combo),
                s.sheet_layer_id_changed.connect(self._sync_sheet_combo),
                s.sheet_face_layer_id_changed.connect(
                    lambda v: self._sync_role_combo(self.sheet_face_combo, v)
                ),
                s.sheet_hair_layer_id_changed.connect(
                    lambda v: self._sync_role_combo(self.sheet_hair_combo, v)
                ),
                s.sheet_clothing_layer_id_changed.connect(
                    lambda v: self._sync_role_combo(self.sheet_clothing_combo, v)
                ),
                s.sheet_pose_layer_id_changed.connect(
                    lambda v: self._sync_role_combo(self.sheet_pose_combo, v)
                ),
                s.preset_changed.connect(self._sync_preset_buttons),
                s.loras_changed.connect(self._rebuild_loras),
                s.sliders_changed.connect(self._rebuild_sliders),
                s.face_lora_changed.connect(lambda _: self._sync_role_loras_from_model()),
                s.face_lora_strength_changed.connect(lambda _: self._sync_role_loras_from_model()),
                s.body_lora_changed.connect(lambda _: self._sync_role_loras_from_model()),
                s.body_lora_strength_changed.connect(lambda _: self._sync_role_loras_from_model()),
                s.style_lora_changed.connect(lambda _: self._sync_role_loras_from_model()),
                s.style_lora_strength_changed.connect(lambda _: self._sync_role_loras_from_model()),
                model.layers.changed.connect(self._rebuild_ref_combos),
            ]
            self.generate_button.model = model
            self.progress_bar.model = model
            self._sync_field(self.character_edit, s.character)
            self._sync_field(self.scene_edit, s.scene)
            self._sync_field(self.lighting_edit, s.lighting)
            self._sync_field(self.other_edit, s.other)
            self._sync_field(self.ref_notes_edit, s.ref_notes)
            self._rebuild_ref_combos()
            self._rebuild_loras()
            self._rebuild_sliders()
            self._sync_role_loras_from_model()
            self._sync_preset_buttons(s.preset)
            self.status_label.setText(s.status)
            self._update_target_label()

    def _krea(self) -> KreaEditWorkspace:
        return self._model.krea_edit

    def _sync_field(self, edit: QPlainTextEdit, text: str):
        if edit.toPlainText() != text:
            edit.blockSignals(True)
            edit.setPlainText(text)
            edit.blockSignals(False)

    def _set_field(self, name: str, edit: QPlainTextEdit):
        setattr(self._krea(), name, edit.toPlainText())

    def _apply_field_preset(self, combo: QComboBox, field: str, edit: QPlainTextEdit):
        """Fill the text box from a starter dropdown; user can edit afterward."""
        text = combo.currentData()
        if not text:
            return
        edit.blockSignals(True)
        edit.setPlainText(str(text))
        edit.blockSignals(False)
        setattr(self._krea(), field, str(text))
        # Reset to placeholder so the same preset can be chosen again later.
        combo.blockSignals(True)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _fill_layer_combo(self, combo: QComboBox, current: str):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(_("None"), "")
        for layer in self._model.layers.images:
            combo.addItem(layer.name, layer.id_string)
        idx = combo.findData(current or "")
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    def _rebuild_ref_combos(self):
        s = self._krea()
        self._fill_layer_combo(self.subject_combo, s.subject_layer_id or s.ref_layer_id)
        self._fill_layer_combo(self.sheet_combo, s.sheet_layer_id)
        self._fill_layer_combo(self.sheet_face_combo, s.sheet_face_layer_id)
        self._fill_layer_combo(self.sheet_hair_combo, s.sheet_hair_layer_id)
        self._fill_layer_combo(self.sheet_clothing_combo, s.sheet_clothing_layer_id)
        self._fill_layer_combo(self.sheet_pose_combo, s.sheet_pose_layer_id)
        self._update_refs_tab_label()

    def _sync_subject_combo(self, layer_id: str):
        idx = self.subject_combo.findData(layer_id or "")
        if idx >= 0 and self.subject_combo.currentIndex() != idx:
            self.subject_combo.blockSignals(True)
            self.subject_combo.setCurrentIndex(idx)
            self.subject_combo.blockSignals(False)
        self._update_refs_tab_label()

    def _sync_sheet_combo(self, layer_id: str):
        idx = self.sheet_combo.findData(layer_id or "")
        if idx >= 0 and self.sheet_combo.currentIndex() != idx:
            self.sheet_combo.blockSignals(True)
            self.sheet_combo.setCurrentIndex(idx)
            self.sheet_combo.blockSignals(False)
        self._update_refs_tab_label()

    def _sync_role_combo(self, combo: QComboBox, layer_id: str):
        idx = combo.findData(layer_id or "")
        if idx >= 0 and combo.currentIndex() != idx:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._update_refs_tab_label()

    def _on_subject_changed(self):
        self._krea().subject_layer_id = self.subject_combo.currentData() or ""
        self._krea().resolve_secondary_ref_id()
        self._update_refs_tab_label()

    def _on_sheet_changed(self):
        self._krea().sheet_layer_id = self.sheet_combo.currentData() or ""
        self._krea().resolve_secondary_ref_id()
        self._update_refs_tab_label()

    def _on_role_sheet_changed(self, attr: str, combo: QComboBox):
        setattr(self._krea(), attr, combo.currentData() or "")
        self._krea().resolve_secondary_ref_id()
        self._update_refs_tab_label()

    def _update_refs_tab_label(self):
        s = self._krea()
        n = 0
        if (s.subject_layer_id or s.ref_layer_id or "").strip():
            n += 1
        n += len(s.role_sheet_entries())
        if not s.role_sheet_entries() and (s.sheet_layer_id or "").strip():
            n += 1
        label = _("Refs") if n == 0 else _("Refs ({n})").format(n=n)
        self.tabs.setTabText(1, label)

    def _sync_preset_buttons(self, preset: str):
        self.preset_turbo.setEnabled(preset != PRESET_TURBO)
        self.preset_raw.setEnabled(preset != PRESET_RAW)

    def _add_lora(self):
        self._sync_loras_to_model()
        self._sync_role_loras_to_model()
        self._sync_sliders_to_model()
        self._krea().add_lora()

    def _add_slider(self):
        self._sync_sliders_to_model()
        self._krea().add_slider()

    def _sync_role_loras_from_model(self):
        s = self._krea()
        self.face_lora_row.set_value(s.face_lora, s.face_lora_strength)
        self.body_lora_row.set_value(s.body_lora, s.body_lora_strength)
        self.style_lora_row.set_value(s.style_lora, s.style_lora_strength)
        self._update_advanced_tab_label()

    def _sync_role_loras_to_model(self):
        s = self._krea()
        if (
            self.face_lora_row.updating
            or self.body_lora_row.updating
            or self.style_lora_row.updating
        ):
            return
        s.face_lora = self.face_lora_row.lora_name()
        s.face_lora_strength = self.face_lora_row.lora_strength()
        s.body_lora = self.body_lora_row.lora_name()
        s.body_lora_strength = self.body_lora_row.lora_strength()
        s.style_lora = self.style_lora_row.lora_name()
        s.style_lora_strength = self.style_lora_row.lora_strength()
        self._update_advanced_tab_label()

    def _rebuild_loras(self):
        while self.lora_list.count():
            item = self.lora_list.takeAt(0)
            if w := item.widget():
                w.deleteLater()
        self._lora_rows.clear()
        for i, pl in enumerate(self._krea().loras):
            row = KreaEditLoraRow(i, pl, self)
            row._clear.clicked.connect(lambda _=False, idx=i: self._remove_lora(idx))
            row._combo.currentIndexChanged.connect(self._sync_loras_to_model)
            row._combo.editTextChanged.connect(lambda _t: self._sync_loras_to_model())
            row._strength.valueChanged.connect(lambda _v: self._sync_loras_to_model())
            self.lora_list.addWidget(row)
            self._lora_rows.append(row)
        self.add_lora_button.setEnabled(len(self._krea().loras) < MAX_EDIT_LORAS)
        self._update_advanced_tab_label()

    def _rebuild_sliders(self):
        while self.slider_list.count():
            item = self.slider_list.takeAt(0)
            if w := item.widget():
                w.deleteLater()
        self._slider_rows.clear()
        for i, pl in enumerate(self._krea().sliders):
            row = KreaEditSliderRow(i, pl, self)
            row._clear.clicked.connect(lambda _=False, idx=i: self._remove_slider(idx))
            row._combo.currentIndexChanged.connect(self._sync_sliders_to_model)
            row._combo.editTextChanged.connect(lambda _t: self._sync_sliders_to_model())
            row._strength.valueChanged.connect(lambda _v: self._sync_sliders_to_model())
            self.slider_list.addWidget(row)
            self._slider_rows.append(row)
        self.add_slider_button.setEnabled(len(self._krea().sliders) < MAX_EDIT_SLIDERS)
        self._update_advanced_tab_label()

    def _update_advanced_tab_label(self):
        s = self._krea()
        named = sum(1 for l in s.loras if l.name)
        named += sum(1 for l in s.sliders if l.name)
        for name in (s.face_lora, s.body_lora, s.style_lora):
            if (name or "").strip():
                named += 1
        label = _("Advanced") if named == 0 else _("Advanced ({n})").format(n=named)
        self.tabs.setTabText(2, label)

    def _remove_lora(self, index: int):
        self._sync_loras_to_model()
        self._krea().remove_lora(index)

    def _remove_slider(self, index: int):
        self._sync_sliders_to_model()
        self._krea().remove_slider(index)

    def _sync_loras_to_model(self):
        for row in self._lora_rows:
            self._krea().update_lora(row.index, row.lora_name(), row.lora_strength())

    def _sync_sliders_to_model(self):
        for row in self._slider_rows:
            self._krea().update_slider(row.index, row.lora_name(), row.lora_strength())

    def _update_target_label(self):
        e = self._krea().estimate_target()
        src = self._model.document.extent
        note = ""
        if e != src:
            note = _(" (downscaled from {w}×{h})").format(w=src.width, h=src.height)
        self.target_label.setText(f"{e.width}×{e.height}{note}")

    def _generate(self):
        self._sync_loras_to_model()
        self._sync_sliders_to_model()
        self._sync_role_loras_to_model()
        self._krea().subject_layer_id = self.subject_combo.currentData() or ""
        self._krea().sheet_layer_id = self.sheet_combo.currentData() or ""
        self._krea().sheet_face_layer_id = self.sheet_face_combo.currentData() or ""
        self._krea().sheet_hair_layer_id = self.sheet_hair_combo.currentData() or ""
        self._krea().sheet_clothing_layer_id = self.sheet_clothing_combo.currentData() or ""
        self._krea().sheet_pose_layer_id = self.sheet_pose_combo.currentData() or ""
        self._krea().ref_notes = self.ref_notes_edit.toPlainText()
        self._krea().generate()
