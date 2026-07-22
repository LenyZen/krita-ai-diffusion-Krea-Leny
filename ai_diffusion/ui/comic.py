from __future__ import annotations

from PyQt5.QtCore import QMetaObject, QSize, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from krita import Krita

from ..files import FileFilter
from ..image import Extent, Image
from ..layer import LayerType
from ..localization import translate as _
from ..model.comic import (
    MAX_CHARACTERS,
    MAX_PAGE_LORAS,
    MAX_SCENES,
    ComicCameraPreset,
    ComicLayout,
    ComicPanelStyle,
    CharacterSheet,
    PageLora,
    ScenePlate,
    layout_panel_count,
    layout_slots,
    slot_label,
)
from ..model.jobs import Job, JobKind, JobState
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind
from ..model.root import root
from . import theme
from .generation import HistoryWidget, ProgressBar
from .widget import ErrorBox, GenerateButton, QueueButton, StyleSelectWidget, WorkspaceSelectWidget


_LAYOUT_LABELS = {
    ComicLayout.grid_4: _("4-panel grid"),
    ComicLayout.grid_6: _("6-panel grid"),
    ComicLayout.grid_8: _("8-panel grid"),
    ComicLayout.nine_grid: _("9-panel grid"),
    ComicLayout.wide_5: _("Wide + 4 (5 panels)"),
    ComicLayout.splash_2: _("Splash + 2 insets"),
    ComicLayout.splash_3: _("Splash + 3 insets"),
    ComicLayout.banner_3: _("Banner + 2"),
    ComicLayout.strip_3: _("3 cinematic strips"),
    ComicLayout.stack_4: _("4 stacked strips"),
    ComicLayout.tall_left_4: _("Tall left + 3"),
    ComicLayout.inverted_splash_3: _("3 top + bottom splash"),
    ComicLayout.mosaic_4: _("Mosaic (1 + 3 overlays)"),
    ComicLayout.offset_6: _("Offset brick 6"),
}

# Display order in the layout combo (creative options grouped after classics)
_LAYOUT_ORDER = [
    ComicLayout.grid_4,
    ComicLayout.grid_6,
    ComicLayout.grid_8,
    ComicLayout.nine_grid,
    ComicLayout.wide_5,
    ComicLayout.splash_2,
    ComicLayout.splash_3,
    ComicLayout.banner_3,
    ComicLayout.strip_3,
    ComicLayout.stack_4,
    ComicLayout.tall_left_4,
    ComicLayout.inverted_splash_3,
    ComicLayout.mosaic_4,
    ComicLayout.offset_6,
]

_PANEL_STYLE_LABELS = {
    ComicPanelStyle.classic: _("Classic"),
    ComicPanelStyle.thin_black: _("Thin black"),
    ComicPanelStyle.bold_black: _("Bold black"),
    ComicPanelStyle.double_black: _("Double black"),
    ComicPanelStyle.red_keyline: _("Red keyline"),
    ComicPanelStyle.white_keyline: _("White keyline"),
    ComicPanelStyle.rounded: _("Rounded"),
    ComicPanelStyle.rounded_bold: _("Rounded bold"),
}

_PANEL_STYLE_ORDER = [
    ComicPanelStyle.classic,
    ComicPanelStyle.thin_black,
    ComicPanelStyle.bold_black,
    ComicPanelStyle.double_black,
    ComicPanelStyle.red_keyline,
    ComicPanelStyle.white_keyline,
    ComicPanelStyle.rounded,
    ComicPanelStyle.rounded_bold,
]

_CAMERA_PRESET_LABELS = {
    ComicCameraPreset.custom: _("Custom / free text"),
    ComicCameraPreset.establishing: _("Establishing (wide set)"),
    ComicCameraPreset.wide: _("Wide"),
    ComicCameraPreset.medium: _("Medium"),
    ComicCameraPreset.close_up: _("Close-up"),
    ComicCameraPreset.extreme_close: _("Extreme close-up"),
    ComicCameraPreset.over_shoulder: _("Over-the-shoulder"),
    ComicCameraPreset.low_angle: _("Low angle"),
    ComicCameraPreset.high_angle: _("High angle"),
    ComicCameraPreset.top_down: _("Top-down"),
}

_CAMERA_PRESET_ORDER = [
    ComicCameraPreset.custom,
    ComicCameraPreset.establishing,
    ComicCameraPreset.wide,
    ComicCameraPreset.medium,
    ComicCameraPreset.close_up,
    ComicCameraPreset.extreme_close,
    ComicCameraPreset.over_shoulder,
    ComicCameraPreset.low_angle,
    ComicCameraPreset.high_angle,
    ComicCameraPreset.top_down,
]


class ComicHistoryWidget(HistoryWidget):
    def is_finished(self, job: Job):
        return (
            job.kind is JobKind.comic
            and job.state is JobState.finished
            and bool(job.params.metadata.get("comic_slots"))
        )

    def add(self, job: Job):
        if not self.is_finished(job):
            return
        for i in range(self.count()):
            item = self.item(i)
            if item and item.text() == "" and item.data(Qt.ItemDataRole.UserRole) == job.id:
                item.setIcon(self._image_thumbnail(job, 0))
                return

        scrollbar = self.verticalScrollBar()
        scroll_to_bottom = scrollbar and scrollbar.value() >= scrollbar.maximum() - 4

        prompt = job.params.name if job.params.name != "" else "<no prompt>"
        header = QListWidgetItem(f"{job.timestamp:%H:%M} - {prompt}")
        header.setFlags(Qt.ItemFlag.NoItemFlags)
        header.setData(Qt.ItemDataRole.UserRole, job.id)
        header.setData(Qt.ItemDataRole.ToolTipRole, job.params.prompt)
        header.setSizeHint(QSize(9999, self.fontMetrics().lineSpacing() + 4))
        self.addItem(header)
        self._add_item(job, QListWidgetItem(self._image_thumbnail(job, 0), None), 0)

        if scroll_to_bottom:
            self.scrollToBottom()

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        self.setCurrentItem(item)
        job_id, index = self.item_info(item)
        job = self._model.jobs.find(job_id)
        if job is None or job.kind is not JobKind.comic:
            super()._show_context_menu(pos)
            return

        menu = QMenu(self)
        menu.addAction(_("Apply Page"), lambda: self._model.apply_generated_result(job_id, index))
        menu.addSeparator()
        menu.addAction(_("Copy Prompt"), self._copy_prompt)
        menu.addAction(_("Discard Image"), self._discard_image)
        menu.exec(self.mapToGlobal(pos))


class ComicLoraRow(QWidget):
    """One page-level LoRA picker row."""

    def __init__(self, index: int, lora: PageLora, parent=None):
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
        self._combo.setToolTip(_("Page LoRA applied to every panel"))
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


class ScenePlateCard(QWidget):
    def __init__(self, scene: ScenePlate, parent=None):
        super().__init__(parent)
        self.scene_id = scene.id
        self.setStyleSheet(f"border-left: 3px solid {theme.grey}; padding-left: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.name_edit = QLineEdit(scene.name, self)
        self.name_edit.setPlaceholderText(_("Scene name (e.g. Kitchen — noon)"))
        header.addWidget(self.name_edit, 1)
        self.remove_button = QPushButton(Krita.instance().icon("deletelayer"), "")
        self.remove_button.setFixedSize(24, 24)
        self.remove_button.setToolTip(_("Remove scene plate"))
        header.addWidget(self.remove_button)
        layout.addLayout(header)

        self.description_edit = QPlainTextEdit(scene.description, self)
        self.description_edit.setPlaceholderText(
            _("Stage bible — architecture, props, lighting, time of day (no main characters)")
        )
        self.description_edit.setMinimumHeight(56)
        layout.addWidget(self.description_edit, 1)

        plate_row = QHBoxLayout()
        plate_row.addWidget(QLabel(_("Plate layer"), self))
        self.plate_combo = QComboBox(self)
        self.plate_combo.setToolTip(
            _(
                "Optional empty-set paint layer. When set, panel generation starts from this "
                "background (img2img) so the room stays locked."
            )
        )
        plate_row.addWidget(self.plate_combo, 1)
        layout.addLayout(plate_row)


class CharacterSheetCard(QWidget):
    def __init__(self, sheet: CharacterSheet, parent=None):
        super().__init__(parent)
        self.character_id = sheet.id
        self.setStyleSheet(f"border-left: 3px solid {theme.grey}; padding-left: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.name_edit = QLineEdit(sheet.name, self)
        self.name_edit.setPlaceholderText(_("Character name"))
        header.addWidget(self.name_edit, 1)
        self.remove_button = QPushButton(Krita.instance().icon("deletelayer"), "")
        self.remove_button.setFixedSize(24, 24)
        self.remove_button.setToolTip(_("Remove character sheet"))
        header.addWidget(self.remove_button)
        layout.addLayout(header)

        self.synopsis_edit = QPlainTextEdit(sheet.synopsis, self)
        self.synopsis_edit.setPlaceholderText(
            _("Character synopsis — appearance, costume, personality cues")
        )
        self.synopsis_edit.setMinimumHeight(56)
        layout.addWidget(self.synopsis_edit, 1)

        face_row = QHBoxLayout()
        face_row.addWidget(QLabel(_("Face"), self))
        self.face_combo = QComboBox(self)
        self.face_combo.setToolTip(_("Optional paint layer used as Face reference"))
        face_row.addWidget(self.face_combo, 1)
        layout.addLayout(face_row)

        lora_row = QHBoxLayout()
        lora_row.addWidget(QLabel(_("LoRA"), self))
        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True
        completer = QCompleter(self._loras)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.lora_combo = QComboBox(self)
        self.lora_combo.setEditable(True)
        self.lora_combo.setModel(self._loras)
        self.lora_combo.setCompleter(completer)
        if line_edit := self.lora_combo.lineEdit():
            line_edit.setPlaceholderText(_("(none)"))
        lora_row.addWidget(self.lora_combo, 1)
        self.lora_strength = QDoubleSpinBox(self)
        self.lora_strength.setRange(0.05, 3.0)
        self.lora_strength.setSingleStep(0.05)
        self.lora_strength.setDecimals(2)
        self.lora_strength.setValue(sheet.lora_strength)
        lora_row.addWidget(self.lora_strength)
        layout.addLayout(lora_row)

        self._set_lora(sheet.lora_name, sheet.lora_strength)

    def _set_lora(self, name: str, strength: float):
        self.lora_strength.setValue(strength)
        if name:
            idx = self.lora_combo.findData(name)
            if idx >= 0:
                self.lora_combo.setCurrentIndex(idx)
            else:
                self.lora_combo.setEditText(name)
        else:
            self.lora_combo.setCurrentIndex(-1)
            self.lora_combo.setEditText("")

    def lora_name(self) -> str:
        data = self.lora_combo.currentData()
        if data:
            return str(data)
        text = self.lora_combo.currentText().strip()
        if not text:
            return ""
        if file := root.files.loras.find(text):
            return file.id
        return text


class ComicWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._panel_location_edits: list[QPlainTextEdit] = []
        self._panel_camera_edits: list[QPlainTextEdit] = []
        self._panel_action_edits: list[QPlainTextEdit] = []
        self._panel_final_edits: list[QPlainTextEdit] = []
        self._panel_same_location: list[QCheckBox] = []
        self._panel_scene_combos: list[QComboBox] = []
        self._panel_camera_combos: list[QComboBox] = []
        self._panel_cast_boxes: list[list[QCheckBox]] = []
        self._panel_thumbs: list[QLabel] = []
        self._panel_status: list[QLabel] = []
        self._panel_lock_buttons: list[QPushButton] = []
        self._panel_finalize_buttons: list[QPushButton] = []
        self._scene_cards: dict[str, ScenePlateCard] = {}
        self._character_cards: dict[str, CharacterSheetCard] = {}
        self._lora_rows: list[ComicLoraRow] = []
        self._rebuilding = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 4, 0)
        outer.setSpacing(4)

        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        outer.addWidget(self.splitter, 1)

        # --- Top: compact page settings ---
        top = QWidget(self.splitter)
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self.workspace_select = WorkspaceSelectWidget(top)
        self.style_select = StyleSelectWidget(top)
        header = QHBoxLayout()
        header.addWidget(self.workspace_select)
        header.addWidget(self.style_select, 1)
        top_layout.addLayout(header)

        info = QLabel(
            _(
                "Character sheets + panel beats drive each panel. Draft for a quick look, "
                "then Refine using the draft as guidance."
            ),
            top,
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {theme.grey};")
        top_layout.addWidget(info)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel(_("Layout"), top))
        self.layout_combo = QComboBox(top)
        for layout_kind in _LAYOUT_ORDER:
            self.layout_combo.addItem(_LAYOUT_LABELS[layout_kind], layout_kind)
        row1.addWidget(self.layout_combo, 1)
        top_layout.addLayout(row1)

        style_frame_row = QHBoxLayout()
        style_frame_row.addWidget(QLabel(_("Panel style"), top))
        self.panel_style_combo = QComboBox(top)
        self.panel_style_combo.setToolTip(
            _("Keyline and frame shape for every panel on this page")
        )
        for style_kind in _PANEL_STYLE_ORDER:
            self.panel_style_combo.addItem(_PANEL_STYLE_LABELS[style_kind], style_kind)
        style_frame_row.addWidget(self.panel_style_combo, 1)
        top_layout.addLayout(style_frame_row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel(_("Title"), top))
        self.title_edit = QLineEdit(top)
        self.title_edit.setPlaceholderText(_("Optional title"))
        row2.addWidget(self.title_edit, 1)
        top_layout.addLayout(row2)

        top_layout.addWidget(QLabel(_("Page synopsis"), top))
        self.synopsis_edit = QPlainTextEdit(top)
        self.synopsis_edit.setPlaceholderText(
            _("Shared story context for every panel on this page")
        )
        self.synopsis_edit.setMaximumHeight(64)
        top_layout.addWidget(self.synopsis_edit)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel(_("Style ref"), top))
        self.style_ref_combo = QComboBox(top)
        self.style_ref_combo.setToolTip(
            _(
                "Unused for panel generation (style vision refs caused multi-panel "
                "duplicates). Use Face refs and page/style LoRAs for look instead."
            )
        )
        style_row.addWidget(self.style_ref_combo, 1)
        top_layout.addLayout(style_row)

        lora_header = QHBoxLayout()
        lora_header.addWidget(QLabel(_("Page LoRAs"), top))
        lora_header.addStretch()
        self.add_lora_button = QPushButton(_("Add LoRA"), top)
        self.add_lora_button.setFixedHeight(22)
        lora_header.addWidget(self.add_lora_button)
        top_layout.addLayout(lora_header)
        self.loras_container = QWidget(top)
        self.loras_layout = QVBoxLayout(self.loras_container)
        self.loras_layout.setContentsMargins(0, 0, 0, 0)
        self.loras_layout.setSpacing(2)
        top_layout.addWidget(self.loras_container)

        self.splitter.addWidget(top)

        # --- Middle: Characters | Panels tabs (expandable) ---
        middle = QWidget(self.splitter)
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget(middle)
        middle_layout.addWidget(self.tabs, 1)

        # Characters tab
        chars_page = QWidget()
        chars_layout = QVBoxLayout(chars_page)
        chars_layout.setContentsMargins(0, 4, 0, 0)
        chars_header = QHBoxLayout()
        self.characters_label = QLabel(_("Character sheets"), chars_page)
        chars_header.addWidget(self.characters_label)
        chars_header.addStretch()
        self.add_character_button = QPushButton(_("Add Character"), chars_page)
        chars_header.addWidget(self.add_character_button)
        chars_layout.addLayout(chars_header)

        chars_scroll = QScrollArea(chars_page)
        chars_scroll.setWidgetResizable(True)
        chars_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.characters_container = QWidget()
        self.characters_layout = QVBoxLayout(self.characters_container)
        self.characters_layout.setContentsMargins(0, 0, 0, 0)
        self.characters_layout.addStretch()
        chars_scroll.setWidget(self.characters_container)
        chars_layout.addWidget(chars_scroll, 1)
        self.tabs.addTab(chars_page, _("Characters"))

        # Scenes tab — locked stages / background plates
        scenes_page = QWidget()
        scenes_layout = QVBoxLayout(scenes_page)
        scenes_layout.setContentsMargins(0, 4, 0, 0)
        scenes_header = QHBoxLayout()
        self.scenes_label = QLabel(_("Scene plates"), scenes_page)
        scenes_header.addWidget(self.scenes_label)
        scenes_header.addStretch()
        self.add_scene_button = QPushButton(_("Add Scene"), scenes_page)
        scenes_header.addWidget(self.add_scene_button)
        scenes_layout.addLayout(scenes_header)
        scenes_hint = QLabel(
            _(
                "Define reusable stages. Description feeds Location; an optional Plate layer "
                "locks the background via img2img when generating panels."
            ),
            scenes_page,
        )
        scenes_hint.setWordWrap(True)
        scenes_hint.setStyleSheet(f"color: {theme.grey};")
        scenes_layout.addWidget(scenes_hint)
        scenes_scroll = QScrollArea(scenes_page)
        scenes_scroll.setWidgetResizable(True)
        scenes_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scenes_container = QWidget()
        self.scenes_layout = QVBoxLayout(self.scenes_container)
        self.scenes_layout.setContentsMargins(0, 0, 0, 0)
        self.scenes_layout.addStretch()
        scenes_scroll.setWidget(self.scenes_container)
        scenes_layout.addWidget(scenes_scroll, 1)
        self.tabs.addTab(scenes_page, _("Scenes"))

        # Panels tab
        panels_page = QWidget()
        panels_layout = QVBoxLayout(panels_page)
        panels_layout.setContentsMargins(0, 4, 0, 0)
        self.panels_label = QLabel(_("Panel beats"), panels_page)
        panels_layout.addWidget(self.panels_label)

        panels_scroll = QScrollArea(panels_page)
        panels_scroll.setWidgetResizable(True)
        panels_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.panels_container = QWidget()
        self.panels_layout = QVBoxLayout(self.panels_container)
        self.panels_layout.setContentsMargins(0, 0, 0, 0)
        self.panels_layout.addStretch()
        panels_scroll.setWidget(self.panels_container)
        panels_layout.addWidget(panels_scroll, 1)
        self.tabs.addTab(panels_page, _("Panels"))

        self.splitter.addWidget(middle)

        # --- Bottom: actions + history ---
        bottom = QWidget(self.splitter)
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        actions = QHBoxLayout()
        self.draft_button = GenerateButton(JobKind.comic, bottom)
        self.draft_button.operation = _("Draft Page")
        self.draft_button.clicked.connect(self.draft)
        actions.addWidget(self.draft_button)

        self.generate_button = GenerateButton(JobKind.comic, bottom)
        self.generate_button.operation = _("Generate Page")
        self.generate_button.clicked.connect(self.generate)
        actions.addWidget(self.generate_button)

        self.remaining_button = QPushButton(_("Generate Remaining"), bottom)
        self.remaining_button.setToolTip(
            _("Generate only unlocked panels; locked panels are pasted as-is")
        )
        self.remaining_button.clicked.connect(self.generate_remaining)
        actions.addWidget(self.remaining_button)

        self.assemble_button = QPushButton(_("Assemble Page"), bottom)
        self.assemble_button.setEnabled(False)
        self.assemble_button.setToolTip(
            _("Composite all locked panels into a page (no new diffusion)")
        )
        self.assemble_button.clicked.connect(self.assemble)
        actions.addWidget(self.assemble_button)

        self.refine_button = GenerateButton(JobKind.comic, bottom)
        self.refine_button.operation = _("Refine Page")
        self.refine_button.setEnabled(False)
        self.refine_button.setToolTip(
            _("Run after Draft Page — uses draft panels as guidance for a full-quality pass")
        )
        self.refine_button.clicked.connect(self.refine)
        actions.addWidget(self.refine_button)

        self.queue_button = QueueButton(parent=bottom)
        self.queue_button.setFixedHeight(self.generate_button.sizeHint().height())
        actions.addWidget(self.queue_button)
        bottom_layout.addLayout(actions)

        self.progress_text = QLabel("", bottom)
        self.progress_text.setStyleSheet(f"color: {theme.grey};")
        bottom_layout.addWidget(self.progress_text)

        self.progress_bar = ProgressBar(bottom)
        bottom_layout.addWidget(self.progress_bar)

        self.error_box = ErrorBox(bottom)
        bottom_layout.addWidget(self.error_box)

        self.history = ComicHistoryWidget(bottom)
        self.history.item_activated.connect(self.apply_result)
        bottom_layout.addWidget(self.history, 1)

        self.splitter.addWidget(bottom)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 3)
        self.splitter.setSizes([220, 360, 280])

        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        self.panel_style_combo.currentIndexChanged.connect(self._on_panel_style_changed)
        self.title_edit.textChanged.connect(self._on_title_changed)
        self.synopsis_edit.textChanged.connect(self._on_synopsis_changed)
        self.style_ref_combo.currentIndexChanged.connect(self._on_style_ref_changed)
        self.add_character_button.clicked.connect(self._add_character)
        self.add_scene_button.clicked.connect(self._add_scene)
        self.add_lora_button.clicked.connect(self._add_page_lora)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "style", self.style_select, "value"),
                bind(model.comic, "can_generate", self.generate_button, "enabled", Bind.one_way),
                bind(model.comic, "can_generate", self.draft_button, "enabled", Bind.one_way),
                bind(model.comic, "can_generate", self.remaining_button, "enabled", Bind.one_way),
                bind(model.comic, "can_assemble", self.assemble_button, "enabled", Bind.one_way),
                bind(model.comic, "progress_text", self.progress_text, "text", Bind.one_way),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                model.comic.layout_changed.connect(self._sync_layout_combo),
                model.comic.panel_style_changed.connect(self._sync_panel_style_combo),
                model.comic.title_changed.connect(self._sync_title),
                model.comic.synopsis_changed.connect(self._sync_synopsis),
                model.comic.panels_changed.connect(self._rebuild_panels),
                model.comic.panel_images_changed.connect(self._refresh_panel_thumbs),
                model.comic.characters_changed.connect(self._rebuild_characters),
                model.comic.scenes_changed.connect(self._rebuild_scenes),
                model.comic.page_loras_changed.connect(self._rebuild_page_loras),
                model.comic.style_layer_id_changed.connect(self._sync_style_ref_combo),
                model.comic.can_generate_changed.connect(self._sync_refine_enabled),
                model.comic.can_refine_changed.connect(self._sync_refine_enabled),
                model.layers.changed.connect(self._on_layers_changed),
            ]
            self.generate_button.model = model
            self.draft_button.model = model
            self.refine_button.model = model
            self.queue_button.model = model
            self.progress_bar.model = model
            self.history.model_ = model
            self._sync_layout_combo()
            self._sync_panel_style_combo()
            self._sync_title()
            self._sync_synopsis()
            self._refresh_style_ref_combo()
            self._rebuild_page_loras()
            self._rebuild_characters()
            self._rebuild_scenes()
            self._rebuild_panels()
            self._sync_refine_enabled()

    def _sync_refine_enabled(self, *_args):
        comic = self._model.comic
        self.refine_button.setEnabled(comic.can_generate and comic.can_refine)

    def _sync_layout_combo(self):
        idx = self.layout_combo.findData(self._model.comic.layout)
        if idx >= 0 and idx != self.layout_combo.currentIndex():
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentIndex(idx)
            self.layout_combo.blockSignals(False)

    def _sync_panel_style_combo(self):
        idx = self.panel_style_combo.findData(self._model.comic.panel_style)
        if idx >= 0 and idx != self.panel_style_combo.currentIndex():
            self.panel_style_combo.blockSignals(True)
            self.panel_style_combo.setCurrentIndex(idx)
            self.panel_style_combo.blockSignals(False)

    def _sync_title(self):
        if self.title_edit.text() != self._model.comic.title:
            self.title_edit.blockSignals(True)
            self.title_edit.setText(self._model.comic.title)
            self.title_edit.blockSignals(False)

    def _sync_synopsis(self):
        if self.synopsis_edit.toPlainText() != self._model.comic.synopsis:
            self.synopsis_edit.blockSignals(True)
            self.synopsis_edit.setPlainText(self._model.comic.synopsis)
            self.synopsis_edit.blockSignals(False)

    def _on_layout_changed(self):
        data = self.layout_combo.currentData()
        if data is not None:
            self._model.comic.layout = data

    def _on_panel_style_changed(self):
        data = self.panel_style_combo.currentData()
        if data is not None:
            self._model.comic.panel_style = data

    def _on_title_changed(self, text: str):
        self._model.comic.title = text

    def _on_synopsis_changed(self):
        self._model.comic.synopsis = self.synopsis_edit.toPlainText()

    def _fill_layer_combo(self, combo: QComboBox, current: str):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(_("(none)"), "")
        for layer in self._model.layers.all:
            if layer.type in (LayerType.paint, LayerType.group, LayerType.file, LayerType.clone):
                combo.addItem(layer.name, layer.id_string)
        idx = combo.findData(current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _refresh_style_ref_combo(self):
        self._fill_layer_combo(self.style_ref_combo, self._model.comic.style_layer_id)

    def _sync_style_ref_combo(self):
        idx = self.style_ref_combo.findData(self._model.comic.style_layer_id)
        if idx >= 0 and idx != self.style_ref_combo.currentIndex():
            self.style_ref_combo.blockSignals(True)
            self.style_ref_combo.setCurrentIndex(idx)
            self.style_ref_combo.blockSignals(False)

    def _on_style_ref_changed(self):
        data = self.style_ref_combo.currentData()
        self._model.comic.style_layer_id = data or ""

    def _on_layers_changed(self):
        self._refresh_style_ref_combo()
        for card in self._character_cards.values():
            current = card.face_combo.currentData() or ""
            self._fill_layer_combo(card.face_combo, current)
        for card in self._scene_cards.values():
            current = card.plate_combo.currentData() or ""
            self._fill_layer_combo(card.plate_combo, current)

    def _add_character(self):
        self._sync_characters_to_model()
        self._model.comic.add_character()

    def _add_scene(self):
        self._sync_scenes_to_model()
        self._model.comic.add_scene()

    def _add_page_lora(self):
        self._sync_page_loras_to_model()
        self._model.comic.add_page_lora()

    def _rebuild_page_loras(self):
        while self.loras_layout.count():
            item = self.loras_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._lora_rows.clear()
        for i, pl in enumerate(self._model.comic.page_loras):
            row = ComicLoraRow(i, pl, self.loras_container)
            row._clear.clicked.connect(lambda _=False, idx=i: self._remove_page_lora(idx))
            row._combo.currentIndexChanged.connect(self._sync_page_loras_to_model)
            row._combo.editTextChanged.connect(lambda _t: self._sync_page_loras_to_model())
            row._strength.valueChanged.connect(lambda _v: self._sync_page_loras_to_model())
            self.loras_layout.addWidget(row)
            self._lora_rows.append(row)
        self.add_lora_button.setEnabled(len(self._model.comic.page_loras) < MAX_PAGE_LORAS)

    def _remove_page_lora(self, index: int):
        self._sync_page_loras_to_model()
        self._model.comic.remove_page_lora(index)

    def _sync_page_loras_to_model(self):
        if self._rebuilding:
            return
        for row in self._lora_rows:
            self._model.comic.update_page_lora(row.index, row.lora_name(), row.lora_strength())

    def _rebuild_characters(self):
        self._rebuilding = True
        while self.characters_layout.count() > 1:
            item = self.characters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._character_cards.clear()

        comic = self._model.comic
        self.characters_label.setText(
            _("Character sheets ({n}/{max})").format(n=len(comic.characters), max=MAX_CHARACTERS)
        )
        self.add_character_button.setEnabled(len(comic.characters) < MAX_CHARACTERS)

        for i, sheet in enumerate(comic.characters):
            card = CharacterSheetCard(sheet, self.characters_container)
            self._fill_layer_combo(card.face_combo, sheet.face_layer_id)
            card.remove_button.clicked.connect(
                lambda _=False, cid=sheet.id: self._remove_character(cid)
            )
            card.name_edit.textChanged.connect(
                lambda text, cid=sheet.id: self._model.comic.update_character(cid, name=text)
            )
            card.synopsis_edit.textChanged.connect(
                lambda cid=sheet.id, edit=card.synopsis_edit: self._model.comic.update_character(
                    cid, synopsis=edit.toPlainText()
                )
            )
            card.face_combo.currentIndexChanged.connect(
                lambda _i, cid=sheet.id, c=card: self._model.comic.update_character(
                    cid, face_layer_id=c.face_combo.currentData() or ""
                )
            )
            card.lora_combo.currentIndexChanged.connect(
                lambda _i, cid=sheet.id, c=card: self._model.comic.update_character(
                    cid, lora_name=c.lora_name(), lora_strength=c.lora_strength.value()
                )
            )
            card.lora_combo.editTextChanged.connect(
                lambda _t, cid=sheet.id, c=card: self._model.comic.update_character(
                    cid, lora_name=c.lora_name(), lora_strength=c.lora_strength.value()
                )
            )
            card.lora_strength.valueChanged.connect(
                lambda v, cid=sheet.id, c=card: self._model.comic.update_character(
                    cid, lora_name=c.lora_name(), lora_strength=v
                )
            )
            self.characters_layout.insertWidget(i, card)
            self._character_cards[sheet.id] = card

        self._rebuilding = False
        # Cast checkboxes depend on character list
        self._rebuild_panels()

    def _remove_character(self, character_id: str):
        self._sync_characters_to_model()
        self._model.comic.remove_character(character_id)

    def _sync_characters_to_model(self):
        for cid, card in self._character_cards.items():
            self._model.comic.update_character(
                cid,
                name=card.name_edit.text(),
                synopsis=card.synopsis_edit.toPlainText(),
                face_layer_id=card.face_combo.currentData() or "",
                lora_name=card.lora_name(),
                lora_strength=card.lora_strength.value(),
            )

    def _rebuild_scenes(self):
        self._rebuilding = True
        while self.scenes_layout.count() > 1:
            item = self.scenes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._scene_cards.clear()

        comic = self._model.comic
        self.scenes_label.setText(
            _("Scene plates ({n}/{max})").format(n=len(comic.scenes), max=MAX_SCENES)
        )
        self.add_scene_button.setEnabled(len(comic.scenes) < MAX_SCENES)

        for i, scene in enumerate(comic.scenes):
            card = ScenePlateCard(scene, self.scenes_container)
            self._fill_layer_combo(card.plate_combo, scene.layer_id)
            card.remove_button.clicked.connect(
                lambda _=False, sid=scene.id: self._remove_scene(sid)
            )
            card.name_edit.textChanged.connect(
                lambda text, sid=scene.id: self._model.comic.update_scene(sid, name=text)
            )
            card.description_edit.textChanged.connect(
                lambda sid=scene.id, edit=card.description_edit: self._model.comic.update_scene(
                    sid, description=edit.toPlainText()
                )
            )
            card.plate_combo.currentIndexChanged.connect(
                lambda _i, sid=scene.id, c=card: self._model.comic.update_scene(
                    sid, layer_id=c.plate_combo.currentData() or ""
                )
            )
            self.scenes_layout.insertWidget(i, card)
            self._scene_cards[scene.id] = card

        self._rebuilding = False
        # Panel scene dropdowns depend on scene list
        self._rebuild_panels()

    def _remove_scene(self, scene_id: str):
        self._sync_scenes_to_model()
        self._model.comic.remove_scene(scene_id)

    def _sync_scenes_to_model(self):
        for sid, card in self._scene_cards.items():
            self._model.comic.update_scene(
                sid,
                name=card.name_edit.text(),
                description=card.description_edit.toPlainText(),
                layer_id=card.plate_combo.currentData() or "",
            )

    def _rebuild_panels(self):
        comic = self._model.comic
        count = layout_panel_count(comic.layout)
        self.panels_label.setText(_("Panel beats ({n})").format(n=count))

        while self.panels_layout.count() > 1:
            item = self.panels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._panel_location_edits.clear()
        self._panel_camera_edits.clear()
        self._panel_action_edits.clear()
        self._panel_final_edits.clear()
        self._panel_same_location.clear()
        self._panel_scene_combos.clear()
        self._panel_camera_combos.clear()
        self._panel_cast_boxes.clear()
        self._panel_thumbs = []
        self._panel_status = []
        self._panel_lock_buttons = []
        self._panel_finalize_buttons = []

        slots = layout_slots(comic.layout)
        for i in range(count):
            wrap = QFrame(self.panels_container)
            wrap.setFrameShape(QFrame.Shape.StyledPanel)
            row = QVBoxLayout(wrap)
            row.setContentsMargins(4, 4, 4, 8)

            header = QHBoxLayout()
            label = slot_label(slots[i]) if i < len(slots) else _("Panel {n}").format(n=i + 1)
            header.addWidget(QLabel(label, wrap))
            status = QLabel("", wrap)
            status.setStyleSheet(f"color: {theme.grey};")
            header.addWidget(status)
            header.addStretch()
            row.addLayout(header)

            body = QHBoxLayout()
            thumb = QLabel(wrap)
            thumb.setFixedSize(72, 72)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setStyleSheet(f"border: 1px solid {theme.grey}; background: transparent;")
            thumb.setScaledContents(False)
            body.addWidget(thumb)

            right = QVBoxLayout()
            panel = comic.panels[i] if i < len(comic.panels) else None

            scene_row = QHBoxLayout()
            scene_row.addWidget(QLabel(_("Scene"), wrap))
            scene_combo = QComboBox(wrap)
            scene_combo.setToolTip(
                _("Lock this panel to a Scene plate (stage bible + optional BG layer)")
            )
            scene_combo.addItem(_("— none —"), "")
            for scene in comic.scenes:
                label_s = scene.name.strip() or _("Unnamed scene")
                if scene.layer_id:
                    label_s = f"{label_s} ★"
                scene_combo.addItem(label_s, scene.id)
            if panel and panel.scene_id:
                idx_s = scene_combo.findData(panel.scene_id)
                if idx_s >= 0:
                    scene_combo.setCurrentIndex(idx_s)
            scene_combo.currentIndexChanged.connect(lambda _i, idx=i: self._sync_panel_fields(idx))
            scene_row.addWidget(scene_combo, 1)
            if i > 0:
                same_box = QCheckBox(_("Same as previous"), wrap)
                same_box.setChecked(bool(panel and panel.same_location))
                same_box.stateChanged.connect(lambda _s, idx=i: self._sync_panel_fields(idx))
                scene_row.addWidget(same_box)
            else:
                same_box = QCheckBox(wrap)
                same_box.setVisible(False)
            right.addLayout(scene_row)

            loc_edit = QPlainTextEdit(wrap)
            loc_edit.setPlaceholderText(
                _("Extra location notes (optional if Scene is set)…")
                if comic.scenes
                else _("Stage, backdrop, time of day, lighting…")
            )
            loc_edit.setMaximumHeight(44)
            if panel:
                loc_edit.setPlainText(panel.location)
            loc_edit.textChanged.connect(lambda idx=i: self._sync_panel_fields(idx))
            right.addWidget(loc_edit)

            cam_row = QHBoxLayout()
            cam_row.addWidget(QLabel(_("Camera"), wrap))
            cam_combo = QComboBox(wrap)
            cam_combo.setToolTip(_("Shot type preset — free text below can refine it"))
            for preset in _CAMERA_PRESET_ORDER:
                cam_combo.addItem(_CAMERA_PRESET_LABELS[preset], preset)
            if panel:
                idx_c = cam_combo.findData(panel.camera_preset)
                if idx_c >= 0:
                    cam_combo.setCurrentIndex(idx_c)
            cam_combo.currentIndexChanged.connect(lambda _i, idx=i: self._sync_panel_fields(idx))
            cam_row.addWidget(cam_combo, 1)
            right.addLayout(cam_row)

            cam_edit = QPlainTextEdit(wrap)
            cam_edit.setPlaceholderText(_("Custom angle notes (optional with a preset)…"))
            cam_edit.setMaximumHeight(44)
            if panel:
                cam_edit.setPlainText(panel.camera)
            cam_edit.textChanged.connect(lambda idx=i: self._sync_panel_fields(idx))
            right.addWidget(cam_edit)

            right.addWidget(QLabel(_("Action"), wrap))
            act_edit = QPlainTextEdit(wrap)
            act_edit.setPlaceholderText(_("Who is where, pose, gesture, emotion…"))
            act_edit.setMaximumHeight(48)
            if panel:
                act_edit.setPlainText(panel.action or panel.text)
            act_edit.textChanged.connect(lambda idx=i: self._sync_panel_fields(idx))
            right.addWidget(act_edit)

            fin_row = QHBoxLayout()
            fin_row.addWidget(QLabel(_("Final description"), wrap))
            finalize_btn = QPushButton(_("Finalize"), wrap)
            finalize_btn.setToolTip(
                _("Polish location + camera + action with Qwen into a Krea-ready prompt")
            )
            finalize_btn.clicked.connect(lambda _=False, idx=i: self._finalize_panel(idx))
            fin_row.addWidget(finalize_btn)
            fin_row.addStretch()
            right.addLayout(fin_row)

            final_edit = QPlainTextEdit(wrap)
            final_edit.setPlaceholderText(
                _("Used for generation. Run Finalize or write your own detailed scene prompt.")
            )
            final_edit.setMinimumHeight(64)
            if panel:
                final_edit.setPlainText(panel.final)
            final_edit.textChanged.connect(lambda idx=i: self._sync_panel_fields(idx))
            right.addWidget(final_edit, 1)

            cast_row = QHBoxLayout()
            cast_row.addWidget(QLabel(_("Cast:"), wrap))
            boxes: list[QCheckBox] = []
            selected = set(panel.character_ids) if panel else set()
            if not comic.characters:
                hint = QLabel(_("Add characters in the Characters tab"), wrap)
                hint.setStyleSheet(f"color: {theme.grey};")
                cast_row.addWidget(hint, 1)
            else:
                for ch in comic.characters:
                    box = QCheckBox(ch.name.strip() or _("Unnamed"), wrap)
                    box.setChecked(ch.id in selected)
                    box.stateChanged.connect(lambda _s, idx=i: self._sync_panel_cast(idx))
                    cast_row.addWidget(box)
                    boxes.append(box)
                cast_row.addStretch()
            right.addLayout(cast_row)

            btns = QHBoxLayout()
            gen_btn = QPushButton(_("Generate"), wrap)
            gen_btn.setToolTip(_("Generate this panel only"))
            gen_btn.clicked.connect(lambda _=False, idx=i: self._generate_panel(idx))
            btns.addWidget(gen_btn)

            lock_btn = QPushButton(_("Lock"), wrap)
            lock_btn.clicked.connect(lambda _=False, idx=i: self._toggle_lock(idx))
            btns.addWidget(lock_btn)

            clear_btn = QPushButton(_("Clear"), wrap)
            clear_btn.clicked.connect(lambda _=False, idx=i: self._clear_panel(idx))
            btns.addWidget(clear_btn)

            reload_btn = QToolButton(wrap)
            reload_btn.setText(_("Layer"))
            reload_btn.setToolTip(_("Reload pixels from the Comic Working layer"))
            reload_btn.clicked.connect(
                lambda _=False, idx=i: self._model.comic.reload_panel_from_layer(idx)
            )
            btns.addWidget(reload_btn)
            btns.addStretch()
            right.addLayout(btns)

            body.addLayout(right, 1)
            row.addLayout(body)

            self.panels_layout.insertWidget(i, wrap)
            self._panel_location_edits.append(loc_edit)
            self._panel_camera_edits.append(cam_edit)
            self._panel_action_edits.append(act_edit)
            self._panel_final_edits.append(final_edit)
            self._panel_same_location.append(same_box)
            self._panel_scene_combos.append(scene_combo)
            self._panel_camera_combos.append(cam_combo)
            self._panel_cast_boxes.append(boxes)
            self._panel_thumbs.append(thumb)
            self._panel_status.append(status)
            self._panel_lock_buttons.append(lock_btn)
            self._panel_finalize_buttons.append(finalize_btn)
            if i > 0 and same_box.isChecked():
                loc_edit.setEnabled(False)
                scene_combo.setEnabled(False)

        self._refresh_panel_thumbs()

    def _refresh_panel_thumbs(self):
        comic = self._model.comic
        if not hasattr(self, "_panel_thumbs"):
            return
        for i, thumb in enumerate(self._panel_thumbs):
            img = comic.panel_image(i)
            locked = i < len(comic.panels) and comic.panels[i].locked
            if img is not None:
                tip = Image.scale_to_fit(img, Extent(72, 72))
                thumb.setPixmap(tip.to_pixmap())
            else:
                thumb.setPixmap(QPixmap())
                thumb.setText("—")
            if i < len(self._panel_status):
                if locked:
                    self._panel_status[i].setText(_("Locked"))
                elif img is not None:
                    self._panel_status[i].setText(_("Ready"))
                else:
                    self._panel_status[i].setText(_("Empty"))
            if i < len(self._panel_lock_buttons):
                self._panel_lock_buttons[i].setText(_("Unlock") if locked else _("Lock"))

    def _generate_panel(self, index: int):
        self._sync_all_to_model()
        self._model.comic.generate_panel(index)

    def _finalize_panel(self, index: int):
        self._sync_all_to_model()
        self._model.comic.finalize_panel(index)

    def _toggle_lock(self, index: int):
        self._sync_all_to_model()
        comic = self._model.comic
        if index < len(comic.panels) and comic.panels[index].locked:
            comic.unlock_panel(index)
        else:
            comic.lock_panel(index)
        self._refresh_panel_thumbs()

    def _clear_panel(self, index: int):
        self._model.comic.clear_panel(index)
        self._refresh_panel_thumbs()

    def _sync_panel_fields(self, index: int):
        if index >= len(self._panel_location_edits):
            return
        same = False
        if index < len(self._panel_same_location):
            same = self._panel_same_location[index].isChecked()
        scene_id = ""
        if index < len(self._panel_scene_combos):
            scene_id = self._panel_scene_combos[index].currentData() or ""
        camera_preset = ComicCameraPreset.custom
        if index < len(self._panel_camera_combos):
            data = self._panel_camera_combos[index].currentData()
            if isinstance(data, ComicCameraPreset):
                camera_preset = data
        self._model.comic.update_panel_fields(
            index,
            location=self._panel_location_edits[index].toPlainText(),
            camera=self._panel_camera_edits[index].toPlainText(),
            action=self._panel_action_edits[index].toPlainText(),
            final=self._panel_final_edits[index].toPlainText(),
            same_location=same,
            scene_id=scene_id,
            camera_preset=camera_preset,
        )
        # Same-as-previous inherits scene + location from the prior panel
        if index < len(self._panel_same_location) and index > 0:
            self._panel_location_edits[index].setEnabled(not same)
            if index < len(self._panel_scene_combos):
                self._panel_scene_combos[index].setEnabled(not same)

    def _sync_panel_cast(self, index: int):
        if index >= len(self._panel_cast_boxes):
            return
        ids = []
        for box, ch in zip(self._panel_cast_boxes[index], self._model.comic.characters):
            if box.isChecked():
                ids.append(ch.id)
        self._model.comic.update_panel_cast(index, ids)

    def _sync_all_to_model(self):
        self._sync_characters_to_model()
        self._sync_scenes_to_model()
        self._sync_page_loras_to_model()
        for i in range(len(self._panel_location_edits)):
            self._sync_panel_fields(i)
            self._sync_panel_cast(i)
        self._model.comic.title = self.title_edit.text()
        self._model.comic.synopsis = self.synopsis_edit.toPlainText()

    def draft(self):
        self._sync_all_to_model()
        self._model.comic.draft_page()

    def generate(self):
        self._sync_all_to_model()
        self._model.comic.generate_page()

    def generate_remaining(self):
        self._sync_all_to_model()
        self._model.comic.generate_remaining()

    def assemble(self):
        self._sync_all_to_model()
        self._model.comic.assemble_page()

    def refine(self):
        self._sync_all_to_model()
        self._model.comic.refine_page()

    def apply_result(self, item: QListWidgetItem):
        job_id, index = self.history.item_info(item)
        self._model.apply_generated_result(job_id, index)
