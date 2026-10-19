"""MiniMax H3 Reference→Video workspace UI."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QMetaObject, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..files import FileFilter
from ..localization import translate as _
from ..model.minimax_h3 import (
    ASPECT_PRESETS,
    DURATION_PRESETS,
    MAX_H3_LORAS,
    MAX_REF_IMAGES,
    MEGAPIXEL_PRESETS,
    H3Lora,
    MiniMaxH3Workspace,
    length_from_duration,
    size_from_megapixels,
)
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind
from ..model.root import root
from . import theme
from .generation import ProgressBar
from .widget import WorkspaceSelectWidget


class H3LoraRow(QWidget):
    """One H3 LoRA picker row (name + strength)."""

    def __init__(self, index: int, lora: H3Lora, parent=None):
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
        self._combo.setMinimumWidth(100)
        self._combo.setToolTip(
            _("MiniMax H3 LoRA from ComfyUI/models/loras (incl. MINIMAX_H3/ subfolder)")
        )
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


class MiniMaxH3Widget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._lora_rows: list[H3LoraRow] = []
        self._rebuilding_loras = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 2, 4, 0)

        header = QHBoxLayout()
        self.workspace_select = WorkspaceSelectWidget(self)
        header.addWidget(self.workspace_select)
        header.addStretch(1)
        root_layout.addLayout(header)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(body)
        root_layout.addWidget(scroll, 1)

        tip = QLabel(
            _(
                "MiniMax H3 Reference→Video — multi-ref stills + prompt → clip. "
                "Turbo = 5-step pack; add your own LoRAs below. "
                "Generated soundtrack is saved as MP3 (Timeline is frames-only in Krita)."
            ),
            body,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(tip)

        presets = QHBoxLayout()
        self.preset_turbo = QPushButton(_("Turbo (5-step)"), body)
        self.preset_turbo.setToolTip(_("Match extreme-fast: 5 steps, CFG 1, turbo LoRAs, 0.4 MP"))
        self.preset_quality = QPushButton(_("Quality"), body)
        self.preset_quality.setToolTip(_("20 steps, ~1 MP, no turbo LoRAs"))
        presets.addWidget(self.preset_turbo)
        presets.addWidget(self.preset_quality)
        layout.addLayout(presets)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self.aspect_combo = QComboBox(body)
        for label, aw, ah in ASPECT_PRESETS:
            self.aspect_combo.addItem(label, (aw, ah))
        form.addRow(_("Aspect"), self.aspect_combo)

        self.mp_combo = QComboBox(body)
        for label, mp in MEGAPIXEL_PRESETS:
            self.mp_combo.addItem(label, mp)
        form.addRow(_("Size"), self.mp_combo)

        self.duration_combo = QComboBox(body)
        for label, sec in DURATION_PRESETS:
            self.duration_combo.addItem(label, sec)
        form.addRow(_("Duration"), self.duration_combo)

        self.size_label = QLabel("", body)
        self.size_label.setStyleSheet(f"color: {theme.grey}; font-size: 11px;")
        form.addRow("", self.size_label)

        self.turbo_check = QCheckBox(_("Turbo LoRAs (5-step pack)"), body)
        self.turbo_check.setToolTip(
            _(
                "Loads minimax_h3 turbo LoRAs when present under models/loras "
                "(or Minimax H3/ subfolder). Stacked before your LoRAs below."
            )
        )
        form.addRow("", self.turbo_check)

        self.steps_spin = QSpinBox(body)
        self.steps_spin.setRange(1, 64)
        form.addRow(_("Steps"), self.steps_spin)

        self.cfg_spin = QDoubleSpinBox(body)
        self.cfg_spin.setRange(0.0, 15.0)
        self.cfg_spin.setSingleStep(0.1)
        form.addRow(_("CFG"), self.cfg_spin)

        layout.addLayout(form)

        lora_header = QHBoxLayout()
        lora_header.addWidget(QLabel(_("LoRAs"), body))
        lora_header.addStretch(1)
        self.add_lora_button = QPushButton(_("Add LoRA"), body)
        self.add_lora_button.setFixedHeight(22)
        self.add_lora_button.setToolTip(
            _("H3 / style LoRAs from ComfyUI/models/loras (e.g. MINIMAX_H3/)")
        )
        lora_header.addWidget(self.add_lora_button)
        layout.addLayout(lora_header)

        self.loras_container = QWidget(body)
        self.loras_layout = QVBoxLayout(self.loras_container)
        self.loras_layout.setContentsMargins(0, 0, 0, 0)
        self.loras_layout.setSpacing(2)
        layout.addWidget(self.loras_container)

        layout.addWidget(QLabel(_("Reference layers"), body))
        refs_hint = QLabel(
            _("Picture 1…6 in the prompt map to these rows. Empty rows are skipped."),
            body,
        )
        refs_hint.setWordWrap(True)
        refs_hint.setStyleSheet(f"color: {theme.grey}; font-size: 11px;")
        layout.addWidget(refs_hint)

        self.ref_combos: list[QComboBox] = []
        for i in range(MAX_REF_IMAGES):
            combo = QComboBox(body)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            self.ref_combos.append(combo)
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{i + 1}.", body))
            row.addWidget(combo, 1)
            layout.addLayout(row)

        layout.addWidget(QLabel(_("Prompt"), body))
        self.prompt_edit = QPlainTextEdit(body)
        self.prompt_edit.setPlaceholderText(
            _(
                "Scene / cuts… Optional tags: <Picture 1>, <Picture 2>, <Audio 1>. "
                "If omitted, tags are added automatically from the refs above."
            )
        )
        self.prompt_edit.setMinimumHeight(100)
        layout.addWidget(self.prompt_edit)

        self.audio_path_label = QLabel(_("No audio selected"), body)
        self.audio_path_label.setWordWrap(True)
        self.audio_path_label.setStyleSheet(f"color: {theme.grey}; font-size: 11px;")
        self.audio_browse_button = QPushButton(_("Find audio…"), body)
        self.audio_browse_button.setToolTip(
            _("Pick a reference audio file. Uploaded to ComfyUI on Generate.")
        )
        self.audio_clear_button = QPushButton(_("Clear"), body)
        audio_row = QHBoxLayout()
        audio_row.addWidget(self.audio_browse_button)
        audio_row.addWidget(self.audio_clear_button)
        audio_row.addStretch(1)
        layout.addLayout(audio_row)
        layout.addWidget(self.audio_path_label)

        actions = QHBoxLayout()
        self.generate_button = QPushButton(_("Generate Clip"), body)
        self.generate_button.setMinimumHeight(32)
        self.apply_button = QPushButton(_("Apply to Timeline"), body)
        self.apply_button.setMinimumHeight(32)
        self.open_soundtrack_button = QPushButton(_("Open soundtrack…"), body)
        self.open_soundtrack_button.setMinimumHeight(32)
        self.open_soundtrack_button.setToolTip(
            _(
                "Open the folder with the last generated H3 MP3. "
                "Krita Timeline / Render Animation does not include this audio."
            )
        )
        self.export_mp4_button = QPushButton(_("Export MP4…"), body)
        self.export_mp4_button.setMinimumHeight(32)
        self.export_mp4_button.setToolTip(
            _("Merge last H3 frames + soundtrack into one MP4 file.")
        )
        actions.addWidget(self.generate_button, 1)
        actions.addWidget(self.apply_button, 1)
        layout.addLayout(actions)
        actions2 = QHBoxLayout()
        actions2.addWidget(self.open_soundtrack_button, 1)
        actions2.addWidget(self.export_mp4_button, 1)
        layout.addLayout(actions2)

        self.progress_bar = ProgressBar(body)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("", body)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.preset_turbo.clicked.connect(lambda: self._h3().apply_preset_turbo())
        self.preset_quality.clicked.connect(lambda: self._h3().apply_preset_quality())
        self.generate_button.clicked.connect(self._generate)
        self.apply_button.clicked.connect(lambda: self._h3().apply_last_clip())
        self.audio_browse_button.clicked.connect(self._browse_audio)
        self.audio_clear_button.clicked.connect(self._clear_audio)
        self.open_soundtrack_button.clicked.connect(lambda: self._h3().open_last_audio())
        self.export_mp4_button.clicked.connect(lambda: self._h3().export_mp4())
        self.add_lora_button.clicked.connect(self._add_lora)
        self.aspect_combo.currentIndexChanged.connect(self._sync_from_ui)
        self.mp_combo.currentIndexChanged.connect(self._sync_from_ui)
        self.duration_combo.currentIndexChanged.connect(self._sync_from_ui)
        self.turbo_check.toggled.connect(self._sync_from_ui)
        self.steps_spin.valueChanged.connect(self._sync_from_ui)
        self.cfg_spin.valueChanged.connect(self._sync_from_ui)
        self.prompt_edit.textChanged.connect(self._sync_prompt)
        for combo in self.ref_combos:
            combo.currentIndexChanged.connect(self._sync_refs)

    def _h3(self) -> MiniMaxH3Workspace:
        return self._model.minimax_h3

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            h3 = model.minimax_h3
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                h3.status_changed.connect(self._update_status),
                h3.prompt_changed.connect(self._update_from_model),
                h3.megapixels_changed.connect(self._update_from_model),
                h3.duration_sec_changed.connect(self._update_from_model),
                h3.turbo_changed.connect(self._update_from_model),
                h3.steps_changed.connect(self._update_from_model),
                h3.cfg_changed.connect(self._update_from_model),
                h3.audio_path_changed.connect(self._update_audio_label),
                h3.loras_changed.connect(self._rebuild_loras),
                model.layers.changed.connect(self._rebuild_layer_combos),
                model.progress_changed.connect(self._update_progress),
            ]
            self.progress_bar.model = model
            self._rebuild_layer_combos()
            self._rebuild_loras()
            self._update_from_model()
            self._update_status()
            self._update_progress()

    def _rebuild_layer_combos(self):
        h3 = self._h3()
        current = [
            h3.ref_layer_0,
            h3.ref_layer_1,
            h3.ref_layer_2,
            h3.ref_layer_3,
            h3.ref_layer_4,
            h3.ref_layer_5,
        ]
        for combo, cur in zip(self.ref_combos, current):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_("— none —"), "")
            for layer in self._model.layers.images:
                combo.addItem(layer.name, layer.id_string)
            idx = combo.findData(cur)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _add_lora(self):
        self._sync_loras_to_model()
        self._h3().add_lora()

    def _remove_lora(self, index: int):
        self._sync_loras_to_model()
        self._h3().remove_lora(index)

    def _rebuild_loras(self):
        self._rebuilding_loras = True
        while self.loras_layout.count():
            item = self.loras_layout.takeAt(0)
            if w := item.widget():
                w.deleteLater()
        self._lora_rows.clear()
        for i, pl in enumerate(self._h3().loras):
            row = H3LoraRow(i, pl, self.loras_container)
            row._clear.clicked.connect(lambda _=False, idx=i: self._remove_lora(idx))
            row._combo.currentIndexChanged.connect(self._sync_loras_to_model)
            row._combo.editTextChanged.connect(lambda _t: self._sync_loras_to_model())
            row._strength.valueChanged.connect(lambda _v: self._sync_loras_to_model())
            self.loras_layout.addWidget(row)
            self._lora_rows.append(row)
        self.add_lora_button.setEnabled(len(self._h3().loras) < MAX_H3_LORAS)
        self._rebuilding_loras = False

    def _sync_loras_to_model(self):
        if self._rebuilding_loras:
            return
        for row in self._lora_rows:
            self._h3().update_lora(row.index, row.lora_name(), row.lora_strength())

    def _update_from_model(self, *_args):
        h3 = self._h3()
        if self.prompt_edit.toPlainText() != h3.prompt:
            self.prompt_edit.blockSignals(True)
            self.prompt_edit.setPlainText(h3.prompt)
            self.prompt_edit.blockSignals(False)

        self.turbo_check.blockSignals(True)
        self.turbo_check.setChecked(h3.turbo)
        self.turbo_check.blockSignals(False)
        self.steps_spin.blockSignals(True)
        self.steps_spin.setValue(h3.steps)
        self.steps_spin.blockSignals(False)
        self.cfg_spin.blockSignals(True)
        self.cfg_spin.setValue(h3.cfg)
        self.cfg_spin.blockSignals(False)

        mp_idx = self.mp_combo.findData(h3.megapixels)
        if mp_idx < 0:
            best_i, best_d = 0, 1e9
            for i in range(self.mp_combo.count()):
                d = abs(float(self.mp_combo.itemData(i)) - h3.megapixels)
                if d < best_d:
                    best_i, best_d = i, d
            mp_idx = best_i
        self.mp_combo.blockSignals(True)
        self.mp_combo.setCurrentIndex(mp_idx)
        self.mp_combo.blockSignals(False)

        aspect_idx = self.aspect_combo.findData((h3.aspect_w, h3.aspect_h))
        if aspect_idx < 0:
            aspect_idx = 0
        self.aspect_combo.blockSignals(True)
        self.aspect_combo.setCurrentIndex(aspect_idx)
        self.aspect_combo.blockSignals(False)

        dur_idx = self.duration_combo.findData(h3.duration_sec)
        if dur_idx < 0:
            best_i, best_d = 0, 1e9
            for i in range(self.duration_combo.count()):
                d = abs(float(self.duration_combo.itemData(i)) - h3.duration_sec)
                if d < best_d:
                    best_i, best_d = i, d
            dur_idx = best_i
        self.duration_combo.blockSignals(True)
        self.duration_combo.setCurrentIndex(dur_idx)
        self.duration_combo.blockSignals(False)

        self._update_size_label()
        self._update_audio_label()

    def _update_audio_label(self, *_args):
        path = (self._h3().audio_path or "").strip()
        if not path:
            self.audio_path_label.setText(_("No reference audio selected"))
            self.audio_clear_button.setEnabled(False)
            return
        name = Path(path).name
        exists = Path(path).is_file()
        if exists:
            self.audio_path_label.setText(name + f"\n{path}")
        else:
            self.audio_path_label.setText(_("Missing") + f": {name}\n{path}")
        self.audio_clear_button.setEnabled(True)

    def _browse_audio(self):
        start = self._h3().audio_path or str(Path.home() / "Desktop")
        if start and Path(start).is_file():
            start = str(Path(start).parent)
        elif start and not Path(start).is_dir():
            start = str(Path.home())
        filename, _filter = QFileDialog.getOpenFileName(
            self,
            _("Select reference audio"),
            start,
            _("Audio files")
            + " (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma);;"
            + _("All files")
            + " (*)",
        )
        if filename:
            self._h3().audio_path = filename

    def _clear_audio(self):
        self._h3().audio_path = ""

    def _update_size_label(self):
        h3 = self._h3()
        w, h = size_from_megapixels(h3.megapixels, h3.aspect_w, h3.aspect_h)
        n = length_from_duration(h3.duration_sec, h3.fps)
        self.size_label.setText(
            _("{w}×{h} · {n} frames @ {fps} fps").format(w=w, h=h, n=n, fps=h3.fps)
        )

    def _sync_prompt(self):
        self._h3().prompt = self.prompt_edit.toPlainText()

    def _sync_from_ui(self, *_args):
        h3 = self._h3()
        aspect = self.aspect_combo.currentData()
        if aspect:
            h3.aspect_w, h3.aspect_h = aspect
        mp = self.mp_combo.currentData()
        if mp is not None:
            h3.megapixels = float(mp)
        dur = self.duration_combo.currentData()
        if dur is not None:
            h3.duration_sec = float(dur)
        h3.turbo = self.turbo_check.isChecked()
        h3.steps = int(self.steps_spin.value())
        h3.cfg = float(self.cfg_spin.value())
        self._update_size_label()

    def _sync_refs(self, *_args):
        h3 = self._h3()
        values = [c.currentData() or "" for c in self.ref_combos]
        h3.ref_layer_0 = values[0]
        h3.ref_layer_1 = values[1]
        h3.ref_layer_2 = values[2]
        h3.ref_layer_3 = values[3]
        h3.ref_layer_4 = values[4]
        h3.ref_layer_5 = values[5]

    def _generate(self):
        self._sync_from_ui()
        self._sync_refs()
        self._sync_prompt()
        self._sync_loras_to_model()
        self._h3().generate()

    def _update_status(self, *_args):
        text = self._h3().status or self._h3().idle_hint()
        self.status_label.setText(text)

    def _update_progress(self, *_args):
        pass  # ProgressBar binds via model
