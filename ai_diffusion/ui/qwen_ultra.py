"""Qwen Image Edit 2.1 Ultra workspace UI."""

from __future__ import annotations

from PyQt5.QtCore import QMetaObject, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
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

from ..backend.api import QwenUltraMode
from ..files import FileFilter
from ..localization import translate as _
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind
from ..model.qwen_ultra import (
    MAX_QWEN_LORAS,
    MAX_REF_IMAGES,
    MEGAPIXEL_PRESETS,
    MODE_LABELS,
    SAMPLER_PRESETS,
    T2I_SIZE_PRESETS,
    QwenUltraLora,
    QwenUltraWorkspace,
)
from ..model.root import root
from . import theme
from .generation import ProgressBar
from .widget import WorkspaceSelectWidget


class QwenUltraLoraRow(QWidget):
    def __init__(self, index: int, lora: QwenUltraLora, parent=None):
        super().__init__(parent)
        self.index = index
        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True

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
        self._combo.setToolTip(_("Qwen 2.1 LoRA from models/loras"))
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
        if lora.name:
            idx = self._combo.findData(lora.name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            else:
                self._combo.setEditText(lora.name)

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


class QwenUltraWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._lora_rows: list[QwenUltraLoraRow] = []
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
                "Qwen Image Edit 2.1 Ultra — Multi-Edit / T2I / Inpaint / Outpaint. "
                "Uses qwen_image_2.1 + qwen3vl + texture-fix VAE (ComfyUI ≥0.38)."
            ),
            body,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(tip)

        presets = QHBoxLayout()
        self.preset_edit = QPushButton(_("Edit preset"), body)
        self.preset_edit.setToolTip(_("25 steps · euler/simple · Multi-Edit"))
        self.preset_t2i = QPushButton(_("T2I preset"), body)
        self.preset_t2i.setToolTip(_("50 steps · res_2s/beta · Text-to-Image"))
        self.preset_outpaint = QPushButton(_("Outpaint preset"), body)
        presets.addWidget(self.preset_edit)
        presets.addWidget(self.preset_t2i)
        presets.addWidget(self.preset_outpaint)
        layout.addLayout(presets)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self.mode_combo = QComboBox(body)
        for mode, label in MODE_LABELS.items():
            self.mode_combo.addItem(_(label), mode)
        form.addRow(_("Mode"), self.mode_combo)

        self.mp_combo = QComboBox(body)
        for label, mp in MEGAPIXEL_PRESETS:
            self.mp_combo.addItem(_(label), mp)
        form.addRow(_("Edit size"), self.mp_combo)

        self.size_combo = QComboBox(body)
        for label, w, h in T2I_SIZE_PRESETS:
            self.size_combo.addItem(label, (w, h))
        form.addRow(_("T2I size"), self.size_combo)

        self.sampler_combo = QComboBox(body)
        for label, sampler, scheduler, steps, cfg in SAMPLER_PRESETS:
            self.sampler_combo.addItem(_(label), (sampler, scheduler, steps, cfg))
        form.addRow(_("Sampler pack"), self.sampler_combo)

        self.steps_spin = QSpinBox(body)
        self.steps_spin.setRange(1, 80)
        form.addRow(_("Steps"), self.steps_spin)

        self.cfg_spin = QDoubleSpinBox(body)
        self.cfg_spin.setRange(0.0, 15.0)
        self.cfg_spin.setSingleStep(0.1)
        form.addRow(_("CFG"), self.cfg_spin)

        self.denoise_spin = QDoubleSpinBox(body)
        self.denoise_spin.setRange(0.05, 1.0)
        self.denoise_spin.setSingleStep(0.05)
        form.addRow(_("Denoise"), self.denoise_spin)

        self.canvas_check = QCheckBox(_("Use canvas as <image1>"), body)
        self.canvas_check.setToolTip(
            _("When on, the current document is sent as the base edit image.")
        )
        form.addRow("", self.canvas_check)

        layout.addLayout(form)

        # Outpaint pads
        pad_row = QHBoxLayout()
        self.pad_left = QSpinBox(body)
        self.pad_top = QSpinBox(body)
        self.pad_right = QSpinBox(body)
        self.pad_bottom = QSpinBox(body)
        for spin, label in (
            (self.pad_left, "L"),
            (self.pad_top, "T"),
            (self.pad_right, "R"),
            (self.pad_bottom, "B"),
        ):
            spin.setRange(0, 2048)
            spin.setSingleStep(32)
            pad_row.addWidget(QLabel(label, body))
            pad_row.addWidget(spin)
        layout.addWidget(QLabel(_("Outpaint pad (px)"), body))
        layout.addLayout(pad_row)

        # LoRAs
        lora_header = QHBoxLayout()
        lora_header.addWidget(QLabel(_("LoRAs"), body))
        lora_header.addStretch(1)
        self.add_lora_button = QPushButton(_("Add LoRA"), body)
        self.add_lora_button.setFixedHeight(22)
        lora_header.addWidget(self.add_lora_button)
        layout.addLayout(lora_header)
        self.loras_container = QWidget(body)
        self.loras_layout = QVBoxLayout(self.loras_container)
        self.loras_layout.setContentsMargins(0, 0, 0, 0)
        self.loras_layout.setSpacing(2)
        layout.addWidget(self.loras_container)

        # Refs
        layout.addWidget(QLabel(_("Reference layers"), body))
        refs_hint = QLabel(
            _(
                "<image1>… tags map to canvas (if checked) then these rows. "
                "Inpaint uses the active selection."
            ),
            body,
        )
        refs_hint.setWordWrap(True)
        refs_hint.setStyleSheet(f"color: {theme.grey}; font-size: 11px;")
        layout.addWidget(refs_hint)

        self.ref_combos: list[QComboBox] = []
        for i in range(MAX_REF_IMAGES):
            combo = QComboBox(body)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
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
                "Edit instruction… Optional tags: <image1>, <image2>… "
                "Example: start with <image1> as the base, replace the head with <image2>."
            )
        )
        self.prompt_edit.setMinimumHeight(100)
        layout.addWidget(self.prompt_edit)

        layout.addWidget(QLabel(_("Negative"), body))
        self.negative_edit = QPlainTextEdit(body)
        self.negative_edit.setMaximumHeight(48)
        layout.addWidget(self.negative_edit)

        self.generate_button = QPushButton(_("Generate"), body)
        self.generate_button.setMinimumHeight(36)
        layout.addWidget(self.generate_button)

        self.progress_bar = ProgressBar(body)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("", body)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.preset_edit.clicked.connect(lambda: self._qwen().apply_preset_edit())
        self.preset_t2i.clicked.connect(lambda: self._qwen().apply_preset_t2i())
        self.preset_outpaint.clicked.connect(lambda: self._qwen().apply_preset_outpaint())
        self.generate_button.clicked.connect(self._generate)
        self.add_lora_button.clicked.connect(self._add_lora)
        self.mode_combo.currentIndexChanged.connect(self._sync_from_ui)
        self.mp_combo.currentIndexChanged.connect(self._sync_from_ui)
        self.size_combo.currentIndexChanged.connect(self._sync_size)
        self.sampler_combo.currentIndexChanged.connect(self._sync_sampler_pack)
        self.steps_spin.valueChanged.connect(self._sync_from_ui)
        self.cfg_spin.valueChanged.connect(self._sync_from_ui)
        self.denoise_spin.valueChanged.connect(self._sync_from_ui)
        self.canvas_check.toggled.connect(self._sync_from_ui)
        for spin in (self.pad_left, self.pad_top, self.pad_right, self.pad_bottom):
            spin.valueChanged.connect(self._sync_from_ui)
        self.prompt_edit.textChanged.connect(self._sync_prompt)
        self.negative_edit.textChanged.connect(self._sync_prompt)
        for combo in self.ref_combos:
            combo.currentIndexChanged.connect(self._sync_refs)

    def _qwen(self) -> QwenUltraWorkspace:
        return self._model.qwen_ultra

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            q = model.qwen_ultra
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                q.status_changed.connect(self._update_status),
                q.prompt_changed.connect(self._update_from_model),
                q.mode_changed.connect(self._update_from_model),
                q.steps_changed.connect(self._update_from_model),
                q.cfg_changed.connect(self._update_from_model),
                q.megapixels_changed.connect(self._update_from_model),
                q.loras_changed.connect(self._rebuild_loras),
                model.layers.changed.connect(self._rebuild_layer_combos),
            ]
            self.progress_bar.model = model
            self._rebuild_layer_combos()
            self._rebuild_loras()
            self._update_from_model()
            self._update_status()

    def _rebuild_layer_combos(self):
        q = self._qwen()
        current = [
            q.ref_layer_0,
            q.ref_layer_1,
            q.ref_layer_2,
            q.ref_layer_3,
            q.ref_layer_4,
            q.ref_layer_5,
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
        self._qwen().add_lora()

    def _remove_lora(self, index: int):
        self._sync_loras_to_model()
        self._qwen().remove_lora(index)

    def _rebuild_loras(self):
        self._rebuilding_loras = True
        while self.loras_layout.count():
            item = self.loras_layout.takeAt(0)
            if w := item.widget():
                w.deleteLater()
        self._lora_rows.clear()
        for i, pl in enumerate(self._qwen().loras):
            row = QwenUltraLoraRow(i, pl, self.loras_container)
            row._clear.clicked.connect(lambda _=False, idx=i: self._remove_lora(idx))
            row._combo.currentIndexChanged.connect(self._sync_loras_to_model)
            row._combo.editTextChanged.connect(lambda _t: self._sync_loras_to_model())
            row._strength.valueChanged.connect(lambda _v: self._sync_loras_to_model())
            self.loras_layout.addWidget(row)
            self._lora_rows.append(row)
        self.add_lora_button.setEnabled(len(self._qwen().loras) < MAX_QWEN_LORAS)
        self._rebuilding_loras = False

    def _sync_loras_to_model(self):
        if self._rebuilding_loras:
            return
        for row in self._lora_rows:
            self._qwen().update_lora(row.index, row.lora_name(), row.lora_strength())

    def _update_from_model(self, *_args):
        q = self._qwen()
        if self.prompt_edit.toPlainText() != q.prompt:
            self.prompt_edit.blockSignals(True)
            self.prompt_edit.setPlainText(q.prompt)
            self.prompt_edit.blockSignals(False)
        if self.negative_edit.toPlainText() != q.negative:
            self.negative_edit.blockSignals(True)
            self.negative_edit.setPlainText(q.negative)
            self.negative_edit.blockSignals(False)

        mode_idx = self.mode_combo.findData(q.mode)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
        self.mode_combo.blockSignals(False)

        mp_idx = self.mp_combo.findData(q.megapixels)
        if mp_idx < 0:
            mp_idx = 1
        self.mp_combo.blockSignals(True)
        self.mp_combo.setCurrentIndex(mp_idx)
        self.mp_combo.blockSignals(False)

        size_idx = self.size_combo.findData((q.width, q.height))
        self.size_combo.blockSignals(True)
        self.size_combo.setCurrentIndex(size_idx if size_idx >= 0 else 0)
        self.size_combo.blockSignals(False)

        for spin, val in (
            (self.steps_spin, q.steps),
            (self.cfg_spin, q.cfg),
            (self.denoise_spin, q.denoise),
            (self.pad_left, q.pad_left),
            (self.pad_top, q.pad_top),
            (self.pad_right, q.pad_right),
            (self.pad_bottom, q.pad_bottom),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self.canvas_check.blockSignals(True)
        self.canvas_check.setChecked(q.use_canvas_as_image1)
        self.canvas_check.blockSignals(False)
        self._update_mode_visibility()

    def _update_mode_visibility(self):
        mode = self.mode_combo.currentData()
        is_t2i = mode is QwenUltraMode.t2i
        is_out = mode is QwenUltraMode.outpaint
        self.size_combo.setEnabled(is_t2i)
        self.mp_combo.setEnabled(not is_t2i)
        for spin in (self.pad_left, self.pad_top, self.pad_right, self.pad_bottom):
            spin.setEnabled(is_out)

    def _sync_prompt(self):
        self._qwen().prompt = self.prompt_edit.toPlainText()
        self._qwen().negative = self.negative_edit.toPlainText()

    def _sync_size(self, *_args):
        data = self.size_combo.currentData()
        if data:
            self._qwen().width, self._qwen().height = data
            self._qwen().resolution = max(data)

    def _sync_sampler_pack(self, *_args):
        data = self.sampler_combo.currentData()
        if not data:
            return
        sampler, scheduler, steps, cfg = data
        q = self._qwen()
        q.sampler = sampler
        q.scheduler = scheduler
        q.steps = int(steps)
        q.cfg = float(cfg)
        self.steps_spin.blockSignals(True)
        self.steps_spin.setValue(steps)
        self.steps_spin.blockSignals(False)
        self.cfg_spin.blockSignals(True)
        self.cfg_spin.setValue(cfg)
        self.cfg_spin.blockSignals(False)

    def _sync_from_ui(self, *_args):
        q = self._qwen()
        mode = self.mode_combo.currentData()
        if mode is not None:
            q.mode = mode
        mp = self.mp_combo.currentData()
        if mp is not None:
            q.megapixels = float(mp)
        q.steps = int(self.steps_spin.value())
        q.cfg = float(self.cfg_spin.value())
        q.denoise = float(self.denoise_spin.value())
        q.use_canvas_as_image1 = self.canvas_check.isChecked()
        q.pad_left = int(self.pad_left.value())
        q.pad_top = int(self.pad_top.value())
        q.pad_right = int(self.pad_right.value())
        q.pad_bottom = int(self.pad_bottom.value())
        self._update_mode_visibility()

    def _sync_refs(self, *_args):
        q = self._qwen()
        values = [c.currentData() or "" for c in self.ref_combos]
        q.ref_layer_0 = values[0]
        q.ref_layer_1 = values[1]
        q.ref_layer_2 = values[2]
        q.ref_layer_3 = values[3]
        q.ref_layer_4 = values[4]
        q.ref_layer_5 = values[5]

    def _generate(self):
        self._sync_from_ui()
        self._sync_refs()
        self._sync_prompt()
        self._sync_loras_to_model()
        self._qwen().generate()

    def _update_status(self, *_args):
        text = self._qwen().status or self._qwen().idle_hint()
        self.status_label.setText(text)
