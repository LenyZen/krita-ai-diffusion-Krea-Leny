"""Wan 2.2 Animate workspace UI — T2V, I2V, and start/end video from Krita."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QMetaObject, QSize, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..files import FileFilter
from ..localization import translate as _
from ..model.animate import (
    LENGTH_PRESETS,
    MAX_ANIMATE_LORAS,
    RESOLUTION_PRESETS,
    SAMPLING_PRESETS,
    AnimateLora,
    AnimateMode,
    AnimateWorkspace,
)
from ..model.model import DocumentModel, Workspace
from ..model.properties import Bind, Binding, bind
from ..model.root import root
from . import theme
from .generation import ProgressBar
from .widget import StyleSelectWidget, WorkspaceSelectWidget


class AnimateLoraRow(QWidget):
    """One Wan video LoRA picker row (applied to high + low UNETs)."""

    def __init__(self, index: int, lora: AnimateLora, parent=None):
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
        self._combo.setToolTip(_("Wan / video LoRA applied to both high and low noise models"))
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


class AnimateWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._lora_rows: list[AnimateLoraRow] = []
        self._rebuilding_loras = False
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        layout.setSpacing(6)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.style_select = StyleSelectWidget(self)
        header = QHBoxLayout()
        header.addWidget(self.workspace_select)
        header.addWidget(self.style_select, 1)
        layout.addLayout(header)

        tip = QLabel(
            _(
                "Wan 2.2 short clips (~5s max). T2V needs wan2.2_t2v_* models; I2V/FLF2V need "
                "wan2.2_i2v_*. On 12GB start at 640×640 / 33 frames. Apply imports into the Timeline."
            ),
            self,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(tip)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem(_("Text → Video (T2V)"), AnimateMode.t2v)
        self.mode_combo.addItem(_("Image → Video (I2V)"), AnimateMode.i2v)
        self.mode_combo.addItem(_("Start / End frames (FLF2V)"), AnimateMode.flf2v)
        form.addRow(_("Mode"), self.mode_combo)

        self._start_label = QLabel(_("Start layer"), self)
        self.start_combo = QComboBox(self)
        self.start_combo.setToolTip(_("Paint layer used as the first frame"))
        form.addRow(self._start_label, self.start_combo)

        self._end_label = QLabel(_("End layer"), self)
        self.end_combo = QComboBox(self)
        self.end_combo.setToolTip(_("Paint layer used as the last frame (FLF2V / Stitch)"))
        form.addRow(self._end_label, self.end_combo)

        self._comic_start_label = QLabel(_("Comic start"), self)
        self.comic_start_combo = QComboBox(self)
        self.comic_start_combo.setToolTip(
            _("Optional: use a locked Comic panel thumbnail as the start frame")
        )
        form.addRow(self._comic_start_label, self.comic_start_combo)

        self._comic_end_label = QLabel(_("Comic end"), self)
        self.comic_end_combo = QComboBox(self)
        self.comic_end_combo.setToolTip(
            _("Optional: use a locked Comic panel thumbnail as the end frame")
        )
        form.addRow(self._comic_end_label, self.comic_end_combo)

        self.res_combo = QComboBox(self)
        for label, w, h in RESOLUTION_PRESETS:
            self.res_combo.addItem(_(label), (w, h))
        form.addRow(_("Resolution"), self.res_combo)

        self.length_combo = QComboBox(self)
        for label, frames in LENGTH_PRESETS:
            self.length_combo.addItem(_(label), frames)
        form.addRow(_("Length"), self.length_combo)

        layout.addLayout(form)

        layout.addWidget(QLabel(_("Motion prompt"), self))
        self.prompt_edit = QPlainTextEdit(self)
        self.prompt_edit.setPlaceholderText(
            _("Scene, camera move, character action, lighting…")
        )
        self.prompt_edit.setMinimumHeight(72)
        layout.addWidget(self.prompt_edit)

        layout.addWidget(QLabel(_("Negative (optional)"), self))
        self.negative_edit = QLineEdit(self)
        self.negative_edit.setPlaceholderText(_("motion blur, morphing, flicker…"))
        layout.addWidget(self.negative_edit)

        adv = QFormLayout()
        self.high_combo = QComboBox(self)
        self.low_combo = QComboBox(self)
        self.te_combo = QComboBox(self)
        self.vae_combo = QComboBox(self)
        adv.addRow(_("High noise"), self.high_combo)
        adv.addRow(_("Low noise"), self.low_combo)
        adv.addRow(_("Text encoder"), self.te_combo)
        adv.addRow(_("VAE"), self.vae_combo)

        self.preset_combo = QComboBox(self)
        self.preset_combo.addItem(_("— sampling presets —"), "")
        for label, _settings in SAMPLING_PRESETS:
            self.preset_combo.addItem(_(label), label)
        self.preset_combo.setToolTip(
            _(
                "Quick setups for Wan. Distilled lightx2v I2V → lcm / 4-step. "
                "T2V with lightx2v LoRAs → euler / simple / shift 5. "
                "You can still tweak sampler, scheduler, steps, and shift after."
            )
        )
        adv.addRow(_("Preset"), self.preset_combo)

        self.sampler_combo = QComboBox(self)
        self.sampler_combo.setToolTip(
            _("KSampler name. euler is best for official T2V + lightx2v; lcm for distilled I2V.")
        )
        self.scheduler_combo = QComboBox(self)
        self.scheduler_combo.setToolTip(
            _("Noise schedule. simple pairs with euler/T2V; sgm_uniform with distilled lcm I2V.")
        )
        adv.addRow(_("Sampler"), self.sampler_combo)
        adv.addRow(_("Scheduler"), self.scheduler_combo)

        self.steps_spin = QSpinBox(self)
        self.steps_spin.setRange(2, 40)
        self.steps_spin.setToolTip(_("Total denoising steps (high+low experts share this count)"))
        self.split_spin = QSpinBox(self)
        self.split_spin.setRange(1, 39)
        self.split_spin.setToolTip(
            _("Step where sampling switches from high-noise to low-noise UNET (usually steps/2)")
        )
        self.cfg_spin = QDoubleSpinBox(self)
        self.cfg_spin.setRange(0.0, 10.0)
        self.cfg_spin.setSingleStep(0.1)
        self.cfg_spin.setToolTip(_("Guidance. Keep ~1.0 for distilled / lightx2v 4-step runs."))
        self.shift_spin = QDoubleSpinBox(self)
        self.shift_spin.setRange(1.0, 16.0)
        self.shift_spin.setSingleStep(0.5)
        self.shift_spin.setToolTip(
            _("ModelSamplingSD3 shift. ~8 for distilled I2V; ~5 for official T2V / lightx2v.")
        )
        adv.addRow(_("Steps"), self.steps_spin)
        adv.addRow(_("Split step"), self.split_spin)
        adv.addRow(_("CFG"), self.cfg_spin)
        adv.addRow(_("Shift"), self.shift_spin)
        layout.addLayout(adv)

        lora_header = QHBoxLayout()
        lora_header.addWidget(QLabel(_("Video LoRAs"), self))
        lora_header.addStretch(1)
        self.add_lora_button = QPushButton(_("Add LoRA"), self)
        self.add_lora_button.setFixedHeight(22)
        self.add_lora_button.setToolTip(
            _(
                "Wan video LoRAs from ComfyUI/models/loras — applied to both high and low UNETs. "
                "Use LoRAs trained for Wan 2.2 T2V / I2V. Optional lightx2v T2V LoRAs help 4-step T2V."
            )
        )
        lora_header.addWidget(self.add_lora_button)
        layout.addLayout(lora_header)

        self.loras_container = QWidget(self)
        self.loras_layout = QVBoxLayout(self.loras_container)
        self.loras_layout.setContentsMargins(0, 0, 0, 0)
        self.loras_layout.setSpacing(2)
        layout.addWidget(self.loras_container)

        btns = QHBoxLayout()
        self.generate_button = QPushButton(_("Generate Animation"), self)
        self.generate_button.setToolTip(_("Run Wan 2.2 and collect frames for the Timeline"))
        self.apply_button = QPushButton(_("Apply to Timeline"), self)
        btns.addWidget(self.generate_button, 1)
        btns.addWidget(self.apply_button)
        layout.addLayout(btns)

        btns2 = QHBoxLayout()
        self.extend_button = QPushButton(_("Extend from last"), self)
        self.extend_button.setToolTip(
            _(
                "After a clip: I2V from its last frame using the current motion prompt and length, "
                "then append those frames (continue the shot — no End frame needed)."
            )
        )
        self.stitch_button = QPushButton(_("Stitch"), self)
        self.stitch_button.setToolTip(
            _(
                "After a clip: FLF2V from its last frame to your End layer/panel, "
                "then append those frames (join toward a target end pose)."
            )
        )
        btns2.addWidget(self.extend_button, 1)
        btns2.addWidget(self.stitch_button)
        layout.addLayout(btns2)

        self.progress_bar = ProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_ui)
        self.generate_button.clicked.connect(self._generate)
        self.apply_button.clicked.connect(self._apply)
        self.extend_button.clicked.connect(self._extend)
        self.stitch_button.clicked.connect(self._stitch)
        self.add_lora_button.clicked.connect(self._add_lora)
        self.res_combo.currentIndexChanged.connect(self._on_res)
        self.length_combo.currentIndexChanged.connect(self._on_length)
        self.start_combo.currentIndexChanged.connect(self._sync_layers)
        self.end_combo.currentIndexChanged.connect(self._sync_layers)
        self.comic_start_combo.currentIndexChanged.connect(self._sync_comic)
        self.comic_end_combo.currentIndexChanged.connect(self._sync_comic)
        self.high_combo.currentIndexChanged.connect(self._sync_models)
        self.low_combo.currentIndexChanged.connect(self._sync_models)
        self.te_combo.currentIndexChanged.connect(self._sync_models)
        self.vae_combo.currentIndexChanged.connect(self._sync_models)
        self.prompt_edit.textChanged.connect(self._sync_prompt)
        self.negative_edit.textChanged.connect(self._sync_prompt)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        self.sampler_combo.currentIndexChanged.connect(self._sync_sampling)
        self.scheduler_combo.currentIndexChanged.connect(self._sync_sampling)
        self.steps_spin.valueChanged.connect(self._sync_sampling)
        self.split_spin.valueChanged.connect(self._sync_sampling)
        self.cfg_spin.valueChanged.connect(self._sync_sampling)
        self.shift_spin.valueChanged.connect(self._sync_sampling)

        self._on_mode_ui()
        for combo in (
            self.mode_combo,
            self.start_combo,
            self.end_combo,
            self.comic_start_combo,
            self.comic_end_combo,
            self.res_combo,
            self.length_combo,
            self.high_combo,
            self.low_combo,
            self.te_combo,
            self.vae_combo,
            self.preset_combo,
            self.sampler_combo,
            self.scheduler_combo,
        ):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def minimumSizeHint(self) -> QSize:
        return QSize(180, super().minimumSizeHint().height())

    @staticmethod
    def _short_model_label(name: str) -> str:
        base = Path(name.replace("\\", "/")).name
        if len(base) <= 36:
            return base
        return base[:16] + "…" + base[-16:]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            a = model.animate
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "style", self.style_select, "value", Bind.two_way),
                a.status_changed.connect(self._refresh_status),
                model.progress_changed.connect(self._refresh_status),
                model.layers.changed.connect(self._rebuild_layer_combos),
                model.comic.modified.connect(lambda *_: self._rebuild_comic_combos()),
                a.loras_changed.connect(self._rebuild_loras),
            ]
            self.progress_bar.model = model
            self._sync_from_model()
            self._rebuild_layer_combos()
            self._rebuild_comic_combos()
            self._rebuild_model_combos()
            self._rebuild_sampling_combos()
            self._rebuild_loras()
            self._refresh_status()

    def _animate(self) -> AnimateWorkspace:
        return self._model.animate

    def _refresh_status(self, *_args):
        """Status line + live percent while Generate / Extend / Stitch is running."""
        a = self._animate()
        base = a.status or a.idle_hint()
        p = self._model.progress
        if base and 0 < p < 1:
            self.status_label.setText(f"{base}  {int(p * 100)}%")
        else:
            self.status_label.setText(base)

    def _sync_from_model(self):
        a = self._animate()
        idx = self.mode_combo.findData(a.mode)
        if idx >= 0:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)
        self.prompt_edit.blockSignals(True)
        self.prompt_edit.setPlainText(a.prompt)
        self.prompt_edit.blockSignals(False)
        self.negative_edit.blockSignals(True)
        self.negative_edit.setText(a.negative_prompt)
        self.negative_edit.blockSignals(False)
        for i in range(self.res_combo.count()):
            w, h = self.res_combo.itemData(i)
            if w == a.width and h == a.height:
                self.res_combo.blockSignals(True)
                self.res_combo.setCurrentIndex(i)
                self.res_combo.blockSignals(False)
                break
        for i in range(self.length_combo.count()):
            if self.length_combo.itemData(i) == a.length:
                self.length_combo.blockSignals(True)
                self.length_combo.setCurrentIndex(i)
                self.length_combo.blockSignals(False)
                break
        self._rebuild_sampling_combos()
        self.steps_spin.blockSignals(True)
        self.steps_spin.setValue(a.steps)
        self.steps_spin.blockSignals(False)
        self.split_spin.blockSignals(True)
        self.split_spin.setValue(a.split_step)
        self.split_spin.blockSignals(False)
        self.cfg_spin.blockSignals(True)
        self.cfg_spin.setValue(a.cfg)
        self.cfg_spin.blockSignals(False)
        self.shift_spin.blockSignals(True)
        self.shift_spin.setValue(a.shift)
        self.shift_spin.blockSignals(False)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self._on_mode_ui()
        self._refresh_status()

    def _on_mode_ui(self):
        data = self.mode_combo.currentData()
        if data is not None:
            self._animate().mode = data
        is_t2v = data is AnimateMode.t2v
        # T2V is prompt-only; start/end still matter for Stitch after a T2V clip
        for w in (
            self._start_label,
            self.start_combo,
            self._comic_start_label,
            self.comic_start_combo,
        ):
            w.setVisible(not is_t2v)
        # End layer stays available for Stitch even in T2V / I2V
        self.end_combo.setEnabled(True)
        self.comic_end_combo.setEnabled(True)
        self._rebuild_model_combos()
        # Mode-specific idle hint when no run status yet
        self._refresh_status()

    def _rebuild_layer_combos(self):
        a = self._animate()
        for combo, current in (
            (self.start_combo, a.start_layer_id),
            (self.end_combo, a.end_layer_id),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_("— canvas / none —"), "")
            for layer in self._model.layers.images:
                combo.addItem(layer.name, layer.id_string)
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _rebuild_comic_combos(self):
        a = self._animate()
        comic = self._model.comic
        for combo, current in (
            (self.comic_start_combo, a.comic_start_panel),
            (self.comic_end_combo, a.comic_end_panel),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_("— none —"), -1)
            for i, panel in enumerate(comic.panels):
                has = comic.panel_image(i) is not None
                label = _("Panel {n}").format(n=i + 1)
                if has:
                    label += " ★"
                elif panel.locked:
                    label += " (" + _("locked empty") + ")"
                combo.addItem(label, i)
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _rebuild_model_combos(self):
        a = self._animate()
        client = self._model._connection.client_if_connected
        if client is None:
            return
        high, low, te, vae = a.resolve_models(client)

        def fill(combo: QComboBox, items: list[str], current: str):
            combo.blockSignals(True)
            combo.clear()
            for name in items:
                combo.addItem(self._short_model_label(name), name)
                combo.setItemData(combo.count() - 1, name, Qt.ItemDataRole.ToolTipRole)
            if current:
                idx = combo.findData(current)
                if idx < 0:
                    combo.addItem(self._short_model_label(current), current)
                    combo.setItemData(combo.count() - 1, current, Qt.ItemDataRole.ToolTipRole)
                    idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        fill(self.high_combo, a.list_unet_models() or ([high] if high else []), high)
        fill(self.low_combo, a.list_unet_models() or ([low] if low else []), low)
        fill(self.te_combo, a.list_text_encoders(), te)
        fill(self.vae_combo, a.list_vaes(), vae)

    def _rebuild_sampling_combos(self):
        a = self._animate()

        def fill(combo: QComboBox, items: list[str], current: str):
            combo.blockSignals(True)
            combo.clear()
            for name in items:
                combo.addItem(name, name)
            if current:
                idx = combo.findData(current)
                if idx < 0:
                    combo.addItem(current, current)
                    idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        fill(self.sampler_combo, a.list_samplers(), a.sampler or "euler")
        fill(self.scheduler_combo, a.list_schedulers(), a.scheduler or "simple")

    def _on_preset(self):
        key = self.preset_combo.currentData()
        if not key:
            return
        a = self._animate()
        if a.apply_sampling_preset(str(key)):
            self._rebuild_sampling_combos()
            self.steps_spin.blockSignals(True)
            self.steps_spin.setValue(a.steps)
            self.steps_spin.blockSignals(False)
            self.split_spin.blockSignals(True)
            self.split_spin.setValue(a.split_step)
            self.split_spin.blockSignals(False)
            self.cfg_spin.blockSignals(True)
            self.cfg_spin.setValue(a.cfg)
            self.cfg_spin.blockSignals(False)
            self.shift_spin.blockSignals(True)
            self.shift_spin.setValue(a.shift)
            self.shift_spin.blockSignals(False)
            # Reset preset combo to placeholder so re-selecting same entry works
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(0)
            self.preset_combo.blockSignals(False)
            self._refresh_status()

    def _sync_layers(self):
        a = self._animate()
        a.start_layer_id = self.start_combo.currentData() or ""
        a.end_layer_id = self.end_combo.currentData() or ""

    def _sync_comic(self):
        a = self._animate()
        a.comic_start_panel = (
            self.comic_start_combo.currentData()
            if self.comic_start_combo.currentData() is not None
            else -1
        )
        a.comic_end_panel = (
            self.comic_end_combo.currentData()
            if self.comic_end_combo.currentData() is not None
            else -1
        )

    def _sync_models(self):
        a = self._animate()
        a.high_noise_model = self.high_combo.currentData() or ""
        a.low_noise_model = self.low_combo.currentData() or ""
        a.text_encoder = self.te_combo.currentData() or ""
        a.vae = self.vae_combo.currentData() or ""

    def _sync_prompt(self):
        a = self._animate()
        a.prompt = self.prompt_edit.toPlainText()
        a.negative_prompt = self.negative_edit.text()

    def _sync_sampling(self):
        a = self._animate()
        a.sampler = self.sampler_combo.currentData() or self.sampler_combo.currentText() or "euler"
        a.scheduler = (
            self.scheduler_combo.currentData() or self.scheduler_combo.currentText() or "simple"
        )
        a.steps = self.steps_spin.value()
        a.split_step = self.split_spin.value()
        a.cfg = self.cfg_spin.value()
        a.shift = self.shift_spin.value()
        # Keep split within steps-1
        if a.split_step >= a.steps:
            a.split_step = max(1, a.steps - 1)
            self.split_spin.blockSignals(True)
            self.split_spin.setValue(a.split_step)
            self.split_spin.blockSignals(False)

    def _on_res(self):
        data = self.res_combo.currentData()
        if data:
            self._animate().width, self._animate().height = data

    def _on_length(self):
        data = self.length_combo.currentData()
        if data:
            self._animate().length = int(data)

    def _add_lora(self):
        self._sync_loras_to_model()
        self._animate().add_lora()

    def _remove_lora(self, index: int):
        self._sync_loras_to_model()
        self._animate().remove_lora(index)

    def _rebuild_loras(self):
        self._rebuilding_loras = True
        while self.loras_layout.count():
            item = self.loras_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._lora_rows.clear()
        for i, pl in enumerate(self._animate().loras):
            row = AnimateLoraRow(i, pl, self.loras_container)
            row._clear.clicked.connect(lambda _=False, idx=i: self._remove_lora(idx))
            row._combo.currentIndexChanged.connect(self._sync_loras_to_model)
            row._combo.editTextChanged.connect(lambda _t: self._sync_loras_to_model())
            row._strength.valueChanged.connect(lambda _v: self._sync_loras_to_model())
            self.loras_layout.addWidget(row)
            self._lora_rows.append(row)
        self.add_lora_button.setEnabled(len(self._animate().loras) < MAX_ANIMATE_LORAS)
        self._rebuilding_loras = False

    def _sync_loras_to_model(self):
        if self._rebuilding_loras:
            return
        for row in self._lora_rows:
            self._animate().update_lora(row.index, row.lora_name(), row.lora_strength())

    def _generate(self):
        self._sync_layers()
        self._sync_comic()
        self._sync_models()
        self._sync_prompt()
        self._sync_sampling()
        self._sync_loras_to_model()
        self._on_res()
        self._on_length()
        self._animate().generate()

    def _apply(self):
        try:
            self._sync_layers()
            self._sync_comic()
            self._animate().apply_last_clip()
        except Exception as e:
            self._model.report_error(str(e))
            self._animate()._set_status(
                _("Could not apply: {msg}").format(msg=str(e))
                + " "
                + _("Generate a clip first, then Apply to Timeline.")
            )

    def _extend(self):
        self._sync_models()
        self._sync_prompt()
        self._sync_sampling()
        self._sync_loras_to_model()
        self._on_res()
        self._on_length()
        self._animate().generate_extend()

    def _stitch(self):
        self._sync_layers()
        self._sync_comic()
        self._sync_models()
        self._sync_prompt()
        self._sync_sampling()
        self._sync_loras_to_model()
        self._animate().generate_stitch()

