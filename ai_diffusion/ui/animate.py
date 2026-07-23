"""Wan 2.2 Animate workspace UI — T2V, I2V, and start/end video from Krita."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt5.QtCore import QMetaObject, QSize, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..files import FileFilter
from ..localization import translate as _
from ..backend.api import WanArch
from ..model.animate import (
    LENGTH_PRESETS,
    MAX_ANIMATE_LORAS,
    RESOLUTION_PRESETS,
    SAMPLING_PRESETS,
    AnimateLora,
    AnimateLoraTarget,
    AnimateMode,
    AnimateWorkspace,
)
from ..model.model import DocumentModel, Workspace
from ..model.properties import Bind, Binding, bind
from ..model.root import root
from . import theme
from .generation import ProgressBar
from .widget import StyleSelectWidget, WorkspaceSelectWidget


def _format_eta(seconds: float) -> str:
    """Human-readable remaining-time estimate for the status line."""
    if seconds < 0 or seconds != seconds or seconds == float("inf"):  # NaN / inf
        return ""
    s = max(0, int(seconds + 0.5))
    if s < 8:
        return _("a few seconds left")
    if s < 60:
        return _("~{n}s left").format(n=s)
    m, sec = divmod(s, 60)
    if m < 60:
        if sec == 0:
            return _("~{m}m left").format(m=m)
        return _("~{m}m {s}s left").format(m=m, s=sec)
    h, m = divmod(m, 60)
    return _("~{h}h {m}m left").format(h=h, m=m)


class AnimateLoraRow(QWidget):
    """One Wan video LoRA picker row with high / low / both expert targeting."""

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
        self._combo.setToolTip(
            _("Wan video LoRA — use Target to put lightx2v high/low on the matching expert")
        )
        layout.addWidget(self._combo, 1)

        self._target = QComboBox(self)
        self._target.addItem(_("Both"), AnimateLoraTarget.both)
        self._target.addItem(_("High"), AnimateLoraTarget.high)
        self._target.addItem(_("Low"), AnimateLoraTarget.low)
        self._target.setToolTip(
            _(
                "Both = same LoRA on high + low UNETs. High / Low = only that expert "
                "(required for lightx2v acceleration LoRAs)."
            )
        )
        layout.addWidget(self._target)

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
        self._set_lora(lora.name, lora.strength, lora.target)

    def _set_lora(self, name: str, strength: float, target: AnimateLoraTarget):
        self._updating = True
        self._strength.setValue(strength)
        tidx = self._target.findData(target)
        self._target.setCurrentIndex(tidx if tidx >= 0 else 0)
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

    def lora_target(self) -> AnimateLoraTarget:
        data = self._target.currentData()
        return data if isinstance(data, AnimateLoraTarget) else AnimateLoraTarget.both


class AnimateVideoSettingsDialog(QDialog):
    """Modal pop-out for Wan sampling / models / Video LoRAs (OK or X to close)."""

    def __init__(self, panel: QWidget, host: QWidget, parent=None):
        super().__init__(parent)
        self._panel = panel
        self._host = host
        self.setWindowTitle(_("Wan Video Settings"))
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setMinimumHeight(480)
        self.resize(460, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hint = QLabel(
            _(
                "Presets, models, sampler, steps, and Video LoRAs. Changes apply immediately. "
                "OK or ✕ closes this window."
            ),
            self,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.grey};")
        root.addWidget(hint)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(panel)
        panel.show()
        root.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, self)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    def done(self, result: int):
        # Keep the panel alive on the main widget's hidden host
        self._panel.setParent(self._host)
        self._panel.hide()
        super().done(result)


class AnimateWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._lora_rows: list[AnimateLoraRow] = []
        self._rebuilding_loras = False
        self._progress_t0: float | None = None
        self._progress_last = 0.0
        self._settings_dialog: AnimateVideoSettingsDialog | None = None
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
                "Wan 2.2 short clips (~5s max). Use Video Settings for presets / models / LoRAs. "
                "On 12GB start at 640×640 / 33 frames."
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

        # --- Advanced settings live in a hidden host; opened via Video Settings dialog ---
        self._settings_host = QWidget(self)
        self._settings_host.hide()
        host_layout = QVBoxLayout(self._settings_host)
        host_layout.setContentsMargins(0, 0, 0, 0)

        self.settings_panel = QWidget(self._settings_host)
        panel_layout = QVBoxLayout(self.settings_panel)
        panel_layout.setContentsMargins(0, 0, 4, 0)
        panel_layout.setSpacing(6)

        adv = QFormLayout()
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(4)
        self.preset_combo = QComboBox(self.settings_panel)
        self.preset_combo.addItem(_("— sampling / model presets —"), "")
        for label, _settings in SAMPLING_PRESETS:
            self.preset_combo.addItem(_(label), label)
        self.preset_combo.setToolTip(
            _(
                "5B Config 1–5 and 14B Config 1–5. 14B lightx2v/native presets follow Mode "
                "(I2V vs T2V) for models and LoRAs. Configs using res_2m / bong_tangent / "
                "beta57 need RES4LYF or ClownShark custom nodes."
            )
        )
        adv.addRow(_("Preset"), self.preset_combo)

        self._high_label = QLabel(_("High noise"), self.settings_panel)
        self.high_combo = QComboBox(self.settings_panel)
        self._low_label = QLabel(_("Low noise"), self.settings_panel)
        self.low_combo = QComboBox(self.settings_panel)
        self.te_combo = QComboBox(self.settings_panel)
        self.vae_combo = QComboBox(self.settings_panel)
        adv.addRow(self._high_label, self.high_combo)
        adv.addRow(self._low_label, self.low_combo)
        adv.addRow(_("Text encoder"), self.te_combo)
        adv.addRow(_("VAE"), self.vae_combo)

        self.sampler_combo = QComboBox(self.settings_panel)
        self.sampler_combo.setToolTip(
            _("High-expert (14B) or sole (5B) KSampler. Config 5 can use a different Low sampler.")
        )
        self._sampler_low_label = QLabel(_("Low sampler"), self.settings_panel)
        self.sampler_low_combo = QComboBox(self.settings_panel)
        self.sampler_low_combo.setToolTip(
            _(
                "14B low-noise expert only. “(same)” uses the High sampler. "
                "Config 5 uses res_2s here (needs RES4LYF)."
            )
        )
        self.scheduler_combo = QComboBox(self.settings_panel)
        self.scheduler_combo.setToolTip(
            _("Noise schedule for both experts unless a preset sets a low override.")
        )
        adv.addRow(_("Sampler"), self.sampler_combo)
        adv.addRow(self._sampler_low_label, self.sampler_low_combo)
        adv.addRow(_("Scheduler"), self.scheduler_combo)

        self.steps_spin = QSpinBox(self.settings_panel)
        self.steps_spin.setRange(2, 40)
        self.steps_spin.setToolTip(
            _(
                "Total denoising steps. lightx2v: 4 or 8. Native quality: 20–30. "
                "5B Config 2: 25–30 (avoid >35)."
            )
        )
        self._split_label = QLabel(_("Split step"), self.settings_panel)
        self.split_spin = QSpinBox(self.settings_panel)
        self.split_spin.setRange(1, 39)
        self.split_spin.setToolTip(
            _(
                "14B only: switches high→low noise UNET. Prefer ~50/50 (e.g. 4 of 8, 10 of 20). "
                "Avoid split 2 with 8 total steps (stiff motion / prompt ignore)."
            )
        )
        self.cfg_spin = QDoubleSpinBox(self.settings_panel)
        self.cfg_spin.setRange(0.0, 10.0)
        self.cfg_spin.setSingleStep(0.1)
        self.cfg_spin.setToolTip(
            _(
                "Guidance. ~1.0 for distilled / lightx2v; ~3.5 for native 14B; "
                "~4.0 for 5B Config 2; ~5.0 for official 5B uni_pc."
            )
        )
        self.shift_spin = QDoubleSpinBox(self.settings_panel)
        self.shift_spin.setRange(1.0, 16.0)
        self.shift_spin.setSingleStep(0.5)
        self.shift_spin.setToolTip(
            _(
                "ModelSamplingSD3 shift. ~8 distilled I2V / 5B Config 1; "
                "~5 for lightx2v, native 14B, and 5B Config 2 (dpmpp_2m)."
            )
        )
        adv.addRow(_("Steps"), self.steps_spin)
        adv.addRow(self._split_label, self.split_spin)
        adv.addRow(_("CFG"), self.cfg_spin)
        adv.addRow(_("Shift"), self.shift_spin)
        panel_layout.addLayout(adv)

        lora_header = QHBoxLayout()
        lora_header.addWidget(QLabel(_("Video LoRAs"), self.settings_panel))
        lora_header.addStretch(1)
        self.add_lora_button = QPushButton(_("Add LoRA"), self.settings_panel)
        self.add_lora_button.setFixedHeight(22)
        self.add_lora_button.setToolTip(
            _(
                "Wan video LoRAs from ComfyUI/models/loras. Set Target to High or Low for "
                "lightx2v acceleration (must match the expert UNET). Native quality presets "
                "clear LoRAs and use more steps."
            )
        )
        lora_header.addWidget(self.add_lora_button)
        panel_layout.addLayout(lora_header)

        self.loras_container = QWidget(self.settings_panel)
        self.loras_layout = QVBoxLayout(self.loras_container)
        self.loras_layout.setContentsMargins(0, 0, 0, 0)
        self.loras_layout.setSpacing(2)
        panel_layout.addWidget(self.loras_container)
        panel_layout.addStretch(1)

        host_layout.addWidget(self.settings_panel)
        self.settings_panel.hide()

        self.video_settings_button = QPushButton(_("Video Settings…"), self)
        self.video_settings_button.setMinimumHeight(32)
        self.video_settings_button.setToolTip(
            _("Open presets, models, sampler, steps, CFG, shift, and Video LoRAs")
        )
        layout.addWidget(self.video_settings_button)

        self.settings_summary = QLabel("", self)
        self.settings_summary.setWordWrap(True)
        self.settings_summary.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(self.settings_summary)

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
        self.video_settings_button.clicked.connect(self._open_video_settings)
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
        self.sampler_low_combo.currentIndexChanged.connect(self._sync_sampling)
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
            self.sampler_low_combo,
            self.scheduler_combo,
        ):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._update_settings_summary()

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
                a.missing_loras_alert.connect(self._popup_missing_loras),
            ]
            self.progress_bar.model = model
            self._sync_from_model()
            self._rebuild_layer_combos()
            self._rebuild_comic_combos()
            self._rebuild_model_combos()
            self._rebuild_sampling_combos()
            self._rebuild_loras()
            self._refresh_status()
            self._update_settings_summary()

    def _animate(self) -> AnimateWorkspace:
        return self._model.animate

    def _open_video_settings(self):
        """Show sampling / models / LoRAs in a modal dialog (OK or ✕ to close)."""
        self._sync_sampling()
        self._sync_models()
        self._sync_loras_to_model()
        self._rebuild_model_combos()
        self._rebuild_sampling_combos()
        self._rebuild_loras()
        dlg = AnimateVideoSettingsDialog(self.settings_panel, self._settings_host, self)
        self._settings_dialog = dlg
        dlg.exec_()
        self._settings_dialog = None
        self._sync_sampling()
        self._sync_models()
        self._sync_loras_to_model()
        self._update_settings_summary()

    def _update_settings_summary(self):
        a = self._animate()
        arch = "5B" if a.arch is WanArch.ti2v5b else "14B"
        samp = a.sampler or "—"
        if a.sampler_low and a.sampler_low != a.sampler and a.arch is not WanArch.ti2v5b:
            samp = f"{samp}→{a.sampler_low}"
        sched = a.scheduler or "—"
        split = ""
        if a.arch is not WanArch.ti2v5b:
            split = _(" · split {n}").format(n=a.split_step)
        n_loras = sum(1 for l in a.loras if l.name)
        lora_txt = _(" · {n} LoRA").format(n=n_loras) if n_loras == 1 else (
            _(" · {n} LoRAs").format(n=n_loras) if n_loras else ""
        )
        self.settings_summary.setText(
            _("{arch} · {sampler} / {scheduler} · {steps} steps{split} · CFG {cfg} · shift {shift}{loras}").format(
                arch=arch,
                sampler=samp,
                scheduler=sched,
                steps=a.steps,
                split=split,
                cfg=a.cfg,
                shift=a.shift,
                loras=lora_txt,
            )
        )

    def _refresh_status(self, *_args):
        """Status line + live percent / ETA while Generate / Extend / Stitch is running."""
        a = self._animate()
        base = a.status or a.idle_hint()
        p = self._model.progress
        if base and 0 < p < 1:
            now = time.monotonic()
            # New run or progress reset (queue restart) — restart the clock
            if self._progress_t0 is None or p + 0.001 < self._progress_last:
                self._progress_t0 = now
            self._progress_last = p
            elapsed = now - self._progress_t0
            pct = int(p * 100)
            # Wait until we have a little signal so the estimate isn't wild
            if p >= 0.03 and elapsed >= 1.5:
                eta = elapsed * (1.0 - p) / p
                eta_txt = _format_eta(eta)
                self.status_label.setText(f"{base}  {pct}% · {eta_txt}" if eta_txt else f"{base}  {pct}%")
            else:
                self.status_label.setText(f"{base}  {pct}%")
        else:
            self._progress_t0 = None
            self._progress_last = 0.0
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
        self._apply_arch_ui(a.arch is WanArch.ti2v5b)
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
        self._update_settings_summary()

    def _on_mode_ui(self):
        data = self.mode_combo.currentData()
        if data is not None:
            self._animate().mode = data
        is_t2v = data is AnimateMode.t2v
        is_5b = self._animate().arch is WanArch.ti2v5b
        # T2V is prompt-only; start/end still matter for Stitch after a T2V clip
        for w in (
            self._start_label,
            self.start_combo,
            self._comic_start_label,
            self.comic_start_combo,
        ):
            w.setVisible(not is_t2v)
        # End layer stays available for Stitch even in T2V / I2V (14B)
        self.end_combo.setEnabled(True)
        self.comic_end_combo.setEnabled(True)
        self._apply_arch_ui(is_5b)
        self._rebuild_model_combos()
        # Mode-specific idle hint when no run status yet
        self._refresh_status()

    def _apply_arch_ui(self, is_5b: bool):
        self._high_label.setText(_("Model") if is_5b else _("High noise"))
        self._low_label.setVisible(not is_5b)
        self.low_combo.setVisible(not is_5b)
        self._split_label.setVisible(not is_5b)
        self.split_spin.setVisible(not is_5b)
        self._sampler_low_label.setVisible(not is_5b)
        self.sampler_low_combo.setVisible(not is_5b)
        self.stitch_button.setEnabled(not is_5b)
        if is_5b:
            self.stitch_button.setToolTip(
                _("Stitch needs 14B FLF2V — switch to a 14B preset first.")
            )
        else:
            self.stitch_button.setToolTip(
                _(
                    "After a clip: FLF2V from its last frame to your End layer/panel, "
                    "then append those frames (join toward a target end pose)."
                )
            )

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
        # Low sampler: "(same)" + live list
        self.sampler_low_combo.blockSignals(True)
        self.sampler_low_combo.clear()
        self.sampler_low_combo.addItem(_("(same as high)"), "")
        for name in a.list_samplers():
            self.sampler_low_combo.addItem(name, name)
        low = a.sampler_low or ""
        idx = self.sampler_low_combo.findData(low)
        if low and idx < 0:
            self.sampler_low_combo.addItem(low, low)
            idx = self.sampler_low_combo.findData(low)
        self.sampler_low_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.sampler_low_combo.blockSignals(False)

    def _on_preset(self):
        key = self.preset_combo.currentData()
        if not key:
            return
        a = self._animate()
        if a.apply_sampling_preset(str(key)):
            self._apply_arch_ui(a.arch is WanArch.ti2v5b)
            self._rebuild_model_combos()
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
            self._rebuild_loras()
            self._refresh_status()
            missing = a.find_missing_loras(self._model._connection.client_if_connected)
            if missing:
                self._popup_missing_loras(missing, blocking_generate=False)
            self._update_settings_summary()

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
        low = self.sampler_low_combo.currentData()
        a.sampler_low = "" if low is None else str(low)
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
        self._update_settings_summary()

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
            row._target.currentIndexChanged.connect(self._sync_loras_to_model)
            row._strength.valueChanged.connect(lambda _v: self._sync_loras_to_model())
            self.loras_layout.addWidget(row)
            self._lora_rows.append(row)
        self.add_lora_button.setEnabled(len(self._animate().loras) < MAX_ANIMATE_LORAS)
        self._rebuilding_loras = False
        self._update_settings_summary()

    def _sync_loras_to_model(self):
        if self._rebuilding_loras:
            return
        for row in self._lora_rows:
            self._animate().update_lora(
                row.index, row.lora_name(), row.lora_strength(), row.lora_target()
            )

    def _popup_missing_loras(self, missing: list, *, blocking_generate: bool = True):
        """Obvious dialog when selected Video LoRAs are not on disk / Comfy."""
        if not missing:
            return
        names = "\n".join(f"  • {n}" for n in missing)
        body = _(
            "Required LoRA file(s) were not found:\n\n"
            "{names}\n\n"
            "Put them in:\n"
            "  ComfyUI/models/loras\n"
            "(or your extra_model_paths loras folder)\n\n"
            "Then refresh ComfyUI models / restart the server, and try again.\n"
            "Or clear the Video LoRA rows / switch to a native (no LoRA) preset."
        ).format(names=names)
        if blocking_generate:
            QMessageBox.critical(
                self,
                _("Missing Wan Video LoRA — generation stopped"),
                _("Generation was not started.") + "\n\n" + body,
            )
            self._animate()._set_status(
                _("Missing Video LoRA(s) — generation stopped.")
                + " "
                + _("Install into models/loras, then retry.")
            )
        else:
            QMessageBox.warning(
                self,
                _("Missing Wan Video LoRA"),
                _(
                    "This preset needs LoRA(s) that are not installed yet.\n"
                    "Generate will stay blocked until they are found.\n\n"
                )
                + body,
            )
            self._animate()._set_status(
                _("Preset needs missing Video LoRA(s).")
                + " "
                + _("Install into models/loras before Generate.")
            )

    def _ensure_loras_ready(self) -> bool:
        """Return False (and show popup) if any selected Video LoRA is missing."""
        self._sync_loras_to_model()
        client = self._model._connection.client_if_connected
        missing = self._animate().find_missing_loras(client)
        if missing:
            self._popup_missing_loras(missing)
            return False
        return True

    def _generate(self):
        self._sync_layers()
        self._sync_comic()
        self._sync_models()
        self._sync_prompt()
        self._sync_sampling()
        self._sync_loras_to_model()
        self._on_res()
        self._on_length()
        if not self._ensure_loras_ready():
            return
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
        if not self._ensure_loras_ready():
            return
        self._animate().generate_extend()

    def _stitch(self):
        self._sync_layers()
        self._sync_comic()
        self._sync_models()
        self._sync_prompt()
        self._sync_sampling()
        self._sync_loras_to_model()
        if not self._ensure_loras_ready():
            return
        self._animate().generate_stitch()

