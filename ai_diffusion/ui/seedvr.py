"""SeedVR2 image enhance workspace UI."""

from __future__ import annotations

from PyQt5.QtCore import QMetaObject, QSize, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..localization import translate as _
from ..model.jobs import JobKind
from ..model.model import DocumentModel, Workspace
from ..model.properties import Bind, Binding, bind, bind_toggle
from ..model.root import root
from ..model.seedvr import (
    ATTENTION_OPTIONS,
    COLOR_CORRECTION_OPTIONS,
    RESOLUTION_PRESETS,
    SeedVRWorkspace,
)
from . import theme
from .generation import ProgressBar
from .widget import ErrorBox, GenerateButton, WorkspaceSelectWidget


class SeedVRWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
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
                "SeedVR2 diffusion enhance / upscale (image). Separate from ESRGAN Upscale. "
                "Requires ComfyUI-SeedVR2_VideoUpscaler. Models download on first use into "
                "ComfyUI/models/SEEDVR2."
            ),
            self,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.grey};")
        outer.addWidget(tip)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        presets = QHBoxLayout()
        self.preset_12gb = QPushButton(_("12GB Balanced"), body)
        self.preset_quality = QPushButton(_("Quality"), body)
        self.preset_low = QPushButton(_("Low VRAM"), body)
        self.preset_12gb.setToolTip(_("3B FP8 + BlockSwap 16 + tiled VAE — good for RTX 3060 12GB"))
        self.preset_quality.setToolTip(_("7B FP8 mixed, less tiling — needs more VRAM"))
        self.preset_low.setToolTip(_("3B GGUF Q4 + heavy BlockSwap / small tiles"))
        presets.addWidget(self.preset_12gb)
        presets.addWidget(self.preset_quality)
        presets.addWidget(self.preset_low)
        layout.addLayout(presets)

        # --- Output ---
        out_box = QGroupBox(_("Output"), body)
        out_form = QFormLayout(out_box)
        self.res_combo = QComboBox(out_box)
        for label, val in RESOLUTION_PRESETS:
            self.res_combo.addItem(_(label), val)
        self.res_combo.setToolTip(_("Target size for the shortest edge (aspect preserved)"))
        out_form.addRow(_("Resolution"), self.res_combo)

        self.res_spin = QSpinBox(out_box)
        self.res_spin.setRange(16, 8192)
        self.res_spin.setSingleStep(2)
        self.res_spin.setToolTip(_("Custom shortest-edge resolution"))
        out_form.addRow(_("Custom res."), self.res_spin)

        self.max_res_spin = QSpinBox(out_box)
        self.max_res_spin.setRange(0, 16384)
        self.max_res_spin.setSingleStep(64)
        self.max_res_spin.setSpecialValueText(_("No limit"))
        self.max_res_spin.setToolTip(_("Cap the longest edge (0 = unlimited)"))
        out_form.addRow(_("Max edge"), self.max_res_spin)

        self.color_combo = QComboBox(out_box)
        for name in COLOR_CORRECTION_OPTIONS:
            self.color_combo.addItem(name, name)
        self.color_combo.setToolTip(_("Match colors back to the original after enhance"))
        out_form.addRow(_("Color corr."), self.color_combo)

        self.target_label = QLabel("", out_box)
        self.target_label.setStyleSheet(f"color: {theme.grey};")
        out_form.addRow(_("Est. size"), self.target_label)
        layout.addWidget(out_box)

        # --- Models ---
        model_box = QGroupBox(_("Models"), body)
        model_form = QFormLayout(model_box)
        self.dit_combo = QComboBox(model_box)
        self.vae_combo = QComboBox(model_box)
        self.device_combo = QComboBox(model_box)
        self.attention_combo = QComboBox(model_box)
        for name in ATTENTION_OPTIONS:
            self.attention_combo.addItem(name, name)
        model_form.addRow(_("DiT"), self.dit_combo)
        model_form.addRow(_("VAE"), self.vae_combo)
        model_form.addRow(_("Device"), self.device_combo)
        model_form.addRow(_("Attention"), self.attention_combo)
        layout.addWidget(model_box)

        # --- Memory ---
        mem_box = QGroupBox(_("Memory (VRAM)"), body)
        mem_form = QFormLayout(mem_box)
        self.blocks_spin = QSpinBox(mem_box)
        self.blocks_spin.setRange(0, 36)
        self.blocks_spin.setToolTip(_("BlockSwap: 0 = off. Higher = less VRAM, slower"))
        mem_form.addRow(_("Blocks to swap"), self.blocks_spin)

        self.swap_io_check = QCheckBox(_("Swap I/O components"), mem_box)
        mem_form.addRow(self.swap_io_check)

        self.dit_offload_combo = QComboBox(mem_box)
        self.vae_offload_combo = QComboBox(mem_box)
        self.tensor_offload_combo = QComboBox(mem_box)
        for c in (self.dit_offload_combo, self.vae_offload_combo, self.tensor_offload_combo):
            c.addItem("none", "none")
            c.addItem("cpu", "cpu")
            c.addItem("cuda:0", "cuda:0")
        mem_form.addRow(_("DiT offload"), self.dit_offload_combo)
        mem_form.addRow(_("VAE offload"), self.vae_offload_combo)
        mem_form.addRow(_("Tensor offload"), self.tensor_offload_combo)

        self.cache_check = QCheckBox(_("Cache models between runs"), mem_box)
        self.cache_check.setToolTip(_("Requires DiT offload ≠ none"))
        mem_form.addRow(self.cache_check)
        layout.addWidget(mem_box)

        # --- VAE tiling ---
        tile_box = QGroupBox(_("VAE tiling"), body)
        tile_form = QFormLayout(tile_box)
        self.encode_tiled = QCheckBox(_("Encode tiled"), tile_box)
        self.decode_tiled = QCheckBox(_("Decode tiled"), tile_box)
        tile_form.addRow(self.encode_tiled)
        tile_form.addRow(self.decode_tiled)
        self.enc_tile = QSpinBox(tile_box)
        self.enc_tile.setRange(64, 4096)
        self.enc_tile.setSingleStep(32)
        self.dec_tile = QSpinBox(tile_box)
        self.dec_tile.setRange(64, 4096)
        self.dec_tile.setSingleStep(32)
        self.enc_overlap = QSpinBox(tile_box)
        self.enc_overlap.setRange(0, 512)
        self.enc_overlap.setSingleStep(32)
        self.dec_overlap = QSpinBox(tile_box)
        self.dec_overlap.setRange(0, 512)
        self.dec_overlap.setSingleStep(32)
        tile_form.addRow(_("Encode tile"), self.enc_tile)
        tile_form.addRow(_("Encode overlap"), self.enc_overlap)
        tile_form.addRow(_("Decode tile"), self.dec_tile)
        tile_form.addRow(_("Decode overlap"), self.dec_overlap)
        layout.addWidget(tile_box)

        # --- Quality extras ---
        qual_box = QGroupBox(_("Quality extras"), body)
        qual_form = QFormLayout(qual_box)
        self.input_noise = QDoubleSpinBox(qual_box)
        self.input_noise.setRange(0.0, 1.0)
        self.input_noise.setSingleStep(0.01)
        self.input_noise.setDecimals(3)
        self.latent_noise = QDoubleSpinBox(qual_box)
        self.latent_noise.setRange(0.0, 1.0)
        self.latent_noise.setSingleStep(0.01)
        self.latent_noise.setDecimals(3)
        qual_form.addRow(_("Input noise"), self.input_noise)
        qual_form.addRow(_("Latent noise"), self.latent_noise)
        self.debug_check = QCheckBox(_("Debug logging (Comfy console)"), qual_box)
        qual_form.addRow(self.debug_check)
        layout.addWidget(qual_box)

        self.layer_check = QCheckBox(_("Use active layer only (else full document)"), body)
        layout.addWidget(self.layer_check)

        credit = QLabel(
            '<a href="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler">SeedVR2</a> '
            + _("by numz / ByteDance Seed — Apache-2.0 style research model."),
            body,
        )
        credit.setOpenExternalLinks(True)
        credit.setWordWrap(True)
        credit.setStyleSheet(f"color: {theme.grey}; font-size: 10px;")
        layout.addWidget(credit)
        layout.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        self.enhance_button = GenerateButton(JobKind.upscaling, self)
        self.enhance_button.operation = _("Enhance with SeedVR")
        self.enhance_button.clicked.connect(self._enhance)
        outer.addWidget(self.enhance_button)

        self.progress_bar = ProgressBar(self)
        outer.addWidget(self.progress_bar)

        self.error_box = ErrorBox(self)
        outer.addWidget(self.error_box)

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.grey};")
        outer.addWidget(self.status_label)

        self.preset_12gb.clicked.connect(lambda: self._seedvr().apply_preset_12gb())
        self.preset_quality.clicked.connect(lambda: self._seedvr().apply_preset_quality())
        self.preset_low.clicked.connect(lambda: self._seedvr().apply_preset_low_vram())
        self.res_combo.currentIndexChanged.connect(self._on_res_preset)
        self.res_spin.valueChanged.connect(self._on_res_spin)
        self.dit_combo.currentIndexChanged.connect(self._sync_models)
        self.vae_combo.currentIndexChanged.connect(self._sync_models)
        self.device_combo.currentIndexChanged.connect(self._sync_models)
        self.attention_combo.currentIndexChanged.connect(self._sync_models)
        self.color_combo.currentIndexChanged.connect(self._sync_models)
        self.dit_offload_combo.currentIndexChanged.connect(self._sync_models)
        self.vae_offload_combo.currentIndexChanged.connect(self._sync_models)
        self.tensor_offload_combo.currentIndexChanged.connect(self._sync_models)

        for combo in (
            self.dit_combo,
            self.vae_combo,
            self.device_combo,
            self.res_combo,
            self.color_combo,
            self.attention_combo,
            self.dit_offload_combo,
            self.vae_offload_combo,
            self.tensor_offload_combo,
        ):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)

    def minimumSizeHint(self) -> QSize:
        return QSize(180, super().minimumSizeHint().height())

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            s = model.seedvr
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(s, "resolution", self.res_spin, "value"),
                bind(s, "max_resolution", self.max_res_spin, "value"),
                bind(s, "blocks_to_swap", self.blocks_spin, "value"),
                bind(s, "encode_tile_size", self.enc_tile, "value"),
                bind(s, "encode_tile_overlap", self.enc_overlap, "value"),
                bind(s, "decode_tile_size", self.dec_tile, "value"),
                bind(s, "decode_tile_overlap", self.dec_overlap, "value"),
                bind(s, "input_noise_scale", self.input_noise, "value"),
                bind(s, "latent_noise_scale", self.latent_noise, "value"),
                bind_toggle(s, "swap_io_components", self.swap_io_check),
                bind_toggle(s, "cache_model", self.cache_check),
                bind_toggle(s, "encode_tiled", self.encode_tiled),
                bind_toggle(s, "decode_tiled", self.decode_tiled),
                bind_toggle(s, "enable_debug", self.debug_check),
                bind_toggle(s, "use_active_layer", self.layer_check),
                bind(s, "can_generate", self.enhance_button, "enabled", Bind.one_way),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                s.status_changed.connect(self.status_label.setText),
                s.resolution_changed.connect(lambda *_: self._update_target()),
                s.max_resolution_changed.connect(lambda *_: self._update_target()),
                s.dit_model_changed.connect(lambda *_: self._sync_combo_data(self.dit_combo, s.dit_model)),
                s.vae_model_changed.connect(lambda *_: self._sync_combo_data(self.vae_combo, s.vae_model)),
            ]
            self.enhance_button.model = model
            self.progress_bar.model = model
            self._rebuild_model_combos()
            self._sync_from_model()
            self._update_target()

    def _seedvr(self) -> SeedVRWorkspace:
        return self._model.seedvr

    def _sync_combo_data(self, combo: QComboBox, value: str):
        idx = combo.findData(value)
        if idx >= 0 and combo.currentIndex() != idx:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _rebuild_model_combos(self):
        s = self._seedvr()
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

        fill(self.dit_combo, s.list_dit_models(), s.dit_model)
        fill(self.vae_combo, s.list_vae_models(), s.vae_model)
        fill(self.device_combo, s.list_devices(), s.device)

    def _sync_from_model(self):
        s = self._seedvr()
        self._sync_combo_data(self.dit_combo, s.dit_model)
        self._sync_combo_data(self.vae_combo, s.vae_model)
        self._sync_combo_data(self.device_combo, s.device)
        self._sync_combo_data(self.attention_combo, s.attention_mode)
        self._sync_combo_data(self.color_combo, s.color_correction)
        self._sync_combo_data(self.dit_offload_combo, s.dit_offload_device)
        self._sync_combo_data(self.vae_offload_combo, s.vae_offload_device)
        self._sync_combo_data(self.tensor_offload_combo, s.tensor_offload_device)
        idx = self.res_combo.findData(s.resolution)
        self.res_combo.blockSignals(True)
        if idx >= 0:
            self.res_combo.setCurrentIndex(idx)
        self.res_combo.blockSignals(False)
        self.status_label.setText(s.status)

    def _sync_models(self):
        s = self._seedvr()
        if self.dit_combo.currentData():
            s.dit_model = self.dit_combo.currentData()
        if self.vae_combo.currentData():
            s.vae_model = self.vae_combo.currentData()
        if self.device_combo.currentData():
            s.device = self.device_combo.currentData()
        if self.attention_combo.currentData():
            s.attention_mode = self.attention_combo.currentData()
        if self.color_combo.currentData():
            s.color_correction = self.color_combo.currentData()
        if self.dit_offload_combo.currentData() is not None:
            s.dit_offload_device = self.dit_offload_combo.currentData()
        if self.vae_offload_combo.currentData() is not None:
            s.vae_offload_device = self.vae_offload_combo.currentData()
        if self.tensor_offload_combo.currentData() is not None:
            s.tensor_offload_device = self.tensor_offload_combo.currentData()

    def _on_res_preset(self):
        data = self.res_combo.currentData()
        if data:
            self._seedvr().resolution = int(data)

    def _on_res_spin(self):
        self._update_target()

    def _update_target(self):
        try:
            e = self._seedvr().estimate_target()
            self.target_label.setText(f"{e.width} × {e.height}")
        except Exception:
            self.target_label.setText("—")

    def _enhance(self):
        self._sync_models()
        self._seedvr().enhance()
