from krita import Krita
from PyQt5.QtCore import QMetaObject, Qt, QUuid
from PyQt5.QtWidgets import (
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..files import FileFilter
from ..layer import LayerType
from ..localization import translate as _
from ..model.augment import MAX_REGIONS, OVERLAY_COLORS, AnnotationRegion
from ..model.jobs import JobKind
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind
from ..model.root import root
from . import theme
from .widget import ErrorBox, GenerateButton, WorkspaceSelectWidget


class RegionLoraWidget(QWidget):
    """Optional per-region LoRA applied only during that region's inpaint pass."""

    def __init__(self, region: AnnotationRegion, parent=None):
        super().__init__(parent)
        self._region_id = region.id
        self._loras = FileFilter(root.files.loras)
        self._loras.available_only = True
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lora_label = QLabel(_("LoRA:"), self)
        lora_label.setToolTip(
            _(
                "Optional LoRA used only while editing this region. "
                "Style LoRAs still apply to every pass."
            )
        )
        layout.addWidget(lora_label)

        completer = QCompleter(self._loras)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        self._combo = QComboBox(self)
        self._combo.setEditable(True)
        self._combo.setModel(self._loras)
        self._combo.setCompleter(completer)
        self._combo.setMaxVisibleItems(16)
        self._combo.setMinimumWidth(120)
        self._combo.setToolTip(lora_label.toolTip())
        self._combo.currentIndexChanged.connect(self._on_lora_changed)
        layout.addWidget(self._combo, 1)

        self._strength = QDoubleSpinBox(self)
        self._strength.setRange(0.05, 3.0)
        self._strength.setSingleStep(0.05)
        self._strength.setDecimals(2)
        self._strength.setValue(region.lora_strength)
        self._strength.setPrefix(_("Strength") + ": ")
        self._strength.setToolTip(_("LoRA strength for this region (1.0 is typical)."))
        self._strength.valueChanged.connect(self._on_strength_changed)
        layout.addWidget(self._strength)

        self._clear = QToolButton(self)
        self._clear.setIcon(theme.icon("discard"))
        self._clear.setToolTip(_("Clear LoRA for this region"))
        self._clear.clicked.connect(self._clear_lora)
        layout.addWidget(self._clear)

        if line_edit := self._combo.lineEdit():
            line_edit.setPlaceholderText(_("(none)"))

        self._set_lora(region.lora_name, region.lora_strength)
        root.files.loras.rowsInserted.connect(self._refresh_lora_list)
        root.files.loras.rowsRemoved.connect(self._refresh_lora_list)

    def _clear_lora(self):
        self._updating = True
        self._combo.setCurrentIndex(-1)
        self._combo.setEditText("")
        self._updating = False
        self._update_strength_enabled()

    def _refresh_lora_list(self):
        current = self._combo.currentData()
        self._combo.blockSignals(True)
        if current is None:
            current = ""
        self._combo.setCurrentIndex(-1)
        self._combo.setEditText("")
        if current:
            idx = self._combo.findData(current)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            else:
                self._combo.setEditText(str(current))
        self._combo.blockSignals(False)
        self._update_strength_enabled()

    def _set_lora(self, lora_name: str, lora_strength: float):
        self._updating = True
        self._strength.setValue(lora_strength)
        if lora_name:
            idx = self._combo.findData(lora_name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            else:
                self._combo.setEditText(lora_name)
        else:
            self._combo.setCurrentIndex(-1)
            self._combo.setEditText("")
        self._updating = False
        self._update_strength_enabled()

    def _update_strength_enabled(self):
        self._strength.setEnabled(bool(self.lora_name()))

    def lora_name(self) -> str:
        data = self._combo.currentData()
        if data:
            return str(data)
        text = self._combo.currentText().strip()
        if not text:
            return ""
        if file := root.files.loras.find(text):
            return file.id
        for i in range(self._loras.rowCount()):
            if self._loras[i].name.lower() == text.lower():
                return self._loras[i].id
        return text

    def lora_strength(self) -> float:
        return self._strength.value()

    def _on_lora_changed(self):
        if self._updating:
            return
        self._update_strength_enabled()
        if self.lora_name() and self._combo.currentData():
            file = root.files.loras.find(self.lora_name())
            if file and (default := file.meta("lora_strength")):
                self._strength.setValue(float(default))

    def _on_strength_changed(self):
        pass

    def sync_to_model(self, augment):
        augment.update_region_lora(self._region_id, self.lora_name(), self.lora_strength())


class RegionCard(QWidget):
    def __init__(self, region: AnnotationRegion, index: int, parent=None):
        super().__init__(parent)
        self.region_id = region.id
        color = OVERLAY_COLORS[region.color_index % len(OVERLAY_COLORS)]
        self.setStyleSheet(
            f"RegionCard {{ border-left: 4px solid rgb({color[0]},{color[1]},{color[2]}); }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 4, 4)

        header = QHBoxLayout()
        b = region.bounds
        header.addWidget(
            QLabel(_("Region {n}: {w}×{h} at ({x},{y})").format(n=index + 1, w=b.width, h=b.height, x=b.x, y=b.y))
        )
        self.remove_button = QPushButton(Krita.instance().icon("deletelayer"), "")
        self.remove_button.setFixedSize(24, 24)
        self.remove_button.setToolTip(_("Remove this region"))
        header.addWidget(self.remove_button)
        layout.addLayout(header)

        self.text_edit = QPlainTextEdit(region.description, self)
        self.text_edit.setPlaceholderText(
            _("Describe the change, e.g. \"add a black rose tattoo on the thigh\"")
        )
        self.text_edit.setMaximumHeight(80)
        layout.addWidget(self.text_edit)

        self.lora_widget = RegionLoraWidget(region, self)
        layout.addWidget(self.lora_widget)


class AugmentWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._region_cards: dict[str, RegionCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)

        self.workspace_select = WorkspaceSelectWidget(self)
        header = QHBoxLayout()
        header.addWidget(self.workspace_select)
        layout.addLayout(header)

        info = QLabel(
            _(
                "Pick a layer, draw a rectangular selection, add regions with descriptions "
                "(and optional per-region LoRAs), then generate localized changes."
            ),
            self,
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(info)

        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel(_("Source layer:"), self))
        self.layer_select = QComboBox(self)
        self.layer_select.setToolTip(_("Layer whose pixels will be used as the starting image."))
        layer_row.addWidget(self.layer_select, 1)
        layout.addLayout(layer_row)

        self.add_region_button = QPushButton(_("Add region from selection"), self)
        self.add_region_button.setToolTip(
            _("Use Krita's rectangular selection tool, then click to mark a region.")
        )
        layout.addWidget(self.add_region_button)

        self.global_prompt = QPlainTextEdit(self)
        self.global_prompt.setPlaceholderText(
            _("Optional prompt added to every marked region (unmarked areas stay unchanged).")
        )
        self.global_prompt.setMaximumHeight(60)
        layout.addWidget(self.global_prompt)

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel(_("Change strength:"), self))
        self.strength_spin = QDoubleSpinBox(self)
        self.strength_spin.setRange(0.25, 0.85)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setToolTip(
            _(
                "How much each region is re-painted (0.6–0.75 works well). "
                "Higher = stronger prompt effect; lower = subtler blend. "
                "Use action prompts like \"add a red bow in the hair\".\n\n"
                "Edge softness: Settings → Configure AI Diffusion → "
                "Selection Feather / Selection Blend.\n"
                "Color harmonization: same dialog → Color Match (on by default)."
            )
        )
        strength_row.addWidget(self.strength_spin)
        layout.addLayout(strength_row)

        self.regions_label = QLabel(self)
        layout.addWidget(self.regions_label)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.regions_container = QWidget()
        self.regions_layout = QVBoxLayout(self.regions_container)
        self.regions_layout.setContentsMargins(0, 0, 0, 0)
        self.regions_layout.addStretch()
        scroll.setWidget(self.regions_container)
        layout.addWidget(scroll, 1)

        self.generate_button = GenerateButton(JobKind.augment, self)
        self.generate_button.operation = _("Generate")
        self.generate_button.clicked.connect(self.generate)
        layout.addWidget(self.generate_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

        self.add_region_button.clicked.connect(self._add_region)
        self.layer_select.currentIndexChanged.connect(self._layer_changed)
        self.global_prompt.textChanged.connect(self._on_global_prompt_edited)

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
                bind(model.augment, "can_generate", self.generate_button, "enabled", Bind.one_way),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                bind(model.augment, "strength", self.strength_spin, "value"),
                model.augment.regions_changed.connect(self._rebuild_regions),
                model.augment.global_prompt_changed.connect(self._sync_global_prompt),
                model.progress_changed.connect(self.update_progress),
                model.layers.changed.connect(self._refresh_layers),
            ]
            self.generate_button.model = model
            self._refresh_layers()
            self._rebuild_regions()
            self._sync_global_prompt()
            self.update_progress()

    def _refresh_layers(self):
        model = self._model
        current = model.augment.source_layer_id
        self.layer_select.blockSignals(True)
        self.layer_select.clear()
        for layer in model.layers.all:
            if layer.type in (LayerType.paint, LayerType.group, LayerType.file, LayerType.clone):
                if layer.name == "[AI Augment]" or layer.name.startswith("Region "):
                    parent = layer.parent_layer
                    if parent and parent.name == "[AI Augment]":
                        continue
                self.layer_select.addItem(layer.name, layer.id_string)
        if current:
            idx = self.layer_select.findData(current)
            if idx >= 0:
                self.layer_select.setCurrentIndex(idx)
            elif self.layer_select.count() > 0:
                self.layer_select.setCurrentIndex(0)
                self._layer_changed()
        elif self.layer_select.count() > 0:
            active = model.layers.active
            if active.type.is_image:
                idx = self.layer_select.findData(active.id_string)
                if idx >= 0:
                    self.layer_select.setCurrentIndex(idx)
            self._layer_changed()
        self.layer_select.blockSignals(False)

    def _layer_changed(self):
        layer_id = self.layer_select.currentData()
        if layer_id:
            layer = self._model.layers.find(QUuid(layer_id))
            self._model.augment.set_source_layer(layer)

    def _sync_global_prompt(self):
        text = self._model.augment.global_prompt
        if text != self.global_prompt.toPlainText():
            self.global_prompt.setPlainText(text)

    def _on_global_prompt_edited(self):
        self._model.augment.global_prompt = self.global_prompt.toPlainText()

    def _add_region(self):
        self._model.augment.add_region_from_selection()

    def _rebuild_regions(self):
        augment = self._model.augment
        self.regions_label.setText(
            _("Regions: {n}/{max}").format(n=len(augment.regions), max=MAX_REGIONS)
        )

        while self.regions_layout.count() > 1:
            item = self.regions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._region_cards.clear()

        for i, region in enumerate(augment.regions):
            card = RegionCard(region, i, self.regions_container)
            card.remove_button.clicked.connect(lambda _, rid=region.id: augment.remove_region(rid))
            card.text_edit.textChanged.connect(
                lambda rid=region.id, edit=card.text_edit: augment.update_region_description(
                    rid, edit.toPlainText()
                )
            )
            self._region_cards[region.id] = card
            self.regions_layout.insertWidget(i, card)

        self.add_region_button.setEnabled(len(augment.regions) < MAX_REGIONS)

    def update_progress(self):
        self.progress_bar.setValue(int(max(self.model.progress, 0) * 100))

    def generate(self):
        self._sync_region_descriptions()
        self._sync_region_loras()
        self._model.augment.global_prompt = self.global_prompt.toPlainText()
        self._model.augment.generate()

    def _sync_region_descriptions(self):
        for region_id, card in self._region_cards.items():
            self._model.augment.update_region_description(
                region_id, card.text_edit.toPlainText()
            )

    def _sync_region_loras(self):
        for card in self._region_cards.values():
            card.lora_widget.sync_to_model(self._model.augment)
