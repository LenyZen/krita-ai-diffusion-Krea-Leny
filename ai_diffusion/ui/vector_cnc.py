from __future__ import annotations

import asyncio
from pathlib import Path

from PyQt5.QtCore import QMetaObject, QPoint, Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..localization import translate as _
from ..model.jobs import Job, JobKind
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind
from ..model.root import root
from ..model.vector_cnc import (
    BackgroundMode,
    Complexity,
    Composition,
    EdgeStyle,
    VectorOutputMode,
    build_vector_prompt,
)
from ..vector_export import write_text_file
from . import theme
from .generation import HistoryWidget, ProgressBar
from .widget import ErrorBox, GenerateButton, QueueButton, StyleSelectWidget, WorkspaceSelectWidget


class VectorCncHistoryWidget(HistoryWidget):
    def _vector_widget(self) -> VectorCncWidget | None:
        widget = self.parent()
        while widget is not None and not isinstance(widget, VectorCncWidget):
            widget = widget.parent()
        return widget if isinstance(widget, VectorCncWidget) else None

    def _run_export(self, job: Job, index: int):
        path = self._model.vector_cnc.suggest_export_path(job, index, ".svg")
        filepath, _filter = QFileDialog.getSaveFileName(
            self, _("Export SVG"), str(path), "SVG (*.svg)"
        )
        if not filepath:
            return

        parent = self._vector_widget()
        if parent:
            parent._set_export_busy(True)

        async def task():
            try:
                content = await asyncio.to_thread(
                    self._model.vector_cnc.trace_result_svg, job, index
                )
                await asyncio.to_thread(write_text_file, Path(filepath), content)
            except Exception as e:
                self._model.report_error(str(e))
            finally:
                if parent:
                    parent._set_export_busy(False)

        eventloop.run(task())

    def _show_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        if item is None:
            return
        self.setCurrentItem(item)
        job_id, index = self.item_info(item)
        job = self._model.jobs.find(job_id)
        if job is None or job.kind is not JobKind.diffusion or index >= len(job.results):
            super()._show_context_menu(pos)
            return

        menu = QMenu(self)
        menu.addAction(_("Export SVG"), lambda: self._export_svg(job, index))
        menu.addAction(_("Add Vector Layer"), lambda: self._add_vector_layer(job, index))
        menu.addSeparator()
        menu.addAction(_("Copy Prompt"), self._copy_prompt)
        menu.addAction(_("Save Image"), self._save_image)
        menu.addAction(_("Discard Image"), self._discard_image)
        menu.exec(self.mapToGlobal(pos))

    def _export_svg(self, job: Job, index: int):
        self._run_export(job, index)

    def _add_vector_layer(self, job: Job, index: int):
        parent = self._vector_widget()
        if parent:
            parent._run_add_vector_layer(job, index)


class VectorCncWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []
        self._updating_preview = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 4, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)

        content = QWidget(scroll)
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.style_select = StyleSelectWidget(self)
        header = QHBoxLayout()
        header.addWidget(self.workspace_select)
        header.addWidget(self.style_select, 1)
        layout.addLayout(header)

        info = QLabel(
            _(
                "Generate flat, vector-friendly images for SVG tracing, CNC, and laser cutting. "
                "Select a result below to export SVG or add a vector layer in Krita."
            ),
            self,
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(info)

        mode_box = QGroupBox(_("Output mode"), self)
        mode_layout = QVBoxLayout(mode_box)
        self._mode_group = QButtonGroup(self)
        self._mode_silhouette = QRadioButton(_("Silhouette — laser stencils, vinyl cuts"), self)
        self._mode_line_art = QRadioButton(_("Line art — engraving, pen plotter"), self)
        self._mode_limited = QRadioButton(_("Limited color — layered laser/CNC (2–6 colors)"), self)
        for i, btn in enumerate(
            [self._mode_silhouette, self._mode_line_art, self._mode_limited]
        ):
            self._mode_group.addButton(btn, i)
            mode_layout.addWidget(btn)
        self._mode_silhouette.setChecked(True)
        layout.addWidget(mode_box)

        layout.addWidget(QLabel(_("What to draw"), self))
        self.description_edit = QPlainTextEdit(self)
        self.description_edit.setPlaceholderText(_("e.g. oak leaf wreath, geometric fox head"))
        self.description_edit.setMaximumHeight(72)
        layout.addWidget(self.description_edit)

        preview_label = QLabel(_("Prompt preview"), self)
        layout.addWidget(preview_label)
        self.prompt_preview = QPlainTextEdit(self)
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setMaximumHeight(100)
        self.prompt_preview.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(self.prompt_preview)

        toggles = QHBoxLayout()
        toggles.addWidget(self._make_option_group(_("Background"), self._background_widgets()))
        toggles.addWidget(self._make_option_group(_("Complexity"), self._complexity_widgets()))
        layout.addLayout(toggles)

        toggles2 = QHBoxLayout()
        toggles2.addWidget(self._make_option_group(_("Edge / style"), self._edge_widgets()))
        toggles2.addWidget(self._make_option_group(_("Composition"), self._composition_widgets()))
        layout.addLayout(toggles2)

        post_box = QGroupBox(_("Vector prep (post-process)"), self)
        post_layout = QVBoxLayout(post_box)
        self.post_contrast = QCheckBox(_("Boost contrast after generation"), self)
        self.post_posterize = QCheckBox(_("Posterize to limited colors"), self)
        self.post_threshold = QCheckBox(_("Threshold to pure black and white"), self)
        post_layout.addWidget(self.post_contrast)
        post_layout.addWidget(self.post_posterize)
        post_layout.addWidget(self.post_threshold)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel(_("Posterize colors:"), self))
        self.color_count = QComboBox(self)
        for n in (2, 3, 4, 6):
            self.color_count.addItem(str(n), n)
        self.color_count.setCurrentIndex(2)
        color_row.addWidget(self.color_count)
        color_row.addStretch()
        post_layout.addLayout(color_row)
        layout.addWidget(post_box)

        self.generate_button = GenerateButton(JobKind.diffusion, self)
        self.generate_button.operation = _("Generate")
        self.generate_button.clicked.connect(self._generate)
        self.generate_button.ctrl_clicked.connect(self._generate_replace)

        self.queue_button = QueueButton(parent=self)
        self.queue_button.setFixedHeight(self.generate_button.minimumSizeHint().height() - 2)

        gen_row = QHBoxLayout()
        gen_row.addWidget(self.generate_button, 1)
        gen_row.addWidget(self.queue_button)
        layout.addLayout(gen_row)

        self.progress_bar = ProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

        export_row = QHBoxLayout()
        self.export_svg_button = QPushButton(_("Export SVG"), self)
        self.export_svg_button.setToolTip(
            _("Trace the selected result and save as an SVG file for Inkscape or CAM tools.")
        )
        self.export_svg_button.clicked.connect(self._export_svg)
        self.add_vector_button = QPushButton(_("Add Vector Layer"), self)
        self.add_vector_button.setToolTip(
            _("Trace the selected result and insert it as a vector layer in this document.")
        )
        self.add_vector_button.clicked.connect(self._add_vector_layer)
        export_row.addWidget(self.export_svg_button)
        export_row.addWidget(self.add_vector_button)
        layout.addLayout(export_row)

        self.history = VectorCncHistoryWidget(self)
        self.history.item_activated.connect(self._apply_result)
        outer.addWidget(self.history, 1)

        self._wire_preview_updates()

    def _make_option_group(self, title: str, widgets: list[QWidget]) -> QGroupBox:
        box = QGroupBox(title, self)
        box_layout = QVBoxLayout(box)
        for w in widgets:
            box_layout.addWidget(w)
        return box

    def _background_widgets(self) -> list[QWidget]:
        self._bg_group = QButtonGroup(self)
        widgets = []
        options = [
            (_("Pure white"), BackgroundMode.pure_white),
            (_("Trace-friendly (white bg)"), BackgroundMode.transparent_style),
            (_("Solid black bg"), BackgroundMode.bg_black),
            (_("Solid white bg"), BackgroundMode.bg_white),
            (_("Single accent color bg"), BackgroundMode.bg_accent),
        ]
        for i, (label, mode) in enumerate(options):
            btn = QRadioButton(label, self)
            self._bg_group.addButton(btn, mode.value)
            btn.setProperty("vector_mode", mode)
            widgets.append(btn)
            if i == 0:
                btn.setChecked(True)
        return widgets

    def _complexity_widgets(self) -> list[QWidget]:
        self._complexity_group = QButtonGroup(self)
        widgets = []
        options = [
            (_("Simple (CNC-friendly)"), Complexity.simple),
            (_("Medium"), Complexity.medium),
            (_("Detailed (harder to vectorize)"), Complexity.detailed),
        ]
        for i, (label, mode) in enumerate(options):
            btn = QRadioButton(label, self)
            self._complexity_group.addButton(btn, mode.value)
            btn.setProperty("vector_mode", mode)
            widgets.append(btn)
            if i == 0:
                btn.setChecked(True)
        return widgets

    def _edge_widgets(self) -> list[QWidget]:
        self._edge_group = QButtonGroup(self)
        widgets = []
        options = [
            (_("Sharp clean edges"), EdgeStyle.sharp),
            (_("Bold outline / thick stroke"), EdgeStyle.bold_outline),
            (_("Symmetrical / decorative"), EdgeStyle.symmetrical),
        ]
        for i, (label, mode) in enumerate(options):
            btn = QRadioButton(label, self)
            self._edge_group.addButton(btn, mode.value)
            btn.setProperty("vector_mode", mode)
            widgets.append(btn)
            if i == 0:
                btn.setChecked(True)
        return widgets

    def _composition_widgets(self) -> list[QWidget]:
        self._composition_group = QButtonGroup(self)
        widgets = []
        options = [
            (_("Centered isolated subject"), Composition.centered),
            (_("Seamless pattern tile"), Composition.seamless_tile),
            (_("Border / safe margin"), Composition.border_margin),
        ]
        for i, (label, mode) in enumerate(options):
            btn = QRadioButton(label, self)
            self._composition_group.addButton(btn, mode.value)
            btn.setProperty("vector_mode", mode)
            widgets.append(btn)
            if i == 0:
                btn.setChecked(True)
        return widgets

    def _wire_preview_updates(self):
        self.description_edit.textChanged.connect(self._sync_description)
        self._mode_group.idClicked.connect(self._sync_output_mode)
        self._bg_group.idClicked.connect(self._sync_background)
        self._complexity_group.idClicked.connect(self._sync_complexity)
        self._edge_group.idClicked.connect(self._sync_edges)
        self._composition_group.idClicked.connect(self._sync_composition)
        self.post_contrast.toggled.connect(self._sync_post_contrast)
        self.post_posterize.toggled.connect(self._sync_post_posterize)
        self.post_threshold.toggled.connect(self._sync_post_threshold)
        self.color_count.currentIndexChanged.connect(self._on_color_count_changed)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            vc = model.vector_cnc
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "style", self.style_select, "value"),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                model.progress_changed.connect(self.progress_bar._update_progress),
                model.progress_kind_changed.connect(self.progress_bar._update_progress_kind),
            ]
            self.generate_button.model = model
            self.queue_button.model = model
            self.progress_bar.model = model
            self.history.model_ = model
            self._load_from_model()
            self._update_preview()
            self._update_export_buttons()
            self._model_bindings.append(
                model.jobs.selection_changed.connect(self._update_export_buttons)
            )

    def _selected_job(self) -> tuple[Job, int] | None:
        if not self._model.jobs.selection:
            return None
        item = self._model.jobs.selection[0]
        job = self._model.jobs.find(item.job)
        if job is None or job.kind is not JobKind.diffusion or item.image >= len(job.results):
            return None
        return job, item.image

    def _update_export_buttons(self):
        enabled = self._selected_job() is not None
        self.export_svg_button.setEnabled(enabled)
        self.add_vector_button.setEnabled(enabled)

    def _set_export_busy(self, busy: bool):
        self.export_svg_button.setEnabled(not busy and self._selected_job() is not None)
        self.add_vector_button.setEnabled(not busy and self._selected_job() is not None)
        if busy:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QGuiApplication.restoreOverrideCursor()

    def _run_add_vector_layer(self, job: Job, index: int):
        self._set_export_busy(True)

        async def task():
            try:
                svg = await asyncio.to_thread(self._model.vector_cnc.trace_result_svg, job, index)
                self._model.vector_cnc.add_vector_layer_from_svg(job, svg)
            except Exception as e:
                self._model.report_error(str(e))
            finally:
                self._set_export_busy(False)

        eventloop.run(task())

    def _export_svg(self):
        selected = self._selected_job()
        if not selected:
            return
        job, index = selected
        self.history._export_svg(job, index)

    def _add_vector_layer(self):
        selected = self._selected_job()
        if not selected:
            return
        job, index = selected
        self._run_add_vector_layer(job, index)

    def _load_from_model(self):
        vc = self._model.vector_cnc
        self._updating_preview = True
        self.description_edit.setPlainText(vc.description)
        self._set_mode_button(self._mode_group, vc.output_mode.value)
        self._set_mode_button(self._bg_group, vc.background.value)
        self._set_mode_button(self._complexity_group, vc.complexity.value)
        self._set_mode_button(self._edge_group, vc.edges.value)
        self._set_mode_button(self._composition_group, vc.composition.value)
        self.post_contrast.setChecked(vc.post_contrast)
        self.post_posterize.setChecked(vc.post_posterize)
        self.post_threshold.setChecked(vc.post_threshold)
        idx = self.color_count.findData(vc.color_count)
        if idx >= 0:
            self.color_count.setCurrentIndex(idx)
        self._updating_preview = False

    def _set_mode_button(self, group: QButtonGroup, value: int):
        if btn := group.button(value):
            btn.setChecked(True)

    def _sync_description(self):
        if self._updating_preview:
            return
        self._model.vector_cnc.description = self.description_edit.toPlainText()
        self._update_preview()

    def _sync_output_mode(self, mode_id: int):
        if self._updating_preview:
            return
        self._model.vector_cnc.output_mode = VectorOutputMode(mode_id)
        self._model.vector_cnc.apply_mode_defaults()
        self._load_from_model()
        self._update_preview()

    def _sync_background(self, mode_id: int):
        if self._updating_preview:
            return
        self._model.vector_cnc.background = BackgroundMode(mode_id)
        self._update_preview()

    def _sync_complexity(self, mode_id: int):
        if self._updating_preview:
            return
        self._model.vector_cnc.complexity = Complexity(mode_id)
        self._update_preview()

    def _sync_edges(self, mode_id: int):
        if self._updating_preview:
            return
        self._model.vector_cnc.edges = EdgeStyle(mode_id)
        self._update_preview()

    def _sync_composition(self, mode_id: int):
        if self._updating_preview:
            return
        self._model.vector_cnc.composition = Composition(mode_id)
        self._update_preview()

    def _sync_post_contrast(self, checked: bool):
        if self._updating_preview:
            return
        self._model.vector_cnc.post_contrast = checked

    def _sync_post_posterize(self, checked: bool):
        if self._updating_preview:
            return
        self._model.vector_cnc.post_posterize = checked

    def _sync_post_threshold(self, checked: bool):
        if self._updating_preview:
            return
        self._model.vector_cnc.post_threshold = checked

    def _on_color_count_changed(self):
        if self._updating_preview:
            return
        self._model.vector_cnc.color_count = int(self.color_count.currentData())
        self._update_preview()

    def _update_preview(self):
        vc = self._model.vector_cnc
        positive, negative = build_vector_prompt(
            self.description_edit.toPlainText(), vc.options()
        )
        self.prompt_preview.setPlainText(f"+ {positive}\n\n- {negative}")

    def _generate(self):
        self._model.vector_cnc.generate()

    def _generate_replace(self):
        self._model.vector_cnc.generate_replace()

    def _apply_result(self, item):
        job_id, index = self.history.item_info(item)
        self._model.apply_generated_result(job_id, index)
