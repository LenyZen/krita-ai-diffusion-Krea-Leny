from __future__ import annotations

import asyncio
import random

from PyQt5.QtCore import QMetaObject, Qt
from PyQt5.QtGui import QGuiApplication, QPalette
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..localization import translate as _
from ..model.jobs import JobKind
from ..model.model import DocumentModel
from ..model.prompt_writer import (
    FIELD_LABELS,
    PREFERRED_MODEL,
    WRITER_CHOICES,
    list_models,
    prefer_model,
    write_prompt,
)
from ..model.properties import Bind, Binding, bind, bind_toggle
from ..model.randomizer import (
    DEFAULT_ITEMS,
    DIALOG_FIELDS,
    MAX_ALWAYS_LORAS,
    MAX_LORAS,
    append_missing,
    delete_lora_set,
    delete_prompt_set,
    load_lora_sets,
    load_prompt_sets,
    normalize_lora_stack,
    save_lora_set,
    save_prompt_set,
    split_items,
)
from ..model.root import root
from . import theme
from .generation import HistoryWidget, ProgressBar
from .widget import (
    ErrorBox,
    GenerateButton,
    QueueButton,
    StrengthWidget,
    StyleSelectWidget,
    WorkspaceSelectWidget,
)


class RandomizerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._model: DocumentModel = root.active_model
        self._model_bindings: list[QMetaObject.Connection | Binding] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 2, 0)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.style_select = StyleSelectWidget(self)
        style_layout = QHBoxLayout()
        style_layout.addWidget(self.workspace_select)
        style_layout.addWidget(self.style_select)
        layout.addLayout(style_layout)

        lora_row = QHBoxLayout()
        self.lora_button = QPushButton(_("Add LoRAs to the stack"), self)
        self.lora_button.clicked.connect(self._open_lora_stack)
        self.lora_save_button = QPushButton(_("Save"), self)
        self.lora_save_button.setToolTip(_("Save the current LoRA stack under a name."))
        self.lora_save_button.clicked.connect(self._save_lora_set)
        self.lora_load_button = QPushButton(_("Load"), self)
        self.lora_load_button.setToolTip(
            _("Load a saved LoRA stack. The prompt lists stay as they are.")
        )
        self.lora_load_button.clicked.connect(self._load_lora_set)
        lora_row.addWidget(self.lora_button, 1)
        lora_row.addWidget(self.lora_save_button)
        lora_row.addWidget(self.lora_load_button)
        layout.addLayout(lora_row)

        self.always_button = QPushButton(_("Always apply these LoRAs"), self)
        self.always_button.setToolTip(
            _(
                "These LoRAs are added on every Generate, along with the one LoRA "
                "picked from the stack."
            )
        )
        self.always_button.clicked.connect(self._open_always_loras)
        layout.addWidget(self.always_button)

        self.chooser_button = QPushButton(_("The Chooser"), self)
        self.chooser_button.clicked.connect(self._open_chooser)
        layout.addWidget(self.chooser_button)

        sets = QHBoxLayout()
        self.defaults_button = QPushButton(_("Populate defaults"), self)
        self.defaults_button.setToolTip(
            _("Add 20 starter items to each list. Existing text stays. LoRAs are left alone.")
        )
        self.defaults_button.clicked.connect(self._populate_defaults)
        self.save_button = QPushButton(_("Save"), self)
        self.save_button.setToolTip(_("Save these lists under a name, such as sci fi or old west."))
        self.save_button.clicked.connect(self._save_set)
        self.load_button = QPushButton(_("Load"), self)
        self.load_button.setToolTip(_("Load a saved prompt set into the lists. LoRAs stay as they are."))
        self.load_button.clicked.connect(self._load_set)
        sets.addWidget(self.defaults_button, 1)
        sets.addWidget(self.save_button)
        sets.addWidget(self.load_button)
        layout.addLayout(sets)

        write_row = QHBoxLayout()
        self.write_button = QPushButton(_("Write prompt"), self)
        self.write_button.setToolTip(
            _(
                "Pick one theme in each dropdown. Ollama expands them into a detailed image prompt."
            )
        )
        self.write_button.clicked.connect(self._open_writer)
        self.use_written_check = QCheckBox(_("Use written prompt"), self)
        self.use_written_check.setToolTip(
            _("Generate uses the Ollama prompt. Turn this off to go back to random list picks.")
        )
        write_row.addWidget(self.write_button, 1)
        write_row.addWidget(self.use_written_check)
        layout.addLayout(write_row)

        self.strength_slider = StrengthWidget(parent=self)
        layout.addWidget(self.strength_slider)

        self.preview_frame = QFrame(self)
        self.preview_frame.setObjectName("lastRoll")
        text = self.palette().color(QPalette.ColorRole.Text).name()
        self.preview_frame.setStyleSheet(
            f"""
            QFrame#lastRoll {{
                border: 1px solid {theme.highlight};
                border-radius: 4px;
                background-color: {theme.base};
            }}
            QFrame#lastRoll QLabel {{
                color: {text};
                background: transparent;
                border: none;
            }}
            QFrame#lastRoll QPlainTextEdit {{
                color: {text};
                background: transparent;
                border: none;
                selection-background-color: {theme.highlight};
                selection-color: {text};
            }}
            """
        )
        self.preview_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.preview_frame.setMinimumHeight(96)
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(8, 6, 8, 6)
        preview_layout.setSpacing(4)
        preview_header = QHBoxLayout()
        self.preview_title = QLabel(_("Last roll"), self.preview_frame)
        self.preview_copy = QPushButton(_("Copy"), self.preview_frame)
        self.preview_copy.setToolTip(_("Copy the text in this box."))
        self.preview_copy.clicked.connect(self._copy_preview)
        preview_header.addWidget(self.preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.preview_copy)
        preview_layout.addLayout(preview_header)
        self.preview = QPlainTextEdit(self.preview_frame)
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.preview.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.preview.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.preview.setMinimumHeight(72)
        self.preview.setToolTip(
            _("Drag the bar under this box to change its height. Ctrl+C copies the selection.")
        )
        preview_layout.addWidget(self.preview, 1)

        self.generate_button = GenerateButton(JobKind.diffusion, self)
        self.generate_button.operation = _("Generate")
        self.generate_button.setIcon(theme.icon("workspace-generation"))
        self.queue_button = QueueButton(parent=self)
        self.queue_button.setFixedHeight(self.generate_button.height() - 2)
        actions = QHBoxLayout()
        actions.addWidget(self.generate_button, 1)
        actions.addWidget(self.queue_button)

        self.progress_bar = ProgressBar(self)
        self.error_box = ErrorBox(self)
        self.history = HistoryWidget(self)
        self.history.item_activated.connect(self._apply_result)

        self._lower = QWidget(self)
        lower_layout = QVBoxLayout(self._lower)
        lower_layout.setContentsMargins(0, 6, 0, 0)
        lower_layout.setSpacing(0)
        lower_layout.addLayout(actions)
        lower_layout.addWidget(self.progress_bar)
        lower_layout.addWidget(self.error_box)
        lower_layout.addWidget(self.history, 1)

        self._splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(8)
        self._splitter.addWidget(self.preview_frame)
        self._splitter.addWidget(self._lower)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([160, 480])
        layout.addSpacing(10)
        layout.addWidget(self._splitter, 1)

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
                bind(model, "strength", self.strength_slider, "value"),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                bind_toggle(model.randomizer, "use_written", self.use_written_check),
                self.generate_button.clicked.connect(model.generate),
                self.generate_button.ctrl_clicked.connect(model.generate_replace),
                model.randomizer.modified.connect(self._refresh_summary),
            ]
            self.generate_button.model = model
            self.queue_button.model = model
            self.progress_bar.model = model
            self.strength_slider.model = model
            self.history.model_ = model
            self._refresh_summary()

    def _apply_result(self, item):
        job_id, index = self.history.item_info(item)
        self.model.apply_generated_result(job_id, index)

    def _refresh_summary(self, *_args):
        rand = self._model.randomizer
        counts = rand.item_counts()
        filled = sum(1 for count in counts.values() if count)
        total = sum(counts.values())
        self.lora_button.setText(
            _("Add LoRAs to the stack") + f"  ({len(rand.lora_stack)}/{MAX_LORAS})"
        )
        self.always_button.setText(
            _("Always apply these LoRAs") + f"  ({len(rand.fixed_loras)}/{MAX_ALWAYS_LORAS})"
        )
        self.chooser_button.setText(_("The Chooser") + f"  ({total} items, {filled} lists)")
        if rand.last_prompt:
            lora = rand.last_lora or _("no extra LoRA")
            if rand.last_lora:
                lora = f"{rand.last_lora} @ {rand.last_lora_strength:g}"
            title = _("Written prompt") if rand.use_written else _("Last roll")
            self.preview_title.setText(title)
            lines = [rand.last_prompt, f"LoRA: {lora}"]
            always = _format_fixed_loras(rand.fixed_loras)
            if always:
                lines.append(f"Always: {always}")
            self._set_preview_text("\n".join(lines))
        else:
            self.preview_title.setText(_("Last roll"))
            self._set_preview_text(
                _(
                    "Each Generate picks one item from every list that has text, "
                    "and one LoRA from the stack. LoRAs in Always apply are added "
                    "on every image. Style sampler LoRAs still apply."
                )
            )

    def _set_preview_text(self, text: str):
        if self.preview.toPlainText() == text:
            return
        scrollbar = self.preview.verticalScrollBar()
        position = scrollbar.value()
        self.preview.setPlainText(text)
        scrollbar.setValue(position)

    def _copy_preview(self):
        text = self.preview.textCursor().selectedText() or self.preview.toPlainText()
        text = text.replace("\u2029", "\n")
        if clipboard := QGuiApplication.clipboard():
            clipboard.setText(text)

    def _open_writer(self):
        dialog = PromptWriterDialog(self._model.randomizer, self)
        dialog.exec_()
        self._refresh_summary()

    def _open_chooser(self):
        dialog = ChooserDialog(self._model.randomizer, self)
        dialog.exec_()
        self._refresh_summary()

    def _open_lora_stack(self):
        names = _server_loras()
        dialog = LoraStackDialog(self._model.randomizer, names, self)
        dialog.exec_()
        self._refresh_summary()

    def _open_always_loras(self):
        names = _server_loras()
        dialog = AlwaysLoraDialog(self._model.randomizer, names, self)
        dialog.exec_()
        self._refresh_summary()

    def _save_lora_set(self):
        ask_save_lora_set(self, list(self._model.randomizer.lora_stack))

    def _load_lora_set(self):
        stack = ask_load_lora_set(self)
        if stack is not None:
            self._model.randomizer.lora_stack = stack
            self._refresh_summary()

    def _populate_defaults(self):
        self._model.randomizer.populate_defaults()
        self._refresh_summary()

    def _save_set(self):
        ask_save_prompt_set(self, self._model.randomizer.values())

    def _load_set(self):
        values = ask_load_prompt_set(self)
        if values is not None:
            self._model.randomizer.apply_values(values)
            self._refresh_summary()


def _server_loras() -> list[str]:
    client = root.connection.client_if_connected
    if client is None:
        return []
    return sorted(client.models.loras, key=str.lower)


class ChooserDialog(QDialog):
    def __init__(self, randomizer, parent=None):
        super().__init__(parent)
        self._randomizer = randomizer
        self.setWindowTitle(_("The Chooser"))
        self.resize(520, 640)

        layout = QVBoxLayout(self)
        hint = QLabel(
            _(
                "Separate items with commas. Generate uses one item from each box "
                "and stitches them as subject, wearing, action, place, mood, shot, extras."
            ),
            self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        sets = QHBoxLayout()
        defaults = QPushButton(_("Populate defaults"), self)
        defaults.clicked.connect(self._populate_defaults)
        save = QPushButton(_("Save"), self)
        save.clicked.connect(self._save_set)
        load = QPushButton(_("Load"), self)
        load.clicked.connect(self._load_set)
        sets.addWidget(defaults, 1)
        sets.addWidget(save)
        sets.addWidget(load)
        layout.addLayout(sets)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QVBoxLayout(body)
        self._edits: dict[str, QPlainTextEdit] = {}
        for name, title, help_text, placeholder in DIALOG_FIELDS:
            label = QLabel(f"{title}  —  {help_text}", body)
            label.setWordWrap(True)
            edit = QPlainTextEdit(body)
            edit.setPlaceholderText(placeholder)
            edit.setPlainText(randomizer.field_text(name))
            edit.setFixedHeight(72)
            count = QLabel(body)
            edit.textChanged.connect(lambda _c=count, _e=edit: _set_count(_c, _e.toPlainText()))
            _set_count(count, edit.toPlainText())
            form.addWidget(label)
            form.addWidget(edit)
            form.addWidget(count)
            self._edits[name] = edit
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _editor_values(self) -> dict[str, str]:
        return {name: edit.toPlainText() for name, edit in self._edits.items()}

    def _populate_defaults(self):
        for name, edit in self._edits.items():
            edit.setPlainText(append_missing(edit.toPlainText(), DEFAULT_ITEMS[name]))

    def _save_set(self):
        values = self._editor_values()
        if ask_save_prompt_set(self, values):
            self._randomizer.apply_values(values)

    def _load_set(self):
        values = ask_load_prompt_set(self)
        if values is None:
            return
        for name, edit in self._edits.items():
            edit.setPlainText(values.get(name, ""))

    def _save(self):
        self._randomizer.apply_values(self._editor_values())
        self.accept()


class LoraStackDialog(QDialog):
    def __init__(self, randomizer, names: list[str], parent=None):
        super().__init__(parent)
        self._randomizer = randomizer
        self.setWindowTitle(_("Add LoRAs to the stack"))
        self.resize(480, 420)

        layout = QVBoxLayout(self)
        hint = QLabel(
            _(
                "Add up to {n} LoRAs. Each Generate picks one of them at the strength you set. "
                "LoRAs saved on the style are not stacked on top of this pick.",
                n=MAX_LORAS,
            ),
            self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._list = QListWidget(self)
        layout.addWidget(self._list, 1)
        for item in randomizer.lora_stack:
            name = str(item.get("name", "")).strip()
            if name:
                self._add_row(name, float(item.get("strength", 1.0)))

        row = QHBoxLayout()
        self._combo = QComboBox(self)
        self._combo.setEditable(True)
        self._combo.addItems(names)
        self._combo.setMinimumWidth(220)
        self._strength = QDoubleSpinBox(self)
        self._strength.setRange(0.0, 2.0)
        self._strength.setSingleStep(0.05)
        self._strength.setValue(1.0)
        self._strength.setPrefix(_("Strength") + " ")
        add = QPushButton(_("Add"), self)
        add.clicked.connect(self._add_current)
        remove = QPushButton(_("Remove"), self)
        remove.clicked.connect(self._remove_current)
        row.addWidget(self._combo, 1)
        row.addWidget(self._strength)
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)

        sets = QHBoxLayout()
        save = QPushButton(_("Save set"), self)
        save.setToolTip(_("Save the LoRAs in this window under a name."))
        save.clicked.connect(self._save_set)
        load = QPushButton(_("Load set"), self)
        load.setToolTip(_("Replace the list in this window with a saved LoRA set."))
        load.clicked.connect(self._load_set)
        sets.addStretch(1)
        sets.addWidget(save)
        sets.addWidget(load)
        layout.addLayout(sets)

        self._status = QLabel(self)
        layout.addWidget(self._status)
        self._update_status()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_row(self, name: str, strength: float):
        item = QListWidgetItem(f"{name}   @ {strength:g}")
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setData(Qt.ItemDataRole.UserRole + 1, strength)
        self._list.addItem(item)

    def _add_current(self):
        if self._list.count() >= MAX_LORAS:
            self._status.setText(_("The stack already has {n} LoRAs.", n=MAX_LORAS))
            return
        name = self._combo.currentText().strip()
        if not name:
            return
        self._add_row(name, float(self._strength.value()))
        self._update_status()

    def _remove_current(self):
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)
            self._update_status()

    def _update_status(self):
        self._status.setText(f"{self._list.count()} / {MAX_LORAS}")

    def _stack_from_list(self) -> list[dict]:
        stack = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            strength = item.data(Qt.ItemDataRole.UserRole + 1)
            if name:
                stack.append(
                    {"name": name, "strength": float(strength if strength is not None else 1.0)}
                )
        return stack

    def _replace_list(self, stack: list[dict]):
        self._list.clear()
        for item in normalize_lora_stack(stack):
            self._add_row(item["name"], float(item["strength"]))
        self._update_status()

    def _save_set(self):
        stack = self._stack_from_list()
        if ask_save_lora_set(self, stack):
            self._randomizer.lora_stack = stack

    def _load_set(self):
        stack = ask_load_lora_set(self)
        if stack is not None:
            self._replace_list(stack)

    def _save(self):
        self._randomizer.lora_stack = self._stack_from_list()
        self.accept()


def _format_fixed_loras(stack) -> str:
    parts = []
    for item in stack or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            strength = float(item.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        parts.append(f"{name} @ {strength:g}")
    return ", ".join(parts)


class AlwaysLoraDialog(QDialog):
    def __init__(self, randomizer, names: list[str], parent=None):
        super().__init__(parent)
        self._randomizer = randomizer
        self.setWindowTitle(_("Always apply these LoRAs"))
        self.resize(480, 420)

        layout = QVBoxLayout(self)
        hint = QLabel(
            _(
                "Every LoRA in this list is applied on every Generate, at the strength you set. "
                "The stack still picks one extra LoRA at random. Up to {n}.",
                n=MAX_ALWAYS_LORAS,
            ),
            self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._list = QListWidget(self)
        layout.addWidget(self._list, 1)
        for item in normalize_lora_stack(list(randomizer.fixed_loras), MAX_ALWAYS_LORAS):
            self._add_row(item["name"], float(item["strength"]))

        row = QHBoxLayout()
        self._combo = QComboBox(self)
        self._combo.setEditable(True)
        self._combo.addItems(names)
        self._combo.setMinimumWidth(220)
        self._strength = QDoubleSpinBox(self)
        self._strength.setRange(0.0, 2.0)
        self._strength.setSingleStep(0.05)
        self._strength.setValue(1.0)
        self._strength.setPrefix(_("Strength") + " ")
        add = QPushButton(_("Add"), self)
        add.clicked.connect(self._add_current)
        remove = QPushButton(_("Remove"), self)
        remove.clicked.connect(self._remove_current)
        row.addWidget(self._combo, 1)
        row.addWidget(self._strength)
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)

        self._status = QLabel(self)
        layout.addWidget(self._status)
        self._update_status()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_row(self, name: str, strength: float):
        item = QListWidgetItem(f"{name}   @ {strength:g}")
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setData(Qt.ItemDataRole.UserRole + 1, strength)
        self._list.addItem(item)

    def _add_current(self):
        if self._list.count() >= MAX_ALWAYS_LORAS:
            self._status.setText(_("The list already has {n} LoRAs.", n=MAX_ALWAYS_LORAS))
            return
        name = self._combo.currentText().strip()
        if not name:
            return
        self._add_row(name, float(self._strength.value()))
        self._update_status()

    def _remove_current(self):
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)
            self._update_status()

    def _update_status(self):
        self._status.setText(f"{self._list.count()} / {MAX_ALWAYS_LORAS}")

    def _save(self):
        stack = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            strength = item.data(Qt.ItemDataRole.UserRole + 1)
            if name:
                stack.append(
                    {"name": name, "strength": float(strength if strength is not None else 1.0)}
                )
        self._randomizer.fixed_loras = normalize_lora_stack(stack, MAX_ALWAYS_LORAS)
        self.accept()


def _set_count(label: QLabel, text: str):
    count = len(split_items(text))
    label.setText(_("Items") + f": {count}")


def ask_save_prompt_set(parent, values: dict[str, str]) -> bool:
    name, ok = QInputDialog.getText(
        parent, _("Save prompt set"), _("Save these lists as"), text=""
    )
    if not ok:
        return False
    title = " ".join(name.split())
    if not title:
        QMessageBox.warning(parent, _("Save prompt set"), _("Enter a name for this prompt set."))
        return False
    if title in load_prompt_sets():
        answer = QMessageBox.question(
            parent,
            _("Save prompt set"),
            _('Replace the saved set "{name}"?', name=title),
        )
        if answer != QMessageBox.Yes:
            return False
    save_prompt_set(title, values)
    return True


def ask_load_prompt_set(parent) -> dict[str, str] | None:
    dialog = LoadPromptSetDialog(parent)
    if dialog.exec_():
        return dialog.values
    return None


def ask_save_lora_set(parent, stack: list[dict]) -> bool:
    name, ok = QInputDialog.getText(
        parent, _("Save LoRA set"), _("Save this LoRA stack as"), text=""
    )
    if not ok:
        return False
    title = " ".join(name.split())
    if not title:
        QMessageBox.warning(parent, _("Save LoRA set"), _("Enter a name for this LoRA set."))
        return False
    if title in load_lora_sets():
        answer = QMessageBox.question(
            parent,
            _("Save LoRA set"),
            _('Replace the saved LoRA set "{name}"?', name=title),
        )
        if answer != QMessageBox.Yes:
            return False
    save_lora_set(title, stack)
    return True


def ask_load_lora_set(parent) -> list[dict] | None:
    dialog = LoadLoraSetDialog(parent)
    if dialog.exec_():
        return dialog.stack
    return None


class LoadPromptSetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: dict[str, str] | None = None
        self.setWindowTitle(_("Load prompt set"))
        self.resize(360, 320)

        layout = QVBoxLayout(self)
        self._empty = QLabel(_("No saved prompt sets yet."), self)
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty)
        self._list = QListWidget(self)
        self._list.itemDoubleClicked.connect(lambda _item: self._load())
        layout.addWidget(self._list, 1)

        row = QHBoxLayout()
        load = QPushButton(_("Load"), self)
        load.clicked.connect(self._load)
        remove = QPushButton(_("Delete"), self)
        remove.clicked.connect(self._delete)
        close = QPushButton(_("Close"), self)
        close.clicked.connect(self.reject)
        row.addWidget(load)
        row.addWidget(remove)
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        self._refresh()

    def _refresh(self):
        self._sets = load_prompt_sets()
        self._list.clear()
        for name in sorted(self._sets, key=str.lower):
            self._list.addItem(name)
        has_sets = self._list.count() > 0
        self._list.setVisible(has_sets)
        self._empty.setVisible(not has_sets)

    def _selected(self) -> str:
        item = self._list.currentItem()
        return item.text() if item is not None else ""

    def _load(self):
        name = self._selected()
        if not name or name not in self._sets:
            return
        self.values = self._sets[name]
        self.accept()

    def _delete(self):
        name = self._selected()
        if not name:
            return
        answer = QMessageBox.question(
            self, _("Delete prompt set"), _('Delete "{name}"?', name=name)
        )
        if answer != QMessageBox.Yes:
            return
        delete_prompt_set(name)
        self._refresh()


class LoadLoraSetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stack: list[dict] | None = None
        self.setWindowTitle(_("Load LoRA set"))
        self.resize(360, 320)

        layout = QVBoxLayout(self)
        self._empty = QLabel(_("No saved LoRA sets yet."), self)
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty)
        self._list = QListWidget(self)
        self._list.itemDoubleClicked.connect(lambda _item: self._load())
        layout.addWidget(self._list, 1)

        row = QHBoxLayout()
        load = QPushButton(_("Load"), self)
        load.clicked.connect(self._load)
        remove = QPushButton(_("Delete"), self)
        remove.clicked.connect(self._delete)
        close = QPushButton(_("Close"), self)
        close.clicked.connect(self.reject)
        row.addWidget(load)
        row.addWidget(remove)
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        self._refresh()

    def _refresh(self):
        self._sets = load_lora_sets()
        self._list.clear()
        for name in sorted(self._sets, key=str.lower):
            count = len(self._sets[name])
            self._list.addItem(f"{name}  ({count})")
            self._list.item(self._list.count() - 1).setData(Qt.ItemDataRole.UserRole, name)
        has_sets = self._list.count() > 0
        self._list.setVisible(has_sets)
        self._empty.setVisible(not has_sets)

    def _selected(self) -> str:
        item = self._list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _load(self):
        name = self._selected()
        if not name or name not in self._sets:
            return
        self.stack = self._sets[name]
        self.accept()

    def _delete(self):
        name = self._selected()
        if not name:
            return
        answer = QMessageBox.question(self, _("Delete LoRA set"), _('Delete "{name}"?', name=name))
        if answer != QMessageBox.Yes:
            return
        delete_lora_set(name)
        self._refresh()


class PromptWriterDialog(QDialog):
    def __init__(self, randomizer, parent=None):
        super().__init__(parent)
        self._randomizer = randomizer
        self._busy = False
        self.setWindowTitle(_("Write prompt"))
        self.resize(560, 720)

        layout = QVBoxLayout(self)
        hint = QLabel(
            _(
                "Each dropdown is one theme. Ollama writes a detailed image prompt "
                "that keeps those themes and fills in the visual detail."
            ),
            self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel(_("Ollama model"), self))
        self._model_combo = QComboBox(self)
        self._model_combo.setEditable(True)
        self._model_combo.setMinimumWidth(220)
        saved = randomizer.ollama_model or PREFERRED_MODEL
        self._model_combo.addItem(saved, saved)
        self._model_combo.setCurrentText(saved)
        refresh = QPushButton(_("Refresh"), self)
        refresh.clicked.connect(self._refresh_models)
        model_row.addWidget(self._model_combo, 1)
        model_row.addWidget(refresh)
        layout.addLayout(model_row)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QVBoxLayout(body)
        self._combos: dict[str, QComboBox] = {}
        picks = randomizer.writer_picks()
        for key, label in FIELD_LABELS:
            form.addWidget(QLabel(label, body))
            combo = QComboBox(body)
            combo.addItems(WRITER_CHOICES[key])
            combo.setMaxVisibleItems(20)
            current = picks.get(key) or ""
            index = combo.findText(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            form.addWidget(combo)
            self._combos[key] = combo
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        tools = QHBoxLayout()
        surprise = QPushButton(_("Random themes"), self)
        surprise.clicked.connect(self._randomize)
        self._idea_button = QPushButton(_("Free idea"), self)
        self._idea_button.clicked.connect(self._open_idea)
        self._write_button = QPushButton(_("Ask Ollama"), self)
        self._write_button.clicked.connect(self._ask)
        tools.addWidget(surprise)
        tools.addWidget(self._idea_button)
        tools.addWidget(self._write_button, 1)
        layout.addLayout(tools)

        self._idea = randomizer.writer_idea
        self._idea_label = QLabel(self)
        self._idea_label.setWordWrap(True)
        self._update_idea_label()
        layout.addWidget(self._idea_label)

        self._with_themes = QCheckBox(_("Include dropdown themes"), self)
        self._with_themes.setChecked(bool(randomizer.writer_with_themes))
        self._with_themes.setToolTip(
            _(
                "Checked: Ollama gets the free idea and the dropdown themes. "
                "Unchecked: Ollama gets only the free idea."
            )
        )
        layout.addWidget(self._with_themes)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._result = QPlainTextEdit(self)
        self._result.setPlaceholderText(_("The written prompt appears here."))
        self._result.setPlainText(randomizer.written_prompt)
        self._result.setMinimumHeight(120)
        layout.addWidget(self._result)

        buttons = QHBoxLayout()
        use = QPushButton(_("Use this prompt"), self)
        use.clicked.connect(self._use)
        copy = QPushButton(_("Copy"), self)
        copy.clicked.connect(self._copy)
        close = QPushButton(_("Close"), self)
        close.clicked.connect(self._store_picks)
        close.clicked.connect(self.accept)
        buttons.addWidget(use)
        buttons.addWidget(copy)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self._refresh_models()

    def _picks(self) -> dict[str, str]:
        return {key: combo.currentText() for key, combo in self._combos.items()}

    def _store_picks(self):
        rand = self._randomizer
        picks = self._picks()
        rand.writer_subject = picks["subject"]
        rand.writer_place = picks["place"]
        rand.writer_wearing = picks["wearing"]
        rand.writer_action = picks["action"]
        rand.writer_mood = picks["mood"]
        rand.writer_shot = picks["shot"]
        rand.writer_extras = picks["extras"]
        rand.ollama_model = self._model_combo.currentText().strip() or PREFERRED_MODEL
        rand.writer_idea = self._idea
        rand.writer_with_themes = self._with_themes.isChecked()

    def _randomize(self):
        for combo in self._combos.values():
            if combo.count():
                combo.setCurrentIndex(random.randrange(combo.count()))

    def _refresh_models(self):
        self._status.setText(_("Looking for Ollama models..."))
        task = eventloop.run(self._fetch_models())
        task.add_done_callback(self._models_ready)

    async def _fetch_models(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, list_models)

    def _models_ready(self, task):
        try:
            names = task.result()
        except Exception as exc:
            self._status.setText(str(exc))
            return
        current = self._model_combo.currentText().strip()
        self._model_combo.clear()
        for name in names:
            self._model_combo.addItem(name, name)
        chosen = current if current in names else prefer_model(names)
        index = self._model_combo.findText(chosen)
        if index >= 0:
            self._model_combo.setCurrentIndex(index)
        elif chosen:
            self._model_combo.setCurrentText(chosen)
        self._status.setText(_("Model list updated."))

    def _ask(self):
        if self._busy:
            return
        self._store_picks()
        self._busy = True
        self._write_button.setEnabled(False)
        model = self._model_combo.currentText().strip() or PREFERRED_MODEL
        picks = self._picks()
        idea = self._idea.strip()
        include_themes = self._with_themes.isChecked() or not idea
        if idea and not self._with_themes.isChecked():
            self._status.setText(_("Writing from the free idea only..."))
        elif idea:
            self._status.setText(_("Writing from the free idea and the dropdown themes..."))
        else:
            self._status.setText(_("Writing from the dropdown themes..."))
        task = eventloop.run(self._run_write(model, picks, idea, include_themes))
        task.add_done_callback(self._write_ready)

    async def _run_write(self, model: str, picks: dict[str, str], idea: str, include_themes: bool):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: write_prompt(model, picks, idea, include_themes)
        )

    def _write_ready(self, task):
        self._busy = False
        self._write_button.setEnabled(True)
        try:
            text = task.result()
        except Exception as exc:
            self._status.setText(str(exc))
            return
        self._result.setPlainText(text)
        self._status.setText(_("Prompt ready. Use this prompt to send it to Generate."))

    def _use(self):
        text = self._result.toPlainText().strip()
        if not text:
            self._status.setText(_("Ask Ollama before using a prompt."))
            return
        self._store_picks()
        self._randomizer.written_prompt = text
        self._randomizer.use_written = True
        self._randomizer.last_prompt = text
        self._status.setText(_("Generate will use this prompt until you turn Use written prompt off."))

    def _open_idea(self):
        dialog = FreeIdeaDialog(self._idea, self)
        if dialog.exec_():
            self._idea = dialog.idea
            self._randomizer.writer_idea = self._idea
            self._update_idea_label()

    def _update_idea_label(self):
        idea = " ".join(self._idea.split())
        if not idea:
            self._idea_label.setText(_("No free idea yet. Open Free idea to type one."))
            return
        if len(idea) > 140:
            idea = idea[:137] + "..."
        self._idea_label.setText(_("Free idea") + f": {idea}")

    def _copy(self):
        from PyQt5.QtGui import QGuiApplication

        if clipboard := QGuiApplication.clipboard():
            clipboard.setText(self._result.toPlainText())


class FreeIdeaDialog(QDialog):
    def __init__(self, idea: str, parent=None):
        super().__init__(parent)
        self.idea = idea
        self.setWindowTitle(_("Free idea"))
        self.resize(480, 280)

        layout = QVBoxLayout(self)
        hint = QLabel(
            _(
                "Type a little idea or two. Ask Ollama will turn it into a detailed image prompt. "
                "Include dropdown themes is on the Write prompt window if you also want those themes sent."
            ),
            self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._edit = QPlainTextEdit(self)
        self._edit.setPlainText(idea)
        self._edit.setPlaceholderText(
            _("a rainy goodbye on a train platform, her coat too big, his ticket still in his hand")
        )
        layout.addWidget(self._edit, 1)

        row = QHBoxLayout()
        use = QPushButton(_("Use this idea"), self)
        use.clicked.connect(self._accept)
        cancel = QPushButton(_("Cancel"), self)
        cancel.clicked.connect(self.reject)
        row.addWidget(use)
        row.addStretch(1)
        row.addWidget(cancel)
        layout.addLayout(row)

    def _accept(self):
        self.idea = self._edit.toPlainText()
        self.accept()
