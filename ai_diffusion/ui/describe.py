from krita import Krita
from PyQt5.QtCore import QMetaObject, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..localization import translate as _
from ..model.clio_styles import CLIO_STYLE_NONE, clio_style_names, clio_style_prompt
from ..model.describe_perspectives import Perspective
from ..model.jobs import JobKind
from ..model.model import DescribeLevel, DescribeProvider, DocumentModel
from ..model.properties import Bind, Binding, bind, bind_combo, bind_toggle
from ..model.root import root
from . import theme
from .widget import ErrorBox, GenerateButton, WorkspaceSelectWidget

_CLIO_NONE_LABEL = _("None (genre perspective only)")


class DescribeWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)
        self.setLayout(layout)

        self.workspace_select = WorkspaceSelectWidget(self)

        self.level_select = QComboBox(self)
        self.level_select.addItem(_("Tags (Danbooru style)"), DescribeLevel.tags)
        self.level_select.addItem(_("Brief caption"), DescribeLevel.brief)
        self.level_select.addItem(_("Detailed description"), DescribeLevel.detailed)
        self.level_select.addItem(_("Extensive description"), DescribeLevel.extensive)
        self.level_select.addItem(_("Mixed caption"), DescribeLevel.mixed)
        self.level_select.setToolTip(
            _("How detailed the generated image description should be.")
        )

        header_layout = QHBoxLayout()
        header_layout.addWidget(self.workspace_select)
        header_layout.addWidget(self.level_select, 1)
        layout.addLayout(header_layout)

        self.provider_select = QComboBox(self)
        self.provider_select.addItem(_("GGUF-VLM (Qwen2.5-VL GGUF)"), DescribeProvider.gguf_vlm)
        self.provider_select.addItem(_("Qwen-VL (ComfyUI-QwenVL)"), DescribeProvider.qwenvl)
        self.provider_select.addItem(_("Florence-2"), DescribeProvider.florence)
        self.provider_select.addItem(_("Auto"), DescribeProvider.auto)
        self.provider_select.setToolTip(
            _(
                "Vision backend for Describe / Enhance. "
                "GGUF-VLM uses ComfyUI-GGUF-VLM with models in LLM/GGUF "
                "(needs .gguf + matching mmproj)."
            )
        )
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel(_("Backend") + ":", self))
        provider_layout.addWidget(self.provider_select, 1)
        layout.addLayout(provider_layout)

        self.gguf_cuda_check = QCheckBox(_("Use CUDA for GGUF-VLM"), self)
        self.gguf_cuda_check.setToolTip(
            _(
                "When unchecked, GGUF-VLM Describe runs on CPU (no llama-cpp CUDA). "
                "Use this to A/B test speeds. Does not affect image Generate."
            )
        )
        layout.addWidget(self.gguf_cuda_check)

        self.perspective_select = QComboBox(self)
        self.perspective_select.addItem(_("No perspective (plain)"), Perspective.none)
        self.perspective_select.addItem(_("Cinematic film still"), Perspective.cinematic)
        self.perspective_select.addItem(_("Science fiction movie"), Perspective.sci_fi)
        self.perspective_select.addItem(_("Horror movie"), Perspective.horror)
        self.perspective_select.addItem(_("Comic book"), Perspective.comic_book)
        self.perspective_select.addItem(_("Anime"), Perspective.anime)
        self.perspective_select.addItem(_("Dark fantasy"), Perspective.dark_fantasy)
        self.perspective_select.addItem(_("Epic fantasy"), Perspective.epic_fantasy)
        self.perspective_select.addItem(_("Film noir"), Perspective.film_noir)
        self.perspective_select.addItem(_("Cyberpunk"), Perspective.cyberpunk)
        self.perspective_select.addItem(_("Documentary photo"), Perspective.documentary)
        self.perspective_select.addItem(_("Watercolor painting"), Perspective.watercolor)
        self.perspective_select.addItem(_("Classical oil painting"), Perspective.oil_painting)
        self.perspective_select.addItem(_("Storybook illustration"), Perspective.storybook)
        self.perspective_select.addItem(_("Steampunk"), Perspective.steampunk)
        self.perspective_select.addItem(_("Synthwave / vaporwave"), Perspective.synthwave)
        self.perspective_select.addItem(_("Western"), Perspective.western)
        self.perspective_select.addItem(_("Post-apocalyptic"), Perspective.post_apocalyptic)
        self.perspective_select.addItem(_("Ukiyo-e woodblock print"), Perspective.ukiyo_e)
        self.perspective_select.addItem(_("Surreal dream"), Perspective.surreal_dream)
        self.perspective_select.addItem(_("Vintage photograph"), Perspective.vintage_photo)
        self.perspective_select.addItem(
            _("Gerry Anderson / Supermarionation"), Perspective.gerry_anderson
        )
        self.perspective_select.addItem(_("Playboy magazine pictorial"), Perspective.playboy)
        self.perspective_select.setToolTip(
            _(
                "Optional genre framing. Combines with a Clio art style when one is selected. "
                "Great for prompt-hungry models like Krea 2."
            )
        )

        perspective_layout = QHBoxLayout()
        perspective_label = QLabel(_("Perspective") + ":", self)
        perspective_layout.addWidget(perspective_label)
        perspective_layout.addWidget(self.perspective_select, 1)
        layout.addLayout(perspective_layout)

        self.clio_names = [CLIO_STYLE_NONE] + clio_style_names()
        self.style_select = QComboBox(self)
        self.style_select.setEditable(True)
        self.style_select.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.style_select.setMaxVisibleItems(18)
        self.style_select.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.style_select.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.style_select.setMinimumContentsLength(12)
        self.style_select.addItem(_CLIO_NONE_LABEL, CLIO_STYLE_NONE)
        for name in clio_style_names():
            self.style_select.addItem(name, name)
        completer = QCompleter(self.clio_names[1:], self.style_select)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.style_select.setCompleter(completer)
        self.style_select.setToolTip(
            _(
                "Clio Style Library — 281 art-style prose lenses for Qwen-VL / Krea-ready prompts. "
                "Type to search. Requires ComfyUI-QwenVL."
            )
        )
        self.style_select.currentIndexChanged.connect(self._on_style_combo)
        self.style_select.lineEdit().editingFinished.connect(self._on_style_edited)

        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel(_("Style") + ":", self))
        style_layout.addWidget(self.style_select, 1)
        layout.addLayout(style_layout)

        self.style_preview = QLabel("", self)
        self.style_preview.setWordWrap(True)
        self.style_preview.setMaximumHeight(54)
        self.style_preview.setStyleSheet(f"color: {theme.grey}; font-size: 11px;")
        layout.addWidget(self.style_preview)

        self.custom_perspective_edit = QLineEdit(self)
        self.custom_perspective_edit.setPlaceholderText(
            _("Optional: your own perspective, e.g. 'a 1980s vaporwave album cover'")
        )
        self.custom_perspective_edit.setToolTip(
            _(
                "Describe the image through your own lens. Combines with perspective and Style above."
            )
        )
        self.custom_perspective_edit.setClearButtonEnabled(True)
        layout.addWidget(self.custom_perspective_edit)

        credit = QLabel(self)
        credit.setWordWrap(True)
        credit.setOpenExternalLinks(True)
        credit.setText(
            '<a href="https://github.com/lumenastrum/clio-style-preview">'
            "Clio Style Preview</a> — MIT code; community style text (u/Dear-Spend-2865). "
            + _("Used as optional art-style lenses in Describe.")
        )
        credit.setStyleSheet(f"color: {theme.grey}; font-size: 10px;")
        layout.addWidget(credit)

        info = QLabel(
            _("Analyzes the active layer and writes a text description of its content.")
            + " "
            + _(
                "Or type your own draft below and Enhance Description to expand it with Qwen-VL, "
                "using the active layer as visual context."
            )
            + " "
            + _(
                "Tip: ComfyUI-QwenVL unlocks Clio styles, perspective re-description, and Enhance; "
                "otherwise Describe falls back to Florence templates."
            ),
            self,
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {theme.grey};")
        layout.addWidget(info)

        layout.addWidget(QLabel(_("Your draft"), self))
        self.draft_edit = QPlainTextEdit(self)
        self.draft_edit.setPlaceholderText(
            _("Rough notes or prompt to expand… (used by Enhance Description)")
        )
        self.draft_edit.setMinimumHeight(64)
        self.draft_edit.setMaximumHeight(120)
        self.draft_edit.textChanged.connect(self._draft_edited)
        layout.addWidget(self.draft_edit)

        actions = QHBoxLayout()
        self.describe_button = GenerateButton(JobKind.describe, self)
        self.describe_button.operation = _("Describe Image")
        self.describe_button.clicked.connect(self.describe)
        self.enhance_button = QPushButton(_("Enhance Description"), self)
        self.enhance_button.setToolTip(
            _(
                "Expand your draft with Qwen-VL, grounded in the active layer image. "
                "Uses the same level and perspective settings as Describe."
            )
        )
        self.enhance_button.clicked.connect(self.enhance)
        actions.addWidget(self.describe_button, 1)
        actions.addWidget(self.enhance_button, 1)
        layout.addLayout(actions)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)

        layout.addWidget(QLabel(_("Result"), self))
        self.result_edit = QPlainTextEdit(self)
        self.result_edit.setPlaceholderText(
            _("The image description will appear here. You can edit it before using it.")
        )
        self.result_edit.textChanged.connect(self._text_edited)
        layout.addWidget(self.result_edit, 1)

        self.copy_prompt_button = QPushButton(theme.icon("copy-image"), _("Use as Prompt"), self)
        self.copy_prompt_button.setToolTip(
            _("Copy the description into the text prompt of the Generate workspace.")
        )
        self.copy_prompt_button.clicked.connect(self.use_as_prompt)

        self.copy_clipboard_button = QPushButton(
            Krita.instance().icon("edit-copy"), _("Copy"), self
        )
        self.copy_clipboard_button.setToolTip(_("Copy the description to the clipboard."))
        self.copy_clipboard_button.clicked.connect(self.copy_to_clipboard)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.copy_prompt_button, 1)
        buttons_layout.addWidget(self.copy_clipboard_button)
        layout.addLayout(buttons_layout)

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
                bind_combo(model.describe, "level", self.level_select),
                bind_combo(model.describe, "provider", self.provider_select),
                bind_toggle(model.describe, "gguf_use_cuda", self.gguf_cuda_check),
                bind_combo(model.describe, "perspective", self.perspective_select),
                bind(model.describe, "custom_perspective", self.custom_perspective_edit, "text"),
                bind(
                    model.describe, "can_generate", self.describe_button, "enabled", Bind.one_way
                ),
                bind(
                    model.describe, "can_generate", self.enhance_button, "enabled", Bind.one_way
                ),
                bind(model, "error", self.error_box, "error", Bind.one_way),
                model.describe.provider_changed.connect(self._update_gguf_cuda_visibility),
                model.describe.clio_style_changed.connect(self._sync_style_combo),
                model.describe.draft_text_changed.connect(self._update_draft),
                model.describe.result_text_changed.connect(self._update_text),
                model.progress_changed.connect(self.update_progress),
            ]
            self.describe_button.model = model
            self._update_gguf_cuda_visibility()
            self._sync_style_combo()
            self._update_draft()
            self._update_text()
            self.update_progress()

    def update_progress(self):
        self.progress_bar.setValue(int(max(self.model.progress, 0) * 100))

    def _update_gguf_cuda_visibility(self, *_args):
        provider = self.model.describe.provider
        show = provider in (DescribeProvider.gguf_vlm, DescribeProvider.auto)
        self.gguf_cuda_check.setVisible(show)
        self.gguf_cuda_check.setEnabled(show)

    def describe(self):
        self.model.describe.describe_image()

    def enhance(self):
        self.model.describe.enhance_description()

    def use_as_prompt(self):
        self.model.describe.use_as_prompt()

    def copy_to_clipboard(self):
        if clipboard := QApplication.clipboard():
            clipboard.setText(self.model.describe.result_text)

    def _resolve_style_name(self, text: str) -> str:
        text = (text or "").strip()
        if not text or text == _CLIO_NONE_LABEL:
            return CLIO_STYLE_NONE
        names = clio_style_names()
        if text in names:
            return text
        # Case-insensitive exact match from typed search
        lower = text.lower()
        for name in names:
            if name.lower() == lower:
                return name
        return CLIO_STYLE_NONE

    def _on_style_combo(self, _index: int = 0):
        name = self.style_select.currentData()
        if name is None:
            name = self._resolve_style_name(self.style_select.currentText())
        if name != self.model.describe.clio_style:
            self.model.describe.clio_style = name or CLIO_STYLE_NONE
        self._update_style_preview()

    def _on_style_edited(self):
        name = self._resolve_style_name(self.style_select.currentText())
        self.model.describe.clio_style = name
        self._sync_style_combo()

    def _sync_style_combo(self, *_args):
        name = self.model.describe.clio_style or CLIO_STYLE_NONE
        idx = self.style_select.findData(name)
        self.style_select.blockSignals(True)
        if idx >= 0:
            self.style_select.setCurrentIndex(idx)
        else:
            self.style_select.setCurrentIndex(0)
            if name:
                self.model.describe.clio_style = CLIO_STYLE_NONE
        self.style_select.blockSignals(False)
        self._update_style_preview()

    def _update_style_preview(self):
        name = self.model.describe.clio_style or CLIO_STYLE_NONE
        prompt = clio_style_prompt(name) if name else ""
        if not prompt:
            self.style_preview.setText("")
            return
        preview = prompt if len(prompt) <= 220 else prompt[:217] + "…"
        self.style_preview.setText(preview)

    def _update_draft(self):
        text = self.model.describe.draft_text
        if text != self.draft_edit.toPlainText():
            self.draft_edit.setPlainText(text)

    def _draft_edited(self):
        self.model.describe.draft_text = self.draft_edit.toPlainText()

    def _update_text(self):
        text = self.model.describe.result_text
        if text != self.result_edit.toPlainText():
            self.result_edit.setPlainText(text)

    def _text_edited(self):
        self.model.describe.result_text = self.result_edit.toPlainText()
