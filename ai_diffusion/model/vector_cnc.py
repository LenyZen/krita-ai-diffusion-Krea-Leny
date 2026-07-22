from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage, qAlpha, qBlue, qGray, qGreen, qRed, qRgba

from ..image import Image, ImageCollection
from ..localization import translate as _
from ..util import clamp, ensure, find_unused_path, sanitize_prompt
from ..vector_export import trace_image_to_svg, write_text_file
from .jobs import Job
from .properties import ObservableProperties, Property

CORE_NEGATIVE = (
    "photorealistic, gradient, soft shading, blur, noise, texture, dithering, "
    "jpeg artifacts, complex background, depth of field, bokeh, 3d render"
)


class VectorOutputMode(Enum):
    silhouette = 0
    line_art = 1
    limited_color = 2


class BackgroundMode(Enum):
    pure_white = 0
    transparent_style = 1
    bg_black = 2
    bg_white = 3
    bg_accent = 4


class Complexity(Enum):
    simple = 0
    medium = 1
    detailed = 2


class EdgeStyle(Enum):
    sharp = 0
    bold_outline = 1
    symmetrical = 2


class Composition(Enum):
    centered = 0
    seamless_tile = 1
    border_margin = 2


@dataclass
class VectorCncOptions:
    mode: VectorOutputMode
    background: BackgroundMode
    complexity: Complexity
    edges: EdgeStyle
    composition: Composition
    color_count: int
    post_contrast: bool
    post_posterize: bool
    post_threshold: bool


_MODE_POSITIVE: dict[VectorOutputMode, str] = {
    VectorOutputMode.silhouette: (
        "solid black silhouette shape on pure white background, stencil art, "
        "vinyl cut, laser cut stencil, bold simple outline, single piece cutout"
    ),
    VectorOutputMode.line_art: (
        "clean continuous line art drawing, stroke only, no fills, no shading, "
        "pen plotter style, engraving line art, single-weight outlines"
    ),
    VectorOutputMode.limited_color: (
        "flat color regions with hard edges, poster art, screen print style, "
        "distinct color areas, no gradients, vector illustration"
    ),
}

_MODE_NEGATIVE: dict[VectorOutputMode, str] = {
    VectorOutputMode.silhouette: "internal details, cross-hatching, fine texture, holes, gray tones",
    VectorOutputMode.line_art: "filled areas, solid blocks of color, hatching, cross-hatching, shading",
    VectorOutputMode.limited_color: "gradients, soft edges, color blending, airbrush, painterly",
}

_BACKGROUND: dict[BackgroundMode, str] = {
    BackgroundMode.pure_white: "pure white background",
    BackgroundMode.transparent_style: (
        "plain white background, high contrast subject suitable for bitmap tracing"
    ),
    BackgroundMode.bg_black: "solid black background",
    BackgroundMode.bg_white: "solid white background",
    BackgroundMode.bg_accent: "single flat accent color background",
}

_COMPLEXITY: dict[Complexity, str] = {
    Complexity.simple: "simple shapes, minimal detail, CNC friendly, few elements",
    Complexity.medium: "moderate detail, clear readable forms",
    Complexity.detailed: "intricate detail, fine features, decorative complexity",
}

_EDGES: dict[EdgeStyle, str] = {
    EdgeStyle.sharp: "sharp clean edges, crisp outlines",
    EdgeStyle.bold_outline: "bold thick outline, strong stroke weight",
    EdgeStyle.symmetrical: "symmetrical composition, mirror balanced decorative design",
}

_COMPOSITION: dict[Composition, str] = {
    Composition.centered: "centered isolated subject, plenty of margin, single focal object",
    Composition.seamless_tile: "seamless repeating pattern tile, wallpaper texture",
    Composition.border_margin: "safe margin border, frame inset, laser bed margin",
}


def build_vector_prompt(description: str, opts: VectorCncOptions) -> tuple[str, str]:
    subject = description.strip()
    parts = [_MODE_POSITIVE[opts.mode]]
    if subject:
        parts.append(subject)
    parts.extend(
        [
            _BACKGROUND[opts.background],
            _COMPLEXITY[opts.complexity],
            _EDGES[opts.edges],
            _COMPOSITION[opts.composition],
        ]
    )
    if opts.mode is VectorOutputMode.limited_color:
        parts.append(f"{opts.color_count} distinct flat colors maximum")
    positive = ", ".join(parts)
    negative = f"{CORE_NEGATIVE}, {_MODE_NEGATIVE[opts.mode]}"
    return positive, negative


def _apply_contrast(image: Image, factor: float = 1.35) -> Image:
    qimg = image._qimage.convertToFormat(QImage.Format.Format_ARGB32)
    for y in range(qimg.height()):
        for x in range(qimg.width()):
            c = qimg.pixel(x, y)
            a = qAlpha(c)
            r = clamp(int((qRed(c) - 128) * factor + 128), 0, 255)
            g = clamp(int((qGreen(c) - 128) * factor + 128), 0, 255)
            b = clamp(int((qBlue(c) - 128) * factor + 128), 0, 255)
            qimg.setPixel(x, y, qRgba(r, g, b, a))
    return Image(qimg)


def _apply_posterize(image: Image, levels: int) -> Image:
    levels = clamp(levels, 2, 8)
    step = 255 / (levels - 1)
    qimg = image._qimage.convertToFormat(QImage.Format.Format_ARGB32)
    for y in range(qimg.height()):
        for x in range(qimg.width()):
            c = qimg.pixel(x, y)
            a = qAlpha(c)
            r = int(round(qRed(c) / step) * step)
            g = int(round(qGreen(c) / step) * step)
            b = int(round(qBlue(c) / step) * step)
            qimg.setPixel(x, y, qRgba(r, g, b, a))
    return Image(qimg)


def _apply_threshold(image: Image, level: int = 128) -> Image:
    qimg = image._qimage.convertToFormat(QImage.Format.Format_ARGB32)
    for y in range(qimg.height()):
        for x in range(qimg.width()):
            c = qimg.pixel(x, y)
            a = qAlpha(c)
            v = 255 if qGray(c) >= level else 0
            qimg.setPixel(x, y, qRgba(v, v, v, a))
    return Image(qimg)


class VectorCncWorkspace(QObject, ObservableProperties):
    """Generate images optimized for SVG/CNC/laser vectorization."""

    output_mode = Property(VectorOutputMode.silhouette, persist=True)
    description = Property("", persist=True)
    background = Property(BackgroundMode.pure_white, persist=True)
    complexity = Property(Complexity.simple, persist=True)
    edges = Property(EdgeStyle.sharp, persist=True)
    composition = Property(Composition.centered, persist=True)
    color_count = Property(4, persist=True)
    post_contrast = Property(False, persist=True)
    post_posterize = Property(False, persist=True)
    post_threshold = Property(False, persist=True)

    output_mode_changed = pyqtSignal(VectorOutputMode)
    description_changed = pyqtSignal(str)
    background_changed = pyqtSignal(BackgroundMode)
    complexity_changed = pyqtSignal(Complexity)
    edges_changed = pyqtSignal(EdgeStyle)
    composition_changed = pyqtSignal(Composition)
    color_count_changed = pyqtSignal(int)
    post_contrast_changed = pyqtSignal(bool)
    post_posterize_changed = pyqtSignal(bool)
    post_threshold_changed = pyqtSignal(bool)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model):
        super().__init__()
        self._model = weakref.ref(model)

    def options(self) -> VectorCncOptions:
        return VectorCncOptions(
            mode=self.output_mode,
            background=self.background,
            complexity=self.complexity,
            edges=self.edges,
            composition=self.composition,
            color_count=self.color_count,
            post_contrast=self.post_contrast,
            post_posterize=self.post_posterize,
            post_threshold=self.post_threshold,
        )

    def built_prompts(self) -> tuple[str, str]:
        return build_vector_prompt(self.description, self.options())

    def apply_mode_defaults(self):
        mode = self.output_mode
        if mode is VectorOutputMode.silhouette:
            self.background = BackgroundMode.pure_white
            self.post_threshold = True
            self.post_posterize = False
        elif mode is VectorOutputMode.line_art:
            self.post_threshold = False
            self.post_posterize = False
            self.post_contrast = True
        elif mode is VectorOutputMode.limited_color:
            self.post_posterize = True
            self.post_threshold = False
            self.color_count = 4

    def annotate_job_params(self, job_params):
        job_params.metadata["workspace"] = "vector_cnc"
        job_params.metadata["vector_cnc_mode"] = self.output_mode.name
        job_params.metadata["vector_cnc_edges"] = self.edges.name
        job_params.metadata["vector_cnc_background"] = self.background.name

    def post_process(self, images: ImageCollection) -> ImageCollection:
        opts = self.options()

        def process(img: Image) -> Image:
            result = img
            if opts.post_contrast:
                result = _apply_contrast(result)
            if opts.post_posterize:
                levels = opts.color_count if opts.mode is VectorOutputMode.limited_color else 4
                result = _apply_posterize(result, levels)
            if opts.post_threshold:
                result = _apply_threshold(result)
            return result

        return images.map(process)

    def prepare_prompts(self) -> bool:
        from .model import DocumentModel

        model = ensure(self._model())
        assert isinstance(model, DocumentModel)

        if not self.description.strip():
            model.report_error(_("Please describe what to draw."))
            return False

        positive, negative = self.built_prompts()
        model.regions.positive = positive
        model.regions.negative = negative
        model.clear_error()
        return True

    def generate(self):
        model = ensure(self._model())
        if self.prepare_prompts():
            model.generate()

    def generate_replace(self):
        model = ensure(self._model())
        if self.prepare_prompts():
            model.generate_replace()

    def _resolve_mode(self, job: Job | None) -> VectorOutputMode:
        if job and (name := job.params.metadata.get("vector_cnc_mode")):
            return VectorOutputMode[name]
        return self.output_mode

    def _result_image(self, job: Job, index: int) -> Image:
        assert len(job.results) > index, "Invalid result index"
        return job.results[index]

    def _resolve_edges(self, job: Job | None) -> EdgeStyle:
        if job and (name := job.params.metadata.get("vector_cnc_edges")):
            return EdgeStyle[name]
        return self.edges

    def _resolve_background(self, job: Job | None) -> BackgroundMode:
        if job and (name := job.params.metadata.get("vector_cnc_background")):
            return BackgroundMode[name]
        return self.background

    def trace_result_svg(self, job: Job, index: int) -> str:
        image = self._result_image(job, index)
        mode = self._resolve_mode(job)
        edges = self._resolve_edges(job)
        background = self._resolve_background(job)
        return trace_image_to_svg(image, mode, self.color_count, edges, background)

    def suggest_export_path(self, job: Job, index: int, suffix: str) -> Path:
        from .model import DocumentModel

        model = ensure(self._model())
        assert isinstance(model, DocumentModel)
        prompt = sanitize_prompt(job.params.name)[:48]
        timestamp = job.timestamp.strftime("%Y%m%d-%H%M%S")
        if model.document.filename:
            doc_path = Path(model.document.filename)
            stem = doc_path.stem
            folder = doc_path.parent
        else:
            stem = "vector"
            folder = Path.home()
        path = folder / f"{stem}-vector-{timestamp}-{index}-{prompt}{suffix}"
        return find_unused_path(path)

    def export_svg(self, job: Job, index: int) -> Path:
        path = self.suggest_export_path(job, index, ".svg")
        write_text_file(path, self.trace_result_svg(job, index))
        return path

    def add_vector_layer_from_svg(self, job: Job, svg: str, name_prefix: str = "[Vector] "):
        from .model import DocumentModel

        model = ensure(self._model())
        assert isinstance(model, DocumentModel)
        name = name_prefix + job.params.name[:60]
        model.layers.create_vector(name, svg)
