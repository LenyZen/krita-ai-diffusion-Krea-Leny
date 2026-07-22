from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

from .image import Image

_MAX_TRACE_SIDE = 1024


def _background_name(background: Any | None) -> str:
    if background is None:
        return ""
    return background.name if hasattr(background, "name") else str(background)


def _dark_ink_on_light_background(background: Any | None) -> bool:
    return _background_name(background) not in ("bg_black",)


def _mode_name(mode: Any) -> str:
    return mode.name if hasattr(mode, "name") else str(mode)


def _ensure_svg_vendor() -> Path | None:
    vendor = Path(__file__).resolve().parent.parent / "krita_svg_studio" / "vendor"
    if not vendor.is_dir():
        return None
    path = str(vendor)
    if path not in sys.path:
        sys.path.insert(0, path)
    return vendor


def _image_to_grayscale(image: Image):
    import numpy as np

    arr = image.to_array()
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return (np.clip(gray, 0, 1) * 255).astype(np.uint8)


def _resize_gray_nearest(gray, max_side: int):
    import numpy as np

    height, width = gray.shape
    longest = max(height, width)
    if longest <= max_side:
        return gray, width, height
    scale = max_side / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    from PIL import Image as PilImage

    resized = PilImage.fromarray(gray, mode="L").resize((new_w, new_h), PilImage.Resampling.NEAREST)
    return np.array(resized, dtype=np.uint8), new_w, new_h


def _prepare_trace_arrays(image: Image, background: Any | None, mode_name: str):
    import numpy as np

    gray = _image_to_grayscale(image)
    orig_h, orig_w = gray.shape
    dark_on_light = _dark_ink_on_light_background(background)
    cutoff = 200 if mode_name == "line_art" else 128

    gray, trace_w, trace_h = _resize_gray_nearest(gray, _MAX_TRACE_SIDE)

    if dark_on_light:
        vtracer_img = np.where(gray < cutoff, 0, 255).astype(np.uint8)
    else:
        vtracer_img = np.where(gray >= cutoff, 0, 255).astype(np.uint8)

    return gray, vtracer_img, orig_w, orig_h, trace_w, trace_h


def _point_str(point) -> str:
    return f"{point.x:.3f} {point.y:.3f}"


def _curve_to_path_commands(curve) -> list[str]:
    start = curve.start_point
    if start is None:
        return []
    commands = [f"M {_point_str(start)}"]
    for segment in curve.segments:
        if segment.is_corner:
            commands.append(f"L {_point_str(segment.c)}")
            commands.append(f"L {_point_str(segment.end_point)}")
        else:
            commands.append(
                f"C {_point_str(segment.c1)} {_point_str(segment.c2)} "
                f"{_point_str(segment.end_point)}"
            )
    commands.append("Z")
    return commands


def _potrace_path_to_svg(traced, trace_w: int, trace_h: int, orig_w: int, orig_h: int, fill: str) -> str:
    commands: list[str] = []
    for curve in traced.curves:
        commands.extend(_curve_to_path_commands(curve))

    if not commands:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_w}" height="{orig_h}" '
            f'viewBox="0 0 {trace_w} {trace_h}"></svg>'
        )

    path_data = " ".join(commands)
    bg = "#ffffff" if fill == "#000000" else "#000000"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_w}" height="{orig_h}" '
        f'viewBox="0 0 {trace_w} {trace_h}">\n'
        f'  <rect width="100%" height="100%" fill="{bg}"/>\n'
        f'  <path d="{path_data}" fill="{fill}" fill-rule="evenodd" stroke="none"/>\n'
        f"</svg>"
    )


def _trace_with_potrace(gray, orig_w: int, orig_h: int, trace_w: int, trace_h: int, fill: str) -> str | None:
    try:
        _ensure_svg_vendor()
        import potrace  # type: ignore

        # Pass grayscale (dark pixels = shapes). Bitmap thresholds and traces dark regions.
        traced = potrace.Bitmap(gray).trace(
            turdsize=4,
            alphamax=1.0,
            opttolerance=0.2,
        )
        return _potrace_path_to_svg(traced, trace_w, trace_h, orig_w, orig_h, fill)
    except Exception:
        return None


def _trace_with_vtracer(vtracer_img, orig_w: int, orig_h: int, trace_w: int, trace_h: int) -> str | None:
    try:
        _ensure_svg_vendor()
        import vtracer  # type: ignore
        from PIL import Image as PilImage

        image = PilImage.fromarray(vtracer_img, mode="L")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        svg = vtracer.convert_raw_image_to_svg(
            buffer.getvalue(),
            img_format="png",
            colormode="binary",
            mode="spline",
            filter_speckle=4,
            corner_threshold=60,
        )
        if 'width="' not in svg:
            return svg
        # Normalize dimensions to original image size.
        import re

        svg = re.sub(r'width="[^"]*"', f'width="{orig_w}"', svg, count=1)
        svg = re.sub(r'height="[^"]*"', f'height="{orig_h}"', svg, count=1)
        if f'viewBox="0 0 {trace_w} {trace_h}"' not in svg:
            svg = re.sub(
                r'viewBox="[^"]*"',
                f'viewBox="0 0 {trace_w} {trace_h}"',
                svg,
                count=1,
            )
        return svg
    except Exception:
        return None


def _trace_with_vtracer_color(image: Image, orig_w: int, orig_h: int) -> str | None:
    try:
        _ensure_svg_vendor()
        import vtracer  # type: ignore
        from PIL import Image as PilImage

        qimg = image._qimage.convertToFormat(QImage.Format.Format_ARGB32)
        if max(orig_w, orig_h) > _MAX_TRACE_SIDE:
            qimg = qimg.scaled(
                max(1, int(orig_w * _MAX_TRACE_SIDE / max(orig_w, orig_h))),
                max(1, int(orig_h * _MAX_TRACE_SIDE / max(orig_w, orig_h))),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        trace_w, trace_h = qimg.width(), qimg.height()
        ptr = qimg.constBits()
        assert ptr is not None
        data = bytes(ptr.asstring(qimg.height() * qimg.bytesPerLine()))
        pil = PilImage.frombytes("RGBA", (trace_w, trace_h), data, "raw", "BGRA")
        buffer = BytesIO()
        pil.save(buffer, format="PNG")
        svg = vtracer.convert_raw_image_to_svg(
            buffer.getvalue(),
            img_format="png",
            colormode="color",
            mode="spline",
            filter_speckle=4,
            corner_threshold=60,
            color_precision=8,
        )
        import re

        svg = re.sub(r'width="[^"]*"', f'width="{orig_w}"', svg, count=1)
        svg = re.sub(r'height="[^"]*"', f'height="{orig_h}"', svg, count=1)
        svg = re.sub(r'viewBox="[^"]*"', f'viewBox="0 0 {trace_w} {trace_h}"', svg, count=1)
        return svg
    except Exception:
        return None


def trace_image_to_svg(
    image: Image,
    mode: Any,
    color_count: int = 4,
    edges: Any | None = None,
    background: Any | None = None,
) -> str:
    mode_name = _mode_name(mode)
    orig_w, orig_h = image.width, image.height
    fill = "#000000" if _dark_ink_on_light_background(background) else "#ffffff"

    if mode_name == "limited_color":
        if svg := _trace_with_vtracer_color(image, orig_w, orig_h):
            return svg

    gray, vtracer_img, orig_w, orig_h, trace_w, trace_h = _prepare_trace_arrays(image, background, mode_name)
    if svg := _trace_with_potrace(gray, orig_w, orig_h, trace_w, trace_h, fill):
        return svg
    if svg := _trace_with_vtracer(vtracer_img, orig_w, orig_h, trace_w, trace_h):
        return svg

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_w}" height="{orig_h}" '
        f'viewBox="0 0 {trace_w} {trace_h}"><rect width="100%" height="100%" fill="#ffffff"/>'
        f"</svg>"
    )


def write_text_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
