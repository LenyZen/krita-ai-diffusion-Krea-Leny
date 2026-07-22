"""Clio Style Library loader for Describe (art-style prose paragraphs).

Source: https://github.com/lumenastrum/clio-style-preview
Code MIT; style text is community-shared prompt prose (see NOTICE below).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

# Empty name = no Clio style selected
CLIO_STYLE_NONE = ""

CLIO_CREDIT = (
    "Style library from Clio Style Preview (MIT code; community style text) — "
    "https://github.com/lumenastrum/clio-style-preview"
)

_STYLES_PATH = Path(__file__).resolve().parent.parent / "presets" / "clio_styles.json"


@lru_cache(maxsize=1)
def load_clio_styles() -> dict[str, str]:
    """Return ordered {name: prompt} from presets/clio_styles.json."""
    try:
        with open(_STYLES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        out: dict[str, str] = {}
        for entry in data:
            name = str(entry.get("name", "")).strip()
            prompt = str(entry.get("prompt", "")).strip()
            if name and prompt:
                out[name] = prompt
        return out
    except Exception as e:
        log.warning("Failed to load Clio styles from %s: %s", _STYLES_PATH, e)
        return {}


def clio_style_names() -> list[str]:
    return list(load_clio_styles().keys())


def clio_style_prompt(name: str) -> str:
    if not name:
        return ""
    return load_clio_styles().get(name, "")
