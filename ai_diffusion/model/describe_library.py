"""Personal prompt snippets for the Describe workspace.

Stored under the Krita AI user data dir as describe_library.json.
Users can save drafts/results and insert them later — lightweight alternative
to NO8D-style XLSX libraries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..util import user_data_dir

log = logging.getLogger(__name__)

_LIBRARY_PATH = user_data_dir / "describe_library.json"

_DEFAULT_ENTRIES = [
    {
        "name": "Portrait — soft window light",
        "text": (
            "A close portrait with soft window light from camera-left, natural skin,"
            " shallow depth of field, calm expression, clean background."
        ),
    },
    {
        "name": "Cinematic wide — dusk city",
        "text": (
            "Wide establishing shot of a city street at dusk, wet asphalt reflections,"
            " neon signs, lone figure mid-frame for scale, moody teal-orange grade."
        ),
    },
    {
        "name": "Product — hero tabletop",
        "text": (
            "Product hero shot on a clean tabletop, soft key and gentle rim light,"
            " sharp materials and labels, minimal props, studio backdrop."
        ),
    },
]


def library_path() -> Path:
    return _LIBRARY_PATH


def load_library() -> list[dict[str, str]]:
    path = _LIBRARY_PATH
    if not path.exists():
        save_library(list(_DEFAULT_ENTRIES))
        return list(_DEFAULT_ENTRIES)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return list(_DEFAULT_ENTRIES)
        out: list[dict[str, str]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            text = str(entry.get("text", "")).strip()
            if name and text:
                out.append({"name": name, "text": text})
        return out or list(_DEFAULT_ENTRIES)
    except Exception as e:
        log.warning("Failed to load describe library %s: %s", path, e)
        return list(_DEFAULT_ENTRIES)


def save_library(entries: list[dict[str, str]]) -> None:
    path = _LIBRARY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [
        {"name": e["name"].strip(), "text": e["text"].strip()}
        for e in entries
        if e.get("name") and e.get("text")
    ]
    path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_library_entry(name: str, text: str) -> list[dict[str, str]]:
    name = (name or "").strip()
    text = (text or "").strip()
    if not name or not text:
        raise ValueError("Name and text are required")
    entries = load_library()
    for entry in entries:
        if entry["name"].lower() == name.lower():
            entry["text"] = text
            save_library(entries)
            return entries
    entries.append({"name": name, "text": text})
    save_library(entries)
    return entries


def find_library_text(name: str) -> str:
    name = (name or "").strip().lower()
    for entry in load_library():
        if entry["name"].lower() == name:
            return entry["text"]
    return ""
