from __future__ import annotations

import json
import random
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from ..backend.api import LoraInput
from ..localization import translate as _
from ..util import user_data_dir
from .properties import ObservableProperties, Property

MAX_LORAS = 25
MAX_ALWAYS_LORAS = 8

# Dialog order. The stitched prompt uses STITCH_ORDER so it reads as a sentence.
DIALOG_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "subject",
        "SUBJECT",
        "Who or what is in the picture. One entry per comma.",
        "a woman, a man in a suit, a robot, a cat",
    ),
    (
        "place",
        "PLACE",
        "Where it happens.",
        "a dog house, inside a luxury car, on top of a mountain, in a mad scientist lab",
    ),
    (
        "wearing",
        "WEARING",
        "Clothes, costume, or nothing specified.",
        "a nurse costume, a police costume, shorts with knee-high yellow stockings and an orange halter top",
    ),
    (
        "action",
        "ACTION",
        "What they are doing.",
        "running, sitting on a throne, looking over their shoulder, dancing",
    ),
    (
        "mood",
        "MOOD",
        "Light, weather, and atmosphere.",
        "golden hour, neon rain, candlelight, harsh noon sun",
    ),
    (
        "shot",
        "SHOT",
        "Camera framing.",
        "close-up, full body, low angle, over the shoulder",
    ),
    (
        "extras",
        "EXTRAS",
        "Small details that finish the picture.",
        "film grain, wind in the hair, steam, floating sparks",
    ),
)

# subject wearing, doing something, somewhere. Mood, shot, and extras follow.
STITCH_ORDER = ("subject", "wearing", "action", "place", "mood", "shot", "extras")

# Twenty starter items per list. Populate defaults appends these and leaves LoRAs alone.
DEFAULT_ITEMS: dict[str, tuple[str, ...]] = {
    "subject": (
        "a woman",
        "a man",
        "a young woman with short hair",
        "an older man with a beard",
        "a robot",
        "an android with a human face",
        "a knight",
        "a pirate captain",
        "a scientist",
        "a dancer",
        "a musician",
        "a chef",
        "a detective",
        "an astronaut",
        "a cowboy",
        "a queen",
        "a street performer",
        "a cat",
        "a wolf",
        "a pair of travelers",
    ),
    "place": (
        "a dog house",
        "inside a luxury car",
        "on top of a mountain",
        "in a mad scientist lab",
        "a rainy alley at night",
        "a sunlit kitchen",
        "a castle courtyard",
        "a desert highway",
        "a neon city street",
        "a quiet library",
        "the deck of a ship",
        "a flower market",
        "a snowy forest",
        "a rooftop at dusk",
        "an abandoned theater",
        "a train compartment",
        "a greenhouse",
        "a hotel lobby",
        "a canyon overlook",
        "a carnival midway",
    ),
    "wearing": (
        "a nurse costume",
        "a police costume",
        "shorts with knee-high yellow stockings and an orange halter top",
        "a tailored black suit",
        "a leather jacket and jeans",
        "a ball gown",
        "a lab coat over street clothes",
        "armor and a travel cloak",
        "a summer dress",
        "a cowboy hat and duster",
        "a silk robe",
        "a racing suit",
        "a winter coat and scarf",
        "a sequin stage outfit",
        "a linen shirt and rolled trousers",
        "a space suit with the helmet under one arm",
        "a kimono",
        "a hoodie and sneakers",
        "a velvet cape",
        "a denim jacket and boots",
    ),
    "action": (
        "running",
        "sitting on a throne",
        "looking over their shoulder",
        "dancing",
        "reading a letter",
        "laughing",
        "climbing",
        "pouring a drink",
        "pointing at the horizon",
        "tying a boot",
        "holding a lantern",
        "leaning on a railing",
        "mid stride through a doorway",
        "adjusting a glove",
        "whispering",
        "reaching toward the camera",
        "sitting on the floor",
        "turning in the wind",
        "checking a pocket watch",
        "standing in the rain",
    ),
    "mood": (
        "golden hour",
        "neon rain",
        "candlelight",
        "harsh noon sun",
        "blue hour",
        "foggy morning",
        "storm light",
        "warm tungsten lamps",
        "moonlight",
        "overcast and soft",
        "dusty sunbeams",
        "cold fluorescent light",
        "sunset rim light",
        "firelight",
        "high contrast noir",
        "pastel dawn",
        "humid and hazy",
        "crisp winter light",
        "spotlight on a dark stage",
        "afterglow after a storm",
    ),
    "shot": (
        "close-up",
        "full body",
        "low angle",
        "over the shoulder",
        "wide establishing shot",
        "medium shot",
        "high angle",
        "profile",
        "three-quarter view",
        "from behind",
        "cowboy shot",
        "extreme close-up of the face",
        "dutch angle",
        "looking down from above",
        "through a doorway",
        "reflection in a window",
        "centered and symmetrical",
        "off-center with empty space",
        "tracking from the side",
        "portrait crop at the waist",
    ),
    "extras": (
        "film grain",
        "wind in the hair",
        "steam",
        "floating sparks",
        "falling leaves",
        "rain on the lens",
        "dust in the air",
        "bokeh lights",
        "a long shadow",
        "motion blur on the background",
        "wet pavement reflections",
        "smoke",
        "confetti",
        "lens flare",
        "chalk dust",
        "snowflakes",
        "a shallow depth of field",
        "fabric caught in a breeze",
        "scratches on an old photo",
        "a single practical light source",
    ),
}


def prompt_sets_path() -> Path:
    return user_data_dir / "randomizer_sets.json"


def append_missing(existing: str, additions: tuple[str, ...] | list[str]) -> str:
    current = split_items(existing)
    seen = {item.lower() for item in current}
    for item in additions:
        cleaned = " ".join(item.split())
        if cleaned and cleaned.lower() not in seen:
            current.append(cleaned)
            seen.add(cleaned.lower())
    return ", ".join(current)


def load_prompt_sets() -> dict[str, dict[str, str]]:
    path = prompt_sets_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sets = data.get("sets", data) if isinstance(data, dict) else {}
    cleaned: dict[str, dict[str, str]] = {}
    for name, values in sets.items():
        if not isinstance(values, dict):
            continue
        cleaned[str(name)] = {
            key: str(values.get(key, ""))
            for key, *_rest in DIALOG_FIELDS
        }
    return cleaned


def write_prompt_sets(sets: dict[str, dict[str, str]]):
    path = prompt_sets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "sets": sets}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_prompt_set(name: str, values: dict[str, str]) -> str:
    title = " ".join(name.split())
    if not title:
        raise ValueError(_("Enter a name for this prompt set."))
    sets = load_prompt_sets()
    sets[title] = {key: values.get(key, "") for key, *_rest in DIALOG_FIELDS}
    write_prompt_sets(sets)
    return title


def delete_prompt_set(name: str):
    sets = load_prompt_sets()
    sets.pop(name, None)
    write_prompt_sets(sets)


def lora_sets_path() -> Path:
    return user_data_dir / "randomizer_lora_sets.json"


def normalize_lora_stack(stack, limit: int = MAX_LORAS) -> list[dict]:
    cleaned: list[dict] = []
    if not isinstance(stack, list):
        return cleaned
    for item in stack:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name", "")).split())
        if not name:
            continue
        try:
            strength = float(item.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        cleaned.append({"name": name, "strength": strength})
        if len(cleaned) >= limit:
            break
    return cleaned


def load_lora_sets() -> dict[str, list[dict]]:
    path = lora_sets_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sets = data.get("sets", data) if isinstance(data, dict) else {}
    cleaned: dict[str, list[dict]] = {}
    for name, stack in sets.items():
        if isinstance(stack, list):
            cleaned[str(name)] = normalize_lora_stack(stack)
    return cleaned


def write_lora_sets(sets: dict[str, list[dict]]):
    path = lora_sets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "sets": sets}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_lora_set(name: str, stack: list[dict]) -> str:
    title = " ".join(name.split())
    if not title:
        raise ValueError(_("Enter a name for this LoRA set."))
    sets = load_lora_sets()
    sets[title] = normalize_lora_stack(stack)
    write_lora_sets(sets)
    return title


def delete_lora_set(name: str):
    sets = load_lora_sets()
    sets.pop(name, None)
    write_lora_sets(sets)


def split_items(text: str) -> list[str]:
    items: list[str] = []
    for part in text.replace("\r", "\n").replace("\n", ",").split(","):
        item = " ".join(part.split())
        if item:
            items.append(item)
    return items


def stitch_prompt(values: dict[str, str], rng: random.Random | None = None) -> str:
    choose = rng.choice if rng is not None else random.choice
    parts: list[str] = []
    for key in STITCH_ORDER:
        items = split_items(values.get(key, ""))
        if items:
            parts.append(choose(items))
    return ", ".join(parts)


def pick_lora(stack: list[dict], rng: random.Random | None = None) -> LoraInput | None:
    usable = [item for item in stack if str(item.get("name", "")).strip()]
    if not usable:
        return None
    choose = rng.choice if rng is not None else random.choice
    item = choose(usable)
    strength = float(item.get("strength", 1.0))
    return LoraInput(str(item["name"]).strip(), strength)


class RandomizerWorkspace(QObject, ObservableProperties):
    subject = Property("", persist=True)
    place = Property("", persist=True)
    wearing = Property("", persist=True)
    action = Property("", persist=True)
    mood = Property("", persist=True)
    shot = Property("", persist=True)
    extras = Property("", persist=True)
    lora_stack = Property(list(), persist=True)
    fixed_loras = Property(list(), persist=True)
    last_prompt = Property("", persist=True)
    last_lora = Property("", persist=True)
    last_lora_strength = Property(1.0, persist=True)
    ollama_model = Property("jimscard/new-adult-writer:latest", persist=True)
    written_prompt = Property("", persist=True)
    use_written = Property(False, persist=True)
    writer_subject = Property("", persist=True)
    writer_place = Property("", persist=True)
    writer_wearing = Property("", persist=True)
    writer_action = Property("", persist=True)
    writer_mood = Property("", persist=True)
    writer_shot = Property("", persist=True)
    writer_extras = Property("", persist=True)
    writer_idea = Property("", persist=True)
    writer_with_themes = Property(True, persist=True)

    subject_changed = pyqtSignal(str)
    place_changed = pyqtSignal(str)
    wearing_changed = pyqtSignal(str)
    action_changed = pyqtSignal(str)
    mood_changed = pyqtSignal(str)
    shot_changed = pyqtSignal(str)
    extras_changed = pyqtSignal(str)
    lora_stack_changed = pyqtSignal(list)
    fixed_loras_changed = pyqtSignal(list)
    last_prompt_changed = pyqtSignal(str)
    last_lora_changed = pyqtSignal(str)
    last_lora_strength_changed = pyqtSignal(float)
    ollama_model_changed = pyqtSignal(str)
    written_prompt_changed = pyqtSignal(str)
    use_written_changed = pyqtSignal(bool)
    writer_subject_changed = pyqtSignal(str)
    writer_place_changed = pyqtSignal(str)
    writer_wearing_changed = pyqtSignal(str)
    writer_action_changed = pyqtSignal(str)
    writer_mood_changed = pyqtSignal(str)
    writer_shot_changed = pyqtSignal(str)
    writer_extras_changed = pyqtSignal(str)
    writer_idea_changed = pyqtSignal(str)
    writer_with_themes_changed = pyqtSignal(bool)
    modified = pyqtSignal(QObject, str)

    def field_text(self, name: str) -> str:
        return str(getattr(self, name) or "")

    def values(self) -> dict[str, str]:
        return {name: self.field_text(name) for name, *_rest in DIALOG_FIELDS}

    def item_counts(self) -> dict[str, int]:
        return {name: len(split_items(text)) for name, text in self.values().items()}

    def apply_values(self, values: dict[str, str]):
        for name, *_rest in DIALOG_FIELDS:
            if name in values:
                setattr(self, name, values[name])

    def populate_defaults(self):
        self.apply_values(
            {
                name: append_missing(self.field_text(name), DEFAULT_ITEMS[name])
                for name, *_rest in DIALOG_FIELDS
            }
        )

    def writer_picks(self) -> dict[str, str]:
        return {
            "subject": self.writer_subject,
            "place": self.writer_place,
            "wearing": self.writer_wearing,
            "action": self.writer_action,
            "mood": self.writer_mood,
            "shot": self.writer_shot,
            "extras": self.writer_extras,
        }

    def roll(self) -> tuple[str, LoraInput | None]:
        prompt = self.written_prompt.strip() if self.use_written else ""
        if not prompt:
            prompt = stitch_prompt(self.values())
        if not prompt:
            raise ValueError(
                _(
                    "Add items in the Chooser, or write a prompt with Ollama "
                    "and turn on Use written prompt."
                )
            )
        lora = pick_lora(list(self.lora_stack))
        self.last_prompt = prompt
        self.last_lora = lora.name if lora is not None else ""
        self.last_lora_strength = lora.strength if lora is not None else 1.0
        return prompt, lora

    def fixed_lora_inputs(self) -> list[LoraInput]:
        return [
            LoraInput(item["name"], float(item["strength"]))
            for item in normalize_lora_stack(list(self.fixed_loras), MAX_ALWAYS_LORAS)
        ]

    def loras_for_job(self, rolled: LoraInput | None) -> list[LoraInput]:
        chosen = self.fixed_lora_inputs()
        names = {item.name for item in chosen}
        if rolled is not None and rolled.name not in names:
            chosen.append(rolled)
        return chosen

    def loras_from_last_roll(self) -> list[LoraInput]:
        rolled = (
            LoraInput(self.last_lora, self.last_lora_strength) if self.last_lora else None
        )
        return self.loras_for_job(rolled)
