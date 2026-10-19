"""Turn seven theme picks into one image prompt with a local Ollama model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434"
PREFERRED_MODEL = "jimscard/new-adult-writer:latest"

SYSTEM_INSTRUCTION = """You write one prompt for an AI image generator. The text will be sent to a diffusion model. It is not a story, a script, or a chat reply.

The user gives you seven themes: subject, place, wearing, action, mood, shot, and extras. Expand those themes into a single detailed visual prompt. Keep every theme. Do not swap in a different person, place, outfit, action, light, camera, or extra.

Describe only what the camera sees: who is there, face and body if relevant, clothing and materials, pose, setting, light, color, lens, and small textures. Write one paragraph in present tense.

Adult themes may be sensual and specific. Stay visual. Do not write dialogue, plot, warnings, or an explanation.

If a free idea is included, that idea is the spark. Elaborate it into a richer image prompt. Keep the idea's people, place, and action. Use the theme lines only where they support the idea.

Output only the prompt paragraph. No title, no labels, no bullet list, no quotation marks."""

SYSTEM_INSTRUCTION_IDEA_ONLY = """You write one prompt for an AI image generator. The text will be sent to a diffusion model. It is not a story, a script, or a chat reply.

The user gives you one free idea and nothing else. Elaborate that idea into a single detailed visual prompt. Keep the idea's people, place, clothing, and action. Do not add a different person, place, outfit, or action.

Describe only what the camera sees: who is there, face and body if relevant, clothing and materials, pose, setting, light, color, lens, and small textures. Write one paragraph in present tense.

Adult themes may be sensual and specific. Stay visual. Do not write dialogue, plot, warnings, or an explanation.

Output only the prompt paragraph. No title, no labels, no bullet list, no quotation marks."""

# Forty choices each. Same titles as the Chooser lists.
WRITER_CHOICES: dict[str, tuple[str, ...]] = {
    "subject": (
        "a woman",
        "a man",
        "a young woman with a bob haircut",
        "an older man with silver hair",
        "a couple standing close",
        "a dancer",
        "a musician with an instrument",
        "a chef",
        "a detective",
        "a scientist",
        "a knight",
        "a pirate captain",
        "an astronaut",
        "a cowboy",
        "a queen",
        "a street performer",
        "a boxer",
        "a swimmer",
        "a painter",
        "a pilot",
        "a bride",
        "a groom",
        "a mechanic",
        "a librarian",
        "a pop star",
        "a vampire",
        "an elf",
        "a robot",
        "an android with a human face",
        "a cat",
        "a wolf",
        "a horse and rider",
        "two travelers",
        "a fashion model",
        "a bartender",
        "a sailor",
        "a ballerina",
        "a samurai",
        "a witch",
        "a superhero in a worn costume",
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
        "a candlelit bedroom",
        "a boxing gym",
        "a recording studio",
        "a temple garden",
        "a subway platform",
        "a vineyard at sunset",
        "a spaceship corridor",
        "a western saloon",
        "a rain-soaked diner",
        "a marble staircase",
        "a backstage dressing room",
        "a lighthouse gallery",
        "a bamboo forest",
        "a museum at night",
        "a motel balcony",
        "a palace ballroom",
        "a cluttered artist studio",
        "a frozen lake",
        "a night market",
        "a cliff above the sea",
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
        "a wet white shirt",
        "a red evening dress",
        "a tuxedo with the bow tie undone",
        "a motorcycle jacket and gloves",
        "a sheer blouse and a high-waisted skirt",
        "a knit sweater and wool socks",
        "a gold cocktail dress",
        "a pirate coat and boots",
        "a school blazer and pleated skirt",
        "a wedding dress with mud on the hem",
        "a basketball jersey and shorts",
        "a fur-collared coat",
        "a black bodysuit and heels",
        "a safari shirt and shorts",
        "a priest collar and dark suit",
        "a flamenco dress",
        "a torn ballerina costume",
        "a silver jumpsuit",
        "a bathrobe and bare feet",
        "jewelry and little else",
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
        "lighting a cigarette",
        "fixing an earring",
        "stepping out of a car",
        "kneeling to pick something up",
        "stretching both arms overhead",
        "playing a guitar",
        "aiming a camera",
        "opening a door",
        "walking away",
        "sitting on the edge of a bed",
        "holding a sword at rest",
        "blowing out a candle",
        "catching falling papers",
        "leaning in to kiss",
        "tying back their hair",
        "looking up into the rain",
        "resting a chin on one hand",
        "spinning with a coat flaring",
        "crouching in a start position",
        "offering a hand to the viewer",
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
        "pink motel neon",
        "green aquarium glow",
        "red darkroom light",
        "silver moonlight on snow",
        "amber bar light",
        "bleach white fashion flash",
        "smoky backlit haze",
        "teal and orange cinema grade",
        "soft window light",
        "lightning flash",
        "bioluminescent blue",
        "warm skin tones against cold shadow",
        "muted vintage colors",
        "saturated pop colors",
        "desaturated war-photo tones",
        "honey-colored afternoon",
        "ink-black background",
        "pearly overcast beach light",
        "stained-glass color spill",
        "a single hard spotlight",
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
        "wide lens close to the subject",
        "telephoto compression",
        "from the floor looking up",
        "through rain on glass",
        "framed by foreground foliage",
        "two-shot of both faces",
        "silhouette against the sky",
        "mirror selfie framing",
        "overhead flat lay of the scene",
        "peeking from behind a corner",
        "tight crop on eyes and mouth",
        "full figure with room to walk",
        "rule of thirds",
        "centered portrait on a long lens",
        "handheld and slightly imperfect",
        "locked-off cinematic frame",
        "from inside a vehicle",
        "reflection in a puddle",
        "split by a doorframe",
        "environmental portrait",
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
        "cigarette smoke",
        "water droplets on skin",
        "loose threads and worn cloth",
        "jewelry catching the light",
        "a visible tattoo",
        "mud on the boots",
        "fog at ankle height",
        "city lights out of focus",
        "a glowing screen nearby",
        "petals stuck to the clothes",
        "breath visible in cold air",
        "a torn poster in the background",
        "hands in sharp focus",
        "freckles and skin texture",
        "wind moving the coat",
        "a reflection in the eyes",
        "practical lamps in the room",
        "35mm photo look",
        "crisp digital detail",
        "a faint double exposure",
    ),
}

FIELD_LABELS = (
    ("subject", "SUBJECT"),
    ("place", "PLACE"),
    ("wearing", "WEARING"),
    ("action", "ACTION"),
    ("mood", "MOOD"),
    ("shot", "SHOT"),
    ("extras", "EXTRAS"),
)


def user_message(picks: dict[str, str], idea: str = "", include_themes: bool = True) -> str:
    lines: list[str] = []
    idea = idea.strip()
    send_themes = include_themes or not idea
    if idea:
        lines.append("FREE IDEA:")
        lines.append(idea)
        lines.append("")
        if send_themes:
            lines.append(
                "Turn this idea into one detailed, elaborate image prompt. The idea leads."
            )
            lines.append("Theme lines below are extra color. Keep a theme when it fits the idea.")
        else:
            lines.append("Turn this idea into one detailed, elaborate image prompt.")
            lines.append("Use only this idea. There are no theme lines.")
        lines.append("")
    if send_themes:
        lines.extend(f"{label}: {picks.get(key, '').strip()}" for key, label in FIELD_LABELS)
        lines.append("")
    lines.append("Write the image prompt now.")
    return "\n".join(lines)


def prefer_model(names: list[str]) -> str:
    folded = [(name, name.lower().replace("_", "-")) for name in names]
    for name, key in folded:
        if "jimscard" in key and "adult-writer" in key:
            return name
    return names[0] if names else PREFERRED_MODEL


def list_models() -> list[str]:
    request = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            "Ollama is not answering at 127.0.0.1:11434. Start Ollama, then refresh the model list."
        ) from exc
    names = []
    for item in payload.get("models", []):
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def clean_prompt(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1].strip()
    for prefix in (
        "Prompt:",
        "Image prompt:",
        "Here is the prompt:",
        "Here's the prompt:",
        "Here is a prompt:",
    ):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def write_prompt(
    model: str, picks: dict[str, str], idea: str = "", include_themes: bool = True
) -> str:
    model_name = model.strip() or PREFERRED_MODEL
    idea_only = bool(idea.strip()) and not include_themes
    instruction = SYSTEM_INSTRUCTION_IDEA_ONLY if idea_only else SYSTEM_INSTRUCTION
    body = {
        "model": model_name,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_message(picks, idea, include_themes)},
        ],
        "options": {"temperature": 0.9, "num_predict": 500},
    }
    try:
        payload = _post_chat(body)
    except urllib.error.HTTPError:
        body.pop("think", None)
        payload = _post_chat(body)
    message = payload.get("message") or {}
    text = clean_prompt(str(message.get("content") or ""))
    if not text:
        raise RuntimeError(f"{model_name} returned an empty prompt.")
    return text


def _post_chat(body: dict) -> dict:
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        if isinstance(exc, urllib.error.HTTPError):
            raise
        raise ConnectionError(
            "Ollama is not answering at 127.0.0.1:11434. Start Ollama and try again."
        ) from exc
