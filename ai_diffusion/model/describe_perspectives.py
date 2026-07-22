"""Perspective-based enhancement for image descriptions.

Two paths produce long, evocative, conversational descriptions suited to
prompt-hungry models like Krea 2:
* `build_instruction` creates a full instruction for an instruction-following
  vision-language model (Qwen-VL) which re-describes the image through the
  selected genre lens.
* `enrich_caption` is the fallback used with Florence-2: it rewrites the plain
  caption in the genre's voice. Genre vocabulary is substituted throughout the
  text, evocative adjectives are injected into noun phrases, and atmosphere,
  lighting and mood sentences are interleaved with the caption instead of
  appended. Phrase choices are seeded per caption so the same image gives
  stable output while different images vary.

All strings here are prompt text sent to (or combined for) generation models,
so they intentionally stay in English and are not localized.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum


class Perspective(Enum):
    none = 0
    cinematic = 1
    sci_fi = 2
    horror = 3
    comic_book = 4
    anime = 5
    dark_fantasy = 6
    epic_fantasy = 7
    film_noir = 8
    cyberpunk = 9
    documentary = 10
    watercolor = 11
    oil_painting = 12
    storybook = 13
    steampunk = 14
    synthwave = 15
    western = 16
    post_apocalyptic = 17
    ukiyo_e = 18
    surreal_dream = 19
    vintage_photo = 20
    gerry_anderson = 21
    playboy = 22


@dataclass
class _PerspectiveStyle:
    # Genre framing inserted into the Qwen-VL instruction.
    lens: str
    # Opener templates that absorb the first caption sentence; `{sentence}`
    # is the (rewritten, lowercased) first sentence of the caption.
    openers: list[str] = field(default_factory=list)
    # Genre adjectives injected into noun phrases across the caption.
    adjectives: list[str] = field(default_factory=list)
    # Word/phrase replacements applied across the whole caption. Keys are
    # matched as whole words, case-insensitively.
    substitutions: dict[str, str] = field(default_factory=dict)
    # Standalone sentences interleaved between the caption sentences.
    atmosphere: list[str] = field(default_factory=list)
    lighting: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)


_STYLES: dict[Perspective, _PerspectiveStyle] = {
    Perspective.none: _PerspectiveStyle(
        lens=(
            "Describe it simply as a beautifully observed image, without forcing it"
            " into any particular genre."
        ),
        openers=[
            "In this richly detailed scene, {sentence}",
            "Rendered with care and presence, {sentence}",
            "In a carefully composed picture, {sentence}",
        ],
        adjectives=[
            "finely detailed",
            "beautifully rendered",
            "quietly striking",
            "carefully observed",
            "textured",
        ],
        substitutions={},
        atmosphere=[
            "Every element feels deliberate, the composition drawing the eye naturally through the frame.",
            "There is a quiet sense of place here, as if the moment were caught just before something changes.",
            "Fine textures and subtle detail reward a closer look across the whole image.",
        ],
        lighting=[
            "The light falls softly, shaping forms with gentle gradients and delicate highlights.",
            "Illumination is balanced and natural, giving depth and dimension to every surface.",
            "Highlights and shadows are carefully placed, lending the scene a tangible, almost physical depth.",
        ],
        mood=[
            "The overall mood is immersive and absorbing, inviting the viewer to linger.",
            "It carries an understated emotional weight, memorable without being loud.",
            "The atmosphere feels alive, rich with quiet detail and intent.",
        ],
    ),
    Perspective.cinematic: _PerspectiveStyle(
        lens=(
            "Treat it as a still frame from a prestige feature film: speak in the"
            " language of cinematography — framing, lens choice, depth of field,"
            " color grading, blocking — and the emotional beat of the scene."
        ),
        openers=[
            "In this breathtaking still frame from a prestige feature film, {sentence}",
            "As if lifted from a movie at its most pivotal beat, {sentence}",
            "Framed with the eye of a master cinematographer, {sentence}",
        ],
        adjectives=[
            "beautifully framed",
            "softly out-of-focus",
            "carefully blocked",
            "film-grain textured",
            "golden-hour lit",
            "anamorphic",
            "wind-touched",
        ],
        substitutions={
            "standing": "standing in perfect blocking",
            "looking at": "gazing toward",
            "walking": "moving through the frame",
            "background": "softly blurred background",
            "sky": "widescreen sky",
            "street": "rain-freshened street",
            "bright": "gently overexposed",
            "colorful": "richly graded",
        },
        atmosphere=[
            "The composition is meticulous, a shallow depth of field isolating the subject from a softly blurred world.",
            "Every element in the frame feels blocked by a director's hand, the staging loaded with unspoken narrative tension.",
            "There is a widescreen grandeur to the scene, the kind of shot a cinematographer waits all day for the light to allow.",
        ],
        lighting=[
            "Motivated light spills through the scene in warm and cool contrasts, a filmic color grade of teal shadows and golden highlights.",
            "The lighting is sculpted and dramatic, rim light carving the silhouettes while soft fill preserves detail in the shadows.",
            "Practical lights glow within the frame itself, haloed by gentle atmospheric haze that gives the air visible depth.",
        ],
        mood=[
            "The emotional temperature is unmistakable — a held breath of a moment, heavy with what happens next.",
            "It feels like the exact frame the trailer would freeze on, saturated with quiet, cinematic gravity.",
            "The mood is grand yet intimate, epic scale anchored by raw human feeling.",
        ],
    ),
    Perspective.sci_fi: _PerspectiveStyle(
        lens=(
            "Imagine it as a scene from an ambitious science-fiction film: advanced"
            " technology, vast scale, otherworldly design language, and the sense of"
            " wonder or unease that comes with the unknown."
        ),
        openers=[
            "In this stunning scene from an epic science-fiction film, {sentence}",
            "In a future no one has lived yet, {sentence}",
            "Somewhere beyond the edge of the known, {sentence}",
        ],
        adjectives=[
            "gleaming",
            "engineered",
            "otherworldly",
            "chrome-plated",
            "hologram-lit",
            "impossibly vast",
            "alien",
            "bio-luminescent",
        ],
        substitutions={
            "building": "monolithic structure",
            "buildings": "monolithic structures",
            "city": "sprawling megastructure city",
            "sky": "alien sky",
            "clouds": "strange, drifting cloud formations",
            "car": "sleek transport craft",
            "road": "transit corridor",
            "lights": "arrays of signal lights",
            "machine": "humming machine",
            "wall": "paneled alloy wall",
            "old": "ancient, pre-collapse",
            "standing": "standing amid the hum of unseen machinery",
        },
        atmosphere=[
            "The scale feels immense and technological, sleek engineered surfaces and impossible geometry stretching beyond the frame.",
            "Fine details hint at a deep, lived-in future — worn panels, drifting particulates, machinery humming just out of sight.",
            "The scene hums with speculative wonder, as if the laws of the ordinary world have been quietly rewritten.",
        ],
        lighting=[
            "Cold starlight and the glow of instrumentation paint the scene in electric blues and sterile whites, pierced by warning ambers.",
            "Volumetric beams cut through thin haze, holographic light scattering across reflective alloy surfaces.",
            "The light feels artificial and precise, emitted rather than shone — panels, consoles and distant engines each casting their own signature glow.",
        ],
        mood=[
            "The mood balances awe and unease, the thrill of the unknown pressing against the fragility of everything human in the frame.",
            "It evokes that particular sci-fi vertigo — wonder at the scale of it, and the quiet question of what it costs.",
            "There is a hushed, cathedral-like reverence to it, technology grown so vast it borders on the sacred.",
        ],
    ),
    Perspective.horror: _PerspectiveStyle(
        lens=(
            "Present it as a scene from an atmospheric horror film: dread, wrongness,"
            " and the fear of what remains unseen matter more than gore. Let the"
            " description crawl under the skin."
        ),
        openers=[
            "In this deeply unsettling frame from an atmospheric horror film, {sentence}",
            "In the held breath before the scare, {sentence}",
            "Something is wrong in this picture, though it takes a moment to name it: {sentence}",
        ],
        adjectives=[
            "unnaturally still",
            "decaying",
            "pale",
            "half-glimpsed",
            "long-abandoned",
            "forsaken",
            "mist-strangled",
            "gaunt",
            "watchful",
        ],
        substitutions={
            "standing": "standing motionless",
            "sitting": "sitting unnaturally still",
            "looking at": "staring hollowly at",
            "smiling": "smiling a little too widely",
            "walking": "drifting slowly",
            "house": "decaying house",
            "room": "airless room",
            "trees": "gnarled, leafless trees",
            "forest": "silent, watching forest",
            "fog": "suffocating fog",
            "window": "darkened window",
            "windows": "darkened windows",
            "door": "half-open door",
            "shadow": "creeping shadow",
            "shadows": "watching shadows",
            "empty": "abandoned",
            "dark": "unnaturally dark",
            "white": "bone-white",
            "night": "moonless night",
        },
        atmosphere=[
            "Something in the composition is subtly off, a wrongness at the edge of perception that makes the eye search the shadows twice.",
            "The air itself feels thick and watchful, every dark corner of the frame pregnant with the possibility of movement.",
            "Decay and silence saturate the scene, the ordinary made menacing by stillness that has lasted too long.",
        ],
        lighting=[
            "Sickly, failing light bleeds through the scene, leaving deep pools of black where anything could be waiting.",
            "A single cold source casts long, distorted shadows that stretch toward the viewer like grasping fingers.",
            "The illumination is dim and uneasy, flickering at the threshold of visibility, revealing just enough to make the darkness worse.",
        ],
        mood=[
            "Dread hangs heavy over everything — not the shock of violence but the slow certainty that something terrible is near.",
            "The mood is suffocating and intimate, the fear of being watched by something that does not blink.",
            "It leaves a residue of unease, the kind of image that follows the viewer out of the room.",
        ],
    ),
    Perspective.comic_book: _PerspectiveStyle(
        lens=(
            "Render it as a splash page from a boldly drawn comic book: dynamic"
            " inked linework, dramatic panel-worthy composition, flat vivid colors,"
            " halftone texture and graphic energy."
        ),
        openers=[
            "In this dynamic splash page straight out of a comic book, {sentence}",
            "Drawn in bold ink at the story's most dramatic beat, {sentence}",
            "Bursting with graphic-novel energy, {sentence}",
        ],
        adjectives=[
            "boldly inked",
            "halftone-shaded",
            "dynamic",
            "exaggerated",
            "action-packed",
            "vividly colored",
            "dramatically foreshortened",
        ],
        substitutions={
            "standing": "striking a heroic pose",
            "walking": "striding with motion lines trailing",
            "looking at": "locking eyes dramatically with",
            "running": "charging with kinetic speed lines",
            "sky": "flat, vividly colored sky",
            "background": "graphic, screentone-textured background",
            "shadow": "hard-edged spot-black shadow",
            "shadows": "hard-edged spot-black shadows",
            "colorful": "printed in fearless primary colors",
            "bright": "saturated to the edge of the page",
            "large": "panel-fillingly huge",
        },
        atmosphere=[
            "Thick confident ink outlines define every form, with crosshatched shading and halftone dots texturing the flat planes of color.",
            "The composition is pure panel drama — exaggerated perspective, foreshortened action, and motion lines charging the scene with kinetic energy.",
            "Every shape reads instantly, graphic and iconic, designed to hit the eye like a full-page reveal.",
        ],
        lighting=[
            "Light and shadow are carved into bold graphic shapes, hard-edged blacks against saturated primary colors.",
            "High-contrast cel-style lighting slams highlights against inky shadow, no gradient wasted.",
            "Color is flat, vivid and fearless — printed-page saturation with dramatic spot blacks anchoring the frame.",
        ],
        mood=[
            "The energy is larger than life, heroic and theatrical, a frozen instant that still seems to move.",
            "It carries the pulpy, breathless drama of a story told at full volume.",
            "The mood is bold and electric, every element pushed for maximum graphic impact.",
        ],
    ),
    Perspective.anime: _PerspectiveStyle(
        lens=(
            "Depict it as a key frame from a high-quality anime production: expressive"
            " character design, painterly backgrounds, clean cel shading, and the"
            " heightened emotion anime is loved for."
        ),
        openers=[
            "In this gorgeous key frame from a high-budget anime film, {sentence}",
            "Painted with the loving detail of a theatrical anime production, {sentence}",
            "Alive with anime's heightened beauty and feeling, {sentence}",
        ],
        adjectives=[
            "cel-shaded",
            "luminous",
            "painterly",
            "softly glowing",
            "expressive",
            "delicately drawn",
            "shimmering",
        ],
        substitutions={
            "standing": "standing as wind stirs hair and fabric",
            "looking at": "gazing with large, expressive eyes toward",
            "sky": "luminous, painterly sky",
            "clouds": "towering, softly painted clouds",
            "background": "lushly painted background",
            "trees": "watercolor-soft trees",
            "hair": "flowing, wind-caught hair",
            "eyes": "large, light-filled eyes",
            "bright": "radiantly glowing",
            "colorful": "saturated with sunset color",
            "beautiful": "breathtakingly drawn",
        },
        atmosphere=[
            "The background is painted with lush, painterly detail while the subjects carry clean, confident cel-shaded lines.",
            "Small atmospheric touches — drifting petals, shimmering air, wind-stirred fabric — give the still frame a sense of breath and motion.",
            "The composition has that distinct anime clarity, every element stylized yet emotionally precise.",
        ],
        lighting=[
            "Light blooms softly across the scene, lens-flare sparkles and glowing rim light giving everything a dreamlike radiance.",
            "The palette is luminous and saturated, sunset golds and sky blues layered with delicate gradient shading.",
            "Crisp cel shadows meet soft ambient glow, the signature contrast of a lovingly lit anime key frame.",
        ],
        mood=[
            "The emotion is dialed up in the way only anime allows — sincere, luminous, and quietly overwhelming.",
            "It feels like the pivotal frame of an episode, the moment the music swells.",
            "The mood is tender and vivid at once, nostalgia and wonder sharing the same frame.",
        ],
    ),
    Perspective.dark_fantasy: _PerspectiveStyle(
        lens=(
            "Cast it as a scene from a grim dark-fantasy tale: ancient places, gothic"
            " weight, beauty entwined with menace, and a world where magic costs"
            " dearly."
        ),
        openers=[
            "In this haunting scene from a grim dark-fantasy saga, {sentence}",
            "Steeped in gothic myth and shadowed grandeur, {sentence}",
            "In a world where beauty and menace share a throne, {sentence}",
        ],
        adjectives=[
            "ancient",
            "tarnished",
            "moonstruck",
            "rune-carved",
            "mist-wreathed",
            "crumbling",
            "cursed",
            "ornate",
        ],
        substitutions={
            "castle": "crumbling, moss-eaten castle",
            "building": "ancient gothic edifice",
            "stone": "cracked, weathered stone",
            "sword": "rune-etched blade",
            "trees": "twisted, ancient trees",
            "forest": "primeval forest",
            "fog": "coiling mist",
            "standing": "standing like a figure from an old ballad",
            "old": "centuries-old",
            "dark": "shadow-drowned",
            "night": "long, starless night",
            "light": "pale, reluctant light",
        },
        atmosphere=[
            "Ancient, weathered detail saturates the scene — cracked stone, tarnished metal, and symbols whose meanings were lost centuries ago.",
            "Mist coils low through the composition, softening edges and hiding whatever patient things wait beyond the frame.",
            "The scene carries gothic weight, ornate and oppressive, splendor gone to seed under the long pressure of dark ages.",
        ],
        lighting=[
            "Pale, moonstruck light drains the color toward silver and shadow, broken by the ember glow of distant fire.",
            "Candle flame and sorcerous luminescence gutter against a darkness that feels ancient and aware.",
            "The light is cold and reluctant, seeping through the gloom rather than conquering it.",
        ],
        mood=[
            "The mood is somber and mythic, sorrow and grandeur braided together like an old ballad's final verse.",
            "It breathes doomed romance — the sense that every beautiful thing here has already been claimed by the dark.",
            "A quiet menace underlies the splendor, power sleeping lightly and best left unwoken.",
        ],
    ),
    Perspective.epic_fantasy: _PerspectiveStyle(
        lens=(
            "Frame it as a scene from a sweeping high-fantasy epic: majestic"
            " landscapes, heroic scale, wondrous magic, and the golden light of"
            " legend."
        ),
        openers=[
            "In this majestic scene from a sweeping high-fantasy epic, {sentence}",
            "As if lifted from legend and painted at the scale of myth, {sentence}",
            "Glowing with the wonder of a grand fantasy tale, {sentence}",
        ],
        adjectives=[
            "majestic",
            "golden",
            "storied",
            "gleaming",
            "wind-swept",
            "enchanted",
            "heroic",
            "sunlit",
        ],
        substitutions={
            "mountains": "soaring, snow-crowned mountains",
            "castle": "gleaming citadel",
            "city": "storied city of spires",
            "field": "wind-rippled meadow",
            "sky": "vast, legend-bright sky",
            "river": "silver ribbon of river",
            "forest": "deep, enchanted forest",
            "standing": "standing as if destiny itself paused to watch",
            "old": "age-honored",
            "big": "heroic in scale",
            "large": "monumental",
        },
        atmosphere=[
            "The scale is heroic and breathtaking, the landscape rolling away toward horizons that promise entire untold stories.",
            "Wondrous detail fills the frame — banners in the wind, distant spires, the shimmer of enchantment on the air itself.",
            "Everything feels storied and significant, as though bards will sing of this exact moment for generations.",
        ],
        lighting=[
            "Golden, honeyed light pours across the scene like a blessing, gilding every edge it touches.",
            "Sunbeams break through vast skies in radiant shafts, painting the world in the palette of legend.",
            "Magical luminance mingles with the natural light, soft auroras and gleaming enchantments warming the air.",
        ],
        mood=[
            "The mood is soaring and hopeful, adventure and destiny audible like distant horns.",
            "It swells with wonder — the feeling of standing at the very beginning of a great tale.",
            "Nobility and awe suffuse the scene, grandeur worn lightly, like a crown deserved.",
        ],
    ),
    Perspective.film_noir: _PerspectiveStyle(
        lens=(
            "Shoot it as a frame of classic film noir: stark black-and-white"
            " chiaroscuro, venetian-blind shadows, cigarette haze, rain-slicked"
            " streets, moral ambiguity thick in the air."
        ),
        openers=[
            "In this stark frame of classic film noir, {sentence}",
            "Soaked in shadow and cigarette smoke, {sentence}",
            "On the wrong side of midnight, shot like a 1940s crime picture, {sentence}",
        ],
        adjectives=[
            "rain-slicked",
            "smoke-wreathed",
            "hard-shadowed",
            "world-weary",
            "monochrome",
            "half-lit",
            "crooked",
        ],
        substitutions={
            "street": "rain-slicked street",
            "city": "sleepless monochrome city",
            "man": "man in a rumpled coat",
            "woman": "woman with trouble in her eyes",
            "standing": "leaning against the scene like an alibi",
            "looking at": "sizing up",
            "light": "hard slanting light",
            "lights": "guttering neon lights",
            "window": "blind-striped window",
            "night": "long, complicit night",
            "dark": "ink-black",
            "smoke": "slow-curling cigarette smoke",
        },
        atmosphere=[
            "Rain-slicked surfaces mirror the scene back at itself, every reflection a little more crooked than the truth.",
            "Smoke curls through the composition, softening hard edges while the geometry of the frame stays merciless.",
            "The setting feels compromised and weary, a world where every doorway hides a debt and every window is watching.",
        ],
        lighting=[
            "Hard chiaroscuro slices the frame apart, venetian-blind shadows striping across every surface like prison bars.",
            "A single harsh key light carves faces and forms from absolute black, no mercy in the falloff.",
            "Neon and streetlamp glare fight the darkness to a draw, monochrome contrast pushed to its unforgiving limit.",
        ],
        mood=[
            "The mood is fatalistic and seductive, trouble arriving in its best coat.",
            "Moral ambiguity hangs heavier than the smoke — nobody in this frame is entirely innocent.",
            "It hums with tense, weary elegance, the calm voice of a story that already knows how it ends.",
        ],
    ),
    Perspective.cyberpunk: _PerspectiveStyle(
        lens=(
            "Wire it into a cyberpunk vision: neon-drenched megacity sprawl, high"
            " tech pressed against low life, chrome, rain, holograms, and the hum of"
            " a future that never sleeps."
        ),
        openers=[
            "In this neon-soaked vision straight out of a cyberpunk epic, {sentence}",
            "In the sleepless sprawl of a high-tech, low-life future, {sentence}",
            "Humming with cyberpunk electricity, {sentence}",
        ],
        adjectives=[
            "neon-lit",
            "chrome-trimmed",
            "rain-wet",
            "holographic",
            "augmented",
            "flickering",
            "wired",
            "grimy",
        ],
        substitutions={
            "city": "neon-drenched megacity",
            "building": "towering arcology",
            "buildings": "towering arcologies",
            "street": "rain-wet neon street",
            "sky": "smog-veiled, ad-lit sky",
            "lights": "oceans of neon signage",
            "car": "low-slung hover vehicle",
            "sign": "flickering holographic sign",
            "signs": "flickering holographic signs",
            "night": "perpetual electric night",
            "crowd": "restless augmented crowd",
            "wall": "cable-strangled wall",
        },
        atmosphere=[
            "The dense urban sprawl stacks layer upon layer, cables and catwalks and crowded signage compressing the frame with restless energy.",
            "High technology grinds against street-level grit — chrome and circuitry spliced onto the worn and the improvised.",
            "Holographic advertisements flicker over rain-wet surfaces, the city selling dreams to people too tired to sleep.",
        ],
        lighting=[
            "Neon bleeds across the scene in electric magenta and cyan, reflections shattering in puddles and polished chrome.",
            "The glow of a thousand signs and screens replaces the sky, artificial color saturating the perpetual night.",
            "Hard LED glare and soft holographic shimmer mix into a synthetic twilight that never quite becomes day.",
        ],
        mood=[
            "The mood is wired and restless, glamour and desperation running on the same current.",
            "It thrums with the loneliness of a crowded future, human warmth flickering somewhere behind the glass.",
            "Everything feels fast, cheap and luminous — a beautiful machine that long ago stopped asking permission.",
        ],
    ),
    Perspective.documentary: _PerspectiveStyle(
        lens=(
            "Observe it as an award-winning documentary photograph: honest natural"
            " light, unposed authenticity, journalistic intimacy, and the weight of a"
            " real moment truthfully witnessed."
        ),
        openers=[
            "In this award-winning documentary photograph, honest and unposed, {sentence}",
            "In a real moment truthfully witnessed, {sentence}",
            "Captured with quiet journalistic intimacy, {sentence}",
        ],
        adjectives=[
            "candid",
            "unposed",
            "weathered",
            "honest",
            "lived-in",
            "quietly telling",
        ],
        substitutions={
            "standing": "caught mid-moment",
            "looking at": "absorbed in",
            "posing": "unaware of the camera",
            "beautiful": "quietly striking",
            "bright": "honestly exposed",
            "old": "well-worn",
        },
        atmosphere=[
            "Nothing here is staged; the details are candid and specific, the texture of actual life caught mid-sentence.",
            "The framing is patient and observational, close enough to feel present, respectful enough not to intrude.",
            "Small imperfections — motion blur at the edges, honest grain — only deepen the sense of being there.",
        ],
        lighting=[
            "Available light does all the work, natural and unforced, falling exactly as it did in the moment.",
            "Soft window light or open sky models the scene gently, true to color and unembellished.",
            "The exposure honors reality — shadows kept honest, highlights unpolished, everything believable.",
        ],
        mood=[
            "The mood is quietly powerful, empathy carried in observation rather than commentary.",
            "It holds the weight of the real, a moment that mattered to someone, preserved with care.",
            "There is dignity in its restraint — the story told simply, and trusted to be enough.",
        ],
    ),
    Perspective.watercolor: _PerspectiveStyle(
        lens=(
            "Paint it as a delicate watercolor: translucent washes, soft bleeding"
            " edges, granulating pigment on textured paper, and the luminous white"
            " of the page breathing through every color."
        ),
        openers=[
            "In this delicate watercolor painting, {sentence}",
            "Washed onto textured paper in translucent pigment, {sentence}",
            "With the softness of wet-on-wet watercolor, {sentence}",
        ],
        adjectives=[
            "softly bled",
            "translucent",
            "loosely brushed",
            "granulating",
            "dappled",
            "wash-tinted",
        ],
        substitutions={
            "sky": "wet-on-wet washed sky",
            "clouds": "blooming pigment clouds",
            "water": "loosely suggested water",
            "trees": "softly bleeding trees",
            "background": "faded wash background",
            "bright": "luminously tinted",
            "dark": "deepened with layered washes",
            "detailed": "loosely suggested",
        },
        atmosphere=[
            "Edges dissolve softly into one another, pigment blooming where wet color met wet paper.",
            "The texture of cold-press paper shows through the washes, granulating pigment settling into its tooth.",
            "Much is suggested rather than stated, loose confident strokes trusting the eye to complete the forms.",
        ],
        lighting=[
            "Light lives in the untouched white of the paper, glowing through the translucent color laid around it.",
            "The palette stays luminous and airy, colors thinned to whispers and layered wet into wet.",
            "Soft gradients of wash model the light, delicate and unforced, brightest where the paper is barest.",
        ],
        mood=[
            "The mood is tender and unhurried, like a page from a travel journal painted on location.",
            "It feels fresh and breathing, the charm of the medium carried in every accidental bloom.",
            "There is a gentle, contemplative quiet to it, fragile in the loveliest way.",
        ],
    ),
    Perspective.oil_painting: _PerspectiveStyle(
        lens=(
            "Render it as a classical oil painting by an old master: rich impasto"
            " and glazing, chiaroscuro modeling, museum-grade composition, and the"
            " deep warm resonance of oil pigment on canvas."
        ),
        openers=[
            "In this classical oil painting worthy of a museum wall, {sentence}",
            "Rendered in rich oils with an old master's hand, {sentence}",
            "Built up in glazes and confident impasto, {sentence}",
        ],
        adjectives=[
            "richly glazed",
            "impasto-textured",
            "masterfully rendered",
            "varnished",
            "deeply toned",
            "canvas-textured",
        ],
        substitutions={
            "light": "golden, Rembrandt-like light",
            "shadow": "warm umber shadow",
            "shadows": "warm umber shadows",
            "background": "softly scumbled background",
            "colorful": "jewel-toned",
            "bright": "luminously glazed",
            "old": "time-darkened",
            "face": "carefully modeled face",
        },
        atmosphere=[
            "Brushwork is visible and alive — thick impasto catching real light in the highlights, thin glazes pooling in the depths.",
            "The composition follows classical geometry, the eye guided in a slow deliberate circuit of the canvas.",
            "Centuries of craft sit in every passage, from the underpainting's warmth to the final varnish's glow.",
        ],
        lighting=[
            "Chiaroscuro carves the scene from darkness, one warm light source modeling every form with painterly patience.",
            "The palette is deep and resonant — burnt sienna, raw umber, vermilion and gold against velvety darks.",
            "Light seems to come from within the paint itself, layered glazes lending the surfaces an inner glow.",
        ],
        mood=[
            "The mood is grave and timeless, a moment elevated to permanence by the weight of the medium.",
            "It carries the hushed gravity of a gallery, rich and contemplative.",
            "There is grandeur in its stillness, the dignity of a scene deemed worthy of oil and canvas.",
        ],
    ),
    Perspective.storybook: _PerspectiveStyle(
        lens=(
            "Illustrate it as a page from a beloved children's storybook: warm"
            " whimsical charm, friendly rounded shapes, cozy detail, and the gentle"
            " magic of a bedtime tale."
        ),
        openers=[
            "In this charming illustration from a beloved storybook, {sentence}",
            "As if painted for the best page of a bedtime tale, {sentence}",
            "With warm storybook whimsy, {sentence}",
        ],
        adjectives=[
            "cozy",
            "whimsical",
            "friendly",
            "rounded",
            "snug",
            "cheerful",
            "hand-painted",
        ],
        substitutions={
            "house": "crooked little cottage",
            "trees": "friendly round-topped trees",
            "forest": "gentle storybook forest",
            "path": "winding pebble path",
            "sky": "soft candy-colored sky",
            "animal": "bright-eyed woodland creature",
            "standing": "standing storybook-neat",
            "old": "charmingly weathered",
            "small": "delightfully tiny",
        },
        atmosphere=[
            "Every shape is rounded and friendly, drawn with the warm simplicity that makes small readers lean closer.",
            "Cozy details fill the corners — the kind of tiny touches children hunt for on every reread.",
            "The scene feels safe and inviting even in its adventure, wonder kept gentle at the edges.",
        ],
        lighting=[
            "Warm honeyed light washes the page, soft as a reading lamp at bedtime.",
            "Colors are sweet and harmonious, pastel skies and glowing windows painted with a loving brush.",
            "Everything glows faintly, as if the story itself were shining up through the paper.",
        ],
        mood=[
            "The mood is warm and wondering, the hush right before 'once upon a time'.",
            "It feels like being tucked in — gentle magic, safe adventure, a promise of a happy ending.",
            "Whimsy and comfort share the page, the world made small and kind.",
        ],
    ),
    Perspective.steampunk: _PerspectiveStyle(
        lens=(
            "Engineer it as a steampunk vision: Victorian brass and copper, exposed"
            " gears and rivets, hissing steam, airships and clockwork, industrial"
            " romance in amber gaslight."
        ),
        openers=[
            "In this gleaming steampunk vision of brass and steam, {sentence}",
            "Amid the clockwork and gaslight of a Victorian future, {sentence}",
            "Wrought in copper, rivets and industrial romance, {sentence}",
        ],
        adjectives=[
            "brass-fitted",
            "riveted",
            "steam-wreathed",
            "clockwork",
            "gaslit",
            "copper-clad",
            "gear-driven",
        ],
        substitutions={
            "machine": "wheezing brass contraption",
            "car": "steam-driven carriage",
            "building": "soot-stained Victorian edifice",
            "buildings": "soot-stained Victorian edifices",
            "city": "smoke-crowned industrial city",
            "sky": "airship-dotted sky",
            "lights": "flickering gas lamps",
            "metal": "polished brass",
            "wearing": "wearing goggles and",
            "modern": "retro-futuristic",
        },
        atmosphere=[
            "Exposed gears, pressure gauges and copper piping crowd the scene, every mechanism ornate and proudly overbuilt.",
            "Steam hisses from brass joints and drifts through the composition, softening the hard Victorian ironwork.",
            "The engineering is romantic rather than sterile — machines with personality, invention worn like jewelry.",
        ],
        lighting=[
            "Amber gaslight glows through the haze, glinting off polished brass and warming the soot-dark corners.",
            "Furnace light and lamplight mix in oranges and golds, shadows striped by girders and clockwork.",
            "Everything gleams with the warm patina of copper and oiled metal under flickering flame.",
        ],
        mood=[
            "The mood is adventurous and inventive, the thrill of a world building itself into the sky.",
            "It hums with industrial optimism — soot, steam and wonder in equal measure.",
            "There is a gentlemanly swagger to it, danger and elegance riding the same airship.",
        ],
    ),
    Perspective.synthwave: _PerspectiveStyle(
        lens=(
            "Style it as a 1980s synthwave/vaporwave artwork: neon grids, chrome"
            " lettering, sunset gradients of hot pink and electric purple, retro"
            " futurism soaked in nostalgia."
        ),
        openers=[
            "In this neon-drenched synthwave dreamscape, {sentence}",
            "Bathed in the pink and purple glow of a 1980s retro-future, {sentence}",
            "Straight off a chrome-lettered album cover from 1986, {sentence}",
        ],
        adjectives=[
            "neon-outlined",
            "chrome-finished",
            "grid-lined",
            "sunset-gradient",
            "retro-futuristic",
            "glowing",
            "airbrushed",
        ],
        substitutions={
            "sun": "huge striped retro sun",
            "sky": "magenta-to-purple gradient sky",
            "road": "neon-gridded highway",
            "car": "gleaming chrome sports car",
            "city": "silhouetted neon skyline",
            "lights": "glowing neon tubes",
            "night": "electric synthwave night",
            "mountains": "wireframe mountains",
            "bright": "neon-hot",
        },
        atmosphere=[
            "A glowing perspective grid rolls toward the horizon, every element outlined in electric light.",
            "The scene drips with airbrushed 1980s nostalgia — chrome, palm silhouettes, and gradients smooth as tape hiss.",
            "Retro-futurism saturates everything, a tomorrow imagined from the driver's seat of a dream machine.",
        ],
        lighting=[
            "Hot pink and electric purple wash the scene, cyan accents burning along every edge.",
            "The palette is pure sunset cassette — magenta skies, laser grids, and the glow of a striped chrome sun.",
            "Neon light blooms and reflects everywhere, soft VHS haze rounding the highlights.",
        ],
        mood=[
            "The mood is nostalgic euphoria, cruising at midnight with the synth arpeggios rising.",
            "It feels endless and electric, a summer night from a decade that never quite existed.",
            "Cool confidence hums through it, stylish, luminous and just a little melancholy.",
        ],
    ),
    Perspective.western: _PerspectiveStyle(
        lens=(
            "Stage it as a scene from a classic Western: sun-bleached frontier"
            " land, dust and leather, long shadows at high noon, laconic figures"
            " under an enormous sky."
        ),
        openers=[
            "In this sun-scorched frame from a classic Western, {sentence}",
            "Under the enormous sky of the open frontier, {sentence}",
            "With dust hanging in the golden air of high noon, {sentence}",
        ],
        adjectives=[
            "sun-bleached",
            "dust-caked",
            "weather-beaten",
            "leathery",
            "wind-scoured",
            "frontier-worn",
        ],
        substitutions={
            "man": "squinting stranger",
            "standing": "standing with a gunslinger's stillness",
            "street": "dusty main street",
            "town": "one-street frontier town",
            "sky": "vast merciless sky",
            "sun": "punishing noon sun",
            "horse": "trail-weary horse",
            "building": "weatherboard saloon",
            "walking": "walking slow and deliberate",
            "old": "hard-worn",
        },
        atmosphere=[
            "Dust drifts through the scene and settles on everything, the land itself vast, dry and indifferent.",
            "The frontier stretches to the horizon in every direction, human figures small and stubborn against it.",
            "Creaking leather, sun-cracked wood and trail dust give the scene its grit — nothing here came easy.",
        ],
        lighting=[
            "Harsh noon light hammers down, carving short black shadows and bleaching every color toward bone.",
            "Golden-hour sun rakes low across the plain, long shadows striping the dust red and amber.",
            "The light is dry and unforgiving, heat shimmering off the ground under a cloudless glare.",
        ],
        mood=[
            "The mood is laconic and tense, a held silence where the next move decides everything.",
            "It breathes lonesome grandeur — freedom and hardship sharing the same saddle.",
            "There is quiet myth in it, the West the way the ballads remember it.",
        ],
    ),
    Perspective.post_apocalyptic: _PerspectiveStyle(
        lens=(
            "Set it after the fall: a post-apocalyptic world of rust and ruin,"
            " nature reclaiming the broken works of civilization, survivors' traces"
            " in the overgrowth, beauty persisting in decay."
        ),
        openers=[
            "In this haunting scene from a world after the fall, {sentence}",
            "Among the rust and ruins of a civilization gone quiet, {sentence}",
            "Where the overgrowth is slowly winning, {sentence}",
        ],
        adjectives=[
            "rust-eaten",
            "overgrown",
            "crumbling",
            "abandoned",
            "scavenged",
            "weed-cracked",
            "sun-faded",
        ],
        substitutions={
            "city": "silent ruined city",
            "building": "gutted, vine-strangled building",
            "buildings": "gutted, vine-strangled buildings",
            "car": "rusted-out husk of a car",
            "road": "weed-cracked highway",
            "street": "debris-strewn street",
            "grass": "waist-high wild grass",
            "windows": "shattered windows",
            "new": "salvaged",
            "clean": "scavenged-clean",
        },
        atmosphere=[
            "Nature is reclaiming everything — vines strangling steel, saplings splitting the asphalt, birdsong where traffic used to be.",
            "The ruins keep their old shapes like memories, every rusted sign and empty doorway hinting at the world before.",
            "Silence has settled over the scene like dust, broken only by wind moving through structures built for crowds.",
        ],
        lighting=[
            "Hazy shafts of sunlight fall through broken roofs and dust, gilding the decay with unexpected tenderness.",
            "The palette is rust, ash and overgrowth green, warmed by a low sun that no longer answers to clocks.",
            "Light pools quietly in the ruins, soft and elegiac, indifferent to what was lost.",
        ],
        mood=[
            "The mood is elegiac but strangely hopeful — the world ended, and yet the light is beautiful on the ruins.",
            "It carries the loneliness of the last page of a story, and the stubborn green of a sequel beginning.",
            "Melancholy and resilience share the frame, survival written in small, quiet details.",
        ],
    ),
    Perspective.ukiyo_e: _PerspectiveStyle(
        lens=(
            "Print it as a Japanese ukiyo-e woodblock: flat elegant color fields,"
            " bold contour lines, stylized waves and clouds, asymmetric composition"
            " in the tradition of Hokusai and Hiroshige."
        ),
        openers=[
            "In this elegant ukiyo-e woodblock print, {sentence}",
            "Carved and printed in the tradition of Hokusai, {sentence}",
            "With the flat grace of a floating-world print, {sentence}",
        ],
        adjectives=[
            "flatly printed",
            "bold-outlined",
            "stylized",
            "indigo-tinted",
            "elegantly simplified",
            "paper-textured",
        ],
        substitutions={
            "waves": "great stylized waves with clawed foam",
            "water": "patterned indigo water",
            "sky": "graded indigo-to-cream sky",
            "clouds": "flat ribbon-like clouds",
            "mountain": "serene distant mountain",
            "mountains": "serene distant mountains",
            "trees": "gnarled pines drawn in bold contour",
            "rain": "slanting lines of woodblock rain",
            "detailed": "elegantly simplified",
        },
        atmosphere=[
            "Forms are flattened into elegant shapes bounded by confident carved lines, depth suggested by overlap and placement alone.",
            "The composition is boldly asymmetric, empty space doing as much work as the printed color.",
            "Patterns stylize the natural world — waves, clouds and rain rendered as rhythm rather than realism.",
        ],
        lighting=[
            "Color sits in serene flat fields — indigo, ochre, seafoam and cream — graded softly where the sky meets the horizon.",
            "There are no cast shadows, only the calm luminosity of layered pigment on mulberry paper.",
            "The bokashi gradients breathe quiet atmosphere into the flat planes, dawn or dusk implied in a single wash.",
        ],
        mood=[
            "The mood is serene and contemplative, the floating world caught mid-breath.",
            "It carries the poised stillness of the ukiyo-e masters, transience made beautiful.",
            "Elegance and restraint govern the scene, every element placed like a considered word.",
        ],
    ),
    Perspective.surreal_dream: _PerspectiveStyle(
        lens=(
            "Dream it as a surrealist vision: impossible juxtapositions, melting"
            " logic, symbols standing in for feelings, the familiar made strange in"
            " the manner of Dalí and Magritte."
        ),
        openers=[
            "In this surreal dream where logic has quietly left the room, {sentence}",
            "As if painted from the inside of a sleeping mind, {sentence}",
            "Where the familiar turns strange and the strange feels inevitable, {sentence}",
        ],
        adjectives=[
            "impossibly floating",
            "dreamlike",
            "melting",
            "uncanny",
            "symbol-laden",
            "gravity-defying",
            "half-remembered",
        ],
        substitutions={
            "sky": "sky that might be a ceiling",
            "clouds": "clouds drifting indoors",
            "clock": "soft melting clock",
            "door": "door opening onto elsewhere",
            "mirror": "mirror reflecting the wrong room",
            "floor": "floor dissolving into horizon",
            "standing": "hovering just above the ground",
            "normal": "deceptively ordinary",
            "strange": "dream-logical",
        },
        atmosphere=[
            "The scene obeys dream logic — scale drifts, horizons misbehave, and the impossible is presented with a straight face.",
            "Familiar objects appear where they cannot be, each juxtaposition feeling less like a mistake than a message.",
            "There is a hush of deep sleep over everything, details crisp in the center and dissolving at the edges of attention.",
        ],
        lighting=[
            "The light comes from nowhere in particular, casting long theatrical shadows that don't quite agree with each other.",
            "Colors are heightened and slightly wrong, a sunset palette at noon, ocean light in a closed room.",
            "Everything is lit with the even, breathless clarity of a dream just before waking.",
        ],
        mood=[
            "The mood is uncanny and tender at once, meaning hovering just out of reach like a word on the tip of the tongue.",
            "It feels significant the way dreams do — every object a symbol, every silence full.",
            "Wonder and unease hold hands here, the strange rendered with perfect calm.",
        ],
    ),
    Perspective.vintage_photo: _PerspectiveStyle(
        lens=(
            "Expose it as a vintage photograph from decades past: film grain, faded"
            " chemistry, light leaks and soft vignettes, a captured moment aged into"
            " memory."
        ),
        openers=[
            "In this faded vintage photograph pulled from an old shoebox, {sentence}",
            "Fixed in silver and time on aging film, {sentence}",
            "Through the soft grain of a decades-old exposure, {sentence}",
        ],
        adjectives=[
            "sepia-toned",
            "grainy",
            "sun-faded",
            "creased",
            "softly vignetted",
            "time-worn",
        ],
        substitutions={
            "photo": "aging photograph",
            "colorful": "faded toward sepia",
            "bright": "gently overexposed",
            "sharp": "softened by old glass",
            "new": "then-new",
            "modern": "of its era",
            "smiling": "smiling across the decades",
        },
        atmosphere=[
            "Film grain textures every surface, and a soft vignette draws the eye to the heart of the frame.",
            "The chemistry has aged — blacks gone brown, whites gone cream — giving the moment the warmth of memory itself.",
            "Small imperfections tell the story: a light leak blushing one corner, a crease where someone carried it for years.",
        ],
        lighting=[
            "The exposure is gentle and slightly blown, sunlight remembered rather than recorded.",
            "Tones settle into sepia and silver, shadows soft, highlights breathing with halation.",
            "The light has the unhurried quality of an older world, patient enough for a long shutter.",
        ],
        mood=[
            "The mood is wistful and warm, a moment that mattered enough for someone to keep.",
            "Nostalgia rises off it like dust in projector light — tender, distant, and very human.",
            "It feels like holding someone else's memory, precious and slightly out of reach.",
        ],
    ),
    Perspective.gerry_anderson: _PerspectiveStyle(
        lens=(
            "Stage it as a frame from a classic Gerry Anderson Supermarionation"
            " production — Thunderbirds, Captain Scarlet, Stingray: articulated"
            " marionette characters, miniature sets and vehicles, mid-century"
            " British futurism, bold primary colors, and that slightly stiff,"
            " heroic puppet-show adventure look."
        ),
        openers=[
            "In this frame from a classic Gerry Anderson Supermarionation series, {sentence}",
            "Staged like a Thunderbirds miniature set with marionette heroes, {sentence}",
            "With the bold mid-century puppet futurism of Gerry Anderson, {sentence}",
        ],
        adjectives=[
            "marionette-like",
            "miniature-scale",
            "boldly painted",
            "retro-futuristic",
            "stiffly posed",
            "heroically staged",
            "model-kit precise",
        ],
        substitutions={
            "person": "marionette figure",
            "man": "heroic marionette man",
            "woman": "elegantly posed marionette woman",
            "people": "marionette characters",
            "face": "painted puppet face with glass eyes",
            "hands": "articulated puppet hands",
            "building": "detailed miniature building",
            "buildings": "detailed miniature buildings",
            "city": "miniature futuristic city set",
            "car": "gleaming model vehicle",
            "vehicle": "detailed scale-model craft",
            "ship": "miniature craft on wires",
            "plane": "sleek model aircraft",
            "rocket": "Thunderbirds-style rocket craft",
            "sky": "painted studio sky backdrop",
            "standing": "standing with marionette stiffness",
            "looking at": "gazing with fixed glass eyes toward",
        },
        atmosphere=[
            "Everything reads as a lovingly built miniature — scale models, painted backdrops, and characters that move with the deliberate charm of marionettes.",
            "The staging is theatrical and heroic, every figure posed for adventure the way Anderson's puppets always were.",
            "Visible craftsmanship saturates the scene: tiny rivets, glossy vehicle paint, and that unmistakable Supermarionation stillness between movements.",
        ],
        lighting=[
            "Bright studio lighting models the miniatures cleanly, colors popping in the bold primaries of 1960s British TV sci-fi.",
            "Hard, even set lighting keeps every vehicle and face readable, with little naturalistic mess — pure television adventure polish.",
            "The palette is saturated and cheerful: reds, blues, and chrome catching the lights like a Century 21 studio stage.",
        ],
        mood=[
            "The mood is earnest adventure — danger faced with stiff upper lips and impeccable uniforms.",
            "It carries the optimistic futurism of Anderson's world: tomorrow as a thrilling toybox of craft and courage.",
            "There is a charming, slightly uncanny puppet gravity to it, heroic and handmade at once.",
        ],
    ),
    Perspective.playboy: _PerspectiveStyle(
        lens=(
            "Shoot it as a classic Playboy magazine pictorial from the golden era:"
            " tasteful glamour photography, soft high-key beauty light, satin and"
            " skin tones, elegant posing, luxurious sets, and that polished"
            " mid-century to 1980s centerfold sensuality — glamorous and"
            " sophisticated rather than crude."
        ),
        openers=[
            "In this classic Playboy magazine pictorial, {sentence}",
            "Shot like a golden-era Playboy centerfold spread, {sentence}",
            "With the soft glamorous polish of a vintage Playboy photo essay, {sentence}",
        ],
        adjectives=[
            "glamorous",
            "softly lit",
            "satin-smooth",
            "elegantly posed",
            "luxurious",
            "airbrushed",
            "centerfold-polished",
        ],
        substitutions={
            "woman": "glamorous Playmate",
            "man": "suave leading man",
            "smile": "inviting centerfold smile",
            "looking at": "gazing invitingly toward",
            "standing": "posing with elegant confidence",
            "sitting": "reclining with studied grace",
            "room": "luxurious boudoir set",
            "bed": "satin-draped bed",
            "sofa": "plush velvet sofa",
            "light": "soft beauty light",
            "skin": "luminous, satin-smooth skin",
            "hair": "perfectly styled hair",
            "bright": "high-key and flattering",
        },
        atmosphere=[
            "The set is luxurious and curated — soft fabrics, warm woods, and the kind of tasteful excess that defined classic magazine pictorials.",
            "Every pose feels intentional and flattering, composed for the page with the confidence of a centerfold spread.",
            "The image has that unmistakable Playboy polish: glamorous, adult, and carefully produced rather than candid.",
        ],
        lighting=[
            "Soft high-key beauty lighting wraps the subject, gentle fill smoothing shadows and making skin tones glow.",
            "A classic butterfly or Rembrandt key flatters the face, with warm rim light separating the figure from a tasteful background.",
            "The palette is warm and inviting — honey highlights, creamy midtones, and just enough contrast for magazine print.",
        ],
        mood=[
            "The mood is sensual and sophisticated — invitation without vulgarity, glamour as its own atmosphere.",
            "It feels like turning a page in a golden-era issue: aspirational, playful, and unmistakably adult.",
            "Confidence and allure share the frame, polished to that timeless Playboy pictorial sheen.",
        ],
    ),
}


_LENGTH_GUIDANCE = {
    "caption": (
        "Write two to three flowing sentences — brief, but vivid and emotionally"
        " charged rather than dry."
    ),
    "detailed_caption": (
        "Write one generous, richly detailed paragraph of five to eight sentences."
    ),
    "more_detailed_caption": (
        "Write two to three full paragraphs of long, flowing prose. Be lavish with"
        " detail; leave nothing important undescribed."
    ),
    "prompt_gen_mixed_caption_plus": (
        "Write one rich descriptive paragraph, then end with a single line of"
        " comma-separated keyword tags summarizing the subjects, style, lighting"
        " and mood."
    ),
}
_DEFAULT_LENGTH_GUIDANCE = _LENGTH_GUIDANCE["more_detailed_caption"]


def _genre_lens(perspective: Perspective, custom: str = "") -> str:
    style = _STYLES[perspective]
    custom = custom.strip()
    if custom and perspective is Perspective.none:
        return (
            f"Describe it entirely through this lens: {custom}. Let that framing"
            " color every sentence — the vocabulary, the atmosphere, the way each"
            " element is seen."
        )
    if custom:
        return (
            f"{style.lens} In addition, apply this lens throughout: {custom}."
            " Let both framings color every sentence."
        )
    return style.lens


def _compose_lens(
    perspective: Perspective,
    custom: str = "",
    style_name: str = "",
    style_prompt: str = "",
) -> str:
    """Combine optional Clio art-style paragraph with genre / custom lenses."""
    style_name = style_name.strip()
    style_prompt = style_prompt.strip()
    genre = _genre_lens(perspective, custom)
    if not style_prompt:
        return genre
    clio = (
        f'Apply the art style "{style_name}" as the primary visual language for'
        " the description. Use the following style paragraph as the aesthetic lens"
        " — medium, rendering, line, color, and atmosphere — while still describing"
        f" what is actually visible in the image:\n{style_prompt}"
    )
    if perspective is Perspective.none and not custom.strip():
        return clio
    return f"{clio}\nAlso apply this additional framing: {genre}"


def build_instruction(
    perspective: Perspective,
    level_task: str,
    custom: str = "",
    style_name: str = "",
    style_prompt: str = "",
) -> str:
    """Full instruction for an instruction-following VLM (Qwen-VL).

    `level_task` is the Florence-2 task name from `DescribeLevel`, reused here
    to scale the requested length of the response. `custom` is free-form user
    text describing an additional perspective; it takes the leading role and
    is combined with the dropdown genre when one is selected.
    Optional `style_name` / `style_prompt` inject a Clio Style Library entry.
    """
    length = _LENGTH_GUIDANCE.get(level_task, _DEFAULT_LENGTH_GUIDANCE)
    lens = _compose_lens(perspective, custom, style_name, style_prompt)
    return (
        "You are writing an evocative image description that will be used as a"
        " prompt for an image generation model, so describe only what should"
        " appear in the picture — no meta commentary, no preamble, no lists of"
        f" observations. {lens} Describe this image in long, conversational,"
        " emotionally evocative prose: name the subjects and their appearance,"
        " the setting and its details, the composition and point of view, the"
        " lighting and color palette, the textures and materials, and above all"
        f" the atmosphere and the feelings the scene evokes. {length}"
        " Respond with the description text only."
    )


def build_enhance_instruction(
    draft: str,
    perspective: Perspective,
    level_task: str,
    custom: str = "",
    style_name: str = "",
    style_prompt: str = "",
) -> str:
    """Instruction for Qwen-VL: expand the user's draft using the attached image."""
    draft = draft.strip()
    lens = _compose_lens(perspective, custom, style_name, style_prompt)
    if level_task == "prompt_gen_tags":
        length = (
            "Output a single line of comma-separated Danbooru-style tags that"
            " capture the draft intent and what is visible in the image — subjects,"
            " style, lighting, and mood. No sentences, no preamble."
        )
    else:
        length = _LENGTH_GUIDANCE.get(level_task, _DEFAULT_LENGTH_GUIDANCE)
    return (
        "You are enhancing a user's draft description into a generation-ready"
        " image prompt. Use the attached image as ground truth for what is"
        " actually visible — keep the draft's intent, but correct contradictions,"
        " fill in concrete visual detail, and expand weakly specified parts.\n"
        f"{lens}\n"
        "Write in long, conversational, emotionally evocative prose suited to"
        " prompt-hungry image models: subjects and appearance, setting, composition,"
        " lighting, color, textures/materials, atmosphere and mood.\n"
        "Rules:\n"
        "- Output ONLY the enhanced description (no preamble, no bullet lists,"
        " no 'here is the improved prompt').\n"
        "- Stay faithful to the image; do not invent major subjects that are absent.\n"
        "- Preserve distinctive names, style cues, and creative direction from the draft.\n"
        f"- {length}\n\n"
        f"User draft to enhance:\n{draft}\n"
    )


def _clean_caption(caption: str) -> str:
    text = caption.strip()
    for prefix in (
        "The image shows ",
        "The image is ",
        "The image depicts ",
        "The image features ",
        "This image shows ",
        "This image is ",
        "This image depicts ",
        "This is ",
        "A picture of ",
        "An image of ",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :]
            break
    return text.strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NOUN_PHRASE = re.compile(r"\b(a|an|the)\s+([A-Za-z]+)\b", re.IGNORECASE)

# Words after an article that shouldn't receive an injected adjective.
_SKIP_NOUNS = {
    "few",
    "lot",
    "bit",
    "kind",
    "sort",
    "type",
    "number",
    "couple",
    "pair",
    "very",
    "same",
    "single",
    "little",
    "most",
    "first",
    "second",
    "third",
    "last",
    "other",
    "top",
    "bottom",
    "left",
    "right",
    "middle",
    "center",
    "front",
    "back",
    "side",
    "image",
    "picture",
    "photo",
    "scene",
    "view",
    "foreground",
    "background",
}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _apply_substitutions(sentence: str, substitutions: dict[str, str]) -> str:
    if not substitutions:
        return sentence
    # Single combined pass (longest keys first) so replacement text is never
    # re-matched by another substitution.
    lookup = {k.lower(): v for k, v in substitutions.items()}
    keys = sorted(substitutions, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE
    )

    def repl(m: re.Match) -> str:
        replacement = lookup[m.group(0).lower()]
        if m.group(0)[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return pattern.sub(repl, sentence)


def _inject_adjectives(
    sentence: str,
    pool: list[str],
    rng: random.Random,
    skip: set[str],
    max_injections: int = 2,
) -> str:
    """Insert genre adjectives into noun phrases: 'a castle' -> 'a crumbling castle'."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        noun = m.group(2)
        if count >= max_injections or not pool or noun.lower() in _SKIP_NOUNS:
            return m.group(0)
        if noun.lower() in skip:
            # This noun gets genre treatment from a substitution instead.
            return m.group(0)
        if rng.random() > 0.55:
            return m.group(0)
        adjective = pool.pop(rng.randrange(len(pool)))
        count += 1
        article = m.group(1)
        if article.lower() in ("a", "an"):
            fixed = "an" if adjective[0].lower() in "aeiou" else "a"
            article = fixed.capitalize() if article[0].isupper() else fixed
        return f"{article} {adjective} {noun}"

    return _NOUN_PHRASE.sub(repl, sentence)


def _lower_first(sentence: str) -> str:
    # Lowercase the leading word unless it looks like a proper noun or acronym.
    first_word = sentence.split(" ", 1)[0]
    if first_word in ("A", "An", "The", "There", "It", "This", "Two", "Three", "Several", "Some"):
        return sentence[0].lower() + sentence[1:]
    if len(first_word) > 1 and first_word[1:].islower() and first_word.istitle():
        # Ordinary capitalized word (not an acronym); risky for names, but
        # captions from Florence-2 rarely start with a proper noun.
        return sentence[0].lower() + sentence[1:]
    return sentence


def _end_punctuated(sentence: str) -> str:
    if not sentence.endswith((".", "!", "?")):
        return sentence + "."
    return sentence


_CUSTOM_OPENERS = [
    "Seen as {custom}, {sentence}",
    "Imagined as {custom}, {sentence}",
    "Rendered as {custom}, {sentence}",
]

_CUSTOM_CLOSERS = [
    "Every detail of the scene is filtered through that vision — {custom} — coloring the light, the textures and the mood alike.",
    "The whole image commits to its framing as {custom}, letting that sensibility saturate every element.",
]


def enrich_caption(perspective: Perspective, caption: str, custom: str = "") -> str:
    """Rewrite a plain Florence-2 caption in the perspective's genre voice.

    The whole caption is transformed: genre vocabulary replaces neutral words
    in every sentence, evocative adjectives are woven into noun phrases, the
    first sentence is absorbed into a genre opener, and atmosphere/lighting
    sentences are interleaved with the remaining caption before closing on a
    mood note. Free-form `custom` perspective text is worked into the opener
    (and a closing note) on top of the selected genre's rewriting.
    """
    text = _clean_caption(caption)
    if not text:
        return caption

    style = _STYLES[perspective]
    custom = custom.strip().rstrip(".")
    rng = random.Random(text)  # str seed is stable across sessions

    adjective_pool = list(style.adjectives)
    substituted_words = {k.lower() for k in style.substitutions}
    sentences = []
    for sentence in _split_sentences(text):
        # Inject adjectives first, on original caption nouns only, so genre
        # substitutions don't end up with stacked modifiers.
        sentence = _inject_adjectives(sentence, adjective_pool, rng, substituted_words)
        sentence = _apply_substitutions(sentence, style.substitutions)
        sentences.append(_end_punctuated(sentence))

    if custom:
        opener = rng.choice(_CUSTOM_OPENERS).format(
            custom=custom, sentence=_lower_first(sentences[0])
        )
    else:
        opener = rng.choice(style.openers).format(sentence=_lower_first(sentences[0]))
    body = sentences[1:]
    extras = [rng.choice(style.atmosphere), rng.choice(style.lighting)]

    parts = [opener]
    if body:
        # Interleave atmosphere and lighting between the caption sentences so
        # the genre voice runs through the whole text instead of trailing it.
        insert_points = sorted(
            rng.sample(range(1, len(body) + 1), min(len(extras), len(body)))
        )
        remaining = [e for i, e in enumerate(extras) if i >= len(insert_points)]
        position = 0
        for i, sentence in enumerate(body, start=1):
            parts.append(sentence)
            if position < len(insert_points) and i == insert_points[position]:
                parts.append(extras[position])
                position += 1
        parts.extend(remaining)
    else:
        parts.extend(extras)
    parts.append(rng.choice(style.mood))
    if custom:
        parts.append(rng.choice(_CUSTOM_CLOSERS).format(custom=custom))
    return " ".join(parts)
