"""List every available edge-tts voice and render a playable sound sample.

Generates one audio file per voice, a JSON manifest, and a self-contained HTML
gallery with players, filtering, and search.

Usage::

    python scripts\\tts_voice_samples.py                 # every voice
    python scripts\\tts_voice_samples.py --lang en       # English only
    python scripts\\tts_voice_samples.py --list-only     # print table, no audio
    python scripts\\tts_voice_samples.py --format wav    # also write 16 kHz WAV
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import edge_tts
import miniaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("alvin.voice_samples")

DEFAULT_OUT = ROOT / "voice_samples"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

DEFAULT_TEXT = (
    "Hello! This is Alvin. The quick brown fox jumps over the lazy dog. "
    "Numbers: one, two, three."
)

# Short native sample lines so each locale is judged on its own phonemes.
SAMPLE_TEXT: dict[str, str] = {
    "en": "Hello! This is a sample of my voice. One, two, three.",
    "es": "Hola, esta es una muestra de mi voz. Uno, dos, tres.",
    "fr": "Bonjour, voici un échantillon de ma voix. Un, deux, trois.",
    "de": "Hallo, dies ist eine Probe meiner Stimme. Eins, zwei, drei.",
    "it": "Ciao, questo è un campione della mia voce. Uno, due, tre.",
    "pt": "Olá, esta é uma amostra da minha voz. Um, dois, três.",
    "nl": "Hallo, dit is een voorbeeld van mijn stem. Eén, twee, drie.",
    "ru": "Привет, это образец моего голоса. Один, два, три.",
    "pl": "Cześć, to jest próbka mojego głosu. Jeden, dwa, trzy.",
    "tr": "Merhaba, bu benim sesimin bir örneği. Bir, iki, üç.",
    "sv": "Hej, det här är ett exempel på min röst. Ett, två, tre.",
    "da": "Hej, dette er et eksempel på min stemme. En, to, tre.",
    "no": "Hei, dette er et eksempel på stemmen min. En, to, tre.",
    "fi": "Hei, tämä on näyte äänestäni. Yksi, kaksi, kolme.",
    "cs": "Ahoj, toto je ukázka mého hlasu. Jedna, dva, tři.",
    "sk": "Ahoj, toto je ukážka môjho hlasu. Jeden, dva, tri.",
    "hu": "Szia, ez az én hangom mintája. Egy, kettő, három.",
    "ro": "Bună, aceasta este o mostră a vocii mele. Unu, doi, trei.",
    "el": "Γεια σας, αυτό είναι ένα δείγμα της φωνής μου. Ένα, δύο, τρία.",
    "uk": "Привіт, це зразок мого голосу. Один, два, три.",
    "ar": "مرحبًا، هذا نموذج من صوتي. واحد، اثنان، ثلاثة.",
    "he": "שלום, זוהי דוגמה לקול שלי. אחת, שתיים, שלוש.",
    "hi": "नमस्ते, यह मेरी आवाज़ का एक नमूना है। एक, दो, तीन।",
    "id": "Halo, ini adalah contoh suara saya. Satu, dua, tiga.",
    "ms": "Hai, ini adalah contoh suara saya. Satu, dua, tiga.",
    "vi": "Xin chào, đây là mẫu giọng nói của tôi. Một, hai, ba.",
    "th": "สวัสดี นี่คือตัวอย่างเสียงของฉัน หนึ่ง สอง สาม",
    "zh": "你好，这是我的声音示例。一，二，三。",
    "ja": "こんにちは。これが私の声のサンプルです。一、二、三。",
    "ko": "안녕하세요, 이것은 제 목소리 샘플입니다. 하나, 둘, 셋.",
    "bn": "হ্যালো, এটি আমার কণ্ঠের একটি নমুনা। এক, দুই, তিন।",
    "ta": "வணக்கம், இது என் குரலின் ஒரு மாதிரி. ஒன்று, இரண்டு, மூன்று.",
    "ur": "سلام، یہ میری آواز کا ایک نمونہ ہے۔ ایک، دو، تین۔",
    "sw": "Habari, hii ni sampuli ya sauti yangu. Moja, mbili, tatu.",
}

LOCALE_TEXT: dict[str, str] = {
    "es-MX": "es",
    "es-US": "es",
    "pt-BR": "pt",
    "pt-PT": "pt",
    "zh-CN": "zh",
    "zh-HK": "zh",
    "zh-TW": "zh",
    "nb-NO": "no",
}


@dataclass
class VoiceResult:
    """Per-voice synthesis outcome recorded in the manifest."""

    short_name: str
    name: str
    locale: str
    gender: str
    tags: list[str]
    file: str | None
    bytes: int | None
    seconds: float | None
    error: str | None


def voice_tags(voice: dict) -> list[str]:
    """Return the voice's roles, preferring explicit Tags then VoiceTag detail."""
    tags = list(voice.get("Tags") or [])
    if tags:
        return tags
    detail = voice.get("VoiceTag") or {}
    return list(
        detail.get("VoicePersonalities") or detail.get("ContentCategories") or []
    )


def text_for_voice(locale: str, override: str | None) -> str:
    """Pick a native sample line for the voice locale."""
    if override:
        return override
    base = LOCALE_TEXT.get(locale)
    if base is None:
        base = locale.split("-")[0].lower()
    return SAMPLE_TEXT.get(base, DEFAULT_TEXT)


def filter_voices(
    voices: list[dict], langs: list[str], genders: list[str]
) -> list[dict]:
    """Filter the edge-tts catalogue by language prefix and gender."""
    wanted_langs = {lang.lower() for lang in langs}
    wanted_genders = {gender.lower() for gender in genders}

    selected = []
    for voice in voices:
        locale = voice.get("Locale", "")
        if wanted_langs and not any(
            locale.lower() == lang or locale.lower().startswith(lang + "-")
            for lang in wanted_langs
        ):
            continue
        if wanted_genders and voice.get("Gender", "").lower() not in wanted_genders:
            continue
        selected.append(voice)
    return selected


def print_catalogue(voices: list[dict]) -> None:
    """Print every voice grouped by locale with gender, roles, and file name."""
    by_locale: dict[str, list[dict]] = {}
    for voice in voices:
        by_locale.setdefault(voice.get("Locale", "?"), []).append(voice)

    total = len(voices)
    print(f"\n{total} voices across {len(by_locale)} locales\n" + "=" * 72)

    for locale in sorted(by_locale):
        entries = by_locale[locale]
        roles = sorted({tag for v in entries for tag in voice_tags(v)})
        print(f"\n{locale}  ({len(entries)} voices, roles: {', '.join(roles) or '-'})")
        print("-" * 72)
        for voice in sorted(entries, key=lambda v: v.get("ShortName", "")):
            tags = ", ".join(voice_tags(voice))
            print(
                f"  {voice.get('ShortName', ''):<28}"
                f" {voice.get('Gender', ''):<7} {tags}"
            )


async def synthesize_mp3(text: str, voice: str, retries: int) -> bytes:
    """Fetch MP3 audio for one voice, retrying transient network failures."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.extend(chunk["data"])
            if not buffer:
                raise RuntimeError("empty audio response")
            return bytes(buffer)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(1.5 * attempt)
    raise RuntimeError(str(last_error))


def write_wav_file(mp3: bytes, path: Path) -> float:
    """Write a 16 kHz mono WAV from MP3 bytes; return duration in seconds."""
    decoded = miniaudio.decode(
        mp3,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=TARGET_CHANNELS,
        sample_rate=TARGET_SAMPLE_RATE,
    )
    miniaudio.wav_write_file(str(path), decoded)
    return round(decoded.duration, 2)


async def render_voice(
    voice: dict,
    out_dir: Path,
    override_text: str | None,
    retries: int,
    semaphore: asyncio.Semaphore,
    write_wav: bool,
) -> VoiceResult:
    """Synthesize one voice, save MP3 (and optionally WAV), return the result."""
    short_name = voice.get("ShortName", "unknown")
    result = VoiceResult(
        short_name=short_name,
        name=voice.get("Name", ""),
        locale=voice.get("Locale", ""),
        gender=voice.get("Gender", ""),
        tags=voice_tags(voice),
        file=None,
        bytes=None,
        seconds=None,
        error=None,
    )

    text = text_for_voice(result.locale, override_text)
    async with semaphore:
        try:
            mp3 = await synthesize_mp3(text, short_name, retries)
            mp3_path = out_dir / f"{short_name}.mp3"
            mp3_path.write_bytes(mp3)

            seconds = None
            if write_wav:
                seconds = write_wav_file(mp3, out_dir / f"{short_name}.wav")

            result.file = mp3_path.name
            result.bytes = len(mp3)
            result.seconds = seconds
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)[:200]
            log.warning("%s failed: %s", short_name, result.error)

    return result


def build_gallery(results: list[VoiceResult], out_dir: Path) -> None:
    """Write a self-contained HTML gallery with audio players and filters."""
    entries = [asdict(result) for result in results if result.file]
    payload = json.dumps(entries, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alvin TTS voices ({len(entries)} samples)</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px; background: #14161a; color: #e8eaed; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.meta {{ margin: 0 0 20px; color: #9aa0a6; }}
  .controls {{ position: sticky; top: 0; display: flex; flex-wrap: wrap; gap: 8px;
               margin-bottom: 20px; padding: 12px 0; background: #14161a; z-index: 2; }}
  input, select {{ padding: 8px 10px; border-radius: 6px; border: 1px solid #333a42;
                   background: #1e2228; color: #e8eaed; font-size: 13px; }}
  input[type=search] {{ flex: 1; min-width: 220px; }}
  .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }}
  .card {{ background: #1e2228; border: 1px solid #2b3138; border-radius: 10px; padding: 12px 14px; }}
  .card h2 {{ font-size: 14px; margin: 0; word-break: break-word; }}
  .card .sub {{ color: #9aa0a6; font-size: 12px; margin: 2px 0 10px; }}
  audio {{ width: 100%; height: 32px; }}
  code {{ color: #8ab4f8; }}
  .empty {{ color: #9aa0a6; }}
</style>
</head>
<body>
<h1>Alvin TTS voices</h1>
<p class="meta">{len(entries)} generated samples &mdash; filter, then play to compare.</p>
<div class="controls">
  <input type="search" id="q" placeholder="Search name, locale, role...">
  <select id="lang"><option value="">All languages</option></select>
  <select id="gender">
    <option value="">All genders</option>
    <option>Female</option><option>Male</option>
  </select>
</div>
<div class="grid" id="grid"></div>
<p class="empty" id="empty" hidden>No voices match.</p>
<script>
const voices = {payload};
const grid = document.getElementById('grid');
const q = document.getElementById('q');
const lang = document.getElementById('lang');
const gender = document.getElementById('gender');
const empty = document.getElementById('empty');

for (const v of voices) {{
  const base = v.locale.split('-')[0];
  if (![...lang.options].some(o => o.value === base)) {{
    lang.insertAdjacentHTML('beforeend', `<option>${{base}}</option>`);
  }}
}}

function render() {{
  const needle = q.value.trim().toLowerCase();
  const matches = voices.filter(v =>
    (!needle || `${{v.short_name}} ${{v.name}} ${{v.locale}} ${{v.tags.join(' ')}}`.toLowerCase().includes(needle)) &&
    (!lang.value || v.locale.split('-')[0] === lang.value) &&
    (!gender.value || v.gender === gender.value)
  );
  grid.innerHTML = matches.map(v => `
    <div class="card">
      <h2><code>${{v.short_name}}</code></h2>
      <div class="sub">${{v.locale}} &middot; ${{v.gender}} &middot; ${{v.tags.join(', ')}} &middot; ${{(v.bytes / 1024).toFixed(0)}} kB${{v.seconds ? ' &middot; ' + v.seconds + 's' : ''}}</div>
      <audio controls preload="none" src="${{v.file}}"></audio>
    </div>`).join('');
  empty.hidden = matches.length > 0;
}}
[q, lang, gender].forEach(el => el.addEventListener('input', render));
render();
</script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogue = await edge_tts.list_voices()
    selected = filter_voices(
        catalogue,
        [lang for lang in args.lang.split(",") if lang] if args.lang != "all" else [],
        [g for g in args.gender.split(",") if g] if args.gender else [],
    )
    if args.limit:
        selected = selected[: args.limit]

    print_catalogue(selected)

    if args.list_only:
        return 0

    results: list[VoiceResult] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    print(
        f"\nSynthesizing {len(selected)} samples with concurrency "
        f"{args.concurrency} -> {out_dir}\n"
    )

    total = len(selected)
    for index, (voice, result) in enumerate(
        zip(
            selected,
            await asyncio.gather(
                *(
                    render_voice(
                        voice,
                        out_dir,
                        args.text,
                        args.retries,
                        semaphore,
                        args.format == "wav",
                    )
                    for voice in selected
                )
            ),
        ),
        start=1,
    ):
        results.append(result)
        status = "ok" if result.file else f"FAILED ({result.error})"
        print(f"[{index:>{len(str(total))}}/{total}] {result.short_name:<28} {status}")

    failures = [r for r in results if not r.file]
    manifest = {
        "text_override": args.text,
        "generated": len(results),
        "failed": len(failures),
        "voices": [asdict(result) for result in results],
    }
    (out_dir / "voices.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    build_gallery(results, out_dir)

    print(f"\nSamples written: {len(results) - len(failures)}/{total}")
    print(f"Manifest: {out_dir / 'voices.json'}")
    print(f"Gallery:  {out_dir / 'index.html'}")
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument(
        "--lang",
        default="all",
        help="comma-separated language codes, or 'all' (default)",
    )
    parser.add_argument(
        "--gender", default="", help="comma-separated genders, e.g. Female,Male"
    )
    parser.add_argument("--text", default=None, help="override the sample text")
    parser.add_argument(
        "--format", choices=["mp3", "wav"], default="mp3", help="output audio format"
    )
    parser.add_argument("--concurrency", type=int, default=8, help="parallel requests")
    parser.add_argument("--retries", type=int, default=3, help="attempts per voice")
    parser.add_argument("--limit", type=int, default=0, help="only first N voices")
    parser.add_argument(
        "--list-only", action="store_true", help="print the catalogue, skip audio"
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
