"""
Stage 1: Speech-to-Text Normalizer, Homophone Repair, Pakistani Minglish /
Roman Urdu Translator, and Intra-Utterance Self-Correction Parser.

This module is 100% deterministic and rule-driven (no external LLMs).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.models import NormalizationResult


# ---------------------------------------------------------------------------
# Urdu & English Number Lexicons
# ---------------------------------------------------------------------------

URDU_NUMBER_MAP: Dict[str, int] = {
    "ek": 1,
    "aik": 1,
    "do": 2,
    "teen": 3,
    "char": 4,
    "chaar": 4,
    "panch": 5,
    "paanch": 5,
    "chhe": 6,
    "che": 6,
    "chay": 6,
    "saat": 7,
    "sat": 7,
    "aath": 8,
    "ath": 8,
    "nau": 9,
    "das": 10,
    "dus": 10,
    "gyarah": 11,
    "barah": 12,
    "terah": 13,
    "chaudah": 14,
    "pandrah": 15,
    "solah": 16,
    "satrah": 17,
    "atharah": 18,
    "unnis": 19,
    "bees": 20,
    "pachees": 25,
    "tees": 30,
    "paintees": 35,
    "chalees": 40,
    "pachas": 50,
}

ENGLISH_NUMBER_MAP: Dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty five": 25,
    "thirty": 30,
    "thirty five": 35,
    "forty": 40,
    "forty five": 45,
    "fifty": 50,
}

# Filler words that add noise at the start/middle of spoken turns
FILLER_PATTERNS = [
    r"\b(?:um+|uh+|er+|ah+|hmm+|like\s+uh)\b",
]


# ---------------------------------------------------------------------------
# Homophone Repair Rules (STT Misrecognitions)
# ---------------------------------------------------------------------------

HOMOPHONE_RULES: List[Tuple[re.Pattern, str, str]] = [
    # "for people / for guests / for persons / table for for"
    (
        re.compile(r"\btable\s+for\s+for\b", re.IGNORECASE),
        "table for 4",
        "table for for -> table for 4",
    ),
    (
        re.compile(r"\bfor\s+(people|persons|guests|members|seats|of\s+us|log|logon|banday)\b", re.IGNORECASE),
        r"4 \1",
        "for <guests> -> 4 <guests>",
    ),
    (
        re.compile(r"\b(?:to|too)\s+(people|persons|guests|members|seats|of\s+us|log|logon|banday)\b", re.IGNORECASE),
        r"2 \1",
        "to/too <guests> -> 2 <guests>",
    ),
    (
        re.compile(r"\bwon\s+(person|guest|seat|table)\b", re.IGNORECASE),
        r"1 \1",
        "won <guest> -> 1 <guest>",
    ),
    (
        re.compile(r"\bate\s+(people|persons|guests|members|seats|of\s+us|log|logon)\b", re.IGNORECASE),
        r"8 \1",
        "ate <guests> -> 8 <guests>",
    ),
    # Time homophones: "ate pm", "ate thirty", "ate o'clock", "at ate", "ate baje"
    (
        re.compile(r"\bate\s+(pm|p\.m\.|am|a\.m\.|o'?clock|thirty|fifteen|forty\s*five|baje)\b", re.IGNORECASE),
        r"8 \1",
        "ate <time> -> 8 <time>",
    ),
    (
        re.compile(r"\bat\s+ate\b", re.IGNORECASE),
        "at 8",
        "at ate -> at 8",
    ),
    (
        re.compile(r"\bfor\s+(pm|p\.m\.|o'?clock|thirty|baje)\b", re.IGNORECASE),
        r"4 \1",
        "for <time> -> 4 <time>",
    ),
    (
        re.compile(r"\b(?:to|too)\s+(pm|p\.m\.|o'?clock|thirty|baje)\b", re.IGNORECASE),
        r"2 \1",
        "to/too <time> -> 2 <time>",
    ),
    # English-STT phonetic misrecognitions of Pakistani Urdu / Minglish speech
    (
        re.compile(r"\b(\d{1,2}|ek|do|teen|char|panch|chhe|saat|aath|nau|das)\s+(?:budget|bajaj|badger|budge)\b", re.IGNORECASE),
        r"\1 baje",
        "STT phonetic 'budget/bajaj' -> 'baje'",
    ),
    (
        re.compile(r"\b(?:sari|sadly|study)\s+(\d{1,2}|saat|sat|aath|ath|nau|das|chhe|panch|char|teen)\b", re.IGNORECASE),
        r"saarhe \1",
        "STT phonetic 'sari <num>' -> 'saarhe <num>'",
    ),
    (
        re.compile(r"\b(?:kull|cull|curl)\b(?=\s+(?:saarhe|sawa|paune|shaam|raat|\d|char|panch|chhe|saat|aath))", re.IGNORECASE),
        "kal",
        "STT phonetic 'kull/cull' -> 'kal'",
    ),
    (
        re.compile(r"\b(?:art|heart)\s+(?:baje|bjy)\b", re.IGNORECASE),
        "8 baje",
        "STT phonetic 'art baje' -> '8 baje'",
    ),
    (
        re.compile(r"\b(?:south)\s+(?:baje|bjy)\b", re.IGNORECASE),
        "7 baje",
        "STT phonetic 'south baje' -> '7 baje'",
    ),
]


# ---------------------------------------------------------------------------
# Native Urdu Script (اردو) Transliteration (for ur-PK Speech Recognition)
# ---------------------------------------------------------------------------

URDU_DIGITS_TRANS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

URDU_SCRIPT_PHRASE_MAP: List[Tuple[str, str]] = [
    ("السلام علیکم", "salam"),
    ("اسلام علیکم", "salam"),
    ("وعلیکم السلام", "salam"),
    ("نہیں بلکہ", "nahi balkay"),
    ("نہیں بلکے", "nahi balkay"),
    ("ہاں جی بالکل", "haan jee bilkul"),
    ("ہاں جی بلکل", "haan jee bilkul"),
    ("جی ہاں بالکل", "jee haan bilkul"),
    ("ہاں جی", "haan jee"),
    ("جی ہاں", "jee haan"),
    ("رہنے دو", "rehne do"),
    ("رہنے دیں", "rehne do"),
    ("کنفرم کر دیں", "confirm kar dein"),
    ("کنفرم کریں", "confirm kar dein"),
    ("کینسل کر دیں", "cancel kar dein"),
    ("بک کروانا ہے", "book karwana hai"),
    ("بک کرنا ہے", "book karna hai"),
    ("خدا حافظ", "khuda hafiz"),
    ("اللہ حافظ", "allah hafiz"),
    ("پرائیویٹ فیملی مجلس", "private family majlis"),
    ("فیملی مجلس", "family majlis"),
    ("روف ٹاپ", "rooftop"),
    ("میرا نام", "mera naam"),
    ("فون نمبر", "phone number"),
    ("ساڑھے", "saarhe"),
    ("سوا", "sawa"),
    ("پونے", "paune"),
    ("ڈیڑھ", "derh"),
    ("ڈھائی", "dhai"),
    ("بجے", "baje"),
    ("پرسوں", "parso"),
    ("کل", "kal"),
    ("آج", "aaj"),
    ("شام", "shaam"),
    ("رات", "raat"),
    ("دوپہر", "dopehar"),
    ("صبح", "subah"),
    ("لوگوں", "logon"),
    ("لوگ", "log"),
    ("بندوں", "bandon"),
    ("بندے", "banday"),
    ("مہمانوں", "mehmanon"),
    ("مہمان", "mehman"),
    ("افراد", "afraad"),
    ("انڈور", "indoor"),
    ("اندر", "indoor"),
    ("چھت", "rooftop"),
    ("مجلس", "majlis"),
    ("ٹیبل", "table"),
    ("میز", "table"),
    ("بک", "book"),
    ("کنفرم", "confirm"),
    ("کینسل", "cancel"),
    ("پارکنگ", "parking"),
    ("ویلٹ", "valet"),
    ("بارش", "rain"),
    ("کارڈ", "card"),
    ("بلکل", "bilkul"),
    ("بالکل", "bilkul"),
    ("ہاں", "haan"),
    ("جی", "jee"),
    ("نہیں", "nahi"),
    ("نہ", "na"),
    ("شکریہ", "shukriya"),
    ("سلام", "salam"),
    ("ایک", "1"),
    ("دو", "2"),
    ("تین", "3"),
    ("چار", "4"),
    ("پانچ", "5"),
    ("چھ", "6"),
    ("سات", "7"),
    ("آٹھ", "8"),
    ("نو", "9"),
    ("دس", "10"),
    ("گیارہ", "11"),
    ("بارہ", "12"),
    ("پندرہ", "15"),
    ("بیس", "20"),
    ("کا", "ka"),
    ("کی", "ki"),
    ("کے", "ke"),
    ("لیے", "liye"),
    ("میں", "mein"),
    ("پر", "par"),
    ("ہے", "hai"),
    ("کر", "kar"),
    ("دو", "do"),
    ("دیں", "dein"),
    ("چاہیے", "chahiye"),
    ("سوری", "sorry"),
    ("طارق", "Tariq"),
    ("محمود", "Mahmood"),
    ("بلال", "Bilal"),
    ("احمد", "Ahmed"),
    ("علی", "Ali"),
    ("خان", "Khan"),
    ("عائشہ", "Ayesha"),
    ("صدیقی", "Siddiqui"),
    ("کامران", "Kamran"),
    ("زینب", "Zainab"),
    ("رضا", "Raza"),
    ("سارہ", "Sarah"),
    ("حسن", "Hassan"),
    ("حسین", "Hussain"),
    ("عمر", "Umar"),
    ("فاروق", "Farooq"),
    ("فاطمہ", "Fatima"),
    ("حمزہ", "Hamza"),
    ("عثمان", "Usman"),
    ("نادیہ", "Nadia"),
    ("کاشف", "Kashif"),
    ("ملک", "Malik"),
    ("شاہ", "Shah"),
]

URDU_CHAR_FALLBACK_MAP: Dict[str, str] = {
    "آ": "aa", "ا": "a", "أ": "a", "إ": "i",
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh",
    "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh",
    "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z",
    "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m",
    "ن": "n", "ں": "n", "و": "o", "ؤ": "o",
    "ہ": "h", "ھ": "h", "ة": "h", "ء": "",
    "ی": "i", "ي": "i", "ے": "e", "ئ": "i",
}


def _transliterate_urdu_script(text: str) -> Tuple[str, List[str]]:
    """
    Translates native Urdu Perso-Arabic script (e.g. from ur-PK SpeechRecognition)
    and Urdu numerals into Roman Urdu / English tokens.
    """
    mappings: List[str] = []
    out = text.translate(URDU_DIGITS_TRANS)
    has_urdu_chars = bool(re.search(r"[\u0600-\u06FF]", out))
    if not has_urdu_chars:
        return out, mappings

    for urdu_phrase, roman_equiv in URDU_SCRIPT_PHRASE_MAP:
        if urdu_phrase in out:
            out = out.replace(urdu_phrase, f" {roman_equiv} ")

    # Fallback character-level Romanization for any remaining Urdu script words (e.g. arbitrary names)
    if re.search(r"[\u0600-\u06FF]", out):
        chars: List[str] = []
        for ch in out:
            if "\u0600" <= ch <= "\u06FF":
                chars.append(URDU_CHAR_FALLBACK_MAP.get(ch, ""))
            else:
                chars.append(ch)
        out = "".join(chars)

    mappings.append("Urdu Script (اردو) -> Roman Urdu/English")
    out = re.sub(r"\s+", " ", out).strip()
    return out, mappings


# ---------------------------------------------------------------------------
# Pakistani Minglish / Roman Urdu Normalization
# ---------------------------------------------------------------------------

def _apply_minglish_normalization(text: str) -> Tuple[str, List[str]]:
    """
    Translates Roman Urdu / Pakistani Minglish date, time, number, guest, and
    seating expressions into canonical English tokens while preserving the
    original conversational cues for intent classification.
    """
    mappings: List[str] = []
    out = text

    # 1. Convert Urdu number words before guest/time keywords into digits
    all_urdu_nums = "|".join(sorted(URDU_NUMBER_MAP.keys(), key=len, reverse=True))
    urdu_num_context_re = re.compile(
        rf"\b({all_urdu_nums})\b(?=\s+(?:log|logon|logo|banday|bando|bande|afraad|mehman|people|guests|persons|seats|baje|bjy|pm|am))",
        re.IGNORECASE,
    )

    def _replace_urdu_num(match: re.Match) -> str:
        word = match.group(1).lower()
        val = URDU_NUMBER_MAP[word]
        mappings.append(f"{word} -> {val}")
        return str(val)

    out = urdu_num_context_re.sub(_replace_urdu_num, out)

    # 2. Fractional Urdu times:
    # "derh baje" -> "1:30 pm", "dhai baje" -> "2:30 pm"
    derh_re = re.compile(r"\b(?:derh|dedh)\s*(?:baje|bjy)?\b", re.IGNORECASE)
    if derh_re.search(out):
        mappings.append("derh baje -> 1:30 pm")
        out = derh_re.sub("1:30 pm", out)

    dhai_re = re.compile(r"\b(?:dhai|dhaai)\s*(?:baje|bjy)?\b", re.IGNORECASE)
    if dhai_re.search(out):
        mappings.append("dhai baje -> 2:30 pm")
        out = dhai_re.sub("2:30 pm", out)

    # "saarhe / sadhe <N> baje" -> "<N>:30"
    saarhe_re = re.compile(
        rf"\b(?:saarhe|sadhe|sarhe|sade)\s+(\d{{1,2}}|{all_urdu_nums})(?:\s*(?:baje|bjy))?\b",
        re.IGNORECASE,
    )

    def _replace_saarhe(m: re.Match) -> str:
        raw_n = m.group(1).lower()
        hour = int(raw_n) if raw_n.isdigit() else URDU_NUMBER_MAP.get(raw_n, 8)
        mappings.append(f"{m.group(0)} -> {hour}:30")
        return f"{hour}:30"

    out = saarhe_re.sub(_replace_saarhe, out)

    # "sawa <N> baje" -> "<N>:15"
    sawa_re = re.compile(
        rf"\bsawa\s+(\d{{1,2}}|{all_urdu_nums})(?:\s*(?:baje|bjy))?\b",
        re.IGNORECASE,
    )

    def _replace_sawa(m: re.Match) -> str:
        raw_n = m.group(1).lower()
        hour = int(raw_n) if raw_n.isdigit() else URDU_NUMBER_MAP.get(raw_n, 8)
        mappings.append(f"{m.group(0)} -> {hour}:15")
        return f"{hour}:15"

    out = sawa_re.sub(_replace_sawa, out)

    # "paune <N> baje" -> "<N-1>:45"
    paune_re = re.compile(
        rf"\b(?:paune|pone)\s+(\d{{1,2}}|{all_urdu_nums})(?:\s*(?:baje|bjy))?\b",
        re.IGNORECASE,
    )

    def _replace_paune(m: re.Match) -> str:
        raw_n = m.group(1).lower()
        hour = int(raw_n) if raw_n.isdigit() else URDU_NUMBER_MAP.get(raw_n, 8)
        prev_hour = 12 if hour == 1 else hour - 1
        mappings.append(f"{m.group(0)} -> {prev_hour}:45")
        return f"{prev_hour}:45"

    out = paune_re.sub(_replace_paune, out)

    # 3. "<N> baje" with time-of-day context ("raat 8 baje", "shaam 7 baje", "8 baje")
    # First normalize "raat/shaam/dopehar/subah" around "<N> baje"
    raat_before_re = re.compile(
        r"\b(?:raat|shaam|dopehar|dupehri)(?:\s+ko)?\s+(\d{1,2}(?::\d{2})?)\s*(?:baje|bjy)?\b",
        re.IGNORECASE,
    )
    def _replace_raat_before(m: re.Match) -> str:
        t = m.group(1)
        mappings.append(f"{m.group(0)} -> {t} pm")
        return f"{t} pm"

    out = raat_before_re.sub(_replace_raat_before, out)

    subah_before_re = re.compile(
        r"\bsubah(?:\s+ko)?\s+(\d{1,2}(?::\d{2})?)\s*(?:baje|bjy)?\b",
        re.IGNORECASE,
    )
    def _replace_subah_before(m: re.Match) -> str:
        t = m.group(1)
        mappings.append(f"{m.group(0)} -> {t} am")
        return f"{t} am"

    out = subah_before_re.sub(_replace_subah_before, out)

    # "<N> baje raat/shaam"
    baje_after_tod_re = re.compile(
        r"\b(\d{1,2}(?::\d{2})?)\s*(?:baje|bjy)\s+(?:raat|shaam|dopehar)\b",
        re.IGNORECASE,
    )
    def _replace_baje_after(m: re.Match) -> str:
        t = m.group(1)
        mappings.append(f"{m.group(0)} -> {t} pm")
        return f"{t} pm"

    out = baje_after_tod_re.sub(_replace_baje_after, out)

    # Standalone "<N> baje" -> "at <N> baje" (kept recognizable as time for slot extractor)
    baje_re = re.compile(r"\b(\d{1,2}(?::\d{2})?)\s*(?:baje|bjy)\b", re.IGNORECASE)
    def _replace_baje(m: re.Match) -> str:
        t = m.group(1)
        mappings.append(f"{m.group(0)} -> {t} baje")
        return f"{t} baje"

    out = baje_re.sub(_replace_baje, out)

    # 4. Guest nouns: "4 logon ka", "4 log", "6 banday", "5 mehman"
    guest_re = re.compile(
        r"\b(\d{1,3})\s*(?:logon|logo|log|bandon|bando|banday|bande|afraad|mehmanon|mehman)(?:\s+(?:ka|ki|ke|k\s+liye|ke\s+liye))?\b",
        re.IGNORECASE,
    )
    def _replace_guest(m: re.Match) -> str:
        n = m.group(1)
        mappings.append(f"{m.group(0)} -> {n} people")
        return f"{n} people"

    out = guest_re.sub(_replace_guest, out)

    # 5. Relative date words: "parso/parson", "aaj", "kal"
    parso_re = re.compile(r"\b(?:parso|parson|narso)\b", re.IGNORECASE)
    if parso_re.search(out):
        mappings.append("parso -> day after tomorrow")
        out = parso_re.sub("day after tomorrow", out)

    aaj_re = re.compile(r"\baaj\b", re.IGNORECASE)
    if aaj_re.search(out):
        mappings.append("aaj -> today")
        out = aaj_re.sub("today", out)

    # "guzra hua kal" / "pichle kal" -> yesterday
    past_kal_re = re.compile(r"\b(?:guzra\s+hua\s+kal|pichle\s+kal|kal\s+raat\s+aya\s+tha)\b", re.IGNORECASE)
    if past_kal_re.search(out):
        mappings.append("guzra kal -> yesterday")
        out = past_kal_re.sub("yesterday", out)

    kal_re = re.compile(r"\bkal\b", re.IGNORECASE)
    if kal_re.search(out):
        mappings.append("kal -> tomorrow")
        out = kal_re.sub("tomorrow", out)

    # 6. Seating expressions in Urdu
    andar_re = re.compile(r"\b(?:andar(?:\s+ac)?(?:\s+mein)?|indoor\s+mein|ac\s+hall\s+mein)\b", re.IGNORECASE)
    if andar_re.search(out):
        mappings.append("andar/indoor mein -> indoor")
        out = andar_re.sub("indoor", out)

    chhat_re = re.compile(r"\b(?:chhat\s*(?:pe|par)?|bahar\s+open\s+air|rooftop\s+pe|rooftop\s+mein)\b", re.IGNORECASE)
    if chhat_re.search(out):
        mappings.append("chhat/bahar -> rooftop")
        out = chhat_re.sub("rooftop", out)

    majlis_re = re.compile(r"\b(?:private\s+majlis|family\s+majlis|majlis\s+room|parda\s+hall|family\s+cabin)\b", re.IGNORECASE)
    if majlis_re.search(out):
        mappings.append("majlis/cabin -> private family majlis")
        out = majlis_re.sub("private family majlis", out)

    return out, mappings


# ---------------------------------------------------------------------------
# Intra-Utterance Self-Correction / Speech Repair Parser
# ---------------------------------------------------------------------------

# Markers indicating the caller is correcting something they just said
REPAIR_MARKERS_ORDERED = [
    "nahi balkay",
    "sorry i meant",
    "sorry i mean",
    "no wait make it",
    "wait no make it",
    "wait actually",
    "actually wait",
    "no wait",
    "wait no",
    "scratch that",
    "cancel that",
    "my bad",
    "i meant",
    "i mean",
    "or rather",
    "balkay",
    "sorry",
    "actually",
]

# Slot-like patterns we can selectively prune from the pre-repair prefix
_PARTY_SIZE_TOKEN_RE = re.compile(
    r"\b(?:for\s+)?(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s*"
    r"(?:people|persons|guests|members|seats|of\s+us|pax)\b"
    r"|\btable\s+for\s+(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.IGNORECASE,
)

_TIME_TOKEN_RE = re.compile(
    r"\b(?:at\s+|around\s+|by\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock|baje)\b"
    r"|\b(?:at|around)\s+\d{1,2}(?::\d{2})\b",
    re.IGNORECASE,
)

_DATE_TOKEN_RE = re.compile(
    r"\b(?:today|tomorrow|day\s+after\s+tomorrow|yesterday|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_SEATING_TOKEN_RE = re.compile(
    r"\b(?:indoor(?:\s+ac)?|rooftop|terrace|outdoor|outside|inside|private(?:\s+family)?\s+majlis|majlis|private\s+room)\b",
    re.IGNORECASE,
)


def _resolve_self_correction(text: str) -> Tuple[str, Optional[str], bool, Optional[str]]:
    """
    Detects intra-utterance self-corrections and returns:
    (effective_text, discarded_prefix, repair_detected, repair_marker)

    Handles:
    1. Urdu/English negated slot replacements:
       "Nahi 8 baje nahi, 9 baje kar do" / "not 8 pm, make it 9 pm"
       "4 people nahi, 6 people"
    2. Inline parenthetical / dashed repairs:
       "Book a table for 3 people—sorry, 5 people—tomorrow at 8 PM"
    3. Clause-level repair markers ("no wait", "actually", "sorry", "I mean", "nahi balkay"):
       Removes only the superseded slot values from the prefix while preserving
       non-conflicting slots (e.g., keeping "tomorrow" when only party_size changed).
    """
    working = text

    # Case 1: Explicit Urdu/English negation of a time or party size:
    # e.g. "Nahi 8 baje nahi, 9 baje kar do" or "8 pm nahi 9 pm" or "not at 7 pm, at 8 pm"
    negated_time_re = re.compile(
        r"(?:\b(?:nahi|no|not)\s+)?(\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|baje|o'?clock)?)\s+nahi\s*[,;—\-]*\s*"
        r"(?:balkay\s+)?(\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|baje|o'?clock)?)",
        re.IGNORECASE,
    )
    m_neg_time = negated_time_re.search(working)
    if m_neg_time:
        old_val = m_neg_time.group(1).strip()
        new_val = m_neg_time.group(2).strip()
        effective = working[: m_neg_time.start()] + new_val + working[m_neg_time.end() :]
        effective = re.sub(r"\s+", " ", effective).strip(" ,;-—")
        return effective, old_val, True, "nahi (negated time repair)"

    not_x_but_y_re = re.compile(
        r"\bnot\s+((?:at\s+|for\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|baje|people|guests|persons)?)\s*[,;—\-]+\s*"
        r"(?:make\s+it\s+|balkay\s+|instead\s+)?((?:at\s+|for\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|baje|people|guests|persons)?)",
        re.IGNORECASE,
    )
    m_not = not_x_but_y_re.search(working)
    if m_not:
        old_val = m_not.group(1).strip()
        new_val = m_not.group(2).strip()
        effective = working[: m_not.start()] + new_val + working[m_not.end() :]
        effective = re.sub(r"\s+", " ", effective).strip(" ,;-—")
        return effective, old_val, True, "not X -> Y"

    # Case 2: Inline targeted repair for party size:
    # e.g., "for 3 people—sorry, 5 people—tomorrow at 8 PM"
    # or "for 3, sorry 5 people tomorrow"
    inline_party_repair_re = re.compile(
        r"\b(for\s+)?(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
        r"(?:people|persons|guests|members|seats)?\s*"
        r"(?:[,;—\-]+\s*|\s+)"
        r"(sorry\s+i\s+meant|sorry\s+i\s+mean|no\s+wait|wait\s+no|i\s+meant|i\s+mean|sorry|actually|nahi\s+balkay|balkay|my\s+bad|or\s+rather)"
        r"(?:\s+make\s+it|\s+for)?\s*[,;—\-]*\s*"
        r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
        r"(people|persons|guests|members|seats)\b",
        re.IGNORECASE,
    )
    m_inline_party = inline_party_repair_re.search(working)
    if m_inline_party:
        prefix_for = m_inline_party.group(1) or "for "
        old_num = m_inline_party.group(2)
        marker = m_inline_party.group(3).lower()
        new_num = m_inline_party.group(4)
        noun = m_inline_party.group(5)
        replacement = f"{prefix_for}{new_num} {noun}"
        effective = (
            working[: m_inline_party.start()]
            + replacement
            + working[m_inline_party.end() :]
        )
        effective = re.sub(r"[—\-]+", " ", effective)
        effective = re.sub(r"\s+", " ", effective).strip()
        return effective, f"{old_num} {noun}", True, marker

    # Case 3: General clause-level repair markers
    # Ignore polite "sorry" at the very beginning of an utterance (e.g. "Sorry, do you have parking?")
    marker_pattern = "|".join(re.escape(m) for m in REPAIR_MARKERS_ORDERED)
    clause_repair_re = re.compile(
        rf"^(.+?)(?:[,;—\-]+\s*|\s+)({marker_pattern})(?:[,;—\-]+\s*|\s+)(.+)$",
        re.IGNORECASE,
    )
    m_clause = clause_repair_re.match(working)
    if m_clause:
        prefix = m_clause.group(1).strip(" ,;—-")
        marker = m_clause.group(2).strip().lower()
        suffix = m_clause.group(3).strip(" ,;—-")

        # Guard: if prefix has no content words (e.g., "uh", "well", "um"), don't treat as slot repair
        if len(prefix.split()) >= 1 and suffix:
            pruned_prefix = prefix
            discarded_parts: List[str] = []

            # Check which slot domains appear in suffix, and remove superseded ones from prefix
            if _PARTY_SIZE_TOKEN_RE.search(suffix):
                for found in _PARTY_SIZE_TOKEN_RE.findall(pruned_prefix):
                    discarded_parts.append(str(found))
                pruned_prefix = _PARTY_SIZE_TOKEN_RE.sub(" ", pruned_prefix)

            if _TIME_TOKEN_RE.search(suffix):
                for m_t in _TIME_TOKEN_RE.finditer(pruned_prefix):
                    discarded_parts.append(m_t.group(0))
                pruned_prefix = _TIME_TOKEN_RE.sub(" ", pruned_prefix)

            if _DATE_TOKEN_RE.search(suffix):
                for m_d in _DATE_TOKEN_RE.finditer(pruned_prefix):
                    discarded_parts.append(m_d.group(0))
                pruned_prefix = _DATE_TOKEN_RE.sub(" ", pruned_prefix)

            if _SEATING_TOKEN_RE.search(suffix):
                for m_s in _SEATING_TOKEN_RE.finditer(pruned_prefix):
                    discarded_parts.append(m_s.group(0))
                pruned_prefix = _SEATING_TOKEN_RE.sub(" ", pruned_prefix)

            # What if suffix is a bare number correction like "make it 6" or "no wait 8"?
            bare_num_suffix_re = re.compile(
                r"^(?:no\s*,?\s*)?(?:make\s+it\s+|for\s+|at\s+)?(\d{1,2})\b(.*)$",
                re.IGNORECASE,
            )
            m_bare = bare_num_suffix_re.match(suffix)
            if m_bare and not discarded_parts:
                # Check if prefix had a party size or time
                if _PARTY_SIZE_TOKEN_RE.search(pruned_prefix) and not _TIME_TOKEN_RE.search(pruned_prefix):
                    for m_p in _PARTY_SIZE_TOKEN_RE.finditer(pruned_prefix):
                        discarded_parts.append(m_p.group(0))
                    pruned_prefix = _PARTY_SIZE_TOKEN_RE.sub(f" {m_bare.group(1)} people ", pruned_prefix)
                    effective = re.sub(r"\s+", " ", f"{pruned_prefix} {m_bare.group(2)}").strip()
                    return effective, ", ".join(discarded_parts) or prefix, True, marker

            pruned_prefix = re.sub(r"\s+", " ", pruned_prefix).strip(" ,;—-")
            combined = f"{pruned_prefix} {suffix}".strip() if pruned_prefix else suffix
            combined = re.sub(r"\s+", " ", combined).strip()
            return combined, prefix, True, marker

    return working, None, False, None


# ---------------------------------------------------------------------------
# Main Normalizer Entrypoint
# ---------------------------------------------------------------------------

def normalize_utterance(raw_text: str) -> NormalizationResult:
    """
    Runs the full Stage 1 normalization pipeline on a caller's raw utterance:
    1. Whitespace & punctuation cleanup + filler removal
    2. STT homophone repair ("for people" -> "4 people", "ate pm" -> "8 pm")
    3. Pakistani Minglish / Roman Urdu normalization ("kal raat 8 baje 4 logon" -> "tomorrow 8 pm 4 people")
    4. English number-word conversion in slot contexts ("four people" -> "4 people")
    5. Intra-utterance self-correction resolution ("3 people—sorry, 5 people" -> "5 people")
    """
    if not raw_text or not raw_text.strip():
        return NormalizationResult(
            raw_text=raw_text or "",
            cleaned_text="",
            effective_text="",
        )

    text = raw_text.strip()

    # Replace em-dashes and multiple hyphens with spaced em-dash so tokens separate cleanly
    text = re.sub(r"[\u2014\u2013]+", " — ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove disfluency fillers
    for pat in FILLER_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Step 1.5: Transliterate native Urdu Perso-Arabic script & numerals (if ur-PK STT was used)
    text, script_mappings = _transliterate_urdu_script(text)

    # Step 2: Homophone repair
    homophone_repairs: List[str] = []
    for pattern, replacement, label in HOMOPHONE_RULES:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            homophone_repairs.append(label)

    # Step 3: Pakistani Minglish / Roman Urdu normalization
    text, minglish_mappings = _apply_minglish_normalization(text)
    if script_mappings:
        minglish_mappings = script_mappings + minglish_mappings

    # Step 4: Convert English number words before guest/time keywords ("four people" -> "4 people", "seven thirty" -> "7:30")
    eng_time_thirty_re = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+thirty\b",
        re.IGNORECASE,
    )
    def _replace_eng_thirty(m: re.Match) -> str:
        hr = ENGLISH_NUMBER_MAP[m.group(1).lower()]
        return f"{hr}:30"

    text = eng_time_thirty_re.sub(_replace_eng_thirty, text)

    eng_guest_re = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b"
        r"(?=\s+(?:people|persons|guests|members|seats|of\s+us|tables?))",
        re.IGNORECASE,
    )
    def _replace_eng_guest(m: re.Match) -> str:
        return str(ENGLISH_NUMBER_MAP[m.group(1).lower()])

    text = eng_guest_re.sub(_replace_eng_guest, text)

    cleaned_text = re.sub(r"\s+", " ", text).strip()

    # Step 5: Resolve intra-utterance self-corrections
    effective_text, discarded_prefix, repair_detected, repair_marker = _resolve_self_correction(cleaned_text)

    return NormalizationResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        effective_text=effective_text,
        discarded_prefix=discarded_prefix,
        repair_detected=repair_detected,
        repair_marker=repair_marker,
        homophone_repairs=homophone_repairs,
        minglish_mappings=minglish_mappings,
    )
