"""
Stage 3: Context-Aware Multi-Slot Extractor with Fuzzy Matching, Pakistani Phone
Normalization, Relative Date/Time Resolution, and Bare-Number Disambiguation.

100% deterministic (Regex + RapidFuzz + Contextual State Rules). No LLMs.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from rapidfuzz import fuzz, process

from app.brain.normalizer import ENGLISH_NUMBER_MAP, URDU_NUMBER_MAP
from app.models import (
    CLOSING_HOUR,
    CLOSING_MINUTE,
    MAX_PARTY_SIZE,
    OPENING_HOUR,
    SeatingArea,
    SlotExtractionResult,
)

PKT_TZ = ZoneInfo("Asia/Karachi")


# ---------------------------------------------------------------------------
# Seating Preference Lexicon & Fuzzy Targets
# ---------------------------------------------------------------------------

SEATING_SYNONYMS: Dict[str, str] = {
    "indoor ac": SeatingArea.INDOOR_AC.value,
    "indoor": SeatingArea.INDOOR_AC.value,
    "indoors": SeatingArea.INDOOR_AC.value,
    "inside": SeatingArea.INDOOR_AC.value,
    "ac hall": SeatingArea.INDOOR_AC.value,
    "air conditioned": SeatingArea.INDOOR_AC.value,
    "main hall": SeatingArea.INDOOR_AC.value,
    "rooftop": SeatingArea.ROOFTOP.value,
    "roof top": SeatingArea.ROOFTOP.value,
    "terrace": SeatingArea.ROOFTOP.value,
    "open air": SeatingArea.ROOFTOP.value,
    "outdoor": SeatingArea.ROOFTOP.value,
    "outdoors": SeatingArea.ROOFTOP.value,
    "outside": SeatingArea.ROOFTOP.value,
    "courtyard": SeatingArea.ROOFTOP.value,
    "margalla view": SeatingArea.ROOFTOP.value,
    "private family majlis": SeatingArea.PRIVATE_MAJLIS.value,
    "private majlis": SeatingArea.PRIVATE_MAJLIS.value,
    "family majlis": SeatingArea.PRIVATE_MAJLIS.value,
    "majlis": SeatingArea.PRIVATE_MAJLIS.value,
    "majlis room": SeatingArea.PRIVATE_MAJLIS.value,
    "private room": SeatingArea.PRIVATE_MAJLIS.value,
    "family room": SeatingArea.PRIVATE_MAJLIS.value,
    "family cabin": SeatingArea.PRIVATE_MAJLIS.value,
    "parda hall": SeatingArea.PRIVATE_MAJLIS.value,
}

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

SPOKEN_DIGIT_MAP = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


# ---------------------------------------------------------------------------
# Helper: Reference Date Resolution
# ---------------------------------------------------------------------------

def get_reference_date(ref_date_str: Optional[str] = None) -> dt.date:
    if ref_date_str:
        try:
            return dt.date.fromisoformat(ref_date_str)
        except ValueError:
            pass
    return dt.datetime.now(PKT_TZ).date()


# ---------------------------------------------------------------------------
# 1. Booking Reference Extraction
# ---------------------------------------------------------------------------

_BOOKING_REF_RE = re.compile(r"\b(?:DD|HG|D\s+D|H\s+G)[-\s]?(\d{3,6})\b", re.IGNORECASE)


def extract_booking_reference(text: str) -> Optional[str]:
    m = _BOOKING_REF_RE.search(text)
    if m:
        return f"DD-{m.group(1)}"
    # Also support spoken digit reference codes like "DD one zero zero one" or "H G one zero zero one"
    spoken_ref_re = re.compile(
        r"\b(?:dd|hg|d\s+d|h\s+g)[-\s]+((?:(?:zero|oh|o|one|two|three|four|five|six|seven|eight|nine|\d)\s*){3,6})\b",
        re.IGNORECASE,
    )
    sm = spoken_ref_re.search(text)
    if sm:
        raw_digits = sm.group(1).lower().split()
        digits = "".join(SPOKEN_DIGIT_MAP.get(tok, tok) for tok in raw_digits)
        if digits.isdigit() and 3 <= len(digits) <= 6:
            return f"DD-{digits}"
    return None


# ---------------------------------------------------------------------------
# 2. Phone Number Extraction (Digits + Spoken Words + Double/Triple)
# ---------------------------------------------------------------------------

def _convert_spoken_digits(text: str) -> str:
    """
    Converts spoken phone sequences like:
    'zero three zero zero one two three four five six seven'
    or 'zero three two one triple five one two three four'
    into contiguous digit strings while preserving colons and hyphens.
    """
    tokens = re.findall(r"[a-z0-9:+-]+", text.lower())
    out_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("double", "triple") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            digit = SPOKEN_DIGIT_MAP.get(nxt) or (nxt if nxt.isdigit() and len(nxt) == 1 else None)
            if digit is not None:
                repeat = 2 if tok == "double" else 3
                out_tokens.append(digit * repeat)
                i += 2
                continue
        if tok in SPOKEN_DIGIT_MAP:
            out_tokens.append(SPOKEN_DIGIT_MAP[tok])
        else:
            out_tokens.append(tok)
        i += 1

    # Collapse adjacent digit groups separated by spaces when total digits >= 10
    joined = " ".join(out_tokens)

    def _collapse_digits(m: re.Match) -> str:
        collapsed = re.sub(r"\s+", "", m.group(0))
        return collapsed if len(collapsed) >= 10 else m.group(0)

    return re.sub(r"\b(?:\d{1,4}\s+){3,}\d{1,4}\b", _collapse_digits, joined)


_PK_PHONE_RE = re.compile(
    r"(?:\+92|0092|92|0)[\s\-]?(3\d{2}|51|21|42)[\s\-]?(\d{7})\b"
)


def extract_phone_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (formatted_phone, validation_error).
    Canonical format for Pakistani mobile: 03XX-XXXXXXX (or 051-XXXXXXX for landline).
    """
    expanded = _convert_spoken_digits(text)
    m = _PK_PHONE_RE.search(expanded)
    if m:
        prefix = m.group(1)
        rest = m.group(2)
        return f"0{prefix}-{rest}", None

    # Check if there is a 10-11 digit sequence that looks like an attempted phone number
    raw_digits_m = re.search(r"\b0\d{9,10}\b", expanded)
    if raw_digits_m:
        raw_d = raw_digits_m.group(0)
        if len(raw_d) == 11 and raw_d.startswith("03"):
            return f"{raw_d[:4]}-{raw_d[4:]}", None
        if len(raw_d) == 10 and raw_d.startswith("051"):
            return f"{raw_d[:3]}-{raw_d[3:]}", None

    return None, None


# ---------------------------------------------------------------------------
# 3. Date Extraction
# ---------------------------------------------------------------------------

def extract_date(text: str, ref_date: dt.date) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (iso_date_str, validation_error, matched_phrase).
    """
    low = text.lower()

    # Check past date keywords first
    past_re = re.compile(
        r"\b(?:yesterday|last\s+night|last\s+week|last\s+month|pichle\s+kal|guzra\s+hua\s+kal|\d+\s+days?\s+ago)\b",
        re.IGNORECASE,
    )
    m_past = past_re.search(low)
    if m_past:
        return None, "past_date", m_past.group(0)

    # Check "day after tomorrow" before "tomorrow"
    if re.search(r"\bday\s+after\s+tomorrow\b", low):
        target = ref_date + dt.timedelta(days=2)
        return target.isoformat(), None, "day after tomorrow"

    if re.search(r"\b(?:tomorrow|tmrw)\b", low):
        target = ref_date + dt.timedelta(days=1)
        return target.isoformat(), None, "tomorrow"

    if re.search(r"\b(?:today|tonight|this\s+evening|this\s+afternoon)\b", low):
        return ref_date.isoformat(), None, "today"

    # Check explicit ISO date YYYY-MM-DD
    m_iso = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", low)
    if m_iso:
        try:
            parsed = dt.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
            if parsed < ref_date:
                return None, "past_date", m_iso.group(0)
            return parsed.isoformat(), None, m_iso.group(0)
        except ValueError:
            pass

    # Check Month Day or Day Month ("October 28", "28th October")
    month_pattern = "|".join(MONTH_NAMES.keys())
    m_md = re.search(rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", low)
    m_dm = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_pattern})\b", low)
    if m_md or m_dm:
        if m_md:
            mon_name, day_str = m_md.group(1), m_md.group(2)
            raw_m = m_md.group(0)
        else:
            assert m_dm is not None
            day_str, mon_name = m_dm.group(1), m_dm.group(2)
            raw_m = m_dm.group(0)
        month_num = MONTH_NAMES[mon_name]
        day_num = int(day_str)
        try:
            candidate = dt.date(ref_date.year, month_num, day_num)
            if candidate < ref_date:
                return None, "past_date", raw_m
            return candidate.isoformat(), None, raw_m
        except ValueError:
            pass

    # Check weekday names ("friday", "this saturday", "next monday")
    for day_name, weekday_idx in WEEKDAYS.items():
        m_wd = re.search(rf"\b(next\s+|this\s+)?{day_name}\b", low)
        if m_wd:
            days_ahead = (weekday_idx - ref_date.weekday()) % 7
            if days_ahead == 0 and m_wd.group(1) and "next" in m_wd.group(1):
                days_ahead = 7
            target = ref_date + dt.timedelta(days=days_ahead)
            return target.isoformat(), None, m_wd.group(0)

    return None, None, None


# ---------------------------------------------------------------------------
# 4. Time Extraction & Restaurant Hours Validation
# ---------------------------------------------------------------------------

def _validate_and_format_time(
    hour: int,
    minute: int,
    meridiem: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Converts (hour, minute, meridiem) into 24-hour 'HH:MM' and validates against
    Haveli Grill & Courtyard operating hours (12:00 PM – 11:30 PM).
    """
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None, "out_of_hours"

    mer = (meridiem or "").lower().replace(".", "").strip()

    if mer == "am":
        # 12:00 AM is midnight (00:00), 1 AM .. 11 AM are 01:00 .. 11:00 -> all closed!
        return None, "out_of_hours"
    elif mer == "pm":
        if hour < 12:
            hour_24 = hour + 12
        elif hour == 12:
            hour_24 = 12
        else:
            hour_24 = hour
    else:
        # No explicit AM/PM provided:
        # Operating hours are 12:00 (noon) to 23:30 (11:30 PM).
        if hour == 12:
            hour_24 = 12
        elif 1 <= hour <= 11:
            hour_24 = hour + 12
        elif 13 <= hour <= 23:
            hour_24 = hour
        else:
            return None, "out_of_hours"

    # Check operating window: 12:00 to 23:30
    total_mins = hour_24 * 60 + minute
    open_mins = OPENING_HOUR * 60
    close_mins = CLOSING_HOUR * 60 + CLOSING_MINUTE
    if total_mins < open_mins or total_mins > close_mins:
        return None, "out_of_hours"

    return f"{hour_24:02d}:{minute:02d}", None


def extract_time(
    text: str,
    expected_slot: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts reservation time in 24-hour 'HH:MM' format.
    Returns (time_24h, validation_error, raw_match).
    """
    # Strip phone numbers and booking refs first so their digits aren't misread as times
    scrubbed = _PK_PHONE_RE.sub(" ", _convert_spoken_digits(text))
    scrubbed = _BOOKING_REF_RE.sub(" ", scrubbed)
    # Strip explicit "<N> people/guests/persons" so "8 people" is never parsed as a time
    scrubbed = re.sub(
        r"\b\d{1,3}\s*(?:people|persons|guests|members|seats|of\s+us|pax|tables?)\b",
        " ",
        scrubbed,
        flags=re.IGNORECASE,
    )

    # Pattern A: Explicit meridiem or 'baje' / 'o'clock': e.g. "7:30 pm", "8 pm", "3:00 am", "8 baje"
    explicit_time_re = re.compile(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.|o'?clock|baje)\b",
        re.IGNORECASE,
    )
    matches = list(explicit_time_re.finditer(scrubbed))
    if matches:
        # Take the last match in case of "from 7 pm to 8:30 pm"
        m = matches[-1]
        hr = int(m.group(1))
        mn = int(m.group(2)) if m.group(2) else 0
        marker = m.group(3).lower()
        mer = "am" if marker.startswith("a") else ("pm" if marker.startswith("p") else None)
        t_val, err = _validate_and_format_time(hr, mn, mer)
        return t_val, err, m.group(0)

    # Pattern B: "HH:MM" with colon (e.g. "7:30", "19:30", "20:00")
    colon_time_re = re.compile(r"\b(\d{1,2}):(\d{2})\b")
    m_colon = colon_time_re.findall(scrubbed)
    if m_colon:
        hr_s, mn_s = m_colon[-1]
        t_val, err = _validate_and_format_time(int(hr_s), int(mn_s), None)
        return t_val, err, f"{hr_s}:{mn_s}"

    # Pattern C: Preposition + hour ("at 7", "around 8", "by 9")
    prep_time_re = re.compile(r"\b(?:at|around|by)\s+(\d{1,2})\b", re.IGNORECASE)
    m_prep = list(prep_time_re.finditer(scrubbed))
    if m_prep:
        hr = int(m_prep[-1].group(1))
        t_val, err = _validate_and_format_time(hr, 0, None)
        return t_val, err, m_prep[-1].group(0)

    # Pattern D: Bare number when expected_slot == "time"
    if expected_slot == "time":
        bare_hr_re = re.compile(r"^\s*(?:make\s+it\s+|for\s+|at\s+)?(\d{1,2})\s*$")
        m_bare = bare_hr_re.match(scrubbed.strip())
        if m_bare:
            hr = int(m_bare.group(1))
            t_val, err = _validate_and_format_time(hr, 0, None)
            return t_val, err, m_bare.group(0)

    return None, None, None


# ---------------------------------------------------------------------------
# 5. Party Size Extraction & Bare-Number Ambiguity Detection
# ---------------------------------------------------------------------------

def extract_party_size(
    text: str,
    expected_slot: Optional[str] = None,
    time_already_extracted: bool = False,
    current_party_size: Optional[int] = None,
) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Returns:
    (party_size, validation_error, ambiguous_number, ambiguity_reason, raw_match)
    """
    scrubbed = _PK_PHONE_RE.sub(" ", _convert_spoken_digits(text))
    scrubbed = _BOOKING_REF_RE.sub(" ", scrubbed)

    # Remove time expressions so "8 pm" or "7:30" isn't confused with party size
    scrubbed_no_time = re.sub(
        r"\b\d{1,2}(?::\d{2})\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock|baje)?\b"
        r"|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock|baje)\b"
        r"|\b(?:at|around|by)\s+\d{1,2}\b",
        " ",
        scrubbed,
        flags=re.IGNORECASE,
    )

    # Pattern A0: Additive guest counts ("4 adults and 2 kids", "4 adults aur 2 bachay")
    additive_re = re.compile(
        r"\b(\d{1,2})\s*(?:adults?|grownups?|baray|people|men|women)\s*(?:and|aur|\+|plus)\s*"
        r"(\d{1,2})\s*(?:kids?|children|bachay|bache|babies|infants?)\b",
        re.IGNORECASE,
    )
    m_add = additive_re.search(scrubbed_no_time)
    if m_add:
        total = int(m_add.group(1)) + int(m_add.group(2))
        if total > MAX_PARTY_SIZE:
            return None, "exceeds_max_capacity", None, None, str(total)
        return total, None, None, None, m_add.group(0)

    # Pattern A1: Relative delta ("add 2 more people", "2 more guests joining") when current_party_size is known
    delta_re = re.compile(
        r"\b(?:add\s+)?(\d{1,2})\s+more\s+(?:people|persons|guests|members|friends|log)\b",
        re.IGNORECASE,
    )
    m_delta = delta_re.search(scrubbed_no_time)
    if m_delta and current_party_size is not None and not re.search(r"\bmake\s+it\s+\d+", scrubbed_no_time, re.IGNORECASE):
        total = int(current_party_size) + int(m_delta.group(1))
        if total > MAX_PARTY_SIZE:
            return None, "exceeds_max_capacity", None, None, str(total)
        return total, None, None, None, m_delta.group(0)

    # Pattern A: Explicit guest noun ("4 people", "6 guests", "party of 8", "table for 4", "3 of us")
    explicit_guest_re = re.compile(
        r"\b(?:for\s+|party\s+of\s+|group\s+of\s+)?(\d{1,3})\s*"
        r"(?:people|persons|guests|members|seats|of\s+us|pax|adults|friends|family\s+members)\b"
        r"|\b(?:table|reservation|spot|seating)\s+for\s+(\d{1,3})\b"
        r"|\b(?:party|group)\s+of\s+(\d{1,3})\b",
        re.IGNORECASE,
    )
    matches = list(explicit_guest_re.finditer(scrubbed_no_time))
    if matches:
        m = matches[-1]
        num_str = m.group(1) or m.group(2) or m.group(3)
        val = int(num_str)
        if val <= 0:
            return None, "invalid_party_size", None, None, m.group(0)
        if val > MAX_PARTY_SIZE:
            return None, "exceeds_max_capacity", None, None, str(val)
        return val, None, None, None, m.group(0)

    # Pattern B: "make it <N>" when changing party size or expected_slot == "party_size"
    if expected_slot == "party_size":
        bare_re = re.compile(r"\b(\d{1,3})\b")
        m_bare = bare_re.search(scrubbed_no_time)
        if m_bare:
            val = int(m_bare.group(1))
            if val <= 0:
                return None, "invalid_party_size", None, None, m_bare.group(0)
            if val > MAX_PARTY_SIZE:
                return None, "exceeds_max_capacity", None, None, str(val)
            return val, None, None, None, m_bare.group(0)

    # Pattern C: Ambiguous bare number in booking request ("Book for 8", "Reserve for 6", or bare "6")
    # when neither guest noun nor time marker is present
    if not time_already_extracted and expected_slot is None:
        ambig_re = re.compile(
            r"^(?:please\s+|can\s+you\s+|i\s+want\s+to\s+)?"
            r"(?:book|reserve|make\s+a\s+booking|make\s+a\s+reservation)?\s*"
            r"(?:for\s+)?(\d{1,2})\s*(?:please|today|tomorrow|tonight)?$",
            re.IGNORECASE,
        )
        m_ambig = ambig_re.match(scrubbed_no_time.strip())
        if m_ambig:
            num = int(m_ambig.group(1))
            if 1 <= num <= 12:
                return None, None, num, "bare_number_guest_or_time", m_ambig.group(0)
            elif 13 <= num <= MAX_PARTY_SIZE:
                return num, None, None, None, m_ambig.group(0)
            elif num > MAX_PARTY_SIZE:
                return None, "exceeds_max_capacity", None, None, str(num)

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# 6. Seating Preference Extraction (Exact + RapidFuzz)
# ---------------------------------------------------------------------------

def extract_seating_preference(text: str) -> Optional[str]:
    low = text.lower()

    # Check longest exact phrases first (note: if the caller said "instead of indoor", ignore the "instead of X" part)
    cleaned_for_seating = re.sub(
        r"\binstead\s+of\s+(?:indoor(?:\s+ac)?|rooftop|terrace|private(?:\s+family)?\s+majlis|majlis)\b",
        " ",
        low,
        flags=re.IGNORECASE,
    )
    cleaned_for_seating = re.sub(
        r"\b(?:indoor(?:\s+ac)?|rooftop|terrace|majlis)\s+(?:nahi|not)\b",
        " ",
        cleaned_for_seating,
        flags=re.IGNORECASE,
    )

    for phrase in sorted(SEATING_SYNONYMS.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", cleaned_for_seating):
            return SEATING_SYNONYMS[phrase]

    # Fuzzy match token windows using RapidFuzz for minor typos / STT distortions
    words = re.findall(r"[a-z]+", cleaned_for_seating)
    candidates = list(SEATING_SYNONYMS.keys())
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i : i + n])
            if len(chunk) < 5:
                continue
            match = process.extractOne(chunk, candidates, scorer=fuzz.ratio)
            if match and match[1] >= 86:
                return SEATING_SYNONYMS[match[0]]

    return None


# ---------------------------------------------------------------------------
# 7. Customer Name Extraction
# ---------------------------------------------------------------------------

_NON_NAME_STOPWORDS = {
    "yes", "no", "yeah", "nope", "haan", "nahi", "jee", "ji", "bilkul",
    "indoor", "rooftop", "terrace", "majlis", "private", "family",
    "today", "tomorrow", "tonight", "yesterday", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "book", "table", "reserve", "reservation", "cancel", "modify",
    "parking", "valet", "menu", "halal", "cards", "credit", "weather",
    "rain", "open", "close", "hours", "location", "address", "hello",
    "salam", "hi", "hey", "thanks", "thank", "you", "bye", "goodbye",
    "people", "guests", "persons", "please", "confirm", "okay", "ok",
    "make", "change", "wait", "actually", "sorry", "calling", "hungry",
}

_EXPLICIT_NAME_PATTERNS = [
    re.compile(
        r"\b(?:my\s+name\s+is|this\s+is|i\s+am|i'm|it\s+is\s+under|it's\s+under|"
        r"under(?:\s+the\s+name(?:\s+of)?)?|book\s+under|reservation\s+under|"
        r"change\s+(?:the\s+)?name\s+to|mera\s+naam|naam)\s+"
        r"([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,2}?)"
        r"(?=\s+(?:and|my|phone|number|contact|book|reserve|for|at|on|tomorrow|today|indoor|rooftop|hai|he|\d)|[,.;!?]|$)",
        re.IGNORECASE,
    ),
]


def _clean_candidate_name(raw_name: str) -> Optional[str]:
    tokens = [t for t in re.findall(r"[A-Za-z]+", raw_name) if t.lower() not in {"hai", "he", "ji", "jee", "please"}]
    if not tokens or len(tokens) > 3:
        return None
    if any(t.lower() in _NON_NAME_STOPWORDS for t in tokens):
        return None
    return " ".join(t.capitalize() for t in tokens)


def extract_customer_name(
    text: str,
    expected_slot: Optional[str] = None,
) -> Optional[str]:
    for pat in _EXPLICIT_NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cleaned = _clean_candidate_name(m.group(1))
            if cleaned:
                return cleaned

    # If the caller says "<Name> <Phone>" in one turn when prompted for name/phone
    # e.g., "Kashif Mehmood 0312 4455667"
    name_before_phone_re = re.compile(
        r"^\s*([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,2})\s+(?:and\s+(?:my\s+)?(?:number|phone)\s+is\s+)?(?:03|\+92|zero\s+three)",
        re.IGNORECASE,
    )
    m_np = name_before_phone_re.search(text)
    if m_np:
        cleaned = _clean_candidate_name(m_np.group(1))
        if cleaned:
            return cleaned

    # If the assistant specifically asked for customer_name, accept a bare 1-3 word name
    if expected_slot == "customer_name":
        scrubbed = _PK_PHONE_RE.sub("", text).strip(" .,!?")
        if re.match(r"^[A-Za-z]+(?:\s+[A-Za-z]+){0,2}$", scrubbed):
            cleaned = _clean_candidate_name(scrubbed)
            if cleaned:
                return cleaned

    return None


# ---------------------------------------------------------------------------
# 8. Special Requests Extraction
# ---------------------------------------------------------------------------

_SPECIAL_REQUEST_KEYWORDS = [
    ("high chair", "High chair for child"),
    ("baby chair", "High chair for child"),
    ("birthday cake", "Birthday cake arrangement"),
    ("birthday", "Birthday table setup"),
    ("anniversary", "Anniversary table setup"),
    ("wheelchair", "Wheelchair accessible seating"),
    ("window seat", "Window table preference"),
    ("window table", "Window table preference"),
    ("quiet table", "Quiet corner table"),
    ("quiet corner", "Quiet corner table"),
    ("nut allergy", "Nut allergy alert for kitchen"),
    ("allergy", "Food allergy alert"),
]


def extract_special_requests(text: str) -> Optional[str]:
    low = text.lower()
    found: List[str] = []
    for kw, label in _SPECIAL_REQUEST_KEYWORDS:
        if kw in low and label not in found:
            found.append(label)
    return ", ".join(found) if found else None


# ---------------------------------------------------------------------------
# Main Multi-Slot Extraction Entrypoint
# ---------------------------------------------------------------------------

def extract_slots(
    effective_text: str,
    raw_text: Optional[str] = None,
    expected_slot: Optional[str] = None,
    current_slots: Optional[Dict[str, Any]] = None,
    reference_date: Optional[str] = None,
) -> SlotExtractionResult:
    """
    Runs context-aware extraction across all 8 reservation slot types.
    Uses `expected_slot` and `current_slots` to disambiguate bare numbers
    or flag ambiguity when neither party_size nor time is known.
    """
    ref_date = get_reference_date(reference_date)
    current = current_slots or {}
    slots: Dict[str, Any] = {}
    validation_errors: Dict[str, str] = {}
    raw_matches: Dict[str, str] = {}

    # 1. Booking reference
    b_ref = extract_booking_reference(effective_text) or (
        extract_booking_reference(raw_text) if raw_text else None
    )
    if b_ref:
        slots["booking_reference"] = b_ref
        raw_matches["booking_reference"] = b_ref

    # 2. Phone number
    phone, phone_err = extract_phone_number(effective_text)
    if not phone and raw_text:
        phone, phone_err = extract_phone_number(raw_text)
    if phone:
        slots["phone_number"] = phone
        raw_matches["phone_number"] = phone
    elif phone_err:
        validation_errors["phone_number"] = phone_err

    # 3. Date
    date_val, date_err, date_raw = extract_date(effective_text, ref_date)
    if date_val:
        slots["date"] = date_val
        raw_matches["date"] = date_raw or date_val
    elif date_err:
        validation_errors["date"] = date_err
        if date_raw:
            raw_matches["date"] = date_raw

    # 4. Time
    time_val, time_err, time_raw = extract_time(effective_text, expected_slot=expected_slot)
    if time_val:
        slots["time"] = time_val
        raw_matches["time"] = time_raw or time_val
    elif time_err:
        validation_errors["time"] = time_err
        if time_raw:
            raw_matches["time"] = time_raw

    # 5. Party size & Bare-number disambiguation
    p_size, p_err, ambig_num, ambig_reason, p_raw = extract_party_size(
        effective_text,
        expected_slot=expected_slot,
        time_already_extracted=(time_val is not None or time_err is not None),
        current_party_size=current.get("party_size"),
    )

    # Disambiguate bare number using existing session state if one of (party_size, time) is already known
    if ambig_num is not None:
        has_party = current.get("party_size") is not None
        has_time = current.get("time") is not None or "time" in slots
        if has_party and not has_time:
            t_resolved, t_res_err = _validate_and_format_time(ambig_num, 0, None)
            if t_resolved:
                slots["time"] = t_resolved
                raw_matches["time"] = str(ambig_num)
                ambig_num = None
                ambig_reason = None
        elif has_time and not has_party:
            slots["party_size"] = ambig_num
            raw_matches["party_size"] = str(ambig_num)
            ambig_num = None
            ambig_reason = None

    if p_size is not None:
        slots["party_size"] = p_size
        raw_matches["party_size"] = p_raw or str(p_size)
    elif p_err:
        validation_errors["party_size"] = p_err
        if p_raw:
            raw_matches["party_size"] = p_raw

    # 6. Seating preference
    seating = extract_seating_preference(effective_text)
    if seating:
        slots["seating_preference"] = seating
        raw_matches["seating_preference"] = seating

    # 7. Customer name (fall back to effective_text if raw_text was in Urdu script)
    cname = extract_customer_name(raw_text or effective_text, expected_slot=expected_slot) or extract_customer_name(
        effective_text, expected_slot=expected_slot
    )
    if cname:
        slots["customer_name"] = cname
        raw_matches["customer_name"] = cname

    # 8. Special requests
    spec = extract_special_requests(raw_text or effective_text)
    if spec:
        slots["special_requests"] = spec
        raw_matches["special_requests"] = spec

    return SlotExtractionResult(
        slots=slots,
        ambiguous_number=ambig_num,
        ambiguity_reason=ambig_reason,
        validation_errors=validation_errors,
        raw_matches=raw_matches,
    )
