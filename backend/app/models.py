"""
Pydantic schemas, enums, restaurant constants, and SQLite models for
DawatDesk Voice Concierge (Haveli Grill & Courtyard, F-7 Markaz, Islamabad).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Restaurant Operational Constants
# ---------------------------------------------------------------------------

RESTAURANT_NAME = "Haveli Grill & Courtyard"
RESTAURANT_BRANCH = "College Road, F-7 Markaz, Islamabad"
RESTAURANT_PHONE = "051-265-4321"
OPENING_HOUR = 12  # 12:00 PM (Noon)
CLOSING_HOUR = 23  # 11:30 PM last seating at 23:00 / 23:30
CLOSING_MINUTE = 30
MAX_PARTY_SIZE = 20
DEFAULT_SEATING = "Indoor AC"

# Required slots to complete a table reservation
REQUIRED_BOOKING_SLOTS = [
    "party_size",
    "date",
    "time",
    "seating_preference",
    "customer_name",
    "phone_number",
]

# Prompting order when slots are missing
SLOT_PROMPT_ORDER = [
    "party_size",
    "date",
    "time",
    "seating_preference",
    "customer_name",
    "phone_number",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DialogueState(str, Enum):
    GREETING = "GREETING"
    COLLECTING_SLOTS = "COLLECTING_SLOTS"
    CLARIFYING_AMBIGUITY = "CLARIFYING_AMBIGUITY"
    RESOLVING_CONFLICT = "RESOLVING_CONFLICT"
    CONFIRMING_BOOKING = "CONFIRMING_BOOKING"
    MODIFYING_EXISTING = "MODIFYING_EXISTING"
    CANCELLING_EXISTING = "CANCELLING_EXISTING"
    COMPLETED = "COMPLETED"


class IntentName(str, Enum):
    GREET = "GREET"
    BOOK_TABLE = "BOOK_TABLE"
    CHECK_AVAILABILITY = "CHECK_AVAILABILITY"
    PROVIDE_SLOT = "PROVIDE_SLOT"
    CHANGE_SLOT = "CHANGE_SLOT"
    MODIFY_BOOKING = "MODIFY_BOOKING"
    CANCEL_BOOKING = "CANCEL_BOOKING"
    CONFIRM_YES = "CONFIRM_YES"
    CONFIRM_NO = "CONFIRM_NO"
    FAQ_PARKING = "FAQ_PARKING"
    FAQ_HOURS = "FAQ_HOURS"
    FAQ_LOCATION = "FAQ_LOCATION"
    FAQ_MENU_HALAL = "FAQ_MENU_HALAL"
    FAQ_ROOFTOP_WEATHER = "FAQ_ROOFTOP_WEATHER"
    FAQ_PAYMENT_CARDS = "FAQ_PAYMENT_CARDS"
    FAQ_PRIVATE_EVENTS = "FAQ_PRIVATE_EVENTS"
    GOODBYE = "GOODBYE"
    UNCLEAR = "UNCLEAR"


class SeatingArea(str, Enum):
    INDOOR_AC = "Indoor AC"
    ROOFTOP = "Rooftop"
    PRIVATE_MAJLIS = "Private Family Majlis"


FAQ_INTENTS = {
    IntentName.FAQ_PARKING.value,
    IntentName.FAQ_HOURS.value,
    IntentName.FAQ_LOCATION.value,
    IntentName.FAQ_MENU_HALAL.value,
    IntentName.FAQ_ROOFTOP_WEATHER.value,
    IntentName.FAQ_PAYMENT_CARDS.value,
    IntentName.FAQ_PRIVATE_EVENTS.value,
}


# ---------------------------------------------------------------------------
# Stage 1: Normalizer Schemas
# ---------------------------------------------------------------------------

class NormalizationResult(BaseModel):
    raw_text: str
    cleaned_text: str
    effective_text: str
    discarded_prefix: Optional[str] = None
    repair_detected: bool = False
    repair_marker: Optional[str] = None
    homophone_repairs: List[str] = Field(default_factory=list)
    minglish_mappings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 2: Intent Classifier Schemas
# ---------------------------------------------------------------------------

class IntentScore(BaseModel):
    intent: str
    probability: float


class IntentPrediction(BaseModel):
    top_intent: str
    confidence: float
    margin: float
    entropy: float
    is_unclear: bool = False
    source: str = "ml"
    top_3: List[IntentScore] = Field(default_factory=list)
    secondary_intents: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3: Slot Extractor Schemas
# ---------------------------------------------------------------------------

class SlotExtractionResult(BaseModel):
    slots: Dict[str, Any] = Field(default_factory=dict)
    ambiguous_number: Optional[int] = None
    ambiguity_reason: Optional[str] = None
    validation_errors: Dict[str, str] = Field(default_factory=dict)
    raw_matches: Dict[str, str] = Field(default_factory=dict)


class SlotMutation(BaseModel):
    slot: str
    old_value: Any = None
    new_value: Any = None
    mutation_type: str = "initial_fill"  # initial_fill | mid_call_overwrite | self_correction | conflict_resolution


# ---------------------------------------------------------------------------
# Stage 4 & 5: Inventory & Decision Trace Schemas
# ---------------------------------------------------------------------------

class AlternativeOption(BaseModel):
    date: str
    time: str
    display_time: str
    seating_preference: str
    table_id: str
    table_name: str
    capacity: int
    reason: str  # "nearest_time" | "alternative_seating"


class InventoryCheckResult(BaseModel):
    checked: bool = False
    available: bool = False
    allocated_table_id: Optional[str] = None
    allocated_table_name: Optional[str] = None
    allocated_seating: Optional[str] = None
    allocated_capacity: Optional[int] = None
    conflict_reason: Optional[str] = None
    alternatives: List[AlternativeOption] = Field(default_factory=list)


class DecisionTrace(BaseModel):
    turn_index: int
    timestamp: str
    processing_ms: float = 0.0
    raw_input: str
    normalized_input: str
    effective_input: str
    homophone_repairs: List[str] = Field(default_factory=list)
    minglish_mappings: List[str] = Field(default_factory=list)
    detected_repair: Optional[Dict[str, Any]] = None
    top_intents: List[IntentScore] = Field(default_factory=list)
    selected_intent: str
    intent_confidence: float
    intent_margin: float = 0.0
    intent_entropy: float = 0.0
    intent_source: str = "ml"
    extracted_slots: Dict[str, Any] = Field(default_factory=dict)
    slot_mutations: List[SlotMutation] = Field(default_factory=list)
    current_slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    expected_slot_before: Optional[str] = None
    expected_slot_after: Optional[str] = None
    state_before: str
    state_after: str
    digression_triggered: bool = False
    faq_topic: Optional[str] = None
    ambiguity_detected: Optional[str] = None
    validation_errors: Dict[str, str] = Field(default_factory=dict)
    inventory_check: Optional[InventoryCheckResult] = None
    booking_record: Optional[Dict[str, Any]] = None
    assistant_response: str


class SessionState(BaseModel):
    session_id: str
    state: DialogueState = DialogueState.GREETING
    return_state: Optional[DialogueState] = None
    expected_slot: Optional[str] = None
    slots: Dict[str, Any] = Field(
        default_factory=lambda: {
            "party_size": None,
            "date": None,
            "time": None,
            "seating_preference": None,
            "customer_name": None,
            "phone_number": None,
            "booking_reference": None,
            "special_requests": None,
        }
    )
    pending_ambiguous_number: Optional[int] = None
    pending_ambiguity_type: Optional[str] = None
    pending_alternatives: List[AlternativeOption] = Field(default_factory=list)
    allocated_table_id: Optional[str] = None
    allocated_table_name: Optional[str] = None
    confirmed_booking_ref: Optional[str] = None
    target_booking_ref: Optional[str] = None
    modification_subflow: Optional[str] = None  # "awaiting_lookup" | "awaiting_changes" | "awaiting_confirm"
    cancellation_subflow: Optional[str] = None  # "awaiting_lookup" | "awaiting_confirm"
    turn_count: int = 0
    reference_date: Optional[str] = None
    history: List[DecisionTrace] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API Request / Response Schemas
# ---------------------------------------------------------------------------

class TurnRequest(BaseModel):
    session_id: Optional[str] = None
    text: str
    reference_date: Optional[str] = None
    include_tts: bool = False


class TurnResponse(BaseModel):
    session_id: str
    response_text: str
    state: str
    slots: Dict[str, Any]
    missing_slots: List[str]
    expected_slot: Optional[str] = None
    booking: Optional[Dict[str, Any]] = None
    decision_trace: DecisionTrace
    audio_base64: Optional[str] = None


class TTSRequest(BaseModel):
    text: str


class BookingRecord(BaseModel):
    booking_ref: str
    customer_name: str
    phone_number: str
    party_size: int
    date: str
    time: str
    seating_preference: str
    table_id: str
    table_name: str
    special_requests: Optional[str] = None
    status: str = "CONFIRMED"
    created_at: str
