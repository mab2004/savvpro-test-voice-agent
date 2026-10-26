"""
Comprehensive Automated Test Suite (`pytest`) for DawatDesk Voice Concierge.

Covers 28 real-world messy phone caller scenarios:
- Homophone & STT repair ("for people" -> "4 people", "ate pm" -> "8 pm")
- Pakistani Minglish / Roman Urdu code-switching ("kal raat 8 baje 4 logon ka table...")
- Intra-utterance self-corrections ("3 people—sorry, 5 people—tomorrow at 8 PM")
- Mid-call slot overwrites (including at the final CONFIRMING_BOOKING step with re-validation)
- FAQ interruptions & digression stack restoration without losing slots
- Bare-number disambiguation ("Book for 8" -> 8 guests vs 8:00 PM)
- SQLite peak-hour conflict resolution (offering nearest times 7:30 PM / 8:30 PM & Rooftop)
- Past dates, out-of-hours times (3:00 AM), >20 capacity banquet policy, and gibberish gating
- Spoken Pakistani phone numbers ("zero three two one triple five...")
- Modifying & cancelling existing reservations + freeing table inventory
- FastAPI REST & TTS audio endpoint verification
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Tuple

import pytest
from fastapi.testclient import TestClient

from app.brain.dialogue_manager import DialogueManager
from app.brain.intent_classifier import get_intent_classifier
from app.brain.inventory import InventoryManager
from app.brain.normalizer import normalize_utterance
from app.brain.slot_extractor import extract_slots
from app.main import app, dialogue_manager
from app.models import DialogueState, IntentName, SeatingArea


FIXED_REF_DATE = "2026-10-26"
FIXED_TOMORROW = "2026-10-27"
FIXED_DAY_AFTER = "2026-10-28"


@pytest.fixture()
def brain(tmp_path: Path) -> DialogueManager:
    """Provides a fresh DialogueManager backed by an isolated temporary SQLite DB."""
    db_file = tmp_path / "test_dawatdesk.db"
    inv = InventoryManager(db_path=str(db_file))
    inv.reset_database(reference_date=FIXED_REF_DATE)
    return DialogueManager(inventory=inv)


# ===========================================================================
# 1. Stage 1 Normalizer: Homophones, Minglish, and Intra-Sentence Repairs
# ===========================================================================

def test_01_homophone_repair_for_people_and_ate_pm() -> None:
    res = normalize_utterance("Book a table for people tomorrow at ate pm")
    assert "4 people" in res.effective_text.lower()
    assert "8 pm" in res.effective_text.lower()
    assert len(res.homophone_repairs) >= 2

    res2 = normalize_utterance("We need a table for for at to pm")
    assert "table for 4" in res2.effective_text.lower()
    assert "2 pm" in res2.effective_text.lower()


def test_02_minglish_roman_urdu_normalization() -> None:
    res = normalize_utterance(
        "Salam, kal raat 8 baje 4 logon ka table book karna hai indoor mein"
    )
    eff = res.effective_text.lower()
    assert "tomorrow" in eff
    assert "8 pm" in eff
    assert "4 people" in eff
    assert "indoor" in eff
    assert len(res.minglish_mappings) >= 3

    res_saarhe = normalize_utterance("parso saarhe saat baje char log chhat pe")
    eff2 = res_saarhe.effective_text.lower()
    assert "day after tomorrow" in eff2
    assert "7:30" in eff2
    assert "4 people" in eff2
    assert "rooftop" in eff2


def test_03_intra_utterance_self_correction_party_size(brain: DialogueManager) -> None:
    turn = brain.process_turn(
        "Book a table for 3 people—sorry, 5 people—tomorrow at 8 PM",
        session_id="s-repair-1",
        reference_date=FIXED_REF_DATE,
    )
    assert turn.slots["party_size"] == 5
    assert turn.slots["date"] == FIXED_TOMORROW
    assert turn.slots["time"] == "20:00"
    assert turn.decision_trace.detected_repair is not None
    assert "sorry" in turn.decision_trace.detected_repair["marker"]


def test_04_intra_utterance_self_correction_time_clause(brain: DialogueManager) -> None:
    turn = brain.process_turn(
        "Book a table for 4 people tomorrow at 7 PM, no wait make it 8:30 PM",
        session_id="s-repair-2",
        reference_date=FIXED_REF_DATE,
    )
    assert turn.slots["party_size"] == 4
    assert turn.slots["date"] == FIXED_TOMORROW
    assert turn.slots["time"] == "20:30"
    assert turn.decision_trace.detected_repair is not None


def test_05_urdu_negated_time_self_correction(brain: DialogueManager) -> None:
    brain.process_turn(
        "Book a table for 4 people tomorrow at 8 PM indoor",
        session_id="s-urdu-neg",
        reference_date=FIXED_REF_DATE,
    )
    turn2 = brain.process_turn(
        "Nahi 8 baje nahi, 9 baje kar do",
        session_id="s-urdu-neg",
        reference_date=FIXED_REF_DATE,
    )
    assert turn2.slots["time"] == "21:00"
    assert any(
        m.slot == "time" and m.old_value == "20:00" and m.new_value == "21:00"
        for m in turn2.decision_trace.slot_mutations
    )


# ===========================================================================
# 2. One-Shot Multi-Slot & Minglish End-to-End Booking Flows
# ===========================================================================

def test_06_multi_slot_one_shot_booking_flow(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Hi I'm Ali, 0321-5551234, book an indoor table for 4 people tomorrow at 7:30 PM",
        session_id="s-oneshot",
        reference_date=FIXED_REF_DATE,
    )
    # Must extract all 6 required slots in a single turn and jump straight to CONFIRMING_BOOKING
    assert t1.state == DialogueState.CONFIRMING_BOOKING.value
    assert t1.missing_slots == []
    assert t1.slots["customer_name"] == "Ali"
    assert t1.slots["phone_number"] == "0321-5551234"
    assert t1.slots["party_size"] == 4
    assert t1.slots["date"] == FIXED_TOMORROW
    assert t1.slots["time"] == "19:30"
    assert t1.slots["seating_preference"] == SeatingArea.INDOOR_AC.value
    assert t1.decision_trace.inventory_check is not None
    assert t1.decision_trace.inventory_check.available is True

    # Confirm the booking
    t2 = brain.process_turn(
        "Yes please confirm it",
        session_id="s-oneshot",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.state == DialogueState.COMPLETED.value
    assert t2.booking is not None
    assert t2.booking["booking_ref"].startswith("DD-")
    assert t2.booking["customer_name"] == "Ali"


def test_07_pakistani_minglish_full_booking_and_confirmation(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Salam, kal raat 8 baje 4 logon ka table book karna hai indoor mein",
        session_id="s-minglish",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.slots["party_size"] == 4
    assert t1.slots["date"] == FIXED_TOMORROW
    assert t1.slots["time"] == "20:00"
    assert t1.slots["seating_preference"] == SeatingArea.INDOOR_AC.value
    assert t1.expected_slot == "customer_name"

    t2 = brain.process_turn(
        "Mera naam Usman Ali hai and phone is 0300 1234567",
        session_id="s-minglish",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.state == DialogueState.CONFIRMING_BOOKING.value
    assert t2.slots["customer_name"] == "Usman Ali"
    assert t2.slots["phone_number"] == "0300-1234567"

    t3 = brain.process_turn(
        "Haan jee bilkul confirm kar dein",
        session_id="s-minglish",
        reference_date=FIXED_REF_DATE,
    )
    assert t3.state == DialogueState.COMPLETED.value
    assert t3.booking is not None
    assert t3.booking["status"] == "CONFIRMED"


# ===========================================================================
# 3. Mid-Call Slot Overwrites & Confirmation Re-Validation
# ===========================================================================

def test_08_mid_call_slot_overwrite_during_collection(brain: DialogueManager) -> None:
    brain.process_turn(
        "Book a table for 4 people tomorrow at 7:00 PM",
        session_id="s-overwrite-1",
        reference_date=FIXED_REF_DATE,
    )
    t2 = brain.process_turn(
        "Actually make it 8:30 PM",
        session_id="s-overwrite-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.slots["time"] == "20:30"
    assert t2.slots["party_size"] == 4
    assert any(
        m.slot == "time" and m.old_value == "19:00" and m.new_value == "20:30" and m.mutation_type == "mid_call_overwrite"
        for m in t2.decision_trace.slot_mutations
    )


def test_09_mid_call_overwrite_at_confirming_booking_revalidates_success(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Hi I'm Bilal, 0333-9876543, book an indoor table for 4 people tomorrow at 7:00 PM",
        session_id="s-overwrite-confirm-ok",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.CONFIRMING_BOOKING.value

    # Overwrite time right at CONFIRMING_BOOKING
    t2 = brain.process_turn(
        "Actually make it 8:30 PM",
        session_id="s-overwrite-confirm-ok",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.state == DialogueState.CONFIRMING_BOOKING.value
    assert t2.slots["time"] == "20:30"
    assert t2.decision_trace.inventory_check is not None
    assert t2.decision_trace.inventory_check.available is True
    assert "8:30 PM" in t2.response_text


def test_10_mid_call_overwrite_at_confirming_booking_triggers_conflict(brain: DialogueManager) -> None:
    """
    Critical requirement: Changing party_size, date, or time during CONFIRMING_BOOKING
    must automatically re-validate table availability.
    At Tomorrow 8:00 PM Indoor AC, 4 guests is available (IND-T2), but 8 guests is FULL (IND-T5 booked).
    """
    t1 = brain.process_turn(
        "Hi I'm Zain, 0300-5554433, book an indoor table for 4 people tomorrow at 8:00 PM",
        session_id="s-overwrite-conflict",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.CONFIRMING_BOOKING.value
    assert t1.decision_trace.inventory_check.available is True

    # Right before confirming, caller increases party size to 8
    t2 = brain.process_turn(
        "Wait, my uncle's family is coming too, make it 8 people",
        session_id="s-overwrite-conflict",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.slots["party_size"] == 8
    assert t2.state == DialogueState.RESOLVING_CONFLICT.value
    assert t2.decision_trace.inventory_check is not None
    assert t2.decision_trace.inventory_check.available is False
    assert len(t2.decision_trace.inventory_check.alternatives) >= 2


# ===========================================================================
# 4. FAQ Interruptions & Digression Stack
# ===========================================================================

def test_11_faq_interruption_valet_parking_preserves_slots_and_reprompts(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "I want to book a table for 4 people tomorrow",
        session_id="s-faq-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.COLLECTING_SLOTS.value
    assert t1.expected_slot == "time"

    # Caller interrupts with an FAQ instead of giving the time
    t2 = brain.process_turn(
        "Wait, do you guys have valet parking?",
        session_id="s-faq-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.decision_trace.digression_triggered is True
    assert t2.decision_trace.faq_topic == IntentName.FAQ_PARKING.value
    assert "valet" in t2.response_text.lower()
    # Must preserve slots and re-prompt for the exact missing slot ('time')
    assert t2.slots["party_size"] == 4
    assert t2.slots["date"] == FIXED_TOMORROW
    assert t2.expected_slot == "time"
    assert "time" in t2.response_text.lower()


def test_12_faq_interruption_rooftop_rain_and_credit_cards_chained(brain: DialogueManager) -> None:
    brain.process_turn(
        "Book a rooftop table for 4 people tomorrow at 7:30 PM under Sarah",
        session_id="s-faq-2",
        reference_date=FIXED_REF_DATE,
    )
    # Currently waiting for phone_number
    t_rain = brain.process_turn(
        "Is the rooftop open if it rains?",
        session_id="s-faq-2",
        reference_date=FIXED_REF_DATE,
    )
    assert t_rain.decision_trace.digression_triggered is True
    assert t_rain.decision_trace.faq_topic == IntentName.FAQ_ROOFTOP_WEATHER.value
    assert "pergola" in t_rain.response_text.lower() or "rain" in t_rain.response_text.lower()
    assert t_rain.expected_slot == "phone_number"

    t_cards = brain.process_turn(
        "And do you take credit cards?",
        session_id="s-faq-2",
        reference_date=FIXED_REF_DATE,
    )
    assert t_cards.decision_trace.digression_triggered is True
    assert t_cards.decision_trace.faq_topic == IntentName.FAQ_PAYMENT_CARDS.value
    assert "visa" in t_cards.response_text.lower() or "mastercard" in t_cards.response_text.lower()
    assert t_cards.expected_slot == "phone_number"

    # Now provide phone number and verify we transition cleanly to CONFIRMING_BOOKING
    t_phone = brain.process_turn(
        "0345-1122334",
        session_id="s-faq-2",
        reference_date=FIXED_REF_DATE,
    )
    assert t_phone.state == DialogueState.CONFIRMING_BOOKING.value
    assert t_phone.slots["customer_name"] == "Sarah"
    assert t_phone.slots["phone_number"] == "0345-1122334"


# ===========================================================================
# 5. Bare-Number Disambiguation & Time Resolution
# ===========================================================================

def test_13_bare_number_ambiguity_clarification_as_guests(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Book for 8",
        session_id="s-ambig-guests",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.CLARIFYING_AMBIGUITY.value
    assert t1.decision_trace.ambiguity_detected == "bare_number_guest_or_time"
    assert "8 guests" in t1.response_text and "8:00 PM" in t1.response_text

    t2 = brain.process_turn(
        "8 guests",
        session_id="s-ambig-guests",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.state == DialogueState.COLLECTING_SLOTS.value
    assert t2.slots["party_size"] == 8
    assert t2.slots["time"] is None


def test_14_bare_number_ambiguity_clarification_as_time(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Book for 8",
        session_id="s-ambig-time",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.CLARIFYING_AMBIGUITY.value

    t2 = brain.process_turn(
        "8:00 PM",
        session_id="s-ambig-time",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.state == DialogueState.COLLECTING_SLOTS.value
    assert t2.slots["time"] == "20:00"
    assert t2.slots["party_size"] is None
    assert t2.expected_slot == "party_size"


def test_15_contextual_bare_number_when_expected_slot_known(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "I want to book a table",
        session_id="s-ctx-bare",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.expected_slot == "party_size"

    # Bare "6" when expected_slot == "party_size" must resolve to 6 guests without ambiguity
    t2 = brain.process_turn(
        "6",
        session_id="s-ctx-bare",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.slots["party_size"] == 6
    assert t2.expected_slot == "date"

    t3 = brain.process_turn(
        "tomorrow",
        session_id="s-ctx-bare",
        reference_date=FIXED_REF_DATE,
    )
    assert t3.slots["date"] == FIXED_TOMORROW
    assert t3.expected_slot == "time"

    # Bare "7" when expected_slot == "time" must resolve to 19:00 (7:00 PM)
    t4 = brain.process_turn(
        "7",
        session_id="s-ctx-bare",
        reference_date=FIXED_REF_DATE,
    )
    assert t4.slots["time"] == "19:00"


# ===========================================================================
# 6. Invalid / Nonsensical Inputs & Low-Confidence Gate
# ===========================================================================

def test_16_dinner_hour_inference_and_out_of_hours_rejection(brain: DialogueManager) -> None:
    # "at 7" -> assumes 7:00 PM (19:00)
    t_ok = brain.process_turn(
        "Book a table for 4 people tomorrow at 7",
        session_id="s-hours-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t_ok.slots["time"] == "19:00"

    # "3:00 AM" -> outside opening hours (12:00 PM - 11:30 PM)
    t_bad = brain.process_turn(
        "Book a table for 4 people tomorrow at 3:00 AM",
        session_id="s-hours-2",
        reference_date=FIXED_REF_DATE,
    )
    assert t_bad.slots["time"] is None
    assert t_bad.decision_trace.validation_errors.get("time") == "out_of_hours"
    assert "12:00 PM" in t_bad.response_text and "11:30 PM" in t_bad.response_text


def test_17_past_date_rejection(brain: DialogueManager) -> None:
    t = brain.process_turn(
        "Book a table for 4 people yesterday at 8 PM",
        session_id="s-past",
        reference_date=FIXED_REF_DATE,
    )
    assert t.slots["date"] is None
    assert t.decision_trace.validation_errors.get("date") == "past_date"
    assert "past date" in t.response_text.lower()


def test_18_exceeds_max_capacity_private_event_policy(brain: DialogueManager) -> None:
    t = brain.process_turn(
        "We want to book a table for 28 people tomorrow at 8 PM",
        session_id="s-maxcap",
        reference_date=FIXED_REF_DATE,
    )
    assert t.slots["party_size"] is None
    assert t.decision_trace.validation_errors.get("party_size") == "exceeds_max_capacity"
    assert "20 guests" in t.response_text
    assert "Events Desk" in t.response_text


def test_19_gibberish_and_out_of_domain_low_confidence_gate(brain: DialogueManager) -> None:
    t_gib = brain.process_turn(
        "blargle qwerty zxcvbnm florp",
        session_id="s-gibberish",
        reference_date=FIXED_REF_DATE,
    )
    assert t_gib.decision_trace.selected_intent == IntentName.UNCLEAR.value
    assert t_gib.decision_trace.intent_confidence < 0.50
    assert "didn't catch" in t_gib.response_text.lower() or "didn't quite catch" in t_gib.response_text.lower()

    t_ood = brain.process_turn(
        "can you repair my spacecraft warp hyperdrive",
        session_id="s-ood",
        reference_date=FIXED_REF_DATE,
    )
    assert t_ood.decision_trace.selected_intent == IntentName.UNCLEAR.value
    assert t_ood.decision_trace.intent_confidence < 0.50


# ===========================================================================
# 7. SQLite Inventory Conflict Resolution (Nearest Time & Alternative Zone)
# ===========================================================================

def test_20_fully_booked_conflict_resolution_by_nearest_time(brain: DialogueManager) -> None:
    """
    Tomorrow at 8:00 PM Indoor AC is seeded as FULL for 6+ guests (IND-T4 & IND-T5 booked).
    Brain must offer 7:30 PM or 8:30 PM Indoor AC (and Rooftop/Majlis at 8:00 PM),
    and accept 'Okay give me 8:30 PM then'.
    """
    t1 = brain.process_turn(
        "Book an indoor table for 6 people tomorrow at 8:00 PM",
        session_id="s-conflict-time",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.RESOLVING_CONFLICT.value
    assert t1.decision_trace.inventory_check is not None
    assert t1.decision_trace.inventory_check.available is False
    assert "7:30 PM" in t1.response_text
    assert "8:30 PM" in t1.response_text

    # Caller accepts 8:30 PM
    t2 = brain.process_turn(
        "Okay give me 8:30 PM then",
        session_id="s-conflict-time",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.slots["time"] == "20:30"
    assert t2.state == DialogueState.COLLECTING_SLOTS.value
    assert t2.decision_trace.inventory_check is not None
    assert t2.decision_trace.inventory_check.available is True


def test_21_fully_booked_conflict_resolution_by_alternative_seating(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Hi I'm Hamza, 0300-7778899, book an indoor table for 6 people tomorrow at 8:00 PM",
        session_id="s-conflict-seat",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.RESOLVING_CONFLICT.value
    assert "Rooftop" in t1.response_text

    # Caller switches to Rooftop
    t2 = brain.process_turn(
        "Rooftop works",
        session_id="s-conflict-seat",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.slots["seating_preference"] == SeatingArea.ROOFTOP.value
    assert t2.slots["time"] == "20:00"
    # Since name & phone were already provided in Turn 1, jumps straight to CONFIRMING_BOOKING!
    assert t2.state == DialogueState.CONFIRMING_BOOKING.value
    assert t2.decision_trace.inventory_check.allocated_table_id == "ROOF-T3"


def test_22_seating_zone_capacity_limit_triggers_zone_alternatives(brain: DialogueManager) -> None:
    # Indoor AC max table size is 8. Requesting 10 people in Indoor AC should suggest Rooftop (10) or Private Family Majlis (12).
    t1 = brain.process_turn(
        "Book an indoor table for 10 people tomorrow at 7:00 PM",
        session_id="s-zone-cap",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.RESOLVING_CONFLICT.value
    assert "up to 8 guests" in t1.response_text
    assert "Rooftop" in t1.response_text or "Private Family Majlis" in t1.response_text


# ===========================================================================
# 8. Spoken Phone Numbers, Modify, Cancel & Fuzzy Matching
# ===========================================================================

def test_23_spoken_pakistani_phone_with_double_and_triple_digits() -> None:
    ext = extract_slots(
        "My name is Ayesha and my number is zero three two one triple five one two three four",
        reference_date=FIXED_REF_DATE,
    )
    assert ext.slots["customer_name"] == "Ayesha"
    assert ext.slots["phone_number"] == "0321-5551234"


def test_24_modify_existing_seeded_booking_by_reference(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "I want to modify my reservation DD-2045",
        session_id="s-mod-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.MODIFYING_EXISTING.value
    assert "Salman Ahmad" in t1.response_text

    t2 = brain.process_turn(
        "Please change the time to 8:30 PM",
        session_id="s-mod-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.state == DialogueState.COMPLETED.value
    assert t2.booking is not None
    assert t2.booking["booking_ref"] == "DD-2045"
    assert t2.booking["time"] == "20:30"


def test_25_cancel_existing_seeded_booking_with_confirmation(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "I need to cancel my booking DD-1001",
        session_id="s-cancel-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.state == DialogueState.CANCELLING_EXISTING.value
    assert "Tariq Mehmood" in t1.response_text

    t2 = brain.process_turn(
        "Yes please confirm",
        session_id="s-cancel-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.state == DialogueState.COMPLETED.value
    assert t2.booking is not None
    assert t2.booking["status"] == "CANCELLED"


def test_26_cancelled_table_becomes_available_for_next_caller(brain: DialogueManager) -> None:
    # First cancel DD-1001 (which held IND-T4 for 6 guests Today at 20:00)
    brain.process_turn("Cancel booking DD-1001", session_id="s-c1", reference_date=FIXED_REF_DATE)
    brain.process_turn("Yes", session_id="s-c1", reference_date=FIXED_REF_DATE)

    # Now a new caller requests 6 guests Today at 8:00 PM Indoor AC -> should succeed on IND-T4!
    t_new = brain.process_turn(
        "Book an indoor table for 6 people today at 8:00 PM",
        session_id="s-c2",
        reference_date=FIXED_REF_DATE,
    )
    assert t_new.state == DialogueState.COLLECTING_SLOTS.value
    assert t_new.decision_trace.inventory_check is not None
    assert t_new.decision_trace.inventory_check.available is True
    assert t_new.decision_trace.inventory_check.allocated_table_id == "IND-T4"


def test_27_rapidfuzz_fuzzy_seating_matching() -> None:
    ext1 = extract_slots("we want rooftp seating please", reference_date=FIXED_REF_DATE)
    assert ext1.slots.get("seating_preference") == SeatingArea.ROOFTOP.value

    ext2 = extract_slots("book in privte family majlis", reference_date=FIXED_REF_DATE)
    assert ext2.slots.get("seating_preference") == SeatingArea.PRIVATE_MAJLIS.value


# ===========================================================================
# 9. FastAPI Endpoints & Offline TTS Integration Test
# ===========================================================================

def test_28_fastapi_rest_endpoints_turn_session_bookings_tts_reset() -> None:
    client = TestClient(app)

    # 1. Reset with fixed reference date
    r_reset = client.post("/api/reset", params={"reference_date": FIXED_REF_DATE})
    assert r_reset.status_code == 200
    assert len(r_reset.json()["bookings"]) == 5

    # 2. Health endpoint
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["llm_used"] is False

    # 3. Process turn with include_tts=True
    r_turn = client.post(
        "/api/turn",
        json={
            "session_id": "api-test-1",
            "text": "Hi I'm Omar, 0300-1239876, book a rooftop table for 2 people tomorrow at 7:30 PM",
            "reference_date": FIXED_REF_DATE,
            "include_tts": True,
        },
    )
    assert r_turn.status_code == 200
    body = r_turn.json()
    assert body["state"] == DialogueState.CONFIRMING_BOOKING.value
    assert body["audio_base64"] is not None and len(body["audio_base64"]) > 100

    # 4. Inspect session state
    r_sess = client.get("/api/session/api-test-1")
    assert r_sess.status_code == 200
    assert r_sess.json()["slots"]["customer_name"] == "Omar"

    # 5. Direct TTS WAV endpoint
    r_tts = client.post("/api/tts", json={"text": "Welcome to Haveli Grill and Courtyard"})
    assert r_tts.status_code == 200
    assert r_tts.headers["content-type"] == "audio/wav"
    assert r_tts.content[:4] == b"RIFF"


def test_29_urdu_fractional_times_derh_dhai_sawa_paune(brain: DialogueManager) -> None:
    t_derh = brain.process_turn(
        "Kal dopehar derh baje 4 logon ka table book karna hai",
        session_id="s-frac-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t_derh.slots["time"] == "13:30"
    assert t_derh.slots["party_size"] == 4

    t_dhai = brain.process_turn(
        "Kal dhai baje 2 logon ka table book karna hai",
        session_id="s-frac-2",
        reference_date=FIXED_REF_DATE,
    )
    assert t_dhai.slots["time"] == "14:30"
    assert t_dhai.slots["party_size"] == 2


def test_30_additive_adults_kids_and_relative_party_delta(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Book a table for 3 adults and 2 kids tomorrow at 7:00 PM indoor",
        session_id="s-delta-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.slots["party_size"] == 5

    t2 = brain.process_turn(
        "Actually add 2 more people to our table",
        session_id="s-delta-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.slots["party_size"] == 7


def test_31_native_urdu_script_ur_pk_speech_recognition(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "اسلام علیکم کل ساڑھے سات بجے چار لوگوں کا انڈور ٹیبل بک کرنا ہے",
        session_id="s-urdu-script-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.slots["party_size"] == 4
    assert t1.slots["date"] == "2026-10-27"
    assert t1.slots["time"] == "19:30"
    assert t1.slots["seating_preference"] == SeatingArea.INDOOR_AC.value

    t2 = brain.process_turn(
        "میرا نام بلال احمد ہے اور فون نمبر ۰۳۲۱۴۴۵۵۶۶۷",
        session_id="s-urdu-script-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t2.slots["customer_name"] == "Bilal Ahmed"
    assert t2.slots["phone_number"] == "0321-4455667"
    assert t2.state == DialogueState.CONFIRMING_BOOKING.value

    t3 = brain.process_turn(
        "ہاں جی بالکل کنفرم کر دیں",
        session_id="s-urdu-script-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t3.state == DialogueState.COMPLETED.value
    assert t3.booking is not None


def test_32_english_stt_phonetic_urdu_misrecognition_repair(brain: DialogueManager) -> None:
    t1 = brain.process_turn(
        "Salam kull sari saat budget char logon ka indoor table book karna hai",
        session_id="s-phonetic-urdu-1",
        reference_date=FIXED_REF_DATE,
    )
    assert t1.slots["party_size"] == 4
    assert t1.slots["date"] == "2026-10-27"
    assert t1.slots["time"] == "19:30"


