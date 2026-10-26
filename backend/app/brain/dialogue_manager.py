"""
Stage 4: Hierarchical Finite State Machine (HFSM) + Digression Stack +
Slot Mutation Tracker + Conflict Resolution Coordinator for DawatDesk Voice Concierge.

100% deterministic, rule- and state-driven decision engine (Zero LLMs).
Returns a complete `DecisionTrace` on every turn for full explainability.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.brain.intent_classifier import get_intent_classifier
from app.brain.inventory import InventoryManager, format_display_time
from app.brain.normalizer import normalize_utterance
from app.brain.slot_extractor import extract_slots, get_reference_date
from app.models import (
    DecisionTrace,
    DialogueState,
    FAQ_INTENTS,
    IntentName,
    InventoryCheckResult,
    REQUIRED_BOOKING_SLOTS,
    RESTAURANT_BRANCH,
    RESTAURANT_NAME,
    RESTAURANT_PHONE,
    SLOT_PROMPT_ORDER,
    SessionState,
    SlotMutation,
    TurnResponse,
)


# ---------------------------------------------------------------------------
# Curated Restaurant FAQ Knowledge Base (Haveli Grill & Courtyard, Islamabad)
# ---------------------------------------------------------------------------

FAQ_ANSWERS: Dict[str, str] = {
    IntentName.FAQ_PARKING.value: (
        "Yes, we offer complimentary valet parking right at our main entrance on College Road, "
        "F-7 Markaz, plus dedicated underground parking."
    ),
    IntentName.FAQ_HOURS.value: (
        "We are open seven days a week from 12:00 PM (noon) until 11:30 PM, with our last dinner "
        "seating at 11:00 PM."
    ),
    IntentName.FAQ_LOCATION.value: (
        f"We are located at {RESTAURANT_BRANCH}, right opposite the main park near Jinnah Super."
    ),
    IntentName.FAQ_MENU_HALAL.value: (
        "All our meat is 100% Zabiha Halal certified. We specialize in Peshawari Shinwari Karahi, "
        "charcoal BBQ platters, mutton ribs, dumpukht, and vegetarian handi options."
    ),
    IntentName.FAQ_ROOFTOP_WEATHER.value: (
        "Our Margalla-view Rooftop has a motorized weather-proof pergola, outdoor gas heaters for "
        "chilly evenings, and mist fans—and if heavy rain occurs, we prioritize moving rooftop "
        "reservations into our Indoor AC hall."
    ),
    IntentName.FAQ_PAYMENT_CARDS.value: (
        "Yes, we accept Cash, Visa, Mastercard, American Express, Raast QR, NayaPay, SadaPay, "
        "and offer a 15% discount on select HBL and Bank Alfalah platinum cards."
    ),
    IntentName.FAQ_PRIVATE_EVENTS.value: (
        "Our Private Family Majlis suites accommodate groups up to 20 guests with full privacy. "
        f"For banquets or corporate events larger than 20 guests, our Events Desk at {RESTAURANT_PHONE} "
        "arranges custom dawat menus."
    ),
}

SLOT_PROMPTS: Dict[str, str] = {
    "party_size": "How many guests will be dining with us?",
    "date": "Which date would you like to book for (for example, today or tomorrow)?",
    "time": "What time would you like the reservation for? We are open from 12:00 PM to 11:30 PM.",
    "seating_preference": "Would you prefer Indoor AC, Rooftop, or our Private Family Majlis seating?",
    "customer_name": "May I have your name for the reservation?",
    "phone_number": "And what is the best phone number to reach you at (e.g., 0300-1234567)?",
}


class DialogueManager:
    """
    Stateful non-LLM conversation engine managing multi-turn phone reservations,
    mid-call slot overwrites, self-corrections, FAQ digressions, ambiguity
    clarification, and SQLite inventory conflicts.
    """

    def __init__(self, inventory: Optional[InventoryManager] = None) -> None:
        self.inventory = inventory or InventoryManager()
        self.classifier = get_intent_classifier()
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        reference_date: Optional[str] = None,
    ) -> SessionState:
        sid = session_id or f"call-{uuid.uuid4().hex[:8]}"
        if sid not in self.sessions:
            self.sessions[sid] = SessionState(
                session_id=sid,
                reference_date=reference_date,
            )
        elif reference_date and not self.sessions[sid].reference_date:
            self.sessions[sid].reference_date = reference_date
        return self.sessions[sid]

    def reset_session(self, session_id: str) -> SessionState:
        self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def reset_all(self, reference_date: Optional[str] = None) -> None:
        self.sessions.clear()
        self.inventory.reset_database(reference_date=reference_date)

    @staticmethod
    def _compute_missing_slots(slots: Dict[str, Any]) -> List[str]:
        return [s for s in SLOT_PROMPT_ORDER if slots.get(s) in (None, "")]

    def _build_reprompt_for_state(self, session: SessionState) -> str:
        """
        Generates the exact contextual continuation prompt when popping back
        from an FAQ digression or recovering from an unclear turn.
        """
        if session.state == DialogueState.CLARIFYING_AMBIGUITY and session.pending_ambiguous_number:
            n = session.pending_ambiguous_number
            return f"Now, back to your booking — when you said {n}, did you mean {n} guests or {n}:00 PM?"

        if session.state == DialogueState.RESOLVING_CONFLICT and session.pending_alternatives:
            alt_Desc = self._format_alternatives_prompt(session.pending_alternatives)
            return f"Now, back to your table options — {alt_Desc}"

        if session.state == DialogueState.CONFIRMING_BOOKING:
            return f"Now, back to your reservation — {self._build_confirmation_summary(session)}"

        if session.state == DialogueState.MODIFYING_EXISTING:
            if not session.target_booking_ref:
                return "To modify your reservation, could you please share your booking reference (like DD-1001) or phone number?"
            return f"What would you like to change on booking {session.target_booking_ref} (date, time, party size, or seating)?"

        if session.state == DialogueState.CANCELLING_EXISTING:
            if not session.target_booking_ref:
                return "To cancel your reservation, could you please share your booking reference (like DD-1001) or phone number?"
            return f"Please confirm: would you like me to cancel booking {session.target_booking_ref}?"

        missing = self._compute_missing_slots(session.slots)
        if missing:
            next_slot = missing[0]
            session.expected_slot = next_slot
            return f"Now, back to your reservation — {SLOT_PROMPTS[next_slot]}"

        return "How else can I assist you with your reservation at Haveli Grill & Courtyard?"

    def _build_confirmation_summary(self, session: SessionState) -> str:
        s = session.slots
        disp_time = format_display_time(s["time"]) if s.get("time") else "your requested time"
        spec = f" (Special notes: {s['special_requests']})" if s.get("special_requests") else ""
        return (
            f"I have a table ready in {s['seating_preference']} for {s['party_size']} guests "
            f"on {s['date']} at {disp_time} under {s['customer_name']} ({s['phone_number']}){spec}. "
            f"Shall I confirm and lock in this reservation?"
        )

    @staticmethod
    def _format_alternatives_prompt(alternatives: List[Any]) -> str:
        if not alternatives:
            return "What other time or date would you like me to check?"
        time_alts = [a for a in alternatives if a.reason == "nearest_time"]
        seat_alts = [a for a in alternatives if a.reason == "alternative_seating"]
        parts: List[str] = []
        if time_alts:
            times_str = " or ".join(a.display_time for a in time_alts)
            parts.append(f"{times_str} in {time_alts[0].seating_preference}")
        if seat_alts:
            zones_str = " or ".join(a.seating_preference for a in seat_alts)
            parts.append(f"{zones_str} at {seat_alts[0].display_time}")
        joined = ", or ".join(parts)
        return f"We can offer you {joined}. Which of these options works best for you?"

    def _resolve_ambiguity_turn(
        self,
        session: SessionState,
        norm_text: str,
        raw_text: str,
        extraction: Any,
        mutations: List[SlotMutation],
    ) -> bool:
        """
        Resolves a pending bare-number ambiguity (e.g. '8' -> 8 guests vs 8:00 PM).
        Returns True if ambiguity was resolved on this turn.
        """
        n = session.pending_ambiguous_number
        if n is None:
            return False

        low = f"{raw_text} {norm_text}".lower()

        # If slot extractor already found explicit party_size or time on this turn
        resolved = False
        if "party_size" in extraction.slots:
            resolved = True
        if "time" in extraction.slots:
            resolved = True

        if not resolved:
            if any(w in low for w in ["guest", "guests", "people", "persons", "log", "logon", "banday", "party", "members"]):
                extraction.slots["party_size"] = n
                resolved = True
            elif any(w in low for w in ["pm", "am", "o'clock", "oclock", "baje", "time", "evening", "night", ":00"]):
                t_24 = f"{n + 12:02d}:00" if 1 <= n <= 11 else "12:00"
                extraction.slots["time"] = t_24
                resolved = True

        if resolved:
            session.pending_ambiguous_number = None
            session.pending_ambiguity_type = None
            session.state = session.return_state or DialogueState.COLLECTING_SLOTS
            session.return_state = None
            return True

        return False

    def process_turn(
        self,
        text: str,
        session_id: Optional[str] = None,
        reference_date: Optional[str] = None,
    ) -> TurnResponse:
        """
        Executes the 5-stage non-LLM Brain pipeline for a single caller turn:
        Stage 1: Normalization, homophone repair, Minglish translation & self-correction parsing
        Stage 2: Contextual slot extraction & bare-number ambiguity detection
        Stage 3: Hybrid intent classification with confidence calibration & OOV gating
        Stage 4: HFSM state transition, digression stack, and SQLite inventory check
        Stage 5: Response synthesis & full DecisionTrace packaging
        """
        session = self.get_or_create_session(session_id=session_id, reference_date=reference_date)
        t_start = dt.datetime.now(dt.timezone.utc)
        session.turn_count += 1
        ref_date_str = reference_date or session.reference_date or get_reference_date().isoformat()
        session.reference_date = ref_date_str

        state_before = session.state
        expected_before = session.expected_slot

        # -------------------------------------------------------------------
        # STAGE 1: Normalization & Self-Correction Resolution
        # -------------------------------------------------------------------
        norm_res = normalize_utterance(text)
        detected_repair: Optional[Dict[str, Any]] = None
        if norm_res.repair_detected:
            detected_repair = {
                "marker": norm_res.repair_marker,
                "discarded_prefix": norm_res.discarded_prefix,
                "effective_text": norm_res.effective_text,
            }

        # -------------------------------------------------------------------
        # STAGE 2: Context-Aware Slot Extraction
        # -------------------------------------------------------------------
        extraction = extract_slots(
            effective_text=norm_res.effective_text,
            raw_text=norm_res.raw_text,
            expected_slot=session.expected_slot,
            current_slots=session.slots,
            reference_date=ref_date_str,
        )

        # If we were in CLARIFYING_AMBIGUITY, check if the caller clarified guests vs time
        mutations: List[SlotMutation] = []
        if session.state == DialogueState.CLARIFYING_AMBIGUITY:
            self._resolve_ambiguity_turn(
                session=session,
                norm_text=norm_res.effective_text,
                raw_text=norm_res.raw_text,
                extraction=extraction,
                mutations=mutations,
            )

        # -------------------------------------------------------------------
        # STAGE 3: Hybrid Intent Classification
        # -------------------------------------------------------------------
        intent_pred = self.classifier.predict(
            text=norm_res.effective_text,
            raw_text=norm_res.raw_text,
            dialogue_state=session.state,
            expected_slot=session.expected_slot,
            has_extracted_slots=bool(extraction.slots),
            repair_detected=norm_res.repair_detected,
        )

        selected_intent = intent_pred.top_intent

        # Check if any FAQ intent was triggered (either as primary or secondary intent)
        faq_intent: Optional[str] = None
        if selected_intent in FAQ_INTENTS:
            faq_intent = selected_intent
        else:
            for sec in intent_pred.secondary_intents:
                if sec in FAQ_INTENTS:
                    faq_intent = sec
                    break

        # -------------------------------------------------------------------
        # STAGE 4: Apply Extracted Slots & Track Mutations / Overwrites
        # -------------------------------------------------------------------
        # Guard: if this turn is an FAQ question about rooftop weather ("Is the rooftop open if it rains?"),
        # do not treat the word "rooftop" inside the FAQ question as a seating preference mutation.
        if faq_intent == IntentName.FAQ_ROOFTOP_WEATHER.value and selected_intent == IntentName.FAQ_ROOFTOP_WEATHER.value:
            extraction.slots.pop("seating_preference", None)

        core_availability_slot_changed = False
        if selected_intent not in (IntentName.MODIFY_BOOKING.value, IntentName.CANCEL_BOOKING.value) and session.state not in (
            DialogueState.MODIFYING_EXISTING,
            DialogueState.CANCELLING_EXISTING,
        ):
            for slot_key, new_val in extraction.slots.items():
                if new_val is None:
                    continue
                old_val = session.slots.get(slot_key)
                if old_val != new_val:
                    if old_val is None:
                        m_type = "self_correction" if norm_res.repair_detected else (
                            "conflict_resolution" if state_before == DialogueState.RESOLVING_CONFLICT else "initial_fill"
                        )
                    else:
                        m_type = "conflict_resolution" if state_before == DialogueState.RESOLVING_CONFLICT else "mid_call_overwrite"
                    mutations.append(
                        SlotMutation(
                            slot=slot_key,
                            old_value=old_val,
                            new_value=new_val,
                            mutation_type=m_type,
                        )
                    )
                    session.slots[slot_key] = new_val
                    if slot_key in ("party_size", "date", "time", "seating_preference"):
                        core_availability_slot_changed = True

        # -------------------------------------------------------------------
        # STAGE 5: Hierarchical State Machine & Inventory Resolution
        # -------------------------------------------------------------------
        digression_triggered = False
        inventory_check: Optional[InventoryCheckResult] = None
        booking_record_dict: Optional[Dict[str, Any]] = None
        response_text = ""

        # A. Handle FAQ Interruption / Digression Stack
        if faq_intent is not None:
            digression_triggered = True
            faq_answer = FAQ_ANSWERS.get(faq_intent, FAQ_ANSWERS[IntentName.FAQ_HOURS.value])

            # If the user was in GREETING and didn't provide any booking slots, stay in GREETING
            has_any_booking_slot = any(
                session.slots.get(k) is not None for k in REQUIRED_BOOKING_SLOTS
            )
            if session.state == DialogueState.GREETING and has_any_booking_slot:
                session.state = DialogueState.COLLECTING_SLOTS

            if session.state == DialogueState.GREETING and not has_any_booking_slot:
                response_text = (
                    f"{faq_answer} Would you like me to reserve a table for you at {RESTAURANT_NAME}?"
                )
            else:
                reprompt = self._build_reprompt_for_state(session)
                response_text = f"{faq_answer} {reprompt}"

        # B. Handle Validation Errors (past date, out-of-hours time, > 20 capacity)
        elif extraction.validation_errors:
            if session.state == DialogueState.GREETING:
                session.state = DialogueState.COLLECTING_SLOTS

            err_messages: List[str] = []
            if extraction.validation_errors.get("date") == "past_date":
                raw_d = extraction.raw_matches.get("date", "that date")
                err_messages.append(
                    f"I cannot book a table for a past date ({raw_d}). Please choose today ({ref_date_str}) or an upcoming date."
                )
                session.expected_slot = "date"

            if extraction.validation_errors.get("time") == "out_of_hours":
                raw_t = extraction.raw_matches.get("time", "that time")
                err_messages.append(
                    f"Our dining hours at {RESTAURANT_NAME} are 12:00 PM (noon) to 11:30 PM, so {raw_t} is outside our operating hours. Please pick a time between 12:00 PM and 11:30 PM."
                )
                session.expected_slot = "time"

            if extraction.validation_errors.get("party_size") == "exceeds_max_capacity":
                raw_p = extraction.raw_matches.get("party_size", "more than 20")
                err_messages.append(
                    f"For groups larger than 20 guests (you requested {raw_p} guests), our Events Desk arranges custom Private Banquet & Dawat packages at {RESTAURANT_PHONE}. Over the phone right now, I can book tables or our Private Family Majlis for up to 20 guests. How many guests (up to 20) would you like me to book for?"
                )
                session.expected_slot = "party_size"

            if extraction.validation_errors.get("party_size") == "invalid_party_size":
                err_messages.append("Please let me know a valid number of guests (between 1 and 20).")
                session.expected_slot = "party_size"

            response_text = " ".join(err_messages)

        # C. Handle Bare-Number Ambiguity ("Book for 8" -> 8 guests or 8:00 PM?)
        elif extraction.ambiguous_number is not None:
            num = extraction.ambiguous_number
            session.pending_ambiguous_number = num
            session.pending_ambiguity_type = extraction.ambiguity_reason
            session.return_state = (
                DialogueState.COLLECTING_SLOTS
                if session.state == DialogueState.GREETING
                else session.state
            )
            session.state = DialogueState.CLARIFYING_AMBIGUITY
            session.expected_slot = None
            response_text = (
                f"I'd love to help with that! When you said {num}, did you mean a party of "
                f"{num} guests, or a reservation at {num}:00 PM?"
            )

        # D. Handle CANCEL_BOOKING Flow
        elif selected_intent == IntentName.CANCEL_BOOKING.value or session.state == DialogueState.CANCELLING_EXISTING:
            response_text, booking_record_dict = self._handle_cancellation_flow(
                session=session,
                selected_intent=selected_intent,
                extraction=extraction,
            )

        # E. Handle MODIFY_BOOKING Flow
        elif selected_intent == IntentName.MODIFY_BOOKING.value or session.state == DialogueState.MODIFYING_EXISTING:
            response_text, inventory_check, booking_record_dict = self._handle_modification_flow(
                session=session,
                selected_intent=selected_intent,
                extraction=extraction,
            )

        # F. Handle RESOLVING_CONFLICT State
        elif session.state == DialogueState.RESOLVING_CONFLICT:
            response_text, inventory_check = self._handle_conflict_resolution_turn(
                session=session,
                selected_intent=selected_intent,
                extraction=extraction,
                norm_text=norm_res.effective_text,
            )

        # G. Handle CONFIRMING_BOOKING State
        elif session.state == DialogueState.CONFIRMING_BOOKING:
            # Critical Edge Case #1: Mid-call slot overwrite right at the confirmation step!
            if core_availability_slot_changed or selected_intent == IntentName.CHANGE_SLOT.value or bool(extraction.slots):
                response_text, inventory_check = self._validate_and_advance_booking(session)
            elif selected_intent == IntentName.CONFIRM_YES.value:
                record, inv_check = self.inventory.create_booking(
                    customer_name=session.slots["customer_name"],
                    phone_number=session.slots["phone_number"],
                    party_size=int(session.slots["party_size"]),
                    date_str=session.slots["date"],
                    time_str=session.slots["time"],
                    seating_preference=session.slots["seating_preference"],
                    special_requests=session.slots.get("special_requests"),
                )
                inventory_check = inv_check
                if record:
                    session.confirmed_booking_ref = record.booking_ref
                    session.allocated_table_id = record.table_id
                    session.allocated_table_name = record.table_name
                    session.state = DialogueState.COMPLETED
                    session.expected_slot = None
                    booking_record_dict = record.model_dump()
                    disp_time = format_display_time(record.time)
                    response_text = (
                        f"Mubarak! Your reservation at {RESTAURANT_NAME} is confirmed under reference "
                        f"{record.booking_ref}. We have reserved {record.table_name} ({record.seating_preference}) "
                        f"for {record.party_size} guests on {record.date} at {disp_time} for "
                        f"{record.customer_name}. We look forward to hosting you!"
                    )
                else:
                    # Table was just taken; transition to conflict resolution
                    session.state = DialogueState.RESOLVING_CONFLICT
                    session.pending_alternatives = inv_check.alternatives
                    alt_str = self._format_alternatives_prompt(inv_check.alternatives)
                    response_text = f"I'm sorry, {inv_check.conflict_reason} {alt_str}"
            elif selected_intent == IntentName.CONFIRM_NO.value:
                session.state = DialogueState.COLLECTING_SLOTS
                session.expected_slot = None
                response_text = (
                    "No problem at all — I haven't locked it in yet. Would you like to change the time, "
                    "date, party size, or seating area, or should we cancel this request?"
                )
            else:
                response_text = (
                    f"Please let me know if I should confirm this booking (say 'Yes, confirm'), "
                    f"or tell me any detail you'd like to change. {self._build_confirmation_summary(session)}"
                )

        # H. Handle Pure Greeting
        elif selected_intent == IntentName.GREET.value and not extraction.slots and session.state == DialogueState.GREETING:
            response_text = (
                f"Assalam-o-Alaikum and welcome to {RESTAURANT_NAME}, F-7 Markaz, Islamabad! "
                "I can help you book a table, check availability, or modify an existing reservation. "
                "How many guests would you like to book for today?"
            )
            session.expected_slot = "party_size"

        # I. Handle Goodbye
        elif selected_intent == IntentName.GOODBYE.value:
            session.state = DialogueState.COMPLETED
            session.expected_slot = None
            response_text = (
                f"Thank you for calling {RESTAURANT_NAME}! Khuda Hafiz and have a wonderful day."
            )

        # J. Handle Unclear / Low-Confidence / Gibberish Input
        elif intent_pred.is_unclear and not extraction.slots:
            if session.state == DialogueState.CLARIFYING_AMBIGUITY and session.pending_ambiguous_number:
                n = session.pending_ambiguous_number
                response_text = (
                    f"Sorry, I didn't quite catch that. Did you mean {n} guests (people) or {n}:00 PM?"
                )
            elif session.expected_slot and session.expected_slot in SLOT_PROMPTS:
                response_text = (
                    f"I'm sorry, I didn't quite catch that clearly. {SLOT_PROMPTS[session.expected_slot]}"
                )
            else:
                response_text = (
                    f"I'm sorry, I didn't catch that. I can help you book a table at {RESTAURANT_NAME}, "
                    "check table availability, or update an existing booking. How can I help you today?"
                )

        # K. Standard Booking & Availability Progression (COLLECTING_SLOTS)
        else:
            if session.state in (DialogueState.GREETING, DialogueState.COMPLETED):
                session.state = DialogueState.COLLECTING_SLOTS

            response_text, inventory_check = self._validate_and_advance_booking(
                session=session,
                is_availability_query=(selected_intent == IntentName.CHECK_AVAILABILITY.value),
            )

        # -------------------------------------------------------------------
        # Package Rich DecisionTrace for Frontend & Automated Tests
        # -------------------------------------------------------------------
        missing_after = self._compute_missing_slots(session.slots)
        elapsed_ms = round((dt.datetime.now(dt.timezone.utc) - t_start).total_seconds() * 1000.0, 2)
        trace = DecisionTrace(
            turn_index=session.turn_count,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            processing_ms=elapsed_ms,
            raw_input=norm_res.raw_text,
            normalized_input=norm_res.cleaned_text,
            effective_input=norm_res.effective_text,
            homophone_repairs=norm_res.homophone_repairs,
            minglish_mappings=norm_res.minglish_mappings,
            detected_repair=detected_repair,
            top_intents=intent_pred.top_3,
            selected_intent=selected_intent,
            intent_confidence=intent_pred.confidence,
            intent_margin=intent_pred.margin,
            intent_entropy=intent_pred.entropy,
            intent_source=intent_pred.source,
            extracted_slots=extraction.slots,
            slot_mutations=mutations,
            current_slots=dict(session.slots),
            missing_slots=missing_after,
            expected_slot_before=expected_before,
            expected_slot_after=session.expected_slot,
            state_before=state_before.value,
            state_after=session.state.value,
            digression_triggered=digression_triggered,
            faq_topic=faq_intent,
            ambiguity_detected=extraction.ambiguity_reason,
            validation_errors=extraction.validation_errors,
            inventory_check=inventory_check,
            booking_record=booking_record_dict,
            assistant_response=response_text,
        )
        session.history.append(trace)

        return TurnResponse(
            session_id=session.session_id,
            response_text=response_text,
            state=session.state.value,
            slots=dict(session.slots),
            missing_slots=missing_after,
            expected_slot=session.expected_slot,
            booking=booking_record_dict,
            decision_trace=trace,
        )

    # -----------------------------------------------------------------------
    # Internal Subflows: Availability Check, Conflict Resolution, Modify, Cancel
    # -----------------------------------------------------------------------

    def _validate_and_advance_booking(
        self,
        session: SessionState,
        is_availability_query: bool = False,
    ) -> Tuple[str, Optional[InventoryCheckResult]]:
        """
        Checks if enough slots are present to run an early SQLite inventory check
        (party_size + date + time + seating_preference, or party_size + date + time for
        availability check), handles conflicts immediately, or prompts for the next
        missing slot in SLOT_PROMPT_ORDER.
        """
        s = session.slots
        has_core_three = all(s.get(k) is not None for k in ("party_size", "date", "time"))
        has_seating = s.get("seating_preference") is not None

        inv_check: Optional[InventoryCheckResult] = None

        # Run SQLite availability check as soon as (party_size, date, time, seating_preference)
        # are known (or on an explicit CHECK_AVAILABILITY query with party_size + date + time)
        if has_core_three and (has_seating or is_availability_query):
            inv_check = self.inventory.check_availability(
                date_str=s["date"],
                time_str=s["time"],
                party_size=int(s["party_size"]),
                seating_preference=s.get("seating_preference"),
            )
            if not inv_check.available:
                session.state = DialogueState.RESOLVING_CONFLICT
                session.pending_alternatives = inv_check.alternatives
                session.expected_slot = None
                alt_prompt = self._format_alternatives_prompt(inv_check.alternatives)
                return f"I just checked our tables — {inv_check.conflict_reason} {alt_prompt}", inv_check

            session.allocated_table_id = inv_check.allocated_table_id
            session.allocated_table_name = inv_check.allocated_table_name
            if not has_seating and inv_check.allocated_seating:
                s["seating_preference"] = inv_check.allocated_seating

        missing = self._compute_missing_slots(s)
        if not missing:
            # All 6 slots are present and inventory is verified available!
            if inv_check is None:
                inv_check = self.inventory.check_availability(
                    date_str=s["date"],
                    time_str=s["time"],
                    party_size=int(s["party_size"]),
                    seating_preference=s["seating_preference"],
                )
                if not inv_check.available:
                    session.state = DialogueState.RESOLVING_CONFLICT
                    session.pending_alternatives = inv_check.alternatives
                    session.expected_slot = None
                    alt_prompt = self._format_alternatives_prompt(inv_check.alternatives)
                    return f"I just checked our tables — {inv_check.conflict_reason} {alt_prompt}", inv_check

            session.state = DialogueState.CONFIRMING_BOOKING
            session.expected_slot = None
            return self._build_confirmation_summary(session), inv_check

        # Otherwise prompt for the next missing slot
        session.state = DialogueState.COLLECTING_SLOTS
        next_slot = missing[0]
        session.expected_slot = next_slot

        # Acknowledge what we have collected so far naturally
        ack_parts: List[str] = []
        if s.get("party_size"):
            ack_parts.append(f"{s['party_size']} guests")
        if s.get("date"):
            ack_parts.append(f"on {s['date']}")
        if s.get("time"):
            ack_parts.append(f"at {format_display_time(s['time'])}")
        if s.get("seating_preference"):
            ack_parts.append(f"in {s['seating_preference']}")

        if inv_check and inv_check.available and is_availability_query:
            prefix = (
                f"Good news! We have {inv_check.allocated_table_name} available for "
                f"{' '.join(ack_parts)}. To lock in this reservation, {SLOT_PROMPTS[next_slot][0].lower() + SLOT_PROMPTS[next_slot][1:]}"
            )
            return prefix, inv_check

        if ack_parts:
            return f"Got it — {' '.join(ack_parts)}. {SLOT_PROMPTS[next_slot]}", inv_check

        return SLOT_PROMPTS[next_slot], inv_check

    def _handle_conflict_resolution_turn(
        self,
        session: SessionState,
        selected_intent: str,
        extraction: Any,
        norm_text: str,
    ) -> Tuple[str, Optional[InventoryCheckResult]]:
        """
        Handles caller responses when their original slot was full and alternatives
        were offered (e.g., 'Okay give me 8:30 PM then' or 'Rooftop works').
        """
        if selected_intent == IntentName.CONFIRM_NO.value and not extraction.slots:
            session.state = DialogueState.COLLECTING_SLOTS
            session.pending_alternatives = []
            session.expected_slot = "time"
            return (
                "No problem. What other date, time, or seating preference would you like me to check?",
                None,
            )

        # If the user said "first option" / "second option" or accepted a single alternative
        low = norm_text.lower()
        if not extraction.slots and session.pending_alternatives:
            if "first" in low or "pehle" in low or selected_intent == IntentName.CONFIRM_YES.value:
                chosen = session.pending_alternatives[0]
                session.slots["time"] = chosen.time
                session.slots["seating_preference"] = chosen.seating_preference
            elif "second" in low or "doosra" in low and len(session.pending_alternatives) > 1:
                chosen = session.pending_alternatives[1]
                session.slots["time"] = chosen.time
                session.slots["seating_preference"] = chosen.seating_preference

        # Re-validate with updated slots
        session.pending_alternatives = []
        return self._validate_and_advance_booking(session)

    def _handle_modification_flow(
        self,
        session: SessionState,
        selected_intent: str,
        extraction: Any,
    ) -> Tuple[str, Optional[InventoryCheckResult], Optional[Dict[str, Any]]]:
        """Handles looking up and modifying an existing confirmed booking in SQLite."""
        session.state = DialogueState.MODIFYING_EXISTING

        # 1. Try to locate target booking if not already bound
        if not session.target_booking_ref:
            b_ref = extraction.slots.get("booking_reference") or session.slots.get("booking_reference")
            phone = extraction.slots.get("phone_number") or session.slots.get("phone_number")
            cname = extraction.slots.get("customer_name") or session.slots.get("customer_name")

            found = self.inventory.find_booking(
                booking_ref=b_ref,
                phone_number=phone,
                customer_name=cname,
            )
            if not found:
                session.expected_slot = "booking_reference"
                return (
                    "I'd be happy to modify your reservation. Could you please share your booking "
                    "reference code (such as DD-1001) or the phone number used for the booking?",
                    None,
                    None,
                )
            session.target_booking_ref = found.booking_ref
            session.slots["booking_reference"] = found.booking_ref

        existing = self.inventory.find_booking(booking_ref=session.target_booking_ref)
        if not existing:
            session.target_booking_ref = None
            session.expected_slot = "booking_reference"
            return (
                "I couldn't find an active reservation with that reference. Could you double-check "
                "your booking reference code or phone number?",
                None,
                None,
            )

        # 2. Check if the caller provided any new slot values to update
        new_party = extraction.slots.get("party_size")
        new_date = extraction.slots.get("date")
        new_time = extraction.slots.get("time")
        new_seating = extraction.slots.get("seating_preference")
        new_special = extraction.slots.get("special_requests")

        if any(x is not None for x in (new_party, new_date, new_time, new_seating, new_special)):
            updated, inv_check = self.inventory.update_booking(
                booking_ref=existing.booking_ref,
                party_size=new_party,
                date_str=new_date,
                time_str=new_time,
                seating_preference=new_seating,
                special_requests=new_special,
            )
            if updated:
                session.state = DialogueState.COMPLETED
                session.expected_slot = None
                # Sync session slots for transparency
                session.slots.update(
                    {
                        "party_size": updated.party_size,
                        "date": updated.date,
                        "time": updated.time,
                        "seating_preference": updated.seating_preference,
                        "customer_name": updated.customer_name,
                        "phone_number": updated.phone_number,
                        "booking_reference": updated.booking_ref,
                    }
                )
                return (
                    f"Done! I have updated reservation {updated.booking_ref} for {updated.customer_name}. "
                    f"It is now set for {updated.party_size} guests on {updated.date} at "
                    f"{format_display_time(updated.time)} in {updated.seating_preference} ({updated.table_name}).",
                    inv_check,
                    updated.model_dump(),
                )
            else:
                alt_str = self._format_alternatives_prompt(inv_check.alternatives)
                return (
                    f"I found booking {existing.booking_ref}, but {inv_check.conflict_reason} {alt_str}",
                    inv_check,
                    None,
                )

        # Found booking, now ask what they want to change
        session.expected_slot = None
        return (
            f"I found reservation {existing.booking_ref} under {existing.customer_name} for "
            f"{existing.party_size} guests on {existing.date} at {format_display_time(existing.time)} "
            f"({existing.seating_preference}). What would you like to change—the time, date, party size, or seating area?",
            None,
            existing.model_dump(),
        )

    def _handle_cancellation_flow(
        self,
        session: SessionState,
        selected_intent: str,
        extraction: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handles looking up and cancelling an existing reservation in SQLite."""
        session.state = DialogueState.CANCELLING_EXISTING

        if not session.target_booking_ref:
            b_ref = extraction.slots.get("booking_reference") or session.slots.get("booking_reference")
            phone = extraction.slots.get("phone_number") or session.slots.get("phone_number")
            cname = extraction.slots.get("customer_name") or session.slots.get("customer_name")

            found = self.inventory.find_booking(
                booking_ref=b_ref,
                phone_number=phone,
                customer_name=cname,
            )
            if not found:
                session.expected_slot = "booking_reference"
                return (
                    "I can help cancel your reservation. Could you please provide your booking "
                    "reference code (like DD-1001) or the phone number on the booking?",
                    None,
                )
            session.target_booking_ref = found.booking_ref
            session.slots["booking_reference"] = found.booking_ref

            # Ask for confirmation before destructive cancellation unless user already said confirm
            session.cancellation_subflow = "awaiting_confirm"
            session.expected_slot = None
            return (
                f"I located booking {found.booking_ref} under {found.customer_name} for "
                f"{found.party_size} guests on {found.date} at {format_display_time(found.time)} "
                f"({found.seating_preference}). Are you sure you want me to cancel this reservation?",
                found.model_dump(),
            )

        # Already have target_booking_ref; check confirmation
        if selected_intent == IntentName.CONFIRM_YES.value or selected_intent == IntentName.CANCEL_BOOKING.value:
            cancelled = self.inventory.cancel_booking(session.target_booking_ref)
            session.state = DialogueState.COMPLETED
            session.expected_slot = None
            if cancelled:
                return (
                    f"Your reservation {cancelled.booking_ref} under {cancelled.customer_name} has been "
                    f"cancelled and your table has been released. We hope to welcome you at {RESTAURANT_NAME} another time!",
                    cancelled.model_dump(),
                )
            return ("That reservation was already cancelled.", None)

        if selected_intent == IntentName.CONFIRM_NO.value:
            session.state = DialogueState.COMPLETED
            session.expected_slot = None
            return (
                f"Got it — I have kept your reservation {session.target_booking_ref} active and unchanged!",
                None,
            )

        return (
            f"Please confirm with 'Yes' if you want me to cancel reservation {session.target_booking_ref}, or 'No' to keep it.",
            None,
        )
