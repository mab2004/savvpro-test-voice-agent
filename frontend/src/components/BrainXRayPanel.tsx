import React, { useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  Calendar,
  CheckCircle2,
  Clock,
  GitBranch,
  Layers,
  MapPin,
  Phone,
  RefreshCw,
  Scissors,
  ShieldAlert,
  Sparkles,
  Users,
  UserCheck,
  Wand2,
} from 'lucide-react';
import {
  BookingRecord,
  DecisionTrace,
  DialogueState,
  RestaurantTable,
  SlotMutation,
} from '../types';

interface BrainXRayPanelProps {
  trace: DecisionTrace | null;
  currentSlots: Record<string, unknown>;
  expectedSlot: string | null;
  currentState: DialogueState | string;
  bookings: BookingRecord[];
  tables: RestaurantTable[];
  onRefreshInventory: () => void;
  onQuickSendText: (text: string) => void;
}

const FRIENDLY_STATE_LABELS: Record<
  string,
  { label: string; description: string; tone: 'neutral' | 'amber' | 'green' }
> = {
  GREETING: {
    label: 'Ready for your request',
    description: 'Tell the receptionist how many guests or ask a question to begin.',
    tone: 'neutral',
  },
  COLLECTING_SLOTS: {
    label: 'Collecting reservation details',
    description: 'Filling in guests, date, time, seating area, name, and phone number.',
    tone: 'neutral',
  },
  CLARIFYING_AMBIGUITY: {
    label: 'Clarifying a detail',
    description: 'Checking whether the number you mentioned was guests or time.',
    tone: 'amber',
  },
  RESOLVING_CONFLICT: {
    label: 'Finding an available table',
    description: 'That exact slot is full—choose one of the available times or areas below.',
    tone: 'amber',
  },
  CONFIRMING_BOOKING: {
    label: 'Ready to confirm',
    description: 'A table is held for you. Say "Yes, confirm" or change any detail.',
    tone: 'amber',
  },
  MODIFYING_EXISTING: {
    label: 'Updating an existing booking',
    description: 'Looking up or changing an existing reservation.',
    tone: 'amber',
  },
  CANCELLING_EXISTING: {
    label: 'Cancelling a booking',
    description: 'Confirming cancellation of an existing reservation.',
    tone: 'amber',
  },
  COMPLETED: {
    label: 'Completed',
    description: 'Your request has been completed.',
    tone: 'green',
  },
};

const FRIENDLY_INTENT_LABELS: Record<string, string> = {
  GREET: 'Greeting',
  BOOK_TABLE: 'Book a table',
  CHECK_AVAILABILITY: 'Check table availability',
  PROVIDE_SLOT: 'Provide booking detail',
  CHANGE_SLOT: 'Change a detail',
  MODIFY_BOOKING: 'Modify existing booking',
  CANCEL_BOOKING: 'Cancel existing booking',
  CONFIRM_YES: 'Confirm (Yes)',
  CONFIRM_NO: 'Decline / Change (No)',
  FAQ_PARKING: 'Question: Parking',
  FAQ_HOURS: 'Question: Opening hours',
  FAQ_LOCATION: 'Question: Location & address',
  FAQ_MENU_HALAL: 'Question: Menu & Halal info',
  FAQ_ROOFTOP_WEATHER: 'Question: Rooftop & weather',
  FAQ_PAYMENT_CARDS: 'Question: Payment methods',
  FAQ_PRIVATE_EVENTS: 'Question: Private events',
  GOODBYE: 'End call',
  UNCLEAR: 'Unclear / Needs clarification',
};

const FRIENDLY_SLOT_NAMES: Record<string, string> = {
  party_size: 'Number of guests',
  date: 'Date',
  time: 'Time',
  seating_preference: 'Seating area',
  customer_name: 'Your name',
  phone_number: 'Phone number',
  booking_reference: 'Booking reference',
  special_requests: 'Special requests',
};

function format24hTo12h(timeStr: unknown): string {
  if (!timeStr || typeof timeStr !== 'string') return '';
  const m = timeStr.match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return String(timeStr);
  const hour24 = parseInt(m[1], 10);
  const min = m[2];
  const period = hour24 >= 12 ? 'PM' : 'AM';
  const hour12 = hour24 % 12 === 0 ? 12 : hour24 % 12;
  return `${hour12}:${min} ${period}`;
}

export const BrainXRayPanel: React.FC<BrainXRayPanelProps> = ({
  trace,
  currentSlots,
  expectedSlot,
  currentState,
  bookings,
  tables,
  onRefreshInventory,
  onQuickSendText,
}) => {
  const [activeTab, setActiveTab] = useState<'summary' | 'bookings'>('summary');

  const displaySlots = trace ? trace.current_slots : currentSlots;
  const activeExpectedSlot = trace ? trace.expected_slot_after : expectedSlot;
  const activeState = String(trace ? trace.state_after : currentState);
  const stateInfo = FRIENDLY_STATE_LABELS[activeState] || {
    label: activeState,
    description: '',
    tone: 'neutral',
  };

  const getMutationForSlot = (slotKey: string): SlotMutation | undefined => {
    if (!trace || !trace.slot_mutations) return undefined;
    return trace.slot_mutations.find((m) => m.slot === slotKey);
  };

  const reservationFields = [
    {
      key: 'party_size',
      label: 'Guests',
      icon: Users,
      format: (v: unknown) => (v ? `${v} ${Number(v) === 1 ? 'guest' : 'guests'}` : null),
    },
    {
      key: 'date',
      label: 'Date',
      icon: Calendar,
      format: (v: unknown) => (v ? String(v) : null),
    },
    {
      key: 'time',
      label: 'Time',
      icon: Clock,
      format: (v: unknown) => (v ? format24hTo12h(v) : null),
    },
    {
      key: 'seating_preference',
      label: 'Seating Area',
      icon: MapPin,
      format: (v: unknown) => (v ? String(v) : null),
    },
    {
      key: 'customer_name',
      label: 'Guest Name',
      icon: UserCheck,
      format: (v: unknown) => (v ? String(v) : null),
    },
    {
      key: 'phone_number',
      label: 'Phone Number',
      icon: Phone,
      format: (v: unknown) => (v ? String(v) : null),
    },
  ];

  const zones = ['Indoor AC', 'Rooftop', 'Private Family Majlis'];

  return (
    <section
      aria-labelledby="summary-heading"
      className="flex flex-col h-full bg-white border border-stone-200 rounded-2xl shadow-sm overflow-clip"
    >
      {/* Header with 2 Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-stone-200 bg-white">
        <div>
          <h2 id="summary-heading" className="text-base font-semibold text-stone-900">
            Reservation &amp; Assistant Decisions
          </h2>
          <p className="text-xs text-stone-500">
            Live reservation ticket &amp; non-LLM brain decision pipeline
          </p>
        </div>

        <div
          className="flex items-center bg-stone-100 p-1 rounded-xl border border-stone-200"
          role="tablist"
          aria-label="Reservation views"
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'summary'}
            onClick={() => setActiveTab('summary')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'summary'
                ? 'bg-white text-stone-900 shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Current Call
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'bookings'}
            onClick={() => setActiveTab('bookings')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'bookings'
                ? 'bg-white text-stone-900 shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Restaurant Bookings ({bookings.length})
          </button>
        </div>
      </div>

      {/* Single Smooth Scrollable Body for the Right Panel */}
      <div className="scrollable-panel flex-1 p-5 space-y-4 bg-stone-50/50">
        {activeTab === 'summary' ? (
          <>
            {/* ==============================================================
                CARD 1: LIVE RESERVATION TICKET
               ============================================================== */}
            <div className="bg-white border border-stone-200 rounded-xl p-4 shadow-sm space-y-4">
              {/* Status Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-stone-100">
                <div className="flex items-center gap-2.5">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      stateInfo.tone === 'green'
                        ? 'bg-emerald-500'
                        : stateInfo.tone === 'amber'
                        ? 'bg-amber-500'
                        : 'bg-stone-400'
                    }`}
                  />
                  <div>
                    <div className="text-sm font-semibold text-stone-900">{stateInfo.label}</div>
                    <div className="text-xs text-stone-500">{stateInfo.description}</div>
                  </div>
                </div>

                {activeExpectedSlot && FRIENDLY_SLOT_NAMES[activeExpectedSlot] && (
                  <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200">
                    Asking next: <strong>{FRIENDLY_SLOT_NAMES[activeExpectedSlot]}</strong>
                  </span>
                )}
              </div>

              {/* 6-Field Booking Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {reservationFields.map(({ key, label, icon: Icon, format }) => {
                  const rawVal = displaySlots?.[key];
                  const formatted = format(rawVal);
                  const mutation = getMutationForSlot(key);
                  const isAsking = activeExpectedSlot === key;
                  const isUpdated =
                    mutation &&
                    mutation.mutation_type !== 'initial_fill' &&
                    mutation.old_value !== null &&
                    mutation.old_value !== undefined;
                  const isNew = mutation && mutation.mutation_type === 'initial_fill';

                  return (
                    <div
                      key={key}
                      className={`p-3 rounded-xl border transition ${
                        isUpdated
                          ? 'bg-amber-50/70 border-amber-300'
                          : isNew
                          ? 'bg-emerald-50/60 border-emerald-300'
                          : isAsking
                          ? 'bg-white border-amber-300 ring-2 ring-amber-100'
                          : formatted
                          ? 'bg-stone-50/80 border-stone-200'
                          : 'bg-stone-50/40 border-stone-200/70'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1 text-xs text-stone-500 mb-1">
                        <span className="inline-flex items-center gap-1.5 font-medium">
                          <Icon className="w-3.5 h-3.5 text-stone-400" aria-hidden="true" />
                          {label}
                        </span>
                        {isAsking && (
                          <span className="text-[11px] font-semibold text-amber-700">Needed</span>
                        )}
                      </div>

                      <div className="text-sm font-semibold text-stone-900 truncate">
                        {formatted || (
                          <span className="text-stone-400 font-normal">Not set yet</span>
                        )}
                      </div>

                      {mutation && (
                        <div className="mt-1.5 text-[11px] font-medium truncate">
                          {isUpdated ? (
                            <span className="text-amber-800">
                              ↻ {String(mutation.old_value)} → {String(mutation.new_value)}
                            </span>
                          ) : (
                            <span className="text-emerald-700">✓ Just added</span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Confirmed Booking Banner */}
              {trace?.booking_record && (
                <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-start gap-2.5">
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" aria-hidden="true" />
                    <div>
                      <div className="text-sm font-bold text-emerald-950">
                        Reservation {trace.booking_record.status} — Ref #{trace.booking_record.booking_ref}
                      </div>
                      <div className="text-xs text-emerald-800 mt-0.5">
                        {trace.booking_record.customer_name} • {trace.booking_record.party_size} guests •{' '}
                        {trace.booking_record.date} at {format24hTo12h(trace.booking_record.time)} •{' '}
                        {trace.booking_record.seating_preference} (Table {trace.booking_record.table_id})
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActiveTab('bookings')}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-700 transition"
                  >
                    View in Bookings List
                  </button>
                </div>
              )}

              {/* Table Conflict & Suggested Alternatives */}
              {trace?.inventory_check?.checked && !trace.inventory_check.available && (
                <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 space-y-3">
                  <div className="flex items-start gap-2.5">
                    <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" aria-hidden="true" />
                    <div>
                      <div className="text-sm font-semibold text-amber-950">
                        Requested time or area is currently full
                      </div>
                      <p className="text-xs text-amber-800 mt-0.5">
                        {trace.inventory_check.conflict_reason}
                      </p>
                    </div>
                  </div>

                  {trace.inventory_check.alternatives.length > 0 && (
                    <div className="space-y-2 pt-1">
                      <div className="text-xs font-semibold text-amber-900">
                        Click an available option below to select it:
                      </div>
                      <div className="balanced-pill-row">
                        {trace.inventory_check.alternatives.map((alt, i) => (
                          <button
                            key={`${alt.table_id}-${alt.time}-${i}`}
                            type="button"
                            onClick={() =>
                              onQuickSendText(
                                alt.reason === 'nearest_time'
                                  ? `Let's do ${alt.display_time} instead`
                                  : `Switch to ${alt.seating_preference} please`
                              )
                            }
                            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white hover:bg-amber-100 text-amber-950 border border-amber-300 shadow-sm transition"
                          >
                            {alt.display_time} • {alt.seating_preference} ({alt.capacity}-seat table)
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Table Available & Held Notice */}
              {trace?.inventory_check?.checked &&
                trace.inventory_check.available &&
                !trace.booking_record && (
                  <div className="p-3 rounded-xl bg-emerald-50/70 border border-emerald-200 flex items-center justify-between gap-2 text-xs text-emerald-900">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" aria-hidden="true" />
                      <span>
                        <strong>Table available:</strong> {trace.inventory_check.allocated_table_name} (
                        {trace.inventory_check.allocated_seating}, seats up to{' '}
                        {trace.inventory_check.allocated_capacity})
                      </span>
                    </div>
                  </div>
                )}
            </div>

            {/* ==============================================================
                CARD 2: BRAIN DECISION PIPELINE (Visible by Default on Active Turns)
               ============================================================== */}
            {trace ? (
              <div className="bg-white border border-stone-200 rounded-xl p-4 shadow-sm space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-stone-100">
                  <div>
                    <h3 className="text-sm font-semibold text-stone-900">
                      Brain Decision Pipeline — Turn #{trace.turn_index}
                    </h3>
                    <p className="text-xs text-stone-500">
                      100% custom non-LLM decision trace ({trace.processing_ms.toFixed(1)} ms)
                    </p>
                  </div>
                  <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
                    0% LLM • Deterministic + SVM
                  </span>
                </div>

                {/* STAGE 1: Speech Cleanup & Self-Correction Diff */}
                <div className="p-3.5 rounded-xl bg-stone-50 border border-stone-200/90 space-y-2.5 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-stone-800 flex items-center gap-1.5">
                      <Wand2 className="w-3.5 h-3.5 text-emerald-700" aria-hidden="true" />
                      Stage 1 — Speech Cleanup &amp; Self-Correction Diff
                    </span>
                    {trace.detected_repair && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-100 text-amber-900 border border-amber-300">
                        <Scissors className="w-3 h-3" aria-hidden="true" />
                        Self-Repair Marker: &ldquo;{trace.detected_repair.marker}&rdquo;
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    <div className="p-2.5 rounded-lg bg-white border border-stone-200 min-w-0">
                      <div className="text-[11px] text-stone-500 mb-0.5">Raw Caller Speech:</div>
                      <div className="font-medium text-stone-900 break-words">
                        &ldquo;{trace.raw_input}&rdquo;
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white border border-emerald-200 min-w-0">
                      <div className="text-[11px] text-emerald-700 font-medium mb-0.5">
                        Normalized Clause (Sent to Brain):
                      </div>
                      <div className="font-semibold text-emerald-950 break-words">
                        &ldquo;{trace.effective_input}&rdquo;
                      </div>
                    </div>
                  </div>

                  {trace.detected_repair && (
                    <div className="p-2.5 rounded-lg bg-amber-50/80 border border-amber-200 flex flex-wrap items-center gap-2">
                      <span className="text-[11px] font-semibold text-amber-900">
                        Discarded pre-repair clause:
                      </span>
                      {trace.detected_repair.discarded_prefix && (
                        <span className="px-2 py-0.5 rounded bg-rose-100 text-rose-800 line-through font-mono text-[11px]">
                          {trace.detected_repair.discarded_prefix}
                        </span>
                      )}
                      <ArrowRight className="w-3.5 h-3.5 text-stone-400 shrink-0" aria-hidden="true" />
                      <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 font-mono font-semibold text-[11px]">
                        {trace.detected_repair.effective_text}
                      </span>
                    </div>
                  )}

                  {(trace.homophone_repairs?.length > 0 ||
                    trace.minglish_mappings?.length > 0) && (
                    <div className="balanced-pill-row pt-0.5">
                      {trace.homophone_repairs?.map((r, i) => (
                        <span
                          key={`homo-${i}`}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-50 text-purple-900 border border-purple-200 font-mono text-[11px]"
                        >
                          <Sparkles className="w-3 h-3 text-purple-600" aria-hidden="true" />
                          Homophone: {r}
                        </span>
                      ))}
                      {trace.minglish_mappings?.map((m, i) => (
                        <span
                          key={`ming-${i}`}
                          className="px-2 py-0.5 rounded bg-sky-50 text-sky-900 border border-sky-200 font-mono text-[11px]"
                        >
                          Urdu/Minglish: {m}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* STAGE 2: Hybrid Classifier & Confidence Gate */}
                <div className="p-3.5 rounded-xl bg-stone-50 border border-stone-200/90 space-y-3 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-stone-800 flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-emerald-700" aria-hidden="true" />
                      Stage 2 — Hybrid Intent Classifier &amp; Uncertainty Gate
                    </span>
                    <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-white text-stone-700 border border-stone-200">
                      Winning Layer:{' '}
                      <strong>
                        {trace.intent_source === 'rule'
                          ? 'Deterministic Dialogue-Act Rule'
                          : 'Calibrated LinearSVC (TF-IDF)'}
                      </strong>
                    </span>
                  </div>

                  {/* Top-3 Intent Bars */}
                  <div className="space-y-2">
                    {trace.top_intents.map((item, idx) => {
                      const pct = Math.round(item.probability * 100);
                      const isWinner = idx === 0;
                      return (
                        <div key={item.intent} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className={isWinner ? 'font-semibold text-stone-900' : 'text-stone-500'}>
                              #{idx + 1} {FRIENDLY_INTENT_LABELS[item.intent] || item.intent}{' '}
                              <span className="font-mono text-[11px] text-stone-400">
                                ({item.intent})
                              </span>
                            </span>
                            <span
                              className={`font-mono ${
                                isWinner ? 'font-bold text-emerald-700' : 'text-stone-500'
                              }`}
                            >
                              {(item.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full h-2 bg-stone-200/80 rounded-full overflow-clip">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${
                                isWinner
                                  ? item.intent === 'UNCLEAR'
                                    ? 'bg-rose-500'
                                    : 'bg-emerald-600'
                                  : 'bg-stone-400'
                              }`}
                              style={{ width: `${Math.max(pct, 3)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Metrics Row */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                    <div className="p-2 rounded-lg bg-white border border-stone-200">
                      <div className="text-[10px] uppercase text-stone-500">Confidence</div>
                      <div className="text-xs font-mono font-bold text-stone-900">
                        {(trace.intent_confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-white border border-stone-200">
                      <div className="text-[10px] uppercase text-stone-500">Top-2 Margin</div>
                      <div className="text-xs font-mono font-bold text-stone-900">
                        {trace.intent_margin.toFixed(3)}
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-white border border-stone-200">
                      <div className="text-[10px] uppercase text-stone-500">Shannon Entropy</div>
                      <div className="text-xs font-mono font-bold text-stone-900">
                        {trace.intent_entropy.toFixed(3)} bits
                      </div>
                    </div>
                    <div
                      className={`p-2 rounded-lg border ${
                        trace.selected_intent === 'UNCLEAR' || trace.intent_confidence < 0.5
                          ? 'bg-rose-50 border-rose-200 text-rose-800'
                          : 'bg-emerald-50 border-emerald-200 text-emerald-800'
                      }`}
                    >
                      <div className="text-[10px] uppercase opacity-80">&lt;0.50 Gate</div>
                      <div className="text-xs font-mono font-bold flex items-center gap-1">
                        {trace.selected_intent === 'UNCLEAR' || trace.intent_confidence < 0.5 ? (
                          <>
                            <ShieldAlert className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                            GATED (UNCLEAR)
                          </>
                        ) : (
                          <>
                            <CheckCircle2 className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                            PASSED
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* STAGE 3 & 4: Slot Mutations & State Machine Transition */}
                <div className="p-3.5 rounded-xl bg-stone-50 border border-stone-200/90 space-y-2.5 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-stone-800 flex items-center gap-1.5">
                      <GitBranch className="w-3.5 h-3.5 text-emerald-700" aria-hidden="true" />
                      Stage 3 &amp; 4 — Slot Mutations &amp; State Machine Transition
                    </span>
                    <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-white border border-stone-200 text-stone-800">
                      {trace.state_before}{' '}
                      <ArrowRight className="inline w-3 h-3 text-emerald-600 mx-0.5" aria-hidden="true" />{' '}
                      <strong>{trace.state_after}</strong>
                    </span>
                  </div>

                  <div className="balanced-pill-row">
                    {trace.slot_mutations?.length > 0 ? (
                      trace.slot_mutations.map((m, idx) => (
                        <span
                          key={idx}
                          className={`px-2.5 py-1 rounded-lg font-mono text-[11px] border ${
                            m.mutation_type === 'initial_fill'
                              ? 'bg-emerald-50 text-emerald-900 border-emerald-200'
                              : 'bg-amber-50 text-amber-900 border-amber-300 font-semibold'
                          }`}
                        >
                          {m.mutation_type === 'initial_fill'
                            ? `+ ${m.slot}: ${String(m.new_value)}`
                            : `↻ ${m.slot}: ${String(m.old_value)} → ${String(m.new_value)} (${m.mutation_type})`}
                        </span>
                      ))
                    ) : (
                      <span className="text-stone-500 italic">
                        No slot values changed on this turn
                      </span>
                    )}

                    {trace.expected_slot_after && (
                      <span className="px-2.5 py-1 rounded-lg font-mono text-[11px] bg-white text-stone-700 border border-stone-200">
                        Next expected slot: <strong>{trace.expected_slot_after}</strong>
                      </span>
                    )}

                    {trace.digression_triggered && (
                      <span className="px-2.5 py-1 rounded-lg font-mono text-[11px] bg-sky-50 text-sky-900 border border-sky-200 font-semibold">
                        FAQ Digression Stack: Answered {trace.faq_topic} &amp; resumed{' '}
                        {trace.state_after}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white border border-stone-200 border-dashed rounded-xl p-6 text-center space-y-1.5">
                <div className="text-sm font-semibold text-stone-700">
                  No conversation turns yet
                </div>
                <p className="text-xs text-stone-500 max-w-md mx-auto">
                  Speak into your microphone, type a message, or click one of the example prompts on the left. You&apos;ll see the live reservation ticket and the 5-stage decision trace right here.
                </p>
              </div>
            )}
          </>
        ) : (
          /* ==============================================================
             TAB 2: ALL RESTAURANT BOOKINGS & TABLES (Clean Native Card List, Single Scrollbar)
             ============================================================== */
          <div className="space-y-4">
            <div className="bg-white border border-stone-200 rounded-xl p-4 shadow-sm space-y-3.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-semibold text-stone-900">
                    Current Reservations ({bookings.length})
                  </h3>
                  <p className="text-xs text-stone-500">
                    Live SQLite database — click Change or Cancel on any reservation to test updating it.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={onRefreshInventory}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-stone-100 hover:bg-stone-200 text-stone-700 border border-stone-200 transition"
                >
                  <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" />
                  Refresh
                </button>
              </div>

              {/* Clean Native Booking Cards (No Nested Scrollbars or Clipped Rows) */}
              <ul className="space-y-2.5" role="list">
                {bookings.map((b) => (
                  <li
                    key={b.booking_ref}
                    className="p-3.5 rounded-xl bg-stone-50/70 hover:bg-stone-50 border border-stone-200 transition space-y-2.5"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span className="px-2.5 py-0.5 rounded-md font-mono text-xs font-bold bg-emerald-100 text-emerald-900 border border-emerald-200">
                          {b.booking_ref}
                        </span>
                        <span className="text-sm font-semibold text-stone-900">
                          {b.customer_name}
                        </span>
                        <span className="text-xs text-stone-500 font-mono">
                          {b.phone_number}
                        </span>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => {
                            setActiveTab('summary');
                            onQuickSendText(`I want to modify my booking ${b.booking_ref}`);
                          }}
                          className="px-2.5 py-1 rounded-lg bg-white hover:bg-stone-100 text-stone-800 border border-stone-200 font-medium text-xs shadow-sm transition"
                        >
                          Change
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setActiveTab('summary');
                            onQuickSendText(`Please cancel my booking ${b.booking_ref}`);
                          }}
                          className="px-2.5 py-1 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 font-medium text-xs transition"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-2 border-t border-stone-200/70 text-xs text-stone-700">
                      <div className="flex items-center gap-1.5">
                        <Users className="w-3.5 h-3.5 text-stone-400 shrink-0" aria-hidden="true" />
                        <span>
                          <strong>{b.party_size}</strong> guests
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-stone-400 shrink-0" aria-hidden="true" />
                        <span>
                          {b.date} at <strong>{format24hTo12h(b.time)}</strong>
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <MapPin className="w-3.5 h-3.5 text-stone-400 shrink-0" aria-hidden="true" />
                        <span className="truncate">
                          <strong>{b.seating_preference}</strong> ({b.table_id})
                        </span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>

            {/* Dining Areas & Table Capacities Summary Grid */}
            <div className="bg-white border border-stone-200 rounded-xl p-4 shadow-sm space-y-2.5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-500">
                Dining Areas &amp; Table Capacities (13 Physical Tables)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {zones.map((zone) => {
                  const zoneTables = tables.filter((t) => t.seating_area === zone);
                  return (
                    <div
                      key={zone}
                      className="p-3 rounded-lg bg-stone-50 border border-stone-200 space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-stone-900">{zone}</span>
                        <span className="text-[11px] text-stone-500">{zoneTables.length} tables</span>
                      </div>
                      <div className="balanced-pill-row">
                        {zoneTables.map((t) => (
                          <span
                            key={t.table_id}
                            className="px-2 py-0.5 rounded text-[11px] bg-white text-stone-700 border border-stone-200"
                          >
                            {t.table_id} (up to {t.capacity})
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
