export type DialogueState =
  | 'GREETING'
  | 'COLLECTING_SLOTS'
  | 'CLARIFYING_AMBIGUITY'
  | 'RESOLVING_CONFLICT'
  | 'CONFIRMING_BOOKING'
  | 'MODIFYING_EXISTING'
  | 'CANCELLING_EXISTING'
  | 'COMPLETED';

export type CallStatus =
  | 'Ready'
  | 'Listening...'
  | 'Brain Processing (~3ms)'
  | 'Assistant Speaking...'
  | 'Call Completed';

export interface IntentScore {
  intent: string;
  probability: number;
}

export interface SlotMutation {
  slot: string;
  old_value: unknown;
  new_value: unknown;
  mutation_type: 'initial_fill' | 'mid_call_overwrite' | 'self_correction' | 'conflict_resolution' | string;
}

export interface AlternativeOption {
  date: string;
  time: string;
  display_time: string;
  seating_preference: string;
  table_id: string;
  table_name: string;
  capacity: number;
  reason: 'nearest_time' | 'alternative_seating' | string;
}

export interface InventoryCheckResult {
  checked: boolean;
  available: boolean;
  allocated_table_id?: string | null;
  allocated_table_name?: string | null;
  allocated_seating?: string | null;
  allocated_capacity?: number | null;
  conflict_reason?: string | null;
  alternatives: AlternativeOption[];
}

export interface DetectedRepair {
  marker?: string | null;
  discarded_prefix?: string | null;
  effective_text?: string;
}

export interface DecisionTrace {
  turn_index: number;
  timestamp: string;
  processing_ms: number;
  raw_input: string;
  normalized_input: string;
  effective_input: string;
  homophone_repairs: string[];
  minglish_mappings: string[];
  detected_repair?: DetectedRepair | null;
  top_intents: IntentScore[];
  selected_intent: string;
  intent_confidence: number;
  intent_margin: number;
  intent_entropy: number;
  intent_source: 'rule' | 'ml' | string;
  extracted_slots: Record<string, unknown>;
  slot_mutations: SlotMutation[];
  current_slots: Record<string, unknown>;
  missing_slots: string[];
  expected_slot_before?: string | null;
  expected_slot_after?: string | null;
  state_before: DialogueState | string;
  state_after: DialogueState | string;
  digression_triggered: boolean;
  faq_topic?: string | null;
  ambiguity_detected?: string | null;
  validation_errors: Record<string, string>;
  inventory_check?: InventoryCheckResult | null;
  booking_record?: BookingRecord | null;
  assistant_response: string;
}

export interface BookingRecord {
  booking_ref: string;
  customer_name: string;
  phone_number: string;
  party_size: number;
  date: string;
  time: string;
  seating_preference: string;
  table_id: string;
  table_name: string;
  special_requests?: string | null;
  status: string;
  created_at: string;
}

export interface RestaurantTable {
  table_id: string;
  table_name: string;
  seating_area: string;
  capacity: number;
}

export interface TurnResponse {
  session_id: string;
  response_text: string;
  state: DialogueState | string;
  slots: Record<string, unknown>;
  missing_slots: string[];
  expected_slot?: string | null;
  booking?: BookingRecord | null;
  decision_trace: DecisionTrace;
  audio_base64?: string | null;
}

export interface ChatTurnItem {
  id: string;
  turnIndex: number;
  callerText: string;
  assistantText: string;
  timestamp: string;
  trace: DecisionTrace;
}

export interface PresetScenario {
  id: string;
  title: string;
  badge: string;
  description: string;
  turns: string[];
}
