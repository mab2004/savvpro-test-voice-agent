# DawatDesk Voice Concierge
### Haveli Grill & Courtyard (F-7 Markaz, Islamabad)
**SAVV Pro Voice Assistant Assessment Submission**

A production-grade, end-to-end voice receptionist that answers restaurant phone calls, checks live table availability in SQLite, handles messy real-caller mid-call changes, resolves busy-hour conflicts with smart recommendations, and answers mid-flow FAQ questions without losing conversational state.

> **CRITICAL RULE COMPLIANCE (`TASK.md`):**  
> *"The brain must be your own code. Do not use a big AI model (like GPT, Claude or Gemini) to decide what happens. Your code decides what the caller wants, what to ask next, and what to do."*  
> **Runtime LLM Usage: 0.00%**. There are **zero** external API calls to OpenAI, Anthropic, Gemini, or any hosted LLM. Every decision is driven by our custom 5-stage deterministic and calibrated machine-learning pipeline.

---

## Quick Links
- **Technical Design Document:** [`DESIGN.md`](./DESIGN.md) (Full 5-stage pipeline specs, Mermaid state machine diagrams, and SVM math)
- **AI Usage & Token Log:** [`AI_USAGE.md`](./AI_USAGE.md) (Transparent reflection report, prompts, debugging stories, and token metrics)
- **Task Specification:** [`TASK.md`](./TASK.md)

---

## 1. Quick Start (Run in Under 1 Minute)

### Option A: One-Click Startup Script (Recommended)

#### Windows:
```cmd
start.bat
```

#### macOS / Linux / Git Bash:
```bash
chmod +x start.sh
./start.sh
```

This starts:
- **Backend API (FastAPI):** `http://127.0.0.1:8000` (Swagger docs at `/docs`)
- **Frontend UI (React + Vite):** `http://localhost:5173`

---

### Option B: Manual Setup

#### 1. Backend (Python 3.10+):
```bash
cd backend
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### 2. Frontend (Node.js 18+):
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 2. Running Automated Tests

Run the comprehensive 32-test edge case suite:

```bash
cd backend
# Windows:
.\.venv\Scripts\pytest.exe -v
# macOS/Linux:
pytest -v
```

**Result:** `32 passed in ~3.8s` (100% pass rate, zero errors).

---

## 3. How the 5-Stage Non-LLM Brain Works

```
Caller Speaks (Microphone / Text)
       │
       ▼
Stage 1: STT Cleanup, Homophone Repair & Roman Urdu / Script Transliteration (normalizer.py)
       │
       ▼
Stage 2: Hybrid Intent Classifier & Shannon Entropy Gate (intent_classifier.py + training_data.py)
       │
       ▼
Stage 3: Context-Aware Slot Extractor & Bare-Number Disambiguation (slot_extractor.py)
       │
       ▼
Stage 4: Hierarchical Finite State Machine with Digression Stack (dialogue_manager.py)
       │
       ▼
Stage 5: SQLite Table Inventory & Conflict Resolver (inventory.py)
       │
       ▼
Assistant Speaks Reply (TTS) + Rich Decision Summary in UI
```

### The 5 Stages Explained:
1. **Stage 1 — STT Cleanup & Intra-Sentence Self-Correction (`normalizer.py`):**
   - Repairs STT homophones (`"table for for"` → `"table for 4"`, `"ate pm"` → `"8 pm"`).
   - Translates Roman Urdu and native Urdu script (`کل`, `ساڑھے سات`, `چار لوگ`, `ہاں جی بالکل`, `نہیں رہنے دو`, `۰-۹`).
   - Prunes intra-sentence speech self-corrections: *"Book for 3 people—sorry, 5 people—tomorrow at 8 PM"* extracts `5`, automatically discarding `3`.
2. **Stage 2 — Hybrid Intent Classifier (`intent_classifier.py`):**
   - Layer 1: Deterministic Dialogue-Act rules for instant confirmations/negations.
   - Layer 2: `CalibratedClassifierCV(LinearSVC)` with Word (1–3) + Char (3–5) TF-IDF n-grams trained on 485 Pakistani caller utterances across 17 intents.
   - Confidence Gate: Computes margin and Shannon entropy; inputs with confidence `< 0.50` or out-of-domain gibberish are flagged as `UNCLEAR` and gracefully clarified.
3. **Stage 3 — Context-Aware Slot Extractor (`slot_extractor.py`):**
   - Extracts `party_size` (including deltas like `"+2 people"` and additive `"3 adults and 2 kids"`), `date`, `time` (12:00 PM – 11:30 PM window inference), `seating_preference` (via `RapidFuzz`), `customer_name`, `phone_number` (Pakistani `03XX-XXXXXXX`, `+92`, and spoken double/triple digits), and `booking_reference`.
   - Disambiguates bare numbers (e.g. *"I want 6"*) based on `expected_slot` or asks for clarification.
4. **Stage 4 — Hierarchical FSM & Digression Stack (`dialogue_manager.py`):**
   - 8 states: `GREETING`, `COLLECTING_SLOTS`, `CLARIFYING_AMBIGUITY`, `RESOLVING_CONFLICT`, `CONFIRMING_BOOKING`, `MODIFYING_EXISTING`, `CANCELLING_EXISTING`, `COMPLETED`.
   - **Digression Stack:** If asked a question mid-booking (*"Wait, do you guys have valet parking?"*), answers the FAQ and seamlessly returns to asking for the exact missing slot without losing state.
   - **Mid-Call Overwrites:** If the caller changes the time or party size right at `CONFIRMING_BOOKING`, the brain automatically overwrites the slot and re-validates table inventory before confirming.
5. **Stage 5 — SQLite Table Inventory (`inventory.py`):**
   - 13 physical tables across `Indoor AC`, `Rooftop`, and `Private Family Majlis`.
   - Seeded peak-hour bookings (`DD-1001` through `DD-1004`) occupy large tables at 8:00 PM Today and Tomorrow so requests for 6+ guests deterministically trigger conflict resolution and suggest `7:30 PM`, `8:30 PM`, or `Rooftop`.

---

## 4. Messy Caller Scenarios Verified in Tests

| # | Scenario | Expected Brain Decision | Tested In |
| :--- | :--- | :--- | :--- |
| 1 | **Intra-Sentence Self-Correction** | *"Book for 3 people—sorry, 5 people—tomorrow at 8 PM"* → Extracts `5`, prunes `3`. | `test_03`, `test_04` |
| 2 | **Mid-Call Overwrite at Confirmation** | Caller reaches `CONFIRMING_BOOKING`, then says *"Actually make it 6 people at 8:30 PM"* → Overwrites slots, re-checks SQLite table capacity, updates reservation. | `test_09`, `test_10` |
| 3 | **Peak-Hour Table Conflict** | 6 guests request 8:00 PM Indoor AC (full in SQLite) → Triggers `RESOLVING_CONFLICT`, offers `7:30 PM`, `8:30 PM`, or `Rooftop`. | `test_20`, `test_21` |
| 4 | **FAQ Interruption (Digression Stack)** | Asked for phone number, caller asks *"Do you have valet parking? Is rooftop open in rain?"* → Answers both, pushes return state, re-prompts for phone number. | `test_11`, `test_12` |
| 5 | **Pakistani Minglish & Urdu Fractions** | *"Salam, kal saarhe saat baje char logon ka indoor table"* → Normalizes to `tomorrow 19:30 4 people Indoor AC`. | `test_02`, `test_07`, `test_29` |
| 6 | **Native Urdu Script (`ur-PK`)** | `"اسلام علیکم کل ساڑھے سات بجے چار لوگوں کا ٹیبل"` → Transliterates and extracts all slots. | `test_31` |
| 7 | **Modify & Cancel Existing Booking** | Quotes `DD-1001` to change time or cancel; frees table in SQLite for the next caller. | `test_24`, `test_25`, `test_26` |
| 8 | **Gibberish & OOV Input** | *"Banana spaceship quantum pineapple"* → Flagged as `UNCLEAR` via `< 0.50` entropy gate without crashing. | `test_19` |

---

## 5. Project File Structure

```
savvpro-test-voice-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI REST API (/api/turn, /api/bookings, /api/reset, /api/tts)
│   │   ├── models.py                   # Pydantic schemas, Enums, DecisionTrace, SessionState
│   │   └── brain/
│   │       ├── normalizer.py           # Stage 1: STT repair, Urdu script/Minglish transliterator, self-correction
│   │       ├── training_data.py        # 485 labeled utterances across 17 intents
│   │       ├── intent_classifier.py    # Stage 2: Calibrated LinearSVC TF-IDF + Shannon entropy gate
│   │       ├── slot_extractor.py       # Stage 3: Context-aware slot extractor + RapidFuzz
│   │       ├── dialogue_manager.py     # Stage 4: Hierarchical FSM + Digression Stack + DecisionTrace
│   │       ├── inventory.py            # Stage 5: SQLite tables & conflict resolution
│   │       └── tts_stt.py              # Zero-key offline audio helpers
│   ├── tests/
│   │   └── test_brain_edge_cases.py    # 32 comprehensive pytest integration tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Live voice console, continuous Web Speech STT, audio playback
│   │   ├── components/
│   │   │   └── BrainXRayPanel.tsx      # Reservation ticket & plain-English decision breakdown
│   │   ├── scenarios.ts                # 6 multi-turn demo scenarios & contextual quick replies
│   │   └── types.ts                    # TypeScript models matching backend API schemas
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
├── start.bat                           # 1-click Windows startup script
├── start.sh                            # 1-click Unix/macOS startup script
├── DESIGN.md                           # Comprehensive technical system design
├── AI_USAGE.md                         # Transparent AI usage and reflection report
├── TASK.md                             # Original assessment task specification
└── README.md                           # This document
```

---

## 6. How to Submit (PR Instructions)

1. Ensure all tests pass: `pytest -v`
2. Commit your work:
   ```bash
   git add .
   git commit -m "feat: complete voice-agent assessment"
   git push origin candidate-<your-github-username>
   ```
3. Open a Pull Request from your branch into `savvpro/savvpro-test-voice-agent:main` titled:
   `feat: complete voice-agent assessment`
