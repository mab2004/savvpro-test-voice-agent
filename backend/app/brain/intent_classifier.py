"""
Stage 2: Hybrid Intent Classifier.

Combines:
1. High-precision deterministic dialogue-act & contextual rules (handling Minglish
   affirmations/negations, explicit booking/cancel/modify phrases, FAQ keywords,
   and expected-slot responses).
2. Local scikit-learn FeatureUnion (Word N-gram TF-IDF + Char-WB N-gram TF-IDF)
   + CalibratedClassifierCV(LinearSVC) trained on startup from 485+ utterances
   in `training_data.py`.
3. Confidence calibration, top-1/top-2 margin, Shannon entropy, and an explicit
   Low-Confidence / Out-of-Vocabulary Gate (< 0.50 -> UNCLEAR).
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.brain.normalizer import normalize_utterance
from app.brain.training_data import TRAINING_CORPUS
from app.models import DialogueState, FAQ_INTENTS, IntentName, IntentPrediction, IntentScore


CONFIDENCE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Deterministic Dialogue-Act Patterns
# ---------------------------------------------------------------------------

_FAQ_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        IntentName.FAQ_PARKING.value,
        re.compile(r"\b(?:valet|parking|park\s+our\s+car|gari\s+park|car\s+park)\b", re.IGNORECASE),
    ),
    (
        IntentName.FAQ_ROOFTOP_WEATHER.value,
        re.compile(
            r"\b(?:rain|rains|raining|drizzle|barish|weather|heaters?|mist\s+fans?|pergola|canopy|shaded|covered\s+rooftop|sardi)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentName.FAQ_PAYMENT_CARDS.value,
        re.compile(
            r"\b(?:credit\s+cards?|debit\s+cards?|visa|mastercard|amex|apple\s+pay|jazzcash|easypaisa|raast|nayapay|sadapay|card\s+payment|pay\s+by\s+card|bank\s+discount|cash\s+only)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentName.FAQ_HOURS.value,
        re.compile(
            r"\b(?:opening\s+hours|operating\s+hours|business\s+hours|what\s+time\s+do\s+you\s+(?:open|close)|when\s+do\s+you\s+(?:open|close)|closing\s+time|timings\s+kya|kitne\s+baje\s+(?:tak\s+)?(?:khula|band|open)|how\s+late\s+are\s+you\s+open|kitchen\s+close)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentName.FAQ_LOCATION.value,
        re.compile(
            r"\b(?:where\s+(?:exactly\s+)?are\s+you\s+located|your\s+address|exact\s+address|location\s+kahan|address\s+bata|which\s+sector|directions\s+to|nearest\s+landmark|where\s+is\s+haveli)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentName.FAQ_MENU_HALAL.value,
        re.compile(
            r"\b(?:halal|zabiha|menu|signature\s+dish|special\s+dish|karahi|bbq\s+platter|vegetarian|vegan|gluten\s+free|what\s+kind\s+of\s+food|cuisine|buffet|a\s+la\s+carte|mutton\s+ribs|chapli\s+kabab|dumpukht)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentName.FAQ_PRIVATE_EVENTS.value,
        re.compile(
            r"\b(?:banquet|corporate\s+(?:event|dinner)|private\s+party|bridal\s+shower|aqeeqa|minimum\s+spend|maximum\s+capacity|max\s+party\s+size|large\s+group\s+policy)\b",
            re.IGNORECASE,
        ),
    ),
]

_CONFIRM_YES_EXACT_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|correct|right|absolutely|definitely|confirm|confirmed|"
    r"yes\s+please|yes\s+confirm(?:\s+it)?|yeah\s+that'?s\s+(?:right|correct)|"
    r"go\s+ahead(?:\s+and\s+book(?:\s+it)?)?|lock\s+it\s+in|book\s+it|"
    r"haan|han|jee|ji|haan\s+jee|haan\s+ji|jee\s+haan|ji\s+haan|bilkul|"
    r"haan\s+jee\s+bilkul(?:\s+confirm\s+kar\s+(?:dein|do))?|"
    r"bilkul\s+theek\s+hai(?:\s+confirm\s+kar\s+(?:do|dein))?|"
    r"theek\s+hai(?:\s+confirm\s+kar\s+(?:dein|do)|\s+book\s+kar\s+(?:dein|do))?|"
    r"ji\s+confirm\s+kar\s+(?:dein|do)|sahi\s+hai(?:\s+confirm)?|done(?:\s+kar\s+dein)?|"
    r"yes\s+everything\s+looks\s+good|looks\s+good\s+confirm|okay\s+yes(?:\s+confirm)?)$",
    re.IGNORECASE,
)

_CONFIRM_NO_EXACT_RE = re.compile(
    r"^(?:no|nope|nah|not\s+now|no\s+thanks|no\s+thank\s+you|"
    r"no\s+don'?t\s+(?:book|confirm|cancel)(?:\s+it|\s+yet)?|"
    r"no\s+wait\s+don'?t\s+confirm(?:\s+yet)?|"
    r"nahi|na|nahi\s+rehne\s+(?:do|dein)|rehne\s+(?:do|dein)(?:\s+bhai)?|"
    r"nahi\s+chhoro|chhoro\s+rehne\s+dein|nahi\s+abhi\s+nahi|"
    r"nahi\s+confirm\s+(?:na|mat)\s+karein|no\s+forget\s+it|no\s+i\s+changed\s+my\s+mind)$",
    re.IGNORECASE,
)

_CANCEL_BOOKING_RE = re.compile(
    r"\b(?:cancel\s+(?:my|our|the|this)?\s*(?:booking|reservation|table)|"
    r"(?:booking|reservation|table)\s+cancel\s+(?:kar|karna|karni|karwani)|"
    r"cancel\s+DD-\d{3,6}|meri\s+(?:booking|reservation)\s+cancel)\b",
    re.IGNORECASE,
)

_MODIFY_BOOKING_RE = re.compile(
    r"\b(?:modify\s+(?:my|our|the|existing)?\s*(?:booking|reservation)|"
    r"(?:change|update|edit|reschedule)\s+(?:my|our|the)?\s*existing\s+(?:booking|reservation)|"
    r"(?:modify|update|edit|reschedule)\s+(?:booking|reservation)\s+DD-\d{3,6}|"
    r"DD-\d{3,6}\s+(?:ka\s+time\s+change|modify|update|reschedule)|"
    r"already\s+have\s+a\s+(?:booking|reservation).*(?:change|modify|move|reschedule)|"
    r"existing\s+(?:booking|reservation)\s+(?:change|modify|update))\b",
    re.IGNORECASE,
)

_CHANGE_SLOT_RE = re.compile(
    r"^(?:actually|wait|hold\s+on|sorry|no\s+wait|wait\s+no|nahi|balke|balkay)\b.*"
    r"\b(?:make\s+it|change\s+(?:it|the|time|date|party|seating)|switch\s+to|instead|kar\s+do|kar\s+dein|"
    r"\d+\s*(?:people|guests|persons|pm|am|baje)|rooftop|indoor|majlis|tomorrow|today)\b"
    r"|\b(?:change\s+(?:the\s+)?(?:time|date|party\s+size|number\s+of\s+guests|seating)\s+to|"
    r"make\s+it\s+\d+\s*(?:people|guests|pm|am)|"
    r"instead\s+of\s+\w+\s+(?:make\s+it|let'?s\s+do)|"
    r"ki\s+jagah\s+.+\s+kar\s+(?:do|dein))\b",
    re.IGNORECASE,
)

_BOOK_TABLE_RE = re.compile(
    r"\b(?:book\s+(?:a|an|us\s+a)?\s*(?:indoor|rooftop|majlis|family)?\s*table|"
    r"make\s+a\s+(?:table\s+)?reservation|reserve\s+a\s+(?:rooftop\s+|indoor\s+)?table|"
    r"table\s+book\s+(?:karna|karni|karwana|kar\s+dein|kar\s+do)|"
    r"reservation\s+(?:karni|karwani|kar\s+dein)|"
    r"book\s+for\s+\d+|want\s+to\s+book\s+a\s+table|need\s+a\s+table\s+for)\b",
    re.IGNORECASE,
)

_CHECK_AVAILABILITY_RE = re.compile(
    r"\b(?:check\s+(?:table\s+)?availability|do\s+you\s+have\s+(?:any\s+)?(?:tables?|space|room|seating)\s+(?:available|free|open)|"
    r"is\s+(?:a\s+table|rooftop|indoor|majlis|there)\s+(?:available|free|open)|"
    r"are\s+(?:any\s+)?tables\s+available|table\s+mil\s+(?:jayega|sakta\s+hai)|"
    r"jagah\s+(?:available\s+)?hai|khali\s+hai)\b",
    re.IGNORECASE,
)

_GOODBYE_RE = re.compile(
    r"\b(?:goodbye|bye\s+bye|khuda\s+hafiz|allah\s+hafiz|fi\s+amanillah|"
    r"that'?s\s+all\s+i\s+needed|nothing\s+else\s+thanks)\b",
    re.IGNORECASE,
)

_GREET_ONLY_RE = re.compile(
    r"^(?:hello|hi|hey|salam|assalam\s*[ou]?\s*alaikum|aoa|adaab|"
    r"good\s+(?:morning|afternoon|evening)|hello\s+ji|salam\s+bhai|salam\s+jee|"
    r"hi\s+there|hey\s+there)(?:\s+haveli(?:\s+grill(?:\s+and\s+courtyard)?)?)?[!.?]*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Hybrid Intent Classifier Implementation
# ---------------------------------------------------------------------------

class HybridIntentClassifier:
    """
    Two-tier local intent classifier:
    Tier 1: Contextual dialogue-act & pattern rules
    Tier 2: Calibrated LinearSVC over Word (1-3) + Char_WB (3-5) TF-IDF features
    Tier 3: Confidence & Out-of-Vocabulary Gate (< 0.50 -> UNCLEAR)
    """

    def __init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
        )
        self.features = FeatureUnion(
            [
                ("word_tfidf", self.word_vectorizer),
                ("char_tfidf", self.char_vectorizer),
            ]
        )
        base_svc = LinearSVC(C=1.2, class_weight="balanced", random_state=42)
        self.classifier = CalibratedClassifierCV(estimator=base_svc, cv=5, method="sigmoid")
        self.pipeline = Pipeline(
            [
                ("features", self.features),
                ("clf", self.classifier),
            ]
        )
        self._is_trained = False
        self._known_vocab: set[str] = set()
        self.train()

    def train(self) -> None:
        """Trains the TF-IDF + Calibrated LinearSVC pipeline on startup."""
        texts: List[str] = []
        labels: List[str] = []
        for raw_utt, label in TRAINING_CORPUS:
            norm = normalize_utterance(raw_utt)
            texts.append(norm.effective_text.lower())
            labels.append(label)

        self.pipeline.fit(texts, labels)
        self._known_vocab = set(self.word_vectorizer.vocabulary_.keys())
        self._is_trained = True

    def _compute_vocab_coverage(self, text: str) -> float:
        """
        Computes the proportion of tokens in `text` that exist in the trained
        word vocabulary or look like domain entities (digits, booking refs, phone numbers).
        Prevents gibberish/out-of-domain strings from getting artificially high
        char-ngram scores.
        """
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return 0.0
        matched = 0
        for tok in tokens:
            if tok in self._known_vocab or tok.isdigit() or re.match(r"^dd\d+$", tok):
                matched += 1
        return matched / len(tokens)

    def predict(
        self,
        text: str,
        raw_text: Optional[str] = None,
        dialogue_state: Optional[DialogueState] = None,
        expected_slot: Optional[str] = None,
        has_extracted_slots: bool = False,
        repair_detected: bool = False,
    ) -> IntentPrediction:
        """
        Predicts the caller's primary and secondary intents with calibrated
        probabilities, margin, entropy, and low-confidence gating.
        """
        clean = (text or "").strip()
        raw_check = (raw_text or clean).strip()

        if not clean:
            return IntentPrediction(
                top_intent=IntentName.UNCLEAR.value,
                confidence=0.0,
                margin=0.0,
                entropy=1.0,
                is_unclear=True,
                source="low_confidence_gate",
                top_3=[IntentScore(intent=IntentName.UNCLEAR.value, probability=1.0)],
            )

        # 1. Run ML pipeline to get full probability distribution
        probs = self.pipeline.predict_proba([clean.lower()])[0]
        classes = list(self.pipeline.classes_)

        # Adjust confidence by vocabulary coverage to catch gibberish / OOD queries
        coverage = self._compute_vocab_coverage(clean)
        if coverage == 0.0 and not has_extracted_slots:
            return IntentPrediction(
                top_intent=IntentName.UNCLEAR.value,
                confidence=0.12,
                margin=0.0,
                entropy=1.0,
                is_unclear=True,
                source="low_confidence_gate",
                top_3=[IntentScore(intent=IntentName.UNCLEAR.value, probability=0.88)],
            )

        ranked_indices = np.argsort(probs)[::-1]
        top_3_scores = [
            IntentScore(intent=str(classes[idx]), probability=round(float(probs[idx]), 4))
            for idx in ranked_indices[:3]
        ]

        ml_top_intent = str(classes[ranked_indices[0]])
        ml_top_prob = float(probs[ranked_indices[0]])
        ml_second_prob = float(probs[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0

        # Scale ML probability if vocabulary coverage is very weak (< 0.35) and no slots extracted
        if coverage < 0.35 and not has_extracted_slots:
            ml_top_prob *= coverage

        margin = round(max(0.0, ml_top_prob - ml_second_prob), 4)

        # Normalized Shannon entropy over classes [0..1]
        max_entropy = math.log2(len(classes)) if len(classes) > 1 else 1.0
        raw_entropy = -sum(float(p) * math.log2(max(float(p), 1e-12)) for p in probs)
        norm_entropy = round(min(1.0, raw_entropy / max_entropy), 4)

        # 2. Check deterministic rules & detect secondary intents
        rule_intents: List[str] = []

        # Check FAQ patterns (on both raw and normalized text)
        for faq_intent, faq_pat in _FAQ_PATTERNS:
            if faq_pat.search(raw_check) or faq_pat.search(clean):
                rule_intents.append(faq_intent)

        # Check explicit Cancel / Modify / Change / Book / Availability / Confirm / Greet / Goodbye
        if _CANCEL_BOOKING_RE.search(raw_check) or _CANCEL_BOOKING_RE.search(clean):
            rule_intents.append(IntentName.CANCEL_BOOKING.value)

        if _MODIFY_BOOKING_RE.search(raw_check) or _MODIFY_BOOKING_RE.search(clean):
            rule_intents.append(IntentName.MODIFY_BOOKING.value)

        if _CONFIRM_YES_EXACT_RE.match(raw_check) or _CONFIRM_YES_EXACT_RE.match(clean):
            rule_intents.append(IntentName.CONFIRM_YES.value)

        if _CONFIRM_NO_EXACT_RE.match(raw_check) or _CONFIRM_NO_EXACT_RE.match(clean):
            rule_intents.append(IntentName.CONFIRM_NO.value)

        if _GREET_ONLY_RE.match(raw_check) or _GREET_ONLY_RE.match(clean):
            rule_intents.append(IntentName.GREET.value)

        if _CHANGE_SLOT_RE.search(raw_check) or _CHANGE_SLOT_RE.search(clean):
            # Only treat as CHANGE_SLOT if we aren't starting a brand new booking from GREETING
            if dialogue_state not in (None, DialogueState.GREETING):
                rule_intents.append(IntentName.CHANGE_SLOT.value)

        if _BOOK_TABLE_RE.search(raw_check) or _BOOK_TABLE_RE.search(clean):
            rule_intents.append(IntentName.BOOK_TABLE.value)

        if _CHECK_AVAILABILITY_RE.search(raw_check) or _CHECK_AVAILABILITY_RE.search(clean):
            rule_intents.append(IntentName.CHECK_AVAILABILITY.value)

        if _GOODBYE_RE.search(raw_check) or _GOODBYE_RE.search(clean):
            if not has_extracted_slots or dialogue_state == DialogueState.COMPLETED:
                rule_intents.append(IntentName.GOODBYE.value)

        # Context-aware slot response rule:
        # If the caller is actively answering a slot prompt or conflict resolution and slots were extracted
        if has_extracted_slots and not rule_intents:
            if dialogue_state in (
                DialogueState.COLLECTING_SLOTS,
                DialogueState.CLARIFYING_AMBIGUITY,
                DialogueState.RESOLVING_CONFLICT,
                DialogueState.MODIFYING_EXISTING,
                DialogueState.CANCELLING_EXISTING,
            ):
                if repair_detected and dialogue_state == DialogueState.CONFIRMING_BOOKING:
                    rule_intents.append(IntentName.CHANGE_SLOT.value)
                else:
                    rule_intents.append(IntentName.PROVIDE_SLOT.value)
            elif dialogue_state == DialogueState.CONFIRMING_BOOKING:
                rule_intents.append(IntentName.CHANGE_SLOT.value)
            elif dialogue_state in (None, DialogueState.GREETING):
                rule_intents.append(IntentName.BOOK_TABLE.value)

        # 3. Arbitrate between Rule Intents and ML Top Intent
        if rule_intents:
            primary = rule_intents[0]
            secondary = [i for i in rule_intents[1:] if i != primary]
            if ml_top_intent != primary and ml_top_prob >= 0.35 and ml_top_intent not in secondary:
                secondary.append(ml_top_intent)

            # Boost confidence when deterministic rule and ML agree, or set rule confidence
            boosted_conf = round(max(ml_top_prob, 0.88 if ml_top_intent == primary else 0.84), 4)
            source = "hybrid" if ml_top_intent == primary else "rule"

            # Ensure primary appears at top of top_3 for transparent trace
            merged_top_3: List[IntentScore] = [IntentScore(intent=primary, probability=boosted_conf)]
            for item in top_3_scores:
                if item.intent != primary and len(merged_top_3) < 3:
                    merged_top_3.append(item)

            return IntentPrediction(
                top_intent=primary,
                confidence=boosted_conf,
                margin=round(max(margin, 0.45), 4),
                entropy=norm_entropy,
                is_unclear=False,
                source=source,
                top_3=merged_top_3,
                secondary_intents=secondary,
            )

        # 4. Pure ML path with Low-Confidence Gate (< 0.50 -> UNCLEAR)
        if ml_top_prob < CONFIDENCE_THRESHOLD:
            return IntentPrediction(
                top_intent=IntentName.UNCLEAR.value,
                confidence=round(ml_top_prob, 4),
                margin=margin,
                entropy=norm_entropy,
                is_unclear=True,
                source="low_confidence_gate",
                top_3=top_3_scores,
                secondary_intents=[],
            )

        return IntentPrediction(
            top_intent=ml_top_intent,
            confidence=round(ml_top_prob, 4),
            margin=margin,
            entropy=norm_entropy,
            is_unclear=False,
            source="ml",
            top_3=top_3_scores,
            secondary_intents=[],
        )


# Module-level singleton trained once on import
_CLASSIFIER_INSTANCE: Optional[HybridIntentClassifier] = None


def get_intent_classifier() -> HybridIntentClassifier:
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = HybridIntentClassifier()
    return _CLASSIFIER_INSTANCE
