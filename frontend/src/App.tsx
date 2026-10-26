import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Clock,
  Globe,
  MapPin,
  MessageSquare,
  Mic,
  MicOff,
  PhoneCall,
  Play,
  RotateCcw,
  Send,
  Square,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { BrainXRayPanel } from './components/BrainXRayPanel';
import { getContextualSuggestions, MESSY_CALLER_SCENARIOS } from './scenarios';
import {
  BookingRecord,
  CallStatus,
  ChatTurnItem,
  DecisionTrace,
  DialogueState,
  PresetScenario,
  RestaurantTable,
  TurnResponse,
} from './types';

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onend: (() => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionInstance;
    webkitSpeechRecognition?: new () => SpeechRecognitionInstance;
  }
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// Wait 2.6 seconds of silence after the caller stops speaking before auto-sending
const SILENCE_AUTO_SEND_MS = 2600;

const INITIAL_SLOTS: Record<string, unknown> = {
  party_size: null,
  date: null,
  time: null,
  seating_preference: null,
  customer_name: null,
  phone_number: null,
  booking_reference: null,
  special_requests: null,
};

export const App: React.FC = () => {
  // Session & Dialogue State
  const [sessionId, setSessionId] = useState<string>(() => `call-${Date.now().toString(36)}`);
  const [dialogueState, setDialogueState] = useState<DialogueState | string>('GREETING');
  const [currentSlots, setCurrentSlots] = useState<Record<string, unknown>>(INITIAL_SLOTS);
  const [expectedSlot, setExpectedSlot] = useState<string | null>('party_size');
  const [turns, setTurns] = useState<ChatTurnItem[]>([]);
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number | null>(null);

  // Call & Audio State
  const [callStatus, setCallStatus] = useState<CallStatus>('Ready');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [autoListen, setAutoListen] = useState<boolean>(false);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [speechLang, setSpeechLang] = useState<'en-PK' | 'ur-PK' | 'en-US'>('en-PK');
  const [micNotice, setMicNotice] = useState<string | null>(null);
  const [inputText, setInputText] = useState<string>('');
  const [sttSupported, setSttSupported] = useState<boolean>(true);

  // Demo Scenario & Inventory State
  const [activeScenarioId, setActiveScenarioId] = useState<string>(MESSY_CALLER_SCENARIOS[0].id);
  const [autoPlayingScenarioId, setAutoPlayingScenarioId] = useState<string | null>(null);
  const [bookings, setBookings] = useState<BookingRecord[]>([]);
  const [tables, setTables] = useState<RestaurantTable[]>([]);
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [politeAnnouncement, setPoliteAnnouncement] = useState<string>('');

  // Refs for STT, Silence Detection, TTS, and Auto-Play
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const silenceTimerRef = useRef<number | null>(null);
  const accumulatedTranscriptRef = useRef<string>('');
  const isListeningDesiredRef = useRef<boolean>(false);
  const isSpeakingRef = useRef<boolean>(false);
  const autoPlayAbortRef = useRef<boolean>(false);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  const clearSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current !== null) {
      window.clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }, []);

  // Fetch SQLite Inventory & Bookings
  const fetchInventory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/bookings`);
      if (!res.ok) throw new Error('Failed to fetch inventory');
      const data = await res.json();
      setBookings(data.bookings || []);
      setTables(data.tables || []);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    fetchInventory();
  }, [fetchInventory]);

  // Scroll transcript to bottom when a new turn arrives
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns.length]);

  // Stop any active STT recognition safely
  const stopListening = useCallback(() => {
    isListeningDesiredRef.current = false;
    clearSilenceTimer();
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      } catch {
        // Ignore if already stopped
      }
      recognitionRef.current = null;
    }
    setIsListening(false);
  }, [clearSilenceTimer]);

  // Speak Assistant Reply (Browser speechSynthesis + Backend /api/tts WAV fallback)
  const speakAssistantReply = useCallback(
    (text: string, audioBase64?: string | null): Promise<void> => {
      return new Promise((resolve) => {
        if (isMuted || !text) {
          setCallStatus(dialogueState === 'COMPLETED' ? 'Call Completed' : 'Ready');
          resolve();
          return;
        }

        // Pause microphone recognition while assistant is speaking
        stopListening();
        isSpeakingRef.current = true;
        setCallStatus('Assistant Speaking...');

        const finishSpeaking = () => {
          isSpeakingRef.current = false;
          setCallStatus('Ready');
          resolve();
        };

        if ('speechSynthesis' in window) {
          try {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.02;
            utterance.pitch = 1.0;
            const voices = window.speechSynthesis.getVoices();
            const preferredVoice =
              voices.find(
                (v) =>
                  v.lang.startsWith('en') &&
                  (v.name.includes('Natural') ||
                    v.name.includes('Google') ||
                    v.name.includes('Aria') ||
                    v.name.includes('Zira') ||
                    v.name.includes('Samantha'))
              ) || voices.find((v) => v.lang.startsWith('en'));
            if (preferredVoice) {
              utterance.voice = preferredVoice;
            }
            utterance.onend = finishSpeaking;
            utterance.onerror = () => {
              if (audioBase64) {
                const audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
                audioElementRef.current = audio;
                audio.onended = finishSpeaking;
                audio.onerror = finishSpeaking;
                audio.play().catch(finishSpeaking);
              } else {
                finishSpeaking();
              }
            };
            window.speechSynthesis.speak(utterance);
            return;
          } catch {
            // Fall through to WAV audio
          }
        }

        if (audioBase64) {
          const audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
          audioElementRef.current = audio;
          audio.onended = finishSpeaking;
          audio.onerror = finishSpeaking;
          audio.play().catch(finishSpeaking);
        } else {
          finishSpeaking();
        }
      });
    },
    [isMuted, dialogueState, stopListening]
  );

  // Send a caller utterance to `/api/turn`
  const sendCallerTurn = useCallback(
    async (rawUtterance: string, currentSessionId?: string): Promise<TurnResponse | null> => {
      const trimmed = rawUtterance.trim();
      if (!trimmed) return null;

      stopListening();
      accumulatedTranscriptRef.current = '';
      setInputText('');
      setCallStatus('Brain Processing (~3ms)');

      const targetSession = currentSessionId || sessionId;

      try {
        const res = await fetch(`${API_BASE}/api/turn`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: targetSession,
            text: trimmed,
            include_tts: false,
          }),
        });

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        const data: TurnResponse = await res.json();
        setBackendOnline(true);
        setSessionId(data.session_id);
        setDialogueState(data.state);
        setCurrentSlots(data.slots);
        setExpectedSlot(data.expected_slot ?? null);

        const newTurn: ChatTurnItem = {
          id: `${data.session_id}-turn-${data.decision_trace.turn_index}`,
          turnIndex: data.decision_trace.turn_index,
          callerText: trimmed,
          assistantText: data.response_text,
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
          trace: data.decision_trace,
        };

        setTurns((prev) => [...prev, newTurn]);
        setSelectedTurnIndex(data.decision_trace.turn_index);
        setPoliteAnnouncement(`Receptionist replied: ${data.response_text}`);

        await fetchInventory();
        await speakAssistantReply(data.response_text, data.audio_base64);

        if (data.state === 'COMPLETED') {
          setCallStatus('Call Completed');
        }

        return data;
      } catch {
        setBackendOnline(false);
        setCallStatus('Ready');
        return null;
      }
    },
    [sessionId, stopListening, fetchInventory, speakAssistantReply]
  );

  // Start Continuous SpeechRecognition with language auto-fallback and pause tolerance
  const startListening = useCallback(
    (candidateIndex: number = 0) => {
      if (isSpeakingRef.current) {
        window.speechSynthesis?.cancel();
        isSpeakingRef.current = false;
      }

      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRec) {
        setSttSupported(false);
        return;
      }

      const langCandidates: string[] =
        speechLang === 'ur-PK'
          ? ['ur-PK', 'ur', 'en-IN', 'en-US']
          : speechLang === 'en-PK'
          ? ['en-PK', 'en-IN', 'en-GB', 'en-US']
          : ['en-US', 'en-GB'];

      const targetLang = langCandidates[candidateIndex] || 'en-US';

      stopListening();
      accumulatedTranscriptRef.current = '';
      setInputText('');
      isListeningDesiredRef.current = true;

      try {
        const recognition = new SpeechRec();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = targetLang;

        let speechReceivedOnThisSession = false;

        recognition.onresult = (event: SpeechRecognitionEvent) => {
          speechReceivedOnThisSession = true;
          let finalPart = '';
          let interimPart = '';

          for (let i = 0; i < event.results.length; i++) {
            const piece = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalPart += piece + ' ';
            } else {
              interimPart += piece;
            }
          }

          const combined = `${finalPart}${interimPart}`.replace(/\s+/g, ' ').trim();
          accumulatedTranscriptRef.current = combined;
          setInputText(combined);

          // Reset the 2.6-second silence timer every time the caller speaks a word
          clearSilenceTimer();
          if (combined.length > 0) {
            silenceTimerRef.current = window.setTimeout(() => {
              if (isListeningDesiredRef.current && accumulatedTranscriptRef.current.trim()) {
                const readyText = accumulatedTranscriptRef.current.trim();
                stopListening();
                void sendCallerTurn(readyText);
              }
            }, SILENCE_AUTO_SEND_MS);
          }
        };

        recognition.onerror = (ev) => {
          const errType = ev.error || '';

          if (errType === 'no-speech') {
            // Keep waiting if no speech yet
            return;
          }

          // If the browser rejects the language code (or immediately fails before speech),
          // automatically switch to the next fallback candidate without shutting down!
          if (
            (errType === 'language-not-supported' ||
              errType === 'service-not-allowed' ||
              errType === 'network' ||
              !speechReceivedOnThisSession) &&
            candidateIndex + 1 < langCandidates.length
          ) {
            const nextLang = langCandidates[candidateIndex + 1];
            setMicNotice(
              `Note: Browser speech engine does not support "${targetLang}". Using "${nextLang}" — our backend will still understand your Urdu & Minglish phrases.`
            );
            window.setTimeout(() => {
              if (isListeningDesiredRef.current) {
                startListening(candidateIndex + 1);
              }
            }, 60);
            return;
          }

          if (errType === 'not-allowed') {
            setMicNotice(
              'Microphone access was blocked. Please allow microphone permission in your browser address bar.'
            );
          }

          stopListening();
          setCallStatus('Ready');
        };

        recognition.onend = () => {
          // If browser ended session while user still wants to speak, seamlessly restart
          if (isListeningDesiredRef.current) {
            try {
              recognition.start();
              return;
            } catch {
              if (candidateIndex + 1 < langCandidates.length) {
                startListening(candidateIndex + 1);
                return;
              }
            }
          }
          setIsListening(false);
          setCallStatus('Ready');
        };

        recognitionRef.current = recognition;
        recognition.start();
        setIsListening(true);
        setCallStatus('Listening...');
        if (candidateIndex === 0) {
          setMicNotice(null);
        }
      } catch {
        if (candidateIndex + 1 < langCandidates.length) {
          startListening(candidateIndex + 1);
        } else {
          setIsListening(false);
          setCallStatus('Ready');
        }
      }
    },
    [speechLang, stopListening, clearSilenceTimer, sendCallerTurn]
  );

  // Hands-free Auto-Listen after reply
  useEffect(() => {
    if (
      autoListen &&
      callStatus === 'Ready' &&
      !isListening &&
      !isSpeakingRef.current &&
      !autoPlayingScenarioId &&
      dialogueState !== 'COMPLETED' &&
      turns.length > 0
    ) {
      const timer = window.setTimeout(() => {
        if (!isSpeakingRef.current && callStatus === 'Ready') {
          startListening();
        }
      }, 450);
      return () => window.clearTimeout(timer);
    }
  }, [autoListen, callStatus, isListening, autoPlayingScenarioId, dialogueState, turns.length, startListening]);

  // Start New Call Session
  const handleStartOrResetCall = useCallback(async () => {
    autoPlayAbortRef.current = true;
    setAutoPlayingScenarioId(null);
    stopListening();
    window.speechSynthesis?.cancel();
    isSpeakingRef.current = false;

    const freshId = `call-${Date.now().toString(36)}`;
    setSessionId(freshId);
    setDialogueState('GREETING');
    setCurrentSlots(INITIAL_SLOTS);
    setExpectedSlot('party_size');
    setTurns([]);
    setSelectedTurnIndex(null);
    setInputText('');
    setCallStatus('Ready');
    await fetchInventory();
  }, [stopListening, fetchInventory]);

  // Reset Entire Demo Database
  const handleFullSystemReset = useCallback(async () => {
    await handleStartOrResetCall();
    try {
      await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
      await fetchInventory();
    } catch {
      // Ignore network errors
    }
  }, [handleStartOrResetCall, fetchInventory]);

  // Auto-Play a Multi-Turn Demo Scenario
  const handleAutoPlayScenario = useCallback(
    async (scenario: PresetScenario) => {
      if (autoPlayingScenarioId) {
        autoPlayAbortRef.current = true;
        setAutoPlayingScenarioId(null);
        window.speechSynthesis?.cancel();
        return;
      }

      stopListening();
      window.speechSynthesis?.cancel();
      isSpeakingRef.current = false;

      const freshId = `sim-${scenario.id}-${Date.now().toString(36)}`;
      setSessionId(freshId);
      setDialogueState('GREETING');
      setCurrentSlots(INITIAL_SLOTS);
      setExpectedSlot('party_size');
      setTurns([]);
      setSelectedTurnIndex(null);

      autoPlayAbortRef.current = false;
      setAutoPlayingScenarioId(scenario.id);

      for (const turnText of scenario.turns) {
        if (autoPlayAbortRef.current) break;
        await sendCallerTurn(turnText, freshId);
        if (autoPlayAbortRef.current) break;
        await new Promise((r) => setTimeout(r, 450));
      }

      setAutoPlayingScenarioId(null);
    },
    [autoPlayingScenarioId, stopListening, sendCallerTurn]
  );

  const inspectedTrace: DecisionTrace | null =
    turns.find((t) => t.turnIndex === selectedTurnIndex)?.trace ??
    (turns.length > 0 ? turns[turns.length - 1].trace : null);

  const selectedScenario =
    MESSY_CALLER_SCENARIOS.find((s) => s.id === activeScenarioId) || MESSY_CALLER_SCENARIOS[0];

  const contextualSuggestions = getContextualSuggestions(String(dialogueState), expectedSlot);

  return (
    <div className="min-h-dvh flex flex-col bg-stone-50 text-stone-900">
      {/* Skip Link */}
      <a href="#content" className="skip-link visually-hidden">
        Skip to main content
      </a>

      {/* Screen Reader Polite Live Region */}
      <div aria-live="polite" aria-atomic="true" className="visually-hidden">
        {politeAnnouncement}
      </div>

      {/* ==================================================================
          CLEAN RESTAURANT HEADER
         ================================================================== */}
      <header className="bg-white border-b border-stone-200 px-4 sm:px-6 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-700 text-white flex items-center justify-center font-bold text-base shadow-sm">
              HG
            </div>
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <h1 className="text-base sm:text-lg font-bold text-stone-900">
                  Haveli Grill &amp; Courtyard
                </h1>
                <span className="text-xs font-medium px-2.5 py-0.5 rounded-full bg-stone-100 text-stone-600 border border-stone-200">
                  Phone Receptionist
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-stone-500 mt-0.5">
                <span className="inline-flex items-center gap-1">
                  <MapPin className="w-3.5 h-3.5 text-stone-400" aria-hidden="true" />
                  F-7 Markaz, Islamabad
                </span>
                <span className="inline-flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-stone-400" aria-hidden="true" />
                  Open 12:00 PM – 11:30 PM
                </span>
              </div>
            </div>
          </div>

          {/* Simple Top Actions */}
          <div className="flex items-center gap-2 flex-wrap">
            {!backendOnline && (
              <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200">
                Backend offline (Port 8000)
              </span>
            )}

            <button
              type="button"
              aria-pressed={isMuted}
              onClick={() => {
                const nextMuted = !isMuted;
                setIsMuted(nextMuted);
                if (nextMuted) {
                  window.speechSynthesis?.cancel();
                  isSpeakingRef.current = false;
                  setCallStatus('Ready');
                }
              }}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                isMuted
                  ? 'bg-stone-100 text-stone-600 border-stone-300'
                  : 'bg-white text-stone-700 border-stone-200 hover:bg-stone-50'
              }`}
            >
              {isMuted ? (
                <>
                  <VolumeX className="w-3.5 h-3.5 text-stone-500" aria-hidden="true" />
                  <span>Voice Muted</span>
                </>
              ) : (
                <>
                  <Volume2 className="w-3.5 h-3.5 text-emerald-600" aria-hidden="true" />
                  <span>Voice On</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleStartOrResetCall}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white hover:bg-stone-50 text-stone-700 border border-stone-200 transition"
            >
              <PhoneCall className="w-3.5 h-3.5 text-stone-500" aria-hidden="true" />
              <span>Start New Call</span>
            </button>

            <button
              type="button"
              onClick={handleFullSystemReset}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-stone-100 hover:bg-stone-200 text-stone-700 border border-stone-200 transition"
            >
              <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
              <span>Reset Demo Data</span>
            </button>
          </div>
        </div>
      </header>

      {/* ==================================================================
          MAIN TWO-COLUMN WORKSPACE
         ================================================================== */}
      <main
        id="content"
        tabIndex={-1}
        className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch"
      >
        {/* ================================================================
            LEFT COLUMN (6 cols): Phone Call & Conversation
           ================================================================ */}
        <section
          aria-labelledby="call-heading"
          className="lg:col-span-6 flex flex-col bg-white border border-stone-200 rounded-2xl shadow-sm overflow-clip h-[calc(100dvh-7rem)] min-h-[600px]"
        >
          {/* 1. Simple Call Status & Demo Scenario Bar */}
          <div className="px-4 py-3 bg-white border-b border-stone-200 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  callStatus === 'Listening...'
                    ? 'bg-rose-500 animate-ping'
                    : callStatus === 'Assistant Speaking...'
                    ? 'bg-emerald-500 animate-pulse'
                    : 'bg-emerald-500'
                }`}
              />
              <h2 id="call-heading" className="text-sm font-semibold text-stone-900">
                {callStatus === 'Listening...'
                  ? 'Listening (take your time — sends after 2.5s pause)...'
                  : callStatus === 'Assistant Speaking...'
                  ? 'Receptionist is speaking...'
                  : callStatus.startsWith('Brain')
                  ? 'Checking availability...'
                  : 'Receptionist on the line'}
              </h2>

              {(callStatus === 'Listening...' || callStatus === 'Assistant Speaking...') && (
                <div className="flex items-center gap-0.5 h-3 ml-1" aria-hidden="true">
                  {[0, 150, 300, 100].map((delay, i) => (
                    <span
                      key={i}
                      className={`w-1 h-3 rounded-full waveform-bar ${
                        callStatus === 'Listening...' ? 'bg-rose-500' : 'bg-emerald-600'
                      }`}
                      style={{ animationDelay: `${delay}ms` }}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* Hands-Free Auto-Listen Toggle */}
            <button
              type="button"
              aria-pressed={autoListen}
              onClick={() => setAutoListen((prev) => !prev)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition ${
                autoListen
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                  : 'bg-stone-50 text-stone-600 border-stone-200 hover:bg-stone-100'
              }`}
            >
              Auto-Listen: {autoListen ? 'On' : 'Off'}
            </button>
          </div>

          {/* Compact Demo Scenario Player Bar */}
          <div className="px-4 py-2.5 bg-stone-50 border-b border-stone-200 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 flex-1 min-w-[220px]">
              <label htmlFor="scenario-select" className="text-xs font-medium text-stone-600 shrink-0">
                Demo Call:
              </label>
              <select
                id="scenario-select"
                value={activeScenarioId}
                onChange={(e) => setActiveScenarioId(e.target.value)}
                className="flex-1 min-w-0 px-2.5 py-1.5 rounded-lg bg-white border border-stone-200 text-xs font-medium text-stone-800"
              >
                {MESSY_CALLER_SCENARIOS.map((scen) => (
                  <option key={scen.id} value={scen.id}>
                    {scen.title}
                  </option>
                ))}
              </select>
            </div>

            <button
              type="button"
              onClick={() => void handleAutoPlayScenario(selectedScenario)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                autoPlayingScenarioId
                  ? 'bg-rose-600 hover:bg-rose-700 text-white'
                  : 'bg-stone-900 hover:bg-stone-800 text-white'
              }`}
            >
              {autoPlayingScenarioId ? (
                <>
                  <Square className="w-3 h-3 fill-current" aria-hidden="true" />
                  <span>Stop Demo</span>
                </>
              ) : (
                <>
                  <Play className="w-3 h-3 fill-current" aria-hidden="true" />
                  <span>Play Demo Call</span>
                </>
              )}
            </button>
          </div>

          {/* 2. Conversation Area */}
          <div className="scrollable-panel flex-1 p-4 sm:p-5 space-y-4 bg-stone-50/40">
            {/* Initial Receptionist Greeting */}
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-800 flex items-center justify-center font-semibold text-xs shrink-0 mt-0.5">
                HG
              </div>
              <div className="bg-white border border-stone-200 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-stone-800 shadow-sm max-w-[85%] leading-relaxed">
                <div className="text-xs font-semibold text-emerald-700 mb-0.5">
                  Haveli Grill Receptionist
                </div>
                Assalam-o-Alaikum! Welcome to Haveli Grill &amp; Courtyard in F-7 Markaz. I can help you book a table, check available times, or update a reservation. How can I help you today?
              </div>
            </div>

            {/* First-Time Onboarding Guide (Visible when no turns have been sent yet) */}
            {turns.length === 0 && (
              <div className="bg-white border border-stone-200 rounded-2xl p-4 sm:p-5 shadow-sm space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-stone-900">
                    How would you like to start?
                  </h3>
                  <p className="text-xs text-stone-500 mt-0.5">
                    Click <strong>Tap to Speak</strong> below (switch mic mode to <strong>Urdu Voice</strong> if speaking Urdu), type a message, or click a sample caller phrase:
                  </p>
                </div>

                {/* Clickable Sample Phrases */}
                <div className="grid grid-cols-1 gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      void sendCallerTurn(
                        'Hi, I would like to book an indoor table for 4 people tomorrow at 7:00 PM'
                      )
                    }
                    className="text-left p-2.5 rounded-xl bg-stone-50 hover:bg-emerald-50/70 border border-stone-200 hover:border-emerald-300 transition group"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-stone-900 group-hover:text-emerald-900">
                        &ldquo;Hi, I&apos;d like to book an indoor table for 4 people tomorrow at 7:00 PM&rdquo;
                      </span>
                      <span className="text-[11px] font-medium text-emerald-700 shrink-0">
                        Book a table →
                      </span>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      void sendCallerTurn(
                        'Book a table for 3 people—no wait, actually 5 people—tomorrow at 8 PM indoor'
                      )
                    }
                    className="text-left p-2.5 rounded-xl bg-stone-50 hover:bg-emerald-50/70 border border-stone-200 hover:border-emerald-300 transition group"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-stone-900 group-hover:text-emerald-900">
                        &ldquo;Book a table for 3 people—no wait, actually 5 people—tomorrow at 8 PM indoor&rdquo;
                      </span>
                      <span className="text-[11px] font-medium text-amber-700 shrink-0">
                        Self-correction →
                      </span>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      void sendCallerTurn(
                        'Can I get an indoor table for 6 people tomorrow at 8:00 PM?'
                      )
                    }
                    className="text-left p-2.5 rounded-xl bg-stone-50 hover:bg-emerald-50/70 border border-stone-200 hover:border-emerald-300 transition group"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-stone-900 group-hover:text-emerald-900">
                        &ldquo;Can I get an indoor table for 6 people tomorrow at 8:00 PM?&rdquo;
                      </span>
                      <span className="text-[11px] font-medium text-amber-700 shrink-0">
                        8 PM busy slot →
                      </span>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      void sendCallerTurn(
                        'Salam, kal saarhe saat baje char logon ka indoor table book karna hai'
                      )
                    }
                    className="text-left p-2.5 rounded-xl bg-stone-50 hover:bg-emerald-50/70 border border-stone-200 hover:border-emerald-300 transition group"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-stone-900 group-hover:text-emerald-900">
                        &ldquo;Salam, kal saarhe saat baje char logon ka indoor table book karna hai&rdquo;
                      </span>
                      <span className="text-[11px] font-medium text-emerald-700 shrink-0">
                        Urdu / Minglish →
                      </span>
                    </div>
                  </button>
                </div>
              </div>
            )}

            {/* Conversation History */}
            {turns.map((item) => {
              const isSelected = inspectedTrace?.turn_index === item.turnIndex;
              const tr = item.trace;
              return (
                <div
                  key={item.id}
                  onClick={() => setSelectedTurnIndex(item.turnIndex)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelectedTurnIndex(item.turnIndex);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`Select turn ${item.turnIndex} to view details on the right`}
                  className={`p-3 rounded-2xl transition cursor-pointer space-y-3 ${
                    isSelected
                      ? 'bg-emerald-50/50 ring-1 ring-emerald-300'
                      : 'hover:bg-stone-100/70'
                  }`}
                >
                  {/* Caller Message */}
                  <div className="flex flex-col items-end space-y-1">
                    <div className="bg-emerald-700 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm shadow-sm max-w-[85%] min-w-0">
                      <p className="break-words">{item.callerText}</p>
                    </div>

                    {/* Subtle Summary Row Under Caller Bubble */}
                    <div className="flex flex-wrap items-center justify-end gap-1.5 text-[11px] text-stone-500">
                      <span>{item.timestamp}</span>
                      {tr.detected_repair && (
                        <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
                          Self-correction cleaned
                        </span>
                      )}
                      {tr.slot_mutations?.map((m, idx) => (
                        <span
                          key={idx}
                          className="px-1.5 py-0.5 rounded bg-stone-200/80 text-stone-800 font-medium"
                        >
                          {m.slot.replace('_', ' ')}: {String(m.new_value)}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Receptionist Reply */}
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-800 flex items-center justify-center font-semibold text-xs shrink-0 mt-0.5">
                      HG
                    </div>
                    <div className="bg-white border border-stone-200 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-stone-800 shadow-sm max-w-[85%] min-w-0 leading-relaxed">
                      <p className="break-words">{item.assistantText}</p>
                    </div>
                  </div>
                </div>
              );
            })}

            <div ref={transcriptEndRef} />
          </div>

          {/* 3. Context-Aware Suggested Replies (Cleanly Wrapped, Never Clipped) */}
          {turns.length > 0 && dialogueState !== 'COMPLETED' && (
            <div className="px-4 py-2.5 bg-stone-50 border-t border-stone-200 space-y-1.5">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-stone-500">
                <MessageSquare className="w-3.5 h-3.5 text-stone-400 shrink-0" aria-hidden="true" />
                <span>Quick replies for this step (click to send):</span>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                {contextualSuggestions.map((item) => (
                  <button
                    key={item.label}
                    type="button"
                    onClick={() => void sendCallerTurn(item.text)}
                    className="px-3 py-1 rounded-lg text-xs font-medium bg-white hover:bg-emerald-50 text-stone-700 hover:text-emerald-900 border border-stone-200 hover:border-emerald-300 transition shadow-sm"
                  >
                    &ldquo;{item.label}&rdquo;
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 4. Bottom Voice & Text Input Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (inputText.trim()) {
                void sendCallerTurn(inputText);
              }
            }}
            className="p-3.5 bg-white border-t border-stone-200 space-y-2.5"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label htmlFor="caller-utterance-input" className="text-xs font-medium text-stone-600">
                {isListening
                  ? 'Speaking... Pause for 2.5 seconds when finished, or click Send now:'
                  : 'Speak with your microphone or type your reply:'}
                {!sttSupported && (
                  <span className="ml-2 text-amber-700">
                    (Browser mic unavailable — type below)
                  </span>
                )}
              </label>

              {/* Microphone Recognition Language Selector (Fixes Urdu & Pakistani Accent STT!) */}
              <div className="flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5 text-stone-400" aria-hidden="true" />
                <label htmlFor="speech-lang-select" className="text-[11px] font-medium text-stone-500">
                  Mic Accent:
                </label>
                <select
                  id="speech-lang-select"
                  value={speechLang}
                  onChange={(e) => setSpeechLang(e.target.value as 'en-PK' | 'ur-PK' | 'en-US')}
                  className="px-2 py-0.5 rounded-md bg-stone-100 border border-stone-200 text-xs font-medium text-stone-700"
                >
                  <option value="en-PK">English / Minglish (PK)</option>
                  <option value="ur-PK">اردو / Urdu Voice (PK)</option>
                  <option value="en-US">English (US)</option>
                </select>
              </div>
            </div>

            {micNotice && (
              <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 px-3 py-1.5 rounded-lg flex items-center justify-between gap-2">
                <span>{micNotice}</span>
                <button
                  type="button"
                  onClick={() => setMicNotice(null)}
                  className="text-amber-600 hover:text-amber-900 font-bold ml-2 shrink-0"
                >
                  ✕
                </button>
              </div>
            )}

            <div className="flex items-center gap-2">
              <button
                type="button"
                aria-pressed={isListening}
                onClick={() => {
                  if (isListening) {
                    // If user already spoke words and clicks Stop, send what they said!
                    const spokenSoFar = (accumulatedTranscriptRef.current || inputText).trim();
                    stopListening();
                    if (spokenSoFar) {
                      void sendCallerTurn(spokenSoFar);
                    }
                  } else {
                    startListening();
                  }
                }}
                className={`shrink-0 inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition shadow-sm ${
                  isListening
                    ? 'bg-rose-600 hover:bg-rose-700 text-white'
                    : 'bg-emerald-600 hover:bg-emerald-700 text-white'
                }`}
              >
                {isListening ? (
                  <>
                    <MicOff className="w-4 h-4" aria-hidden="true" />
                    <span>Done Speaking</span>
                  </>
                ) : (
                  <>
                    <Mic className="w-4 h-4" aria-hidden="true" />
                    <span>Tap to Speak</span>
                  </>
                )}
              </button>

              <input
                id="caller-utterance-input"
                type="text"
                value={inputText}
                onChange={(e) => {
                  setInputText(e.target.value);
                  accumulatedTranscriptRef.current = e.target.value;
                }}
                placeholder="Type or speak (e.g., Table for 4 tomorrow at 8 PM)..."
                className="flex-1 min-w-0 px-3 py-2.5 rounded-xl bg-stone-50 border border-stone-200 text-xs sm:text-sm text-stone-900 placeholder:text-stone-400 focus:bg-white focus:border-emerald-600"
              />

              <button
                type="submit"
                disabled={!inputText.trim()}
                className="shrink-0 inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-xs font-semibold bg-stone-900 hover:bg-stone-800 disabled:opacity-40 disabled:pointer-events-none text-white transition"
              >
                <Send className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Send</span>
              </button>
            </div>
          </form>
        </section>

        {/* ================================================================
            RIGHT COLUMN (6 cols): Reservation Ticket & Decision Summary
           ================================================================ */}
        <div className="lg:col-span-6 h-[calc(100dvh-7rem)] min-h-[600px]">
          <BrainXRayPanel
            trace={inspectedTrace}
            currentSlots={currentSlots}
            expectedSlot={expectedSlot}
            currentState={dialogueState}
            bookings={bookings}
            tables={tables}
            onRefreshInventory={fetchInventory}
            onQuickSendText={(txt) => void sendCallerTurn(txt)}
          />
        </div>
      </main>
    </div>
  );
};

export default App;
