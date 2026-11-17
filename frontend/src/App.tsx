import { useCallback, useEffect, useRef, useState } from "react";
import { Bot, MessageSquare, RotateCcw, Send } from "lucide-react";
import {
  ask,
  askAgent,
  getDrugs,
  getHealth,
  type ChatMessage,
  type Health,
} from "./api";
import AnswerCard, { type Turn } from "./components/AnswerCard";
import Sidebar from "./components/Sidebar";

const EXAMPLE_QUESTIONS = [
  "What are the contraindications of atorvastatin?",
  "What are the side effects of metformin?",
  "Which drugs interact with atorvastatin?",
  "What is the recommended dosage of lisinopril?",
];

type Mode = "chat" | "agent";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [drugs, setDrugs] = useState<string[]>([]);
  const [healthError, setHealthError] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [mode, setMode] = useState<Mode>("chat");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [now, setNow] = useState(Date.now());
  const nextId = useRef(1);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([getHealth(), getDrugs()]);
      setHealth(h);
      setDrugs(d.drugs);
      setHealthError(false);
    } catch {
      setHealthError(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const anyPending = turns.some((t) => t.status === "pending");
  useEffect(() => {
    if (!anyPending) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [anyPending]);

  async function submit(question: string) {
    const q = question.trim();
    if (!q || sending) return;
    setInput("");
    setSending(true);
    const id = nextId.current++;
    const startedAt = Date.now();
    const currentMode = mode;
    setTurns((prev) => [
      ...prev,
      { id, question: q, mode: currentMode, status: "pending", startedAt },
    ]);

    // Chat history: prior turns only, most recent last, capped at 5 turns.
    const history: ChatMessage[] = turns
      .filter((t) => t.status === "done" && t.mode === "chat" && t.answer)
      .slice(-5)
      .flatMap((t) => [
        { role: "user" as const, content: t.question },
        { role: "assistant" as const, content: t.answer ?? "" },
      ]);

    try {
      if (currentMode === "agent") {
        const res = await askAgent(q);
        setTurns((prev) =>
          prev.map((t) =>
            t.id === id
              ? {
                  ...t,
                  status: "done",
                  answer: res.answer,
                  sources: res.sources,
                  model: res.model,
                  trace: res.trace,
                  iterations: res.iterations,
                  indexedDrugs: res.indexed_drugs,
                }
              : t,
          ),
        );
      } else {
        const res = await ask(q, history);
        setTurns((prev) =>
          prev.map((t) =>
            t.id === id
              ? {
                  ...t,
                  status: "done",
                  answer: res.answer,
                  sources: res.sources,
                  model: res.model,
                  rewrittenQuestion: res.rewritten_question,
                  indexedDrug: res.indexed_drug,
                }
              : t,
          ),
        );
      }
      refresh();
    } catch (err) {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id
            ? {
                ...t,
                status: "error",
                error: err instanceof Error ? err.message : "Request failed",
              }
            : t,
        ),
      );
    } finally {
      setSending(false);
    }
  }

  function handleSelectDrug(drug: string) {
    setInput(`What are the side effects of ${drug}?`);
    composerRef.current?.focus();
  }

  const llmUnconfigured = health?.llm && !health.llm.configured;

  return (
    <div className="app">
      <Sidebar
        health={health}
        drugs={drugs}
        healthError={healthError}
        onIndexed={refresh}
        onSelectDrug={handleSelectDrug}
      />

      <main className="main">
        <div className="main-inner">
          <header className="topbar">
            <div
              className="mode-toggle"
              role="group"
              aria-label="Answer mode"
            >
              <button
                type="button"
                className={mode === "chat" ? "mode-btn active" : "mode-btn"}
                onClick={() => setMode("chat")}
                aria-pressed={mode === "chat"}
              >
                <MessageSquare size={16} strokeWidth={1.5} />
                Chat
              </button>
              <button
                type="button"
                className={mode === "agent" ? "mode-btn active" : "mode-btn"}
                onClick={() => setMode("agent")}
                aria-pressed={mode === "agent"}
              >
                <Bot size={16} strokeWidth={1.5} />
                Agent
              </button>
            </div>
            {turns.length > 0 && (
              <button
                type="button"
                className="btn-quiet"
                onClick={() => setTurns([])}
              >
                <RotateCcw size={16} strokeWidth={1.5} />
                New conversation
              </button>
            )}
          </header>

          {turns.length === 0 && (
            <section className="intro">
              <h1>Answers from the official FDA label</h1>
              <p>
                Ask about a prescription drug. Every claim is cited to the
                label section it came from — no guessing, no uncited claims.
              </p>
              <div className="examples">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    className="example-chip"
                    onClick={() => submit(q)}
                    disabled={sending}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </section>
          )}

          <section className="composer-block">
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                submit(input);
              }}
            >
              <textarea
                ref={composerRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit(input);
                  }
                }}
                rows={2}
                maxLength={1000}
                placeholder="Ask about a drug — dosage, side effects, interactions…"
                aria-label="Your question"
                disabled={sending}
              />
              <button
                type="submit"
                className="btn-primary"
                disabled={sending || input.trim().length < 3}
              >
                <Send size={16} strokeWidth={1.5} />
                Ask
              </button>
            </form>
            <p className="composer-note">
              {mode === "chat"
                ? "Follow-ups welcome — the conversation is remembered, and follow-up questions are rewritten before retrieval."
                : "Agent mode plans its own label lookups and shows each step."}
              {mode === "agent" && llmUnconfigured && (
                <> Agent mode needs a working LLM — set NVIDIA_API_KEY on the server.</>
              )}
            </p>
          </section>

          <section className="thread" aria-live="polite">
            {turns.map((t) => (
              <AnswerCard
                key={t.id}
                turn={t}
                elapsedSec={Math.floor((now - t.startedAt) / 1000)}
              />
            ))}
          </section>

          <footer className="main-foot">
            Grounded in DailyMed SPL documents. Answers quote the label — they
            never calculate a dose.
          </footer>
        </div>
      </main>
    </div>
  );
}
