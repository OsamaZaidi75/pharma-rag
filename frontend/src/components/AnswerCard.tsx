import { LoaderCircle, TriangleAlert } from "lucide-react";
import type { AgentResponse, AskResponse } from "../api";
import { renderAnswer } from "../answerText";
import TraceView from "./TraceView";

export type TurnStatus = "pending" | "done" | "error";

export interface Turn {
  id: number;
  question: string;
  mode: "chat" | "agent";
  status: TurnStatus;
  startedAt: number;
  answer?: string;
  sources?: AskResponse["sources"];
  model?: string;
  rewrittenQuestion?: string | null;
  indexedDrug?: string | null;
  indexedDrugs?: string[];
  trace?: AgentResponse["trace"];
  iterations?: number;
  error?: string;
}

interface AnswerCardProps {
  turn: Turn;
  elapsedSec: number;
}

function formatScore(score: number): string {
  return score.toFixed(2);
}

export default function AnswerCard({ turn, elapsedSec }: AnswerCardProps) {
  const isFallback = (turn.model ?? "").startsWith("extractive-fallback");

  return (
    <article className="turn">
      <h2 className="turn-question">{turn.question}</h2>

      {turn.status === "pending" && (
        <div className="working" role="status">
          <LoaderCircle size={16} strokeWidth={1.5} className="spin" />
          <div>
            <p>Searching the indexed labels…</p>
            {elapsedSec >= 8 && (
              <p className="working-note">
                Still working — a first question about a new drug downloads and
                indexes its FDA label, usually 30–60 seconds.
              </p>
            )}
          </div>
          <span className="elapsed">{elapsedSec}s</span>
        </div>
      )}

      {turn.status === "error" && (
        <div className="error-box" role="alert">
          <TriangleAlert size={16} strokeWidth={1.5} />
          <div>
            <p>{turn.error ?? "Something went wrong."}</p>
            {turn.mode === "agent" && (
              <p className="error-note">
                Agent mode needs a working LLM — NVIDIA_API_KEY on the server.
                Chat mode answers without one.
              </p>
            )}
          </div>
        </div>
      )}

      {turn.status === "done" && turn.answer && (
        <>
          <div className="answer-card">
            <div className="answer-body">{renderAnswer(turn.answer)}</div>
            <div className="answer-meta">
              <span className="meta-model">{turn.model}</span>
              {isFallback && (
                <span className="meta-note">
                  Answered from label excerpts — no LLM connected
                </span>
              )}
            </div>
          </div>

          {turn.rewrittenQuestion && (
            <p className="turn-note">
              Understood as: “{turn.rewrittenQuestion}”
            </p>
          )}
          {turn.indexedDrug && (
            <p className="turn-note">
              Indexed {turn.indexedDrug} from DailyMed for this answer.
            </p>
          )}
          {turn.indexedDrugs && turn.indexedDrugs.length > 0 && (
            <p className="turn-note">
              Indexed from DailyMed: {turn.indexedDrugs.join(", ")}.
            </p>
          )}

          {turn.mode === "agent" && turn.trace && (
            <TraceView trace={turn.trace} iterations={turn.iterations ?? 0} />
          )}

          {turn.sources && turn.sources.length > 0 && (
            <div className="sources">
              <div className="sources-label">
                Sources · {turn.sources.length} label{" "}
                {turn.sources.length === 1 ? "section" : "sections"}
              </div>
              <ul className="source-list">
                {turn.sources.map((s) => (
                  <li key={s.ref} className="source-card">
                    <span className="source-ref" aria-hidden="true">
                      {s.ref}
                    </span>
                    <div className="source-body">
                      <div className="source-head">
                        <span className="source-drug">{s.drug_name}</span>
                        <span className="source-section">
                          {s.section_title}
                        </span>
                        <span className="source-score">
                          {formatScore(s.score)}
                        </span>
                      </div>
                      <p className="source-preview">{s.preview}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </article>
  );
}
