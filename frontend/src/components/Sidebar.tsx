import { useState } from "react";
import { Database, Pill, Plus } from "lucide-react";
import { indexDrugs, type Health } from "../api";

interface SidebarProps {
  health: Health | null;
  drugs: string[];
  healthError: boolean;
  onIndexed: () => void;
  onSelectDrug: (drug: string) => void;
}

export default function Sidebar({
  health,
  drugs,
  healthError,
  onIndexed,
  onSelectDrug,
}: SidebarProps) {
  const [drugInput, setDrugInput] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [indexNote, setIndexNote] = useState<string | null>(null);

  const llm = health?.llm;
  const llmLine = llm
    ? llm.configured
      ? `${llm.provider} · ${llm.model}`
      : `${llm.provider} · no key set`
    : "checking";

  async function handleIndex(e: React.FormEvent) {
    e.preventDefault();
    const drug = drugInput.trim().toLowerCase();
    if (!drug || indexing) return;
    setIndexing(true);
    setIndexNote(null);
    try {
      const res = await indexDrugs([drug]);
      if (res.indexed.length > 0) setIndexNote(`Indexed ${drug}`);
      else if (res.already_indexed.length > 0)
        setIndexNote(`${drug} is already indexed`);
      else setIndexNote(`No DailyMed label found for ${drug}`);
      setDrugInput("");
      onIndexed();
    } catch (err) {
      setIndexNote(err instanceof Error ? err.message : "Indexing failed");
    } finally {
      setIndexing(false);
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">
          <Pill size={16} strokeWidth={1.5} />
        </span>
        <div>
          <div className="brand-name">Pharma RAG</div>
          <div className="brand-sub">FDA label Q&amp;A</div>
        </div>
      </div>

      <div className="side-section">
        <div className="side-label">Index</div>
        {healthError ? (
          <p className="side-note">API unreachable — is the server running?</p>
        ) : health ? (
          <dl className="stat-list">
            <div className="stat-row">
              <dt>Label chunks</dt>
              <dd>{health.indexed_chunks.toLocaleString()}</dd>
            </div>
            <div className="stat-row">
              <dt>Drugs indexed</dt>
              <dd>{drugs.length}</dd>
            </div>
            <div className="stat-row">
              <dt>Answer model</dt>
              <dd className="stat-model">{llmLine}</dd>
            </div>
          </dl>
        ) : (
          <p className="side-note">Loading index status…</p>
        )}
        {llm && !llm.configured && (
          <p className="side-note">
            No LLM key set — answers use the extractive fallback, still cited.
            Agent mode needs an LLM key (NVIDIA_API_KEY on the server).
          </p>
        )}
      </div>

      <div className="side-section">
        <div className="side-label">Indexed drugs</div>
        {drugs.length === 0 ? (
          <p className="side-note">
            Nothing indexed yet. Ask about a drug, or index one below — the
            first index takes 30–60 seconds.
          </p>
        ) : (
          <ul className="drug-list">
            {drugs.map((d) => (
              <li key={d}>
                <button
                  type="button"
                  className="drug-row"
                  onClick={() => onSelectDrug(d)}
                  title={`Ask about ${d}`}
                >
                  <span className="drug-dot" aria-hidden="true" />
                  {d}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <form className="side-section index-form" onSubmit={handleIndex}>
        <div className="side-label">Index a drug</div>
        <div className="index-row">
          <input
            type="text"
            value={drugInput}
            onChange={(e) => setDrugInput(e.target.value)}
            placeholder="Generic name"
            aria-label="Drug to index"
            disabled={indexing}
          />
          <button type="submit" className="btn-secondary" disabled={indexing}>
            {indexing ? (
              <Database size={16} strokeWidth={1.5} className="spin" />
            ) : (
              <Plus size={16} strokeWidth={1.5} />
            )}
            {indexing ? "Indexing" : "Index"}
          </button>
        </div>
        {indexing && (
          <p className="side-note">
            Downloading the label from DailyMed — first index takes 30–60
            seconds.
          </p>
        )}
        {indexNote && !indexing && <p className="side-note">{indexNote}</p>}
      </form>

      <p className="side-foot">
        Answers come from official FDA labels (DailyMed). This is not medical
        advice.
      </p>
    </aside>
  );
}
