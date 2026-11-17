// Thin client for the Pharma RAG FastAPI backend. Same origin in production
// (the API serves this frontend); proxied to :8000 by Vite in development.

export interface Source {
  ref: number;
  drug_name: string;
  section_title: string;
  score: number;
  preview: string;
}

export interface Health {
  status: string;
  indexed_chunks: number;
  llm: { provider: string; model: string; configured: boolean };
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  model: string;
  indexed_drug: string | null;
  rewritten_question: string | null;
}

export interface TraceStep {
  iteration: number;
  thought: string;
  action: string | null;
  args: Record<string, unknown>;
  observation: string;
}

export interface AgentResponse {
  answer: string;
  sources: Source[];
  model: string;
  trace: TraceStep[];
  iterations: number;
  indexed_drugs: string[];
}

export interface IndexResponse {
  indexed: string[];
  already_indexed: string[];
  failed: string[];
  total_chunks: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // keep the generic message
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function getDrugs(): Promise<{ drugs: string[] }> {
  return request<{ drugs: string[] }>("/drugs");
}

export function ask(
  question: string,
  history: ChatMessage[],
): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify({ question, history }),
  });
}

export function askAgent(question: string): Promise<AgentResponse> {
  return request<AgentResponse>("/ask-agent", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function indexDrugs(drugs: string[]): Promise<IndexResponse> {
  return request<IndexResponse>("/index", {
    method: "POST",
    body: JSON.stringify({ drugs }),
  });
}
