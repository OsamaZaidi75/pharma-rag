// Renders an answer as React nodes — no innerHTML, so label text can never
// inject markup. Handles the shapes the backend actually produces: **bold**,
// "• [n] excerpt" bullet lines from the extractive fallback, [n] citation
// markers, and the trailing medical disclaimer (shown muted).
import type { ReactNode } from "react";

export const DISCLAIMER =
  "This is not medical advice. Consult a healthcare professional.";

// Split one line into text / bold / citation segments.
function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  // Matches **bold** or [n] citation markers.
  const re = /(\*\*[^*]+\*\*|\[\d+\])/g;
  let last = 0;
  let key = 0;
  for (const m of text.matchAll(re)) {
    const idx = m.index ?? 0;
    if (idx > last) nodes.push(text.slice(last, idx));
    const token = m[0];
    if (token.startsWith("**")) {
      nodes.push(
        <strong key={`${keyPrefix}-${key++}`}>{token.slice(2, -2)}</strong>,
      );
    } else {
      nodes.push(
        <span className="cite" key={`${keyPrefix}-${key++}`}>
          {token}
        </span>,
      );
    }
    last = idx + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export function renderAnswer(text: string): ReactNode {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = (key: string) => {
    if (bullets.length === 0) return;
    blocks.push(
      <ul key={key}>
        {bullets.map((b, i) => (
          <li key={i}>{renderInline(b, `${key}-${i}`)}</li>
        ))}
      </ul>,
    );
    bullets = [];
  };

  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (line === "") {
      flushBullets(`ul-${i}`);
      return;
    }
    if (line === DISCLAIMER) {
      flushBullets(`ul-${i}`);
      blocks.push(
        <p key={i} className="disclaimer">
          {DISCLAIMER}
        </p>,
      );
      return;
    }
    if (line.startsWith("• ")) {
      bullets.push(line.slice(2).trim());
      return;
    }
    flushBullets(`ul-${i}`);
    blocks.push(<p key={i}>{renderInline(line, `p-${i}`)}</p>);
  });
  flushBullets("ul-end");
  return blocks;
}
