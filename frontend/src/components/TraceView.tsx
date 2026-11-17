import type { TraceStep } from "../api";

interface TraceViewProps {
  trace: TraceStep[];
  iterations: number;
}

export default function TraceView({ trace, iterations }: TraceViewProps) {
  if (trace.length === 0) return null;
  return (
    <details className="trace">
      <summary>
        Agent trace · {iterations} {iterations === 1 ? "step" : "steps"}
      </summary>
      <ol className="trace-list">
        {trace.map((step) => (
          <li key={step.iteration} className="trace-step">
            <span className="trace-num">{step.iteration}</span>
            <div className="trace-body">
              {step.thought && <p className="trace-thought">{step.thought}</p>}
              {step.action && (
                <p className="trace-action">
                  <code>{step.action}</code>
                  {Object.keys(step.args).length > 0 && (
                    <code className="trace-args">
                      {JSON.stringify(step.args)}
                    </code>
                  )}
                </p>
              )}
              {step.observation && (
                <p className="trace-obs">{step.observation}</p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}
