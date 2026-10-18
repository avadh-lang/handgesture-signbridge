import type { CoachState } from "../../types";

export function CoachView({ state }: { state: CoachState }) {
  const pct = Math.min(1, Math.max(0, state.score)) * 100;
  const maxBox = Math.max(1, ...state.box_counts);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-baseline justify-between">
        <div>
          <span className="text-xs uppercase text-ink-subtle">
            sign this · {state.target_kind}
          </span>
          <p className="text-2xl font-semibold tracking-tight text-ink">{state.target_name}</p>
        </div>
        <span className="text-xs text-ink-subtle tabular">
          {state.session_ok}/{state.session_n} this session
        </span>
      </div>

      <div>
        <div className="h-2 overflow-hidden rounded-full bg-surface-2">
          <div
            className={`h-full rounded-full transition-[width] duration-150 ${
              state.pass ? "bg-success" : "bg-warn"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between text-xs">
          <span className="text-ink-subtle">{state.msg}</span>
          <span className="text-ink-tertiary tabular">{state.score.toFixed(2)}</span>
        </div>
      </div>

      <div>
        <p className="mb-1.5 text-xs text-ink-subtle">leitner boxes</p>
        <div className="flex items-end gap-1.5">
          {state.box_counts.map((n, i) => (
            <div key={i} className="flex flex-1 flex-col items-center gap-1">
              <div className="flex h-10 w-full items-end overflow-hidden rounded-sm bg-surface-2">
                <div
                  className="w-full rounded-sm bg-accent"
                  style={{ height: `${(n / maxBox) * 100}%` }}
                />
              </div>
              <span className="text-[11px] text-ink-tertiary tabular">{n}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
