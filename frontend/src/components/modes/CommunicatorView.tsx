import { useSignBridgeStore } from "../../state/store";
import type { CommunicatorState } from "../../types";

export function CommunicatorView({ state }: { state: CommunicatorState }) {
  const send = useSignBridgeStore((s) => s.send);

  if (state.reverse) {
    const cards = state.reverse_cards ?? [];
    return (
      <div className="flex flex-col items-center gap-4">
        <p className="text-xs text-ink-subtle">reverse channel — press v to go back</p>
        {cards.length === 0 ? (
          <p className="text-ink-tertiary">…</p>
        ) : (
          <div className="flex flex-wrap justify-center gap-3">
            {cards.map((text, i) => (
              <div
                key={i}
                className={`min-w-[120px] rounded-lg border px-5 py-6 text-center text-lg font-medium ${
                  i === state.reverse_active
                    ? "border-success text-success"
                    : "border-hairline-strong text-ink"
                }`}
              >
                {text}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  const hold = Math.min(1, Math.max(0, state.hold_fraction ?? 0));
  const suggestions = state.suggestions ?? [];

  return (
    <div className="flex flex-col gap-4">
      {state.detecting_letter && (
        <div className="flex items-center gap-3">
          <span className="font-mono text-2xl text-warn tabular">{state.detecting_letter}</span>
          <span className="text-xs text-ink-subtle tabular">
            {(state.detecting_conf ?? 0).toFixed(2)}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
            <div
              className="h-full rounded-full bg-warn transition-[width] duration-75"
              style={{ width: `${hold * 100}%` }}
            />
          </div>
        </div>
      )}

      <div className="text-lg text-ink">
        {state.word || <span className="text-ink-tertiary">waiting for a sign…</span>}
      </div>

      {state.flash && <div className="text-sm text-success">{state.flash}</div>}

      <div className="mt-2 flex items-center gap-3 border-t border-hairline pt-3">
        <span className="text-xs text-ink-subtle">
          {state.type_out ? "typing into focused app" : "not typing into other apps"}
          <kbd className="ml-1.5 rounded bg-surface-2 px-1 py-0.5 text-[11px]">t</kbd>
        </span>
        <span className="text-xs text-ink-tertiary tabular">{state.bank_count ?? 0} signs taught</span>
      </div>

      <div className="rounded-lg border border-hairline bg-surface-1 px-4 py-3">
        <p className="mb-1 min-h-[1.5em] text-base text-ink">
          {state.sentence_text || <span className="text-ink-tertiary">sentence appears here</span>}
        </p>
        {suggestions.length > 0 && (
          <div className="flex gap-2">
            {suggestions.map((word, i) => (
              <button
                key={word}
                type="button"
                onClick={() =>
                  send({ type: "event", action: "communicator.accept_suggestion", index: i })
                }
                className="rounded-md bg-surface-2 px-2.5 py-1 text-sm text-ink-muted hover:bg-surface-3 hover:text-ink"
              >
                <span className="mr-1 text-ink-tertiary tabular">{i + 1}</span>
                {word}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
