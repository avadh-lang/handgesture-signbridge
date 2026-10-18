import { useSignBridgeStore } from "../../state/store";
import type { ControlState } from "../../types";

export function ControlView({ state }: { state: ControlState }) {
  const send = useSignBridgeStore((s) => s.send);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-xs text-ink-subtle tabular">{state.bound_count} bound</span>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => send({ type: "event", action: "control.learn" })}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
          >
            learn + bind
          </button>
          <button
            type="button"
            onClick={() => send({ type: "event", action: "control.reload" })}
            className="rounded-md px-3 py-1.5 text-sm text-ink-subtle hover:bg-surface-1 hover:text-ink"
          >
            reload
          </button>
        </div>
      </div>

      {state.commands.length === 0 ? (
        <p className="text-sm text-ink-tertiary">no commands yet — learn one above</p>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {state.commands.map((c) => (
            <li
              key={c.name}
              className="flex items-center justify-between rounded-md bg-surface-1 px-3 py-2 text-sm"
            >
              <span className="text-ink">{c.name}</span>
              <span className="text-ink-subtle">
                {c.action_type}
                <span className="ml-1.5 text-ink-tertiary">{c.action_value}</span>
              </span>
            </li>
          ))}
        </ul>
      )}

      {state.last_fired && (
        <div className="rounded-md border border-success/30 bg-surface-1 px-3 py-2 text-sm text-success">
          fired {state.last_name}: {state.last_fired}
        </div>
      )}
    </div>
  );
}
