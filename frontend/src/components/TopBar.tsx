import { useSignBridgeStore } from "../state/store";
import type { ModeName } from "../types";

const MODES: { id: ModeName; label: string; hint: string }[] = [
  { id: "communicator", label: "Communicator", hint: "1" },
  { id: "coach", label: "Coach", hint: "2" },
  { id: "control", label: "Control", hint: "3" },
  { id: "pointer", label: "Pointer", hint: "4" },
];

export function TopBar() {
  const mode = useSignBridgeStore((s) => s.mode);
  const status = useSignBridgeStore((s) => s.status);
  const send = useSignBridgeStore((s) => s.send);

  return (
    <header className="flex flex-col gap-3 border-b border-hairline px-5 py-3 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="flex items-center justify-between sm:justify-start sm:gap-6">
        <span className="text-[15px] font-semibold tracking-tight text-ink">SignBridge</span>
        <ConnectionDot status={status} className="sm:hidden" />
      </div>
      <nav className="flex min-w-0 items-center gap-1 overflow-x-auto">
        {MODES.map((m) => {
          const active = m.id === mode;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => send({ type: "event", action: "mode.set", name: m.id })}
              className={`shrink-0 rounded-md px-3 py-1.5 text-sm transition-colors ${
                active
                  ? "bg-surface-2 text-ink"
                  : "text-ink-subtle hover:bg-surface-1 hover:text-ink-muted"
              }`}
            >
              {m.label}
              <span className="ml-1.5 text-ink-tertiary tabular">{m.hint}</span>
            </button>
          );
        })}
      </nav>
      <ConnectionDot status={status} className="hidden sm:flex" />
    </header>
  );
}

function ConnectionDot({
  status,
  className = "",
}: {
  status: "connecting" | "open" | "closed";
  className?: string;
}) {
  const label = status === "open" ? "connected" : status === "connecting" ? "connecting…" : "offline";
  const color =
    status === "open" ? "bg-success" : status === "connecting" ? "bg-warn" : "bg-danger";
  return (
    <div className={`flex items-center gap-2 text-xs text-ink-subtle ${className}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${color}`} />
      {label}
    </div>
  );
}
