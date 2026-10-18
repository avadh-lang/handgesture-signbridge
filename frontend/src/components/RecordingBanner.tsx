import { useSignBridgeStore } from "../state/store";

export function RecordingBanner() {
  const recording = useSignBridgeStore((s) => s.recording);
  const send = useSignBridgeStore((s) => s.send);
  if (!recording?.active) return null;

  return (
    <div className="absolute left-1/2 top-4 flex -translate-x-1/2 items-center gap-3 rounded-full border border-warn/40 bg-surface-1 px-4 py-2 text-sm text-ink shadow-lg">
      <span className="h-1.5 w-1.5 rounded-full bg-warn" />
      {recording.hint}
      <button
        type="button"
        onClick={() => send({ type: "event", action: "recording.cancel" })}
        className="text-ink-subtle hover:text-ink"
      >
        cancel
      </button>
    </div>
  );
}
