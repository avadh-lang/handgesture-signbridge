import { useEffect, useRef, useState } from "react";
import { useSignBridgeStore } from "../state/store";

export function PromptModal() {
  const prompt = useSignBridgeStore((s) => s.prompt);
  const send = useSignBridgeStore((s) => s.send);
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (prompt) {
      setValue(prompt.initial ?? "");
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [prompt]);

  if (!prompt) return null;

  function submit() {
    send({ type: "event", action: "prompt.submit", value });
  }
  function cancel() {
    send({ type: "event", action: "prompt.cancel" });
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={cancel}
    >
      <div
        className="w-full max-w-sm rounded-lg border border-hairline-strong bg-surface-1 p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <label className="mb-2 block text-sm text-ink-muted">{prompt.label}</label>
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
            if (e.key === "Escape") cancel();
          }}
          className="w-full rounded-md border border-hairline bg-canvas px-3 py-2 text-ink outline-none focus:border-accent-focus"
        />
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={cancel}
            className="rounded-md px-3 py-1.5 text-sm text-ink-subtle hover:text-ink"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
}
