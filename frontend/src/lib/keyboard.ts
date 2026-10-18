import { useEffect } from "react";
import { useSignBridgeStore } from "../state/store";
import type { ClientEvent, ModeName } from "../types";

// Mirrors each Mode's KEYMAP in modes.py, so the same keys do the same
// thing here as in the cv2 CLI - one action vocabulary, two keyboards.
type KeyAction = { action: string; kw?: Record<string, unknown> };

const PER_MODE_KEYMAP: Record<ModeName, Record<string, KeyAction>> = {
  communicator: {
    v: { action: "communicator.toggle_reverse" },
    " ": { action: "communicator.space" },
    Backspace: { action: "backspace" },
    "1": { action: "communicator.accept_suggestion", kw: { index: 0 } },
    "2": { action: "communicator.accept_suggestion", kw: { index: 1 } },
    "3": { action: "communicator.accept_suggestion", kw: { index: 2 } },
    Enter: { action: "communicator.speak" },
    t: { action: "communicator.toggle_type_out" },
    r: { action: "communicator.teach_word" },
  },
  coach: {
    " ": { action: "coach.submit" },
    Enter: { action: "coach.submit" },
    r: { action: "coach.retry" },
    n: { action: "coach.skip" },
    a: { action: "coach.add_drill" },
  },
  control: {
    l: { action: "control.learn" },
    e: { action: "control.reload" },
    u: { action: "control.unbind_last" },
  },
  pointer: {},
};

const GLOBAL_KEYMAP: Record<string, KeyAction> = {
  Tab: { action: "mode.next" },
  "`": { action: "mode.prev" },
};

/** Routes keydown events to named backend actions, same vocabulary as the
 * cv2 CLI's keyboard. Disabled while a prompt modal owns input focus. */
export function useKeyboardActions(): void {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const { prompt, mode, send } = useSignBridgeStore.getState();
      if (prompt) return; // the prompt modal owns the keyboard while open
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const global = GLOBAL_KEYMAP[e.key];
      const perMode = PER_MODE_KEYMAP[mode]?.[e.key];
      const hit = global ?? perMode;
      if (!hit) return;

      e.preventDefault();
      const event: ClientEvent = { type: "event", action: hit.action, ...(hit.kw ?? {}) };
      send(event);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
