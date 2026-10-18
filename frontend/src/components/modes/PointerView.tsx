import type { PointerState } from "../../types";

// hud.py's palette constants are BGR tuples (OpenCV convention) - flip to rgb().
function bgrToCss([b, g, r]: [number, number, number]): string {
  return `rgb(${r}, ${g}, ${b})`;
}

export function PointerView({ state }: { state: PointerState }) {
  if (!state.enabled) {
    return (
      <div className="rounded-md border border-danger/30 bg-surface-1 px-4 py-3 text-sm text-danger">
        pointer unavailable — install pyautogui, then grant Accessibility permission
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {state.status_lines.map((line, i) => (
        <p key={i} className="font-mono text-sm tabular" style={{ color: bgrToCss(line.color) }}>
          {line.text}
        </p>
      ))}
      <p className="mt-2 text-xs text-ink-subtle">
        point = move · pinch = click · deep pinch = right-click · hold = drag · two fingers = scroll
      </p>
    </div>
  );
}
