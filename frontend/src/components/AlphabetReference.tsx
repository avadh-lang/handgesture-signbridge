import { useEffect } from "react";
import { LETTER_ROTATION, LETTER_SHAPES, MOTION_LETTERS } from "../data/letterShapes";
import { HAND_CONNECTIONS } from "../lib/handTopology";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

function LetterIcon({ letter }: { letter: string }) {
  const hand = LETTER_SHAPES[letter];
  const rotation = LETTER_ROTATION[letter] ?? 0;
  return (
    <svg viewBox="0 0 1 1" className="h-full w-full" role="img" aria-label={`ASL letter ${letter} handshape`}>
      <g transform={`rotate(${rotation} 0.5 0.5)`}>
        {HAND_CONNECTIONS.map(([a, b], i) => {
          const [ax, ay] = hand[a];
          const [bx, by] = hand[b];
          return (
            <line key={i} x1={ax} y1={ay} x2={bx} y2={by} stroke="currentColor"
                  strokeWidth={0.028} strokeLinecap="round" />
          );
        })}
        {hand.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={0.032} fill="currentColor" />
        ))}
      </g>
    </svg>
  );
}

export function AlphabetReference({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 px-4 py-8" onClick={onClose}>
      <div
        className="w-full max-w-4xl rounded-xl border border-hairline-strong bg-surface-1 p-5 sm:p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-[15px] font-semibold text-ink">ASL fingerspelling alphabet</h2>
          <button type="button" onClick={onClose}
                  className="rounded-md px-2.5 py-1 text-sm text-ink-subtle hover:bg-surface-2 hover:text-ink">
            close
          </button>
        </div>
        <div className="grid grid-cols-4 gap-3 sm:grid-cols-6 md:grid-cols-7">
          {ALPHABET.map((letter) => (
            <div key={letter} className="flex flex-col items-center gap-1.5 rounded-lg border border-hairline bg-surface-2 p-2.5">
              <div className="aspect-square w-full text-ink-muted">
                <LetterIcon letter={letter} />
              </div>
              <span className="text-sm font-medium text-ink">{letter}</span>
              {MOTION_LETTERS[letter] && (
                <span className="text-center text-[10px] leading-tight text-warn">{MOTION_LETTERS[letter]}</span>
              )}
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-ink-subtle">
          Generated from SignBridge's own hand-skeleton geometry, not a copied chart. J and Z are
          air-drawn - shown here as their static base handshape.
        </p>
      </div>
    </div>
  );
}
