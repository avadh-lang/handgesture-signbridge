import { LETTER_ROTATION, LETTER_SHAPES, MOTION_LETTERS } from "../data/letterShapes";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

// Landmark indices (MediaPipe Hands topology, same as skeleton.py).
const WRIST = 0;
const FINGERS = [
  [1, 2, 3, 4],       // thumb
  [5, 6, 7, 8],       // index
  [9, 10, 11, 12],    // middle
  [13, 14, 15, 16],   // ring
  [17, 18, 19, 20],   // pinky
];
const PALM = [0, 5, 9, 13, 17];

function LetterIcon({ letter }: { letter: string }) {
  const hand = LETTER_SHAPES[letter];
  const rotation = LETTER_ROTATION[letter] ?? 0;
  const palmPath = PALM.map((i) => hand[i].join(",")).join(" L ");
  return (
    <svg viewBox="0 0 1 1" className="h-full w-full" role="img" aria-label={`ASL letter ${letter} handshape`}>
      <g transform={`rotate(${rotation} 0.5 0.5)`}>
        <path d={`M ${palmPath} Z`} fill="currentColor" opacity={0.22} />
        {FINGERS.map((joints, fi) => (
          <polyline
            key={fi}
            points={[hand[WRIST], ...joints.map((j) => hand[j])].map((p) => p.join(",")).join(" ")}
            fill="none"
            stroke="currentColor"
            strokeWidth={0.085}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
      </g>
    </svg>
  );
}

export function AlphabetReference() {
  return (
    <section id="alphabet-reference" className="rounded-xl border border-hairline bg-surface-1 p-5 sm:p-6">
      <h2 className="mb-4 text-[15px] font-semibold text-ink">ASL fingerspelling alphabet</h2>
      <div className="grid grid-cols-4 gap-3 sm:grid-cols-6 md:grid-cols-9 xl:grid-cols-[repeat(13,minmax(0,1fr))]">
        {ALPHABET.map((letter) => (
          <div key={letter} className="flex flex-col items-center gap-1.5 rounded-lg border border-hairline bg-surface-2 p-2.5">
            <div className="aspect-square w-full text-ink">
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
        Generated from SignBridge's own hand geometry, not a copied chart. J and Z are air-drawn -
        shown here as their static base handshape.
      </p>
    </section>
  );
}
