import { useState } from "react";

// Points at frontend/public/local-only/, which is gitignored - this is the
// user's own saved reference image for personal practice, never committed
// or pushed to this public repo. Renders nothing if the file isn't present
// (e.g. on anyone else's clone), so this is safe to ship as a component.
const LOCAL_IMAGE_PATH = "/local-only/alphabet-reference.jpeg";

export function PersonalReferenceImage() {
  const [missing, setMissing] = useState(false);
  if (missing) return null;

  return (
    <section className="rounded-xl border border-hairline bg-surface-1 p-5 sm:p-6">
      <h2 className="mb-1 text-[15px] font-semibold text-ink">Your reference image</h2>
      <p className="mb-4 text-xs text-ink-subtle">
        Saved locally for your own practice - not part of the app, not committed to the repo.
      </p>
      <img
        src={LOCAL_IMAGE_PATH}
        alt="Personal ASL alphabet reference"
        className="mx-auto max-h-[70vh] w-auto rounded-lg"
        onError={() => setMissing(true)}
      />
    </section>
  );
}
