import { useEffect, useRef } from "react";
import { HAND_CONNECTIONS } from "../lib/handTopology";
import { useSignBridgeStore } from "../state/store";
import type { HandPoints, LandmarksMessage } from "../types";

const VIDEO_W = 1280;
const VIDEO_H = 720;
const LANDMARK_INTERVAL_MS = 1000 / 30; // the backend targets ~30fps

function lerpHand(a: HandPoints | null, b: HandPoints | null, t: number): HandPoints | null {
  if (!a || !b || a.length !== b.length) return b ?? a ?? null;
  return a.map(([ax, ay], i) => {
    const [bx, by] = b[i];
    return [ax + (bx - ax) * t, ay + (by - ay) * t] as [number, number];
  });
}

function drawHand(ctx: CanvasRenderingContext2D, hand: HandPoints, color: string) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (const [ia, ib] of HAND_CONNECTIONS) {
    const [ax, ay] = hand[ia];
    const [bx, by] = hand[ib];
    ctx.moveTo(ax * w, ay * h);
    ctx.lineTo(bx * w, by * h);
  }
  ctx.stroke();

  ctx.fillStyle = color;
  for (const [x, y] of hand) {
    ctx.beginPath();
    ctx.arc(x * w, y * h, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function VideoOverlay() {
  const videoCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const lastImageRef = useRef<HTMLImageElement | null>(null);
  const lastJpegRef = useRef<string | null>(null);

  // Draw the latest decoded video frame whenever a new one arrives -
  // separate from the overlay's own animation loop below, so a 1280x720
  // JPEG decode doesn't happen on every rAF tick.
  useEffect(() => {
    const unsub = useSignBridgeStore.subscribe((state) => {
      const jpeg = state.landmarks?.jpeg;
      if (!jpeg || jpeg === lastJpegRef.current) return;
      lastJpegRef.current = jpeg;
      const img = new Image();
      img.onload = () => {
        lastImageRef.current = img;
        const canvas = videoCanvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (canvas && ctx) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = `data:image/jpeg;base64,${jpeg}`;
    });
    return unsub;
  }, []);

  // Landmark overlay: redrawn every animation frame, interpolating between
  // the last two received samples so motion reads as fluid rather than
  // stepping at the WebSocket's actual delivery rate.
  useEffect(() => {
    let raf = 0;

    function tick() {
      raf = requestAnimationFrame(tick);
      const canvas = overlayCanvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) return;

      const { landmarks, prevLandmarks } = useSignBridgeStore.getState();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!landmarks) return;

      const elapsed = performance.now() - landmarks.t * 1000;
      const t = Math.min(1, Math.max(0, elapsed / LANDMARK_INTERVAL_MS));
      const prev: LandmarksMessage | null = prevLandmarks;

      const left = lerpHand(prev?.left ?? null, landmarks.left, t);
      const right = lerpHand(prev?.right ?? null, landmarks.right, t);
      if (left) drawHand(ctx, left, "#828fff");
      if (right) drawHand(ctx, right, "#5e6ad2");
    }

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const hasFrame = useSignBridgeStore((s) => s.landmarks !== null);

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-hairline bg-surface-1">
      <canvas ref={videoCanvasRef} width={VIDEO_W} height={VIDEO_H} className="h-full w-full" />
      <canvas
        ref={overlayCanvasRef}
        width={VIDEO_W}
        height={VIDEO_H}
        className="absolute inset-0 h-full w-full"
      />
      {!hasFrame && (
        <div className="absolute inset-0 flex items-center justify-center text-ink-subtle">
          waiting for camera…
        </div>
      )}
    </div>
  );
}
