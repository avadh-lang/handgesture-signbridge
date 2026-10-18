// Mirrors the wire protocol in ../docs/ARCHITECTURE.md and the message
// shapes backend/server.py actually sends (see backend/runtime.py's
// Engine.step() and _pose_to_landmarks()).

export type HandPoints = [number, number][]; // 21 points, normalized image coords

export interface BodyRef {
  origin: [number, number];
  scale: number;
  angle: number;
}

export interface LandmarksMessage {
  type: "landmarks";
  t: number;
  left: HandPoints | null;
  right: HandPoints | null;
  body: BodyRef | null;
  jpeg?: string; // base64, present when the backend streams video frames
}

export type ModeName = "communicator" | "coach" | "control" | "pointer";

export interface CommunicatorState {
  mode: "communicator";
  reverse: boolean;
  type_out?: boolean;
  word?: string;
  sentence_text?: string;
  suggestions?: string[];
  detecting_letter?: string | null;
  detecting_conf?: number;
  hold_fraction?: number;
  flash?: string;
  bank_count?: number;
  reverse_cards?: string[];
  reverse_active?: number;
}

export interface CoachState {
  mode: "coach";
  target_kind: "letter" | "sign";
  target_name: string;
  score: number;
  pass: boolean;
  msg: string;
  session_ok: number;
  session_n: number;
  box_counts: number[];
}

export interface ControlCommand {
  name: string;
  action_type: string;
  action_value: string;
}

export interface ControlState {
  mode: "control";
  commands: ControlCommand[];
  last_fired: string;
  last_name: string;
  bound_count: number;
}

export interface PointerStatusLine {
  text: string;
  color: [number, number, number]; // BGR, as hud.py's palette
}

export interface PointerState {
  mode: "pointer";
  enabled: boolean;
  status_lines: PointerStatusLine[];
}

export type ViewStatePayload = CommunicatorState | CoachState | ControlState | PointerState;

export interface ViewStateMessage {
  type: "view_state";
  mode: ModeName;
  payload: ViewStatePayload;
}

export interface PromptMessage {
  type: "prompt";
  kind: string;
  label: string;
  initial: string;
}

export interface RecordingMessage {
  type: "recording";
  active: boolean;
  state: string;
  hint: string;
}

export type ServerMessage = LandmarksMessage | ViewStateMessage | PromptMessage | RecordingMessage;

export type ClientEvent =
  | { type: "event"; action: "mode.next" | "mode.prev" | "recording.cancel" }
  | { type: "event"; action: "prompt.submit"; value: string }
  | { type: "event"; action: "prompt.cancel" }
  | { type: "event"; action: string; [key: string]: unknown };
