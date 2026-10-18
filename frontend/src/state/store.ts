import { create } from "zustand";
import type {
  ClientEvent,
  LandmarksMessage,
  ModeName,
  PromptMessage,
  RecordingMessage,
  ViewStateMessage,
} from "../types";

export type ConnectionStatus = "connecting" | "open" | "closed";

interface SignBridgeState {
  status: ConnectionStatus;
  mode: ModeName;
  landmarks: LandmarksMessage | null;
  prevLandmarks: LandmarksMessage | null; // for overlay interpolation
  viewState: ViewStateMessage | null;
  prompt: PromptMessage | null;
  recording: RecordingMessage | null;
  send: (event: ClientEvent) => void;

  setStatus: (status: ConnectionStatus) => void;
  setSend: (send: (event: ClientEvent) => void) => void;
  pushLandmarks: (msg: LandmarksMessage) => void;
  setViewState: (msg: ViewStateMessage) => void;
  setPrompt: (msg: PromptMessage | null) => void;
  setRecording: (msg: RecordingMessage | null) => void;
}

export const useSignBridgeStore = create<SignBridgeState>((set, get) => ({
  status: "connecting",
  mode: "communicator",
  landmarks: null,
  prevLandmarks: null,
  viewState: null,
  prompt: null,
  recording: null,
  send: () => {
    // replaced once the socket connects; a no-op keeps early callers safe
  },

  setStatus: (status) => set({ status }),
  setSend: (send) => set({ send }),
  pushLandmarks: (msg) => set({ landmarks: msg, prevLandmarks: get().landmarks }),
  setViewState: (msg) => set({ viewState: msg, mode: msg.mode }),
  setPrompt: (msg) => set({ prompt: msg }),
  setRecording: (msg) => set({ recording: msg }),
}));
