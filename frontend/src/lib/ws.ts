import { useEffect } from "react";
import { useSignBridgeStore } from "../state/store";
import type { ClientEvent, ServerMessage } from "../types";

const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 5000;

function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

/** Owns the one SignBridge WebSocket connection for the app's lifetime,
 * reconnecting with backoff on drop, and writing every message straight
 * into the zustand store. Mount once near the root (see App.tsx). */
export function useSignBridgeConnection(): void {
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectDelay = RECONNECT_MIN_MS;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    const { setStatus, setSend, pushLandmarks, setViewState, setPrompt, setRecording } =
      useSignBridgeStore.getState();

    function connect() {
      if (cancelled) return;
      setStatus("connecting");
      socket = new WebSocket(wsUrl());

      socket.onopen = () => {
        reconnectDelay = RECONNECT_MIN_MS;
        setStatus("open");
        setSend((event: ClientEvent) => {
          if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(event));
          }
        });
      };

      socket.onmessage = (ev) => {
        let msg: ServerMessage;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        switch (msg.type) {
          case "landmarks":
            pushLandmarks(msg);
            break;
          case "view_state":
            setViewState(msg);
            break;
          case "prompt":
            setPrompt(msg);
            break;
          case "recording":
            setRecording(msg);
            break;
        }
      };

      socket.onclose = () => {
        setStatus("closed");
        setSend(() => {});
        if (cancelled) return;
        reconnectTimer = setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 1.6, RECONNECT_MAX_MS);
      };

      socket.onerror = () => {
        socket?.close();
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);
}
