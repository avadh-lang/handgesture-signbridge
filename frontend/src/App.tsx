import { AlphabetReference } from "./components/AlphabetReference";
import { CommunicatorView } from "./components/modes/CommunicatorView";
import { CoachView } from "./components/modes/CoachView";
import { ControlView } from "./components/modes/ControlView";
import { PointerView } from "./components/modes/PointerView";
import { PromptModal } from "./components/PromptModal";
import { RecordingBanner } from "./components/RecordingBanner";
import { TopBar } from "./components/TopBar";
import { VideoOverlay } from "./components/VideoOverlay";
import { useKeyboardActions } from "./lib/keyboard";
import { useSignBridgeConnection } from "./lib/ws";
import { useSignBridgeStore } from "./state/store";
import type {
  CoachState,
  CommunicatorState,
  ControlState,
  PointerState,
} from "./types";

function ModePanel() {
  const viewState = useSignBridgeStore((s) => s.viewState);
  if (!viewState) {
    return <p className="text-sm text-ink-tertiary">connecting to SignBridge…</p>;
  }
  switch (viewState.mode) {
    case "communicator":
      return <CommunicatorView state={viewState.payload as CommunicatorState} />;
    case "coach":
      return <CoachView state={viewState.payload as CoachState} />;
    case "control":
      return <ControlView state={viewState.payload as ControlState} />;
    case "pointer":
      return <PointerView state={viewState.payload as PointerState} />;
  }
}

function App() {
  useSignBridgeConnection();
  useKeyboardActions();

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-5 px-5 py-4">
      <TopBar />
      <main className="grid min-w-0 grid-cols-1 gap-5 lg:grid-cols-[3fr_2fr]">
        <div className="relative min-w-0">
          <VideoOverlay />
          <RecordingBanner />
        </div>
        <div className="min-w-0 rounded-xl border border-hairline bg-surface-1 p-5">
          <ModePanel />
        </div>
      </main>
      <AlphabetReference />
      <PromptModal />
    </div>
  );
}

export default App;
