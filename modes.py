"""The four modes SignBridge runs, over one shared tracking loop.

    1  Communicator  - sign / fingerspell -> text; speak it or type it into the
                       focused app.  `v` flips to the reverse channel: type a
                       phrase, hold up big gloss cards for the other person.
    2  Coach         - practise a target sign / letter, get a live accuracy
                       score, spaced-repetition drills that persist.
    3  Control       - a custom sign fires an app / keystroke / shell / text
                       macro.
    4  Pointer       - hands-free mouse: point to move, pinch to click/drag,
                       two fingers to scroll.

Each mode separates *deciding* from *drawing*:

    mode.compute(ctx) -> dict      pure-ish state update, no frame drawing.
                                    This is what a non-OpenCV frontend (e.g.
                                    the planned web UI) consumes directly.
    mode.update(ctx)               calls compute() then draws the result via
                                    hud.* onto ctx.frame - this is what the
                                    OpenCV CLI (app.py) calls every frame.

Keyboard input works the same way: every mode declares a ``KEYMAP`` from
OpenCV keycodes to named actions, and implements ``on_action(action, ctx,
**kw)``. ``on_key(code, ctx)`` (what app.py's cv2 loop calls) is a thin
shim on the base ``Mode`` class that looks the code up in ``KEYMAP`` and
calls ``on_action`` - so the keyboard path and any future named-action path
(e.g. a web frontend's `{"action": "backspace"}` message) can never drift
apart; there is exactly one action vocabulary, not two.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import hud
import output_sink
from commands import load_commands, parse_action, run_action, save_commands
from fingerspell import ALPHABET, classify_letter_best
from glosser import gloss
from leitner import Leitner
from live import LiveSigner
from pointer import PointerControl
from recorder import SignRecorder
from sessionlog import SessionLog
from signbank import REJECT_DISTANCE, SignBank, slugify
from skeleton import Pose
from speller import Speller
from tracking import dominant_hand


@dataclass
class Ctx:
    frame: np.ndarray
    pose: Pose
    now: float
    bank: SignBank
    live: LiveSigner
    recorder: SignRecorder
    busy: bool = False
    _prompt_request: tuple | None = None

    @property
    def paused(self) -> bool:
        return self.busy or self.recorder.active

    def request_prompt(self, kind: str, label: str, initial: str = "") -> None:
        self._prompt_request = (kind, label, initial)


class Mode:
    title = "?"
    keyhint = ""
    # OpenCV keycode -> action name, or (action name, kwargs dict).
    KEYMAP: dict = {}

    def activate(self, ctx: Ctx) -> None: ...
    def update(self, ctx: Ctx) -> None: ...
    def compute(self, ctx: Ctx) -> dict: return {}

    def on_action(self, action: str, ctx: Ctx, **kw) -> bool:
        return False

    def on_key(self, code: int, ctx: Ctx) -> bool:
        spec = self.KEYMAP.get(code)
        if spec is None:
            return False
        action, kw = spec if isinstance(spec, tuple) else (spec, {})
        return self.on_action(action, ctx, **kw)

    def on_prompt(self, kind: str, value: str, ctx: Ctx) -> None: ...
    def on_recorded(self, result: dict, ctx: Ctx) -> None: ...


# ---------------------------------------------------------------- Communicator
class Communicator(Mode):
    title = "Communicator"
    keyhint = "space | bksp | 1/2/3 pick | enter speak | t type->app | v reverse | r teach word"

    KEYMAP = {
        ord('v'): "communicator.toggle_reverse",
        ord(' '): "communicator.space",
        8: "backspace", 127: "backspace",
        ord('1'): ("communicator.accept_suggestion", {"index": 0}),
        ord('2'): ("communicator.accept_suggestion", {"index": 1}),
        ord('3'): ("communicator.accept_suggestion", {"index": 2}),
        13: "communicator.speak", 10: "communicator.speak",
        ord('t'): "communicator.toggle_type_out",
        ord('r'): "communicator.teach_word",
    }

    def __init__(self) -> None:
        self.speller = Speller()
        self.type_out = False
        self._typed = 0
        self._teach_name: str | None = None
        self.reverse = False
        self.reverse_cards: list = []
        self.reverse_since = 0.0
        self.flash = ""

    # ---- forward: sign -> text
    def compute(self, ctx: Ctx) -> dict:
        if self.reverse:
            return self._compute_reverse(ctx)

        if not ctx.paused:
            m = ctx.live.poll(ctx.bank, ctx.now)
            if m is not None:
                self.speller.sentence.append(m.name)
                self.speller._refresh()
                if self.type_out:
                    output_sink.type_text(m.name + " ")
                self.flash = f'sign: {m.name}'

        letter, conf = None, 0.0
        hand, label = dominant_hand(ctx.pose)
        if hand is not None and not ctx.paused:
            res = classify_letter_best(hand, label)
            if not res.motion:
                letter, conf = res.letter, res.confidence
        committed = self.speller.feed_letter(letter, conf, ctx.now)
        if committed and self.type_out:
            self._sync_typing()

        hold_fraction = self.speller.hold_fraction(ctx.now) if letter else 0.0
        return {
            "mode": "communicator",
            "reverse": False,
            "type_out": self.type_out,
            "word": self.speller.word,
            "sentence_text": self.speller.text(),
            "suggestions": list(self.speller.suggestions),
            "detecting_letter": letter,
            "detecting_conf": conf,
            "hold_fraction": hold_fraction,
            "flash": self.flash,
            "bank_count": len(ctx.bank),
        }

    def _compute_reverse(self, ctx: Ctx) -> dict:
        texts: list[str] = []
        active = -1
        if self.reverse_cards:
            n = len(self.reverse_cards)
            step = max(0.0, ctx.now - self.reverse_since)
            active = min(int(step / 1.1), n - 1)
            for c in self.reverse_cards:
                texts.append("-".join(c.letters) if c.kind == "fingerspell" else c.text)
        return {
            "mode": "communicator",
            "reverse": True,
            "reverse_cards": texts,
            "reverse_active": active,
        }

    def update(self, ctx: Ctx) -> None:
        state = self.compute(ctx)
        f = ctx.frame
        if state["reverse"]:
            hud.cards(f, state["reverse_cards"], state["reverse_active"])
            hud.top_bar(f, "1 Communicator - reverse channel", "v = back")
            return

        hud.top_bar(f, "1 Communicator", "TYPE->APP" if state["type_out"] else "")
        lines = []
        if state["detecting_letter"]:
            frac = state["hold_fraction"]
            lines.append((f"detecting  {state['detecting_letter']}  {state['detecting_conf']:.2f}  "
                          f"{'#' * int(frac * 10):<10}", hud.AMBER))
        lines.append((f"word:  {state['word'] or '_'}", hud.WHITE))
        if state["flash"]:
            lines.append((state["flash"], hud.GREEN))
        hud.panel(f, lines)
        hud.caption_bar(f, state["sentence_text"], state["suggestions"])

    def _sync_typing(self) -> None:
        w = self.speller.word
        if len(w) > self._typed:
            output_sink.type_text(w[self._typed:])
        self._typed = len(w)

    def on_action(self, action: str, ctx: Ctx, index: int | None = None, **kw) -> bool:
        s = self.speller
        if action == "communicator.toggle_reverse":
            self.reverse = not self.reverse
            if self.reverse:
                ctx.request_prompt("say", "say")
            return True
        if self.reverse:
            return True
        if action == "communicator.space":
            s.space()
            if self.type_out:
                output_sink.type_text(" ")
            self._typed = 0
            return True
        if action == "backspace":
            s.backspace()
            if self.type_out:
                output_sink.press_key("backspace")
            self._typed = len(s.word)
            return True
        if action == "communicator.accept_suggestion":
            if index is not None and index < len(s.suggestions):
                sugg = s.suggestions[index]
                if self.type_out:
                    output_sink.type_text(sugg[self._typed:] + " ")
                s.accept(index)
                self._typed = 0
            return True
        if action == "communicator.speak":
            text = s.pop_sentence()
            output_sink.speak(text)
            self.flash = f'spoke: "{text}"'
            self._typed = 0
            return True
        if action == "communicator.toggle_type_out":
            self.type_out = not self.type_out
            self._typed = len(s.word)
            return True
        if action == "communicator.teach_word":
            ctx.request_prompt("teach_word", "teach word (sign its name later)")
            return True
        return False

    def on_prompt(self, kind: str, value: str, ctx: Ctx) -> None:
        if kind == "say":
            if not value:                             # cancelled -> leave reverse
                self.reverse = False
                return
            self.reverse_cards = gloss(value)
            self.reverse_since = ctx.now
            output_sink.speak(value)
            if self.reverse:
                ctx.request_prompt("say", "say")      # keep the channel open
        elif kind == "teach_word":
            if value:
                self._teach_name = value
                ctx.recorder.start(value, reps=3)

    def on_recorded(self, result: dict, ctx: Ctx) -> None:
        if not result["clips"]:
            self.flash = "nothing recorded"
            return
        ctx.bank.add(result["name"], result["clips"],
                     kind=result["kind"], hands=result["hands"])
        ctx.live.reset()
        self.flash = f'learned "{result["name"]}" ({result["kind"]})'


# ----------------------------------------------------------------------- Coach
PASS_SCORE = 0.55


class Coach(Mode):
    title = "Coach"
    keyhint = "space submit | r retry | n skip | a add drill"

    KEYMAP = {
        13: "coach.submit", 10: "coach.submit", ord(' '): "coach.submit",
        ord('r'): "coach.retry", ord('n'): "coach.skip", ord('a'): "coach.add_drill",
    }

    def __init__(self) -> None:
        self.leitner = Leitner()
        self.log = SessionLog()
        self.target: tuple[str, str] | None = None
        self.best = 0.0
        self.msg = "get ready"
        self.session_ok = 0
        self.session_n = 0

    def _pool(self, ctx: Ctx) -> list[str]:
        keys = [f"letter:{c}" for c in ALPHABET if c not in "JZ"]
        keys += [f"sign:{m.name}" for m in ctx.bank.metas()]
        return keys

    def _pick(self, ctx: Ctx) -> None:
        key = self.leitner.next_due(self._pool(ctx), ctx.now)
        self.target = tuple(key.split(":", 1)) if key else ("letter", "A")
        self.best = 0.0

    def _score(self, ctx: Ctx) -> float:
        kind, name = self.target
        if kind == "letter":
            hand, label = dominant_hand(ctx.pose)
            if hand is None:
                return 0.0
            r = classify_letter_best(hand, label)
            if r.letter != name:
                return 0.0
            return max(r.confidence, 0.3)
        clip = ctx.live.recent_clip()
        if clip is None:
            return 0.0
        dists = dict(ctx.bank.distances(clip))
        d = dists.get(slugify(name))
        return 0.0 if d is None else max(0.0, 1.0 - d / REJECT_DISTANCE)

    def compute(self, ctx: Ctx) -> dict:
        if self.target is None:
            self._pick(ctx)
        self.best = max(self.best * 0.97, self._score(ctx))
        kind, name = self.target
        return {
            "mode": "coach",
            "target_kind": kind,
            "target_name": name,
            "score": self.best,
            "pass": self.best >= PASS_SCORE,
            "msg": self.msg,
            "session_ok": self.session_ok,
            "session_n": self.session_n,
            "box_counts": self.leitner.box_counts(),
        }

    def update(self, ctx: Ctx) -> None:
        state = self.compute(ctx)
        f = ctx.frame
        hud.top_bar(f, "2 Coach", f"{state['session_ok']}/{state['session_n']} this session")
        colour = hud.GREEN if state["pass"] else hud.AMBER
        hud.panel(f, [
            (f"sign this:  {state['target_name']}" +
             ("  (letter)" if state["target_kind"] == "letter" else "  (sign)"), hud.WHITE),
            (f"match  [{'#' * int(state['score'] * 20):<20}] {state['score']:.2f}", colour),
            (state["msg"], hud.GREY),
        ])
        boxes = state["box_counts"]
        hud.text(f, "boxes " + " ".join(f"{i}:{n}" for i, n in enumerate(boxes)),
                 (16, f.shape[0] - 24), 0.55, hud.GREY)

    def on_action(self, action: str, ctx: Ctx, **kw) -> bool:
        if self.target is None:
            return False
        key = ":".join(self.target)
        if action == "coach.submit":
            ok = self.best >= PASS_SCORE
            box = self.leitner.grade(key, ok, ctx.now)
            self.session_n += 1
            self.session_ok += int(ok)
            self.log.log("coach", target=key, detail=f"box={box}",
                         score=float(self.best), result="pass" if ok else "fail")
            self.msg = (f"nice - box {box}" if ok else f"not quite ({self.best:.2f}) - box 0")
            ctx.live.reset()
            self._pick(ctx)
            return True
        if action == "coach.retry":
            self.best = 0.0
            ctx.live.reset()
            return True
        if action == "coach.skip":
            ctx.live.reset()
            self._pick(ctx)
            self.msg = "skipped"
            return True
        if action == "coach.add_drill":
            ctx.request_prompt("drill", "add drill (single letter, or a sign name)")
            return True
        return False

    def on_prompt(self, kind: str, value: str, ctx: Ctx) -> None:
        if kind == "drill" and value:
            k = f"letter:{value.upper()}" if len(value) == 1 else f"sign:{value}"
            self.leitner.ensure([k])
            self.msg = f"added {k}"


# --------------------------------------------------------------------- Control
class Control(Mode):
    title = "Control"
    keyhint = "l learn+bind | e reload commands.json | u unbind last"

    KEYMAP = {
        ord('l'): "control.learn",
        ord('e'): "control.reload",
        ord('u'): "control.unbind_last",
    }

    def __init__(self) -> None:
        self.commands = load_commands()
        self.last = ""
        self.last_name = ""
        self._pending_name = ""
        self._pending_action: dict | None = None

    def compute(self, ctx: Ctx) -> dict:
        if not ctx.paused:
            m = ctx.live.poll(ctx.bank, ctx.now)
            if m is not None and m.name in self.commands:
                self.last = run_action(self.commands[m.name])
                self.last_name = m.name
        return {
            "mode": "control",
            "commands": [
                {"name": k, "action_type": v.get("type"), "action_value": v.get("value")}
                for k, v in self.commands.items()
            ],
            "last_fired": self.last,
            "last_name": self.last_name,
            "bound_count": len(self.commands),
        }

    def update(self, ctx: Ctx) -> None:
        state = self.compute(ctx)
        f = ctx.frame
        hud.top_bar(f, "3 Control", f"{state['bound_count']} bound")
        lines = [(f"{c['name']}  ->  {c['action_type']}: {c['action_value']}", hud.WHITE)
                 for c in state["commands"][:6]]
        if not lines:
            lines = [("no commands yet - press 'l' to learn one", hud.GREY)]
        if state["last_fired"]:
            lines.append((f"fired  {state['last_name']}: {state['last_fired']}", hud.GREEN))
        hud.panel(f, lines, scale=0.62, gap=32)

    def on_action(self, action: str, ctx: Ctx, **kw) -> bool:
        if action == "control.learn":
            ctx.request_prompt("ctl_name", "command sign name")
            return True
        if action == "control.reload":
            self.commands = load_commands()
            self.last = "reloaded"
            return True
        if action == "control.unbind_last" and self.last_name in self.commands:
            self.commands.pop(self.last_name)
            save_commands(self.commands)
            self.last = f"unbound {self.last_name}"
            return True
        return False

    def on_prompt(self, kind: str, value: str, ctx: Ctx) -> None:
        if kind == "ctl_name" and value:
            self._pending_name = value
            ctx.request_prompt("ctl_action",
                               "action  app:Safari | key:space | text:hi | shell:say hi")
        elif kind == "ctl_action":
            action = parse_action(value)
            if action is None:
                self.last = "bad action spec"
                return
            self._pending_action = action
            ctx.recorder.start(self._pending_name, reps=3)

    def on_recorded(self, result: dict, ctx: Ctx) -> None:
        if not result["clips"] or self._pending_action is None:
            self.last = "record failed"
            return
        ctx.bank.add(result["name"], result["clips"], kind=result["kind"],
                     hands=result["hands"], action=self._pending_action)
        self.commands[result["name"]] = self._pending_action
        save_commands(self.commands)
        ctx.live.reset()
        self.last = f'bound "{result["name"]}" -> {self._pending_action["type"]}'
        self._pending_action = None


# --------------------------------------------------------------------- Pointer
class Pointer(Mode):
    title = "Pointer"
    keyhint = "point = move | pinch = click | deep pinch = right | pinch+drag | 2 fingers = scroll"

    def __init__(self) -> None:
        self.pc = PointerControl()
        self._prev = 0.0

    def activate(self, ctx: Ctx) -> None:
        self.pc.reset()
        self._prev = 0.0

    def compute(self, ctx: Ctx) -> dict:
        if not self.pc.enabled:
            return {"mode": "pointer", "enabled": False, "status_lines": []}
        dt = (ctx.now - self._prev) if self._prev else None
        self._prev = ctx.now
        hand, _ = dominant_hand(ctx.pose)
        lines = self.pc.update(hand if not ctx.paused else None, ctx.now, dt)
        return {
            "mode": "pointer",
            "enabled": True,
            "status_lines": [{"text": s, "color": c} for s, c in lines],
        }

    def update(self, ctx: Ctx) -> None:
        state = self.compute(ctx)
        f = ctx.frame
        if not state["enabled"]:
            hud.top_bar(f, "4 Pointer", "unavailable")
            hud.panel(f, [("pip install pyautogui, then grant Accessibility", hud.RED)])
            return
        hud.top_bar(f, "4 Pointer", "hands-free mouse")
        hud.panel(f, [(l["text"], l["color"]) for l in state["status_lines"]], scale=0.62, gap=30)


def all_modes() -> list[Mode]:
    return [Communicator(), Coach(), Control(), Pointer()]
