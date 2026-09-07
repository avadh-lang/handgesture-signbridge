"""The three modes SignBridge runs, over one shared tracking loop.

    1  Communicator  - sign / fingerspell -> text; speak it or type it into the
                       focused app.  `v` flips to the reverse channel: type a
                       phrase, hold up big gloss cards for the other person.
    2  Coach         - practise a target sign / letter, get a live accuracy
                       score, spaced-repetition drills that persist.
    3  Control       - a custom sign fires an app / keystroke / shell / text
                       macro.

App calls ``mode.update(ctx)`` each frame and routes keys / prompts / finished
recordings back to the active mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import hud
import output_sink
from commands import load_commands, parse_action, run_action, save_commands
from fingerspell import ALPHABET, classify_letter
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

    def activate(self, ctx: Ctx) -> None: ...
    def update(self, ctx: Ctx) -> None: ...
    def on_key(self, code: int, ctx: Ctx) -> bool:
        return False
    def on_prompt(self, kind: str, value: str, ctx: Ctx) -> None: ...
    def on_recorded(self, result: dict, ctx: Ctx) -> None: ...


# ---------------------------------------------------------------- Communicator
class Communicator(Mode):
    title = "Communicator"
    keyhint = "space | bksp | 1/2/3 pick | enter speak | t type->app | v reverse | r teach word"

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
    def update(self, ctx: Ctx) -> None:
        f = ctx.frame
        if self.reverse:
            self._draw_reverse(ctx)
            return

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
            res = classify_letter(hand, label)
            if not res.motion:
                letter, conf = res.letter, res.confidence
        committed = self.speller.feed_letter(letter, conf, ctx.now)
        if committed and self.type_out:
            self._sync_typing()

        hud.top_bar(f, "1 Communicator", "TYPE->APP" if self.type_out else "")
        lines = []
        if letter:
            frac = self.speller.hold_fraction(ctx.now)
            lines.append((f"detecting  {letter}  {conf:.2f}  {'#' * int(frac * 10):<10}", hud.AMBER))
        lines.append((f"word:  {self.speller.word or '_'}", hud.WHITE))
        if self.flash:
            lines.append((self.flash, hud.GREEN))
        hud.panel(f, lines)
        hud.caption_bar(f, self.speller.text(), self.speller.suggestions)

    def _sync_typing(self) -> None:
        w = self.speller.word
        if len(w) > self._typed:
            output_sink.type_text(w[self._typed:])
        self._typed = len(w)

    # ---- reverse: text -> gloss cards
    def _draw_reverse(self, ctx: Ctx) -> None:
        if self.reverse_cards:
            n = len(self.reverse_cards)
            step = max(0.0, ctx.now - self.reverse_since)
            active = min(int(step / 1.1), n - 1)
            texts = []
            for c in self.reverse_cards:
                texts.append("-".join(c.letters) if c.kind == "fingerspell" else c.text)
            hud.cards(ctx.frame, texts, active)
        else:
            hud.cards(ctx.frame, [])
        hud.top_bar(ctx.frame, "1 Communicator - reverse channel", "v = back")

    def on_key(self, code: int, ctx: Ctx) -> bool:
        s = self.speller
        if code == ord('v'):
            self.reverse = not self.reverse
            if self.reverse:
                ctx.request_prompt("say", "say")
            return True
        if self.reverse:
            return True
        if code == ord(' '):
            s.space()
            if self.type_out:
                output_sink.type_text(" ")
            self._typed = 0
            return True
        if code in (8, 127):
            s.backspace()
            if self.type_out:
                output_sink.press_key(51)
            self._typed = len(s.word)
            return True
        if code in (ord('1'), ord('2'), ord('3')):
            i = code - ord('1')
            if i < len(s.suggestions):
                sugg = s.suggestions[i]
                if self.type_out:
                    output_sink.type_text(sugg[self._typed:] + " ")
                s.accept(i)
                self._typed = 0
            return True
        if code in (13, 10):
            text = s.pop_sentence()
            output_sink.speak(text)
            self.flash = f'spoke: "{text}"'
            self._typed = 0
            return True
        if code == ord('t'):
            self.type_out = not self.type_out
            self._typed = len(s.word)
            return True
        if code == ord('r'):
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
            r = classify_letter(hand, label)
            if r.letter != name:
                return 0.0
            return max(r.confidence, 0.3)
        clip = ctx.live.recent_clip()
        if clip is None:
            return 0.0
        dists = dict(ctx.bank.distances(clip))
        d = dists.get(slugify(name))
        return 0.0 if d is None else max(0.0, 1.0 - d / REJECT_DISTANCE)

    def update(self, ctx: Ctx) -> None:
        if self.target is None:
            self._pick(ctx)
        self.best = max(self.best * 0.97, self._score(ctx))

        kind, name = self.target
        f = ctx.frame
        hud.top_bar(f, "2 Coach", f"{self.session_ok}/{self.session_n} this session")
        colour = hud.GREEN if self.best >= PASS_SCORE else hud.AMBER
        hud.panel(f, [
            (f"sign this:  {name}" + ("  (letter)" if kind == "letter" else "  (sign)"), hud.WHITE),
            (f"match  [{'#' * int(self.best * 20):<20}] {self.best:.2f}", colour),
            (self.msg, hud.GREY),
        ])
        boxes = self.leitner.box_counts()
        hud.text(f, "boxes " + " ".join(f"{i}:{n}" for i, n in enumerate(boxes)),
                 (16, f.shape[0] - 24), 0.55, hud.GREY)

    def on_key(self, code: int, ctx: Ctx) -> bool:
        if self.target is None:
            return False
        key = ":".join(self.target)
        if code in (13, 10, ord(' ')):
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
        if code == ord('r'):
            self.best = 0.0
            ctx.live.reset()
            return True
        if code == ord('n'):
            ctx.live.reset()
            self._pick(ctx)
            self.msg = "skipped"
            return True
        if code == ord('a'):
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

    def __init__(self) -> None:
        self.commands = load_commands()
        self.last = ""
        self.last_name = ""
        self._pending_name = ""
        self._pending_action: dict | None = None

    def update(self, ctx: Ctx) -> None:
        if not ctx.paused:
            m = ctx.live.poll(ctx.bank, ctx.now)
            if m is not None and m.name in self.commands:
                self.last = run_action(self.commands[m.name])
                self.last_name = m.name

        f = ctx.frame
        hud.top_bar(f, "3 Control", f"{len(self.commands)} bound")
        lines = [(f"{k}  ->  {v.get('type')}: {v.get('value')}", hud.WHITE)
                 for k, v in list(self.commands.items())[:6]]
        if not lines:
            lines = [("no commands yet - press 'l' to learn one", hud.GREY)]
        if self.last:
            lines.append((f"fired  {self.last_name}: {self.last}", hud.GREEN))
        hud.panel(f, lines, scale=0.62, gap=32)

    def on_key(self, code: int, ctx: Ctx) -> bool:
        if code == ord('l'):
            ctx.request_prompt("ctl_name", "command sign name")
            return True
        if code == ord('e'):
            self.commands = load_commands()
            self.last = "reloaded"
            return True
        if code == ord('u') and self.last_name in self.commands:
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

    def update(self, ctx: Ctx) -> None:
        f = ctx.frame
        if not self.pc.enabled:
            hud.top_bar(f, "4 Pointer", "unavailable")
            hud.panel(f, [("pip install pyautogui, then grant Accessibility", hud.RED)])
            return
        dt = (ctx.now - self._prev) if self._prev else None
        self._prev = ctx.now
        hand, _ = dominant_hand(ctx.pose)
        lines = self.pc.update(hand if not ctx.paused else None, ctx.now, dt)
        hud.top_bar(f, "4 Pointer", "hands-free mouse")
        hud.panel(f, [(s, c) for s, c in lines], scale=0.62, gap=30)


def all_modes() -> list[Mode]:
    return [Communicator(), Coach(), Control(), Pointer()]
