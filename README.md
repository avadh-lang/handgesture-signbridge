# SignBridge

A real-time sign-language studio that runs on one webcam, fully offline.

Built on [MediaPipe](https://developers.google.com/mediapipe) Holistic (both
hands **+** a body pose) and OpenCV. Three modes over one tracking loop:

| Mode | What it does |
|---|---|
| **1 Communicator** | Fingerspell or sign → live text. Speak it aloud (`say`) or type it straight into the focused app (Slack, Notes, a search bar). `v` flips to the **reverse channel**: type a phrase, hold up big ASL-gloss cards for the other person. |
| **2 Coach** | Practise a target sign or letter, get a live accuracy score, and spaced-repetition drills (Leitner boxes) that persist across sessions. Every attempt is logged to `data/practice_log.csv`. |
| **3 Control** | A sign you invented fires a macro — open an app, send a keystroke, type a snippet, run a shell command. |
| **4 Pointer** | Hands-free mouse: point to move the cursor, pinch to click (deep pinch = right-click, hold = drag), two fingers to scroll. So you can navigate *and* communicate with no keyboard, mouse or voice. Needs `pyautogui` + Accessibility permission. |

### The part that isn't available anywhere else

**Teach it your own signs in ~10 seconds.** Demonstrate a sign three times and
it's in your personal dictionary — a moving, two-handed, body-relative sign, or
a still handshape, it doesn't matter. No training, no dataset, no network:

* each demo becomes a fixed-length **feature clip** — de-rotated, palm-scaled
  handshape descriptors + where the hands sit *relative to your torso* + how
  they move;
* recognition is nearest-template with [dynamic time warping](dtw.py), so a sign
  done a little faster or slower still matches;
* a rejection rule means an unknown gesture is ignored, not mis-fired.

```bash
python learn_sign.py "pizza"                  # 3 demos, saved to signs/
python learn_sign.py "mom" --reps 4
python learn_sign.py "flat spin" --bind app:Safari    # also bind it for Control
python learn_sign.py --list
python learn_sign.py --check "pizza"          # live distance readout for tuning
```

Your custom vocabulary works in every mode: whole-word signs drop into the
Communicator sentence, become drill targets in Coach, and trigger macros in
Control.

## Setup

MediaPipe has **no working macOS wheel for Python 3.13/3.14**, so this project
uses **Python 3.12**.

```bash
cd "handgesture project"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

First run needs **Camera** permission for your terminal. Typing into other apps
(`t` in Communicator, Control macros) additionally needs **Accessibility**
permission — both under System Settings → Privacy & Security.

## Run

```bash
.venv/bin/python app.py                    # Communicator, Holistic tracker
.venv/bin/python app.py --mode coach
.venv/bin/python app.py --mode control
.venv/bin/python app.py --mode pointer      # hands-free mouse
.venv/bin/python app.py --tracker hands    # lighter; no body reference
.venv/bin/python app.py --camera 1 --no-flip
.venv/bin/python app.py --swap-hands       # if left/right come through mirrored
```

**Keys**

| Scope | Key | Action |
|---|---|---|
| global | `Tab` / `` ` `` | next / previous mode |
| global | `s` · `h` · `q`/`Esc` | snapshot · toggle help · quit |
| while recording a sign | `Esc` | cancel |
| Communicator | `space` · `⌫` | commit word · delete |
| Communicator | `1` `2` `3` | accept the numbered word suggestion |
| Communicator | `Enter` | speak the sentence, then clear |
| Communicator | `t` · `v` · `r` | type-into-app on/off · reverse channel · teach a word |
| Coach | `space` · `r` · `n` · `a` | submit attempt · retry · skip · add a drill |
| Control | `l` · `e` · `u` | learn + bind a command · reload `commands.json` · unbind last |
| Pointer | — | drop your hand out of frame to release the cursor |

## How fingerspelling works

`fingerspell.py` classifies the ASL A–Z from a single hand geometrically. It's
solid on the well-separated letters and less sure on the genuine look-alikes
(A/S/T/M/N are all near-fists); **J** and **Z** are air-drawn and handled by the
motion layer. For rock-solid fingerspelling, record your own A–Z into the bank —
the bank is always checked first.

`glosser.py` (reverse channel) is a fast gloss, not a grammatical ASL
translator: it drops articles and the copula, points pronouns, and marks unknown
names for fingerspelling.

## No webcam? 

```bash
.venv/bin/python demo_image.py path/to/hand.jpg      # still-image gesture + letter
.venv/bin/python selftest.py                         # 77 camera-free checks
```

## Files

| File | Purpose |
|---|---|
| `skeleton.py` | Landmark normalisation, body-relative location, per-frame & sequence features |
| `dtw.py` | Banded dynamic time warping |
| `signbank.py` | Personal sign dictionary: record → match → reject |
| `live.py` | Rolling-window sign matcher with debounce |
| `fingerspell.py` | Rule-based ASL A–Z from one hand |
| `predict.py` + `data/words_en.txt` | Offline prefix word prediction |
| `speller.py` | Fingerspelling → word → sentence state machine |
| `glosser.py` | English → ASL-gloss cards |
| `leitner.py` · `sessionlog.py` | Spaced-repetition scheduler · practice CSV log |
| `tracking.py` | Unified MediaPipe Holistic / Hands tracker → `skeleton.Pose` |
| `filters.py` | One-Euro filter + hysteresis debounce |
| `pointer.py` | Hands-free mouse: move / click / drag / scroll |
| `hud.py` · `textinput.py` | OpenCV overlays · in-window text field |
| `recorder.py` | Non-blocking "demonstrate a sign N times" capture |
| `modes.py` | Communicator / Coach / Control / Pointer |
| `commands.py` + `data/commands.json` | Sign → app/key/text/shell bindings |
| `app.py` · `learn_sign.py` | Mode host · teach/list/delete/check CLI |
| `gestures.py` · `hand_tracker.py` · `actions.py` · `demo_image.py` | The original single-frame gesture recogniser (still used by `demo_image.py`) |

## Credits

The **Pointer** mode — One-Euro cursor smoothing, active-zone → screen mapping,
the pinch click/drag state machine, two-finger scroll — and the CSV practice log
are adapted from
[Prathamesh-2803/Real-time-hand-gesture-control-system](https://github.com/Prathamesh-2803/Real-time-hand-gesture-control-system),
rebuilt on top of SignBridge's `skeleton.Pose` so they share one tracker.
