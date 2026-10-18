# Competitor / Related-Work Landscape

Research conducted 2026-10-18 (web search). Accuracy claims below are self-reported
by each project's own authors/vendors unless noted otherwise — treat them as
directional, not independently verified.

## Camera-based sign *recognition* (input direction — closest competitors)

| Project | What it does | Offline? | Personalization | Vocabulary scope | Hardware |
|---|---|---|---|---|---|
| **SignAll** (Budapest) | Enterprise ASL↔English translation | No (cloud-dependent product) | No | Fixed | Camera array (multi-camera rig) |
| **SignSpeak** | Flutter app: on-device MediaPipe + on-device LLM, ASL + custom hand signs → full English sentences | Yes | Yes — editable vocabulary, no retraining | Open-ended via LLM sentence construction | Phone camera |
| **Signspell** | Open-source webcam **ASL alphabet only** (LSTM) | Yes | No | Fingerspelling only (A–Z) | Any webcam |
| Course-project repos (Random Forest / PyTorch, various GitHub) | MediaPipe landmarks → classifier, config-file vocabulary | Yes (student projects) | Partial (retrain to add signs) | Small, fixed per project | Webcam |
| CVPR 2026 workshop paper (landmark-based) | Isolated-sign recognition, <10M params, ~100MB | Yes (research) | No | Fixed benchmark set | Webcam |
| **SpellRing** (Cornell) | Continuous fingerspelling via acoustic sensing | Yes | No | Fingerspelling only | Custom hardware ring |

## Sign *generation* (output direction — different problem, adjacent space)

| Project | What it does |
|---|---|
| **Hand Talk** (acquired by Sorenson, Jan 2025) | Text/speech → 3D avatar that *signs out* (Libras/ASL). Opposite direction from recognition — doesn't read a user's signs at all. |

## Where SignBridge sits

No project found combines all of the following in one tool:

1. **Zero-training personal sign teaching** — record a sign 3× and it's usable immediately (DTW nearest-template matching, no model weights, no retraining, no dataset). SignSpeak is the closest in spirit (editable vocabulary, no retraining) but still depends on an on-device LLM for sentence construction; SignBridge's personal-vocabulary path needs no model at all.
2. **Fingerspelling + whole-word signs in the same tool.** Signspell and the fingerspelling-focused research projects (SpellRing, course projects) only do the alphabet. SignBridge does both, with the sign-bank checked before the rule-based fingerspelling fallback.
3. **Four integrated modes on one tracking loop** — Communicator (text/speech), Coach (spaced-repetition practice with persistent Leitner state), Control (sign → OS macro), Pointer (hands-free mouse). No competitor found bundles recognition with practice/coaching *and* OS-level automation *and* pointer control.
4. **100% offline, no cloud, no LLM.** SignAll is cloud-dependent; SignSpeak needs an on-device LLM. SignBridge's core recognition path (DTW + rule-based fingerspelling) needs neither — a meaningful offline/privacy claim for the paper.
5. **Commodity hardware.** A single regular webcam — no camera array (SignAll) and no custom hardware (SpellRing's ring).

## Known gap competitors don't share (worth stating honestly in the paper)

SignBridge's rule-based fingerspelling classifier (`fingerspell.py`) is tuned to one hand's proportions and doesn't generalize across hand shapes (see `docs/ARCHITECTURE.md` and the Phase 4 ML-classifier plan in `docs/REWORK_ROADMAP.md`). This is a real, specific, fixable limitation — and addressing it with a trained, person-held-out-evaluated classifier is itself a legitimate paper contribution, not just an engineering cleanup.

## Sources

- [SignAll — TechCrunch](https://techcrunch.com/?p=1587362)
- [SignAll — AI for Sign Language (Veritone)](https://veritone.com/blog/ai-translation-for-sign-language)
- [SignAll — CORDIS EU project record](https://cordis.europa.eu/article/id/411590-first-system-to-automatically-translate-sign-language)
- [Real-Time Sign Language Translation with MediaPipe, Flutter, and Gemini Nano (SignSpeak) — DEV Community](https://dev.to/noor_y/real-time-sign-language-translation-with-mediapipe-flutter-and-gemini-nano-21e9)
- [Signspell — Product Hunt](https://www.producthunt.com/products/signspell)
- [Isolated Sign Language Recognition via MediaPipe Landmarks — CVPR 2026 Workshop (MSLR)](https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Varanasi_Isolated_Sign_Language_Recognition_via_MediaPipe_Landmarks_A_Case_Study_CVPRW_2026_paper.html)
- [SpellRing: Recognizing Continuous Fingerspelling in ASL Using a Ring (arXiv)](https://arxiv.org/html/2502.10830v1)
- [AI ring tracks spelled words in American Sign Language — Cornell Chronicle](https://news.cornell.edu/stories/2025/03/ai-ring-tracks-spelled-words-american-sign-language)
- [Hand Talk — World Summit Award](https://wsa-global.org/winner/hand-talk/)
- [Hand Talk acquisition — innovatorsunder35.com](https://www.innovatorsunder35.com/the-list/ronaldo-tenorio/)
- [SignLang-AI — GitHub](https://github.com/ashish-4169/SignLang-AI)
- [asl-fingerspelling-recognition — GitHub](https://github.com/Ticasslo/asl-fingerspelling-recognition)
