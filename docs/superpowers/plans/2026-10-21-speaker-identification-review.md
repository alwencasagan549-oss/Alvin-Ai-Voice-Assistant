# Speaker Identification Review Fixes Implementation Plan

> **For agentic workers:** Use this plan as the task checklist; each change has a regression test before its implementation.

**Goal:** Make speaker identification optional at import time, correct its inference and audio path, and remove the review’s startup, latency, and personalization failures.

**Architecture:** Keep the speaker module importable without torch or SpeechBrain, loading those packages only when the model is initialized or run. Keep speaker inference off the turn path, collect only VAD-active audio in a bounded queue, and always rebuild the system message when personalizing a turn.

**Tech Stack:** Python 3.12, FastAPI, NumPy, pytest, SpeechBrain ECAPA-TDNN.

**Spec:** User-supplied “Speaker Identification Review — Not Ready”.

## Global Constraints

- Speaker identification remains disabled by default.
- Importing `app.main` with torch, torchaudio, and SpeechBrain unavailable must succeed.
- The turn handler must not await speaker embedding inference.
- Main runtime requirements must not install speaker-only dependencies.

## Review Focus

- Optional speaker dependencies absent at server import: test a subprocess import with those imports blocked.
- Invalid/missing model after partial initialization: test identification returns unknown without raising.
- VAD starts and ends around audio frames: test only active speech is buffered and the cap is maintained.
- An identification task is still running at the next turn: test final handling does not wait on it.
- Conversation history already contains a system message: test a contextualized prompt replaces it without mutating history.

---

### Task 1: Optional dependencies and valid embeddings

**Files:** `app/speaker.py`, `app/main.py`, `app/connection.py`, `tests/unit/test_speaker.py`, `tests/integration/test_speaker_identification.py`

**Interfaces:** `SpeakerService.identify(audio: np.ndarray) -> IdentificationResult` remains the public identification API.

- [ ] Add tests for importing `app.main` with speaker packages blocked, converting NumPy audio before `encode_batch`, and returning an embedding of shape `(192,)` when the real optional model is available.
- [ ] Run the focused tests and confirm they fail on the current import and embedding paths; the real model test may skip when extras are absent.
- [ ] Lazy-import torch in `extract_embedding`, pass contiguous float tensors and lengths under `torch.inference_mode()`, and keep service startup errors contained when speaker support is enabled.
- [ ] Make `identify` return an unknown result if initialization left `_model` unavailable.
- [ ] Re-run focused tests.

### Task 2: Non-blocking identification, clean audio, and prompt context

**Files:** `app/connection.py`, `app/llm.py`, `tests/unit/test_connection.py`, `tests/unit/test_llm.py`

**Interfaces:** `_identify_speaker()` remains the background worker; `build_messages()` accepts the existing optional `system_prompt` argument.

- [ ] Add failing tests for final handling with a blocked identification worker, VAD-gated bounded buffering, and a system prompt override over existing history.
- [ ] Run those tests and confirm the expected failures.
- [ ] Schedule speaker identification as an observed background task, buffer only after VAD marks speech active, use a deque and tracked sample count, reset both on consume, and replace the first system message in a fresh message list.
- [ ] Run focused tests.

### Task 3: Optional install, health response, and profile hygiene

**Files:** `requirements.txt`, `requirements-speaker.txt`, `README.md`, `AGENTS.md`, `.gitignore`, `app/main.py`, `tests/unit/test_main.py`

**Interfaces:** `requirements-speaker.txt` includes `requirements.txt` and installs torch, torchaudio, and SpeechBrain as opt-in dependencies.

- [ ] Add a failing health test that confirms the speaker similarity threshold is absent.
- [ ] Run the test and confirm it fails on the current response.
- [ ] Split speaker packages into `requirements-speaker.txt`, document the install command, omit the threshold from `/health`, and ignore the local pickle profile file.
- [ ] Run focused tests and inspect the final diff for changes outside this review.
