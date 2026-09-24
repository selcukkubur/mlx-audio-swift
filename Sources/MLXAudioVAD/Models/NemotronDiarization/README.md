# Nemotron 3 Diarization

Swift port of NVIDIA's Nemotron 3 Diarization: a 31-layer rotary
FastConformer-replacement encoder feeding Sortformer's arrival-order speaker
cache (AOSC) / FIFO streaming speaker head. Predicts per-frame, per-speaker
activity probabilities (up to 8 speakers) for both offline and low-latency
streaming use.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/Nemotron-3-Diarization)

Ported against `mlx-audio`'s reference implementation at commit `03a4d99`
(`mlx_audio/vad/models/nemotron_diarization/nemotron_diarization.py`), whose
own docstring credits NVIDIA-NeMo/Speech (Apache-2.0) for the architecture
and AOSC semantics.

## Architecture

1. **`NemotronMelFeatures`** — 128-bin log-mel frontend using the
   checkpoint's own `fb`/`window` buffers, gathering frames on a global
   sample clock so a streaming caller can featurize a chunk as though it sat
   at its true position in an entire session.
2. **`NemotronEncoder`** — `NemotronFeatureStacking` (8x time-domain
   subsampling + projection) followed by 31 pre-norm transformer blocks with
   **rotary position embeddings**, then a final `LayerNorm`.
3. **`SpeakerModules`** — subclasses Sortformer's `SortformerModules` (shared
   in this fork with the Sortformer port: `encoder_proj`,
   `first_hidden_to_hidden`, `single_hidden_to_spks`, `hidden_to_spks`, and
   `SortformerModules.lengthToMask` all come from there, not duplicated
   here) and adds a subpixel-upsampling `Conv1d` that restores the encoder's
   8x-subsampled time resolution before scoring, plus a learnable silence
   embedding AOSC compression uses to fill disabled cache slots.
4. **`NemotronDiarizationModel`** — wires the three together, and reuses
   Sortformer's own AOSC compression (`SortformerModel.compressSpkcacheAosc`)
   for the streaming cache, rather than reimplementing it.

## The rotary convention — and why it is written down

`NemotronAttention` builds its `RoPE` with `traditional: false`. This is
upstream's `nn.RoPE(..., traditional=False, ...)` — the split-half
("rotate half") rotation convention, not the interleaved variant that
rotates consecutive pairs. mlx-swift's own `traditional` flag is named
after the *interleaved* form, so `false` here is the correct, checkpoint-
matching choice, not an oversight.

This is called out explicitly, in the source and here, because getting it
wrong is the single most dangerous failure mode in this port: choosing the
wrong RoPE convention does not crash and does not change a single tensor
shape. Every `@ModuleInfo` key still matches, every array still has the
right dimensions, and the model still runs at full speed — it just computes
attention over the wrong geometry, silently. Nothing downstream (a shape
check, a "does it run" smoke test) catches this; only comparing actual
numbers against an independent reference does. That is what the parity
figures below are for.

## Parity gate

Nothing downstream of this port proceeds until this gate passes. It is not
a formality: an earlier port in this same repository (Audio8) loaded, ran,
matched every shape, passed a spot check of weight values and an all-ones
convolution test, and transcribed nonsense — because both sides of that
comparison read the same wrong bytes. Cosine similarity against an
independent (Python) implementation of the same checkpoint is the check
that would have caught it on the first day.

Measured against `mlx-audio-ref` at commit `03a4d99`, running the published
`mlx-community/Nemotron-3-Diarization` checkpoint. Provenance is called out
per row below, because that distinction matters: this project's discipline
is measured, not claimed, and a number this README cannot be re-derived
from is a claim, not a measurement.

| Stage | Input | Cosine similarity | Threshold | Result | Reproducible how |
|---|---|---|---|---|---|
| mel (log-mel frontend) | 4 s noise | `0.9999999990` | ≥ 0.9999 | pass | `NemotronParity`, this repo |
| encoder (31-layer rotary transformer output) | 4 s noise | `0.9999488035` | ≥ 0.999 | pass | `NemotronParity`, this repo |
| probs (single-window speaker probabilities) | 4 s noise | `0.9999749655` | ≥ 0.999 | pass | `NemotronParity`, this repo |
| streaming (chunked `feed`) | 4 s noise | `0.9827825139` | *(reported, not gated)* | measured | `NemotronParity`, this repo |
| streaming (chunked `feed`) | 5.2 s real speech (`say`) | `0.9999999453` | *(reported, not gated)* | measured | `NemotronParity`, this repo |
| streaming (chunked `feed`) | 30 s real speech (`Tests/media/multi_speaker.wav`, this repo's own fixture) | `0.9996263850` | *(reported, not gated)* | measured | `NemotronParity`, this repo |
| streaming (chunked `feed`, **AOSC compression exercised**: `spkcache_compressed: true`) | 58.4 s real speech (`say`, two voices) | `0.9999995964` | *(reported, not gated)* | measured | `NemotronParity`, this repo |
| streaming (chunked `feed`, **AOSC compression exercised**) | 17.4 s / 53.4 s real speech (unspecified source) | `0.9999997031` / `0.9999777511` | *(reported, not gated)* | measured | external harness, run during review only — not reproducible from this repo |

The encoder and probs stages were run with the *reference's own* mel output
as input (not this port's mel output), isolating each stage's own
correctness from any small upstream drift, per the brief's diagnostic
guidance (`encoder` near zero with `mel` fine would point at the rotary
convention; a `mel` mismatch would point at the frontend or the
`fb`/`window` buffers — neither was observed).

The two `say`-sourced streaming-on-speech rows are the headline result for
the streaming path, and — unlike the rest of this README's numbers being
merely described as reproducible — actually are: see "Reproducing the
streaming-on-speech figures" below to regenerate them from nothing but this
repo and the `say` command already on any Mac. Both real-speech figures sit
in the same range as the single-window stages, and the 58.4 s clip is long
enough to overflow the FIFO and trigger AOSC compression mid-session
(confirmed via `state.spkcache_compressed: True` on the Python side); the
compressed cache's frozen predictions (the `spkcacheCompressed` freeze in
`NemotronStreamingState`, implemented per upstream but originally
unexercised by this port's own short test signal) were exercised and
matched the reference exactly.

The bottom row (17.4 s / 53.4 s) is kept because it is real evidence and
was independently confirmed, not fabricated, but it was measured with an
external harness during review, not with this repo's own tooling — so it
cannot be reproduced by running anything checked in here. It corroborates
the two `say`-sourced rows above it (same order of magnitude, same
AOSC-compression behavior at the longer duration) but is not itself the
reproducible claim; treat the `say`-sourced rows as the ones this README
stands behind.

### Reproducing the streaming-on-speech figures

All four `NemotronParity`-labeled streaming-on-speech rows above were
produced with this repo's own tooling end-to-end — `nemotron_parity.py`
(Python reference dump) feeding `NemotronParity` (this fork's Swift
harness, `Sources/Tools/NemotronParity`) — including the Swift side parsing
the WAV file itself via a real RIFF chunk walker (`loadWav16kMonoPCM16` in
`main.swift`), not a fixed 44-byte-header skip: `say -o file.wav` writes a
`JUNK`/`FLLR` filler chunk ahead of `data`, so PCM audio does not start at
byte 44 (it starts at byte 4096 for the short clip below) — a naive
fixed-offset reader would read the filler as audio. `Tests/media/`'s own
WAVs exercise a different real-world case: a `LIST`/`INFO` metadata chunk
(this project's files are SoX-processed) ahead of `data`.

```bash
# Simplest: this repo's own fixture, no audio generation needed.
PYTHONPATH=<mlx-audio-ref checkout> python3 nemotron_parity.py \
  <model-dir> <out-dir> Tests/media/multi_speaker.wav
swift run -c release NemotronParity <model-dir> <out-dir> \
  Tests/media/multi_speaker.wav <out-dir>/speech_streaming_probs.npy

# Or generate a clip with `say` (also exercises the JUNK/FLLR chunk-walking path):
say -o speech_test.wav --data-format=LEI16@16000 -v Samantha \
  "This is a short test of the streaming speaker diarization parity check for the Nemotron model."
PYTHONPATH=<mlx-audio-ref checkout> python3 nemotron_parity.py \
  <model-dir> <out-dir> speech_test.wav
swift run -c release NemotronParity <model-dir> <out-dir> \
  speech_test.wav <out-dir>/speech_streaming_probs.npy
```

The 58.4 s two-voice clip that exercises AOSC compression is the same
process, generated from two longer `say` calls (different voices,
`-v Samantha` / `-v Daniel`) concatenated at the PCM level with Python's
`wave` module into one plain-header WAV, then run through the same two
tool invocations above.

### Why the 4 s noise streaming figure is lower — and why it is still worth keeping

The `0.9827825139` row above uses the *same* fixed-seed Gaussian noise
waveform as the mel/encoder/probs rows, run through streaming `feed` instead
of a single window. It is not the headline streaming number — the real-
speech rows are — but it is kept because it is the evidence for why a
near-zero-activation input scores measurably lower here than on real
speech, which is itself a useful thing to know about this metric.

Diagnosis, checked directly rather than assumed, before the real-speech
figures above existed to confirm it:

- **Not a boundary/indexing bug.** The per-window central-frame extraction,
  FIFO growth (`chunk[:, :n]`), and `frames_processed` bookkeeping were
  checked frame-by-frame against the reference's own `streaming_step` for
  an isolated single window (fresh, empty cache) and matched the same
  window count, same FIFO length (9 encoder frames), and a comparable
  cosine (~0.9994) to what a from-scratch encoder pass shows — i.e. no
  extra error is introduced by the streaming bookkeeping itself.
- **Not a chunked-mel-gather bug.** This port's own chunked mel output was
  checked against its own full-buffer mel output on identical underlying
  samples and matched to `0.0` absolute difference (bit-exact) — the
  chunked/global-sample-clock gather in `NemotronMelFeatures` is internally
  self-consistent.
- **What it is:** the noise input has no real speech, so every stage —
  offline or streaming — correctly predicts near-silence throughout
  (reference probabilities max out around `2.3e-5`, most much smaller).
  Streaming re-runs the *entire* 31-layer encoder from scratch on every
  window (upstream's own design: `self.encoder(combined, lengths)` over the
  concatenated spkcache+FIFO+chunk, not an incremental/cached attention),
  so the small (~`1e-3` absolute, ordinary float32-precision-level) mel and
  BF16 encoder rounding differences already present in the single-window
  figures get freshly reintroduced and re-amplified at every one of the
  session's streaming windows, rather than being computed once. Sigmoid
  outputs this close to 0 are highly sensitive in *relative* terms to tiny
  upstream perturbations, which is exactly what a magnitude-sensitive
  metric like cosine similarity penalizes — even though the qualitative
  behavior (correctly predicting silence throughout) is preserved at every
  stage. On real speech, where activations are not saturating a sigmoid's
  flattest region, this effect essentially disappears — which is exactly
  what the 17.4 s and 53.4 s rows above show.

## Quick Start

```swift
import MLXAudioVAD

let model = try await NemotronDiarizationModel.fromPretrained(
    "mlx-community/Nemotron-3-Diarization"
)

// Offline: mel features in, per-frame speaker probabilities out.
let mel = model.preprocessor(audioSamples)          // (1, 128, T)
let lengths = MLXArray([Int32(mel.dim(2))])
let probs = model(mel, lengths: lengths)             // (1, ceil(T/8)*8, numSpeakers)

// Streaming: select a preset once, then start as many streams as you like
// under it. `setStreamingConfig` mutates the MODEL (matching upstream's
// `set_streaming_config`, kept as an explicit model-level call for exactly
// that reason) — one model instance cannot run two streams under two
// different presets concurrently.
model.setStreamingConfig(.low)
var state = model.initStreamingState()
let segments = model.feed(chunkOfSamples, state: &state, final: false)
// ... more chunks ...
let finalSegments = model.feed([], state: &state, final: true)
```

`setStreamingConfig(.offline | .low | .veryLow | .ultraLow)` picks one of
NVIDIA's latency presets (input-buffer latency, excluding compute and the
STFT window: offline=30.4s, low=1.04s, very_low=0.64s, ultra_low=0.32s).
`initStreamingState()` is genuinely per-stream — a fresh, empty state that
mutates nothing on the model — but it always runs under whichever preset
`setStreamingConfig` last selected on that model instance.
