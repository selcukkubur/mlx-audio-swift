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
`mlx-community/Nemotron-3-Diarization` checkpoint on a fixed-seed (`seed=0`)
~4-second 16 kHz Gaussian noise waveform (`NemotronParity`, this fork's
`Sources/Tools/NemotronParity`):

| Stage | Cosine similarity | Threshold | Result |
|---|---|---|---|
| mel (log-mel frontend) | `0.9999999990` | ≥ 0.9999 | pass |
| encoder (31-layer rotary transformer output) | `0.9999488035` | ≥ 0.999 | pass |
| probs (single-window speaker probabilities) | `0.9999749655` | ≥ 0.999 | pass |
| streaming (chunked `feed`, AOSC/FIFO exercised) | `0.9827825139` | *(reported, not gated — see below)* | measured |

The encoder and probs stages were run with the *reference's own* mel output
as input (not this port's mel output), isolating each stage's own
correctness from any small upstream drift, per the brief's diagnostic
guidance (`encoder` near zero with `mel` fine would point at the rotary
convention; a `mel` mismatch would point at the frontend or the
`fb`/`window` buffers — neither was observed).

### Streaming: measured, not gated, and honestly lower

The brief's own three thresholds (mel/encoder/probs) are all single-window
figures and all pass comfortably. The streaming figure was added on top of
those — the brief itself does not require it — specifically because a
single-window comparison never exercises the speaker cache or FIFO at all,
and those accumulate error silently rather than announcing it.

The measured streaming cosine, `0.9827825139`, is real and is noticeably
lower than the single-window figures. Diagnosis, checked directly rather
than assumed:

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
- **What it is:** the input is synthetic Gaussian noise with no real
  speech, so every stage — offline or streaming — correctly predicts
  near-silence throughout (probabilities in the `1e-5`–`1e-14` range).
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
  stage.

This is reported as a genuine finding, not smoothed over: streaming parity
on real speech, where activations are not saturating a sigmoid's flattest
region, would be a stronger and more representative test than this
synthetic-noise waveform provides. A follow-up parity run against a real
speech sample is worth doing before treating the streaming path as fully
proven, even though nothing examined here points at an implementation bug.

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

// Streaming:
var state = model.initStreamingState(preset: .low)
let segments = model.feed(chunkOfSamples, state: &state, final: false)
// ... more chunks ...
let finalSegments = model.feed([], state: &state, final: true)
```

`initStreamingState(preset:)` combines what upstream splits across two
calls (`init_streaming_state()` + `set_streaming_config(preset)`) into one:
selecting a preset (`.offline`, `.low`, `.veryLow`, `.ultraLow`) and getting
a fresh, empty state are always wanted together.
