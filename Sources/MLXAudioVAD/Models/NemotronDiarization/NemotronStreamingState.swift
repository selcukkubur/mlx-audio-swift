import Foundation
@preconcurrency import MLX

/// Everything carried between chunks.
///
/// The four model arrays mirror upstream's `StreamingState` dataclass
/// exactly (`nemotron_diarization.py` lines 184-198): `spkcache`,
/// `spkcache_preds`, `fifo`, `fifo_preds`, each starting at sequence length
/// **zero**, not pre-sized to `spkcacheLen`. `init_streaming_state` (lines
/// 228-236) builds all four with `mx.zeros((1, 0, ...))` — the caches grow as
/// audio arrives and are trimmed back down by the AOSC/FIFO update logic in a
/// later task, they are never pre-allocated at capacity. This matches the
/// existing Sortformer port in this fork (`Sortformer.initStreamingState()`),
/// which starts its own cache/fifo arrays empty for the same reason.
///
/// Upstream also sets `dtype=self.dtype` on `spkcache`/`fifo` specifically
/// (not on the two prediction arrays) — the checkpoint's encoder weights are
/// BF16, so those two caches are created BF16 here too. Sortformer's own
/// `initStreamingState()` defaults its caches to Float32 instead; that gap
/// is pre-existing there and out of scope for this port, but it is not
/// repeated here.
///
/// This is the part that fails silently and cumulatively: a speaker cache
/// updated wrongly does not throw, it drifts, and the labels degrade over
/// minutes in a way a short clip will not reveal. It is a value type so a
/// caller cannot mutate a session's state by accident.
public struct NemotronStreamingState: Sendable {
    /// Arrival-Order Speaker Cache: accumulated per-speaker evidence,
    /// `(1, frames, dModel)`, BF16. Starts empty and grows/compacts as
    /// chunks are accepted.
    public var spkcache: MLXArray
    /// Per-frame speaker probabilities for `spkcache`, `(1, frames, numSpeakers)`, Float32.
    public var spkcachePreds: MLXArray
    /// Recent frames not yet folded into the cache, `(1, frames, dModel)`, BF16.
    public var fifo: MLXArray
    /// Per-frame speaker probabilities for `fifo`, `(1, frames, numSpeakers)`, Float32.
    public var fifoPreds: MLXArray

    /// PCM retained so the next chunk's first STFT window has real left
    /// context instead of zero padding. Only the very first chunk of a
    /// session should ever see a zero-padded left edge.
    public var pcmTail: [Float]
    /// Encoder frames consumed so far — the clock all emitted spans are dated
    /// against, and the same clock `NemotronMelFeatures`'s `sampleOffset`
    /// keeps the frontend's global sample positions aligned with. Native
    /// 10 ms frames, before output downsampling (mirrors upstream's
    /// `frames_processed`).
    public var frameOffset: Int
    /// Frames already emitted as finalized, so a span is not reported twice.
    public var emittedUpTo: Int

    /// `dtype` is the encoder's own weight dtype (query it from the loaded
    /// model, e.g. the `pre_encode.proj` weight, as upstream's `self.dtype`
    /// property does) — not hardcoded here, since a caller loading a
    /// differently-quantized checkpoint should get caches that match it.
    /// The published checkpoint is BF16 throughout, hence the default.
    public init(
        config: NemotronDiarizationConfig,
        dtype: DType = .bfloat16
    ) {
        let dModel = config.encoderConfig.dModel
        let numSpeakers = config.numSpeakers
        self.spkcache = MLXArray.zeros([1, 0, dModel], dtype: dtype)
        self.spkcachePreds = MLXArray.zeros([1, 0, numSpeakers])
        self.fifo = MLXArray.zeros([1, 0, dModel], dtype: dtype)
        self.fifoPreds = MLXArray.zeros([1, 0, numSpeakers])
        self.pcmTail = []
        self.frameOffset = 0
        self.emittedUpTo = 0
    }
}
