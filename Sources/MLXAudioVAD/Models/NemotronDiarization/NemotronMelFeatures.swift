import Foundation
import MLX
import MLXNN

/// Log-mel guard, matching upstream's `2 ** -24` (not a rounder `1e-10`):
/// `log(power @ fb.T + 2**-24)`. Using a different epsilon shifts every
/// near-silent bin by a tiny but nonzero amount — invisible on a spot check,
/// visible on a parity diff.
private let logGuard: Float = pow(2.0, -24)

/// 128-bin log-mel frontend for Nemotron 3 Diarization, at 16 kHz.
///
/// Unlike a from-scratch mel implementation, `fb` and `window` are the
/// checkpoint's own buffers (`preprocessor.fb` [1, 128, 257] F32,
/// `preprocessor.window` [400] F32), loaded like any other weight. Upstream's
/// `MelFeatures` docstring calls this out explicitly ("retaining the
/// checkpoint's filter buffers") — computing a Slaney/HTK filterbank instead
/// would be a plausible-looking approximation, not the tensor the encoder was
/// trained against.
///
/// `callAsFunction` is a direct port of upstream's `__call__`
/// (`nemotron_diarization.py` lines 24-71), not a build atop this fork's
/// generic `MLXAudioCore.stft`. That utility unconditionally re-pads
/// `nFFT/2` zeros onto *both* edges of whatever buffer it receives and sizes
/// its frame count from the padded buffer's length — neither matches
/// upstream, whose frame count is `total_samples / hop_length` and whose
/// framing gathers directly against a **global** sample position (`start`,
/// `sampleOffset`, `totalSamples` together let a chunk be featurised as
/// though it sat at its true position in an entire streaming session, while
/// the caller holds only a suffix of the audio). Re-padding per call is not
/// an approximation of that: it manufactures 256 synthetic zero samples at
/// every chunk's leading edge regardless of what real audio precedes it, so
/// interior chunks would never actually get continuous STFT context no
/// matter what a caller prepends. Positions are computed once, on the same
/// global clock the caller's `NemotronStreamingState.frameOffset` advances,
/// and preemphasis falls out of gathering `positions` and `positions - 1`
/// from that same clock rather than needing a carried-over scalar.
public class NemotronMelFeatures: Module {
    let config: NemotronProcessorConfig

    @ParameterInfo(key: "fb") var fb: MLXArray
    @ParameterInfo(key: "window") var window: MLXArray

    public init(_ config: NemotronProcessorConfig) {
        self.config = config
        // Placeholder shapes until the checkpoint's real buffers are loaded;
        // matches the fb/window shapes verified against the safetensors
        // header (`preprocessor.fb` [1, 128, 257], `preprocessor.window` [400]).
        self._fb.wrappedValue = MLXArray.zeros([1, config.featureSize, config.nFFT / 2 + 1])
        self._window.wrappedValue = MLXArray.zeros([config.winLength])
        super.init()
    }

    /// `audio` is 1-D float samples in [-1, 1] — the buffer this call was
    /// handed, which may be the whole recording or just a retained suffix.
    /// `start`/`count` select which global frames to compute; `sampleOffset`
    /// is `audio`'s own position on that same global sample clock, and
    /// `totalSamples` is the true (unpadded) recording length so trailing
    /// silence isn't counted as valid frames. Defaults reproduce a plain
    /// whole-buffer call.
    ///
    /// Returns `(1, featureSize, count)`, matching upstream's convention —
    /// `NemotronFeatureStacking` transposes this itself, so it is not done here.
    public func callAsFunction(
        _ audio: MLXArray,
        start: Int = 0,
        count: Int? = nil,
        sampleOffset: Int = 0,
        totalSamples: Int? = nil
    ) -> MLXArray {
        let totalSamples = totalSamples ?? audio.dim(0)
        let validFrames = totalSamples / config.hopLength
        let count = count ?? validFrames
        guard count > 0 else {
            return MLXArray.zeros([1, config.featureSize, 0])
        }

        // GLOBAL sample positions for every (frame, tap) pair. `sampleOffset`
        // is the only place this clock maps back onto `audio`'s own indices
        // — that decoupling is the whole mechanism, not an implementation
        // detail: it is what lets a streaming caller ask for specific global
        // frames while physically holding only a suffix of the recording.
        let frameIds = MLXArray(start..<(start + count))
        let positions = frameIds.expandedDimensions(axis: 1) * config.hopLength
            + MLXArray(0..<config.nFFT).expandedDimensions(axis: 0)
            - config.nFFT / 2

        let maxLocalIndex = max(0, audio.dim(0) - 1)

        // Out-of-range positions read as silence via the validity mask, not
        // via the clip: clip alone would read a real (wrong) sample at the
        // buffer's edge instead of the zero upstream returns there.
        func gather(_ indices: MLXArray) -> MLXArray {
            let local = indices - sampleOffset
            let clipped = MLX.clip(local, min: 0, max: maxLocalIndex)
            let values = audio.take(clipped)
            let valid = (indices .>= 0) .&& (indices .< totalSamples)
            return MLX.which(valid, values, 0 as Float)
        }

        // Preemphasis on the gathered values, not on `audio` directly:
        // gathering `positions - 1` is what makes preemphasis continuous
        // across a chunk boundary. Position -1 for the recording's very
        // first sample gathers as 0 (correct: no sample precedes it), while
        // position (chunkStart - 1) for any later chunk gathers the true
        // previous sample from wherever the caller's buffer puts it —
        // carrying continuity by construction, with nothing to remember
        // between calls.
        var frames = gather(positions) - config.preemphasis * gather(positions - 1)
        let positionsValid = (positions .>= 0) .&& (positions .< totalSamples)
        frames = MLX.which(positionsValid, frames, 0 as Float)

        // The checkpoint's window is `winLength` (400) samples; centered
        // inside an `nFFT` (512) buffer with symmetric zero padding —
        // matching torch.stft's behavior for win_length < n_fft. Skipping
        // this pad would run a 400-point transform where the model expects
        // a 512-point one.
        let padTotal = config.nFFT - config.winLength
        let padLeft = padTotal / 2
        let padRight = padTotal - padLeft
        let paddedWindow: MLXArray = padTotal > 0
            ? MLX.concatenated([MLXArray.zeros([padLeft]), window.asType(.float32), MLXArray.zeros([padRight])])
            : window.asType(.float32)

        let spectrum = MLX.rfft(frames * paddedWindow, axis: -1)
        let power = MLX.abs(spectrum).square()
        // fb is (1, featureSize, nFreqs); drop the leading batch dim so the
        // matmul contracts over frequency bins, as upstream's `self.fb[0]` does.
        var features = MLX.log(MLX.matmul(power, fb[0].asType(.float32).transposed()) + logGuard)
        // Frames whose global id falls past the recording's real length
        // (i.e. in hop-length rounding slack) are zeroed, not left as
        // whatever the position mask happened to compute.
        let frameValid = frameIds.expandedDimensions(axis: 1) .< validFrames
        features = MLX.which(frameValid, features, 0 as Float)
        return features.transposed().contiguous().expandedDimensions(axis: 0)
    }
}
