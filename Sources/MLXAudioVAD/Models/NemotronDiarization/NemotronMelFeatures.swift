import Foundation
import MLX
import MLXNN
import MLXAudioCore

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
/// Streaming-aware by design: `previousSample` lets preemphasis stay
/// continuous across chunk boundaries, and the caller (Diarizer.accept, a
/// later task) is expected to prepend the previous chunk's PCM tail onto
/// `audio` before calling this, so the STFT's left-edge window sees real
/// samples instead of zero padding except at the very start of a session.
/// Upstream is explicit that chunk boundaries must not reset preemphasis or
/// STFT context — resetting them puts a discontinuity at every edge, which a
/// diarizer reads as a speaker change. That is a plausible-looking wrong
/// answer, not a crash.
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

    /// `audio` is 1-D float samples in [-1, 1]. `previousSample` is the last
    /// sample preceding `audio` (the previous chunk's final sample, or 0 for
    /// the first chunk of a session), so preemphasis is continuous rather
    /// than restarting at 0 on every call.
    ///
    /// Returns `(1, featureSize, numFrames)`, matching upstream's convention.
    public func callAsFunction(
        _ audio: MLXArray, previousSample: Float = 0
    ) -> MLXArray {
        let shifted = MLX.concatenated(
            [MLXArray([previousSample]), audio[0..<(audio.dim(0) - 1)]], axis: 0
        )
        let emphasised = audio - config.preemphasis * shifted

        // The checkpoint's window is `winLength` (400) samples; STFT needs it
        // centered inside an `nFFT` (512) buffer, zero-padded symmetrically —
        // matching torch.stft's behavior for win_length < n_fft, which is
        // what the reference's own padding does.
        let padTotal = config.nFFT - config.winLength
        let padLeft = padTotal / 2
        let padRight = padTotal - padLeft
        let paddedWindow: MLXArray = padTotal > 0
            ? MLX.concatenated([MLXArray.zeros([padLeft]), window, MLXArray.zeros([padRight])])
            : window

        // `.constant` (zero) padding matches NeMo/Sortformer convention, not
        // Whisper's reflect padding.
        let spectrum = stft(
            audio: emphasised, window: paddedWindow,
            nFft: config.nFFT, hopLength: config.hopLength, padMode: .constant
        )
        let power = MLX.abs(spectrum).square()
        // fb is (1, featureSize, nFreqs); drop the leading batch dim so the
        // matmul contracts over frequency bins, as upstream's `self.fb[0]` does.
        let mel = MLX.matmul(power, fb[0].transposed())
        let logMel = MLX.log(mel + logGuard)
        return logMel.transposed(1, 0).expandedDimensions(axis: 0)
    }
}
