import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXAudioCore
import HuggingFace

/// Nemotron 3 Diarization's speaker head. Subclasses Sortformer's
/// `SortformerModules` — matching upstream's `SpeakerModules(SortformerModules)`
/// (`nemotron_diarization.py` lines 161-180) — to share `encoder_proj`,
/// `first_hidden_to_hidden`, `single_hidden_to_spks` and `hidden_to_spks`
/// rather than duplicate them.
///
/// Adds exactly the checkpoint tensors Sortformer's base class does not have
/// (verified against `checkpoint-tensors.txt`'s `sortformer_modules.*` keys):
/// a subpixel-upsampling `Conv1d` that restores the encoder's 8x-subsampled
/// time resolution before scoring, a learnable silence embedding AOSC
/// compression uses to fill disabled cache slots, and an activity head that
/// is present in every shipped checkpoint but is not on this model's forward
/// path (upstream's own `__call__`, lines 176-180, never touches it either).
public class SpeakerModules: SortformerModules {
    @ModuleInfo(key: "subpixel_upsample") var subpixelUpsample: Conv1d
    @ParameterInfo(key: "learnable_sil_emb") var learnableSilEmb: MLXArray
    @ModuleInfo(key: "activity_head") var activityHead: Sequential?

    override init(_ config: ModulesConfig) {
        // Checkpoint-verified: subpixel_upsample.weight is [1536, 3, 192] =
        // [tf_d_model * subsampling_factor, kernel=3, tf_d_model] — already
        // MLX's native Conv1d layout ([O, K, I]), not PyTorch's ([O, I, K]),
        // so no permute is needed when loading (unlike Sortformer's own
        // torch-native pointwise/depthwise convs).
        self._subpixelUpsample.wrappedValue = Conv1d(
            inputChannels: config.tfDModel,
            outputChannels: config.tfDModel * config.subsamplingFactor,
            kernelSize: 3,
            padding: 1
        )
        // fc_d_model (512), not tf_d_model (192): this fills disabled
        // spkcache slots at encoder-embedding resolution, before
        // `encoder_proj` — matching the checkpoint's [512] shape.
        self._learnableSilEmb.wrappedValue = MLXArray.zeros([config.fcDModel])
        self._activityHead.wrappedValue = config.useActivityHead
            ? Sequential(layers: LayerNorm(dimensions: config.tfDModel), Linear(config.tfDModel, 3))
            : nil
        super.init(config)
    }

    /// Matches upstream's `SpeakerModules.__call__`: project to `tf_d_model`,
    /// subpixel-upsample the time axis back to native (10 ms) resolution by
    /// reshaping the upsample conv's channel-multiplied output, then run the
    /// shared speaker-sigmoid head.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = encoderProj(x)
        let (b, _, hDim) = (h.dim(0), h.dim(1), h.dim(2))
        let upsampled = subpixelUpsample(h).reshaped(b, -1, hDim)
        return forwardSpeakerSigmoids(upsampled)
    }
}

/// A single speaker-active time span produced by `feed`.
public struct NemotronSpeakerSegment: Sendable {
    public let speakerIndex: Int
    public let start: Double
    public let end: Double
    /// `true` once the session has seen `feed(..., final: true)` and this
    /// span's end time reflects the fully flushed stream rather than a
    /// provisional boundary that a later chunk could still extend.
    public let isFinal: Bool
}

/// Nemotron 3 Diarization: the 31-layer rotary FastConformer-replacement
/// encoder (`NemotronEncoder`) feeding Sortformer's AOSC/FIFO streaming
/// speaker head (`SpeakerModules`, above), fronted by a checkpoint-exact
/// log-mel frontend (`NemotronMelFeatures`). See this directory's README for
/// the parity figures that gate this port.
public class NemotronDiarizationModel: Module {
    public private(set) var config: NemotronDiarizationConfig

    // `encoder` and `preprocessor` are `public`, not just `internal`, so the
    // parity harness (a separate executable target, gating this whole port)
    // can run and compare each stage in isolation rather than only the
    // fully-assembled model.
    @ModuleInfo(key: "encoder") public var encoder: NemotronEncoder
    @ModuleInfo(key: "sortformer_modules") var sortformerModules: SpeakerModules
    @ModuleInfo(key: "preprocessor") public var preprocessor: NemotronMelFeatures

    public init(_ config: NemotronDiarizationConfig) {
        self.config = config
        self._encoder.wrappedValue = NemotronEncoder(config.encoderConfig)
        self._sortformerModules.wrappedValue = SpeakerModules(config.modulesConfig)
        self._preprocessor.wrappedValue = NemotronMelFeatures(config.processorConfig)
        super.init()
    }

    /// The encoder's own weight dtype (BF16 throughout the published
    /// checkpoint), matching upstream's `Model.dtype` property.
    public var dtype: DType { encoder.preEncode.proj.weight.dtype }

    public var sampleRate: Int { config.processorConfig.samplingRate }

    /// Single-window forward: mel features `(B, mel, T)` -> speaker
    /// probabilities `(B, ceil(T/8)*8, numSpeakers)`. Matches upstream's
    /// `Model.__call__` (lines 219-226) exactly, including masking by the
    /// *pre*-stacking length (`lengths`), not the post-stacking one — the
    /// subpixel upsample inside `sortformerModules` restores native time
    /// resolution, so the original mel-frame count is what bounds validity.
    public func callAsFunction(_ melFeatures: MLXArray, lengths: MLXArray) -> MLXArray {
        let (x, stackedLengths) = encoder.preEncode(melFeatures.asType(dtype), lengths: lengths)
        let probs = sortformerModules(encoder(x, lengths: stackedLengths))
        let valid = MLXArray(0..<probs.dim(1)).expandedDimensions(axis: 0) .< lengths.expandedDimensions(axis: 1)
        return probs * valid.expandedDimensions(axis: 2).asType(probs.dtype)
    }

    // MARK: - Streaming

    /// Selects one of NVIDIA's latency presets, matching upstream's
    /// `set_streaming_config(preset)` (`nemotron_diarization.py` lines
    /// 237-260) both in name and in shape: **this mutates the model**, not
    /// any particular stream. Upstream keeps this as an explicit method on
    /// the model precisely so that global-ness is visible at the call site
    /// rather than discovered later — folding it into a state factory would
    /// hide it. One `NemotronDiarizationModel` instance can therefore only
    /// serve one streaming configuration at a time: call this before
    /// `initStreamingState()`, and do not run two concurrent streams that
    /// want different presets against the same model instance — create a
    /// second `NemotronDiarizationModel` (sharing weights is not supported
    /// by this API) instead.
    public func setStreamingConfig(_ preset: NemotronStreamingPreset) {
        var modules = config.modulesConfig
        modules.chunkLen = preset.chunk
        modules.chunkRightContext = preset.rightContext
        modules.fifoLen = preset.fifo
        modules.spkcacheLen = 264
        modules.spkcacheUpdatePeriod = preset.cacheUpdatePeriod
        config.modulesConfig = modules
    }

    /// Fresh, empty streaming state under whatever preset
    /// `setStreamingConfig` last selected (or the checkpoint's own default
    /// `modules_config` if it was never called). Unlike the model-level
    /// config, this is genuinely per-stream: it mutates nothing on the
    /// model and a caller can hold as many independent states as it likes
    /// against one model instance, as long as they all share that one
    /// active preset.
    public func initStreamingState() -> NemotronStreamingState {
        NemotronStreamingState(config: config, dtype: dtype)
    }

    /// Process one feature window against the current cache/FIFO, mutating
    /// `state` and returning the native-resolution speaker probabilities for
    /// exactly the window's `centralFrames` new frames. Direct port of
    /// upstream's `Model.streaming_step` (lines 262-323); context
    /// predictions (spkcache/FIFO lookback and right-context lookahead)
    /// never themselves enter the FIFO, only the central frames do.
    public func streamingStep(
        _ features: MLXArray,
        state: inout NemotronStreamingState,
        centralFrames: Int,
        featureLength: Int
    ) -> MLXArray {
        let cfg = config.modulesConfig
        let factor = config.encoderConfig.subsamplingFactor
        let (chunk, lengths) = encoder.preEncode(features.asType(dtype), lengths: MLXArray([Int32(featureLength)]))
        let cacheLen = state.spkcache.dim(1)
        let fifoLen = state.fifo.dim(1)
        let combined = MLX.concatenated([state.spkcache, state.fifo, chunk], axis: 1)
        var high = sortformerModules(encoder(combined, lengths: lengths + (cacheLen + fifoLen)))

        // Validity is bounded by the *real* audio length (`lengths[0]`),
        // not `high.shape[1]`: a window's chunk is padded up to a multiple
        // of `factor` regardless of how many of its frames are real, and
        // upstream masks against that real count for exactly that reason.
        let validCount = (lengths[0] + (cacheLen + fifoLen)) * factor
        let validMask = MLXArray(0..<high.dim(1)) .< validCount
        high = high * validMask.expandedDimensions(axis: 0).expandedDimensions(axis: -1).asType(high.dtype)

        // Cache scoring always operates at encoder resolution: mean-pool
        // native-resolution probabilities back down by `factor`.
        let low = high.asType(.float32).reshaped(1, -1, factor, config.numSpeakers).mean(axis: 2)

        let n = (centralFrames + factor - 1) / factor
        let start = cacheLen + fifoLen
        let result = high[0, (start * factor)..<(start * factor + centralFrames)].asType(.float32)

        state.fifo = MLX.concatenated([state.fifo, chunk[0..., 0..<n, 0...]], axis: 1)
        state.fifoPreds = MLX.concatenated(
            [low[0..., cacheLen..<start, 0...], low[0..., start..<(start + n), 0...]], axis: 1
        )

        if state.fifo.dim(1) > cfg.fifoLen {
            let pop = min(state.fifo.dim(1), max(cfg.spkcacheUpdatePeriod, state.fifo.dim(1) - cfg.fifoLen))
            state.spkcache = MLX.concatenated([state.spkcache, state.fifo[0..., 0..<pop, 0...]], axis: 1)
            // Once compressed, the cache's own predictions are frozen and
            // reused verbatim — this window's `low` no longer lines up
            // with the compressed (non-contiguous) cache frames.
            let previous = state.spkcacheCompressed ? state.spkcachePreds : low[0..., 0..<cacheLen, 0...]
            state.spkcachePreds = MLX.concatenated([previous, state.fifoPreds[0..., 0..<pop, 0...]], axis: 1)
            state.fifo = state.fifo[0..., pop..., 0...]
            state.fifoPreds = state.fifoPreds[0..., pop..., 0...]

            if state.spkcache.dim(1) > cfg.spkcacheLen {
                let (compressedCache, compressedPreds) = SortformerModel.compressSpkcacheAosc(
                    embs: state.spkcache,
                    preds: state.spkcachePreds,
                    meanSilEmb: sortformerModules.learnableSilEmb.expandedDimensions(axis: 0),
                    modulesCfg: cfg
                )
                state.spkcache = compressedCache
                state.spkcachePreds = compressedPreds
                state.spkcacheCompressed = true
            }
        }

        state.frameOffset += centralFrames
        eval(result, state.spkcache, state.spkcachePreds, state.fifo, state.fifoPreds)
        return result
    }

    /// The raw per-frame speaker probabilities `feed` would emit for this
    /// call, before segment extraction — exposed publicly (beyond what
    /// `feed`'s own `[NemotronSpeakerSegment]` return type can carry) so a
    /// caller — including the parity harness this task is gated on — can
    /// compare against upstream's `result.speaker_probs` directly. Applies
    /// upstream's `_output` pooling by `outputSubsamplingFactor` (a no-op
    /// for the published checkpoint, which sets it to 1).
    ///
    /// - Precondition: `state.finished` must be `false`. Upstream raises a
    ///   catchable `ValueError` on a finished stream; this is not `throws`,
    ///   so calling it again after a `final: true` call traps instead
    ///   (`precondition`, which stays live under `-O`) rather than
    ///   returning an error a caller could recover from.
    public func feedProbabilities(
        _ samples: [Float], state: inout NemotronStreamingState, final: Bool
    ) -> MLXArray {
        precondition(!state.finished, "This stream is finished; call initStreamingState() for a new one")
        state.pcmTail.append(contentsOf: samples)
        state.samplesReceived += samples.count

        let cfg = config.modulesConfig
        let proc = config.processorConfig
        let factor = config.encoderConfig.subsamplingFactor
        let central = cfg.chunkLen * factor
        let right = cfg.chunkRightContext * factor

        var outputs: [MLXArray] = []
        while true {
            let available = state.samplesReceived / proc.hopLength - state.frameOffset
            let needed = (state.frameOffset + central + right - 1) * proc.hopLength + proc.nFFT / 2
            if available <= 0 || (!final && state.samplesReceived < needed) { break }
            let n = min(central, available)
            let windowFrames: Int
            if final {
                // NeMo masks the extra centered STFT frame, then pads to pad_to.
                var totalFrames = state.samplesReceived / proc.hopLength + 1
                if proc.padTo > 0 {
                    totalFrames += (proc.padTo - totalFrames % proc.padTo) % proc.padTo
                }
                windowFrames = min(central + right, totalFrames - state.frameOffset)
            } else {
                windowFrames = central + right
            }
            let audioArray = MLXArray(state.pcmTail)
            let features = preprocessor(
                audioArray,
                start: state.frameOffset,
                count: windowFrames,
                sampleOffset: state.pcmOffset,
                totalSamples: state.samplesReceived
            )
            let result = streamingStep(
                features, state: &state, centralFrames: n, featureLength: min(windowFrames, available)
            )
            outputs.append(result)

            let keepFrom = max(0, state.frameOffset * proc.hopLength - proc.nFFT / 2 - 1)
            let localKeepFrom = keepFrom - state.pcmOffset
            if localKeepFrom > 0 {
                state.pcmTail.removeFirst(min(localKeepFrom, state.pcmTail.count))
            }
            state.pcmOffset = keepFrom
        }

        state.finished = final
        if final { state.pcmTail = [] }

        var probs = outputs.isEmpty
            ? MLXArray.zeros([0, config.numSpeakers])
            : MLX.concatenated(outputs, axis: 0)

        let outFactor = config.outputSubsamplingFactor
        if outFactor > 1 && probs.dim(0) > 0 {
            let length = probs.dim(0)
            let padAmount = ((-length % outFactor) + outFactor) % outFactor
            let padded = MLX.padded(probs, widths: [IntOrPair((0, padAmount)), IntOrPair(0)])
            let groups = padded.dim(0) / outFactor
            let counts = MLX.minimum(
                MLXArray(outFactor),
                MLXArray(length) - MLXArray(0..<groups) * outFactor
            )
            probs = padded.reshaped(groups, outFactor, config.numSpeakers).sum(axis: 1)
                / counts.expandedDimensions(axis: 1).asType(.float32)
        }
        return probs
    }

    /// Feed arbitrary mono PCM chunks at the model's sample rate. Call once
    /// with `final: true` to flush lookahead. Direct port of upstream's
    /// `Model.feed` (lines 366-421); segment extraction (upstream's
    /// `_output`) is applied here on top of `feedProbabilities`.
    ///
    /// - Warning: calling this again on a `state` that has already seen
    ///   `final: true` traps (see `feedProbabilities`), it does not throw —
    ///   `feed` itself is not `throws`. Start a new stream instead: build a
    ///   fresh `state` from `initStreamingState()`.
    public func feed(
        _ samples: [Float], state: inout NemotronStreamingState, final: Bool,
        threshold: Float = 0.5, minDuration: Float = 0.0, mergeGap: Float = 0.0
    ) -> [NemotronSpeakerSegment] {
        let proc = config.processorConfig
        let offsetSeconds = Double(state.frameOffset) * Double(proc.hopLength) / Double(sampleRate)
        let probs = feedProbabilities(samples, state: &state, final: final)

        let stride = Float(proc.hopLength * config.outputSubsamplingFactor) / Float(sampleRate)
        let segments = SortformerModel.predsToSegments(
            probs, frameDuration: stride, threshold: threshold, minDuration: minDuration, mergeGap: mergeGap
        )
        let sessionEnd = Double(state.frameOffset) * Double(proc.hopLength) / Double(sampleRate)
        return segments.map {
            NemotronSpeakerSegment(
                speakerIndex: $0.speaker,
                start: Double($0.start) + offsetSeconds,
                end: min(Double($0.end) + offsetSeconds, sessionEnd),
                isFinal: final
            )
        }
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ repoId: String, cache: HubCache = .default
    ) async throws -> NemotronDiarizationModel {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NSError(
                domain: "NemotronDiarizationModel", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)"]
            )
        }
        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: ".safetensors", cache: cache
        )
        return try fromModelDirectory(modelURL)
    }

    public static func fromModelDirectory(_ modelURL: URL) throws -> NemotronDiarizationModel {
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(NemotronDiarizationConfig.self, from: configData)

        let model = NemotronDiarizationModel(config)

        let weightFiles = try FileManager.default.contentsOfDirectory(
            at: modelURL, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }

        var allWeights = [String: MLXArray]()
        for file in weightFiles {
            for (k, v) in try loadArrays(url: file) { allWeights[k] = v }
        }

        // Checkpoint-native layout throughout (Conv1d included, see
        // `SpeakerModules`'s comment on `subpixel_upsample`) — no
        // sanitize/permute pass is needed, unlike Sortformer's loader.
        try model.update(parameters: ModuleParameters.unflattened(allWeights), verify: .noUnusedKeys)
        eval(model.parameters())
        return model
    }
}
