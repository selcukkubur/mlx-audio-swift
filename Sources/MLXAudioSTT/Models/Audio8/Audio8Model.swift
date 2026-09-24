import Foundation
import MLX
import MLXNN
import MLXAudioCore
import HuggingFace
import Tokenizers

/// Edge0's Audio8-ASR-Infinite: a Voxtral Realtime audio tower in front of a
/// Qwen2 decoder, decoding one text token per audio clock step.
///
/// **Most of this model was already in this package.** Its `audio_config` names
/// `voxtral_realtime_encoder` and means it: the tower is the same 32 layers
/// with the same bias pattern, and `VoxtralRealtimeAudioEncoder` ends in
/// `downsampleAndProject`, which groups encoder frames by `downsampleFactor`
/// and runs gelu(proj0) -> proj2 over them. That grouping and that projector
/// are exactly what this checkpoint calls `multi_modal_projector`, down to the
/// 10240-wide input (1280 x 8). So the whole audio path is reused rather than
/// rewritten, and the weights are renamed onto it.
///
/// What is genuinely new is the decoder — Qwen2 carries attention biases that
/// Voxtral's does not — and the run itself: text trails the audio by a fixed
/// number of tokens, which is what makes it streaming and what `t_cond`
/// encodes.
public final class Audio8Model: Module, STTGenerationModel {
    public let config: Audio8Config

    /// Emitted when a clock step decodes no new word.
    static let streamingPadToken = 151_665
    /// Emitted at a word boundary.
    static let streamingWordToken = 151_666

    @ModuleInfo(key: "encoder") var encoder: VoxtralRealtimeAudioEncoder
    @ModuleInfo(key: "decoder") var decoder: Audio8Decoder
    /// One row per supported clock (4, 6, 8 frames). Added to the delay
    /// embedding so the decoder knows which clock it is running at.
    @ModuleInfo(key: "frame_len_embedding") var frameLenEmbedding: Embedding

    var tokenizer: (any Tokenizer)?
    private var melFilters: MLXArray?

    public init(config: Audio8Config) {
        self.config = config
        let a = config.audioConfig
        let encoderConfig = VoxtralRealtimeEncoderConfig(
            dim: a.hiddenSize,
            nLayers: a.numHiddenLayers,
            nHeads: a.numAttentionHeads,
            headDim: a.headDim,
            hiddenDim: a.intermediateSize,
            nKvHeads: a.numKeyValueHeads,
            normEps: a.rmsNormEps,
            ropeTheta: a.ropeTheta,
            slidingWindow: a.slidingWindow,
            causal: true,
            useBiases: true,
            downsampleFactor: config.maxFrameLen
        )
        self._encoder.wrappedValue = VoxtralRealtimeAudioEncoder(
            encoderConfig, decoderDim: config.textConfig.hiddenSize)
        self._decoder.wrappedValue = Audio8Decoder(config.textConfig)
        self._frameLenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.frameLens.count,
            dimensions: config.textConfig.hiddenSize)
    }

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters()
    }

    /// How many tokens the text runs behind the audio, for this clock.
    ///
    /// The checkpoint post-trains specific pairs — at frame_len 8 those are a
    /// 320 ms delay (2 tokens) and 480 ms (3). 2 is taken as the default
    /// because it is the lower latency of the pair and the one its own table
    /// lists first.
    var delayTokens: Int { 2 }

    /// Audio the model is trained to hear before the first word. 9 tokens at
    /// frame_len 8, from `streaming_n_left_pad_tokens_by_frame_len`. Without it
    /// the decoder is asked for a token while the encoder still has nothing
    /// behind it, and the opening words are lost.
    var leftPadTokens: Int { 9 }

    private func ensureMelFilters() -> MLXArray {
        if let melFilters { return melFilters }
        let f = VoxtralRealtimeAudio.computeMelFilters(
            numMelBins: config.audioConfig.numMelBins,
            windowSize: 400,
            sampleRate: 16000
        ).asType(.float32)
        melFilters = f
        return f
    }

    /// The conditioning every layer is scaled by: a sinusoid of the delay,
    /// plus the row for this clock.
    private func buildTCond() -> MLXArray {
        let delay = voxtralComputeTimeEmbedding(
            tValue: Float(delayTokens), dim: config.textConfig.hiddenSize)
        let index = config.frameLens.firstIndex(of: config.maxFrameLen) ?? 0
        return delay + frameLenEmbedding.weight[index]
    }

    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters
    ) -> STTOutput {
        let samplesPerToken = config.maxFrameLen * 320  // 20 ms tower frames at 16 kHz
        let leftPad = leftPadTokens * samplesPerToken
        // Run past the end as well: with a delay the last words only arrive in
        // the steps after the audio has stopped.
        let rightPad = (delayTokens + 1 + 10) * samplesPerToken

        var flat = audio.ndim > 1 ? audio.mean(axis: -1) : audio

        // **Align the clip to a whole number of groups before padding, or the
        // encoder silently eats the first frames.**
        //
        // `VoxtralRealtimeAudioEncoder.convStem` trims its output down to a
        // multiple of `downsampleFactor` and trims from the FRONT
        // (`x = x[trunc...]`). That is Voxtral's convention and not Audio8's:
        // this checkpoint completes the last group by padding at the END and
        // keeps every frame it was given. Left alone the two disagree by
        // `convLen % 8` frames, which offsets every audio group against the
        // text. One group is exactly `samplesPerToken` samples, so rounding the
        // clip up to a multiple of it makes the trim a no-op.
        let aligned = (flat.shape[0] + samplesPerToken - 1) / samplesPerToken * samplesPerToken
        flat = MLX.padded(
            flat, widths: [IntOrPair((leftPad, rightPad + (aligned - flat.shape[0])))])

        var mel = VoxtralRealtimeAudio.computeMelSpectrogram(
            audio: flat, melFilters: ensureMelFilters(), windowSize: 400, hopLength: 160)
        if mel.shape[1] % 2 != 0 { mel = mel[0..., 1...] }

        let audioEmbeds = encoder(mel)
        let steps = audioEmbeds.shape[0]
        guard steps > 0 else { return STTOutput(text: "") }

        decoder.precomputeAdaScales(tCond: buildTCond())

        var out: [Int] = []
        var cache: [VoxtralRealtimeDecoderKVCache?]? = nil
        var next = config.bosTokenId
        var position = 0

        for step in 0..<steps {
            // One decode step per audio group: the token embedding and the
            // group's audio embedding are ADDED, which is how this model fuses
            // the two streams — there is no audio placeholder token.
            let tokenEmbed = decoder.embedTokens.weight[next].expandedDimensions(axis: 0)
            let fused = tokenEmbed + audioEmbeds[step ..< (step + 1), 0...]

            let (hidden, newCache) = decoder(fused, startPos: position, cache: cache)
            cache = newCache
            position += 1

            // Tied embeddings: the checkpoint ships no lm_head.
            let logits = MLX.matmul(hidden[-1], decoder.embedTokens.weight.transposed(1, 0))
            next = logits.argMax().item(Int.self)
            if next == config.eosTokenId { break }

            // The run-up is not speech and its tokens are not transcript. The
            // model is fed `leftPadTokens` of silence before the first word so
            // its encoder has something behind it, and it still emits one token
            // per step through that silence; kept, they arrive as a few commas
            // and a stray word before the first real one.
            guard step >= leftPadTokens + delayTokens else { continue }

            // `[STREAMING_PAD]` and `[STREAMING_WORD]` are the clock's own
            // punctuation — one is emitted whenever a step decodes no new word,
            // the other marks a word boundary. They carry no text.
            if next == Audio8Model.streamingPadToken || next == Audio8Model.streamingWordToken {
                continue
            }
            out.append(next)
        }

        let text = (try? tokenizer?.decode(tokens: out)) ?? nil
        return STTOutput(text: text ?? "")
    }

    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let out = self.generate(audio: audio, generationParameters: generationParameters)
            continuation.yield(.token(out.text))
            continuation.yield(.result(out))
            continuation.finish()
        }
    }

    static func fromPretrained(
        _ modelPath: String, cache: HubCache = .default
    ) async throws -> Audio8Model {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(domain: "Audio8Model", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"
            ])
        }
        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: "safetensors", hfToken: hfToken, cache: cache)
        return try await fromDirectory(dir)
    }

    static func fromDirectory(_ dir: URL) async throws -> Audio8Model {
        let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Audio8Config.self, from: configData)
        let model = Audio8Model(config: config)

        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" {
            for (k, v) in try MLX.loadArrays(url: file) { weights[k] = v }
        }
        // The converter writes the names this model declares, so nothing is
        // remapped here — a mismatch is a conversion bug and should surface as
        // one rather than being silently patched at load time.
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        model.tokenizer = try? await AutoTokenizer.from(modelFolder: dir)
        return model
    }
}
