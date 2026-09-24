import Foundation

/// Edge0's Audio8-ASR-Infinite, as its own `config.json` describes it.
///
/// **The audio half is not new and is not reimplemented.** Its `audio_config`
/// says `model_type: voxtral_realtime_encoder`, and it is one: 32 layers,
/// hidden 1280, 128 mel bins, and — the detail that settles it — exactly the
/// bias pattern `VoxtralRealtimeEncoderAttention` hardcodes, with a bias on the
/// query, value and output projections and none on the key. So the tower is
/// `VoxtralRealtimeAudioEncoder` with renamed weights, and only the decoder
/// side of this file describes anything that did not already exist here.
/// `rope_theta` sits inside a `rope_parameters` object in this checkpoint,
/// rather than beside the other fields where most configs this package reads
/// put it.
private struct Audio8RopeParameters: Decodable {
    let ropeTheta: Float?
    enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
}

public struct Audio8Config: Decodable, Sendable {

    /// The Qwen2 decoder, plus the modulation that makes it streaming.
    public struct TextConfig: Decodable, Sendable {
        public var hiddenSize: Int
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var numKeyValueHeads: Int
        public var intermediateSize: Int
        public var rmsNormEps: Float
        public var ropeTheta: Float
        public var vocabSize: Int
        /// The bottleneck the delay conditioning is squeezed through before it
        /// scales each layer. Voxtral calls the same thing
        /// `ada_rms_norm_t_cond_dim`.
        public var adaRmsNormTCondDim: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case intermediateSize = "intermediate_size"
            case rmsNormEps = "rms_norm_eps"
            case ropeParameters = "rope_parameters"
            case vocabSize = "vocab_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
            numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
            numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try Audio8Config.ropeTheta(from: c, key: .ropeParameters)
            vocabSize = try c.decode(Int.self, forKey: .vocabSize)
            // Not in the checkpoint's config at all; 32 is what the weights
            // say, `ada_rms_norm.ada_down` being (32, 2048), and it is also
            // Voxtral's own default for the same bottleneck.
            adaRmsNormTCondDim = 32
        }
    }

    /// Only the fields the audio tower needs, read from `audio_config` and
    /// handed to `VoxtralRealtimeEncoderConfig`.
    public struct AudioConfig: Decodable, Sendable {
        public var hiddenSize: Int
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var numKeyValueHeads: Int
        public var headDim: Int
        public var intermediateSize: Int
        public var numMelBins: Int
        public var slidingWindow: Int
        public var rmsNormEps: Float
        public var ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case numMelBins = "num_mel_bins"
            case slidingWindow = "sliding_window"
            case rmsNormEps = "rms_norm_eps"
            case ropeParameters = "rope_parameters"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
            numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
            numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
                ?? (try c.decode(Int.self, forKey: .numAttentionHeads))
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 750
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            ropeTheta = try Audio8Config.ropeTheta(from: c, key: .ropeParameters)
        }
    }

    /// `rope_theta` is nested under `rope_parameters` in both halves of this
    /// checkpoint, not left at the top of the section the way most configs
    /// this package reads put it.
    fileprivate static func ropeTheta<K: CodingKey>(
        from container: KeyedDecodingContainer<K>, key: K
    ) throws -> Float {
        let rope = try container.decodeIfPresent(Audio8RopeParameters.self, forKey: key)
        return rope?.ropeTheta ?? 1_000_000
    }

    public var textConfig: TextConfig
    public var audioConfig: AudioConfig
    /// How many encoder frames make one decode step. The card post-trains
    /// 4, 6 and 8 — an 80, 120 or 160 ms clock — and 8 is the default here.
    public var maxFrameLen: Int
    public var frameLens: [Int]
    public var bosTokenId: Int
    public var eosTokenId: Int
    /// What the projector reads: `frame_len` encoder frames concatenated.
    public var projectionSize: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case audioConfig = "audio_config"
        case maxFrameLen = "max_frame_len"
        case frameLens = "frame_lens"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case projectionSize = "projection_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(TextConfig.self, forKey: .textConfig)
        audioConfig = try c.decode(AudioConfig.self, forKey: .audioConfig)
        maxFrameLen = try c.decodeIfPresent(Int.self, forKey: .maxFrameLen) ?? 8
        frameLens = try c.decodeIfPresent([Int].self, forKey: .frameLens) ?? [4, 6, 8]
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 151_644
        // The checkpoint writes this as a single-element list.
        if let single = try? c.decode(Int.self, forKey: .eosTokenId) {
            eosTokenId = single
        } else {
            eosTokenId = (try? c.decode([Int].self, forKey: .eosTokenId))?.first ?? 151_645
        }
        projectionSize = try c.decodeIfPresent(Int.self, forKey: .projectionSize)
            ?? (audioConfig.hiddenSize * maxFrameLen)
    }
}
