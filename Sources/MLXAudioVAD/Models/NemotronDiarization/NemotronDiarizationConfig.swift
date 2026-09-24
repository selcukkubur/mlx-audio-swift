import Foundation

/// Field names match the JSON in mlx-community/Nemotron-3-Diarization, and
/// the defaults match upstream's config.py so a missing key behaves the way
/// the reference does rather than silently becoming zero.
public struct NemotronEncoderConfig: Codable, Sendable {
    public var featIn: Int = 128
    public var dModel: Int = 512
    public var nLayers: Int = 31
    public var nHeads: Int = 8
    public var subsamplingFactor: Int = 8
    public var ffExpansion: Float = 4.0
    public var qkvBias: Bool = false
    public var qkNorm: Bool = false
    public var preBlockNorm: Bool = true
    public var xscaling: Bool = false
    public var ropeBase: Float = 10_000.0
    public var rotaryFraction: Float = 1.0

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case dModel = "d_model"
        case nLayers = "n_layers"
        case nHeads = "n_heads"
        case subsamplingFactor = "subsampling_factor"
        case ffExpansion = "ff_expansion"
        case qkvBias = "qkv_bias"
        case qkNorm = "qk_norm"
        case preBlockNorm = "pre_block_norm"
        case xscaling
        case ropeBase = "rope_base"
        case rotaryFraction = "rotary_fraction"
    }

    public var headDim: Int { dModel / nHeads }
}

/// No separate Nemotron modules config: the checkpoint labels this section
/// `sortformer_modules` and the existing `ModulesConfig` already models it,
/// declaring 19 of the 26 fields with correct snake_case CodingKeys and sensible
/// defaults. Reusing it means `SortformerModules.init` takes the value directly
/// without conversion, and avoids duplicating a subset of an existing type.

public struct NemotronDiarizationConfig: Codable, Sendable {
    public var modelType: String = "nemotron_diarization"
    public var numSpeakers: Int = 8
    public var outputSubsamplingFactor: Int = 1
    public var encoderConfig = NemotronEncoderConfig()
    public var modulesConfig: ModulesConfig

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numSpeakers = "num_speakers"
        case outputSubsamplingFactor = "output_subsampling_factor"
        case encoderConfig = "encoder_config"
        case modulesConfig = "modules_config"
    }
}

/// NVIDIA's recommended streaming values, in 80 ms encoder frames. The app
/// uses `.low`; `.offline` exists for post-hoc scoring of a finished file.
public enum NemotronStreamingPreset: String, Sendable, CaseIterable {
    case offline, low, veryLow, ultraLow

    public var chunk: Int {
        switch self {
        case .offline: return 340
        case .low: return 9
        case .veryLow: return 6
        case .ultraLow: return 3
        }
    }
    public var rightContext: Int {
        switch self {
        case .offline: return 40
        case .low: return 4
        case .veryLow: return 2
        case .ultraLow: return 1
        }
    }
    public var fifo: Int {
        switch self {
        case .offline: return 40
        default: return 264
        }
    }
    public var cacheUpdatePeriod: Int {
        switch self {
        case .offline: return 300
        default: return 222
        }
    }
}
