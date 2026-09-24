import Foundation
import MLX
import MLXNN
import MLXFast

/// Multi-head attention with rotary position embeddings.
///
/// **`traditional: false` is load-bearing and must not be "tidied".** Upstream
/// builds `nn.RoPE(..., traditional=False, ...)`, which is the split-half
/// rotation — the `rotate_half` convention. `true` is the interleaved variant
/// that rotates consecutive pairs, per mlx-swift's own documentation on
/// `PositionalEncoding.swift`.
///
/// Choosing the wrong one does not crash and does not change a single shape.
/// It produces a model that loads, runs at full speed, and is confidently
/// wrong — which is exactly how the Audio8 port failed, through every shape
/// check and spot check that passed.
class NemotronAttention: Module {
    let nHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "w_qkv") var wQKV: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let rope: RoPE

    // `qk_norm` (a LayerNorm applied to q/k before RoPE when config.qkNorm is
    // true) is deliberately unimplemented: no shipped Nemotron 3 Diarization
    // checkpoint sets it (verified qkNorm: false), so this is a known,
    // intentional gap rather than an oversight.
    init(_ config: NemotronEncoderConfig) {
        self.nHeads = config.nHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self._wQKV.wrappedValue = Linear(
            config.dModel, 3 * config.dModel, bias: config.qkvBias
        )
        self._outProj.wrappedValue = Linear(config.dModel, config.dModel)
        self.rope = RoPE(
            dimensions: Int(Float(config.headDim) * config.rotaryFraction),
            traditional: false,
            base: config.ropeBase
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, t, d) = (x.dim(0), x.dim(1), x.dim(2))
        // Upstream reshapes to (b, t, 3, heads, headDim) then selects along
        // axis 2 — one fused projection, not three. Splitting it into three
        // Linears would need the checkpoint's single w_qkv tensor sliced in
        // the same order, so the fused form is kept.
        let qkv = wQKV(x).reshaped(b, t, 3, nHeads, headDim)
        let q = qkv[0..., 0..., 0].transposed(0, 2, 1, 3)
        let k = qkv[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)

        let qr = rope(q)
        let kr = rope(k)
        let out = MLXFast.scaledDotProductAttention(
            queries: qr, keys: kr, values: v, scale: scale,
            mask: mask.map { .array($0) } ?? .none
        )
        return outProj(out.transposed(0, 2, 1, 3).reshaped(b, t, d))
    }
}

class NemotronFeedForward: Module {
    @ModuleInfo(key: "linear1") var fc1: Linear
    @ModuleInfo(key: "linear2") var fc2: Linear

    init(_ config: NemotronEncoderConfig) {
        let hidden = Int(Float(config.dModel) * config.ffExpansion)
        self._fc1.wrappedValue = Linear(config.dModel, hidden)
        self._fc2.wrappedValue = Linear(hidden, config.dModel)
        super.init()
    }

    // Exact gelu, matching upstream's `nn.gelu` — not the tanh approximation.
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// Pre-norm: `preBlockNorm` is true for this model, so the norm is applied
/// before the sublayer and the residual carries the unnormalised input.
class NemotronTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: NemotronAttention
    @ModuleInfo(key: "ffn") var ff: NemotronFeedForward

    init(_ config: NemotronEncoderConfig) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._attn.wrappedValue = NemotronAttention(config)
        self._ff.wrappedValue = NemotronFeedForward(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + attn(norm1(x), mask: mask)
        h = h + ff(norm2(h))
        return h
    }
}

/// Stacks `subsamplingFactor` consecutive frames into one (10 ms mel frames
/// into 80 ms encoder frames) and owns the projection into `dModel`, matching
/// upstream's `FeatureStacking`, which fuses stacking and projection into one
/// module rather than splitting them across two.
/// `public`: the top-level model (and the parity harness that gates this
/// port) calls `preEncode` directly, ahead of `NemotronEncoder`, exactly as
/// upstream's `Model.__call__` calls `encoder.pre_encode(...)` before
/// `encoder(...)`.
public class NemotronFeatureStacking: Module {
    let factor: Int
    @ModuleInfo(key: "proj") public var proj: Linear

    init(_ config: NemotronEncoderConfig) {
        self.factor = config.subsamplingFactor
        // The checkpoint stores no bias for this projection — bias: false
        // is not the Linear default and must be explicit, or a bias term
        // gets trained-from-nothing (initialised to zero, then silently
        // present as an extra degree of freedom the checkpoint never set).
        self._proj.wrappedValue = Linear(
            config.featIn * config.subsamplingFactor, config.dModel, bias: false
        )
        super.init()
    }

    public func callAsFunction(_ features: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        // Mel arrives as (B, mel, T); transpose to (B, T, mel) before grouping.
        let x = features.transposed(0, 2, 1)
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        // Pad UP to a whole number of groups, never truncate down. Dropping
        // the tail to reach a multiple of `factor` silently shifts every
        // later frame's group membership by the dropped remainder — that
        // exact front-trim-vs-end-pad defect offset every group by two
        // frames in a previous port here and produced fluent nonsense.
        let padAmount = ((-t % factor) + factor) % factor
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, padAmount)), IntOrPair(0)])
        let grouped = padded.reshaped(b, (t + padAmount) / factor, c * factor)
        return (proj(grouped), (lengths + factor - 1) / factor)
    }
}

/// `public`: exposed beyond this file (and beyond this module) so a caller
/// — including the parity harness this port is gated on — can run
/// `preEncode` and `callAsFunction` as the two separate steps upstream's own
/// `Model.__call__` keeps them as, rather than only through the top-level
/// `NemotronDiarizationModel`.
public class NemotronEncoder: Module {
    @ModuleInfo(key: "pre_encode") public var preEncode: NemotronFeatureStacking
    @ModuleInfo(key: "embed_norm") var embedNorm: UnaryLayer
    @ModuleInfo(key: "layers") var layers: [NemotronTransformerBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    let scale: Float

    init(_ config: NemotronEncoderConfig) {
        self._preEncode.wrappedValue = NemotronFeatureStacking(config)
        self._embedNorm.wrappedValue =
            config.preBlockNorm ? LayerNorm(dimensions: config.dModel) : Identity()
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            NemotronTransformerBlock(config)
        }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        self.scale = config.xscaling ? Float(config.dModel).squareRoot() : 1.0
        super.init()
    }

    // Matches upstream's `Model`, which calls `encoder.pre_encode(mel, lengths)`
    // separately before `encoder(x, lengths)` — this method does NOT call
    // `preEncode` itself. `x` here is already stacked+projected, and
    // `lengths` are the post-stacking lengths `preEncode` returned.
    public func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> MLXArray {
        let valid = MLXArray(0..<x.dim(1)).expandedDimensions(axis: 0) .< lengths.expandedDimensions(axis: 1)
        let mask = valid.expandedDimensions(axes: [1, 2])
        var h = embedNorm(x * scale)
        for layer in layers { h = layer(h, mask: mask) }
        return finalNorm(h)
    }
}
