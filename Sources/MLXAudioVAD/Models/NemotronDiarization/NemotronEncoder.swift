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
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: NemotronEncoderConfig) {
        let hidden = Int(Float(config.dModel) * config.ffExpansion)
        self._fc1.wrappedValue = Linear(config.dModel, hidden)
        self._fc2.wrappedValue = Linear(hidden, config.dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

/// Pre-norm: `preBlockNorm` is true for this model, so the norm is applied
/// before the sublayer and the residual carries the unnormalised input.
class NemotronTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: NemotronAttention
    @ModuleInfo(key: "ff") var ff: NemotronFeedForward

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

/// Stacks `subsamplingFactor` consecutive frames into one, which is how the
/// 10 ms mel frames become 80 ms encoder frames.
class NemotronFeatureStacking: Module {
    let factor: Int
    init(_ config: NemotronEncoderConfig) {
        self.factor = config.subsamplingFactor
        super.init()
    }

    func callAsFunction(_ features: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        let (b, t, d) = (features.dim(0), features.dim(1), features.dim(2))
        let usable = (t / factor) * factor
        let trimmed = features[0..., 0..<usable, 0...]
        return (trimmed.reshaped(b, usable / factor, d * factor), lengths / factor)
    }
}

class NemotronEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: Linear
    @ModuleInfo(key: "layers") var layers: [NemotronTransformerBlock]
    @ModuleInfo(key: "stacking") var stacking: NemotronFeatureStacking

    init(_ config: NemotronEncoderConfig) {
        self._stacking.wrappedValue = NemotronFeatureStacking(config)
        self._preEncode.wrappedValue = Linear(
            config.featIn * config.subsamplingFactor, config.dModel
        )
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            NemotronTransformerBlock(config)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        let (stacked, outLengths) = stacking(x, lengths: lengths)
        var h = preEncode(stacked)
        let mask = SortformerModules.lengthToMask(outLengths, maxLength: h.dim(1))
        for layer in layers { h = layer(h, mask: mask) }
        return (h, outLengths)
    }
}
