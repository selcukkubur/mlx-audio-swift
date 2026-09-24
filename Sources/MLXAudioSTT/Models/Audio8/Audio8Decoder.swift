import Foundation
import MLX
import MLXNN

/// Audio8's text decoder: Qwen2, modulated per layer by how far behind the
/// audio the text is allowed to run.
///
/// **Why this is not `VoxtralRealtimeDecoder`.** The two are the same shape and
/// differ in one detail that cannot be configured away: Qwen2 carries a bias on
/// the query, key and value projections and none on the output, where Voxtral's
/// decoder carries none at all. Everything else is shared outright —
/// `VoxtralRealtimeAdaRMSNorm` is the same bottleneck against the same
/// conditioning vector, and `VoxtralRealtimeDecoderKVCache` is the same cache —
/// so only the parts that genuinely differ are written here.
final class Audio8Attention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: Audio8Config.TextConfig) {
        nHeads = config.numAttentionHeads
        nKvHeads = config.numKeyValueHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        scale = pow(Float(headDim), -0.5)
        ropeTheta = config.ropeTheta

        let qDim = nHeads * headDim
        let kvDim = nKvHeads * headDim
        // Qwen2's bias pattern, and the whole reason this class exists.
        self._wq.wrappedValue = Linear(config.hiddenSize, qDim, bias: true)
        self._wk.wrappedValue = Linear(config.hiddenSize, kvDim, bias: true)
        self._wv.wrappedValue = Linear(config.hiddenSize, kvDim, bias: true)
        self._wo.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
    }

    private func rope(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let idx = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
        let invFreq = 1.0 / MLX.pow(MLXArray(ropeTheta), idx / Float(headDim))
        let angles = positions.asType(.float32).expandedDimensions(axis: 1)
            * invFreq.expandedDimensions(axis: 0)
        let cosA = MLX.cos(angles).expandedDimensions(axis: 1)
        let sinA = MLX.sin(angles).expandedDimensions(axis: 1)
        let half = x.shape[x.ndim - 1] / 2
        let x1 = x[.ellipsis, 0..<half]
        let x2 = x[.ellipsis, half...]
        return MLX.concatenated([x1 * cosA - x2 * sinA, x1 * sinA + x2 * cosA], axis: -1)
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        let seqLen = x.shape[0]
        var q = wq(x).reshaped(seqLen, nHeads, headDim)
        var k = wk(x).reshaped(seqLen, nKvHeads, headDim)
        let v = wv(x).reshaped(seqLen, nKvHeads, headDim)

        q = rope(q, positions: positions)
        k = rope(k, positions: positions)

        var keys = k
        var values = v
        if let cache {
            keys = MLX.concatenated([cache.keys, k], axis: 0)
            values = MLX.concatenated([cache.values, v], axis: 0)
        }
        let updated = VoxtralRealtimeDecoderKVCache(
            keys: keys, values: values, positionOffset: 0)

        // Grouped-query attention: repeat the key/value heads to meet the
        // query heads. 16 query heads against 2 key/value heads here.
        let repeats = nHeads / nKvHeads
        var kk = keys, vv = values
        if repeats > 1 {
            kk = MLX.repeated(keys, count: repeats, axis: 1)
            vv = MLX.repeated(values, count: repeats, axis: 1)
        }

        let qT = q.transposed(1, 0, 2)
        let kT = kk.transposed(1, 2, 0)
        let vT = vv.transposed(1, 0, 2)
        var scores = MLX.matmul(qT, kT) * scale

        // Causal over the cached prefix: position i may read up to i.
        let total = kk.shape[0]
        if seqLen > 1 {
            let rows = MLXArray(0..<seqLen).reshaped(seqLen, 1) + (total - seqLen)
            let cols = MLXArray(0..<total).reshaped(1, total)
            let mask = MLX.where(cols .<= rows, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            scores = scores + mask.expandedDimensions(axis: 0)
        }

        let weights = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        let out = MLX.matmul(weights, vT).transposed(1, 0, 2).reshaped(seqLen, nHeads * headDim)
        return (wo(out), updated)
    }
}

final class Audio8DecoderLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: Audio8Attention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ada_rms_norm") var adaRmsNorm: VoxtralRealtimeAdaRMSNorm

    @ModuleInfo(key: "feed_forward_w1") var w1: Linear
    @ModuleInfo(key: "feed_forward_w3") var w3: Linear
    @ModuleInfo(key: "feed_forward_w2") var w2: Linear

    init(_ config: Audio8Config.TextConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attention.wrappedValue = Audio8Attention(config)
        self._ffnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._adaRmsNorm.wrappedValue = VoxtralRealtimeAdaRMSNorm(
            dim: config.hiddenSize, bottleneckDim: config.adaRmsNormTCondDim)
        self._w1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._w3.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, adaScale: MLXArray,
        cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        let (attn, newCache) = attention(attentionNorm(x), positions: positions, cache: cache)
        var h = x + attn
        // The delay modulation sits between the norm and the MLP, which is
        // where `Qwen2RealtimeV1DecoderLayer` puts it: the block is scaled by
        // how far behind the audio this text is running before it is mixed.
        let normed = adaRmsNorm(ffnNorm(h), adaScale: adaScale)
        h = h + w2(silu(w1(normed)) * w3(normed))
        return (h, newCache)
    }
}

final class Audio8Decoder: Module {
    let config: Audio8Config.TextConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Audio8DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private var adaScales: [MLXArray] = []

    init(_ config: Audio8Config.TextConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            Audio8DecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// The conditioning is the same for every layer and every step of a run, so
    /// it is squeezed through each layer's bottleneck once rather than per
    /// token — the same reason `VoxtralRealtimeDecoder` precomputes it.
    func precomputeAdaScales(tCond: MLXArray) {
        adaScales = layers.map { $0.adaRmsNorm.computeScale(tCond: tCond) }
    }

    func callAsFunction(
        _ embeds: MLXArray, startPos: Int, cache: [VoxtralRealtimeDecoderKVCache?]?
    ) -> (MLXArray, [VoxtralRealtimeDecoderKVCache?]) {
        var h = embeds
        let positions = MLXArray(startPos..<(startPos + h.shape[0])).asType(.int32)
        var newCache: [VoxtralRealtimeDecoderKVCache?] = []
        newCache.reserveCapacity(layers.count)
        for (i, layer) in layers.enumerated() {
            let (out, c) = layer(
                h, positions: positions, adaScale: adaScales[i], cache: cache?[i])
            h = out
            newCache.append(c)
        }
        return (norm(h), newCache)
    }
}
