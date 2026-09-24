import Foundation
import MLX

/// Split-half rotary embedding — the convention `transformers` applies in
/// `apply_rotary_pos_emb`, pairing the first half of each head's channels with
/// the second half.
///
/// This sits beside `voxtralApplyInterleavedRoPE` because the two are not
/// interchangeable: interleaved pairs adjacent channels (0 with 1, 2 with 3),
/// split-half pairs across the midpoint (0 with 64, 1 with 65). The reference
/// implementation of this encoder uses the latter.
func voxtralApplySplitRoPE(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    nHeads: Int,
    headDim: Int
) -> MLXArray {
    let seqLen = x.shape[0]
    let halfDim = headDim / 2
    let reshaped = x.reshaped(seqLen, nHeads, headDim)

    let x1 = reshaped[0..., 0..., 0..<halfDim]
    let x2 = reshaped[0..., 0..., halfDim...]

    let cosE = cos.expandedDimensions(axis: 1)
    let sinE = sin.expandedDimensions(axis: 1)

    let o1 = x1 * cosE - x2 * sinE
    let o2 = x2 * cosE + x1 * sinE
    return MLX.concatenated([o1, o2], axis: -1).reshaped(seqLen, nHeads * headDim)
}
