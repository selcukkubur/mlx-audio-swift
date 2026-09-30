import Foundation
import MLX

// MARK: - Metadata

/// One entry of the `phonon_packed` metadata key in the published safetensors.
///
/// The file describes itself so a caller can enumerate what is packed without
/// string surgery. `name` is the prefix the converter used for the block's
/// tensors (`<name>.trits`, `<name>.hi_bits`, ...); `tensor` is the exact
/// logical tensor the block replaces. Use `tensor` as given rather than
/// appending `.weight` to `name`: the suffix is a property of the source
/// checkpoint, and the int6 LSTM tables genuinely do not have one.
public struct PhononPackedEntry: Codable, Sendable, Equatable {
    public let name: String
    public let tensor: String
    /// Logical shape: `[O, I]` for most modules, `[O, 1, I]` for pointwise
    /// convolutions.
    public let shape: [Int]
    public let tile: Int
}

/// One entry of the `phonon_int6` metadata key. Same contract as
/// `PhononPackedEntry`, including that `tensor` may lack a `.weight` suffix.
public struct PhononInt6Entry: Codable, Sendable, Equatable {
    public let name: String
    public let tensor: String
    public let shape: [Int]
    public let bits: Int
}

// MARK: - Five-value

/// One five-value weight matrix, as Fermion pack it.
///
/// Each weight takes one of five levels — `{-hi, -lo, 0, +lo, +hi}` — with `lo`
/// and `hi` per output row. The sign is a base-3 trit, five to a byte; the
/// choice between `lo` and `hi` is one bit, stored **only for non-zero
/// weights**, which is what makes the format 2.09 bits per weight and what
/// makes it awkward to address: bit *n* belongs to the *n*-th non-zero in
/// row-major order, so a reader needs to know how many non-zeros precede it.
/// `nzRow` and `nzTile` are the precomputed answer.
// `@unchecked` because MLXArray is not declared Sendable; every field here is a
// `let` that is never written after load, so sharing across tasks is safe.
public struct PhononFiveValue: @unchecked Sendable {
    public let trits: MLXArray      // uint8  [O, ceil(I/5)]
    public let hiBits: MLXArray     // uint8  [ceil(nnz/8)]
    public let nzTile: MLXArray     // uint16 [O, ceil(I/tile)]
    public let nzRow: MLXArray      // uint32 [O]
    public let lo: MLXArray         // float16 [O]
    public let hi: MLXArray         // float16 [O]
    public let outFeatures: Int
    public let inFeatures: Int
    public let tile: Int

    public init(trits: MLXArray, hiBits: MLXArray, nzTile: MLXArray, nzRow: MLXArray,
                lo: MLXArray, hi: MLXArray, outFeatures: Int, inFeatures: Int, tile: Int) {
        self.trits = trits; self.hiBits = hiBits; self.nzTile = nzTile
        self.nzRow = nzRow; self.lo = lo; self.hi = hi
        self.outFeatures = outFeatures; self.inFeatures = inFeatures; self.tile = tile
    }

    /// Builds one module from the arrays of a converted file, or nil if any of
    /// its six pieces is missing or has a shape the metadata does not describe.
    ///
    /// `shape` is the module's logical shape from the file's metadata. It is
    /// `[O, I]` for most modules but `[O, 1, I]` for pointwise convolutions, so
    /// the two features are read as the first and last elements, which is
    /// right for both forms without the caller having to squeeze anything.
    ///
    /// The dimension checks matter more than they look on the path that ships.
    /// `dequantized()` never reads `nzTile` at all — it recomputes `seen`
    /// incrementally from `nzRow` — and `.dequantized` drops the packed arrays
    /// once the dense matrix exists, so a corrupt `nz_tile` is invisible there
    /// and would only surface under the kernel. A wrong-but-in-range `nz_row`
    /// is worse: the trits still give every weight the right sign, so the model
    /// decodes to plausible magnitudes and produces a plausible transcript with
    /// nothing to trip over. `tile: 0` would divide by zero computing `ntile`
    /// in `phononFiveValueMatmul` (PhononFiveValueKernel.swift). None of this
    /// is a consistency check across the tables — it cannot tell a valid
    /// `nz_row` from another valid one — it only rejects arrays whose shapes
    /// contradict the metadata.
    public static func load(from arrays: [String: MLXArray], name: String,
                            shape: [Int], tile: Int) -> PhononFiveValue? {
        guard let outFeatures = shape.first, let inFeatures = shape.last,
              outFeatures > 0, inFeatures > 0, tile > 0,
              let trits = arrays["\(name).trits"],
              let hiBits = arrays["\(name).hi_bits"],
              let nzTile = arrays["\(name).nz_tile"],
              let nzRow = arrays["\(name).nz_row"],
              let lo = arrays["\(name).lo"],
              let hi = arrays["\(name).hi"],
              trits.ndim == 2, trits.dim(0) == outFeatures,
              trits.dim(1) == (inFeatures + 4) / 5,
              nzTile.ndim == 2, nzTile.dim(0) == outFeatures,
              nzTile.dim(1) == (inFeatures + tile - 1) / tile,
              nzRow.size == outFeatures,
              lo.size == outFeatures, hi.size == outFeatures
        else { return nil }
        return PhononFiveValue(trits: trits, hiBits: hiBits, nzTile: nzTile, nzRow: nzRow,
                               lo: lo, hi: hi, outFeatures: outFeatures,
                               inFeatures: inFeatures, tile: tile)
    }

    /// Straightforward decode, used by the parity gate and as the fallback when
    /// the kernel is disabled. Deliberately simple rather than fast: its job is
    /// to be obviously correct so the kernel has something trustworthy to be
    /// checked against.
    public func dequantized() -> MLXArray {
        let O = outFeatures, I = inFeatures
        let tritBytes = trits.asArray(UInt8.self)
        let bitBytes = hiBits.asArray(UInt8.self)
        let rowBase = nzRow.asArray(UInt32.self)
        let loV = lo.asType(.float32).asArray(Float.self)
        let hiV = hi.asType(.float32).asArray(Float.self)
        let rb = (I + 4) / 5

        var w = [Float](repeating: 0, count: O * I)
        for o in 0..<O {
            var seen = Int(rowBase[o])
            // Bounded by I, never by rb * 5. The unused digits in a row's last
            // byte are written as 0, which decodes as -1 rather than 0 —
            // Fermion's own encoding, kept as-is so the container stays
            // byte-identical to theirs. Reading past column I would therefore
            // produce spurious -1 weights that look like plausible data.
            for i in 0..<I {
                let byte = Int(tritBytes[o * rb + i / 5])
                var code = byte
                for _ in 0..<(i % 5) { code /= 3 }
                let sign = Int(code % 3) - 1        // {-1, 0, +1}
                if sign == 0 { continue }
                let bit = (Int(bitBytes[seen >> 3]) >> (seen & 7)) & 1
                seen += 1
                w[o * I + i] = Float(sign) * (bit == 1 ? hiV[o] : loV[o])
            }
        }
        return MLXArray(w, [O, I]).asType(.float16)
    }
}

// MARK: - Int6

/// A matrix kept as symmetric per-row int6, as Fermion pack the tables that
/// are not worth the five-value format (embedding, joint head, LSTM matrices,
/// projectors, `pre_encode.out`).
///
/// Four 6-bit values share three bytes, little-endian, so a row of `cols`
/// values takes `cols * 3 / 4` bytes. Values are stored offset by 32 and
/// multiplied by one float16 scale per row.
// `@unchecked` for the same reason as `PhononFiveValue`: immutable MLXArrays.
public struct PhononInt6: @unchecked Sendable {
    public let q6: MLXArray         // uint8  [O, cols * 3 / 4]
    public let scale: MLXArray      // float16 [O]
    public let outFeatures: Int
    public let inFeatures: Int

    public init(q6: MLXArray, scale: MLXArray, outFeatures: Int, inFeatures: Int) {
        self.q6 = q6; self.scale = scale
        self.outFeatures = outFeatures; self.inFeatures = inFeatures
    }

    /// Builds one table from the arrays of a converted file, or nil if either
    /// piece is missing. `shape` is the logical shape from the metadata; as for
    /// `PhononFiveValue.load`, its first and last elements are the features.
    ///
    /// Also nil when the packed row width does not match `shape`, since that
    /// would mean the metadata and the tensors describe different things, and
    /// decoding on regardless would shear every row after the first.
    public static func load(from arrays: [String: MLXArray], name: String,
                            shape: [Int]) -> PhononInt6? {
        guard let outFeatures = shape.first, let inFeatures = shape.last,
              inFeatures % 4 == 0,
              let q6 = arrays["\(name).q6"],
              let scale = arrays["\(name).scale"],
              q6.ndim == 2, q6.dim(0) == outFeatures, q6.dim(1) == inFeatures * 3 / 4,
              scale.size == outFeatures
        else { return nil }
        return PhononInt6(q6: q6, scale: scale, outFeatures: outFeatures, inFeatures: inFeatures)
    }

    /// Straightforward decode, in the same spirit as `PhononFiveValue`: plain
    /// enough to be the reference a faster path is checked against.
    public func dequantized() -> MLXArray {
        let O = outFeatures, I = inFeatures
        let bytes = q6.asArray(UInt8.self)
        let scales = scale.asType(.float32).asArray(Float.self)
        let rowBytes = I * 3 / 4

        var w = [Float](repeating: 0, count: O * I)
        for o in 0..<O {
            for group in 0..<(I / 4) {
                let base = o * rowBytes + group * 3
                let packed = UInt32(bytes[base])
                    | (UInt32(bytes[base + 1]) << 8)
                    | (UInt32(bytes[base + 2]) << 16)
                for k in 0..<4 {
                    let u = Int((packed >> UInt32(6 * k)) & 0x3F)
                    w[o * I + group * 4 + k] = Float(u - 32) * scales[o]
                }
            }
        }
        return MLXArray(w, [O, I]).asType(.float16)
    }
}
