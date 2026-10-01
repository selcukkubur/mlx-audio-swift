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

    /// A byte of five trits, unpacked once instead of per weight.
    ///
    /// `tritSigns[b * 5 + k]` is `(b / 3^k) % 3 - 1` — the sign of the k-th
    /// weight packed in byte `b`, the same arithmetic the per-weight division
    /// loop used to do. 256 entries rather than 243 so that a byte above the
    /// valid base-3 range decodes to exactly what the division produced, which
    /// keeps a corrupt file decoding the way it always did rather than trapping
    /// on a table it does not fit.
    private static let tritSigns: [Int8] = {
        var table = [Int8](repeating: 0, count: 256 * 5)
        for b in 0..<256 {
            var code = b
            for k in 0..<5 {
                table[b * 5 + k] = Int8(code % 3) - 1
                code /= 3
            }
        }
        return table
    }()

    /// Expands the block to a dense float16 matrix. This is what loading a
    /// Phonon-2 model spends its time in, and it is the reference the parity
    /// gate and the kernel are both checked against.
    ///
    /// Two things make it fast without making it clever:
    ///
    /// - **Rows are decoded in parallel.** The hi/lo bit of a weight is found by
    ///   counting the non-zeros before it, which reads like a serial scan over
    ///   the whole matrix — but `nzRow[o]` is that count for the start of row
    ///   `o`, already absolute, so each row carries its own entry point and no
    ///   row needs to know what any other row decoded. (`nzTile` would cut the
    ///   same seam finer, inside a row; with 1024 to 4096 rows per block there
    ///   is nothing to gain by it, so it stays unread here.)
    /// - **A byte of trits is unpacked once**, through `tritSigns`, rather than
    ///   re-divided by 3 up to four times for each of the five weights in it.
    ///
    /// Two further changes look obvious, were tried, and are deliberately not
    /// here. Writing float16 straight into the buffer rounds nothing — every
    /// value is `±lo` or `±hi`, which are float16 already — but measured 0.89 s
    /// against 0.69 s on otherwise identical code. Allocating the staging buffer
    /// with `UnsafeMutablePointer.allocate` rather than this `[Float]` cost 8 MB
    /// of peak footprint (1.511 GiB against 1.503) and was not faster either. Neither is an obstacle to a later attempt;
    /// both need a measurement rather than a glance.
    ///
    /// Bounded by `I`, never by `rb * 5`. The unused digits in a row's last byte
    /// are written as 0, which decodes as -1 rather than 0 — Fermion's own
    /// encoding, kept as-is so the container stays byte-identical to theirs.
    /// Reading past column `I` would therefore produce spurious -1 weights that
    /// look like plausible data.
    public func dequantized() -> MLXArray {
        let O = outFeatures, I = inFeatures
        let tritBytes = trits.asArray(UInt8.self)
        let bitBytes = hiBits.asArray(UInt8.self)
        let rowBase = nzRow.asArray(UInt32.self)
        let loV = lo.asType(.float32).asArray(Float.self)
        let hiV = hi.asType(.float32).asArray(Float.self)
        let rb = (I + 4) / 5

        var w = [Float](repeating: 0, count: O * I)
        w.withUnsafeMutableBufferPointer { wb in
            tritBytes.withUnsafeBufferPointer { tritBuf in
                Self.tritSigns.withUnsafeBufferPointer { signBuf in
                    // Base addresses rather than the buffers themselves:
                    // `UnsafePointer` is Sendable and `UnsafeBufferPointer` is
                    // not. Every index into these three is bounded by something
                    // already checked — `trits` is `[O, rb]` by `load`, the sign
                    // table is indexed by a byte, the output by `o * I + i` —
                    // which is exactly why they can be read unchecked and
                    // `hiBits` below cannot.
                    let out = wb.baseAddress!
                    let tb = tritBuf.baseAddress!
                    let signs = signBuf.baseAddress!
                    // One chunk per 64 rows: enough chunks that every thread
                    // gets work on the smallest blocks here (O = 1024), large
                    // enough that the dispatch is noise against the rows it
                    // hands over.
                    let rowsPerChunk = 64
                    let chunks = (O + rowsPerChunk - 1) / rowsPerChunk
                    DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                        let rowEnd = min(O, (chunk + 1) * rowsPerChunk)
                        for o in (chunk * rowsPerChunk)..<rowEnd {
                            var seen = Int(rowBase[o])
                            let loRow = loV[o], hiRow = hiV[o]
                            let rowOut = out + o * I
                            var byteIndex = o * rb
                            var i = 0
                            while i < I {
                                let base = Int(tb[byteIndex]) * 5
                                for k in 0..<min(5, I - i) {
                                    let sign = signs[base + k]
                                    if sign == 0 { continue }
                                    // `bitBytes` here, not a raw pointer, and
                                    // deliberately. `seen` starts at `nzRow[o]`
                                    // and advances once per non-zero, both of
                                    // which come from the file, and `hi_bits` is
                                    // the one packed array whose length nothing
                                    // can check at load: its length is the
                                    // non-zero count, and that is only known by
                                    // doing this decode. Swift's bounds check is
                                    // what stands between a malformed download
                                    // and an out-of-bounds read, and it measured
                                    // free behind the dependent load.
                                    let bit = (Int(bitBytes[seen >> 3]) >> (seen & 7)) & 1
                                    seen += 1
                                    rowOut[i + k] = Float(sign) * (bit == 1 ? hiRow : loRow)
                                }
                                byteIndex += 1
                                i += 5
                            }
                        }
                    }
                }
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
