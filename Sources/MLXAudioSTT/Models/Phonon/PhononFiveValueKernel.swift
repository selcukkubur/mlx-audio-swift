import Foundation
import MLX
import MLXFast

/// `y = x · Wᵀ` with W held in Fermion's five-value packing, never expanded.
///
/// Keeping the packed bytes is the point: the model stays ~189 MB resident
/// instead of 1.30 GB dequantised.
///
/// One threadgroup per (output row, time step). The threads of a group stride
/// across a tile of the row, each accumulating its share, and the partial sums
/// are reduced at the end.
///
/// `nzTile` is what makes the bit-plane addressable: Fermion index it by
/// cumulative non-zero count, so without a precomputed per-tile offset each
/// thread would have to scan the whole preceding row to find its bit.
///
/// This is the deliberately naive version. Counting the non-zeros before an
/// element rescans the tile from its start, O(TILE) per element. It exists to
/// be correct first; a cooperative unpack with a local prefix scan is the
/// obvious speed-up and belongs after the parity gate, not before it.
let phononKernelHeader = """
    // Digit `k` of a base-3 byte, mapped to a sign in {-1, 0, +1}.
    inline int phonon_sign(uint byte, uint k) {
        for (uint d = 0; d < k; ++d) byte /= 3u;
        return (int)(byte % 3u) - 1;
    }
"""

let phononKernelSource = """
    uint tid = thread_position_in_threadgroup.x;
    uint nthreads = threads_per_threadgroup.x;
    // Grid is (O groups, T groups): x picks the weight row, y picks the input
    // time step. Both are uniform across a group, so the early return and the
    // barrier below are safe.
    uint row = threadgroup_position_in_grid.x;
    uint step = threadgroup_position_in_grid.y;
    if (row >= (uint)O) return;

    threadgroup float partial[256];

    uint I_ = (uint)I;
    uint rb = (I_ + 4u) / 5u;
    float loV = (float)lo[row];
    float hiV = (float)hi[row];
    uint xBase = step * I_;
    float acc = 0.0f;

    for (uint t0 = 0; t0 < I_; t0 += (uint)TILE) {
        uint tileIdx = t0 / (uint)TILE;
        uint seen = nz_row[row] + (uint)nz_tile[row * (uint)NTILE + tileIdx];
        uint tileEnd = min(t0 + (uint)TILE, I_);

        // Bounded by I_, never by rb * 5: the padding digits in a row's last
        // byte are 0, which decodes as -1, and would leak into the sum.
        for (uint i = t0 + tid; i < tileEnd; i += nthreads) {
            int sgn = phonon_sign((uint)trits[row * rb + i / 5u], i % 5u);
            if (sgn == 0) continue;

            uint before = 0;
            for (uint j = t0; j < i; ++j) {
                if (phonon_sign((uint)trits[row * rb + j / 5u], j % 5u) != 0) before++;
            }
            uint bitIdx = seen + before;
            uint bit = ((uint)hi_bits[bitIdx >> 3] >> (bitIdx & 7u)) & 1u;
            acc += (float)sgn * (bit == 1u ? hiV : loV) * (float)x[xBase + i];
        }
    }

    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint k = 0; k < nthreads; ++k) total += partial[k];
        out[step * (uint)O + row] = (half)total;
    }
"""

/// The kernel object, built once for the process.
///
/// It is a description of the kernel, not the compiled pipeline — MLX caches
/// that behind it — but rebuilding the description allocated an object and
/// re-hashed the source text on every matmul, and one transcription makes tens
/// of thousands of them.
///
/// `nonisolated(unsafe)` because the type is not declared `Sendable`. It is
/// immutable once built and MLX guards its own kernel cache, so the only thing
/// being asserted here is that a `let` nobody writes can be read from more than
/// one task.
nonisolated(unsafe) let phononFiveValueKernel = MLXFast.metalKernel(
    name: "phonon_five_value",
    inputNames: ["x", "trits", "hi_bits", "nz_tile", "nz_row", "lo", "hi"],
    outputNames: ["out"],
    source: phononKernelSource,
    header: phononKernelHeader
)

/// `x` is `[..., I]`; the result is `[..., O]`, float16.
///
/// Every call site is batched over time, so the leading dimensions are
/// flattened to `T` and the kernel runs over `[T, I]`. A single vector is just
/// `T == 1`, not a separate path.
public func phononFiveValueMatmul(_ x: MLXArray, _ w: PhononFiveValue) -> MLXArray {
    precondition(x.dim(-1) == w.inFeatures,
                 "phononFiveValueMatmul: x has \(x.dim(-1)) features, weight expects \(w.inFeatures)")
    let leading = Array(x.shape.dropLast())
    let steps = leading.reduce(1, *)
    let x2 = x.reshaped([steps, w.inFeatures])

    let ntile = (w.inFeatures + w.tile - 1) / w.tile
    let group = min(w.tile, 256)
    let out = phononFiveValueKernel(
        [x2, w.trits, w.hiBits, w.nzTile, w.nzRow, w.lo, w.hi],
        template: [("O", w.outFeatures), ("I", w.inFeatures),
                   ("TILE", w.tile), ("NTILE", ntile)],
        // MLX's grid counts threads, not groups, so a group per row is
        // O * group threads wide.
        grid: (w.outFeatures * group, steps, 1),
        threadGroup: (group, 1, 1),
        outputShapes: [[steps, w.outFeatures]],
        outputDTypes: [.float16]
    )[0]
    return out.reshaped(leading + [w.outFeatures])
}
