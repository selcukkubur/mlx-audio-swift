// Parity gate for the Phonon-2 port. Stage one proves this repo's unpacking of
// the converted file matches Fermion's own reader, tensor by tensor. Stage two
// (`--kernel`) proves the Metal kernel computes the same product as a dense
// matmul against that decode.
//
// Usage: PhononParity [--all] [--kernel] <converted-model-dir> <reference-dump-dir>
//
// The reference dump comes from Scripts/phonon2_parity.py in the app repo, which
// decodes Fermion's original container with Fermion's own code. That is the
// point of the gate: a previous port in this codebase passed every shape check,
// a spot check of weight values and an all-ones convolution test, and
// transcribed nonsense, because both sides of every one of those comparisons
// read the same wrong bytes. This one compares against an implementation that
// shares no code with ours.
//
// Exits non-zero if any module is below the bar or differs element-wise beyond
// float16 rounding, if a reference has no module in the file, or if any int6
// table in the file has no reference. With `--kernel` it also exits non-zero if
// the kernel disagrees with the decode on any module, on the synthetic ragged
// case, or if it checked nothing. Nothing here short-circuits: every figure is
// printed pass or fail.

import Foundation
import MLX
import MLXAudioSTT

/// Both sides claim to produce the SAME numbers from the same bytes, so this is
/// an equality check with a tolerance, not a similarity check. Anything
/// under 0.9999 is a decode bug, and a wrong trit order, bit-plane offset or
/// row/column swap all land far below it.
let bar = 0.9999

/// The cosine bar alone is blind to a localised fault, measured on real
/// corruptions: a flipped trit changes one byte but mis-aligns `hi_bits` for the
/// rest of that row, giving a couple of hundred wrong elements and a cosine of
/// about 0.99997, still above the bar; a wrong `nz_tile` entry, or one wrong int6
/// byte, reads 1.0 and 0.99999998. So elements are compared one by one, and
/// exactly. Five_value is float16 on both sides. The int6 reference is float32
/// while we emit float16, so it is rounded to float16 first, which is what the
/// port's own arithmetic does; any slack wider than that (half an ulp is
/// 2^-11 relative) lets a one-ulp corruption of a scale through.
func mismatches(_ a: [Float], _ b: [Float]) -> (count: Int, maxAbs: Float) {
    var count = 0
    var maxAbs: Float = 0
    for i in 0..<a.count {
        let want = Float(Float16(b[i]))
        // NaN compares unequal to everything, itself included, so it counts.
        if a[i] == want { continue }
        count += 1
        maxAbs = max(maxAbs, abs(a[i] - want))
        if a[i].isNaN || want.isNaN { maxAbs = .infinity }
    }
    return (count, maxAbs)
}

let diagnosis = """

How to read a failure:
  near zero on every tensor              trit order reversed (digits are little-endian by power of three)
  correct for row 0, wrong afterwards    nz_row off by one, or hi_bits packed with the wrong bitorder
  mirrored values                        lo/hi swapped
  right magnitudes, wrong signs          sign = code - 1 inverted (cosine near -1)
  int6 only                              6-bit groups read big-endian, or the +32 bias dropped
  cosine fine but mismatched > 0         a localised fault: one wrong byte, one nz_row, one int6 scale
  coverage failures                      the dump is truncated or wrong, not the decode: regenerate it
                                         with Scripts/phonon2_parity.py (add --all for --all)
  nz tables bad > 0                      nz_tile / nz_row disagree with the reference's non-zero
                                         pattern; decoding may still be right, a kernel would not be
Do not proceed to the kernel until every module passes: a kernel checked
against a wrong decode certifies nothing.

How to read a kernel failure (only meaningful once the decode stage is clean):
  cosine below 0.99                      a bug, not rounding
  cosine fine, elements over tolerance   a localised fault, the kind a cosine bar cannot see: one
                                         wrong nz_tile entry (dequantized() never reads it), one
                                         thread reading the wrong hi bit
  only the synthetic case fails          the partial-final-tile or trit-padding branch: a loop
                                         bounded by rb * 5 instead of I. No real module reaches it
  fails at T=7, passes at T=1            the time axis: step indexing of x or out
"""

struct ReferenceEntry: Decodable {
    let key: String
    let kind: String?           // absent in dumps written before int6 was added
    let module: String          // the metadata `name` (block prefix) it corresponds to
    let shape: [Int]
}

func cosine(_ a: [Float], _ b: [Float]) -> Double {
    precondition(a.count == b.count, "shape mismatch: \(a.count) vs \(b.count)")
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<a.count {
        dot += Double(a[i]) * Double(b[i])
        na += Double(a[i]) * Double(a[i])
        nb += Double(b[i]) * Double(b[i])
    }
    guard na > 0, nb > 0 else { return 0 }
    return dot / (na.squareRoot() * nb.squareRoot())
}

/// Counts the entries of `nz_tile` and `nz_row` that disagree with the ones the
/// reference's own non-zero pattern implies.
///
/// `dequantized()` never reads `nz_tile`; only a kernel will, and it will trust
/// it blindly. A wrong entry would therefore pass every check of the decoded
/// values and still send a GPU thread to the wrong bit. The reference supplies
/// the pattern independently of both tables, so this pins them without a
/// kernel. (A row whose `lo` is exactly zero would hide non-zeros from that
/// pattern; none of the sampled rows does, and a failure here says so first.)
func tableMismatches(_ module: PhononFiveValue, reference b: [Float]) -> Int {
    let O = module.outFeatures, I = module.inFeatures, tile = module.tile
    let ntile = (I + tile - 1) / tile
    let tiles = module.nzTile.asArray(UInt16.self)
    let rows = module.nzRow.asArray(UInt32.self)
    guard tiles.count == O * ntile, rows.count == O else { return O * ntile + O }
    var bad = 0
    var runningRows = 0
    for o in 0..<O {
        if Int(rows[o]) != runningRows { bad += 1 }
        var seen = 0
        for t in 0..<ntile {
            if Int(tiles[o * ntile + t]) != seen { bad += 1 }
            for i in (t * tile)..<min((t + 1) * tile, I) where b[o * I + i] != 0 { seen += 1 }
        }
        runningRows += seen
    }
    return bad
}

func flat(_ x: MLXArray) -> [Float] {
    x.asType(.float32).reshaped([-1]).asArray(Float.self)
}

/// The kernel accumulates in float32 and the dense reference is float16 in and
/// out, so a little divergence is arithmetic, not error. Looser than the decode
/// stage on purpose. Below 0.99 is a bug outright.
let kernelBar = 0.999

/// Every call site in the model is batched over time, so a lone vector would
/// certify a kernel that fails at integration. 7 is deliberately not a power of
/// two or a multiple of anything the kernel tiles by.
let kernelSteps = [1, 7]

/// Cosine over a whole output is close to blind to a localised fault: Task 3
/// measured it catching 3 of 11 real corruptions, and one wrong output element
/// among thousands moves it by less than the bar's slack. So every output
/// element is also held to a tolerance of its own.
///
/// Both sides round a float32 sum to float16, so two correct results can differ
/// by one ulp (each is within half an ulp of the exact sum, and they land on
/// different neighbours when it sits near a rounding midpoint). Half an ulp
/// would fail correct results at random. On top of that, the two float32 sums
/// add the same terms in different orders; their gap is bounded by the
/// rounding error of a sum of I terms, which grows like sqrt(I) * 2^-24 times
/// the sum of the terms' magnitudes (not the sum itself, which cancels). 2^-23
/// gives that a factor of two of slack. Measured against float16 rounding it is
/// about four orders smaller, so it only matters where an output nearly cancels.
///
/// The tolerance is taken per element from that element's own magnitude rather
/// than once from the largest output. A single bound sized by the largest
/// output would be several ulps wide at every smaller one and would let an
/// error on a small output through. Per-element is never looser than the
/// global bound, which is reported alongside.
let accumulationEps: Float = 0x1p-23

struct KernelResult {
    var cosine = 0.0
    var maxAbs: Float = 0
    var globalTolerance: Float = 0      // the bound if sized once, at the largest output
    var worstRatio = 0.0                // max over elements of |diff| / that element's tolerance
    var exceeding = 0                   // elements over their own tolerance
    var nonFinite = 0
    var maxOutput: Float = 0
    var ok: Bool { cosine >= kernelBar && exceeding == 0 && nonFinite == 0 }
}

/// Runs the kernel and the dense product on the same fixed-seed `x` of `steps`
/// rows and compares them. `dense` is `[O, I]` float16.
func kernelCompare(_ w: PhononFiveValue, dense: MLXArray, steps: Int, seed: UInt64) -> KernelResult {
    let O = w.outFeatures, I = w.inFeatures
    let x = MLXRandom.normal([steps, I], dtype: .float16, key: MLXRandom.key(seed))
    let got = phononFiveValueMatmul(x, w)
    let want = matmul(x, dense.T)
    // Sum of |term| per output, for the accumulation allowance.
    let magnitude = matmul(abs(x).asType(.float32), abs(dense).asType(.float32).T)
    eval(got, want, magnitude)

    var result = KernelResult()
    guard got.shape == [steps, O], want.shape == [steps, O] else {
        result.nonFinite = Int.max      // a wrong shape is not comparable; fail it
        return result
    }
    let g = flat(got), r = flat(want), l1 = flat(magnitude)
    result.cosine = cosine(g, r)
    let allowanceScale = accumulationEps * Float(I).squareRoot()
    for i in 0..<g.count {
        // Written so NaN fails: any comparison with NaN is false.
        guard g[i].isFinite, r[i].isFinite else { result.nonFinite += 1; continue }
        let diff = abs(g[i] - r[i])
        result.maxAbs = max(result.maxAbs, diff)
        result.maxOutput = max(result.maxOutput, abs(r[i]))
        let ulp = Float(Float16(max(abs(g[i]), abs(r[i]))).ulp)
        let tolerance = ulp + allowanceScale * l1[i]
        result.worstRatio = max(result.worstRatio, Double(diff / tolerance))
        if !(diff <= tolerance) { result.exceeding += 1 }
    }
    let worstAllowance = allowanceScale * (l1.max() ?? 0)
    result.globalTolerance = Float(Float16(result.maxOutput).ulp) + worstAllowance
    return result
}

func kernelLine(_ label: String, _ shape: [Int], steps: Int, _ r: KernelResult) -> String {
    var line = "\(r.ok ? "PASS" : "FAIL")  kernel  \(label)  \(shape)  T=\(steps)"
    line += "  cosine \(String(format: "%.8f", r.cosine))"
    line += "  max|diff| \(r.maxAbs) (bound at the largest output, \(r.maxOutput): \(r.globalTolerance))"
    // Over 100% is a failure. Healthy runs sit in the 90s at times: two correct
    // roundings on adjacent floats use nearly all of the one-ulp allowance.
    line += "  worst element \(String(format: "%.0f", r.worstRatio * 100))% of its tolerance"
    line += "  over tolerance \(r.exceeding)"
    if r.nonFinite > 0 { line += "  non-finite/unshaped \(r.nonFinite)" }
    return line
}

// MARK: - Synthetic ragged module

/// SplitMix64: a fixed, dependency-free stream so the synthetic module is the
/// same on every run and every machine.
struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
}

/// A five-value module packed here, by hand, with a width no real module has.
///
/// Every real module has I of 1024 or 4096, multiples of the tile (128), so the
/// kernel's partial-final-tile branch and the trit padding in a row's last byte
/// are unreachable with this model and could regress unnoticed. I = 999 is
/// neither a multiple of 128 nor of 5: the last tile is 103 wide, and the last
/// byte of every row carries one padding digit. Padding digits are written as 0,
/// which decodes as -1 (Fermion's own encoding), so any loop that reads them
/// adds a spurious weight.
///
/// The dense matrix the packing was built from is returned too, so the decode is
/// checked against what was intended and not just against the kernel.
func syntheticFiveValue(outFeatures O: Int, inFeatures I: Int, tile: Int, seed: UInt64)
    -> (module: PhononFiveValue, intended: MLXArray)
{
    var rng = SplitMix64(state: seed)
    let rb = (I + 4) / 5
    let ntile = (I + tile - 1) / tile
    let pow3: [Int] = [1, 3, 9, 27, 81]

    var lo = [Float16](), hi = [Float16]()
    for _ in 0..<O {
        let l = Float16(0.01 + 0.04 * rng.unit())
        lo.append(l)
        hi.append(Float16(Float(l) * (2 + 2 * rng.unit())))
    }

    var dense = [Float](repeating: 0, count: O * I)
    var tritBytes = [UInt8](repeating: 0, count: O * rb)
    var bits = [Bool]()
    var nzTile = [UInt16](repeating: 0, count: O * ntile)
    var nzRow = [UInt32](repeating: 0, count: O)
    for o in 0..<O {
        nzRow[o] = UInt32(bits.count)
        var inRow = 0
        for i in 0..<I {
            if i % tile == 0 { nzTile[o * ntile + i / tile] = UInt16(inRow) }
            let sign = rng.unit() < 0.35 ? 0 : (rng.next() & 1 == 0 ? -1 : 1)
            tritBytes[o * rb + i / 5] += UInt8((sign + 1) * pow3[i % 5])
            guard sign != 0 else { continue }
            let big = rng.next() & 1 == 1
            bits.append(big)
            inRow += 1
            dense[o * I + i] = Float(sign) * Float(big ? hi[o] : lo[o])
        }
    }
    var hiBytes = [UInt8](repeating: 0, count: (bits.count + 7) / 8)
    for (n, big) in bits.enumerated() where big { hiBytes[n >> 3] |= UInt8(1 << (n & 7)) }

    let module = PhononFiveValue(
        trits: MLXArray(tritBytes, [O, rb]), hiBits: MLXArray(hiBytes),
        nzTile: MLXArray(nzTile, [O, ntile]), nzRow: MLXArray(nzRow),
        lo: MLXArray(lo), hi: MLXArray(hi),
        outFeatures: O, inFeatures: I, tile: tile)
    return (module, MLXArray(dense, [O, I]).asType(.float16))
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

// --all demands a reference for every five_value module in the file rather than
// the sampled set. It needs a dump made with `phonon2_parity.py --all` and is
// the pre-release run; the default is the quick one.
var args = Array(CommandLine.arguments.dropFirst())
let requireAll = args.contains("--all")
let runKernel = args.contains("--kernel")
args.removeAll { $0 == "--all" || $0 == "--kernel" }
guard args.count == 2 else {
    fail("usage: PhononParity [--all] [--kernel] <converted-model-dir> <reference-dump-dir>")
}
let modelDir = URL(fileURLWithPath: args[0]), refDir = URL(fileURLWithPath: args[1])

let arrays: [String: MLXArray]
let metadata: [String: String]
let reference: [String: MLXArray]
let listing: [ReferenceEntry]
do {
    (arrays, metadata) = try loadArraysAndMetadata(url: modelDir.appendingPathComponent("model.safetensors"))
    reference = try loadArrays(url: refDir.appendingPathComponent("phonon2_parity.safetensors"))
    listing = try JSONDecoder().decode(
        [ReferenceEntry].self, from: Data(contentsOf: refDir.appendingPathComponent("picked.json")))
} catch {
    fail("could not read inputs: \(error)")
}

guard let packedJSON = metadata["phonon_packed"], let int6JSON = metadata["phonon_int6"] else {
    fail("model.safetensors has no phonon_packed / phonon_int6 metadata; is this a converted file?")
}
let packed: [PhononPackedEntry]
let int6: [PhononInt6Entry]
do {
    packed = try JSONDecoder().decode([PhononPackedEntry].self, from: Data(packedJSON.utf8))
    int6 = try JSONDecoder().decode([PhononInt6Entry].self, from: Data(int6JSON.utf8))
} catch {
    fail("could not parse phonon metadata: \(error)")
}

var failures = 0
var checked: (five: Int, int6: Int) = (0, 0)
var coveredInt6 = Set<String>()
var coveredFive = Set<String>()
// Kernel stage bookkeeping. `kernelRuns` counts (module, T) pairs so a module
// that quietly skipped one of the batch sizes is caught, not just one that
// skipped both.
var coveredKernel = Set<String>()
var kernelRuns = 0

print("bar \(bar); \(listing.count) reference tensors, model has \(packed.count) five_value + \(int6.count) int6 modules\n")

for ref in listing {
    let kind = ref.kind ?? "five_value"
    guard let want = reference[ref.key] else {
        print("FAIL  \(kind)  \(ref.module): key \(ref.key) missing from the reference dump")
        failures += 1
        continue
    }

    let decoded: MLXArray
    var fiveValue: PhononFiveValue?
    switch kind {
    case "five_value":
        guard let entry = packed.first(where: { $0.name == ref.module }),
              let module = PhononFiveValue.load(
                  from: arrays, name: entry.name, shape: entry.shape, tile: entry.tile)
        else {
            print("FAIL  five_value  \(ref.module): not in phonon_packed, or a piece of it is missing")
            failures += 1
            continue
        }
        decoded = module.dequantized()
        fiveValue = module
        coveredFive.insert(entry.name)
        checked.five += 1
    case "int6":
        guard let entry = int6.first(where: { $0.name == ref.module }),
              let module = PhononInt6.load(from: arrays, name: entry.name, shape: entry.shape)
        else {
            print("FAIL  int6  \(ref.module): not in phonon_int6, or a piece of it is missing")
            failures += 1
            continue
        }
        decoded = module.dequantized()
        coveredInt6.insert(entry.name)
        checked.int6 += 1
    default:
        print("FAIL  \(kind)  \(ref.module): unknown reference kind")
        failures += 1
        continue
    }

    // Compared element for element, so a transposed decode of a square matrix
    // cannot slip through on shape alone; the shapes just have to agree in size.
    guard decoded.size == want.size else {
        print("FAIL  \(kind)  \(ref.module): \(decoded.shape) decoded vs \(want.shape) reference")
        failures += 1
        continue
    }
    let a = flat(decoded), b = flat(want)
    let cos = cosine(a, b)
    // Written so NaN fails: `cos < bar` would be false for NaN and pass it.
    let diff = mismatches(a, b)
    let tableBad = fiveValue.map { tableMismatches($0, reference: b) } ?? 0
    let ok = cos >= bar && diff.count == 0 && tableBad == 0
    if !ok { failures += 1 }

    var line = "\(ok ? "PASS" : "FAIL")  \(kind)  \(ref.module)  \(ref.shape)  cosine \(String(format: "%.8f", cos))"
    line += "  mismatched \(diff.count)/\(a.count)  max|diff| \(diff.maxAbs)"
    if fiveValue != nil { line += "  nz tables bad \(tableBad)" }
    if !ok {
        // Row 0 is the diagnostic that separates "everything wrong" from "wrong
        // once the running non-zero count matters".
        let cols = ref.shape.last ?? 1
        let row0 = cosine(Array(a[0..<cols]), Array(b[0..<cols]))
        line += "  [row 0 cosine \(String(format: "%.6f", row0))]"
    }
    print(line)

    // Run against the decode just verified, while it is in hand. Recomputing it
    // later would cost a second CPU decode of every module.
    if runKernel, let module = fiveValue {
        for (n, steps) in kernelSteps.enumerated() {
            let seed = 20_260_930 &+ UInt64(steps) &* 1_000_003 &+ UInt64(coveredKernel.count) &* 7 &+ UInt64(n)
            let result = kernelCompare(module, dense: decoded, steps: steps, seed: seed)
            if !result.ok { failures += 1 }
            print(kernelLine(ref.module, ref.shape, steps: steps, result))
            kernelRuns += 1
        }
        coveredKernel.insert(ref.module)
    }
}

// A gate that quietly skips a table certifies nothing about it, and one whose
// dump was truncated would otherwise pass having checked nothing. There are only
// nine int6 tables, so all are required.
for entry in int6 where !coveredInt6.contains(entry.name) {
    print("FAIL  int6  \(entry.name): in the file but absent from the reference dump")
    failures += 1
}

// For five_value a sample is enough by design, but it must span every distinct
// geometry in the file: the shapes are the fault classes that matter (a kernel
// tile bug shows on one shape and not another), and a dump that lost a shape
// class is exactly the silent regression to catch. `[O, 1, I]` pointwise
// convolutions count by their `[O, I]`, the geometry a kernel sees.
func shapeName(_ entry: PhononPackedEntry) -> String {
    "\(entry.shape.first ?? 0)x\(entry.shape.last ?? 0)"
}
let shapesInFile = Set(packed.map(shapeName))
let shapesChecked = Set(packed.filter { coveredFive.contains($0.name) }.map(shapeName))
if packed.isEmpty || checked.five == 0 {
    print("FAIL  coverage: no five_value module was checked, so nothing about the packed weights is certified")
    failures += 1
}
let unchecked = shapesInFile.subtracting(shapesChecked).sorted()
if !unchecked.isEmpty {
    print("FAIL  coverage: \(unchecked.count) of \(shapesInFile.count) five_value shapes unchecked: \(unchecked.joined(separator: ", "))")
    failures += 1
}
if requireAll {
    let missing = packed.filter { !coveredFive.contains($0.name) }
    if !missing.isEmpty {
        print("FAIL  coverage: --all, but \(missing.count) of \(packed.count) five_value modules have no reference, first: \(missing[0].name)")
        failures += 1
    }
}

if runKernel {
    // The ragged case runs with every --kernel, not behind its own flag: the
    // branch it covers is unreachable from the real model.
    let O = 32, I = 999, tile = 128
    let (module, intended) = syntheticFiveValue(outFeatures: O, inFeatures: I, tile: tile, seed: 0x5EED_999)
    precondition(I % tile != 0 && I % 5 != 0, "the synthetic case must be ragged in both ways")
    let decodeDiff = mismatches(flat(module.dequantized()), flat(intended))
    let decodeOK = decodeDiff.count == 0
    if !decodeOK { failures += 1 }
    print("\n\(decodeOK ? "PASS" : "FAIL")  synthetic  [\(O), \(I)]  dequantized() vs the matrix it was packed from  mismatched \(decodeDiff.count)/\(O * I)  max|diff| \(decodeDiff.maxAbs)")
    var syntheticRuns = 0
    for steps in kernelSteps {
        let result = kernelCompare(module, dense: module.dequantized(), steps: steps, seed: 999 &+ UInt64(steps))
        if !result.ok { failures += 1 }
        print(kernelLine("synthetic I=999 (tile \(tile), last tile \(I % tile) wide)", [O, I], steps: steps, result))
        syntheticRuns += 1
    }
    if syntheticRuns != kernelSteps.count {
        print("FAIL  coverage: synthetic case ran \(syntheticRuns) of \(kernelSteps.count) batch sizes")
        failures += 1
    }

    // Same standard as the decode stage: a stage that checked nothing has
    // certified nothing, and must not read as a pass.
    if coveredKernel.isEmpty {
        print("FAIL  coverage: --kernel, but no module was kernel-checked, so nothing about the kernel is certified")
        failures += 1
    }
    if coveredKernel.count != checked.five || kernelRuns != checked.five * kernelSteps.count {
        print("FAIL  coverage: \(checked.five) five_value modules decoded but \(coveredKernel.count) kernel-checked (\(kernelRuns) of \(checked.five * kernelSteps.count) runs)")
        failures += 1
    }
    let kernelShapes = Set(packed.filter { coveredKernel.contains($0.name) }.map(shapeName))
    let kernelUnchecked = shapesInFile.subtracting(kernelShapes).sorted()
    if !kernelUnchecked.isEmpty {
        print("FAIL  coverage: \(kernelUnchecked.count) of \(shapesInFile.count) shapes never reached the kernel: \(kernelUnchecked.joined(separator: ", "))")
        failures += 1
    }
    print("\nkernel: \(coveredKernel.count) modules x T in \(kernelSteps) (\(kernelShapes.count) of \(shapesInFile.count) shapes) plus synthetic I=\(I); bar \(kernelBar)")
}

print("\nchecked \(checked.five) of \(packed.count) five_value (\(shapesChecked.count) of \(shapesInFile.count) shapes) and \(checked.int6) of \(int6.count) int6 modules\(requireAll ? " [--all]" : "")\(runKernel ? " [--kernel]" : "")")
if failures > 0 {
    print("\nGATE FAILED: \(failures) problem(s)")
    print(diagnosis)
    exit(1)
}
print("GATE PASSED: every module at or above \(bar)\(runKernel ? "; kernel at or above \(kernelBar) and within tolerance" : "")")
