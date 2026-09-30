// Decode parity gate for the Phonon-2 port: proves this repo's unpacking of the
// converted file matches Fermion's own reader, tensor by tensor.
//
// Usage: PhononParity <converted-model-dir> <reference-dump-dir>
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
// table in the file has no reference. Nothing here short-circuits: every figure
// is printed pass or fail.

import Foundation
import MLX
import MLXAudioSTT

/// Both sides claim to produce the SAME numbers from the same bytes, so this is
/// an equality check with a tolerance, not a similarity check. The only slack
/// it needs is float16 rounding of the int6 tables, which costs ~1e-7. Anything
/// under 0.9999 is a decode bug, and a wrong trit order, bit-plane offset or
/// row/column swap all land far below it.
let bar = 0.9999

/// The cosine bar alone is blind to a localised fault: a single wrong trit byte
/// changes five weights out of four million and still reads 0.99999999. Fermion's
/// reader and ours decode the same bits with the same float16 arithmetic, so for
/// five_value the two must agree element for element, and any difference is a
/// bug. The int6 reference is float32 while we emit float16, so those may differ
/// by float16 rounding (half an ulp, 2^-11 relative) and no more.
func mismatches(_ a: [Float], _ b: [Float], exact: Bool) -> (count: Int, maxAbs: Float) {
    var count = 0
    var maxAbs: Float = 0
    for i in 0..<a.count {
        let d = abs(a[i] - b[i])
        // NaN compares false on both sides of `>`, so test it explicitly.
        if d.isNaN { count += 1; maxAbs = .infinity; continue }
        maxAbs = max(maxAbs, d)
        let slack: Float = exact ? 0 : abs(b[i]) * 0x1p-10 + 1e-7
        if d > slack { count += 1 }
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
  cosine fine but mismatched > 0         a localised fault: one wrong byte, one nz_row
  nz tables bad > 0                      nz_tile / nz_row disagree with the reference's non-zero
                                         pattern; decoding may still be right, a kernel would not be
Do not proceed to the kernel until every module passes: a kernel checked
against a wrong decode certifies nothing.
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

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

let args = CommandLine.arguments
guard args.count == 3 else { fail("usage: PhononParity <converted-model-dir> <reference-dump-dir>") }
let modelDir = URL(fileURLWithPath: args[1]), refDir = URL(fileURLWithPath: args[2])

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
    let diff = mismatches(a, b, exact: kind == "five_value")
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
}

// A gate that quietly skips a table certifies nothing about it. There are only
// nine, so all of them are required; five_value is a sample by design.
for entry in int6 where !coveredInt6.contains(entry.name) {
    print("FAIL  int6  \(entry.name): in the file but absent from the reference dump")
    failures += 1
}

print("\nchecked \(checked.five) of \(packed.count) five_value and \(checked.int6) of \(int6.count) int6 modules")
if failures > 0 {
    print("\nGATE FAILED: \(failures) problem(s)")
    print(diagnosis)
    exit(1)
}
print("GATE PASSED: every module at or above \(bar)")
