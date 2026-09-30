// Decode parity gate for the Phonon-2 port: proves this repo's unpacking of the
// converted file matches Fermion's own reader, tensor by tensor.
//
// Usage: PhononParity [--all] <converted-model-dir> <reference-dump-dir>
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

// --all demands a reference for every five_value module in the file rather than
// the sampled set. It needs a dump made with `phonon2_parity.py --all` and is
// the pre-release run; the default is the quick one.
var args = Array(CommandLine.arguments.dropFirst())
let requireAll = args.contains("--all")
args.removeAll { $0 == "--all" }
guard args.count == 2 else {
    fail("usage: PhononParity [--all] <converted-model-dir> <reference-dump-dir>")
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

print("\nchecked \(checked.five) of \(packed.count) five_value (\(shapesChecked.count) of \(shapesInFile.count) shapes) and \(checked.int6) of \(int6.count) int6 modules\(requireAll ? " [--all]" : "")")
if failures > 0 {
    print("\nGATE FAILED: \(failures) problem(s)")
    print(diagnosis)
    exit(1)
}
print("GATE PASSED: every module at or above \(bar)")
