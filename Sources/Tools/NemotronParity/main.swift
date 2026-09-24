// Parity-gate harness for the Nemotron 3 Diarization port. Loads the
// checkpoint and the same fixed-seed audio a Python reference dump produced
// (see nemotron_parity.py, kept alongside this port's docs, not shipped),
// runs the Swift port over each stage, and reports cosine similarity
// against the reference arrays.
//
// Usage: NemotronParity <model-dir> <parity-data-dir> [speech-wav-path] [speech-reference-npy-path]
//
// The optional trailing pair makes the real-speech streaming figures in
// this model's README independently reproducible with this repo's own
// tooling, not just quoted from an external run: point them at any 16 kHz
// mono 16-bit PCM WAV file and the matching reference dump that
// `nemotron_parity.py <model-dir> <out-dir> <wav-path>` produces alongside
// it (`speech_streaming_probs.npy` by default).

import Foundation
import MLX
import MLXAudioVAD

func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
    precondition(a.count == b.count, "shape mismatch: \(a.count) vs \(b.count)")
    var dot = 0.0, na = 0.0, nb = 0.0
    for index in 0..<a.count {
        dot += Double(a[index]) * Double(b[index])
        na += Double(a[index]) * Double(a[index])
        nb += Double(b[index]) * Double(b[index])
    }
    guard na > 0, nb > 0 else { return 0 }
    return dot / (na.squareRoot() * nb.squareRoot())
}

func flat(_ x: MLXArray) -> [Float] {
    x.asType(.float32).reshaped([-1]).asArray(Float.self)
}

struct WavFormatError: Error, CustomStringConvertible {
    let description: String
}

/// A real RIFF/WAVE chunk walker, not a fixed-44-byte-header skip: chunks
/// (`fmt `, `data`, and anything else — including a `JUNK`/`FLLR` filler
/// chunk, which is exactly what macOS's `say -o file.wav` writes ahead of
/// `data`, pushing PCM start to a non-44 offset, 4096 in that case) are
/// walked by their own declared size, each padded to an even byte boundary
/// per the RIFF spec, until `data` is found.
///
/// Requires 16 kHz mono 16-bit PCM (this model's own sample rate and the
/// format this port already assumes elsewhere) — anything else is a clear
/// error rather than a silent resample or a silent wrong-channel average.
func loadWav16kMonoPCM16(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    let bytes = [UInt8](data)

    func u32(_ offset: Int) -> UInt32 {
        UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
    }
    func u16(_ offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }
    func fourCC(_ offset: Int) -> String {
        String(decoding: bytes[offset..<(offset + 4)], as: UTF8.self)
    }

    guard bytes.count >= 12, fourCC(0) == "RIFF", fourCC(8) == "WAVE" else {
        throw WavFormatError(description: "\(url.lastPathComponent) is not a RIFF/WAVE file")
    }

    var sampleRate: UInt32?
    var channels: UInt16?
    var bitsPerSample: UInt16?
    var pcm: [UInt8]?

    var pos = 12
    while pos + 8 <= bytes.count {
        let chunkID = fourCC(pos)
        let chunkSize = Int(u32(pos + 4))
        let contentStart = pos + 8
        guard contentStart + chunkSize <= bytes.count else {
            // A truncated/streaming-written file's last chunk can overstate
            // its own size; take what is actually there rather than fault.
            if chunkID == "data" {
                pcm = Array(bytes[contentStart...])
            }
            break
        }
        switch chunkID {
        case "fmt ":
            guard chunkSize >= 16 else {
                throw WavFormatError(description: "fmt chunk too small in \(url.lastPathComponent)")
            }
            let audioFormat = u16(contentStart)
            guard audioFormat == 1 || audioFormat == 0xFFFE else {
                throw WavFormatError(
                    description: "\(url.lastPathComponent) is not PCM (format tag \(audioFormat))"
                )
            }
            channels = u16(contentStart + 2)
            sampleRate = u32(contentStart + 4)
            bitsPerSample = u16(contentStart + 14)
        case "data":
            pcm = Array(bytes[contentStart..<(contentStart + chunkSize)])
        default:
            break // JUNK, FLLR, LIST, etc. — skip by declared size, not assumed to be absent.
        }
        // RIFF chunks are word-aligned: an odd-sized chunk has one pad byte.
        pos = contentStart + chunkSize + (chunkSize % 2)
    }

    guard let sampleRate, let channels, let bitsPerSample, let pcm else {
        throw WavFormatError(description: "missing fmt or data chunk in \(url.lastPathComponent)")
    }
    guard sampleRate == 16000 else {
        throw WavFormatError(
            description: "\(url.lastPathComponent) is \(sampleRate) Hz; this model needs 16000 Hz"
                + " — resample before passing it in, this tool does not resample"
        )
    }
    guard bitsPerSample == 16 else {
        throw WavFormatError(
            description: "\(url.lastPathComponent) is \(bitsPerSample)-bit; only 16-bit PCM is supported"
        )
    }
    guard channels == 1 else {
        throw WavFormatError(
            description: "\(url.lastPathComponent) has \(channels) channels; expected mono"
        )
    }

    let bytesPerFrame = 2
    let frameCount = pcm.count / bytesPerFrame
    var samples = [Float](repeating: 0, count: frameCount)
    pcm.withUnsafeBytes { raw in
        for frame in 0..<frameCount {
            let byteOffset = frame * bytesPerFrame
            let lo = Int16(raw[byteOffset])
            let hi = Int16(raw[byteOffset + 1])
            let sample = Int16(bitPattern: UInt16(lo) | (UInt16(hi) << 8))
            samples[frame] = Float(sample) / 32768.0
        }
    }
    return samples
}

/// Runs one full streaming session (fresh state, "low" preset) over
/// `audioSamples`, chunked at `chunkSamples` per `feed` call — deliberately
/// not aligned to the model's own streaming window, matching the Python
/// reference dump's chunking so the two sides exercise identical window
/// boundaries. Returns the concatenated per-frame speaker probabilities.
func runStreaming(model: NemotronDiarizationModel, audioSamples: [Float], chunkSamples: Int) -> MLXArray {
    model.setStreamingConfig(.low)
    var state = model.initStreamingState()
    var outputs: [MLXArray] = []
    var i = 0
    while i < audioSamples.count {
        let end = min(i + chunkSamples, audioSamples.count)
        let probs = model.feedProbabilities(Array(audioSamples[i..<end]), state: &state, final: false)
        if probs.dim(0) > 0 { outputs.append(probs) }
        i = end
    }
    let finalProbs = model.feedProbabilities([], state: &state, final: true)
    if finalProbs.dim(0) > 0 { outputs.append(finalProbs) }
    let result = outputs.isEmpty
        ? MLXArray.zeros([0, model.config.numSpeakers])
        : MLX.concatenated(outputs, axis: 0)
    eval(result)
    return result
}

func runGate() throws {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        print("usage: NemotronParity <model-dir> <parity-data-dir> [speech-wav-path] [speech-reference-npy-path]")
        exit(1)
    }
    let modelDir = URL(fileURLWithPath: args[1])
    let dataDir = URL(fileURLWithPath: args[2])

    print("Loading model from \(modelDir.path)")
    let model = try NemotronDiarizationModel.fromModelDirectory(modelDir)

    let referenceAudio = try MLX.loadArray(url: dataDir.appendingPathComponent("parity_audio.npy"))
    let referenceMel = try MLX.loadArray(url: dataDir.appendingPathComponent("parity_mel.npy"))
    let referenceEncoder = try MLX.loadArray(url: dataDir.appendingPathComponent("parity_encoder.npy"))
    let referenceProbs = try MLX.loadArray(url: dataDir.appendingPathComponent("parity_probs.npy"))
    let referenceStreamingProbs = try MLX.loadArray(
        url: dataDir.appendingPathComponent("parity_streaming_probs.npy")
    )

    // --- Stage 1: mel. Swift frontend on the reference's own fixed-seed audio. ---
    let swiftMel = model.preprocessor(referenceAudio)
    eval(swiftMel)
    let melCosine = cosineSimilarity(flat(swiftMel), flat(referenceMel))
    print("mel      shape=\(swiftMel.shape) vs \(referenceMel.shape)  cosine=\(melCosine)")

    // --- Stage 2: encoder. Fed the REFERENCE mel (not swiftMel) so this stage
    // isolates the encoder's own correctness from any upstream mel drift. ---
    let melLengths = MLXArray([Int32(referenceMel.dim(2))])
    let (stackedX, stackedLengths) = model.encoder.preEncode(
        referenceMel.asType(model.dtype), lengths: melLengths
    )
    let swiftEncoder = model.encoder(stackedX, lengths: stackedLengths)
    eval(swiftEncoder)
    let encoderCosine = cosineSimilarity(flat(swiftEncoder), flat(referenceEncoder))
    print("encoder  shape=\(swiftEncoder.shape) vs \(referenceEncoder.shape)  cosine=\(encoderCosine)")

    // --- Stage 3: full single-window forward, also fed the reference mel. ---
    let swiftProbs = model(referenceMel, lengths: melLengths)
    eval(swiftProbs)
    let probsCosine = cosineSimilarity(flat(swiftProbs), flat(referenceProbs))
    print("probs    shape=\(swiftProbs.shape) vs \(referenceProbs.shape)  cosine=\(probsCosine)")

    // --- Stage 4: streaming, on the same fixed-seed synthetic-noise audio
    // as the other three stages (see the README for why this stage in
    // particular reads lower on noise than on real speech). ---
    let audioSamples: [Float] = referenceAudio.asType(.float32).asArray(Float.self)
    let swiftStreamingProbs = runStreaming(model: model, audioSamples: audioSamples, chunkSamples: 4000)
    let streamingCosine = cosineSimilarity(flat(swiftStreamingProbs), flat(referenceStreamingProbs))
    print(
        "streaming (noise) shape=\(swiftStreamingProbs.shape) vs \(referenceStreamingProbs.shape)"
            + "  cosine=\(streamingCosine)"
    )

    // --- Stage 5 (optional): streaming on real speech, from an actual WAV
    // file this tool parses itself (see `loadWav16kMonoPCM16`), reproducing
    // the README's headline streaming figures rather than only quoting them. ---
    if args.count >= 5 {
        let speechWavURL = URL(fileURLWithPath: args[3])
        let speechReferenceURL = URL(fileURLWithPath: args[4])
        print("Loading speech WAV from \(speechWavURL.path)")
        let speechSamples = try loadWav16kMonoPCM16(speechWavURL)
        print("  \(speechSamples.count) samples (\(Double(speechSamples.count) / 16000.0) s)")
        let speechReference = try MLX.loadArray(url: speechReferenceURL)
        let speechProbs = runStreaming(model: model, audioSamples: speechSamples, chunkSamples: 4000)
        let speechCosine = cosineSimilarity(flat(speechProbs), flat(speechReference))
        print(
            "streaming (speech) shape=\(speechProbs.shape) vs \(speechReference.shape)"
                + "  cosine=\(speechCosine)"
        )
    }

    print("")
    print("GATE mel>=0.9999: \(melCosine >= 0.9999)")
    print("GATE encoder>=0.999: \(encoderCosine >= 0.999)")
    print("GATE probs>=0.999: \(probsCosine >= 0.999)")
    // Streaming has no required threshold in the gate (see this model's
    // README for why noise measures lower than the single-window stages):
    // report it, don't gate on it.
    print("streaming (noise) cosine (reported, not gated): \(streamingCosine)")
}

// This sandbox has no working `metal` compiler, so mlx-swift's GPU kernels
// never get built into a usable metallib unless one is colocated with this
// binary; run on CPU explicitly so results don't depend on that, and to
// match the reference's own `precise_math` test fixture (which forces
// `mx.stream(mx.cpu)` for the same reason: to avoid Apple Silicon's
// reduced-precision float32 GEMMs on GPU).
try Device.withDefaultDevice(Device.cpu) {
    try runGate()
}
