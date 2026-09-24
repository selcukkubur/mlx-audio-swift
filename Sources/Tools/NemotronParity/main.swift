// Parity-gate harness for the Nemotron 3 Diarization port. Loads the
// checkpoint and the same fixed-seed audio a Python reference dump produced
// (see nemotron_parity.py, kept alongside this port's docs, not shipped),
// runs the Swift port over each stage, and reports cosine similarity
// against the reference arrays.
//
// Usage: NemotronParity <model-dir> <parity-data-dir>

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

func runGate() throws {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        print("usage: NemotronParity <model-dir> <parity-data-dir>")
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

    // --- Stage 4: streaming. Same "low" preset, same chunking scheme (4000
    // samples per feed, deliberately not aligned to the streaming window),
    // same reference audio, as the Python side's streaming dump. ---
    model.setStreamingConfig(.low)
    var state = model.initStreamingState()
    let audioSamples: [Float] = referenceAudio.asType(.float32).asArray(Float.self)
    let chunkSamples = 4000
    var streamingOutputs: [MLXArray] = []
    var i = 0
    while i < audioSamples.count {
        let end = min(i + chunkSamples, audioSamples.count)
        let chunk = Array(audioSamples[i..<end])
        let probs = model.feedProbabilities(chunk, state: &state, final: false)
        if probs.dim(0) > 0 { streamingOutputs.append(probs) }
        i = end
    }
    let finalProbs = model.feedProbabilities([], state: &state, final: true)
    if finalProbs.dim(0) > 0 { streamingOutputs.append(finalProbs) }
    let swiftStreamingProbs = streamingOutputs.isEmpty
        ? MLXArray.zeros([0, model.config.numSpeakers])
        : MLX.concatenated(streamingOutputs, axis: 0)
    eval(swiftStreamingProbs)
    let streamingCosine = cosineSimilarity(flat(swiftStreamingProbs), flat(referenceStreamingProbs))
    print(
        "streaming shape=\(swiftStreamingProbs.shape) vs \(referenceStreamingProbs.shape)"
            + "  cosine=\(streamingCosine)  frameOffset=\(state.frameOffset)"
    )

    print("")
    print("GATE mel>=0.9999: \(melCosine >= 0.9999)")
    print("GATE encoder>=0.999: \(encoderCosine >= 0.999)")
    print("GATE probs>=0.999: \(probsCosine >= 0.999)")
    // Streaming has no required threshold in the gate (see this model's
    // README for why it measures lower than the single-window stages):
    // report it, don't gate on it.
    print("streaming cosine (reported, not gated): \(streamingCosine)")
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
