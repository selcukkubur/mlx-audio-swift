import Foundation
import MLX
import MLXNN

/// Which arithmetic a packed layer uses, decided once when the model is loaded.
///
/// Not a per-call switch on purpose: the two modes have different residency —
/// `.dequantized` materialises every packed matrix as float16 and keeps it — so
/// choosing per call would mean paying for both.
public enum PhononExecution: String, Sendable, CaseIterable {
    /// `phononFiveValueMatmul` reads the packed bytes directly; nothing is
    /// expanded. Correct — it transcribes the gate's audio identically — but the
    /// speed gate measured it 622x slower than `.dequantized` on 2026-09-30, so
    /// it is not the default and has to be asked for by name.
    case kernel
    /// `dequantized()` runs once at load and an ordinary dense matmul runs
    /// thereafter — the same fp16 arithmetic the rest of the model already uses.
    /// The shipping mode, per the speed gate.
    case dequantized
}

/// The body both packed layers share: one five-value matrix and the mode it is
/// read in.
///
/// A class rather than a struct, and deliberately not a `Module`: `Module`'s
/// reflection classifies a non-`MLXArray`, non-`Module` ivar as `.other` and
/// leaves it alone, so the packed pieces stay out of `parameters()`. That
/// matters — they must not be handed to `update(parameters:)`, counted among the
/// model's keys, or cast to the compute dtype (`trits` is uint8; casting it
/// would be silent nonsense).
///
/// `@unchecked Sendable` for the same reason as `PhononFiveValue`: every stored
/// property is a `let` that is never written after init.
final class PhononProjection: @unchecked Sendable {
    let execution: PhononExecution
    let outFeatures: Int
    let inFeatures: Int

    /// Kept only in `.kernel`. `.dequantized` drops the packed bytes once it has
    /// decoded them: holding both would put 189 MB on top of the 1.3 GB the
    /// fallback is supposed to cost, and the fallback's memory figure is half of
    /// what the speed gate is deciding.
    let packed: PhononFiveValue?
    /// `[O, I]`, present only in `.dequantized`. Decoded as float16; the loader's
    /// compute-dtype cast may convert it afterwards.
    ///
    /// Whoever holds this must make sure it is reachable by that cast. MLX casts
    /// a parameter with `_updateInternal`, which mutates the `MLXArray` object in
    /// place, so a layer that hands this same object to the parameter tree gets
    /// the conversion for free — and a layer that keeps it only here does not.
    /// `PhononPointwiseConv1d` was the second kind, and at bfloat16 it cost
    /// 0.76 GiB and half the speed until it stopped being.
    let dense: MLXArray?

    /// The dtype of the decoded matrix, or nil in `.kernel`. The loader checks
    /// this after casting rather than trusting the paragraph above.
    var denseDType: DType? { dense?.dtype }

    init(packed: PhononFiveValue, execution: PhononExecution) {
        self.execution = execution
        self.outFeatures = packed.outFeatures
        self.inFeatures = packed.inFeatures
        switch execution {
        case .kernel:
            self.packed = packed
            self.dense = nil
            // None of this is a model parameter, so the `eval(model)` at the end
            // of loading will not reach it. Without this the packed bytes would
            // still be lazy safetensors slices and the first matmul would pay to
            // read them off disk — a cost that would land entirely on whichever
            // transcription ran first.
            eval(packed.trits, packed.hiBits, packed.nzTile, packed.nzRow, packed.lo, packed.hi)
        case .dequantized:
            let w = packed.dequantized()
            eval(w)
            self.packed = nil
            self.dense = w
        }
    }

    /// `x` is `[..., I]`, the result `[..., O]`. Used by the layers that cannot
    /// install the dense matrix as their own `weight`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch execution {
        case .kernel:
            guard let packed else { fatalError("PhononProjection: .kernel without packed weights") }
            return phononFiveValueMatmul(x, packed)
        case .dequantized:
            guard let dense else { fatalError("PhononProjection: .dequantized without a dense matrix") }
            return matmul(x, dense.T)
        }
    }
}

/// A layer whose weights came from a packed block.
///
/// The loader uses this to count what it swapped in and to check that the
/// compute-dtype cast reached every decoded matrix. Both are cheap; both replace
/// an assumption that has already been wrong once.
protocol PhononPackedLayer: Module {
    var packedProjection: PhononProjection { get }
}

/// `Linear` backed by a five-value packed matrix.
///
/// A subclass rather than a protocol both dense and packed layers satisfy. The
/// conformer declares its projections as `@ModuleInfo var linear1: Linear`, and
/// MLX replaces a child module only when the replacement is assignable to that
/// ivar's declared type, so a protocol would mean retyping every projection in
/// `ParakeetFeedForward`, `NemoMultiHeadAttention`,
/// `NemoRelPositionMultiHeadAttention` and `NemoJointNetwork` — the conformer
/// rewritten to admit a format most callers will never use. Subclassing changes
/// nothing outside this file and is the route MLX takes itself
/// (`QuantizedLinear: Linear`).
public final class PhononLinear: Linear, PhononPackedLayer {
    let projection: PhononProjection

    var packedProjection: PhononProjection { projection }

    /// The dimensions of the matrix this layer applies, not of whatever is in
    /// `weight`.
    ///
    /// In `.kernel` `weight` is a 1x1 placeholder, so the inherited `shape` would
    /// describe 216 real projections as 1x1. This override fixes the accessor.
    /// It does **not** fix the parameter: `parameters()` still emits the
    /// placeholder, so a kernel-mode model must not be re-saved or re-verified
    /// against a checkpoint — it would write a 1x1 weight and read back nothing
    /// usable. Nothing in this repo does that today; anything that starts to
    /// needs to reconstitute the matrix from `projection.packed` first.
    public override var shape: (Int, Int) {
        (projection.outFeatures, projection.inFeatures)
    }

    /// In `.dequantized` the decoded matrix is installed as `weight`, so the
    /// inherited `callAsFunction` runs and the fallback is the ordinary dense
    /// path rather than an imitation of it. In `.kernel` there is no dense
    /// matrix and `weight` is a 1x1 placeholder, because `Linear.weight` is a
    /// non-optional `let` that `Linear.shape` reads.
    init(packed: PhononFiveValue, bias: MLXArray?, execution: PhononExecution) {
        let projection = PhononProjection(packed: packed, execution: execution)
        self.projection = projection
        switch execution {
        case .kernel:
            super.init(weight: MLXArray.zeros([1, 1], type: Float16.self), bias: bias)
        case .dequantized:
            guard let dense = projection.dense else {
                fatalError("PhononLinear: .dequantized without a dense matrix")
            }
            super.init(weight: dense, bias: bias)
        }
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard projection.execution == .kernel else {
            return super.callAsFunction(x)
        }
        let y = projection(x)
        guard let bias else { return y }
        return y + bias.asType(y.dtype)
    }

    public override func describeExtra(_ indent: Int) -> String {
        "(inputDimensions=\(projection.inFeatures), outputDimensions=\(projection.outFeatures), bias=\(bias != nil), execution=\(projection.execution.rawValue))"
    }
}

/// The pointwise (kernel size 1) `Conv1d` of the conformer convolution module,
/// backed by the same packed format.
///
/// A 1x1 convolution over `[B, T, C]` is the arithmetic of a linear projection,
/// which is why the converter packs these alongside the attention and
/// feed-forward matrices. They are 48 of the 264 packed modules and about a
/// quarter of the packed weights, so leaving them dense would make both the
/// memory figure and the speed figure describe something the app would not ship.
///
/// Unlike `Linear`, `Conv1d` has no `init(weight:bias:)` and its `weight` and
/// `bias` are `let`s set by its own initialiser, so both modes go through
/// `projection` here and `weight` is always a 1x1x1 placeholder. As in
/// `PhononLinear`, that placeholder is what `parameters()` emits, so one of
/// these must not be re-saved or re-verified against a checkpoint. Every packed
/// module in this checkpoint is bias-free and the loader rejects one that is
/// not, so there is no bias to carry.
public final class PhononPointwiseConv1d: Conv1d, PhononPackedLayer {
    let projection: PhononProjection

    var packedProjection: PhononProjection { projection }

    /// The decoded matrix again — the same object `projection.dense` holds, not a
    /// copy — declared here so that `Module`'s reflection sees it and the
    /// loader's compute-dtype cast reaches it.
    ///
    /// `PhononLinear` gets this for nothing because its decoded matrix *is* its
    /// `weight`. This layer cannot do that: `Conv1d.weight` is a `let` set by the
    /// superclass initialiser and is the wrong rank besides. Without this
    /// declaration the 48 pointwise convolutions stayed float16 while everything
    /// else converted, and at bfloat16 that mixed-dtype promotion cost 0.76 GiB
    /// and doubled transcription time.
    let dense: MLXArray?

    init(packed: PhononFiveValue, execution: PhononExecution) {
        let projection = PhononProjection(packed: packed, execution: execution)
        self.projection = projection
        self.dense = projection.dense
        super.init(inputChannels: 1, outputChannels: 1, kernelSize: 1, bias: false)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(x)
    }

    public override func describeExtra(_ indent: Int) -> String {
        "(inputChannels=\(projection.inFeatures), outputChannels=\(projection.outFeatures), kernelSize=1, bias=false, execution=\(projection.execution.rawValue))"
    }
}
