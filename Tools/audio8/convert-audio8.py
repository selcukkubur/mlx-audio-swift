"""Edge0/Audio8-ASR-Infinite -> the layout mlx-audio-swift's Audio8 model reads.

The audio tower is Voxtral Realtime's encoder (its own config says
`model_type: voxtral_realtime_encoder`), and mlx-audio-swift already implements
that, down to the bias pattern: q/v/o carry one, k does not, and of the MLP only
the down projection does. So the tower is a pure rename onto the existing
`VoxtralRealtimeAudioEncoder`; nothing about it is reimplemented here.

The decoder is Qwen2 with a Voxtral-style adaptive RMS modulation per layer, and
that half is new.
"""
import json, sys, numpy as np
from pathlib import Path
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])

def load(d):
    # bfloat16 on disk, which numpy cannot represent — torch reads it, and
    # float32 is what the MLX side quantises from anyway.
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[k] = v.to(torch.float32).numpy()
    return out

src = load(SRC)
dst, dropped = {}, []

def enc_layer(rest):
    # `rest` is "<N>.<tail>" inside audio_tower.layers
    n, tail = rest.split(".", 1)
    m = {
        "self_attn.q_proj": f"attention.wq",
        "self_attn.k_proj": f"attention.wk",
        "self_attn.v_proj": f"attention.wv",
        "self_attn.o_proj": f"attention.wo",
        "self_attn_layer_norm": "attention_norm",
        "final_layer_norm": "ffn_norm",
        "mlp.gate_proj": "feed_forward_w1",
        "mlp.up_proj": "feed_forward_w3",
        "mlp.down_proj": "feed_forward_w2",
    }
    for src_pre, dst_pre in m.items():
        if tail.startswith(src_pre + "."):
            suffix = tail[len(src_pre) + 1:]
            return f"encoder.transformer_layers.{n}.{dst_pre}.{suffix}"
    return None

def dec_layer(rest):
    n, tail = rest.split(".", 1)
    m = {
        "self_attn.q_proj": "attention.wq",
        "self_attn.k_proj": "attention.wk",
        "self_attn.v_proj": "attention.wv",
        "self_attn.o_proj": "attention.wo",
        "input_layernorm": "attention_norm",
        "post_attention_layernorm": "ffn_norm",
        "mlp.gate_proj": "feed_forward_w1",
        "mlp.up_proj": "feed_forward_w3",
        "mlp.down_proj": "feed_forward_w2",
        "ada_rms_norm.linear1": "ada_rms_norm.ada_down",
        "ada_rms_norm.linear2": "ada_rms_norm.ada_up",
    }
    for src_pre, dst_pre in m.items():
        if tail.startswith(src_pre + "."):
            suffix = tail[len(src_pre) + 1:]
            return f"decoder.layers.{n}.{dst_pre}.{suffix}"
    return None

for k, v in src.items():
    # Photon's voice-activity heads steer its own endpointing; transcription
    # never reads them and the Swift model has nowhere to put them.
    if k.startswith("semantic_vad_heads."):
        dropped.append(k); continue

    n = None
    if k.startswith("audio_tower.embedder.conv"):
        which = "0" if ".conv1." in k else "1"
        suffix = k.rsplit(".", 1)[1]
        n = f"encoder.conv_layers_{which}_conv.conv.{suffix}"
        if suffix == "weight":
            # PyTorch Conv1d is (out, in, k); MLX wants (out, k, in).
            #
            # `ascontiguousarray` is load-bearing, not tidiness. A bare
            # `transpose` returns a VIEW: safetensors then writes the original
            # byte order under the new shape, so the file claims (out, k, in)
            # while holding (out, in, k). Nothing downstream can see it — the
            # shape is right, the values at index [0,0,:] are right, and every
            # consistency check passes because everything reads the same wrong
            # bytes. It shows up only as a model that runs and transcribes
            # nonsense.
            v = np.ascontiguousarray(v.transpose(0, 2, 1))
    elif k.startswith("audio_tower.layers."):
        n = enc_layer(k[len("audio_tower.layers."):])
    elif k == "audio_tower.norm.weight":
        n = "encoder.transformer_norm.weight"
    elif k.startswith("language_model.model.layers."):
        n = dec_layer(k[len("language_model.model.layers."):])
    elif k == "language_model.model.embed_tokens.weight":
        n = "decoder.embed_tokens.weight"
    elif k == "language_model.model.norm.weight":
        n = "decoder.norm.weight"
    elif k.startswith("multi_modal_projector."):
        # Not a separate module on the Swift side. `VoxtralRealtimeAudioEncoder`
        # already ends in `downsampleAndProject`, which reshapes the encoder
        # frames into groups of `downsampleFactor` and runs them through
        # gelu(proj0) -> proj2 — the same grouping and the same two-layer
        # projector this checkpoint calls `multi_modal_projector`, with the
        # same 10240 input width (1280 x 8).
        n = ("encoder.audio_language_projection_0" if ".linear_1." in k
             else "encoder.audio_language_projection_2") + ".weight"
    elif k == "frame_len_embedding.weight":
        n = "frame_len_embedding.weight"

    if n is None:
        raise SystemExit(f"unmapped tensor: {k}  {v.shape}")
    dst[n] = v.astype(np.float32)

OUT.mkdir(parents=True, exist_ok=True)
save_file(dst, str(OUT / "model.safetensors"), metadata={"format": "pt"})
cfg = json.loads((SRC / "config.json").read_text())
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))
for extra in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja"):
    if (SRC / extra).exists():
        (OUT / extra).write_text((SRC / extra).read_text())
print(f"source {len(src)} -> converted {len(dst)}, dropped {len(dropped)} (vad heads)")
