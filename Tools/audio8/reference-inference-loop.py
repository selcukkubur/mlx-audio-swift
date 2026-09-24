import torch, warnings, wave, numpy as np, sys
from safetensors.torch import load_file
warnings.filterwarnings("ignore")
S="/private/tmp/claude-501/-Users-selcukkubur-Documents-GitHub-personal-openmeet-web--claude-worktrees-openmeet-domain-extensions-e6865c/05e1d450-5c7a-4b7f-8526-bc24372d4aca/scratchpad"
p=f"{S}/models/audio8-src"
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, AutoTokenizer
cfg=AutoConfig.from_pretrained(p, trust_remote_code=True)
m=AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
sd=load_file(f"{p}/model.safetensors"); sd.update(load_file(f"{p}/semantic_vad_heads.safetensors"))
m.load_state_dict(sd, strict=False); m=m.to(torch.float32).eval()
m.tie_weights()
tok=AutoTokenizer.from_pretrained(p, trust_remote_code=True)
proc=AutoProcessor.from_pretrained(p, trust_remote_code=True)

def wav(path):
    w=wave.open(path); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)
    return a.astype(np.float32)/32768.0

name=sys.argv[1] if len(sys.argv)>1 else "hard"
FRAME_LEN=int(sys.argv[2]) if len(sys.argv)>2 else 8
DELAY=int(sys.argv[3]) if len(sys.argv)>3 else 2
a=wav(f"{S}/{name}.wav")
# The model is trained with a run-up: `streaming_n_left_pad_tokens` worth of
# audio before the first word, 9 tokens at frame_len 8. Without it the decoder
# is asked for its first token while the encoder still has nothing behind it,
# which is where the junk leading token and the lost opening words came from.
LEFT_PAD_TOKENS=9
run_up=int(LEFT_PAD_TOKENS*FRAME_LEN*0.02*16000)  # frames are 20 ms at the tower
a=np.concatenate([np.zeros(run_up, dtype=np.float32), a])
feats=proc(a, sampling_rate=16000, return_tensors="pt")["input_features"]
print("mel:", tuple(feats.shape))

with torch.no_grad():
    hid,_ = m.get_audio_tower_hidden_states(input_features=feats, use_cache=False, return_outputs=True)
print("audio tower hidden:", tuple(hid.shape))
n_groups = hid.shape[1] // FRAME_LEN
print("frame_len", FRAME_LEN, "-> groups:", n_groups)

bos = cfg.bos_token_id; eos = cfg.eos_token_id
ids = torch.tensor([[bos]], dtype=torch.long)
out_ids = []
with torch.no_grad():
    # Run past the audio: with a delay the text trails the sound, so the last
    # words only arrive in the extra steps after the audio groups run out.
    for step in range(min(n_groups + DELAY + 4, 500)):
        n = ids.shape[1]
        grouped = m.group_audio_hidden_states(hid, frame_len=FRAME_LEN, target_token_count=n)
        aud = m.multi_modal_projector(grouped)
        emb = m.get_input_embeddings()(ids) + aud.to(dtype=torch.float32)
        res = m.forward_language_model_with_delay(
            inputs_embeds=emb, frame_len=FRAME_LEN, num_delay_tokens=DELAY, use_cache=False)
        logits = res.logits if hasattr(res,"logits") else res[0]
        nxt = int(logits[0,-1].argmax())
        if nxt == eos or (isinstance(eos,list) and nxt in eos): break
        out_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
print("TOKENS:", len(out_ids))
print("TEXT:", tok.decode(out_ids, skip_special_tokens=True))
