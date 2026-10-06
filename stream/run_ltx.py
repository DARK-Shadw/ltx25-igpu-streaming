"""Run LTX-2.5 (22B, unquantized bf16) on an Intel iGPU by streaming weights from NVMe.

Phases (each frees its GPU memory before the next):
  A. text encoder (48 Gemma layers streamed)  -> prompt embeddings
  B. connectors (streamed)                    -> conditioning
  C. transformer (48 blocks, prefetch-streamed) x N denoising steps
  D. VAE decode -> frames
"""
import gc
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import transformers
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler, LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from transformers import AutoConfig, AutoTokenizer

from runtime import DEV, Streamed, build_empty
from dev import acc, NAME as DEVNAME, MODELS as M

PROMPT = os.environ.get("PROMPT", "A golden retriever running along a sunny beach, waves crashing, cinematic")
NEG = os.environ.get("NEG", "worst quality, blurry, distorted, jittery")
H, W, FRAMES = int(os.environ.get("H", 256)), int(os.environ.get("W", 384)), int(os.environ.get("FRAMES", 25))
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
DISTILLED = os.environ.get("DISTILLED", "1") == "1"  # repo's default transformer/ is the distilled DiT: unguided, explicit sigmas
SIGMAS = list(DISTILLED_SIGMA_VALUES) if DISTILLED else None
if DISTILLED and os.environ.get("S1SIGMAS"):   # experiment: custom stage-1 schedule, comma-separated
    SIGMAS = [float(x) for x in os.environ["S1SIGMAS"].split(",")]
STEPS = len(SIGMAS) if DISTILLED else int(os.environ.get("STEPS", 8))
SEED = int(os.environ.get("SEED", -1))
LAT = os.environ.get("LAT", "latents.pt")
SKIPDEC = os.environ.get("SKIPDEC", "0") == "1"
TWOSTAGE = os.environ.get("TWOSTAGE", "0") == "1"  # stage 1 at WxH, x2 latent upsample, 3 more steps at 2Wx2H
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES as S2_SIGMAS
S2STEPS = int(os.environ.get("S2STEPS", 3))   # 3 = full recipe; 2 drops the noisiest refinement step (faster, keeps stage-1 content closer)
S2_SIGMAS = S2_SIGMAS[-S2STEPS:]
IMAGE = os.environ.get("IMAGE")              # path of a first-frame conditioning image (image-to-video); empty = text-to-video
FPS = float(os.environ.get("FPS", 24))        # generation frame rate; 12 = anime "on twos" (duplicate frames at export)
TOTAL_PASSES = STEPS + (len(S2_SIGMAS) if TWOSTAGE else 0)
MAXSEQ = int(os.environ.get("MAXSEQ", 256))
OUT = os.environ.get("OUT", "out.mp4")
T0 = time.perf_counter()


def log(*a):
    free = None
    try:
        import psutil
        free = psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    gpu = acc.memory_allocated() / 2**30
    print(f"[{time.perf_counter() - T0:7.1f}s]" + (f"[ram free {free:4.1f}G]" if free else "") + f"[gpu {gpu:4.1f}G]", *a, flush=True)


def cleanup():
    gc.collect()
    acc.synchronize()
    acc.empty_cache()


# ---------- build pieces ----------
log("building components (no weights loaded)")
tok = AutoTokenizer.from_pretrained(f"{M}/tokenizer")
if os.environ.get("BOS", "1") == "1":
    tok.add_bos_token = True  # shipped tokenizer template omits <bos>
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{M}/scheduler")
tcfg = AutoConfig.from_pretrained(f"{M}/text_encoder")
te_full = build_empty(lambda: transformers.Gemma4UnifiedForConditionalGeneration._from_config(tcfg, dtype=torch.bfloat16))
conn = build_empty(lambda: LTX2TextConnectors.from_config(LTX2TextConnectors.load_config(f"{M}/connectors")))
dit = build_empty(lambda: LTX2VideoTransformer3DModel.from_config(LTX2VideoTransformer3DModel.load_config(f"{M}/transformer")))
from diffusers import AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from safetensors import safe_open


def empty_with_stats(cls, comp):
    """Meta-weight model; only the tiny latents_mean/std buffers are loaded (needed by the pipeline)."""
    m = build_empty(lambda: cls.from_config(cls.load_config(f"{M}/{comp}")))
    f = safe_open(f"{M}/{comp}/diffusion_pytorch_model.safetensors", "pt")
    for k in ("latents_mean", "latents_std"):
        if k in m._buffers:
            m._buffers[k] = f.get_tensor(k).to(DEV)
    return m


vae = empty_with_stats(AutoencoderKLLTX2Video, "vae")
audio_vae = empty_with_stats(AutoencoderKLLTX2Audio, "audio_vae")
vocoder = build_empty(lambda: LTX2VocoderWithBWE.from_config(LTX2VocoderWithBWE.load_config(f"{M}/vocoder")))

from diffusers import LTX2ImageToVideoPipeline
PipeCls = LTX2ImageToVideoPipeline if IMAGE else LTX2Pipeline
pipe = PipeCls(scheduler=sched, vae=vae, audio_vae=audio_vae, text_encoder=te_full.model, tokenizer=tok,
                    connectors=conn, transformer=dit, vocoder=vocoder)
LTX2Pipeline._execution_device = property(lambda self: DEV)
LTX2ImageToVideoPipeline._execution_device = property(lambda self: DEV)
log("components ready")

# ---------- A. text encoder ----------
te = Streamed(te_full, f"{M}/text_encoder", [f"model.language_model.layers.{i}" for i in range(48)], log=log)
n = te.load_resident(skip_prefixes=("lm_head", "model.embed_vision", "model.embed_audio", "model.vision_embedder"))
log(f"A: text encoder resident weights {n / 2**30:.2f} GiB loaded; encoding prompt")
t = time.perf_counter()
with torch.no_grad():
    embeds, mask = pipe._get_gemma_prompt_embeds([PROMPT] if DISTILLED else [PROMPT, NEG], max_sequence_length=MAXSEQ, device=DEV,
                                                 dtype=torch.bfloat16)
acc.synchronize()
log(f"A: done in {time.perf_counter() - t:.1f}s (streamed {te.bytes / 2**30:.1f} GiB, waited {te.wait_s:.1f}s on I/O); embeds {tuple(embeds.shape)}")
log("DBG embeds finite:", torch.isfinite(embeds).all().item(), "absmax", embeds.abs().amax().item())
pos_e, pos_m = embeds[0:1], mask[0:1]
neg_e, neg_m = (None, None) if DISTILLED else (embeds[1:2], mask[1:2])
te.free_resident()
te.close()
del te
pipe.text_encoder = None
embeds = mask = None
cleanup()

_live = {}
for o in gc.get_objects():
    try:
        if isinstance(o, torch.Tensor) and o.device.type == DEVNAME:
            k = (tuple(o.shape), str(o.dtype))
            _live[k] = _live.get(k, 0) + o.numel() * o.element_size()
    except Exception:
        pass
log("live gpu tensors after A (top):", sorted(((round(v / 2**20), k) for k, v in _live.items()), reverse=True)[:6])

# ---------- B. connectors ----------
cunits = ["video_text_proj_in", "audio_text_proj_in"] + \
         [f"{s}_connector.transformer_blocks.{i}" for s in ("video", "audio") for i in range(8)]
cs = Streamed(conn, f"{M}/connectors", cunits, log=log)
n = cs.load_resident()
log(f"B: connectors resident {n / 2**20:.0f} MiB; running")
t = time.perf_counter()
with torch.no_grad():
    cat_e = pos_e if DISTILLED else torch.cat([neg_e, pos_e])
    cat_m = pos_m if DISTILLED else torch.cat([neg_m, pos_m])
    c_video, c_audio, c_mask = conn(cat_e, cat_m, padding_side="left")
acc.synchronize()
log("DBG connector out finite:", [torch.isfinite(x).all().item() for x in (c_video, c_audio)],
    "absmax", c_video.abs().amax().item(), c_audio.abs().amax().item())
log(f"B: done in {time.perf_counter() - t:.1f}s (streamed {cs.bytes / 2**30:.1f} GiB)")
cs.free_resident()
cs.close()
del cs, cat_e, cat_m
# connector outputs are cached; the pipeline only needs placeholders with the right dtype/shape
pos_e = torch.zeros(1, 1, 1, dtype=torch.bfloat16, device=DEV)
neg_e = None if DISTILLED else pos_e.clone()
pipe.connectors = None
object.__setattr__(pipe, "connectors", lambda *a, **k: (c_video, c_audio, c_mask))
cleanup()
import collections
_live = collections.Counter()
for o in gc.get_objects():
    try:
        if isinstance(o, torch.Tensor) and o.device.type == DEVNAME:
            _live[(tuple(o.shape), str(o.dtype))] += o.numel() * o.element_size()
    except Exception:
        pass
log("live gpu tensors after B (top):", [(k, round(v / 2**20)) for k, v in _live.most_common(6)])

# ---------- image conditioning (image-to-video) ----------
img_latents = None
if IMAGE and not (TWOSTAGE and os.environ.get("RESUME", "0") == "1" and os.path.exists(LAT + ".stage1")):
    import numpy as _np
    import PIL.Image as _PIL
    from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import retrieve_latents
    from diffusers.pipelines.ltx2.utils import apply_image_conditioning_crf
    _img = _PIL.open(IMAGE).convert("RGB")
    _crf = 18                                    # LTX-2.5 default conditioning CRF (re-compress like the training data)
    _img = _PIL.fromarray(apply_image_conditioning_crf(_np.array(_img), _crf))
    _t = pipe.video_processor.preprocess(_img, height=H, width=W).to(DEV, torch.bfloat16)      # (1, 3, H, W) in [-1, 1]
    _vae = AutoencoderKLLTX2Video.from_pretrained(f"{M}/vae", torch_dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        _z = retrieve_latents(_vae.encode(_t.unsqueeze(2)), None, "argmax")          # (1, 128, 1, H/32, W/32)
    del _vae
    cleanup()
    _F = (FRAMES - 1) // 8 + 1
    img_latents = _z.repeat(1, 1, _F, 1, 1).float()
    log(f"I2V: encoded {os.path.basename(IMAGE)} -> latents {tuple(_z.shape)}, start latents {tuple(img_latents.shape)}")

# ---------- C. transformer denoising ----------
dit_units = [f"transformer_blocks.{i}" for i in range(48)]
def make_ds(passes):
    # passes=0 -> synchronous streaming (no prefetch buffers): slower per step, ~3 GB less RAM; used for memory-hungry stage 2
    return Streamed(dit, f"{M}/transformer", dit_units, prefetch_passes=passes, slots=2, log=log,
                    transpose_linear=os.environ.get("TRANSPOSE", "1") == "1")  # default ON: pre-transposed weights (bit-identical output, faster matmuls)


FFN_CHUNK = int(os.environ.get("FFN_CHUNK", "4096"))  # tokens per feed-forward chunk (cuts peak activation memory at large resolutions)


def _chunked(orig):
    def fwd(x, *a, **k):
        if x.shape[1] <= FFN_CHUNK:
            return orig(x, *a, **k)
        return torch.cat([orig(c, *a, **k) for c in x.split(FFN_CHUNK, dim=1)], dim=1)
    return fwd


for _u in dit_units:
    _b = dit.get_submodule(_u)
    _b.ff.forward = _chunked(_b.ff.forward)

RESUME_S1 = TWOSTAGE and os.environ.get("RESUME", "0") == "1" and os.path.exists(LAT + ".stage1")
ds = make_ds(0 if RESUME_S1 else STEPS)
n = ds.load_resident()
for name, p in dit.named_parameters():  # params absent from the checkpoint (unused keyframe embedding)
    if p.device.type == "meta" and not name.startswith("transformer_blocks."):
        log(f"  [warn] zero-filling missing {name}")
        path, _, leaf = name.rpartition(".")
        sub = dit.get_submodule(path) if path else dit
        sub._parameters[leaf] = torch.nn.Parameter(torch.zeros(p.shape, dtype=p.dtype, device=DEV), requires_grad=False)
log(f"C: transformer resident {n / 2**30:.2f} GiB; denoising {STEPS} steps at {W}x{H}, {FRAMES} frames")
t = time.perf_counter()
step_t = []


PREVIEW_DIR = os.environ.get("PREVIEW_DIR")           # if set, write a live preview picture after every denoising step
_stage = {"tag": "C", "n": STEPS, "F": (FRAMES - 1) // 8 + 1, "h": H // 32, "w": W // 32}


def on_step(pipe_, i, tt, kw):
    acc.synchronize()
    now = time.perf_counter()
    step_t.append(now)
    log(f"{_stage['tag']}: step {i + 1}/{_stage['n']} done ({now - (step_t[-2] if len(step_t) > 1 else t):.1f}s)")
    if PREVIEW_DIR and kw.get("noise_pred_video") is not None:
        try:   # x0 estimate = latents_next - sigma_next * velocity (flow matching, Euler step); never let a preview break generation
            import preview
            sig = float(pipe_.scheduler.sigmas[i + 1]) if i + 1 < len(pipe_.scheduler.sigmas) else 0.0
            lat, vel = kw["latents"].float(), kw["noise_pred_video"].float()
            if vel.ndim == 5:   # the image-to-video pipeline passes the velocity unpacked and without the clean first frame
                lat = LTX2Pipeline._unpack_latents(lat, _stage["F"], _stage["h"], _stage["w"], pipe_.transformer_spatial_patch_size, pipe_.transformer_temporal_patch_size)
                vel = torch.cat([torch.zeros_like(vel[:, :, :1]), vel], dim=2)
            x0 = lat - sig * vel
            preview.save_preview(x0, _stage["F"], _stage["h"], _stage["w"], LTX2Pipeline, vae,
                                 pipe_.transformer_spatial_patch_size, pipe_.transformer_temporal_patch_size,
                                 os.path.join(PREVIEW_DIR, f"{_stage['tag']}_step{i + 1}.png"))
        except Exception as e:
            log(f"  [preview skipped: {type(e).__name__}: {e}]")
    return kw


gen = torch.Generator().manual_seed(SEED) if SEED >= 0 else None
common = dict(
    prompt_embeds=pos_e, prompt_attention_mask=pos_m, negative_prompt_embeds=neg_e, negative_prompt_attention_mask=neg_m,
    guidance_scale=1.0 if DISTILLED else 3.0, stg_scale=0.0, modality_scale=1.0,
    audio_guidance_scale=1.0 if DISTILLED else 7.0, audio_stg_scale=0.0, audio_modality_scale=1.0,
    spatio_temporal_guidance_blocks=None, output_type="latent", return_dict=False,
    callback_on_step_end=on_step, generator=gen, frame_rate=FPS,
)
if PREVIEW_DIR:
    common["callback_on_step_end_tensor_inputs"] = ["latents", "noise_pred_video"]
    pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "noise_pred_video"]
with torch.no_grad():
    if RESUME_S1:
        _d = torch.load(LAT + ".stage1")
        video_lat, audio_lat = _d["video"].to(DEV), _d["audio"].to(DEV)
        log(f"C: resumed stage-1 latents {tuple(video_lat.shape)} from disk")
    else:
        _i2v = dict(image=None, latents=img_latents, noise_scale=1.0) if img_latents is not None else {}
        video_lat, audio_lat = pipe(height=H, width=W, num_frames=FRAMES, num_inference_steps=STEPS, sigmas=SIGMAS, **_i2v, **common)
        if TWOSTAGE:
            torch.save({"video": video_lat.cpu(), "audio": audio_lat.cpu()}, LAT + ".stage1")
    if TWOSTAGE:
        log(f"C: stage 1 done in {time.perf_counter() - t:.1f}s; upsampling latents {tuple(video_lat.shape)} x2")
        from diffusers import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        ups = LTX2LatentUpsamplerModel.from_pretrained(f"{M}/latent_upsampler", torch_dtype=torch.bfloat16).to(DEV)
        LTX2LatentUpsamplePipeline._execution_device = property(lambda self: DEV)
        up_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=ups)
        up_lat = up_pipe(latents=video_lat, output_type="latent", return_dict=False)[0]
        acc.synchronize()
        log(f"C: upsampled to {tuple(up_lat.shape)}")
        del ups, up_pipe, video_lat
        ds.close()
        ds = make_ds(0)
        cleanup()
        _stage.update(tag="C2", n=len(S2_SIGMAS), h=H // 16, w=W // 16)
        t2 = time.perf_counter()
        step_t.clear(); t = t2
        _p2 = pipe
        if IMAGE:
            # the image-to-video pipeline gives every token its own timestep (first-frame conditioning), which needs much more
            # activation memory; the refinement stage uses the plain text-to-video pipeline (stage-1 output already holds the image)
            _p2 = LTX2Pipeline(scheduler=sched, vae=vae, audio_vae=audio_vae, text_encoder=te_full.model, tokenizer=tok,
                               connectors=conn, transformer=dit, vocoder=vocoder)
            object.__setattr__(_p2, "connectors", lambda *a, **k: (c_video, c_audio, c_mask))
            if PREVIEW_DIR: _p2._callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "noise_pred_video"]
        video_lat, audio_lat = _p2(num_frames=FRAMES, num_inference_steps=len(S2_SIGMAS), sigmas=S2_SIGMAS, latents=up_lat,
                                   audio_latents=audio_lat, noise_scale=S2_SIGMAS[0], **common)
log(f"C: denoising done in {time.perf_counter() - t:.1f}s (transformer streamed {ds.bytes / 2**30:.1f} GiB)")
ds.close()
del ds
cleanup()
torch.save({"video": video_lat.cpu(), "audio": audio_lat.cpu()}, LAT)
log("saved latents.pt", tuple(video_lat.shape))

if SKIPDEC:
    sys.exit(0)

# ---------- D. decode ----------
for name in ("pos_e", "neg_e", "c_video", "c_audio", "c_mask"):
    globals()[name] = None
dit = pipe.transformer = None
cleanup()
from diffusers.utils import encode_video

log("D: loading VAE + vocoder for decode")
vae_r = AutoencoderKLLTX2Video.from_pretrained(f"{M}/vae", torch_dtype=torch.bfloat16).to(DEV)
vae_r.enable_tiling()
with torch.no_grad():
    lat = video_lat.to(DEV, torch.bfloat16)
    ts = torch.tensor([0.0], device=DEV, dtype=torch.bfloat16) if vae_r.config.timestep_conditioning else None
    video = vae_r.decode(lat, ts, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
acc.synchronize()
log("D: video decoded", frames.shape)
del vae_r, video, lat
cleanup()
audio_vae_r = AutoencoderKLLTX2Audio.from_pretrained(f"{M}/audio_vae", torch_dtype=torch.bfloat16).to(DEV)
vocoder_r = LTX2VocoderWithBWE.from_pretrained(f"{M}/vocoder", torch_dtype=torch.bfloat16).to(DEV)
with torch.no_grad():
    mel = audio_vae_r.decode(audio_lat.to(DEV, torch.bfloat16), return_dict=False)[0]
    wav = vocoder_r(mel)
sr = vocoder_r.config.output_sampling_rate
encode_video((frames * 255).round().astype("uint8"), fps=24, output_path=OUT, audio=wav[0].float().cpu(),
             audio_sample_rate=sr)
log(f"D: wrote {OUT}")
