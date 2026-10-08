# THE BIBLE — LTX-2.5 (22B, unquantized) on an Intel Arc iGPU

Everything learned between 2026-10-04 and 2026-10-06: what works, how it works, every bug and its fix, what failed, measured numbers,
prompting lessons, and a checklist for porting a newer model. **Read sections 0–3 first; use the rest as reference.**
Companion files: `videos/README.md` (clip index), `prompts/` (all prompts), `BIBLE.md` (this file).
The author of this file (Claude) could not hear audio or see motion; everything about sound/motion quality marked **(user-verified)**
was confirmed by the user, everything else is from stills, spectrograms, speech recognition and numbers.

---------------------------------------------------------------------------------------------------------------------------------

## 0. TL;DR (state at 2026-10-06)

* A **full-precision (bf16, no quantization) 22B LTX-2.5 distilled video+audio model runs end to end on a laptop iGPU with 15.6 GB RAM**
  by streaming weights from NVMe layer by layer. Text→video, image→video, native audio (speech, SFX, music) all work.
* Best practical mode = **FLASH** (12 fps generated, every frame shown twice, 768×448 → 1 refinement step → 1536×896): ~14 min per 5.4 s clip.
  Good for anime ("on twos"). For realistic footage 12 fps judders and moves at half apparent speed → use 24 fps (slower).
* **Audio was silent/garbled for a day because the vocoder returns NaN in bf16. Run audio VAE + vocoder in fp32.** Fixed. Speech is intelligible
  (word-for-word ASR match). (user-verified: audio is good.)
* Custom GPU kernels were attempted and **abandoned** (our GEMMs: 1.5 and 0.7 TFLOPS vs oneDNN 3.5–3.7). Real wins came from pipeline changes
  (12 fps, 1 refinement step, transposed weights) — 47 min → ~14 min.
* The 30-second short "The Lantern Keeper" is **unfinished** (shots 1–4 done, 5–6 pending; a memory-pressure kill stopped it) and the user judged it
  **visually bad because of an art-style conflict** (Ghibli-like girl vs. harder-edged action-anime swordsman). See §9.4.
* Hard rule from the harness: **if Claude Code kills a background job for low memory, do not restart it unless the user asks.**

---------------------------------------------------------------------------------------------------------------------------------

## 1. Goals, user, environment

**User goals:** (1) a showcase: "full unquantized LTX-2.5 on an integrated GPU, 5–8 s HD video in X minutes"; (2) an "anime studio": prompt → 1–10 min
video built from ~5–10 s shots stitched together, with dialogue, BGM, SFX. Wants honesty about limits, speed, and quality; likes autonomy
("don't wait for me") but the harness reaper overrides that after a kill.

**Hardware (measured/queried):**
| Item | Value |
|---|---|
| CPU | Intel Core Ultra 5 125H, 14 cores / 18 threads, AVX2+FMA only (no AVX-512/AMX), 2× 8 GB LPDDR5X-7467 (~120 GB/s theoretical) |
| iGPU | "Intel Arc Graphics" (Meteor Lake) — 112 EUs / 14 subslices, 2.2 GHz, **no XMX/DPAS (no matrix engines), no HW bf16 conversion**, fp16 yes, subgroup sizes 8/16/32, 4 MB L3, 64 KB SLM |
| Memory | **GPU memory IS system RAM** (15.6 GB total; torch reports ~7.9 GiB addressable). Everything the GPU holds is taken from the same pool as Windows/browser |
| Disk | SK hynix 954 GB NVMe; **3.65 GB/s unbuffered read**, 3.1 GB/s write |
| NPU | "Intel AI Boost" present, unused |
| OS | Windows 11 Home 10.0.26200; shell = Git-Bash (+PowerShell); Python 3.14.0 |

**Software (in `.venv`):** torch **2.14.1+xpu** (from `https://download.pytorch.org/whl/xpu`), diffusers **0.40.0**, transformers **5.14.1 (pinned to the
checkpoint's version; 5.18 also ran)**, accelerate 1.15, safetensors 0.8, PyAV 19.0.1, faster-whisper 1.2.1 (ASR checks), pyopencl 2026.1.4,
psutil, numpy 2.5. Triton 3.8 is installed but unusable (see §10).

**Model repo:** `Lightricks/LTX-2.5-Diffusers` on Hugging Face — **gated ("auto")**: the user had to click "Agree" on the model page before any
download worked (401/403 until then). The user's HF login was already cached on the machine. Downloaded to `models/ltx25/` (≈66 GB, took 2 h 51 m ≈ 6.4 MB/s).
Components downloaded: `transformer/` (8 shards), `text_encoder/` (5), `connectors/` (2), `vae/`, `audio_vae/`, `vocoder/`, `latent_upsampler/`, tokenizer, scheduler, configs.
NOT downloaded: `transformer_full/` (the non-distilled SFT DiT, ~35 GB), `prompt_enhancer/` (~9.5 GB), `diffusion_decoder/` (0.78 GB), the 9 GB distilled LoRA
(unnecessary — see §6.1), `temporal_latent_upsampler/`, `duration_head/`.

---------------------------------------------------------------------------------------------------------------------------------

## 2. Project layout

```
C:\Programming\iGPU
  BIBLE.md                  this file
  render.py                 ONE-COMMAND renderer (modes, filing, metadata)  <- use this
  story.py                  6-shot "Lantern Keeper" driver (resumable)
  organize.py               one-time cleanup script (already run)
  stream/                   the engine
    runtime.py              Streamed (hook-based weight streaming), build_empty, transposed-Linear patch
    blockstream.py          NVMe reader, tensor index, aligned repack, prefetching BlockStreamer
    run_ltx.py              the pipeline (text enc -> connectors -> transformer stage1 -> upsample -> stage2), env-var driven
    decode_big.py           staged, memory-lean decode of saved latents -> mp4 (video + fp32 audio)   <- use this
    fix_audio.py            re-decode audio from saved latents and remux into existing mp4 (video untouched)
    stitch.py               join shots (cuts, drop duplicate first frames, audio crossfades)
    transcribe.py           faster-whisper ASR (does speech come out intelligible?)
    audio_report.py / audio_debug.py   audio diagnostics (extract wav, spectrogram, NaN trace)
    decode_to_mp4.py, decode_latents.py   older decoders (decode_to_mp4 hung at 1536x896; kept, fp32 audio fixed)
    check_keys.py, inspect_layout.py, verify_loader.py, debug_*.py   porting/diagnosis tools (see §11)
    (the old *_pre_*.py / run_ltx_cfg_backup.py snapshots and the 4 GB stream/layers.bin scratch file were deleted at the first commit)
  bench/                    microbenchmarks (disk/RAM, GEMM, SDPA, FMA peak, OpenCL GEMMs, Level Zero interop)
  videos/                   all outputs, indexed in videos/README.md
  prompts/                  prompt texts (lantern-keeper/ has the 6 shot prompts + story notes)
  latents/                  saved latents: *.pt (final) and *.pt.stage1 (reusable stage-1 results)
  frames/  logs/  audio/    stills & contact sheets / run logs / extracted wavs + spectrograms
  models/ltx25/             weights (66 GB)         .venv/  Python environment
```
Memory notes for new Claude sessions live in `C:\Users\aswin\.claude\projects\C--Programming-iGPU\memory\` (short; points here).

---------------------------------------------------------------------------------------------------------------------------------

## 3. How to run things (copy-paste)

All commands from `C:\Programming\iGPU`. Use `.venv\Scripts\python.exe`. In Git-Bash set `export MSYS_NO_PATHCONV=1` if you call Windows tools with `/flags`.

```bash
# one clip, filed automatically (videos/<category>/<name>/<mode>_<WxH>_<fps>_<secs>s_seed<N>.mp4 + .json)
.venv/Scripts/python.exe render.py --mode flash --name my-scene --prompt-file prompts/my.txt --seed 7 [--seconds 5.4] [--category anime]
.venv/Scripts/python.exe render.py --mode preview ...                 # stage 1 only (768x448), ~8 min: pick prompt/seed before a full render
.venv/Scripts/python.exe render.py --mode flash ... --image frame.png # image-to-video: frame.png becomes the first frame
.venv/Scripts/python.exe render.py --mode flash ... --reuse-stage1    # skip stage 1 if latents/<tag>.pt.stage1 exists
.venv/Scripts/python.exe story.py                                      # resume the Lantern Keeper film (skips finished shots)
.venv/Scripts/python.exe stream/transcribe.py clip.mp4 [small.en]      # does the speech come out right?
.venv/Scripts/python.exe stream/fix_audio.py "video.mp4=latents/x.pt"  # re-decode audio from latents (fp32) and remux
```
**Modes (`render.py MODES`)** — times are wall-clock for a 5.4 s clip incl. decode, Balanced power plan, Brave closed or one tab:
| Mode | Stage 1 → final | fps | Refine steps | Time | Use |
|---|---|---|---|---|---|
| **flash** | 768×448 → 1536×896 | 12 (×2 dup) | 1 | **~14 min** | anime finals |
| flash2 | same | 12 | 2 | ~18 min | marginally more refinement |
| turbo | 640×352 → 1280×704 | 12 | 1 | ~9.5 min | slow scenes / quick looks; fast action goes to mush |
| preview | 768×448, no refinement | 12 | – | ~8 min | choose seed/prompt |
| audiotest | 512×320, no refinement | 12 | – | ~4.8 min | dialogue/audio checks (video is soft by design) |
| audiohd | refine a saved audiotest → 1024×640 | 12 | 1 | ~3.8 min | (copy the audiotest latents to `<name>_audiohd_seedN.pt.stage1`, pass `--reuse-stage1`) |
| reference | 768×448 → 1536×896 | 24 | 3 | ~47 min | best quality, slow; realistic needs ≥24 fps |

`run_ltx.py` env vars (render.py sets them): `PROMPT, SEED, W, H, FRAMES (must be 8k+1), FPS, TWOSTAGE, S2STEPS, SKIPDEC, LAT, RESUME, IMAGE, BOS=1, MAXSEQ=512, TRANSPOSE=1, FFN_CHUNK=4096, DISTILLED=1`.

---------------------------------------------------------------------------------------------------------------------------------

## 4. How the engine works

**Problem:** 22B bf16 = 35.4 GiB transformer + 22.3 GiB text encoder + 6 GiB connectors, on a machine whose GPU can address ~7.9 GiB.
**Idea:** keep weights on NVMe, stream one block at a time while the previous block computes.

1. **Build empty.** `runtime.build_empty` constructs HF/diffusers models under `accelerate.init_empty_weights(include_buffers=False)` with default dtype bf16:
   parameters are on the *meta* device (zero memory), small buffers (RoPE tables etc.) are real and moved to the GPU.
2. **Index the checkpoints.** `blockstream.ShardIndex` parses safetensors headers (8-byte length + JSON) → every tensor's file/offset/dtype/shape.
   `Plan` groups the tensors of one *unit* (a transformer block, a Gemma layer, a connector block) into aligned read runs (merge gaps ≤1 MiB;
   **a block's tensors are scattered across shards, so min→max range reads are wrong** — see B03).
3. **Stream.** `Streamed` registers a forward-pre-hook (load the unit's weights into the module) and a forward-hook (synchronize, restore meta params,
   release the slot) on each unit module. Two loaders: **prefetch** (`BlockStreamer`: reader thread → pinned slots → H2D on a side stream, overlaps with compute)
   and **sync** (`_load_unit_direct`: chunked unbuffered reads straight into the aligned slot, no prefetch buffers, ~3 GB less RAM).
   Reads use Windows `FILE_FLAG_NO_BUFFERING` via ctypes (page-cache bypass → true NVMe speed, no cache bloat).
4. **Alignment is mandatory.** Tensors sit at arbitrary byte offsets in shard files; viewing them directly on the GPU gives NaN/faults.
   `aligned_unpack` copies each tensor into a 256-byte-aligned slot (see B09).
5. **Resident weights.** Everything not inside a unit (embeddings, norms, small projections, ~0.8 GiB for the DiT) is loaded once and stays.
   Checkpoint **F32 tensors are cast to bf16** (matches `from_pretrained(dtype=bf16)`). Checkpoint **buffers** (Gemma `layer_scalar`) are loaded as buffers.
6. **Transposed Linear weights** (`transpose_linear=True`, default): each 2-D Linear weight is stored `[in,out]` and `forward = addmm(bias, x, Wt)`;
   +~9 % matmul speed, **bit-identical output** (verified byte-for-byte on a full clip).
7. **Phases (each frees its GPU memory before the next):** A text encoder (48 Gemma layers, 20 GiB streamed, ~15 s) → B connectors (5.9 GiB, ~5 s; outputs cached) →
   C stage 1 transformer (8 steps, prefetch streamer) → upsample latents ×2 (0.95 GB upsampler) → stage 2 (sync streamer, `FFN_CHUNK`) → D decode (VAE, audio VAE, vocoder, mp4).
   The pipeline objects are real diffusers pipelines with `_execution_device` monkey-patched and `connectors` replaced by a lambda returning the cached outputs.
8. **Two-stage:** stage 1 at W×H (8 distilled sigmas), `LTX2LatentUpsamplePipeline` ×2, stage 2 re-noises (sigma 0.909375 / 0.725 / 0.421875 — `S2STEPS` keeps the last N) and refines.
   Stage-1 latents are saved to `LAT + ".stage1"` so a stage-2 failure never costs stage 1.
9. **Image-to-video:** encode the image with the real VAE encoder (CRF-18 re-compression like the pipeline), put it in frame 0 of 5-D `latents` repeated over all frames, `noise_scale=1.0`
   (frame 0 stays clean via the conditioning mask), use `LTX2ImageToVideoPipeline` for stage 1. **Stage 2 uses the plain T2V pipeline** (see B20).
10. **Decode** (`decode_big.py`): VAE decode (tiled) → convert to uint8 *on the GPU in 16-frame chunks* → CPU; audio VAE + vocoder in **fp32**; PyAV mux (H.264 + AAC).

---------------------------------------------------------------------------------------------------------------------------------

## 5. Measured numbers

**Microbenchmarks** (`bench/`):
* NVMe seq read 3.65 GB/s (unbuffered); single-thread memcpy 10 GB/s; GPU device copy 52 GB/s (read+write); host→GPU 19–20 GB/s pinned (16–64 MB chunks) vs 2.7–5.8 pageable.
* GEMM (11440×4096×4096): fp32 3.66, fp16 3.67, bf16 3.67 TFLOPS (all the same → oneDNN `jit:gemm:any` runs fp16/bf16 at the fp32 rate). `x @ W.T` with W[N,K] is ~9 % slower than `x @ Wt` (contiguous [K,N]).
* Raw FMA peak (OpenCL): **fp32 3.93, fp16 7.77 TFLOPS** — true 2× fp16 exists in hardware, nothing off-the-shelf uses it.
* SDPA is a fused kernel: ~1.9–2.0 TFLOPS bf16 (fp16 slower, 1.66), extra memory only 0.13 GiB at 17k tokens → **no custom attention kernel is needed for memory**.
* Power plan: Balanced beat "High performance" by ~7 % in raw GPU throughput (4 alternating runs each; shared CPU/GPU power budget). Run-to-run noise ±5 %.

**Cost model (distilled, batch 1, per transformer step):** `t(N) ≈ 14.5·N + 0.428·N²` seconds, N = video tokens in thousands
(tokens = latent_frames × H/32 × W/32, latent_frames = (frames−1)/8+1). Fitted before the transposed-weight patch (−5–10 % since). Below ~1.5k tokens a step is I/O-bound (~12.5 s = reading 35 GiB).
Observed: 384 tok 12.5 s; 1.1k 16 s; 2.4k 43 s; 3.0k 41–52 s; 5.4k (768×448×121f) 74–85 s; 11.4k 222 s; 12.1k 219 s; 21.5k (1536×896×121f) 560–583 s.
Attention is ~34 % of a step at 21k tokens. Each extra guidance pass (STG / modality, on by default in the pipeline) re-streams 35 GiB — the distilled recipe turns them off.

**Phase times, FLASH 5.4 s clip:** text+connectors 20 s · stage 1 ~375 s · upsample 6 s · stage 2 (1 step) ~220 s · decode ~170 s (VAE 140 s, audio 30 s, export few s).

---------------------------------------------------------------------------------------------------------------------------------

## 6. The model: facts and the *recipe*

### 6.1 Distilled by default
`transformer/` in this repo **is the distilled DiT** (model_index default). The 9 GB "distilled LoRA" is for the *dev/SFT* model and is not needed.
**Distilled recipe (mandatory):** `sigmas = DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]` (8 steps),
`guidance_scale = 1.0, audio_guidance_scale = 1.0, stg_scale = 0, audio_stg_scale = 0, modality_scale = 1, audio_modality_scale = 1`, batch 1, positive prompt only.
Stage 2 sigmas `[0.909375, 0.725, 0.421875]`, `noise_scale = first sigma`. The pipeline *defaults* (guidance 3, STG block 28, modality 3, 30 linear steps) are for the SFT model and give 3 passes per step
and off-recipe results — an early mistake here (B10).
Architecture facts: 48 blocks × 738 MiB (84 tensors each: video, audio and cross-modal attention + FFNs), hidden 4096, 32 heads (head dim 128), FFN 16384; video VAE compresses 8× in time and 32×32 spatially
(**FRAMES must be 8k+1; H, W multiples of 32**); text encoder = Gemma-4 "Unified" 12B (48 layers, 3840 hidden, sliding-window + global layers, k=v on global layers); prompt embeddings = all 49 hidden states stacked
(188,160 features/token) → connectors (per-modality projections + learnable registers); audio VAE latent (8 ch × 16 mel) ; vocoder = `LTX2VocoderWithBWE`, 48 kHz stereo.
Duration/length: the pipeline allows up to 20 s per call; we stay ≤5.4 s per clip for time/memory.

### 6.2 Tokenizer
The shipped tokenizer template adds **no `<bos>`**. We set `tok.add_bos_token = True` (`BOS=1`). (Its isolated effect was never proven — see B12 — but keep it.)

---------------------------------------------------------------------------------------------------------------------------------

## 7. AUDIO (read this before touching sound)

* **Root cause of silent/garbled audio: `LTX2VocoderWithBWE` outputs NaN in bf16.** The NaN waveform was encoded as digital silence (-180 dBFS) in almost every clip; a few early clips had quiet hiss.
  **Fix: load `AutoencoderKLLTX2Audio` and the vocoder in `torch.float32`** (the mel stage is fine in bf16, fp32 mel differs negligibly). GPU fp32 == CPU fp32 exactly.
  Implemented in `decode_big.py`, `decode_to_mp4.py`, `fix_audio.py`. All old clips were repaired from their saved audio latents (`fix_audio.py`) — video frames untouched.
* Diagnosis path worth reusing: `audio_report.py` (RMS/peak/silence/centroid per clip) → `audio_debug.py` (stats at latent → mel → wav in bf16/fp32/CPU).
* **Speech works.** Put spoken lines **in double quotes** with a voice description in the prompt; faster-whisper (`base.en`, `small.en`) transcribed every line word for word, with correct timing,
  also after refinement and in image-to-video. One short line per ~2–3 s is comfortable; 3 sentences fit in 5.4 s.
* LTX guidance: describe the sound explicitly ("anything left out of the prompt gets invented"): ambience, SFX, music, voice quality. The audio branch is conditioned jointly with video.
* Spectrograms showed structured events (footstep pulses, rain-like noise bed, rhythmic bass); many clips are low-frequency heavy (80–97 % of energy < 150 Hz) — the user judged them fine.
* Per-clip audio does not match across shots (each shot invents its own BGM) → `stitch.py` equal-power crossfades 0.3 s. A consistent score across a 30 s film would need a separate music track (not built).
* Audio is generated as latents together with video; it cannot be skipped without dropping the audio stream of the DiT.

---------------------------------------------------------------------------------------------------------------------------------

## 8. Image-to-video (works) and chaining shots

* Works with the first frame = given image (first-frame difference vs input ≈ 3/255 after the VAE round trip), identity continues, speech works (`videos/tests/i2v-test`).
* Chained shots: next shot's `--image` = last frame of an earlier shot (extract with PyAV, full res; run_ltx resizes to stage-1 size). The first 2 output frames of a chained clip duplicate the previous
  last frame → `stitch.py drop_first_frames=2` for seamless continuation. Hard cuts: drop 0.
* **Memory caveat (B20):** `LTX2ImageToVideoPipeline` multiplies timesteps per token (conditioning mask) → too much activation memory at the 12k-token refinement stage. Use it for stage 1 only.
* Because there is no text-to-image model here, "reference images" for character design must come from frames of earlier generations.

---------------------------------------------------------------------------------------------------------------------------------

## 9. Prompting and quality lessons

### 9.1 Prompt format (confirmed by LTX/fal/RunDiffusion guides and by our results)
One flowing paragraph, present tense, 4–8 sentences (~150–200 words). Order that worked: **style sentence first** → setting → characters with *observable* traits (hair, scar, clothing colors) → one action →
camera move in prose → **sound sentence last** (speech in quotes). No tags, no weights, no mood words ("tense atmosphere"). Negative prompts do nothing (guidance 1.0). Avoid text/signage.

### 9.2 Motion
* **Slow, deliberate motion + smooth camera glide / push-in = clean frames** (ronin-entrance: frame-to-frame change ≈2.4, no collapse).
* **Fast camera whips, "quick", "fast sword stroke", "speed lines", "motion smear" = paint-smear collapse** for ~2–3 unique frames at the fastest moments (swordsman-bridge). Our own prompt asked for the smear.
  The refinement stage cannot repair frames that are already broken in stage 1.
* At 12 fps one latent step covers 0.67 s (VAE is 8× temporal) so big swings have less temporal detail.
* Faces in wide shots need pixels: at 640×352 stage 1 they turn to mush; a close-up at the end of a push-in is always the cleanest frame.

### 9.3 Resolution / modes
* Softness is resolution, not fps: 512×320 stage-1-only looked soft; refining to 1024×640 (3.8 min) was clearly sharper with identical dialogue.
* 1 refinement step ≈ 2 steps in quality (keeps stage-1 content closer); 3 steps re-noise heavily (details change, e.g. a sword vanished).
* 12 fps doubled: fine for anime. **Realistic** (fox): stills equally good, but 12 Hz stepping (every other frame identical) and ~½ the apparent motion speed (feels floaty). Use native 24 fps for realistic finals
  (a "flash24" mode was not built; estimate ~23 min per 5 s).
* Greasy/scratchy hatching on textures in wide shots persists in every mode; only more pixels/steps reduce it. The untried ideas: `diffusion_decoder/` decode, non-distilled model with CFG (≈1.5 h/clip).

### 9.4 Art-style consistency across shots (the reason "Lantern Keeper" failed)
The user judged the 20 s partial film bad because **the girl looked Ghibli-soft while the swordsman looked like harder-edged action anime**. Both prompts shared one style sentence, but the *character descriptors*
("small girl … copper bob … yellow raincoat" vs "tall swordsman … scar … crimson eyes") pull the model toward different design languages (and I2V from a frame of one character then cut to the other keeps each in its own language).
**Next time:** (a) pick ONE reference style per film and name it *per character* ("in the same hand-drawn Studio-Ghibli film style" for BOTH, or "dark action anime" for both);
(b) keep characters in the same age/design family (a child + a scarred warrior pulls styles apart — consider two adults or two similar designs);
(c) make a style-test pair (one frame of each character via `preview`) and compare side by side BEFORE committing 80 minutes;
(d) put the style words last as well as first, and avoid per-character style hints that differ; (e) check geometry too: shot 2 had the swordsman emerge from the temple gate instead of climbing the steps.

### 9.5 Recipes that worked (see `videos/README.md` for the full index)
* `prompts/anime_ronin-entrance.txt` seed 77 — best action-anime clip (slow walk toward camera, glide back, push-in on glowing eyes). FLASH 13.6 min / TURBO 9.7 min.
* `prompts/anime_ghibli-girl-yellow-raincoat.txt` — cleanest 2D look (one subject, flat colors, watercolor background, slow push-in).
* `prompts/test_dialogue_cafe.txt` seed 5 — realistic two-person dialogue, perfect ASR.
* `prompts/test_i2v_ronin-ignite.txt` + `frames/i2v/ronin_last_frame.png` seed 9 — I2V with spoken line.

---------------------------------------------------------------------------------------------------------------------------------

## 10. THE MISTAKE / BUG LOG (symptom → cause → fix). Do not re-learn these.

| ID | Symptom | Cause | Fix |
|---|---|---|---|
| B01 | torch had no GPU | CPU wheel installed system-wide on Python 3.14 | venv + `pip install torch --index-url https://download.pytorch.org/whl/xpu` (retry with `--retries`; one DNS failure mid-download). Pass `--timeout` |
| B02 | 401/403 on the model repo | gated repo, license not accepted | user clicks "Agree" on the HF page; then `snapshot_download(local_dir=..., allow_patterns=...)` |
| B03 | "max block read 5482 MiB / 143 GiB total" | a block's tensors are scattered over shards; I read min→max offset per block | per-tensor runs, merge gaps ≤1 MiB (`Plan`) |
| B04 | `ReadFile` error 87 | unbuffered I/O needs sector-aligned offsets/sizes/addresses; last chunk past EOF returns short | align to 4096, stop on short read |
| B05 | OOM while loading a 1.9 GB embedding | staged through a full-size pinned buffer + GPU copy; pinned memory is never returned | one shared 64 MB pinned staging buffer, chunked reads into the GPU tensor |
| B06 | buffers not freed between phases → OOM | forward hooks keep the `Streamed` object alive | explicit `close()` removes hooks and drops buffers; call it every phase |
| B07 | OOM at start of denoising | my "fill missing params with zeros" loop also hit every streamed block param | restrict to non-block params |
| B08 | `mat1 and mat2 must have the same dtype (Float, BFloat16)` | a few checkpoint tensors are F32 | cast F32→bf16 on load |
| B09 | NaN everywhere / `OUT_OF_RESOURCES` on trivial ops | tensor views at arbitrary byte offsets (misaligned) in the raw read buffer | repack each tensor into a 256-byte-aligned slot (`aligned_unpack`). Found via per-block finite checks |
| B10 | off-recipe results, 2× compute | used guidance 3.0 + linear sigmas on the **distilled** model | README: distilled sigmas, guidance 1.0, no STG/modality; `DISTILLED=1` default |
| B11 | tokenizer output lacks BOS | shipped TemplateProcessing has no bos | `tok.add_bos_token = True` |
| B12 | videos ignored the prompt (random people) | **unisolated**: fixed after (a) direct aligned loader for the text stages, (b) transformers==5.14.1, (c) BOS | keep all three; always sanity-check with two prompts + fixed seed |
| B13 | "LM test" said the text encoder is broken (same top token for every prompt) | last hidden layer (48) is collapsed by design; LTX uses intermediate layers | judge encoders with a topic-gap test over layers (`debug_topics.py`), not next-token prediction. Also: CPU-vs-GPU per-layer comparison (`debug_gpu_vs_cpu.py`) and byte-exact loader check (`verify_loader.py`) |
| B14 | stage 2 OOM / killed | prefetch buffers (~3 GB) + 12k+ tokens | stage 2 uses sync streaming (`make_ds(0)`), `FFN_CHUNK=4096` (chunked feed-forward) |
| B15 | decode hung at 1536×896 (6 GB RAM, 0 CPU for 14 min) | built huge float32 numpy arrays | `decode_big.py`: uint8 on GPU in chunks, staged logs |
| B16 | RAM exhausted early | loading VAEs/vocoder eagerly | build VAEs empty + load only `latents_mean/std`; load real weights only at decode |
| B17 | JSON open failed on Windows | cp1252 default | always `open(..., encoding="utf-8")` |
| B18 | tooling quirks | `grep` buffers output into pipes (hides logs); `powercfg /setactive` fails in Git-Bash (path conversion); long `sleep` blocked | write logs to files; `MSYS_NO_PATHCONV=1`; use background jobs + notifications |
| B19 | silent / garbled audio in all clips | **vocoder NaN in bf16** | fp32 audio VAE + vocoder (§7) |
| B20 | image-to-video OOM in stage 2 | per-token timesteps (conditioning mask) in `LTX2ImageToVideoPipeline` | refinement stage uses plain `LTX2Pipeline` |
| B21 | PyAV errors | `add_stream(template=)` removed → `add_stream_from_template`; AAC needs float32 `fltp`; streams must be added before the first packet is muxed | fixed in `fix_audio.py`, `stitch.py` |
| B22 | "High performance" plan slower | CPU boost steals power budget from the iGPU | stay on Balanced |
| B23 | I mislabeled `dog.mp4`/`fruit.mp4` as distilled-recipe | they predate it (guidance on, linear sigmas) | renamed `…early-recipe` |
| B24 | `shot` job killed ("low memory") twice at model-loading start | Claude Code reaper (system-critical RAM) | do not auto-restart; free apps; user can set `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` at Claude Code launch |
| B25 | `story.py` stage-1 wasted | I2V stage 2 OOM (B20) cost 7.5 min | now `--reuse-stage1` always passed |

**Dead ends (do not repeat):**
* **Triton** (installed with torch-xpu): needs Level Zero C headers + an MSVC-style toolchain → fails to compile its launcher on this machine.
* **Custom OpenCL fp16 GEMMs** (`bench/cl_gemm.py` v0: 1.5 TFLOPS; `cl_gemm2.py` v2 with shuffles/SIMD8: 0.72, 8-row tiles spill to 0.1) vs oneDNN 3.5–3.7. Load-bandwidth-bound; would need SLM tiling + a profiler/disassembler.
  Even a win is worth only ~15 % of total time. **Level Zero interop does work** (`bench/l0_interop.py`: a kernel compiled by the OpenCL driver, loaded with `zeModuleCreate(NATIVE)` in a fresh L0 context, runs correctly on torch tensors' device pointers) — so a custom kernel *could* be integrated if someone writes a faster one. (OpenCL itself rejects torch's pointers, error −50.)
* fp16 numerics instead of bf16: no speed gain (oneDNN runs both at fp32 rate) and SDPA fp16 is slower.
* NPU / CPU co-processing: not tried (low payoff, high complexity).

---------------------------------------------------------------------------------------------------------------------------------

## 11. PORTING A NEWER MODEL — checklist and caveats

1. **Index first.** `stream/inspect_layout.py <component>`: dtypes, per-block sizes, tensor counts. Is the block group regular (`.N.`)? Are there tensors outside blocks (go resident)? F32 islands?
2. **Match names.** `stream/check_keys.py`: model param names vs checkpoint keys; expect extras (vision/audio towers), missing tied weights (`lm_head`), **buffers stored in the checkpoint** (`layer_scalar`) and params missing from it (`keyframes_abs_pos_embedding`).
3. **Meta init** with `include_buffers=False`; check which buffers are computed in `__init__` (RoPE) vs loaded.
4. **Byte-exact loader test** (`verify_loader.py`), then **CPU vs GPU per-layer test** (`debug_gpu_vs_cpu.py`; relative diff ≈1–5 % is bf16 noise, not a bug).
5. **Alignment** (§4.4) — always repack; test with finite checks per block.
6. **Run every convolutional/vocoder/codec stage in fp32 first** (bf16 NaN'd the vocoder here). Add `isfinite` checks at every stage boundary; log stats.
7. **Memory budget:** GPU memory = system RAM. Phases must free everything (`close()`), keep one shared pinned staging buffer, avoid float32 copies of big tensors (embedding tables!), chunk token-dimension ops (`FFN_CHUNK`), prefer sync streaming for the biggest stage.
   Expect Windows + a browser to cost 4–6 GB; the reaper kills jobs when free RAM hits ~0 at model-loading peaks.
8. **Read the model card's recipe** (sigmas, guidance, resolutions, defaults) — pipeline defaults may be for another variant. Count transformer passes per step (STG/modality/CFG each re-stream the whole model).
9. **Pipeline internals that bite:** per-token timesteps (I2V conditioning), duration head, text max length/padding (we use 512), scheduler config (dynamic shifting off for distilled), `frame_rate` affects RoPE and apparent motion speed, VAE temporal factor (8) → frame counts 8k+1.
10. **Tokenizer:** check BOS/chat template yourself.
11. **Objective checks that saved us:** topic-gap similarity over layers (text), ASR transcript (speech), per-frame sharpness/change metrics + contact sheets (collapse frames), byte-identical output comparisons (patch safety), cost-model fit (is it I/O- or compute-bound?).
12. **Throughput expectation:** compute-bound ≈ 2.7–3.4 TFLOPS effective on this iGPU. Time per step ≈ (2 × params × tokens + attention) / 3 TFLOPS; disk read floor = model bytes / 3.65 GB/s per pass. A 2× bigger model ≈ 2× slower; quantizing weights to 8-bit would halve I/O and is the lever *if* a kernel could use it (not available here).
13. **Windows details:** unbuffered I/O via ctypes `CreateFileW(FILE_FLAG_NO_BUFFERING)`, `ReadFile` in ≤64 MB chunks, sector alignment 4096; `SetFilePointerEx`; keep file handles per shard.
14. **Gated repos / downloads:** accept the license in the browser; downloads were ~6 MB/s — plan hours and make them resumable (`snapshot_download` resumes).

---------------------------------------------------------------------------------------------------------------------------------

## 12. Open items and suggested next steps (ordered)

1. **Redo "The Lantern Keeper" with a consistent art style** (§9.4): style test pair first (`preview`, 8 min each), then FLASH shots. Current assets: shots 1–4 rendered (13.6, 7.3, 14.4, 13.8 min), prompts in `prompts/lantern-keeper/`, partial film 21.5 s,
   shots 5–6 never rendered (killed at start of shot 5). `story.py` resumes; if the style is redone, change prompts + seeds and delete `videos/anime/lantern-keeper/shots/*` + `frames/lantern-keeper/*` first.
2. Fix shot geometry (swordsman climbs the steps from below; keep "scar" as a thin line, not a patch).
3. **Realistic finals at 24 fps** ("flash24": 24 fps, 768×448→1536×896, 1 step; estimate ~23 min/5 s) — build as a `render.py` mode and verify smoothness.
4. Try `diffusion_decoder/` (0.78 GB) for sharper decode of existing latents (cheap: re-decode only).
5. Non-distilled `transformer_full/` with CFG for best motion (needs ~35 GB download; est. ≥1.5 h per 5 s clip) — only for hero shots.
6. A local **story planner** (LLM) → shot list → `story.py`; optional separate music bed + SFX mix; subtitle/credits; consistency tests.
7. Speed (optional): custom packed-fp16 GEMM through Level Zero interop (max ~15 % total gain, high risk); `flash` with 8 stage-1 steps is already the floor for the distilled schedule.
8. Housekeeping: delete `*_pre_*.py` snapshots, `bench/__pycache__`, stale latents when disk is needed (366 GB free at last check).

---------------------------------------------------------------------------------------------------------------------------------

## 13. Timeline (for orientation)
2026-10-04: concept discussion (can an iGPU train/run big models?); hardware benchmarks; streaming prototype on a fake model; model access/download (2 h 51 m).
2026-10-05: engine built; bugs B03–B16; first correct prompt-following clip; distilled recipe discovery; two-stage; HD showcase (4 s 1280×704, 20.6 min); anime tests; power-plan A/B; custom-kernel attempts; FLASH/TURBO modes (47 → 14 min); decode hang fix.
2026-10-06: project reorganized; `render.py`; ronin-entrance; **audio bug found and fixed (B19)**; dialogue + ASR; image-to-video; realistic-vs-FLASH comparison; Lantern Keeper shots 1–4 (B20, B24); this document.

---
## 14. Update 2026-10-06 (later): experiments, CUDA port, friend setup

- **Experiments (swordsman-bridge, seed 7):** `preview8` (8 fps tripled, 6 steps: sigmas 1.0,0.9875,0.975,0.909375,0.725,0.421875) = 25 s/step, 4.4 min total vs ~9 min — user judged quality "meh", dropped. `nativehd` (single stage 1536x896, 8 steps, 17 frames) = 48-65 s/step, 10.2 min for 1.4 s; first half crisp, second half collapsed (fast motion compressed into 1.4 s) -> true HD does NOT remove fast-motion smear; confounded by the short clip, no same-settings FLASH A/B was run.
- **Device abstraction:** `stream/dev.py` picks CUDA if available else XPU (`LTX_DEVICE` overrides); `LTX_MODELS` or `ltx_config.json` sets the model folder. All `torch.xpu.*` calls became `acc.*`. Verified bit-identical loader + full smoke render on XPU after the refactor. **CUDA path is untested** (no NVIDIA card here).
- **New render modes:** `preview8`, `nativehd`, `smoke` (384x256, setup test).
- **Probe + setup for friends:** `friend_probe.py` (hardware report), `setup_ltx.py` (11 resumable, self-verifying stages: preflight, model folder, venv, torch, deps, HF login/license pause, download, GPU self-test, loader self-test, smoke render, FLASH benchmark), `make_pack.py` -> `ltx_igpu_pack.zip`. Bug found while testing: console code page cp1252 crashed on unicode check marks -> ASCII only + tolerant Tee.
- **Friend hardware (probe reports):** RTX 4060 laptop 8 GB, 27.8 GB RAM, D: NVMe 3.25 GB/s (use D:, C: is 2.0 GB/s) -> estimated FLASH ~4 min, native HD ~6-7 min, 22.8 TFLOPS bf16; RTX 2050 4 GB, 15.7 GB RAM, 27 GB free disk, 1.57 GB/s -> cannot hold the 66 GB model.
- **Estimates not measured:** int8/DP4A could give maybe 1.5-2x on matmuls (unproven; weight-only quantization gives ~0 speedup on this iGPU).

### 14b. Emberfall (59.6 s film), 2026-10-06
`story_emberfall.py` + `prompts/emberfall/shot1-9.txt`: existing ronin-entrance (FLASH) + i2v-test (768x448, upscaled) opening, then 9 independent text-to-video FLASH shots (hard cuts, seeds 201-209, no frame chaining). Render 131 min total (14.1-15.3 min/shot), no memory kills, 1430 frames 1536x896, 60.8 MB. Same style sentence + verbatim character descriptions in every prompt gave a coherent look (warlord is bolder-inked than the ronin: mild mismatch only). ASR confirmed all 6 spoken lines. Weak spots seen in frames: shot 5 (low-angle hammer) is mostly a black silhouette; shot 7 (clash) frames show both fighters standing facing each other, no visible impact at the sampled frames; hero details drift between shots (scar becomes a black patch in close-up; coat lining missing in shot 6). Final: `videos/anime/emberfall/emberfall_60s_FLASH_1536x896.mp4`.

### B26 (2026-10-06) Audio drifted ahead of picture in stitched films
`stream/stitch.py` used an overlapping equal-power crossfade, which shortens the audio by xfade_ms at every join (0.3 s x 10 joins = 3 s in Emberfall), and each clip's audio is ~0.05 s shorter than its video; the track was then padded with silence at the end. Symptom: speech 3+ s early (shot 8's line heard at 45.9 s, shot starts at 48.7 s). Fix: each clip's audio is trimmed/padded to its exact frame count and placed under its own video; only a 150 ms fade-out/fade-in at each cut. Verified with ASR timestamps vs shot start times. Films stitched before this fix (Lantern Keeper partial, earlier merges) have the same drift. Lesson: verify A/V sync of any concatenation against known event times, not just total duration.

### 14c. Snow Bridge (59.6 s film), 2026-10-06 - planned T2V/I2V mix
`story_snowbridge.py`, `prompts/snowbridge/`. 11 FLASH shots, 158.6 min render (+~15 min for one rejected take + a stopped partial). Humanoid hero + humanoid rival (NOT a monster) in the bridge-clip style; prompt template = style sentence, place, character sentence (verbatim), short action sentences, camera, dialogue, sound. I2V when the same subject continues (2<-1, 4<-3, 8<-7) and for cut-backs (5<-2, 10<-5); T2V at hard cuts to a new subject/place. Start frame = sharpest clean frame in a window of the source clip (not blindly the last one). Result: identity/style consistent across I2V shots; T2V cuts changed the bridge (wooden-railed vs stone arch).
**Motion QC (new, calibrated):** median frame-to-frame change (real frames, gray, 1/4 res): <8 = static, 8-38 = good, >38 = chaotic smear (liked bridge clip 27.5; acceptable clash 16.6; smeared shot 46.9). Sharpness does NOT detect smear (smear has high edge energy). Failed motion shots retry once with `shot{N}_alt.txt` and a new seed.
**Prompt lesson:** the phrases "dynamic speed lines and a subtle motion smear", "quick half-circle", "charges ... leaps" produced a collapsed smear (shot 7 take 1, median 46.7); a medium-shot version with "steps forward and slashes", "tracks slowly around them in a half-circle", no smear wording gave 28.1 and stayed readable.
**Whisper note:** the ASR adds "Thanks for watching!" on music-only tails (no-speech prob 0.67) - ignore it.

### 14d. LTX Studio (hand-guided UI), 2026-10-06
`studio.py` (+ `start_studio.bat`): Tkinter front-end, ~50 MB RAM (a browser tab is several times that). Fields: optional start image (auto crop-to-fill / pad to the stage-1 aspect), prompt, preset, seconds, gen fps (12/24), stage-1 size, stage-1 steps (8 recipe / 6), refinement steps 0-3, seed, name, "reuse stage 1" (re-run refinement only). It launches `render.py --mode custom` (new CLI overrides: `--w --h --fps --s2 --s1sigmas`), tails the log, and shows phase + progress bar + time left (cost-model estimate, refined by measured step times; default Flash predicted 14m13s vs measured 14.1 min), the log, and a **live denoising preview**.
**Live preview:** `stream/preview.py` + hook in `run_ltx.on_step`: after every step, x0 = latents_next - sigma_next * velocity (flow matching), unpacked, denormalized, mapped to RGB by a linear 128->3 map fitted on our own rendered clips (`stream/fit_latent_rgb.py` -> `stream/latent_rgb.npy`, held-out R^2 0.84, MAE 0.064). It is a colour/composition SKETCH of first/middle/last frame (latent grid is 24x14 per frame at 768x448), not a decode. Distilled schedule means the picture only "snaps in" at steps 6-8 (sigma 0.909 -> 0.725 -> 0.42); steps 1-5 look like fog. The I2V pipeline hands the callback the velocity unpacked and without the clean first frame (handled). Previews land in `previews/<tag>/` (git-ignored). A VAE decode per step was rejected: it would hold 1.4 GB more RAM during stage 1 (reaper risk).
Verified: full UI-driven runs (T2V smoke and I2V smoke with a non-16:9 start image) incl. screenshots (`studio.py --auto-test <dir> [--image p]`, `--shot <png>`).

### 14e. Ye Chen episode: art-style test (2026-10-06)
Design refs live in `episodes/ye-chen-ep1/characters/` (`*_gpt.png` = rich painterly manhwa/donghua style, `*_nano.png` = Gemini, flat cel). I2V stage-1 test (768x448, preview mode, same prompt/seed 31, grey-background 3/4 medium crop of the Gu Changge sheet): GPT style kept brocade, hair strands and face through a push-in (median frame change 9.8, max 14, sharpness 5.2); Gemini flat style also stable but flatter/less motion (6.3, max 8, sharpness 4.0). So the richer style did NOT hurt the video model in this easy case (single character, plain background, slow motion). NOT yet tested: HD refinement on this style (greasiness risk), real backgrounds, two characters, speech. Gemini's own prompts had to be fully expanded - a bare "STYLE." placeholder was pasted literally once.
HD refinement on the GPT style (FLASH, 14.2 min, same image/prompt/seed): edge sharpness 3.7 vs 2.7 for the stage-1 clip enlarged with bicubic, motion unchanged (median 9.6 vs 9.8), no greasiness or smear in face/hair/brocade crops; side effect: skin/face shading becomes slightly smoother and more semi-realistic than the painted original (identity stays within the clip). True 1080p path = stage-1 960x544 + the fixed x2 latent upsampler = 1920x1088 (crop 4 px top/bottom): ~4.6k stage-1 tokens, ~18.4k stage-2 tokens (formula: ~10 min + ~7 min + decode, ~25 min/clip; stage 2 at that size never run here, OOM/reaper risk).

### 14f. Dense wide shots need a higher stage-1 size (Ye Chen ep1 shot 01, 2026-10-07)
At 800x448 the video VAE cannot hold hundreds of 4-6 px figures (VAE round trip of the source: 27.3 dB at 1600x896 vs 22.7 dB at 800x448): frame 0 already smears, stage 2 then invents noisy detail ("grain"). Refinement with 3 steps (sigma 0.909 start) is crisper but RE-IMAGINES the scene (frame-0 PSNR 14.2, different clothes/props): unusable for fixed designs. Fix = stage 1 at 1024x576 -> 2048x1152: crowd/floor pattern much cleaner, frame-0 PSNR 21.2, but 39.5 min for 81 frames (stage 1 16.2, stage 2 15.9, decode 6) vs 18.1 min. Needs SYNC_S1 (synchronous streaming) automatically when W*H*frames > 900*448*65, otherwise OUT_OF_RESOURCES at the first block. Pre-flight (bench/preflight_detail.py: VAE round trip at stage-1 size vs source) flagged shots 1, 11, 5A as < 25 dB. Corrections: my first "14x grain" metric was an artifact of a textured moving scene vs a flat grey background; stage 2 does not cause the grain, the lost crowd detail does. A job started from the session dies with it (a laptop shutdown on battery killed one 16-min run): keep AC power on.
