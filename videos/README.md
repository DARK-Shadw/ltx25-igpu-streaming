# Videos — LTX-2.5 (22B, unquantized bf16) on an Intel Arc iGPU

All clips were generated locally with the distilled LTX-2.5 DiT streamed from NVMe (Core Ultra 5 125H, 15.6 GB RAM, no discrete GPU).
Times are wall-clock for the whole render (text → video → decode/export). Render with `python render.py --mode flash ...` (see top-level `render.py`).

## Render modes

| Mode | What it does | Output | Time (5.4 s clip) | Use for |
|---|---|---|---|---|
| **FLASH** (recommended) | 12 fps generation (each frame shown twice = "anime on twos"), stage 1 at 768×448, 1 refinement step | 1536×896 | **~15 min** | finals |
| FLASH-2 | same, 2 refinement steps | 1536×896 | ~18 min | slightly more refinement |
| TURBO | stage 1 at 640×352, 1 refinement step | 1280×704 | ~9.5 min | quick looks only (wide-shot faces get mushy) |
| PREVIEW | stage 1 only, 12 fps doubled | 768×448 | ~7 min | picking a seed / prompt before a full render |
| REFERENCE | 24 fps, stage 1 at 768×448, 3 refinement steps | 1536×896 | ~47 min | best quality, slow |

## Anime

### swordsman-bridge — same prompt + seed in five modes
| File | Mode | Res | Length | Time |
|---|---|---|---|---|
| `1_reference-stage1-only_768x448_24fps_5s.mp4` | stage 1 only | 768×448 | 5.0 s | 13 min |
| `2_reference-hd_1536x896_24fps_5s_47min.mp4` | REFERENCE | 1536×896 | 5.0 s | 47 min |
| `3_flash-2step_1536x896_12fpsx2_5.4s_18min.mp4` | FLASH-2 | 1536×896 | 5.4 s | 17 min 44 s |
| `4_FLASH_1536x896_12fpsx2_5.4s_15min_RECOMMENDED.mp4` | **FLASH** | 1536×896 | 5.4 s | ~15 min |
| `5_turbo-preview_1280x704_12fpsx2_5.4s_9.5min.mp4` | TURBO | 1280×704 | 5.4 s | 9 min 32 s |

Prompt: `prompts/anime_swordsman-bridge.txt`. Known weakness: ~0.5 s of the fastest camera moves collapses into paint smear (at ~1.4–1.7 s and ~3.3–3.5 s).

### ronin-entrance — new (FLASH mode, rendered with `render.py`)
- `ronin-entrance/flash_1536x896_24fps_5.4s_seed77.mp4` (+ `.json` with settings, timings and the full prompt) — silver-haired swordsman with crimson eyes strides out of a burning, rain-soaked street toward the camera, then a slow push-in to a close-up of his glowing eyes. **13.6 min** total. No smear collapses (frame-to-frame change ~2.4 vs ~27 in swordsman-bridge). Prompt: `prompts/anime_ronin-entrance.txt`.
- `ronin-entrance/turbo_1280x704_24fps_5.4s_seed77.mp4` — same prompt and seed in TURBO mode: **9.7 min** (vs 13.6). Different composition (stage-1 resolution changes the draw): smoky fire at his feet, glowing red eyes in the close-up, no smear frames; but smaller/less detailed faces in wide shots and the face is cropped off the top of the frame around 3.9 s. Side-by-side: `frames/ronin_turbo_vs_flash.png`. On slow prompts TURBO holds up; on fast-action prompts it was much worse.
- Known flaws (FLASH): the "scar across the eye" reads as a black face-paint stripe, the walls have odd carved-relief patterns, and the dragging-sparks detail is barely visible.

### early-tests
- `bamboo-forest-swordsman_768x448_24fps_5s.mp4` — Demon Slayer-style water slash (14 min)
- `ghibli-girl-yellow-raincoat_768x448_24fps_3s.mp4` — simple Ghibli-style prompt, cleanest 2D look (8 min)
- `cyberpunk-swordswoman-mecha-dragon_768x448_24fps_3s.mp4` — first hard anime prompt; looks like a 3D render (7 min)

## Realistic
- `hd-showcase/fox-snowy-forest_1280x704_24fps_4s_two-stage_20min.mp4` — the HD showcase: 4 s, 1280×704, 20.6 min total.
- `tests/dog-beach_1024x640_2s_two-stage.mp4` — 2 s two-stage (~7 min).
- `tests/dog-beach_640x384_3s.mp4` — 3 s single stage (~6 min denoise).
- `tests/dog-beach_384x256_1s_early-recipe.mp4`, `tests/fruit-bowl_384x256_1s_early-recipe.mp4` — 1 s tests made before the distilled recipe was used (guidance on, linear sigmas).

## _failed-or-duplicate
Kept for the record, not deleted: the black NaN-bug video, two clips where the prompt was ignored (early text-encoder bugs), and a byte-identical duplicate.

## Other folders
`prompts/` prompt texts · `frames/` extracted stills and contact sheets · `latents/` saved latents (`*.pt.stage1` = reusable stage-1 results) · `logs/` run logs.
Audio quality has not been verified on any clip.
