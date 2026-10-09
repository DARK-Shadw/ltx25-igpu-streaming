# Launch post drafts (edit freely; all numbers below are measured, see README)

## LinkedIn (engineering story)
I ran a 22-billion-parameter video model, fully unquantized, on a laptop with no GPU.

Not a gaming rig. An Intel Core Ultra 5 with integrated graphics and 15.6 GB of RAM. The model (Lightricks' LTX-2.5) is about 66 GB on disk and normally wants 40+ GB of VRAM.

How: I never load the model. Weights stay on an NVMe drive and are streamed layer by layer - unbuffered aligned reads, a prefetch thread that reads layer n+1 while layer n computes, per-tensor repacking so the GPU kernels accept the memory. Text to video, image to video, and native audio and speech all work.

What it costs: it is not fast. A 5-second anime clip with sound takes about 14 minutes (about 47 for the best-quality setting); a 4-second realistic clip about 20 minutes. The point was to find out whether it could run at all - and to write down every wall I hit: a vocoder that silently produced NaN in bf16, memory that had to be 256-byte aligned, a prompt that "worked" only after three changes at once.

Then I used it to make an anime scene from my own artwork: start frames in, shots out, automatic QC for the failures I kept seeing (faces zooming toward the lens, crowds walking out of frame).

Repo (code, benchmarks, the full bug log): <link>
Clips attached. Questions and criticism welcome - especially from anyone who has tried streaming inference on other hardware.

#AI #VideoGeneration #OpenSource #EdgeAI #Intel

## Reddit (r/StableDiffusion, r/comfyui, r/IntelArc - check each sub's self-promotion rules first)
Title: I ran the full unquantized LTX-2.5 (22B) with audio on a laptop iGPU (15.6 GB RAM, no dGPU) - 5 s anime clip in ~14 min
Body: video first. Then: weights streamed from NVMe layer by layer, bf16, no quantization, distilled model, T2V + I2V + native audio. Table of measured times (anime 14 / 47 min, realistic 20 min). It is slow - that is not the claim; the claim is that it runs at all on this hardware. Known weaknesses: fast motion smears for a few frames, anime is 12 fps doubled. CUDA path exists but is untested. Repo + full bug log: <link>. Happy to answer questions about the streaming engine.

## Instagram / Reels caption (use ltx25_igpu_reel_1080x1920.mp4)
22B-parameter AI video model. Fully unquantized. Running on a laptop's integrated GPU - no graphics card, 15.6 GB RAM, offline.
Each 5-second clip takes 15-30 minutes. Start frames are my own artwork; picture and sound are generated together.
Link in bio for the code and the full bug log. #aivideo #opensource #anime #localai #intel
