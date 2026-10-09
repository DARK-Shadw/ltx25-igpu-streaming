# Launch post drafts (edit freely; all numbers below are measured, see README)

## LinkedIn (engineering story)
I ran a 22-billion-parameter video model, fully unquantized, on a laptop with no GPU.

Not a gaming rig. An Intel Core Ultra 5 with integrated graphics and 15.6 GB of RAM. The model (Lightricks' LTX-2.5) is about 66 GB on disk and normally wants 40+ GB of VRAM.

How: I never load the model. Weights stay on an NVMe drive and are streamed layer by layer - unbuffered aligned reads, a prefetch thread that reads layer n+1 while layer n computes, per-tensor repacking so the GPU kernels accept the memory. Text to video, image to video, and native audio and speech all work.

What it costs: it is not fast. A 5-second anime clip with sound takes about 14 minutes (about 47 for the best-quality setting); a 4-second realistic clip about 20 minutes. The point was to find out whether it could run at all - and to write down every wall I hit: a vocoder that silently produced NaN in bf16, memory that had to be 256-byte aligned, a prompt that "worked" only after three changes at once.

Then I used it to make an anime scene from my own artwork: start frames in, shots out, automatic QC for the failures I kept seeing (faces zooming toward the lens, crowds walking out of frame).

Repo (code, benchmarks, the full bug log): https://github.com/DARK-Shadw/ltx25-igpu-streaming
Clips attached. Questions and criticism welcome - especially from anyone who has tried streaming inference on other hardware.

#AI #VideoGeneration #OpenSource #EdgeAI #Intel

## Reddit (r/StableDiffusion, r/comfyui, r/IntelArc - check each sub's self-promotion rules first; attach the video first)
Title: I'm learning inference engineering, so I tried to run a 22B video model (LTX-2.5, unquantized, with audio) on my laptop's integrated GPU

Body:
I'm trying to get into inference engineering, and I wanted a project that would force me to learn the hard parts instead of just calling a library. So I picked something that shouldn't fit: **LTX-2.5 (22B), in bf16 with no quantization**, on an **Intel Core Ultra 5 iGPU with 15.6 GB of shared RAM**. No discrete GPU.

The model is about 66 GB on disk, so I never load it. The weights stay on NVMe and get streamed **layer by layer**: aligned reads into a small pinned buffer, a prefetch thread loading layer n+1 while layer n computes, and the weights swapped in and out with hooks. It uses the distilled model, and it does **text-to-video, image-to-video and native audio**.

**Measured on this machine:**

| Result | Time |
|---|---|
| Anime clip, 5.4 s, 1536x896, with sound | ~14 min |
| Anime clip, best quality (true 24 fps) | ~47 min |
| Realistic clip, 4 s, 1280x704 | ~20 min |

It's slow, and that's not the point. The point was to find out whether it runs at all on hardware like this, and to learn what breaks. A lot did (memory alignment, a vocoder that gave silence in bf16), and I wrote every bug down.

**Known weaknesses:** fast motion smears for a few frames, anime clips are 12 fps with each frame shown twice, and the CUDA path exists but I haven't tested it on an NVIDIA GPU.

**Repo, with the full bug log:** https://github.com/DARK-Shadw/ltx25-igpu-streaming

I'd really like feedback from people who do this for a living. What would you have done differently in the streaming design? What should I learn next to get better at this?

## Instagram / Reels caption (use ltx25_igpu_reel_1080x1920.mp4)
22B-parameter AI video model. Fully unquantized. Running on a laptop's integrated GPU - no graphics card, 15.6 GB RAM, offline.
Each 5-second clip takes 15-30 minutes. Start frames are my own artwork; picture and sound are generated together.
Link in bio for the code and the full bug log. #aivideo #opensource #anime #localai #intel
