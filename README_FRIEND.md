# LTX-2.5 (22B, unquantized) on your PC - setup

This runs the full-precision 22B LTX-2.5 video+audio model by streaming its weights from your SSD, layer by layer,
so it works even with only 8 GB of VRAM. It needs **Windows, an NVIDIA GPU, ~75 GB free disk and a Hugging Face account**.

## Steps
1. Unzip this folder anywhere (e.g. `D:\ltx_igpu`). Install Python 3.10+ (python.org) if you do not have it.
2. Open a terminal in the folder and run:

       python setup_ltx.py

   It asks where to store the 66 GB model (choose your FASTEST SSD), then does everything by itself:
   installs PyTorch + libraries, helps you log in to Hugging Face, downloads the model with a progress bar,
   self-tests the GPU and the disk streaming, renders a tiny test video, then a full-size benchmark video.
3. **It is safe to stop and re-run at any time** (Ctrl+C, reboot, internet loss). Every finished step is re-checked and skipped;
   downloads resume.
4. When it stops with a message, do what the message says, then run the same command again.
5. When it says ALL DONE, send back `setup_report.json` and `setup_log.txt`. If it fails, send just `setup_log.txt`.

## Hugging Face (the one manual step)
The script pauses and explains it, but in short: make a free account, click "Agree and access repository" on
https://huggingface.co/Lightricks/LTX-2.5-Diffusers , create a *Read* token at https://huggingface.co/settings/tokens and paste it
when asked. The token is stored only on your computer. Never send it to anyone.

## Notes
- Close games/browsers with many tabs during the benchmark: it uses a lot of RAM.
- A full-size 5-second video takes a few minutes on a modern NVIDIA GPU.
- `BIBLE.md` documents how the engine works and every problem found so far (for whoever debugs it, human or AI).
