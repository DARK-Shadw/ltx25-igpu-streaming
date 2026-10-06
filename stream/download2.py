from huggingface_hub import snapshot_download
p = snapshot_download("Lightricks/LTX-2.5-Diffusers", local_dir="models/ltx25", allow_patterns=["latent_upsampler/*"], max_workers=2)
print("DONE", p)
