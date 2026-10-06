from huggingface_hub import snapshot_download
p = snapshot_download(
    "Lightricks/LTX-2.5-Diffusers",
    local_dir="models/ltx25",
    allow_patterns=[
        "*.json", "*.txt", "*.model",
        "transformer/*-of-00008.safetensors", "transformer/*.index.json",
        "text_encoder/*", "tokenizer/*", "processor/*", "scheduler/*",
        "vae/*", "audio_vae/*", "vocoder/*",
        "connectors/*-of-00002.safetensors", "connectors/*.index.json",
    ],
    ignore_patterns=["transformer_full/*", "prompt_enhancer/*", "*lora*"],
    max_workers=4,
)
print("DONE", p)
