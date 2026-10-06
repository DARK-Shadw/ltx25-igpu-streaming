"""One-command setup + self-test for running unquantized LTX-2.5 (22B) on this PC by streaming weights from disk.

  python setup_ltx.py                      # first run: asks where to put the 66 GB model, then does everything
  python setup_ltx.py --model-dir D:\\ltx25  # choose the model folder up front
  python setup_ltx.py --skip-bench         # skip the final full-size benchmark render
  python setup_ltx.py --status             # show which stages are done, change nothing

SAFE TO RE-RUN. Every stage checks its own result ("verify") before it is trusted, so after any failure,
Ctrl+C or reboot, just run the same command again: finished stages are re-verified and skipped, the failed one is retried
(downloads resume where they stopped).

Everything printed is also saved to setup_log.txt. If something fails, send that file back - nothing else is needed.
Nothing is uploaded anywhere by this script. Your Hugging Face token is stored only in Hugging Face's own local cache.
"""
import argparse, ctypes, getpass, json, os, platform, re, shutil, subprocess, sys, time, traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "setup_log.txt")
STATE = os.path.join(ROOT, "setup_state.json")
CONFIG = os.path.join(ROOT, "ltx_config.json")
VENV = os.path.join(ROOT, ".venv")
VPY = os.path.join(VENV, "Scripts", "python.exe")
REPO = "Lightricks/LTX-2.5-Diffusers"
MODEL_URL = "https://huggingface.co/" + REPO
TOKEN_URL = "https://huggingface.co/settings/tokens"
ALLOW = ["*.json", "*.txt", "*.model",
         "transformer/*-of-00008.safetensors", "transformer/*.index.json",
         "text_encoder/*", "tokenizer/*", "processor/*", "scheduler/*", "vae/*", "audio_vae/*", "vocoder/*",
         "connectors/*-of-00002.safetensors", "connectors/*.index.json", "latent_upsampler/*"]
IGNORE = ["transformer_full/*", "prompt_enhancer/*", "*lora*"]
PINS = ["diffusers==0.40.0", "transformers==5.14.1", "accelerate==1.15.0", "safetensors==0.8.0"]
FREE = ["huggingface_hub", "numpy", "pillow", "av", "imageio", "imageio-ffmpeg", "sentencepiece", "protobuf", "tqdm", "regex"]
NEED_GB = 75          # 66 GB model + room for outputs

# ------------------------------------------------------------------ logging (console + file, progress bars kept out of the file)


class Tee:
    def __init__(self, console):
        self.console, self.buf = console, ""
        self.f = open(LOG, "a", encoding="utf-8", errors="replace")

    def write(self, s):
        try:
            self.console.write(s); self.console.flush()
        except UnicodeEncodeError:          # legacy Windows code pages: never crash on a character the console can't show
            enc = getattr(self.console, "encoding", None) or "ascii"
            self.console.write(s.encode(enc, errors="replace").decode(enc, errors="replace")); self.console.flush()
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.split("\r")[-1]          # keep only the final state of a progress-bar line
            if line.strip():
                self.f.write(line + "\n")
        self.f.flush()

    def flush(self):
        self.console.flush()

    def isatty(self):
        return self.console.isatty()


def say(msg=""):
    print(msg, flush=True)


def banner(msg):
    say("\n" + "=" * 78 + f"\n  {msg}\n" + "=" * 78)


def fmt_t(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


class Fail(Exception):
    """A stage could not complete; the message tells the user what happened and what to do."""


def ask(prompt, default=None):
    if not sys.stdin or not sys.stdin.isatty():
        raise Fail(f"input needed but this run is not interactive: {prompt}")
    s = input(prompt).strip()
    return s or default


def pause(msg):
    say("\n" + msg)
    ask("\n  >>> Press ENTER when you have done this (or type q + ENTER to stop here): ")


def run_cmd(cmd, env=None, cwd=ROOT, what=""):
    """Run a command, stream its output live (progress bars included), return the exit code."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
        if not chunk:
            break
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
    return p.wait()


def retry(fn, tries, what):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Fail:
            raise
        except Exception as e:
            say(f"  ! {what} failed (attempt {i}/{tries}): {type(e).__name__}: {str(e)[:200]}")
            if i == tries:
                raise
            time.sleep(min(60, 5 * i * i)); say("  retrying...")


def cfg():
    try:
        return json.load(open(CONFIG))
    except Exception:
        return {}


def models_dir():
    return cfg().get("models")


# ------------------------------------------------------------------ stages. Each: run(ctx) does the work, verify(ctx) returns (ok, info)
# --------- 1. preflight


def v_preflight(ctx):
    if os.name != "nt":
        return False, "this version only supports Windows (it uses Windows unbuffered file reads)"
    if sys.version_info < (3, 10):
        return False, f"Python {sys.version.split()[0]} is too old (need 3.10+)"
    if ctx.device == "cuda":
        smi = shutil.which("nvidia-smi")
        if not smi:
            return False, "nvidia-smi not found: install the NVIDIA driver (https://www.nvidia.com/Download/index.aspx)"
    return True, f"Windows, Python {sys.version.split()[0]}"


def r_preflight(ctx):
    raise Fail(v_preflight(ctx)[1])


# --------- 2. model folder choice


def drives():
    out = []
    mask = ctypes.windll.kernel32.GetLogicalDrives()
    for i in range(26):
        if mask >> i & 1:
            d = f"{chr(65 + i)}:\\"
            if ctypes.windll.kernel32.GetDriveTypeW(d) == 3:       # fixed disk
                try:
                    out.append((d, shutil.disk_usage(d).free / 2**30))
                except OSError:
                    pass
    return out


def v_folder(ctx):
    m = models_dir()
    if not m:
        return False, "not chosen yet"
    try:
        os.makedirs(m, exist_ok=True)
        t = os.path.join(m, ".write_test"); open(t, "w").write("x"); os.remove(t)
    except OSError as e:
        return False, f"cannot write to {m}: {e}"
    return True, m


def r_folder(ctx):
    path = ctx.model_dir
    if not path:
        say("Where should the model be stored? It needs about 66 GB (+ room for videos) and READ SPEED is what matters:")
        say("a fast NVMe SSD is ideal; avoid hard drives and slow SATA SSDs.\n")
        ds = drives()
        for d, f in ds:
            say(f"    {d}   {f:7.0f} GB free")
        best = max((x for x in ds if x[1] >= NEED_GB), key=lambda x: x[1], default=None)
        default = os.path.join(best[0], "ltx25") if best else os.path.join(ROOT, "models", "ltx25")
        path = ask(f"\n  Folder for the model [{default}]: ", default)
    path = os.path.abspath(path)
    free = shutil.disk_usage(os.path.splitdrive(path)[0] + "\\" if os.path.splitdrive(path)[0] else ROOT).free / 2**30
    have = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 2**30 if os.path.isdir(path) else 0
    if free + have < NEED_GB:
        raise Fail(f"only {free:.0f} GB free on that drive (+{have:.0f} GB already there); need about {NEED_GB} GB. Free some space or choose another drive: python setup_ltx.py --model-dir <folder>")
    os.makedirs(path, exist_ok=True)
    json.dump({"models": path}, open(CONFIG, "w"))
    say(f"  model folder: {path}  ({free:.0f} GB free)")


# --------- 3. virtual environment


def v_venv(ctx):
    if not os.path.exists(VPY):
        return False, "no virtual environment yet"
    r = subprocess.run([VPY, "-c", "import sys;print(sys.version.split()[0])"], capture_output=True, text=True)
    return r.returncode == 0, f"{VENV} (Python {r.stdout.strip()})"


def r_venv(ctx):
    if os.path.isdir(VENV):
        shutil.rmtree(VENV, ignore_errors=True)          # half-created -> start clean
    say("  creating virtual environment...")
    if subprocess.call([sys.executable, "-m", "venv", VENV]) != 0:
        raise Fail("could not create a virtual environment (is the 'venv' module installed with this Python?)")


# --------- 4. PyTorch


def cuda_tag():
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        ver = (int(m.group(1)), int(m.group(2)))
    except Exception:
        raise Fail("could not read the CUDA version from nvidia-smi; update the NVIDIA driver")
    if ver >= (13, 0): return "cu130"
    if ver >= (12, 8): return "cu128"
    if ver >= (12, 6): return "cu126"
    raise Fail(f"NVIDIA driver is too old (supports CUDA {ver[0]}.{ver[1]}); update it from nvidia.com, then re-run")


def v_torch(ctx):
    code = ("import torch,sys;d=sys.argv[1];ok=torch.cuda.is_available() if d=='cuda' else torch.xpu.is_available();"
            "x=torch.randn(64,64,device=d)@torch.randn(64,64,device=d) if ok else None;"
            "print(torch.__version__, 'OK' if ok else 'NO-ACCELERATOR')")
    r = subprocess.run([VPY, "-c", code, ctx.device], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
    return r.returncode == 0 and last.endswith("OK"), f"torch {last}"


def r_torch(ctx):
    if ctx.device != "cuda":
        raise Fail("PyTorch with an Intel XPU must be installed manually in .venv")
    tag = cuda_tag()
    say(f"  installing PyTorch for CUDA ({tag}); this is a ~3 GB download...")
    url = f"https://download.pytorch.org/whl/{tag}"
    retry(lambda: _pip(["torch", "--index-url", url]), 3, "PyTorch install")


def _pip(args):
    rc = run_cmd([VPY, "-m", "pip", "install", "--disable-pip-version-check"] + args)
    if rc != 0:
        raise RuntimeError(f"pip exited with code {rc}")


# --------- 5. other Python packages


def v_deps(ctx):
    code = ("import importlib.metadata as m;import sys\n"
            "pins={'diffusers':'0.40.0','transformers':'5.14.1','accelerate':'1.15.0','safetensors':'0.8.0'}\n"
            "bad=[f'{k} {m.version(k)} != {v}' for k,v in pins.items() if m.version(k)!=v]\n"
            "import diffusers,transformers,accelerate,safetensors,huggingface_hub,numpy,PIL,av,sentencepiece,google.protobuf\n"
            "from diffusers import LTX2Pipeline\nprint('BAD:'+';'.join(bad) if bad else 'OK')")
    r = subprocess.run([VPY, "-c", code], capture_output=True, text=True)
    out = (r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode == 0 and out == "OK", ("all imports + pinned versions OK" if out == "OK" else (out or r.stderr.strip()[-300:]))


def r_deps(ctx):
    say("  installing diffusers / transformers / etc. (pinned to the versions this engine was tested with)...")
    retry(lambda: _pip(PINS + FREE), 3, "package install")


# --------- 6. Hugging Face login + license


def hf_check():
    """-> (state, detail) with state in ok / no_token / not_accepted / error"""
    code = r'''
import sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
try:
    who = api.whoami()["name"]
except Exception as e:
    print("NO_TOKEN"); sys.exit(0)
try:
    hf_hub_download("%s", "model_index.json", local_dir=sys.argv[1] + "/_probe")
    print("OK " + who)
except Exception as e:
    code = getattr(getattr(e, "response", None), "status_code", None)
    print("NOT_ACCEPTED " + who if code in (401, 403) or "Gated" in type(e).__name__ else "ERROR " + type(e).__name__ + ": " + str(e)[:150])
''' % REPO
    r = subprocess.run([VPY, "-c", code, os.path.join(ROOT, "_hf_probe")], capture_output=True, text=True, timeout=120)
    shutil.rmtree(os.path.join(ROOT, "_hf_probe"), ignore_errors=True)
    out = (r.stdout.strip().splitlines() or [""])[-1]
    if out.startswith("OK"): return "ok", out[3:]
    if out.startswith("NO_TOKEN"): return "no_token", ""
    if out.startswith("NOT_ACCEPTED"): return "not_accepted", out[13:]
    return "error", out or r.stderr.strip()[-200:]


def v_hf(ctx):
    s, d = hf_check()
    return s == "ok", f"logged in as '{d}', license accepted" if s == "ok" else s


def r_hf(ctx):
    for _ in range(50):
        s, d = hf_check()
        if s == "ok":
            return
        if s == "no_token":
            banner("ACTION NEEDED: Hugging Face account")
            say(f"""  The model is "gated": you must have a free Hugging Face account, accept the model's license, and give this
  script a read-only access token. Do these steps in your web browser:

   1. Create a free account (or log in):   https://huggingface.co/join
   2. Open the model page and click the "Agree and access repository" button:
         {MODEL_URL}
   3. Create a token:  {TOKEN_URL}  ->  "Create new token"  ->  type "Read"  ->  copy it (starts with hf_)
   4. Come back here and paste the token below (you won't see it as you type - that's normal, just paste + ENTER).

  The token stays on THIS computer (Hugging Face's own cache). Never send it to anyone, including me.""")
            tok = getpass.getpass("\n  Paste token (or just press ENTER to stop here): ").strip()
            if not tok:
                raise Fail("stopped at the Hugging Face login step; re-run when you have a token")
            r = subprocess.run([VPY, "-c", "import sys;from huggingface_hub import login;login(token=sys.argv[1], add_to_git_credential=False)", tok],
                               capture_output=True, text=True)
            if r.returncode != 0:
                say("  ! that token was rejected: " + r.stderr.strip()[-200:])
        elif s == "not_accepted":
            banner("ACTION NEEDED: accept the model license")
            say(f"""  You are logged in to Hugging Face as '{d}', but that account has not accepted the model license yet.

   1. Open this page in your browser (logged in as '{d}'):  {MODEL_URL}
   2. Click the button "Agree and access repository" and wait until the page shows the files.
   (If you already clicked it, it can take a minute to take effect.)""")
            pause("")
        else:
            raise Fail(f"could not reach Hugging Face ({d}). Check your internet connection and re-run.")
    raise Fail("still no access after many tries; see the messages above")


# --------- 7. download


def remote_files():
    code = r'''
import json, sys
from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects
info = HfApi().model_info("%s", files_metadata=True)
items = [(s.rfilename, s.size) for s in info.siblings]
keep = set(filter_repo_objects([n for n, _ in items], allow_patterns=%r, ignore_patterns=%r))
print(json.dumps({n: sz for n, sz in items if n in keep}))
''' % (REPO, ALLOW, IGNORE)
    r = subprocess.run([VPY, "-c", code], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError("could not list the model files: " + r.stderr.strip()[-300:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def download_state():
    m = models_dir()
    files = retry(remote_files, 3, "listing model files")
    missing = {}
    for n, sz in files.items():
        p = os.path.join(m, *n.split("/"))
        if not os.path.isfile(p) or (sz is not None and os.path.getsize(p) != sz):
            missing[n] = sz or 0
    return files, missing


def v_download(ctx):
    try:
        files, missing = download_state()
    except Exception as e:
        return False, f"cannot check ({e})"
    tot = sum(v or 0 for v in files.values()) / 2**30
    return not missing, (f"all {len(files)} files present, {tot:.1f} GB" if not missing else f"{len(missing)} files missing/incomplete ({sum(missing.values()) / 2**30:.1f} GB to go)")


def r_download(ctx):
    files, missing = download_state()
    need = sum(missing.values()) / 2**30
    free = shutil.disk_usage(models_dir()).free / 2**30
    if free < need + 2:
        raise Fail(f"need {need:.0f} GB more but only {free:.0f} GB free on that drive")
    say(f"  downloading {need:.1f} GB ({len(missing)} files). Interrupting is fine: re-running resumes from where it stopped.")
    code = ("import sys;from huggingface_hub import snapshot_download;"
            "snapshot_download(%r, local_dir=sys.argv[1], allow_patterns=%r, ignore_patterns=%r, max_workers=4)" % (REPO, ALLOW, IGNORE))
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1")

    def go():
        rc = run_cmd([VPY, "-c", code, models_dir()], env=env)
        if rc != 0:
            raise RuntimeError(f"download exited with code {rc}")
    retry(go, 8, "download")
    mb = need * 1024 / max(1, time.time() - t0)
    ctx.state["download_MBps"] = round(mb, 1)
    say(f"  downloaded in {fmt_t(time.time() - t0)} (~{mb:.0f} MB/s average)")


# --------- 8-9. self-tests


def selftest(ctx, what):
    env = dict(os.environ, LTX_MODELS=models_dir(), LTX_DEVICE=ctx.device, PYTHONUNBUFFERED="1")
    p = subprocess.run([VPY, os.path.join(ROOT, "stream", "selftest.py"), what], capture_output=True, text=True, env=env, cwd=ROOT)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def v_dev(ctx):
    ok, out = selftest(ctx, "device")
    ctx.state["selftest_device"] = out
    speed = next((l.strip() for l in out.splitlines() if "TFLOPS" in l), out.splitlines()[-1] if out else "")
    return ok, speed if ok else out[-600:]


def r_dev(ctx):
    ok, out = selftest(ctx, "device")
    say(out)
    raise Fail("the accelerator self-test failed (details above). Send setup_log.txt")


def v_loader(ctx):
    ok, out = selftest(ctx, "loader")
    ctx.state["selftest_loader"] = out
    return ok, out.splitlines()[-1] if ok else out[-600:]


def r_loader(ctx):
    ok, out = selftest(ctx, "loader")
    say(out)
    raise Fail("the weight-streaming self-test failed (details above). Send setup_log.txt")


# --------- 10-11. real renders


def check_video(path, min_frames):
    code = r'''
import av, numpy as np, sys, json
c = av.open(sys.argv[1]); v = c.streams.video[0]
fr = [f.to_ndarray(format="gray") for f in c.decode(video=0)]
c.close(); c = av.open(sys.argv[1])
au = np.concatenate([f.to_ndarray().astype("float32").ravel() for f in c.decode(audio=0)]) if c.streams.audio else np.zeros(1)
print(json.dumps(dict(frames=len(fr), w=v.width, h=v.height, std=float(np.std([f.std() for f in fr])), mean_frame_std=float(np.mean([f.std() for f in fr])),
      audio_finite=bool(np.isfinite(au).all()), audio_rms=float(np.sqrt(np.mean(au ** 2))))))
'''
    r = subprocess.run([VPY, "-c", code, path], capture_output=True, text=True)
    if r.returncode != 0:
        return False, "cannot decode the video: " + r.stderr.strip()[-200:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    ok = d["frames"] >= min_frames and d["mean_frame_std"] > 8 and d["audio_finite"] and d["audio_rms"] > 1e-5
    return ok, f"{d['w']}x{d['h']}, {d['frames']} frames, picture contrast {d['mean_frame_std']:.0f}, audio rms {d['audio_rms']:.4f}" + ("" if ok else "  <-- looks empty/broken")


def render_out(name, sub):
    d = os.path.join(ROOT, "videos", "tests", name, sub)
    mp4 = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mp4")] if os.path.isdir(d) else []
    return mp4[0] if mp4 else None


def do_render(ctx, mode, name, seconds, label):
    env = dict(os.environ, LTX_MODELS=models_dir(), LTX_DEVICE=ctx.device, PYTHONUNBUFFERED="1")
    sub = mode
    prompt = "prompts/anime_swordsman-bridge.txt"
    say(f"  rendering {label}; progress is written to logs/ (this window shows start/finish)...")
    t0 = time.time()
    rc = run_cmd([VPY, "render.py", "--mode", mode, "--name", name, "--prompt-file", prompt, "--seed", "3", "--seconds", str(seconds),
                  "--category", "tests", "--subdir", sub], env=env)
    dt = time.time() - t0
    if rc != 0:
        tail = ""
        try:
            lg = os.path.join(ROOT, "logs", f"{name}_{mode}_seed3.log")
            tail = "".join(open(lg, errors="replace").readlines()[-25:])
        except Exception:
            pass
        say("  ---- last lines of the render log ----\n" + tail)
        raise Fail(f"the {label} failed (exit code {rc}). The log tail above is in setup_log.txt - send it to me.")
    ctx.state[f"{mode}_seconds"] = round(dt)


def v_smoke(ctx):
    p = render_out("setup-smoke", "smoke")
    return (check_video(p, 17) if p else (False, "not rendered yet"))


def r_smoke(ctx):
    do_render(ctx, "smoke", "setup-smoke", 1.4, "tiny test video (1.4 s, 384x256)")
    ok, info = v_smoke(ctx)
    if not ok:
        raise Fail("the tiny test video was produced but looks broken: " + info)


def v_bench(ctx):
    p = render_out("setup-bench", "flash")
    return (check_video(p, 60) if p else (False, "not rendered yet"))


def r_bench(ctx):
    if ctx.skip_bench:
        raise Fail("benchmark skipped (--skip-bench)")
    do_render(ctx, "flash", "setup-bench", 5.4, "full-size FLASH benchmark (5.4 s video, 1536x896)")
    ok, info = v_bench(ctx)
    if not ok:
        raise Fail("benchmark video looks broken: " + info)


STAGES = [
    ("preflight", "Check this PC", v_preflight, r_preflight),
    ("folder", "Choose the model folder", v_folder, r_folder),
    ("venv", "Create Python environment", v_venv, r_venv),
    ("torch", "Install PyTorch (GPU)", v_torch, r_torch),
    ("deps", "Install other packages", v_deps, r_deps),
    ("hf", "Hugging Face login + license", v_hf, r_hf),
    ("download", "Download the model (66 GB)", v_download, r_download),
    ("device", "Self-test: GPU", v_dev, r_dev),
    ("loader", "Self-test: streaming weights from disk", v_loader, r_loader),
    ("smoke", "Test render (tiny video + audio)", v_smoke, r_smoke),
    ("bench", "Benchmark render (full size)", v_bench, r_bench),
]
BOOT = {"preflight", "folder", "venv"}      # run with the system Python; the rest run inside the venv


def write_report(ctx):
    rep = dict(state=ctx.state, finished=time.strftime("%Y-%m-%d %H:%M"), os=platform.platform(), python=sys.version.split()[0])
    for mode, name in (("flash", "setup-bench"), ("smoke", "setup-smoke")):
        js = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ROOT, "videos", "tests", name)) for f in fs if f.endswith(".json")]
        if js:
            rep[mode] = json.load(open(js[0]))
            rep[mode].pop("prompt", None)
        lg = os.path.join(ROOT, "logs", f"{name}_{mode}_seed3.log")
        if os.path.exists(lg):
            steps = [float(x) for x in re.findall(r"step \d+/\d+ done \(([\d.]+)s\)", open(lg, errors="replace").read())]
            if steps:
                rep.setdefault(mode + "_step_seconds", steps)
    out = os.path.join(ROOT, "setup_report.json")
    json.dump(rep, open(out, "w"), indent=2)
    return out


class Ctx:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir")
    ap.add_argument("--device", default="cuda", choices=["cuda", "xpu"], help="(testing only) accelerator type")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--in-venv", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    sys.stdout = sys.stderr = Tee(sys.__stdout__)
    ctx = Ctx(); ctx.device, ctx.skip_bench, ctx.model_dir, ctx.state = a.device, a.skip_bench, a.model_dir, load_state()
    if a.model_dir and not a.in_venv:
        json.dump({"models": os.path.abspath(a.model_dir)}, open(CONFIG, "w"))
    if not a.in_venv:
        say(f"\nLTX-2.5 setup  |  {time.strftime('%Y-%m-%d %H:%M')}  |  log: {LOG}")
        say("Safe to stop (Ctrl+C) and re-run at any time: it resumes and re-verifies finished stages.")
    todo = [s for s in STAGES if (s[0] in BOOT) != a.in_venv]
    n_all = len(STAGES)
    for sid, title, ver, run in todo:
        i = [s[0] for s in STAGES].index(sid) + 1
        say(f"\n[{i}/{n_all}] {title}")
        try:
            ok, info = ver(ctx)
            if ok:
                say(f"   [OK] already done and verified: {info}")
                continue
            if a.status:
                say(f"   - not done: {info}")
                continue
            say(f"   -> working ({info})")
            t0 = time.time()
            run(ctx)
            ok, info = ver(ctx)
            if not ok:
                raise Fail(f"'{title}' ran but its check still fails: {info}")
            ctx.state[sid] = dict(done=True, seconds=round(time.time() - t0), info=str(info)[:300])
            save_state(ctx.state)
            say(f"   [OK] done in {fmt_t(time.time() - t0)} and verified: {info}")
        except Fail as e:
            if sid == "bench" and ctx.skip_bench:
                say("   - skipped"); continue
            say(f"\n  [X] STOPPED at: {title}\n  {e}\n\n  Fix the above and run the same command again - it will continue from here."
                f"\n  If you are stuck, send me the file {LOG}")
            sys.exit(2)
        except KeyboardInterrupt:
            say("\n  Stopped by you. Re-run the same command to continue."); sys.exit(130)
        except Exception:
            say(f"\n  [X] UNEXPECTED ERROR at: {title}\n" + traceback.format_exc() + f"\n  Run again to retry; if it repeats send me {LOG}")
            sys.exit(3)
    if not a.in_venv and not a.status:        # bootstrap done -> continue inside the venv
        say("\n  switching to the virtual environment...")
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        cmd = [VPY, os.path.abspath(__file__), "--in-venv", "--device", a.device] + (["--skip-bench"] if a.skip_bench else [])
        sys.exit(subprocess.call(cmd))
    if a.in_venv and not a.status:
        rep = write_report(ctx)
        banner("ALL DONE")
        say(f"  Everything is installed and verified. Please send me:\n    {rep}\n    {LOG}\n"
            "  Make a video:   .venv\\Scripts\\python render.py --mode flash --name my-clip --prompt-file prompts\\anime_swordsman-bridge.txt --seed 1")


if __name__ == "__main__":
    main()
