"""LTX Studio - a very small Tk front-end for hand-guided generation (about 30-50 MB of RAM, no browser).

  .venv\\Scripts\\pythonw.exe studio.py        (or double-click start_studio.bat)

Pick a start image (optional), write the prompt, tune the settings, press Start. It runs render.py in the background and shows a
progress bar with time left, a live "what the model currently thinks the video will be" picture after every denoising step
(drag the slider to replay how it formed), and the log. Output goes to videos/studio/<name>/.
"""
import ctypes, json, math, os, random, re, subprocess, sys, time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.abspath(__file__))
_V = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PY = _V if os.path.exists(_V) else sys.executable
INPUTS = os.path.join(ROOT, "studio_inputs")
RECIPE6 = "1.0,0.9875,0.975,0.909375,0.725,0.421875"
PRESETS = {                       # size, fps, stage-1 steps, refinement steps
    "Flash (recommended)": ("768x448", 12, 8, 1),
    "Flash+ (2 refinement steps)": ("768x448", 12, 8, 2),
    "Reference (24 fps, 3 refinement)": ("768x448", 24, 8, 3),
    "Turbo (quick look)": ("640x352", 12, 8, 1),
    "Preview (stage 1 only)": ("768x448", 12, 8, 0),
    "Smoke test (tiny)": ("384x256", 12, 8, 0),
}
SIZES = ["384x256", "512x320", "640x352", "768x448", "896x512", "960x544", "1024x576", "1280x704"]


# ----------------------------------------------------------------------------- helpers
def free_ram_gb():
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]
    ms = MS(); ms.l = ctypes.sizeof(ms); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ap / 2**30


def frames_for(seconds, fps):
    return max(9, int(round(seconds * fps / 8)) * 8 + 1)


def step_seconds(n_tokens):                          # measured cost model on this iGPU (N in thousands of tokens)
    n = n_tokens / 1000.0
    return max(20.0, 14.5 * n + 0.428 * n * n)


def estimate(w, h, seconds, fps, s1_steps, s2_steps):
    F = (frames_for(seconds, fps) - 1) // 8 + 1
    n1 = (w // 32) * (h // 32) * F
    e = dict(text=35.0, s1_n=s1_steps, s1=step_seconds(n1), up=40.0 if s2_steps else 0.0, s2_n=s2_steps)
    e["s2"] = step_seconds(n1 * 4) * 0.95 if s2_steps else 0.0
    fw, fh = (w * 2, h * 2) if s2_steps else (w, h)
    e["dec"] = 150.0 * (fw * fh * frames_for(seconds, fps)) / (1536 * 896 * 65) + 20.0
    e["total"] = e["text"] + s1_steps * e["s1"] + e["up"] + s2_steps * e["s2"] + e["dec"]
    return e


def fmt(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def prep_image(src, w, h, mode):
    im = Image.open(src).convert("RGB")
    if mode == "Crop to fill":
        s = max(w / im.width, h / im.height)
        im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
        x, y = (im.width - w) // 2, (im.height - h) // 2
        return im.crop((x, y, x + w, y + h))
    s = min(w / im.width, h / im.height)                                    # "Pad (black bars)"
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), (0, 0, 0)); bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


class Tracker:
    """Turns the generator's log lines into 'phase', progress fraction and time left."""

    def __init__(self, est):
        self.e, self.t0 = est, time.time()
        self.phase, self.p_t0 = "starting", time.time()
        self.s1_done = self.s2_done = 0
        self.step_times = {"s1": [], "s2": []}
        self.dec_frac = 0.0

    def feed(self, line):
        now = time.time()
        if "A: text encoder" in line: self._set("text encoder", now)
        elif "B: connectors" in line: self._set("text encoder", now)
        elif "I2V: encoded" in line: self._set("encoding start image", now)
        elif "C: transformer resident" in line: self._set("stage 1 (low-res denoising)", now)
        elif "C: resumed stage-1" in line: self.s1_done = self.e["s1_n"]; self._set("upscaling", now)
        elif re.search(r"C: step (\d+)/(\d+) done \(([\d.]+)s\)", line):
            m = re.search(r"C: step (\d+)/(\d+) done \(([\d.]+)s\)", line)
            self.s1_done = int(m[1]); self.step_times["s1"].append(float(m[3])); self.p_t0 = now
            if self.s1_done >= self.e["s1_n"]: self._set("upscaling", now)
        elif "C: upsampled" in line: self._set("stage 2 (refinement at 2x)", now)
        elif re.search(r"C2: step (\d+)/(\d+) done \(([\d.]+)s\)", line):
            m = re.search(r"C2: step (\d+)/(\d+) done \(([\d.]+)s\)", line)
            self.s2_done = int(m[1]); self.step_times["s2"].append(float(m[3])); self.p_t0 = now
        elif "C: denoising done" in line: self._set("decoding video + audio", now)
        elif "vae decoded" in line: self.dec_frac = 0.8
        elif "audio decoded" in line: self.dec_frac = 0.95
        elif "wrote" in line and "D:" not in line and ".mp4" in line: self._set("done", now)

    def _set(self, name, now):
        if self.phase != name:
            self.phase, self.p_t0 = name, now

    def progress(self):
        now, e = time.time(), self.e
        elapsed = now - self.t0
        sc1 = (sum(self.step_times["s1"]) / len(self.step_times["s1"]) / e["s1"]) if self.step_times["s1"] else 1.0
        sc2 = (sum(self.step_times["s2"]) / len(self.step_times["s2"]) / e["s2"]) if self.step_times["s2"] and e["s2"] else sc1
        in_phase = now - self.p_t0
        left = 0.0
        if self.phase == "done": return 1.0, 0.0, "done"
        order = ["starting", "text encoder", "encoding start image", "stage 1 (low-res denoising)", "upscaling", "stage 2 (refinement at 2x)", "decoding video + audio"]
        pi = order.index(self.phase) if self.phase in order else 0
        if pi <= 1: left += max(0.0, e["text"] - in_phase if pi == 1 else e["text"])
        if pi <= 3:
            n_left = e["s1_n"] - self.s1_done
            cur = e["s1"] * sc1
            left += max(0.0, n_left * cur - (in_phase if pi == 3 else 0.0)) if pi == 3 else n_left * cur
        if pi <= 4: left += e["up"] if pi < 4 else max(0.0, e["up"] - in_phase)
        if pi <= 5:
            n_left = e["s2_n"] - (self.s2_done if pi == 5 else 0)
            left += n_left * e["s2"] * sc2 - (min(in_phase, e["s2"] * sc2) if pi == 5 and n_left else 0.0)
        left += e["dec"] * (1 - self.dec_frac) if pi == 6 else e["dec"]
        left = max(left, 0.0)
        return min(0.99, elapsed / (elapsed + left)) if elapsed + left > 0 else 0.0, left, self.phase


# ----------------------------------------------------------------------------- the app
class App:
    def __init__(self, root):
        self.root = root
        root.title("LTX Studio")
        root.geometry(f"1180x{min(940, root.winfo_screenheight() - 70)}")
        self.proc = None; self.tracker = None; self.log_pos = 0; self.tag = None; self.video = None
        self.src_image = None; self.thumb = None; self.prev_files = []; self.prev_img = None; self.follow = True
        v = self.v = {k: tk.StringVar() for k in ("preset", "seconds", "fps", "size", "s1", "s2", "seed", "name", "fit")}
        self.reuse = tk.BooleanVar(value=False)
        os.makedirs(INPUTS, exist_ok=True)

        left = ttk.Frame(root, padding=10); left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(root, padding=10); right.grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)

        b = ttk.Frame(left); b.grid(sticky="ew")
        self.start_btn = ttk.Button(b, text="Start generation", command=self.start); self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(b, text="Stop", command=self.stop, state="disabled"); self.stop_btn.pack(side="left", padx=6)
        self.ram = ttk.Label(b, text=""); self.ram.pack(side="right")
        self.info = ttk.Label(left, text="", wraplength=430, justify="left"); self.info.grid(sticky="w", pady=(2, 8))

        ttk.Label(left, text="Start image (optional - empty = text-to-video)").grid(sticky="w")
        r = ttk.Frame(left); r.grid(sticky="ew")
        ttk.Button(r, text="Browse...", command=self.pick_image).pack(side="left")
        ttk.Button(r, text="Clear", command=self.clear_image).pack(side="left", padx=4)
        ttk.Combobox(r, textvariable=v["fit"], values=["Crop to fill", "Pad (black bars)"], width=16, state="readonly").pack(side="left", padx=4)
        v["fit"].set("Crop to fill"); v["fit"].trace_add("write", lambda *a: self.refresh_thumb())
        self.thumb_lbl = ttk.Label(left, text="(no image)", anchor="center", relief="groove", width=40)
        self.thumb_lbl.grid(sticky="ew", pady=4, ipady=14)

        ttk.Label(left, text="Prompt").grid(sticky="w")
        self.prompt = tk.Text(left, width=52, height=7, wrap="word", font=("Segoe UI", 10)); self.prompt.grid(sticky="ew")
        self.prompt.bind("<<Modified>>", self.count_words)
        self.words = ttk.Label(left, text="0 words"); self.words.grid(sticky="e")

        g = ttk.LabelFrame(left, text="Settings", padding=8); g.grid(sticky="ew", pady=6)
        def row(i, label, widget):
            ttk.Label(g, text=label).grid(row=i, column=0, sticky="w", pady=2); widget.grid(row=i, column=1, sticky="ew", padx=6)
        g.columnconfigure(1, weight=1)
        row(0, "Preset", ttk.Combobox(g, textvariable=v["preset"], values=list(PRESETS), state="readonly", width=32))
        row(1, "Seconds", ttk.Spinbox(g, textvariable=v["seconds"], from_=1.0, to=10.0, increment=0.5, width=8))
        row(2, "Generation fps", ttk.Combobox(g, textvariable=v["fps"], values=["12", "24"], width=8, state="readonly"))
        row(3, "Stage-1 size (x2 after refinement)", ttk.Combobox(g, textvariable=v["size"], values=SIZES, width=12))
        row(4, "Stage-1 steps", ttk.Combobox(g, textvariable=v["s1"], values=["8", "6"], width=8, state="readonly"))
        row(5, "Refinement steps (0-3)", ttk.Spinbox(g, textvariable=v["s2"], from_=0, to=3, width=8))
        sr = ttk.Frame(g); ttk.Entry(sr, textvariable=v["seed"], width=10).pack(side="left"); ttk.Button(sr, text="Random", command=self.rand_seed).pack(side="left", padx=4)
        row(6, "Seed", sr)
        row(7, "Name", ttk.Entry(g, textvariable=v["name"]))
        ttk.Checkbutton(g, text="Reuse stage 1 of the same name+seed (re-run refinement only)", variable=self.reuse).grid(row=8, column=0, columnspan=2, sticky="w")
        v["preset"].trace_add("write", self.apply_preset)
        v["preset"].set("Flash (recommended)"); v["seconds"].set("5.4"); v["name"].set("my-clip"); self.rand_seed()
        for k in ("seconds", "fps", "size", "s1", "s2"): v[k].trace_add("write", lambda *a: self.update_info())


        self.status = ttk.Label(right, text="Idle", font=("Segoe UI", 12, "bold")); self.status.grid(row=0, column=0, sticky="w")
        self.bar = ttk.Progressbar(right, maximum=1000, length=640); self.bar.grid(row=1, column=0, sticky="ew", pady=4)
        self.time_lbl = ttk.Label(right, text=""); self.time_lbl.grid(row=2, column=0, sticky="w")
        self.cap = ttk.Label(right, text="Live preview appears here after the first denoising step (a colour sketch of the model's current guess: first / middle / last frame)", wraplength=640)
        self.cap.grid(row=3, column=0, sticky="w", pady=(10, 2))
        self.prev_lbl = ttk.Label(right, relief="groove", anchor="center"); self.prev_lbl.grid(row=4, column=0, sticky="ew", ipady=40)
        self.scrub = ttk.Scale(right, from_=0, to=0, command=self.scrub_to); self.scrub.grid(row=5, column=0, sticky="ew", pady=2)
        self.logbox = tk.Text(right, height=11, wrap="none", font=("Consolas", 8), state="disabled"); self.logbox.grid(row=6, column=0, sticky="nsew", pady=6)
        right.columnconfigure(0, weight=1); right.rowconfigure(6, weight=1)
        rb = ttk.Frame(right); rb.grid(row=7, sticky="ew")
        self.play_btn = ttk.Button(rb, text="Play result", command=lambda: self.video and os.startfile(self.video), state="disabled"); self.play_btn.pack(side="left")
        ttk.Button(rb, text="Open output folder", command=self.open_folder).pack(side="left", padx=6)
        self.update_info(); self.tick()

    # --- form logic
    def apply_preset(self, *a):
        p = PRESETS.get(self.v["preset"].get())
        if p:
            self.v["size"].set(p[0]); self.v["fps"].set(str(p[1])); self.v["s1"].set(str(p[2])); self.v["s2"].set(str(p[3]))

    def rand_seed(self): self.v["seed"].set(str(random.randint(1, 99999)))

    def count_words(self, *a):
        self.prompt.edit_modified(False)
        n = len(self.prompt.get("1.0", "end").split()); self.words.config(text=f"{n} words" + ("  (long prompts get truncated past ~350 words)" if n > 300 else ""))

    def settings(self):
        w, h = (int(x) for x in self.v["size"].get().lower().split("x"))
        return dict(w=w, h=h, seconds=float(self.v["seconds"].get()), fps=int(self.v["fps"].get()), s1=int(self.v["s1"].get()), s2=int(self.v["s2"].get()),
                    seed=int(self.v["seed"].get()), name=re.sub(r"[^A-Za-z0-9_-]+", "-", self.v["name"].get().strip()) or "clip")

    def update_info(self):
        try:
            s = self.settings()
            if s["w"] % 32 or s["h"] % 32: raise ValueError("size must be multiples of 32")
            fr = frames_for(s["seconds"], s["fps"]); e = estimate(s["w"], s["h"], s["seconds"], s["fps"], s["s1"], s["s2"])
            out = f"{s['w'] * 2}x{s['h'] * 2}" if s["s2"] else f"{s['w']}x{s['h']}"
            self.info.config(text=f"{fr} frames = {fr / s['fps']:.1f} s at {s['fps']} fps  ->  {out}.   Estimated time: ~{fmt(e['total'])}   (rough, from measured step costs)", foreground="")
            self.est = e
        except Exception as ex:
            self.info.config(text=f"check settings: {ex}", foreground="#b00020"); self.est = None
        self.refresh_thumb()

    # --- image
    def pick_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All", "*.*")])
        if p: self.src_image = p; self.refresh_thumb()

    def clear_image(self): self.src_image = None; self.refresh_thumb()

    def refresh_thumb(self):
        if not self.src_image:
            self.thumb_lbl.config(image="", text="(no image)"); self.thumb = None; return
        try:
            s = self.settings(); im = prep_image(self.src_image, s["w"], s["h"], self.v["fit"].get()); im.thumbnail((300, 170))
            self.thumb = ImageTk.PhotoImage(im); self.thumb_lbl.config(image=self.thumb, text="")
        except Exception as ex:
            self.thumb_lbl.config(image="", text=f"cannot use image: {ex}")

    # --- run control
    def start(self):
        if self.proc: return
        if not self.est: return messagebox.showerror("Settings", "Fix the settings first."), None
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt: return messagebox.showerror("Prompt", "Write a prompt first.")
        if free_ram_gb() < 3.0 and not messagebox.askyesno("Low memory", f"Only {free_ram_gb():.1f} GB RAM is free. Generation needs most of it - close browsers/apps first.\n\nStart anyway?"):
            return
        s = self.settings(); self.s = s
        pf = os.path.join(INPUTS, f"{s['name']}_prompt.txt"); open(pf, "w", encoding="utf-8").write(prompt + "\n")
        cmd = [PY, "render.py", "--mode", "custom", "--category", "studio", "--name", s["name"], "--prompt-file", os.path.relpath(pf, ROOT), "--seed", str(s["seed"]),
               "--seconds", str(s["seconds"]), "--w", str(s["w"]), "--h", str(s["h"]), "--fps", str(s["fps"]), "--s2", str(s["s2"])]
        if s["s1"] == 6: cmd += ["--s1sigmas", RECIPE6]
        if self.reuse.get(): cmd += ["--reuse-stage1"]
        if self.src_image:
            ip = os.path.join(INPUTS, f"{s['name']}_start.png"); prep_image(self.src_image, s["w"], s["h"], self.v["fit"].get()).save(ip)
            cmd += ["--image", os.path.relpath(ip, ROOT)]
        self.tag = f"{s['name']}_custom_seed{s['seed']}"
        self.log_path = os.path.join(ROOT, "logs", self.tag + ".log"); self.out_path = os.path.join(ROOT, "logs", "studio_last.out")
        try: os.remove(self.log_path)
        except OSError: pass
        self.log_pos = 0; self.prev_files = []; self.follow = True; self.video = None
        self.logbox.config(state="normal"); self.logbox.delete("1.0", "end"); self.logbox.config(state="disabled")
        self.tracker = Tracker(self.est)
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(self.out_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT, creationflags=0x08000000)
        self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal"); self.play_btn.config(state="disabled")
        self.status.config(text="Starting...")

    def stop(self):
        if self.proc and messagebox.askyesno("Stop", "Stop the generation? Progress of this clip is lost."):
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            self.status.config(text="Stopped by you")

    def open_folder(self):
        d = os.path.join(ROOT, "videos", "studio"); os.makedirs(d, exist_ok=True); os.startfile(d)

    # --- periodic update (cheap: reads only new log text, one small image)
    def tick(self):
        try:
            self.ram.config(text=f"free RAM {free_ram_gb():.1f} GB")
            if self.proc: self.poll()
        finally:
            self.root.after(1000, self.tick)

    def poll(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.log_pos); chunk = f.read(); self.log_pos = f.tell()
            lines = [ln.split("\r")[-1] for ln in chunk.split("\n") if ln.strip()]
            show = []
            for ln in lines:
                self.tracker.feed(ln)
                if ln.lstrip().startswith("[") and "it/s]" not in ln and "Warning" not in ln: show.append(ln.strip()[:150])
            if show:
                self.logbox.config(state="normal"); self.logbox.insert("end", "\n".join(show) + "\n"); self.logbox.see("end"); self.logbox.config(state="disabled")
        frac, left, phase = self.tracker.progress()
        rc = self.proc.poll()
        if rc is None:
            self.bar["value"] = frac * 1000
            self.status.config(text=f"{phase}"); self.time_lbl.config(text=f"elapsed {fmt(time.time() - self.tracker.t0)}   |   about {fmt(left)} left")
            self.load_previews()
        else:
            self.finish(rc)

    def finish(self, rc):
        out = open(self.out_path, encoding="utf-8", errors="replace").read()
        self.proc = None; self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled")
        total = fmt(time.time() - self.tracker.t0)
        m = re.search(r"output: (.+\.mp4)", out)
        if rc == 0 and m and os.path.exists(m[1].strip()):
            self.video = m[1].strip(); self.bar["value"] = 1000; self.status.config(text="Done")
            self.time_lbl.config(text=f"finished in {total}  ->  {self.video}"); self.play_btn.config(state="normal")
        elif "Stopped by you" in self.status.cget("text"):
            self.time_lbl.config(text=f"stopped after {total}")
        else:
            self.status.config(text="Failed")
            self.time_lbl.config(text=f"failed after {total} (exit {rc}). See logs\\{self.tag}.log")
            messagebox.showerror("Generation failed", (out[-900:] or "no output") + f"\n\nFull log: logs\\{self.tag}.log")

    # --- live preview
    def load_previews(self):
        d = os.path.join(ROOT, "previews", self.tag or "")
        if not os.path.isdir(d): return
        files = [f for f in os.listdir(d) if f.endswith(".png") and not f.endswith(".tmp.png")]
        key = lambda f: (0 if f.startswith("C_") else 1, int(re.search(r"step(\d+)", f)[1]))
        files.sort(key=key)
        if files != self.prev_files:
            self.prev_files = files; self.scrub.config(to=max(0, len(files) - 1))
            if self.follow: self.scrub.set(len(files) - 1); self.show_preview(len(files) - 1)

    def scrub_to(self, val):
        i = int(float(val))
        if self.prev_files and i != getattr(self, "_shown", -1):
            self.follow = (i == len(self.prev_files) - 1); self.show_preview(i)

    def show_preview(self, i):
        if not self.prev_files or i >= len(self.prev_files): return
        f = self.prev_files[i]; self._shown = i
        try:
            im = Image.open(os.path.join(ROOT, "previews", self.tag, f)); im.load()
            maxw = max(300, self.prev_lbl.winfo_width() - 12)
            if im.width != maxw: im = im.resize((maxw, round(im.height * maxw / im.width)), Image.BICUBIC)
            self.prev_img = ImageTk.PhotoImage(im); self.prev_lbl.config(image=self.prev_img)
            st = "Stage 1 (low-res)" if f.startswith("C_") else "Stage 2 (refinement)"
            self.cap.config(text=f"{st}, step {re.search(r'step(\\d+)', f)[1]}: the model's current guess of the final video (colour sketch of first / middle / last frame). Drag the slider to replay.")
        except Exception:
            pass


def main():
    try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass
    root = tk.Tk(); app = App(root)
    if "--shot" in sys.argv:
        from PIL import ImageGrab
        def snap():
            root.update(); x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(sys.argv[sys.argv.index("--shot") + 1]); root.destroy()
        root.after(1500, snap)
    if "--auto-test" in sys.argv:                           # verification harness: fill the form, run a tiny clip, screenshot, exit
        from PIL import ImageGrab
        shot_dir = sys.argv[sys.argv.index("--auto-test") + 1]; os.makedirs(shot_dir, exist_ok=True)
        app.v["preset"].set("Smoke test (tiny)"); app.v["seconds"].set("1.4"); app.v["name"].set("ui-autotest"); app.v["seed"].set("11")
        app.prompt.insert("1.0", open(os.path.join(ROOT, "prompts", "anime_swordsman-bridge.txt"), encoding="utf-8").read().strip())
        if "--image" in sys.argv: app.src_image = sys.argv[sys.argv.index("--image") + 1]
        app.update_info(); n = [0]
        def grab(tag):
            root.update(); x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(os.path.join(shot_dir, f"ui_{tag}.png"))
        def step():
            n[0] += 1
            if n[0] == 2: grab("form"); app.start()
            if n[0] in (60, 120, 170): grab(f"run{n[0]}")
            if app.proc is None and n[0] > 5:
                grab("end"); root.destroy(); return
            root.after(1000, step)
        root.after(1500, step)
    root.mainloop()


if __name__ == "__main__":
    main()
