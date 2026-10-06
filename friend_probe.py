"""System probe for the LTX-2.5 streaming port. Run it, then send back friend_probe_report.txt.

  python friend_probe.py                 # tests the drive that holds the current folder
  python friend_probe.py --dir D:\\temp   # test a specific drive/folder (put it on the SSD that will hold the model)

What it does (takes ~2-3 minutes):
  * reads CPU / RAM / OS / GPU model, VRAM, driver (via nvidia-smi if present)
  * measures real disk read speed (writes a 2 GiB temp file, reads it back bypassing the OS cache, then deletes it)
  * if PyTorch with CUDA is installed: measures GPU matmul speed, attention speed and RAM->GPU copy speed
It does NOT collect your username, IP address, file names or anything personal, and it uploads nothing.
Only needs plain Python 3.8+; PyTorch is optional (the report says if it was missing).
"""
import argparse, ctypes, json, mmap, os, platform, shutil, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.getcwd(), help="folder on the drive to benchmark")
ap.add_argument("--tag", default="", help="name for the report file, e.g. --tag diskE")
ap.add_argument("--size-gb", type=float, default=2.0, help="size of the temporary disk-test file")
args = ap.parse_args()
R = {"probe_version": 1}


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str)).stdout.strip()
    except Exception as e:
        return f"(failed: {e})"


def section(name):
    print(f"\n== {name} ==", flush=True)


# ---------------------------------------------------------------- system
section("System")
R["os"] = platform.platform()
R["python"] = sys.version.split()[0]
R["cpu"] = platform.processor()
R["cpu_cores_logical"] = os.cpu_count()
if os.name == "nt":
    R["cpu"] = sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"') or R["cpu"]

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]
    ms = MS(); ms.l = ctypes.sizeof(ms); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    R["ram_total_gb"], R["ram_free_gb"] = round(ms.tp / 2**30, 1), round(ms.ap / 2**30, 1)
else:
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        R["ram_total_gb"], R["ram_free_gb"] = round(mi["MemTotal"] / 2**20, 1), round(mi["MemAvailable"] / 2**20, 1)
        R["cpu"] = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), R["cpu"])
    except Exception as e:
        R["ram_error"] = str(e)
for k in ("os", "cpu", "cpu_cores_logical", "ram_total_gb", "ram_free_gb", "python"):
    print(f"  {k}: {R.get(k)}")

# ---------------------------------------------------------------- GPU (nvidia-smi)
section("GPU (nvidia-smi)")
smi = sh("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,pcie.link.gen.max,pcie.link.width.max,"
         "pcie.link.gen.current,pcie.link.width.current --format=csv,noheader")
R["nvidia_smi"] = smi
print("  " + (smi or "nvidia-smi not found (NVIDIA driver missing or not on PATH)"))

# ---------------------------------------------------------------- disk
section("Disk")
d = os.path.abspath(args.dir)
os.makedirs(d, exist_ok=True)
tot, used, free = shutil.disk_usage(d)
R["disk_dir_drive"] = os.path.splitdrive(d)[0] or "/"
R["disk_total_gb"], R["disk_free_gb"] = round(tot / 2**30), round(free / 2**30)
if os.name == "nt":
    drv = os.path.splitdrive(d)[0].rstrip(":")
    R["disk_type"] = sh(f'powershell -NoProfile -Command "Get-Partition -DriveLetter {drv} | Get-Disk | '
                        f'Get-PhysicalDisk | Select-Object FriendlyName,MediaType,BusType | ConvertTo-Json -Compress"')
print(f"  drive {R['disk_dir_drive']}: {R['disk_free_gb']} GB free of {R['disk_total_gb']} GB")
print(f"  type: {R.get('disk_type', '(unknown)')}")
need = args.size_gb + 1
if free < need * 2**30:
    print(f"  not enough free space for the {args.size_gb} GiB test file; skipping disk benchmark")
    R["disk_bench"] = "skipped: low space"
else:
    BLK = 4 * 2**20
    path = os.path.join(d, "_probe_tmp.bin")
    n_blocks = int(args.size_gb * 2**30 // BLK)
    size = n_blocks * BLK
    print(f"  writing {size / 2**30:.1f} GiB test file...", flush=True)
    chunk = os.urandom(BLK)
    with open(path, "wb") as f:
        for _ in range(n_blocks):
            f.write(chunk)
        f.flush(); os.fsync(f.fileno())
    buf = mmap.mmap(-1, BLK)              # page-aligned buffer, needed for unbuffered I/O
    res = {}
    try:
        if os.name == "nt":
            k32 = ctypes.windll.kernel32
            k32.CreateFileW.restype = ctypes.c_void_p
            k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                                        ctypes.c_ulong, ctypes.c_void_p]
            k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
            k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_ulong]
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 | 0x10000000, None)  # NO_BUFFERING | SEQUENTIAL_SCAN
            assert h not in (None, ctypes.c_void_p(-1).value), "CreateFile failed"
            addr = ctypes.addressof(ctypes.c_char.from_buffer(buf)); got = ctypes.c_ulong()

            def read_at(off):
                k32.SetFilePointerEx(h, off, None, 0)
                k32.ReadFile(h, addr, BLK, ctypes.byref(got), None)
                return got.value
            closer = lambda: k32.CloseHandle(h)
        else:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))

            def read_at(off):
                return os.preadv(fd, [buf], off)
            closer = lambda: os.close(fd)
        t = time.perf_counter(); nb = 0
        for i in range(n_blocks):
            nb += read_at(i * BLK)
        res["sequential_4MiB_GBps"] = round(nb / (time.perf_counter() - t) / 1e9, 2)
        import random
        random.seed(1); offs = [random.randrange(n_blocks) * BLK for _ in range(min(256, n_blocks))]
        t = time.perf_counter(); nb = sum(read_at(o) for o in offs)
        res["random_4MiB_GBps"] = round(nb / (time.perf_counter() - t) / 1e9, 2)
        closer()
    except Exception as e:
        res["error"] = repr(e)
    finally:
        try: os.remove(path)
        except OSError: pass
    R["disk_bench"] = res
    print(f"  uncached read speed: {res}")

# ---------------------------------------------------------------- torch / CUDA
section("PyTorch / CUDA")
try:
    import torch
    R["torch"] = torch.__version__
    R["torch_cuda_built"] = torch.version.cuda
    R["cuda_available"] = torch.cuda.is_available()
    print(f"  torch {torch.__version__}, cuda build {torch.version.cuda}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        R["gpu"] = dict(name=p.name, vram_gb=round(p.total_memory / 2**30, 1), capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count)
        print("  gpu:", R["gpu"])
        dev = "cuda"

        def bench(fn, iters=20, warm=3):
            for _ in range(warm): fn()
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(iters): fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / iters

        g = {}
        for dt_name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
            try:
                M, K, N = 4096, 4096, 4096
                a = torch.randn(M, K, device=dev, dtype=dt); b = torch.randn(K, N, device=dev, dtype=dt)
                s = bench(lambda: a @ b)
                g[f"matmul_{dt_name}_TFLOPS"] = round(2 * M * K * N / s / 1e12, 1)
                del a, b
            except Exception as e:
                g[f"matmul_{dt_name}_TFLOPS"] = f"error {e!r}"[:80]
        try:   # attention at roughly our 12k-token stage-2 size
            q = torch.randn(1, 32, 4096, 128, device=dev, dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn_like(q)
            f = torch.nn.functional.scaled_dot_product_attention
            s = bench(lambda: f(q, k, v))
            g["sdpa_bf16_TFLOPS"] = round(4 * 32 * 4096 * 4096 * 128 / s / 1e12, 1)
            del q, k, v
        except Exception as e:
            g["sdpa_bf16_TFLOPS"] = f"error {e!r}"[:80]
        try:   # RAM -> GPU copy speed (what weight streaming uses)
            n = 256 * 2**20
            host_p = torch.empty(n, dtype=torch.uint8).pin_memory(); host_n = torch.empty(n, dtype=torch.uint8)
            dst = torch.empty(n, dtype=torch.uint8, device=dev)
            s = bench(lambda: dst.copy_(host_p, non_blocking=True), iters=10)
            g["h2d_pinned_GBps"] = round(n / s / 1e9, 1)
            s = bench(lambda: dst.copy_(host_n), iters=5, warm=1)
            g["h2d_pageable_GBps"] = round(n / s / 1e9, 1)
            del host_p, host_n, dst
        except Exception as e:
            g["h2d_GBps"] = f"error {e!r}"[:80]
        free_b, total_b = torch.cuda.mem_get_info()
        g["vram_free_gb"] = round(free_b / 2**30, 1)
        R["gpu_bench"] = g
        for k_, v_ in g.items():
            print(f"  {k_}: {v_}")
    else:
        print("  CUDA not available in this PyTorch (CPU-only wheel, or no NVIDIA driver). GPU benchmarks skipped.")
except ImportError:
    R["torch"] = None
    print("  PyTorch is not installed - GPU benchmarks skipped (the rest of the report is still useful).")
    print("  Optional: pip install torch --index-url https://download.pytorch.org/whl/cu128   and run this again")

# ---------------------------------------------------------------- report
out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"friend_probe_report{'_' + args.tag if args.tag else ''}.json")
out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"friend_probe_report{'_' + args.tag if args.tag else ''}.txt")
json.dump(R, open(out_json, "w"), indent=2)
open(out_txt, "w").write(json.dumps(R, indent=2))
print(f"\nDone. Please send me this file: {out_txt}")
