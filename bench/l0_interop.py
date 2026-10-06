"""Can a kernel compiled by the OpenCL driver run, via Level Zero (ctypes), directly on PyTorch XPU tensors?

Plan: build an OpenCL C kernel with pyopencl -> extract the device-native binary -> load it with zeModuleCreate
(ZE_MODULE_FORMAT_NATIVE) in a NEW Level Zero context -> launch it on torch tensors' device pointers
(torch allocates in its own L0 context; we test whether another context in the same process may use them).
"""
import ctypes as C
import sys
import numpy as np
import pyopencl as cl
import torch

# ---------- 1. compile with the OpenCL driver, grab the native binary ----------
SRC = """
__kernel void scale(__global const float *in, __global float *out, float s) {
    size_t i = get_global_id(0); out[i] = in[i] * s;
}"""
ctx = cl.create_some_context(interactive=False)
prog = cl.Program(ctx, SRC).build()
binary = prog.binaries[0]
print(f"native binary: {len(binary)} bytes, magic {binary[:4]!r}")

# ---------- 2. Level Zero via ctypes ----------
ze = C.WinDLL("ze_loader.dll")
class Desc(C.Structure):
    pass
def ok(r, what):
    if r != 0:
        print(f"{what} failed: 0x{r:08x}"); sys.exit(1)

# enum values from ze_api.h (ze_structure_type_t)
ST_CONTEXT_DESC, ST_CMDQ_DESC, ST_CMDLIST_DESC, ST_MODULE_DESC, ST_KERNEL_DESC = 0xD, 0xE, 0xF, 0x1B, 0x1D
ZE_MODULE_FORMAT_NATIVE = 1

class ContextDesc(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32)]
class CmdQueueDesc(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("ordinal", C.c_uint32), ("index", C.c_uint32),
                ("flags", C.c_uint32), ("mode", C.c_uint32), ("priority", C.c_uint32)]
class ModuleDesc(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("format", C.c_uint32), ("inputSize", C.c_size_t),
                ("pInputModule", C.c_void_p), ("pBuildFlags", C.c_char_p), ("pConstants", C.c_void_p)]
class KernelDesc(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32), ("pKernelName", C.c_char_p)]
class GroupCount(C.Structure):
    _fields_ = [("x", C.c_uint32), ("y", C.c_uint32), ("z", C.c_uint32)]

ok(ze.zeInit(0), "zeInit")
n = C.c_uint32(0); ze.zeDriverGet(C.byref(n), None)
drivers = (C.c_void_p * n.value)(); ok(ze.zeDriverGet(C.byref(n), drivers), "zeDriverGet")
driver = C.c_void_p(drivers[0])
n = C.c_uint32(0); ze.zeDeviceGet(driver, C.byref(n), None)
devs = (C.c_void_p * n.value)(); ok(ze.zeDeviceGet(driver, C.byref(n), devs), "zeDeviceGet")
device = C.c_void_p(devs[0])
print(f"L0: {len(drivers)} driver(s), {len(devs)} device(s)")

zctx = C.c_void_p()
ok(ze.zeContextCreate(driver, C.byref(ContextDesc(ST_CONTEXT_DESC, None, 0)), C.byref(zctx)), "zeContextCreate")
cl_desc = CmdQueueDesc(ST_CMDQ_DESC, None, 0, 0, 0, 1, 0)       # mode 1 = synchronous
cmdlist = C.c_void_p()
ok(ze.zeCommandListCreateImmediate(zctx, device, C.byref(cl_desc), C.byref(cmdlist)), "zeCommandListCreateImmediate")

buf = C.create_string_buffer(binary, len(binary))
mdesc = ModuleDesc(ST_MODULE_DESC, None, ZE_MODULE_FORMAT_NATIVE, len(binary), C.cast(buf, C.c_void_p), None, None)
module = C.c_void_p()
r = ze.zeModuleCreate(zctx, device, C.byref(mdesc), C.byref(module), None)
ok(r, "zeModuleCreate(native binary from OpenCL)")
kernel = C.c_void_p()
ok(ze.zeKernelCreate(module, C.byref(KernelDesc(ST_KERNEL_DESC, None, 0, b"scale")), C.byref(kernel)), "zeKernelCreate")
print("module + kernel loaded in a fresh Level Zero context")

# ---------- 3. run on torch's tensors ----------
N = 1 << 20
x = torch.arange(N, device="xpu", dtype=torch.float32); y = torch.zeros(N, device="xpu", dtype=torch.float32)
torch.xpu.synchronize()
px, py = C.c_void_p(x.data_ptr()), C.c_void_p(y.data_ptr())
ok(ze.zeKernelSetArgumentValue(kernel, 0, C.sizeof(C.c_void_p), C.byref(px)), "set arg0")
ok(ze.zeKernelSetArgumentValue(kernel, 1, C.sizeof(C.c_void_p), C.byref(py)), "set arg1")
s = C.c_float(3.0)
ok(ze.zeKernelSetArgumentValue(kernel, 2, C.sizeof(C.c_float), C.byref(s)), "set arg2")
ok(ze.zeKernelSetGroupSize(kernel, 256, 1, 1), "zeKernelSetGroupSize")
ok(ze.zeCommandListAppendLaunchKernel(cmdlist, kernel, C.byref(GroupCount(N // 256, 1, 1)), None, 0, None), "launch")
ok(ze.zeCommandListHostSynchronize(cmdlist, C.c_uint64(-1 & 0xFFFFFFFFFFFFFFFF)), "sync")
got = y.cpu(); exp = x.cpu() * 3.0
print("RESULT: custom kernel on torch tensors correct =", torch.equal(got, exp), "| sample", got[:4].tolist())
