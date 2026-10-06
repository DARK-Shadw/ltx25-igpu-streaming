"""Can an OpenCL kernel use PyTorch XPU (Level Zero / SYCL) device pointers directly?"""
import ctypes, numpy as np, torch, pyopencl as cl

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]; plat = dev.platform
q = cl.CommandQueue(ctx)
print("device:", dev.name, "| usm ext:", "cl_intel_unified_shared_memory" in dev.extensions)

ocl = ctypes.WinDLL("OpenCL.dll")
ocl.clGetExtensionFunctionAddressForPlatform.restype = ctypes.c_void_p
ocl.clGetExtensionFunctionAddressForPlatform.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
addr = ocl.clGetExtensionFunctionAddressForPlatform(plat.int_ptr, b"clSetKernelArgMemPointerINTEL")
print("clSetKernelArgMemPointerINTEL address:", hex(addr) if addr else None)
SetPtr = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p)(addr)

prog = cl.Program(ctx, """
__kernel void scale(__global const float *in, __global float *out, float s) {
    size_t i = get_global_id(0); out[i] = in[i] * s;
}""").build()
k = prog.scale

n = 1 << 20
x = torch.arange(n, device="xpu", dtype=torch.float32)
y = torch.zeros(n, device="xpu", dtype=torch.float32)
torch.xpu.synchronize()
print("torch ptrs:", hex(x.data_ptr()), hex(y.data_ptr()))

ok_flag = cl.enqueue_nd_range_kernel  # placeholder to keep import used
r0 = SetPtr(k.int_ptr, 0, ctypes.c_void_p(x.data_ptr()))
r1 = SetPtr(k.int_ptr, 1, ctypes.c_void_p(y.data_ptr()))
print("SetKernelArgMemPointerINTEL results:", r0, r1, "(0 = success)")
k.set_arg(2, np.float32(3.0))
try:
    ev = cl.enqueue_nd_range_kernel(q, k, (n,), None); ev.wait(); q.finish()
    got = y.cpu()
    exp = (x.cpu() * 3.0)
    print("kernel ran. correct:", torch.equal(got, exp), "| sample:", got[:4].tolist(), "expected:", exp[:4].tolist())
except Exception as e:
    print("kernel failed:", type(e).__name__, str(e)[:200])
