// Measures unbuffered sequential NVMe read speed and RAM memcpy bandwidth.
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "bench_file.bin";
    const size_t GB = 1ull << 30, total = 6 * GB, chunk = 16ull << 20;
    char* buf = (char*)VirtualAlloc(nullptr, chunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (size_t i = 0; i < chunk; i += 8) *(unsigned long long*)(buf + i) = i * 0x9E3779B97F4A7C15ull;

    // write test file (buffered off so it doesn't linger in cache)
    HANDLE w = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (w == INVALID_HANDLE_VALUE) { printf("create failed %lu\n", GetLastError()); return 1; }
    auto t0 = clk::now();
    for (size_t off = 0; off < total; off += chunk) { DWORD n; WriteFile(w, buf, (DWORD)chunk, &n, nullptr); }
    CloseHandle(w);
    double ws = std::chrono::duration<double>(clk::now() - t0).count();
    printf("disk write: %.2f GB/s\n", total / (double)GB / ws);

    for (int pass = 0; pass < 2; pass++) {
        HANDLE r = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        t0 = clk::now();
        for (size_t off = 0; off < total; off += chunk) { DWORD n; ReadFile(r, buf, (DWORD)chunk, &n, nullptr); }
        CloseHandle(r);
        double rs = std::chrono::duration<double>(clk::now() - t0).count();
        printf("disk read (unbuffered, pass %d): %.2f GB/s\n", pass, total / (double)GB / rs);
    }
    DeleteFileA(path);

    size_t n = 512ull << 20;
    char* a = (char*)VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    char* b = (char*)VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    memset(a, 1, n); memset(b, 2, n);
    t0 = clk::now();
    for (int i = 0; i < 10; i++) memcpy(b, a, n);
    double ms = std::chrono::duration<double>(clk::now() - t0).count();
    printf("RAM memcpy: %.1f GB/s (read+write traffic ~%.1f GB/s)\n", 10.0 * n / GB / ms, 20.0 * n / GB / ms);
    return 0;
}
