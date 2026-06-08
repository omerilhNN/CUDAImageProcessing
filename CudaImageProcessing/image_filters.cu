/*
 * ============================================================================
 *  CENG479 - Parallel Programming
 *  GPU-Accelerated Image Processing:
 *    Gaussian Blur, Sobel Edge Detection, and Median Filtering with CUDA
 *
 *  Omer Faruk Ilhan (21118080023), Eren Diken (21118080077)
 *  Instructor: Asst. Prof. Dr. Huseyin Temucin
 * ----------------------------------------------------------------------------
 *  This single translation unit contains:
 *    - CPU (sequential) baselines for all three filters
 *    - Naive (global-memory) CUDA kernels for all three filters
 *    - Tiled (shared-memory) CUDA kernels for all three filters
 *    - CUDA error checking (CUDA_CHECK / CUDA_CHECK_KERNEL)
 *    - Correctness validation (CPU vs GPU, with tolerance)
 *    - Timing: std::chrono for CPU, cudaEvent for kernel, H2D and D2H transfers
 *    - Speedup (kernel and end-to-end), throughput (MPixels/s)
 *    - Configurable image size and a multi-resolution / multi-blocksize sweep
 *    - CSV output (results.csv) for report tables & graphs
 *    - Optional dependency-free PGM image output (--save) to inspect filters
 *
 *  It compiles with no external dependencies:
 *      nvcc -O3 image_filters.cu -o image_filters
 *
 *  Default run performs a single-resolution benchmark; pass --sweep for the
 *  full experimental matrix described in the proposal (see README.md).
 * ============================================================================
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

 // ============================================================================
 // 1. CUDA ERROR CHECKING  (proposal step 3)
 // ============================================================================

 // Wrap every CUDA API call: reports file, line and human-readable error string.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Use immediately after a kernel launch: catches launch errors AND
// synchronizes so asynchronous execution errors surface here.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// ============================================================================
// 2. CONSTANTS AND CONSTANT MEMORY
// ============================================================================

// Largest Gaussian kernel we support: 15x15 -> 225 weights.
static const int MAX_GAUSS_K = 15;
// Largest median window we support: 7x7 -> 49 samples (kept small on purpose,
// since each thread holds the whole window in registers/local memory).
static const int MAX_MEDIAN_WIN = 49;

// Read-only filter weights live in constant memory: cached and broadcast
// efficiently when every thread in a warp reads the same address.
__constant__ float c_gauss[MAX_GAUSS_K * MAX_GAUSS_K];
__constant__ float c_sobelX[9];
__constant__ float c_sobelY[9];

enum FilterType { F_GAUSSIAN = 0, F_SOBEL = 1, F_MEDIAN = 2 };
enum Variant { V_NAIVE = 0, V_TILED = 1 };

static const char* filterName(FilterType f) {
    switch (f) {
    case F_GAUSSIAN: return "Gaussian";
    case F_SOBEL:    return "Sobel";
    default:         return "Median";
    }
}
static const char* variantName(Variant v) { return v == V_NAIVE ? "naive" : "tiled"; }

// ============================================================================
// 3. SMALL HOST/DEVICE HELPERS
// ============================================================================

__host__ __device__ inline int   clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__host__ __device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Build a normalized 2D Gaussian kernel (weights sum to 1.0).
static void makeGaussianKernel(std::vector<float>& k, int radius, float sigma) {
    const int K = 2 * radius + 1;
    k.assign(K * K, 0.0f);
    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x) {
            float w = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            k[(y + radius) * K + (x + radius)] = w;
            sum += w;
        }
    for (float& w : k) w /= sum;          // normalize so the image stays in [0,255]
}

// Deterministic synthetic 8-bit grayscale image: smooth waves + a checkerboard
// + a disc + salt-and-pepper noise. The noise gives the median filter real work.
static void generateImage(std::vector<unsigned char>& img, int W, int H, unsigned seed) {
    img.resize((size_t)W * H);
    std::srand(seed);
    const float cx = W * 0.5f, cy = H * 0.5f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float fx = x / (float)W, fy = y / (float)H;
            float v = 128.0f + 70.0f * std::sin(fx * 20.0f) * std::cos(fy * 15.0f);
            if (((x / 32) + (y / 32)) & 1) v += 30.0f;            // checkerboard
            float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (d < W * 0.25f) v -= 45.0f;                         // dark disc
            img[(size_t)y * W + x] = (unsigned char)clampf(v, 0.0f, 255.0f);
        }
    }
    // ~2% salt-and-pepper noise
    long n = (long)W * H / 50;
    for (long i = 0; i < n; ++i) {
        int x = std::rand() % W, y = std::rand() % H;
        img[(size_t)y * W + x] = (std::rand() & 1) ? 255 : 0;
    }
}

// Dependency-free binary PGM writer so we can eyeball filter output.
static void writePGM(const char* path, const unsigned char* img, int W, int H) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Could not open %s for writing\n", path); return; }
    f << "P5\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(img), (std::streamsize)W * H);
}

// ============================================================================
// 4. CPU BASELINES  (proposal step 4) - the reference for validation & speedup
// ============================================================================

static void cpuGaussian(const unsigned char* in, unsigned char* out,
    int W, int H, const float* ker, int radius) {
    const int K = 2 * radius + 1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky)
                for (int kx = -radius; kx <= radius; ++kx) {
                    int sx = clampi(x + kx, 0, W - 1);   // replicate border
                    int sy = clampi(y + ky, 0, H - 1);
                    acc += in[(size_t)sy * W + sx] * ker[(ky + radius) * K + (kx + radius)];
                }
            out[(size_t)y * W + x] = (unsigned char)clampf(acc + 0.5f, 0.0f, 255.0f);
        }
}

static void cpuSobel(const unsigned char* in, unsigned char* out, int W, int H) {
    static const int gx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static const int gy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float sx = 0.0f, sy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = clampi(x + kx, 0, W - 1);
                    int py = clampi(y + ky, 0, H - 1);
                    int p = in[(size_t)py * W + px];
                    int idx = (ky + 1) * 3 + (kx + 1);
                    sx += gx[idx] * p;
                    sy += gy[idx] * p;
                }
            float mag = std::sqrt(sx * sx + sy * sy);
            out[(size_t)y * W + x] = (unsigned char)clampf(mag, 0.0f, 255.0f);
        }
}

static void cpuMedian(const unsigned char* in, unsigned char* out,
    int W, int H, int radius) {
    const int K = 2 * radius + 1;
    const int n = K * K;
    unsigned char w[MAX_MEDIAN_WIN];
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int c = 0;
            for (int ky = -radius; ky <= radius; ++ky)
                for (int kx = -radius; kx <= radius; ++kx) {
                    int sx = clampi(x + kx, 0, W - 1);
                    int sy = clampi(y + ky, 0, H - 1);
                    w[c++] = in[(size_t)sy * W + sx];
                }
            // insertion sort (n is small) then take the middle element
            for (int i = 1; i < n; ++i) {
                unsigned char key = w[i]; int j = i - 1;
                while (j >= 0 && w[j] > key) { w[j + 1] = w[j]; --j; }
                w[j + 1] = key;
            }
            out[(size_t)y * W + x] = w[n / 2];
        }
}

// ============================================================================
// 5. NAIVE CUDA KERNELS  (one thread per output pixel, all reads from global)
//    Boundary handling via clamp-to-edge so every thread is in-bounds-safe.
// ============================================================================

__global__ void gaussianNaive(const unsigned char* in, unsigned char* out,
    int W, int H, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;                       // grid may overshoot image
    const int K = 2 * radius + 1;
    float acc = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky)
        for (int kx = -radius; kx <= radius; ++kx) {
            int sx = min(max(x + kx, 0), W - 1);
            int sy = min(max(y + ky, 0), H - 1);
            acc += in[(size_t)sy * W + sx] * c_gauss[(ky + radius) * K + (kx + radius)];
        }
    out[(size_t)y * W + x] = (unsigned char)clampf(acc + 0.5f, 0.0f, 255.0f);
}

__global__ void sobelNaive(const unsigned char* in, unsigned char* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sx = 0.0f, sy = 0.0f;
    for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
            int px = min(max(x + kx, 0), W - 1);
            int py = min(max(y + ky, 0), H - 1);
            int p = in[(size_t)py * W + px];
            int idx = (ky + 1) * 3 + (kx + 1);
            sx += c_sobelX[idx] * p;
            sy += c_sobelY[idx] * p;
        }
    float mag = sqrtf(sx * sx + sy * sy);
    out[(size_t)y * W + x] = (unsigned char)clampf(mag, 0.0f, 255.0f);
}

__global__ void medianNaive(const unsigned char* in, unsigned char* out,
    int W, int H, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int K = 2 * radius + 1;
    const int n = K * K;
    unsigned char w[MAX_MEDIAN_WIN];
    int c = 0;
    for (int ky = -radius; ky <= radius; ++ky)
        for (int kx = -radius; kx <= radius; ++kx) {
            int sx = min(max(x + kx, 0), W - 1);
            int sy = min(max(y + ky, 0), H - 1);
            w[c++] = in[(size_t)sy * W + sx];
        }
    for (int i = 1; i < n; ++i) {
        unsigned char key = w[i]; int j = i - 1;
        while (j >= 0 && w[j] > key) { w[j + 1] = w[j]; --j; }
        w[j + 1] = key;
    }
    out[(size_t)y * W + x] = w[n / 2];
}

// ============================================================================
// 6. TILED (SHARED-MEMORY) CUDA KERNELS  (proposal step 14 / section 4.2)
//    Each block cooperatively loads a (blockDim + 2*radius) halo tile into
//    shared memory, then every thread reads its window from on-chip memory.
//    Dynamic shared memory is sized at launch = tileW * tileH bytes.
// ============================================================================

// Cooperative halo load shared by all tiled kernels.
__device__ inline void loadTile(unsigned char* tile, const unsigned char* in,
    int W, int H, int radius, int tileW, int tileH) {
    int baseX = blockIdx.x * blockDim.x - radius;   // top-left of tile in image space
    int baseY = blockIdx.y * blockDim.y - radius;
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y)
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = min(max(baseX + tx, 0), W - 1);
            int gy = min(max(baseY + ty, 0), H - 1);
            tile[ty * tileW + tx] = in[(size_t)gy * W + gx];
        }
}

__global__ void gaussianTiled(const unsigned char* in, unsigned char* out,
    int W, int H, int radius) {
    extern __shared__ unsigned char tile[];
    int tileW = blockDim.x + 2 * radius;
    int tileH = blockDim.y + 2 * radius;
    loadTile(tile, in, W, H, radius, tileW, tileH);
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int K = 2 * radius + 1;
    float acc = 0.0f;
    for (int ky = 0; ky < K; ++ky)
        for (int kx = 0; kx < K; ++kx)
            acc += tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)] * c_gauss[ky * K + kx];
    out[(size_t)y * W + x] = (unsigned char)clampf(acc + 0.5f, 0.0f, 255.0f);
}

__global__ void sobelTiled(const unsigned char* in, unsigned char* out, int W, int H) {
    extern __shared__ unsigned char tile[];
    const int radius = 1;
    int tileW = blockDim.x + 2 * radius;
    int tileH = blockDim.y + 2 * radius;
    loadTile(tile, in, W, H, radius, tileW, tileH);
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sx = 0.0f, sy = 0.0f;
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) {
            int p = tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)];
            int idx = ky * 3 + kx;
            sx += c_sobelX[idx] * p;
            sy += c_sobelY[idx] * p;
        }
    float mag = sqrtf(sx * sx + sy * sy);
    out[(size_t)y * W + x] = (unsigned char)clampf(mag, 0.0f, 255.0f);
}

__global__ void medianTiled(const unsigned char* in, unsigned char* out,
    int W, int H, int radius) {
    extern __shared__ unsigned char tile[];
    int tileW = blockDim.x + 2 * radius;
    int tileH = blockDim.y + 2 * radius;
    loadTile(tile, in, W, H, radius, tileW, tileH);
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int K = 2 * radius + 1;
    const int n = K * K;
    unsigned char w[MAX_MEDIAN_WIN];
    int c = 0;
    for (int ky = 0; ky < K; ++ky)
        for (int kx = 0; kx < K; ++kx)
            w[c++] = tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)];
    for (int i = 1; i < n; ++i) {
        unsigned char key = w[i]; int j = i - 1;
        while (j >= 0 && w[j] > key) { w[j + 1] = w[j]; --j; }
        w[j + 1] = key;
    }
    out[(size_t)y * W + x] = w[n / 2];
}

// ============================================================================
// 7. VALIDATION  (proposal step 5)
// ============================================================================

struct ValStats { int maxDiff; long mismatches; bool pass; };

// Outputs are 8-bit; CPU and GPU use the same float math, so they should match
// within +/- tol (1 absorbs occasional last-bit rounding differences on the
// linear/gradient filters; median is integer and uses tol = 0).
static ValStats validate(const unsigned char* cpu, const unsigned char* gpu,
    long N, int tol, int reportLimit = 5) {
    ValStats s{ 0, 0, true };
    int shown = 0;
    for (long i = 0; i < N; ++i) {
        int d = std::abs((int)cpu[i] - (int)gpu[i]);
        if (d > s.maxDiff) s.maxDiff = d;
        if (d > tol) {
            ++s.mismatches;
            if (shown < reportLimit) {
                printf("    mismatch @ %ld : CPU=%d GPU=%d (diff=%d)\n",
                    i, (int)cpu[i], (int)gpu[i], d);
                ++shown;
            }
        }
    }
    s.pass = (s.mismatches == 0);
    return s;
}

// ============================================================================
// 8. GPU TIMING DRIVER  (proposal steps 7, 8)
//    Measures H2D, kernel, D2H separately with cudaEvent. Does a warm-up
//    launch first (so context init / caching does not pollute the kernel time),
//    then averages the kernel over `iters` launches.
// ============================================================================

struct GpuTiming { float h2d_ms, kernel_ms, d2h_ms; };

static GpuTiming runGpu(FilterType filter, Variant variant,
    const unsigned char* h_in, unsigned char* h_out,
    unsigned char* d_in, unsigned char* d_out,
    int W, int H, int radius, dim3 block, int iters) {
    const size_t bytes = (size_t)W * H;
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    int tileW = block.x + 2 * radius;
    int tileH = block.y + 2 * radius;
    size_t shmem = (size_t)tileW * tileH * sizeof(unsigned char); // tiled only

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    GpuTiming t{ 0, 0, 0 };

    // --- Host -> Device transfer (step 8) ---
    CUDA_CHECK(cudaEventRecord(a));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, a, b));

    // Helper lambda to dispatch the right kernel.
    auto launch = [&]() {
        if (variant == V_NAIVE) {
            if (filter == F_GAUSSIAN) gaussianNaive << <grid, block >> > (d_in, d_out, W, H, radius);
            else if (filter == F_SOBEL)    sobelNaive << <grid, block >> > (d_in, d_out, W, H);
            else                           medianNaive << <grid, block >> > (d_in, d_out, W, H, radius);
        }
        else {
            if (filter == F_GAUSSIAN) gaussianTiled << <grid, block, shmem >> > (d_in, d_out, W, H, radius);
            else if (filter == F_SOBEL)    sobelTiled << <grid, block, shmem >> > (d_in, d_out, W, H);
            else                           medianTiled << <grid, block, shmem >> > (d_in, d_out, W, H, radius);
        }
        };

    // --- Warm-up (not timed): primes context, caches, JIT ---
    launch();
    CUDA_CHECK_KERNEL();

    // --- Timed kernel: average over `iters` launches (step 7) ---
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaGetLastError());                 // catch any launch error
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    t.kernel_ms = total / iters;

    // --- Device -> Host transfer (step 8) ---
    CUDA_CHECK(cudaEventRecord(a));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, a, b));

    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return t;
}

// ============================================================================
// 9. ONE FULL CONFIGURATION: CPU + GPU + validate + record one CSV row
// ============================================================================

static void uploadWeights(FilterType filter, int radius) {
    if (filter == F_GAUSSIAN) {
        std::vector<float> g;
        makeGaussianKernel(g, radius, /*sigma=*/std::max(1.0f, radius / 2.0f));
        CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, g.data(), g.size() * sizeof(float)));
    }
    // Sobel weights are constant for the whole program (uploaded once in main).
}

static void runConfig(std::ofstream& csv, FilterType filter, Variant variant,
    int W, int H, int radius, dim3 block, int gpuIters,
    const std::vector<unsigned char>& h_in,
    std::vector<unsigned char>& cpuOut, double cpuMs, bool cpuValid,
    unsigned char* d_in, unsigned char* d_out) {
    const long N = (long)W * H;
    std::vector<unsigned char> gpuOut(N);

    uploadWeights(filter, radius);
    GpuTiming gt = runGpu(filter, variant, h_in.data(), gpuOut.data(),
        d_in, d_out, W, H, radius, block, gpuIters);

    int tol = (filter == F_MEDIAN) ? 0 : 1;
    ValStats vs = cpuValid ? validate(cpuOut.data(), gpuOut.data(), N, tol)
        : ValStats{ -1, -1, false };

    double totalGpu = gt.h2d_ms + gt.kernel_ms + gt.d2h_ms;
    double kSpeedup = cpuMs / gt.kernel_ms;
    double e2eSpeedup = cpuMs / totalGpu;
    double mpix = (double)N / (gt.kernel_ms * 1e-3) / 1e6;
    int K = 2 * radius + 1;

    printf("  %-9s %-5s K=%-2d block=%2dx%-2d | CPU %9.3f ms | "
        "H2D %7.3f  ker %8.4f  D2H %7.3f | tot %8.3f | "
        "kSpd %7.1fx  e2e %6.1fx | %6.1f MPix/s | %s (maxDiff=%d)\n",
        filterName(filter), variantName(variant), K, block.x, block.y,
        cpuMs, gt.h2d_ms, gt.kernel_ms, gt.d2h_ms, totalGpu,
        kSpeedup, e2eSpeedup, mpix,
        cpuValid ? (vs.pass ? "PASS" : "FAIL") : "skip", vs.maxDiff);

    csv << filterName(filter) << ',' << W << ',' << H << ',' << K << ','
        << variantName(variant) << ',' << block.x << ',' << block.y << ','
        << cpuMs << ',' << gt.h2d_ms << ',' << gt.kernel_ms << ',' << gt.d2h_ms << ','
        << totalGpu << ',' << kSpeedup << ',' << e2eSpeedup << ',' << mpix << ','
        << vs.maxDiff << ',' << vs.mismatches << ','
        << (cpuValid ? (vs.pass ? "PASS" : "FAIL") : "skip") << '\n';
}

// ============================================================================
// 10. EXPERIMENT DRIVER for one resolution
// ============================================================================

static void runResolution(std::ofstream& csv, int W, int H, int gpuIters,
    bool save, bool fullKernelSizes) {
    printf("\n================ Image %d x %d (%d pixels) ================\n",
        W, H, W * H);
    std::vector<unsigned char> h_in;
    generateImage(h_in, W, H, /*seed=*/12345u);

    unsigned char* d_in = nullptr, * d_out = nullptr;
    const size_t bytes = (size_t)W * H;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    dim3 blocks[] = { dim3(8, 8), dim3(16, 16), dim3(32, 32) };

    struct Cfg { FilterType f; std::vector<int> radii; };
    std::vector<Cfg> cfgs;
    if (fullKernelSizes) {
        cfgs.push_back({ F_GAUSSIAN, {1, 2, 3, 7} });   // 3x3,5x5,7x7,15x15
        cfgs.push_back({ F_SOBEL,    {1} });            // fixed 3x3
        cfgs.push_back({ F_MEDIAN,   {1, 2, 3} });      // 3x3,5x5,7x7
    }
    else {
        cfgs.push_back({ F_GAUSSIAN, {2} });            // 5x5
        cfgs.push_back({ F_SOBEL,    {1} });
        cfgs.push_back({ F_MEDIAN,   {1} });            // 3x3
    }

    for (const Cfg& cfg : cfgs) {
        for (int radius : cfg.radii) {
            // CPU baseline computed ONCE per (filter,radius); reused for every
            // block size / variant since it does not depend on them.
            std::vector<unsigned char> cpuOut(bytes);
            auto t0 = std::chrono::high_resolution_clock::now();
            if (cfg.f == F_GAUSSIAN) {
                std::vector<float> g; makeGaussianKernel(g, radius, std::max(1.0f, radius / 2.0f));
                cpuGaussian(h_in.data(), cpuOut.data(), W, H, g.data(), radius);
            }
            else if (cfg.f == F_SOBEL) {
                cpuSobel(h_in.data(), cpuOut.data(), W, H);
            }
            else {
                cpuMedian(h_in.data(), cpuOut.data(), W, H, radius);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (save) {
                char path[128];
                std::snprintf(path, sizeof(path), "out_%s_K%d_%dx%d.pgm",
                    filterName(cfg.f), 2 * radius + 1, W, H);
                writePGM(path, cpuOut.data(), W, H);
            }

            for (dim3 block : blocks) {
                runConfig(csv, cfg.f, V_NAIVE, W, H, radius, block, gpuIters,
                    h_in, cpuOut, cpuMs, true, d_in, d_out);
                runConfig(csv, cfg.f, V_TILED, W, H, radius, block, gpuIters,
                    h_in, cpuOut, cpuMs, true, d_in, d_out);
            }
        }
    }

    if (save) writePGM(("in_" + std::to_string(W) + "x" + std::to_string(H) + ".pgm").c_str(),
        h_in.data(), W, H);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// 11. MAIN  (proposal steps 10, 11, 13, 17)
//   Usage:
//     image_filters                 single 1024x1024 benchmark
//     image_filters W [H] [iters]   single WxH benchmark (H defaults to W)
//     image_filters --sweep [iters] full matrix: 512/1024/2048/4096
//     extra flags: --save (write PGM images), --full-kernels (all kernel sizes)
// ============================================================================

int main(int argc, char** argv) {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) { fprintf(stderr, "No CUDA device found.\n"); return 1; }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Compute %d.%d | GlobalMem %.1f GB\n",
        prop.name, prop.multiProcessorCount, prop.major, prop.minor,
        prop.totalGlobalMem / 1e9);

    // Upload the (fixed) Sobel kernels to constant memory once.
    const float hSobelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    const float hSobelY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelX, hSobelX, sizeof(hSobelX)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelY, hSobelY, sizeof(hSobelY)));

    // --- argument parsing ---
    bool sweep = false, save = false, fullKernels = false;
    int  W = 1024, H = 0, iters = 10;
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sweep"))        sweep = true;
        else if (!std::strcmp(argv[i], "--save"))         save = true;
        else if (!std::strcmp(argv[i], "--full-kernels")) fullKernels = true;
        else pos.push_back(argv[i]);
    }
    if (sweep) {
        if (!pos.empty()) iters = std::atoi(pos[0]);
    }
    else {
        if (pos.size() > 0) W = std::atoi(pos[0]);
        H = W;
        if (pos.size() > 1) H = std::atoi(pos[1]);
        if (pos.size() > 2) iters = std::atoi(pos[2]);
    }
    if (iters < 1) iters = 1;

    std::ofstream csv("results.csv");
    csv << "Filter,Width,Height,KernelSize,Variant,BlockX,BlockY,"
        "CPU_ms,H2D_ms,Kernel_ms,D2H_ms,Total_GPU_ms,"
        "Kernel_Speedup,EndToEnd_Speedup,Throughput_MPixPerSec,"
        "MaxAbsDiff,Mismatches,Validation\n";

    printf("GPU kernel timing averaged over %d iteration(s).\n", iters);

    if (sweep) {
        int res[] = { 512, 1024, 2048, 4096 };
        for (int r : res) runResolution(csv, r, r, iters, save, fullKernels);
    }
    else {
        runResolution(csv, W, H, iters, save, fullKernels);
    }

    csv.close();
    printf("\nResults written to results.csv\n");
    CUDA_CHECK(cudaDeviceReset());   // flush profiler traces (Nsight/nvprof)
    return 0;
}
