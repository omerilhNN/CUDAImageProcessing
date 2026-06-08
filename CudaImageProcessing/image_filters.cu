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
 *    - Four CUDA variants per filter (where applicable):
 *        * naive      : one thread per pixel, global memory only
 *        * tiled      : shared-memory tiled 2D convolution
 *        * separable  : two 1D passes for Gaussian (O(K^2) -> O(2K))
 *        * stream     : multi-stream pipelined H2D/kernel/D2H overlap
 *    - CUDA error checking (CUDA_CHECK / CUDA_CHECK_KERNEL)
 *    - Correctness validation (CPU vs GPU, with per-variant tolerance)
 *    - Timing: std::chrono for CPU, cudaEvent for kernel, H2D and D2H transfers
 *    - Speedup (kernel and end-to-end), throughput (MPixels/s), GFLOP/s
 *    - Per-config roofline metrics: arithmetic intensity, achieved GB/s
 *    - Theoretical peak memory bandwidth derived from device properties
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
__constant__ float c_gauss[MAX_GAUSS_K * MAX_GAUSS_K];     // 2D weights
__constant__ float c_gauss1D[MAX_GAUSS_K];                 // 1D weights (separable)
__constant__ float c_sobelX[9];
__constant__ float c_sobelY[9];

enum FilterType { F_GAUSSIAN = 0, F_SOBEL = 1, F_MEDIAN = 2 };
// Variants:
//   V_NAIVE     : one thread per pixel, all reads from global memory
//   V_TILED     : shared-memory tiled 2D convolution (KxK)
//   V_SEPARABLE : two-pass 1D convolution (horizontal then vertical),
//                 reduces arithmetic from O(K^2) to O(2K) per output pixel.
//                 Mathematically equivalent for Gaussian (the 2D Gaussian is
//                 the outer product of two 1D Gaussians). Not applicable to
//                 Sobel-magnitude (sqrt of two gradients) or Median (rank op).
//   V_STREAM    : tiled 2D convolution, but with H2D/kernel/D2H overlapped
//                 across multiple CUDA streams (end-to-end pipelining).
enum Variant { V_NAIVE = 0, V_TILED = 1, V_SEPARABLE = 2, V_STREAM = 3 };

static const char* filterName(FilterType f) {
    switch (f) {
    case F_GAUSSIAN: return "Gaussian";
    case F_SOBEL:    return "Sobel";
    default:         return "Median";
    }
}
static const char* variantName(Variant v) {
    switch (v) {
    case V_NAIVE:     return "naive";
    case V_TILED:     return "tiled";
    case V_SEPARABLE: return "separable";
    default:          return "stream";
    }
}

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

// Build a normalized 1D Gaussian kernel for separable filtering.
// The 2D Gaussian G(x,y) = exp(-(x^2+y^2)/2s^2) factors as g(x)*g(y) where
// g(t) = exp(-t^2/2s^2). Applying g horizontally then vertically is therefore
// mathematically equivalent to the full 2D convolution, but cuts the work
// per output pixel from K^2 multiply-adds to 2*K.
static void makeGaussianKernel1D(std::vector<float>& k, int radius, float sigma) {
    const int K = 2 * radius + 1;
    k.assign(K, 0.0f);
    float sum = 0.0f;
    for (int x = -radius; x <= radius; ++x) {
        float w = std::exp(-(x * x) / (2.0f * sigma * sigma));
        k[x + radius] = w;
        sum += w;
    }
    for (float& w : k) w /= sum;
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
// 6b. SEPARABLE GAUSSIAN  (V_SEPARABLE)
//     Two 1D passes instead of one 2D pass. For a KxK kernel this reduces
//     arithmetic intensity from K^2 to 2K MAdds per output pixel (a 7.5x
//     reduction at K=15) while producing a bit-equivalent result up to
//     floating-point rounding. Two kernels: horizontal then vertical. Both
//     use a thin tile (only halo in the pass direction) to maximize shared
//     memory reuse and minimize bank conflicts.
//
//     Pass 1 (horizontal):  out[y,x] = sum_k g1D[k] * in[y, x + k - r]
//     Pass 2 (vertical):    out[y,x] = sum_k g1D[k] * in[y + k - r, x]
// ============================================================================

// Horizontal pass: blockDim.y rows, each thread reads a tile of width
// (blockDim.x + 2*radius) and reduces across radius in X.
__global__ void gaussianSeparableH(const unsigned char* in, float* out,
    int W, int H, int radius) {
    extern __shared__ unsigned char stile[];
    int tileW = blockDim.x + 2 * radius;
    // Load: each row independent, halo only in X.
    int baseX = blockIdx.x * blockDim.x - radius;
    int baseY = blockIdx.y * blockDim.y;
    int ly = threadIdx.y;
    int gy = baseY + ly;
    int gyClamped = min(max(gy, 0), H - 1);
    for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
        int gx = min(max(baseX + tx, 0), W - 1);
        stile[ly * tileW + tx] = in[(size_t)gyClamped * W + gx];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W || gy >= H) return;
    const int K = 2 * radius + 1;
    float acc = 0.0f;
    for (int kx = 0; kx < K; ++kx)
        acc += stile[ly * tileW + (threadIdx.x + kx)] * c_gauss1D[kx];
    // Write intermediate as float to avoid quantizing twice (better PSNR).
    out[(size_t)gy * W + x] = acc;
}

// Vertical pass: reads float intermediate, halo only in Y, writes uint8.
__global__ void gaussianSeparableV(const float* in, unsigned char* out,
    int W, int H, int radius) {
    extern __shared__ float ftile[];
    int tileH = blockDim.y + 2 * radius;
    int baseX = blockIdx.x * blockDim.x;
    int baseY = blockIdx.y * blockDim.y - radius;
    int lx = threadIdx.x;
    int gx = baseX + lx;
    int gxClamped = min(max(gx, 0), W - 1);
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y) {
        int gy = min(max(baseY + ty, 0), H - 1);
        ftile[ty * blockDim.x + lx] = in[(size_t)gy * W + gxClamped];
    }
    __syncthreads();

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= W || y >= H) return;
    const int K = 2 * radius + 1;
    float acc = 0.0f;
    for (int ky = 0; ky < K; ++ky)
        acc += ftile[(threadIdx.y + ky) * blockDim.x + lx] * c_gauss1D[ky];
    out[(size_t)y * W + gx] = (unsigned char)clampf(acc + 0.5f, 0.0f, 255.0f);
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

// Forward declaration for the streamed/pipelined runner (defined below).
static GpuTiming runGpuStreamed(FilterType filter, const unsigned char* h_in,
    unsigned char* h_out, int W, int H, int radius, dim3 block, int iters,
    int nStreams, int nTiles);

static GpuTiming runGpu(FilterType filter, Variant variant,
    const unsigned char* h_in, unsigned char* h_out,
    unsigned char* d_in, unsigned char* d_out,
    int W, int H, int radius, dim3 block, int iters) {
    // Streamed variant has its own dedicated runner: it needs pinned host
    // memory and multiple device buffers, which the single-shot path does
    // not allocate. Delegate.
    if (variant == V_STREAM) {
        return runGpuStreamed(filter, h_in, h_out, W, H, radius, block, iters,
            /*nStreams=*/4, /*nTiles=*/8);
    }

    const size_t bytes = (size_t)W * H;
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    int tileW = block.x + 2 * radius;
    int tileH = block.y + 2 * radius;
    size_t shmem = (size_t)tileW * tileH * sizeof(unsigned char); // tiled only

    // For the separable variant we need an intermediate float buffer of W*H
    // floats to hold the horizontal-pass output.
    float* d_inter = nullptr;
    if (variant == V_SEPARABLE) {
        CUDA_CHECK(cudaMalloc(&d_inter, (size_t)W * H * sizeof(float)));
    }

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    GpuTiming t{ 0, 0, 0 };

    // --- Host -> Device transfer ---
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
        else if (variant == V_TILED) {
            if (filter == F_GAUSSIAN) gaussianTiled << <grid, block, shmem >> > (d_in, d_out, W, H, radius);
            else if (filter == F_SOBEL)    sobelTiled << <grid, block, shmem >> > (d_in, d_out, W, H);
            else                           medianTiled << <grid, block, shmem >> > (d_in, d_out, W, H, radius);
        }
        else { // V_SEPARABLE - only valid for Gaussian
            // Horizontal pass: tile is blockDim.y rows x (blockDim.x + 2r) cols of uint8.
            size_t shH = (size_t)block.y * (block.x + 2 * radius) * sizeof(unsigned char);
            gaussianSeparableH << <grid, block, shH >> > (d_in, d_inter, W, H, radius);
            // Vertical pass: tile is (blockDim.y + 2r) rows x blockDim.x cols of float.
            size_t shV = (size_t)(block.y + 2 * radius) * block.x * sizeof(float);
            gaussianSeparableV << <grid, block, shV >> > (d_inter, d_out, W, H, radius);
        }
        };

    // --- Warm-up (not timed): primes context, caches, JIT ---
    launch();
    CUDA_CHECK_KERNEL();

    // --- Timed kernel: average over `iters` launches ---
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaGetLastError());
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    t.kernel_ms = total / iters;

    // --- Device -> Host transfer ---
    CUDA_CHECK(cudaEventRecord(a));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, a, b));

    if (d_inter) CUDA_CHECK(cudaFree(d_inter));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return t;
}

// ============================================================================
// 8b. STREAMED / PIPELINED RUNNER  (V_STREAM)
//   Splits the image into N horizontal strips and processes them on a
//   round-robin of CUDA streams. Each stream alternates H2D / kernel / D2H,
//   and CUDA overlaps these stages across streams as long as the host buffer
//   is pinned (cudaHostAlloc). The result is dramatic end-to-end (e2e)
//   speedup because the PCIe transfer time is hidden behind kernel execution.
//
//   For correctness with a convolution that has a halo, each strip must
//   include `radius` extra rows of overlap with its neighbours on both sides
//   (top and bottom). We only WRITE the strip's core rows back, so the
//   overlap is read-only and stitching is seamless.
// ============================================================================

static GpuTiming runGpuStreamed(FilterType filter, const unsigned char* h_in,
    unsigned char* h_out, int W, int H, int radius, dim3 block, int iters,
    int nStreams, int nTiles) {
    const size_t bytes = (size_t)W * H;

    // 1) Pinned host buffers (required for true async overlap).
    unsigned char* h_in_pinned = nullptr;
    unsigned char* h_out_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_in_pinned, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out_pinned, bytes, cudaHostAllocDefault));
    std::memcpy(h_in_pinned, h_in, bytes);

    // 2) One device input strip + one device output strip per stream, sized
    //    for the largest possible strip (ceil(H/nTiles) + 2*radius rows).
    int stripRowsMax = (H + nTiles - 1) / nTiles + 2 * radius;
    size_t stripBytesMax = (size_t)W * stripRowsMax;

    std::vector<cudaStream_t> streams(nStreams);
    std::vector<unsigned char*> d_inS(nStreams, nullptr);
    std::vector<unsigned char*> d_outS(nStreams, nullptr);
    for (int s = 0; s < nStreams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_inS[s], stripBytesMax));
        CUDA_CHECK(cudaMalloc(&d_outS[s], stripBytesMax));
    }

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));

    auto launchStrip = [&](int strip, cudaStream_t st) {
        int rowStart = strip * ((H + nTiles - 1) / nTiles);
        int rowEnd = std::min(rowStart + (H + nTiles - 1) / nTiles, H);
        if (rowStart >= rowEnd) return;

        // Strip with halo (clamped to image bounds).
        int srcRowStart = std::max(rowStart - radius, 0);
        int srcRowEnd = std::min(rowEnd + radius, H);
        int stripH = srcRowEnd - srcRowStart;
        size_t stripBytes = (size_t)W * stripH;

        // Async H2D (only this strip + halo).
        CUDA_CHECK(cudaMemcpyAsync(d_inS[strip % nStreams],
            h_in_pinned + (size_t)srcRowStart * W,
            stripBytes, cudaMemcpyHostToDevice, st));

        // Launch tiled kernel on the strip. Grid covers full strip including halo;
        // boundary checks inside the kernel keep us safe.
        dim3 grid((W + block.x - 1) / block.x, (stripH + block.y - 1) / block.y);
        int tileW = block.x + 2 * radius;
        int tileH = block.y + 2 * radius;
        size_t shmem = (size_t)tileW * tileH * sizeof(unsigned char);
        if (filter == F_GAUSSIAN)
            gaussianTiled << <grid, block, shmem, st >> > (d_inS[strip % nStreams],
                d_outS[strip % nStreams], W, stripH, radius);
        else if (filter == F_SOBEL)
            // Sobel kernel signature: (in, out, W, H) -- radius is fixed at 1.
            sobelTiled << <grid, block, shmem, st >> > (d_inS[strip % nStreams],
                d_outS[strip % nStreams], W, stripH);
        else
            medianTiled << <grid, block, shmem, st >> > (d_inS[strip % nStreams],
                d_outS[strip % nStreams], W, stripH, radius);

        // Async D2H: copy only the strip's core rows (skip halo) back.
        int coreOffsetInStrip = rowStart - srcRowStart;   // rows of halo at top
        int coreRows = rowEnd - rowStart;
        CUDA_CHECK(cudaMemcpyAsync(h_out_pinned + (size_t)rowStart * W,
            d_outS[strip % nStreams] + (size_t)coreOffsetInStrip * W,
            (size_t)W * coreRows, cudaMemcpyDeviceToHost, st));
        };

    // Warm-up pass (not timed).
    for (int strip = 0; strip < nTiles; ++strip)
        launchStrip(strip, streams[strip % nStreams]);
    for (auto& st : streams) CUDA_CHECK(cudaStreamSynchronize(st));

    // Timed end-to-end run, averaged over `iters`.
    CUDA_CHECK(cudaEventRecord(a));
    for (int it = 0; it < iters; ++it) {
        for (int strip = 0; strip < nTiles; ++strip)
            launchStrip(strip, streams[strip % nStreams]);
        for (auto& st : streams) CUDA_CHECK(cudaStreamSynchronize(st));
    }
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, a, b));

    // Copy result out of the pinned buffer.
    std::memcpy(h_out, h_out_pinned, bytes);

    // Cleanup.
    for (int s = 0; s < nStreams; ++s) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(d_inS[s]));
        CUDA_CHECK(cudaFree(d_outS[s]));
    }
    CUDA_CHECK(cudaFreeHost(h_in_pinned));
    CUDA_CHECK(cudaFreeHost(h_out_pinned));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));

    // For streamed mode we report the *full pipeline* time under kernel_ms
    // and zero H2D/D2H, since the whole point is that they are overlapped:
    // the wall-clock end-to-end time is the meaningful number.
    GpuTiming t{ 0.0f, total / iters, 0.0f };
    return t;
}

// ============================================================================
// 9. ONE FULL CONFIGURATION: CPU + GPU + validate + record one CSV row
// ============================================================================

static void uploadWeights(FilterType filter, int radius) {
    if (filter == F_GAUSSIAN) {
        std::vector<float> g;
        float sigma = std::max(1.0f, radius / 2.0f);
        makeGaussianKernel(g, radius, sigma);
        CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, g.data(), g.size() * sizeof(float)));
        // 1D weights for the separable variant.
        std::vector<float> g1;
        makeGaussianKernel1D(g1, radius, sigma);
        CUDA_CHECK(cudaMemcpyToSymbol(c_gauss1D, g1.data(), g1.size() * sizeof(float)));
    }
    // Sobel weights are constant for the whole program (uploaded once in main).
}

// Per-pixel arithmetic intensity model (FLOPs/byte from global memory).
// This is a rough analytic estimate used to place each kernel on the
// roofline plot. It counts:
//   - FLOPs per output pixel based on filter math
//   - Bytes transferred per output pixel, assuming perfect shared-memory
//     reuse for the tiled variants (1 byte read + 1 byte written), and
//     a full K*K reload for the naive variants.
//
// Returns: { flops_per_pixel, bytes_per_pixel }
struct RooflineInfo { double flops_pp; double bytes_pp; };

static RooflineInfo computeRoofline(FilterType filter, Variant variant, int radius) {
    int K = 2 * radius + 1;
    double flops = 0.0, bytes = 0.0;
    if (filter == F_GAUSSIAN) {
        if (variant == V_SEPARABLE) {
            // 2 passes, each K muls + K adds = 2K FLOPs/pixel -> 4K total.
            flops = 4.0 * K;
            // H pass: read K bytes, write 4 bytes float (intermediate).
            // V pass: read K floats (4K bytes), write 1 byte.
            // With shared-memory reuse: ~1 byte in, 4 bytes intermediate,
            // 4 bytes intermediate in (perfect reuse), 1 byte out.
            bytes = 1 + 4 + 4 + 1;
        }
        else {
            // 2D Gaussian: K^2 mul + K^2 add = 2*K^2 FLOPs/pixel.
            flops = 2.0 * K * K;
            bytes = (variant == V_NAIVE) ? (double)K * K + 1 : 1 + 1;
        }
    }
    else if (filter == F_SOBEL) {
        // 2 * 9 MAdds + magnitude (mul, mul, add, sqrt ~ 4 FLOPs) = ~40 FLOPs.
        flops = 40.0;
        bytes = (variant == V_NAIVE) ? 9.0 + 1 : 1 + 1;
    }
    else { // F_MEDIAN
        // Insertion sort on K*K elements: ~K^4 / 4 comparisons (avg).
        // We count each compare as 1 FLOP-equivalent op.
        flops = 0.25 * (double)K * K * K * K;
        bytes = (variant == V_NAIVE) ? (double)K * K + 1 : 1 + 1;
    }
    return { flops, bytes };
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

    // Tolerance: Gaussian/Sobel use float math -> last-bit rounding ok.
    // Median is integer -> exact. Separable Gaussian does two float passes
    // with an intermediate quantization, so allow slightly more slack.
    int tol = (filter == F_MEDIAN) ? 0 : (variant == V_SEPARABLE ? 2 : 1);
    ValStats vs = cpuValid ? validate(cpuOut.data(), gpuOut.data(), N, tol)
        : ValStats{ -1, -1, false };

    double totalGpu = gt.h2d_ms + gt.kernel_ms + gt.d2h_ms;
    double kSpeedup = cpuMs / gt.kernel_ms;
    double e2eSpeedup = cpuMs / totalGpu;
    double mpix = (double)N / (gt.kernel_ms * 1e-3) / 1e6;
    int K = 2 * radius + 1;

    // Roofline metrics.
    RooflineInfo ri = computeRoofline(filter, variant, radius);
    double totalFlops = ri.flops_pp * (double)N;
    double totalBytes = ri.bytes_pp * (double)N;
    double gflops = totalFlops / (gt.kernel_ms * 1e-3) / 1e9;
    double gbps = totalBytes / (gt.kernel_ms * 1e-3) / 1e9;
    double ai = ri.flops_pp / ri.bytes_pp;   // arithmetic intensity

    printf("  %-9s %-9s K=%-2d block=%2dx%-2d | CPU %9.3f ms | "
        "ker %8.4f | tot %8.3f | kSpd %7.1fx e2e %6.1fx | "
        "%6.1f MPix/s %6.1f GF/s AI=%4.1f | %s\n",
        filterName(filter), variantName(variant), K, block.x, block.y,
        cpuMs, gt.kernel_ms, totalGpu, kSpeedup, e2eSpeedup,
        mpix, gflops, ai,
        cpuValid ? (vs.pass ? "PASS" : "FAIL") : "skip");

    csv << filterName(filter) << ',' << W << ',' << H << ',' << K << ','
        << variantName(variant) << ',' << block.x << ',' << block.y << ','
        << cpuMs << ',' << gt.h2d_ms << ',' << gt.kernel_ms << ',' << gt.d2h_ms << ','
        << totalGpu << ',' << kSpeedup << ',' << e2eSpeedup << ',' << mpix << ','
        << gflops << ',' << gbps << ',' << ai << ','
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
                // Separable variant only meaningful for Gaussian.
                if (cfg.f == F_GAUSSIAN) {
                    runConfig(csv, cfg.f, V_SEPARABLE, W, H, radius, block, gpuIters,
                        h_in, cpuOut, cpuMs, true, d_in, d_out);
                }
            }
            // Streamed variant: report only at one canonical block size
            // (16x16). It is an end-to-end pipelining optimization, not
            // a kernel optimization, so block-size sweeping is not the
            // point. We do it once per (filter,radius).
            runConfig(csv, cfg.f, V_STREAM, W, H, radius, dim3(16, 16), gpuIters,
                h_in, cpuOut, cpuMs, true, d_in, d_out);
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
    // Note: cudaDeviceProp::memoryClockRate and ::clockRate were deprecated
    // and removed in CUDA 13. Theoretical peak bandwidth is therefore taken
    // as a runtime parameter to analyze_results.py (--peak-bw) rather than
    // computed here. For RTX 4060 Laptop ~256 GB/s; for RTX 3060 ~360 GB/s.
    printf("GPU: %s | SMs: %d | Compute %d.%d | GlobalMem %.1f GB | "
        "MemBus %d-bit\n",
        prop.name, prop.multiProcessorCount, prop.major, prop.minor,
        prop.totalGlobalMem / 1e9, prop.memoryBusWidth);

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
        "GFLOPS,GBps,ArithmeticIntensity,"
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
