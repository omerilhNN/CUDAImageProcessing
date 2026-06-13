/*
 * ============================================================================
 *  CENG479 - Parallel Programming - LIVE DEMO
 *
 *  Real-time webcam (or video file) processing demo for the in-class
 *  presentation. Pulls frames with OpenCV, processes them with the same
 *  CUDA kernels used in image_filters.cu, and overlays a live FPS counter
 *  showing CPU vs GPU performance.
 *
 *  Keys at runtime:
 *      C   -- CPU baseline (single-threaded)
 *      G   -- GPU naive
 *      T   -- GPU tiled (shared memory)
 *      S   -- GPU stream-pipelined (true end-to-end overlap)
 *      1   -- Sobel edge detection
 *      2   -- Gaussian blur
 *      3   -- Median filter
 *      ESC -- quit
 *
 *  Build (Visual Studio command prompt, after installing OpenCV):
 *      nvcc -O3 -arch=sm_89 live_demo.cu ^
 *           -I"C:\opencv\build\include" ^
 *           -L"C:\opencv\build\x64\vc16\lib" -lopencv_world4100 ^
 *           -o live_demo.exe
 *
 *  Run with webcam:
 *      live_demo.exe
 *
 *  Run with video file:
 *      live_demo.exe path\to\video.mp4
 * ============================================================================
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <chrono>
#include <deque>
#include <string>
#include <numeric>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

 // ============================================================================
 // CONSTANT MEMORY for filter weights (same idea as image_filters.cu)
 // ============================================================================
__constant__ float c_sobelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
__constant__ float c_sobelY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
__constant__ float c_gauss5[25];        // 5x5 Gaussian weights

static void initGaussianWeights() {
    float k[25];
    float sigma = 1.4f;
    float sum = 0.0f;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x) {
            float w = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            k[(y + 2) * 5 + (x + 2)] = w;
            sum += w;
        }
    for (int i = 0; i < 25; ++i) k[i] /= sum;
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss5, k, sizeof(k)));
}

// ============================================================================
// CPU REFERENCE IMPLEMENTATIONS
// ============================================================================
static void cpuSobel(const unsigned char* in, unsigned char* out, int W, int H) {
    static const int gx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static const int gy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float sx = 0, sy = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = std::min(std::max(x + kx, 0), W - 1);
                    int py = std::min(std::max(y + ky, 0), H - 1);
                    int p = in[py * W + px];
                    int idx = (ky + 1) * 3 + (kx + 1);
                    sx += gx[idx] * p;
                    sy += gy[idx] * p;
                }
            float mag = std::sqrt(sx * sx + sy * sy);
            out[y * W + x] = (unsigned char)std::min(255.0f, std::max(0.0f, mag));
        }
}

static void cpuGaussian(const unsigned char* in, unsigned char* out, int W, int H) {
    float k[25];
    float sigma = 1.4f, sum = 0.0f;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x) {
            float w = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            k[(y + 2) * 5 + (x + 2)] = w;
            sum += w;
        }
    for (int i = 0; i < 25; ++i) k[i] /= sum;

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ++ky)
                for (int kx = -2; kx <= 2; ++kx) {
                    int sx = std::min(std::max(x + kx, 0), W - 1);
                    int sy = std::min(std::max(y + ky, 0), H - 1);
                    acc += in[sy * W + sx] * k[(ky + 2) * 5 + (kx + 2)];
                }
            out[y * W + x] = (unsigned char)std::min(255.0f, std::max(0.0f, acc + 0.5f));
        }
}

static void cpuMedian(const unsigned char* in, unsigned char* out, int W, int H) {
    unsigned char w[9];
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int c = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    int sx = std::min(std::max(x + kx, 0), W - 1);
                    int sy = std::min(std::max(y + ky, 0), H - 1);
                    w[c++] = in[sy * W + sx];
                }
            for (int i = 1; i < 9; ++i) {
                unsigned char key = w[i]; int j = i - 1;
                while (j >= 0 && w[j] > key) { w[j + 1] = w[j]; --j; }
                w[j + 1] = key;
            }
            out[y * W + x] = w[4];
        }
}

// ============================================================================
// GPU KERNELS (tiled with shared memory + halo)
// ============================================================================

__device__ inline int clampi_dev(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// --- Branchless 3x3 median (19 karsilastirma) ---
// Insertion sort'a gore daha hizli ve warp divergence yaratmaz.
__device__ inline void mmx(unsigned char& a, unsigned char& b) {
    unsigned char lo = a < b ? a : b;
    unsigned char hi = a < b ? b : a;
    a = lo; b = hi;
}
__device__ inline unsigned char median9(unsigned char* p) {
    mmx(p[1], p[2]); mmx(p[4], p[5]); mmx(p[7], p[8]);
    mmx(p[0], p[1]); mmx(p[3], p[4]); mmx(p[6], p[7]);
    mmx(p[1], p[2]); mmx(p[4], p[5]); mmx(p[7], p[8]);
    mmx(p[0], p[3]); mmx(p[5], p[8]); mmx(p[4], p[7]);
    mmx(p[3], p[6]); mmx(p[1], p[4]); mmx(p[2], p[5]);
    mmx(p[4], p[7]); mmx(p[4], p[2]); mmx(p[6], p[4]);
    mmx(p[4], p[2]);
    return p[4];
}

// --- Sobel (tiled) ---
__global__ void sobelTiled(const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out, int W, int H) {
    extern __shared__ unsigned char tile[];
    const int radius = 1;
    int tileW = blockDim.x + 2 * radius;
    int tileH = blockDim.y + 2 * radius;
    int baseX = blockIdx.x * blockDim.x - radius;
    int baseY = blockIdx.y * blockDim.y - radius;
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y)
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = clampi_dev(baseX + tx, 0, W - 1);
            int gy = clampi_dev(baseY + ty, 0, H - 1);
            tile[ty * tileW + tx] = in[gy * W + gx];
        }
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sx = 0, sy = 0;
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            int p = tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)];
            int idx = ky * 3 + kx;
            sx += c_sobelX[idx] * p;
            sy += c_sobelY[idx] * p;
        }
    float mag = sqrtf(sx * sx + sy * sy);
    out[y * W + x] = (unsigned char)fminf(255.0f, fmaxf(0.0f, mag));
}

__global__ void sobelNaive(const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sx = 0, sy = 0;
#pragma unroll
    for (int ky = -1; ky <= 1; ++ky)
#pragma unroll
        for (int kx = -1; kx <= 1; ++kx) {
            int px = clampi_dev(x + kx, 0, W - 1);
            int py = clampi_dev(y + ky, 0, H - 1);
            int p = in[py * W + px];
            int idx = (ky + 1) * 3 + (kx + 1);
            sx += c_sobelX[idx] * p;
            sy += c_sobelY[idx] * p;
        }
    float mag = sqrtf(sx * sx + sy * sy);
    out[y * W + x] = (unsigned char)fminf(255.0f, fmaxf(0.0f, mag));
}

// --- Gaussian (tiled, 5x5) ---
__global__ void gaussianTiled(const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out, int W, int H) {
    extern __shared__ unsigned char tile[];
    const int radius = 2;
    int tileW = blockDim.x + 2 * radius;
    int tileH = blockDim.y + 2 * radius;
    int baseX = blockIdx.x * blockDim.x - radius;
    int baseY = blockIdx.y * blockDim.y - radius;
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y)
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = clampi_dev(baseX + tx, 0, W - 1);
            int gy = clampi_dev(baseY + ty, 0, H - 1);
            tile[ty * tileW + tx] = in[gy * W + gx];
        }
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0;
    for (int ky = 0; ky < 5; ++ky)
        for (int kx = 0; kx < 5; ++kx)
            acc += tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)] * c_gauss5[ky * 5 + kx];
    out[y * W + x] = (unsigned char)fminf(255.0f, fmaxf(0.0f, acc + 0.5f));
}

__global__ void gaussianNaive(const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0;
    for (int ky = -2; ky <= 2; ++ky)
        for (int kx = -2; kx <= 2; ++kx) {
            int sx = clampi_dev(x + kx, 0, W - 1);
            int sy = clampi_dev(y + ky, 0, H - 1);
            acc += in[sy * W + sx] * c_gauss5[(ky + 2) * 5 + (kx + 2)];
        }
    out[y * W + x] = (unsigned char)fminf(255.0f, fmaxf(0.0f, acc + 0.5f));
}

// --- Median (tiled, 3x3) ---
__global__ void medianTiled(const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out, int W, int H) {
    extern __shared__ unsigned char tile[];
    const int radius = 1;
    int tileW = blockDim.x + 2 * radius;
    int tileH = blockDim.y + 2 * radius;
    int baseX = blockIdx.x * blockDim.x - radius;
    int baseY = blockIdx.y * blockDim.y - radius;
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y)
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x) {
            int gx = clampi_dev(baseX + tx, 0, W - 1);
            int gy = clampi_dev(baseY + ty, 0, H - 1);
            tile[ty * tileW + tx] = in[gy * W + gx];
        }
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    unsigned char w[9];
    int c = 0;
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx)
            w[c++] = tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)];
    out[y * W + x] = median9(w);
}

__global__ void medianNaive(const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    unsigned char w[9];
    int c = 0;
    for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
            int sx = clampi_dev(x + kx, 0, W - 1);
            int sy = clampi_dev(y + ky, 0, H - 1);
            w[c++] = in[sy * W + sx];
        }
    out[y * W + x] = median9(w);
}

// ============================================================================
// PROCESSING MODES
// ============================================================================
enum Backend { B_CPU, B_GPU_NAIVE, B_GPU_TILED, B_GPU_STREAM };
enum FilterMode { F_SOBEL, F_GAUSSIAN, F_MEDIAN };

static const char* backendName(Backend b) {
    switch (b) {
    case B_CPU:         return "CPU (1 thread)";
    case B_GPU_NAIVE:   return "GPU naive";
    case B_GPU_TILED:   return "GPU tiled (shared mem)";
    case B_GPU_STREAM:  return "GPU stream-pipelined";
    }
    return "?";
}
static const char* filterModeName(FilterMode f) {
    switch (f) {
    case F_SOBEL:    return "Sobel edges";
    case F_GAUSSIAN: return "Gaussian blur";
    case F_MEDIAN:   return "Median 3x3";
    }
    return "?";
}

// ============================================================================
// GPU RUNNERS
// ============================================================================
struct GpuCtx {
    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    // For stream mode:
    unsigned char* h_in_pinned = nullptr;
    unsigned char* h_out_pinned = nullptr;
    std::vector<cudaStream_t> streams;
    std::vector<unsigned char*> d_inS;
    std::vector<unsigned char*> d_outS;
    size_t bytes = 0;
    int W = 0, H = 0;

    void allocate(int w, int h, int nStreams = 4) {
        if (W == w && H == h) return;
        free();
        W = w; H = h; bytes = (size_t)w * h;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaHostAlloc(&h_in_pinned, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_out_pinned, bytes, cudaHostAllocDefault));
        // Strip size for 8-stream pipeline. Boyut en buyuk radius'a gore
        // ayrilmali: Gaussian 5x5 -> radius 2. Aksi halde Gaussian+stream tasar.
        const int maxRadius = 2;
        int stripRowsMax = (H + 7) / 8 + 2 * maxRadius;
        size_t stripBytes = (size_t)W * stripRowsMax;
        streams.resize(nStreams); d_inS.resize(nStreams); d_outS.resize(nStreams);
        for (int s = 0; s < nStreams; ++s) {
            CUDA_CHECK(cudaStreamCreate(&streams[s]));
            CUDA_CHECK(cudaMalloc(&d_inS[s], stripBytes));
            CUDA_CHECK(cudaMalloc(&d_outS[s], stripBytes));
        }
    }
    void free() {
        if (d_in) { cudaFree(d_in); d_in = nullptr; }
        if (d_out) { cudaFree(d_out); d_out = nullptr; }
        if (h_in_pinned) { cudaFreeHost(h_in_pinned); h_in_pinned = nullptr; }
        if (h_out_pinned) { cudaFreeHost(h_out_pinned); h_out_pinned = nullptr; }
        for (auto s : streams) cudaStreamDestroy(s);
        for (auto p : d_inS) cudaFree(p);
        for (auto p : d_outS) cudaFree(p);
        streams.clear(); d_inS.clear(); d_outS.clear();
    }
    ~GpuCtx() { free(); }
};

static void runGpuSimple(GpuCtx& ctx, Backend backend, FilterMode filter,
    const unsigned char* h_in, unsigned char* h_out) {
    int W = ctx.W, H = ctx.H;
    // Pinned buffer uzerinden stage et: H2D/D2H bandwidth ~2x.
    std::memcpy(ctx.h_in_pinned, h_in, ctx.bytes);
    CUDA_CHECK(cudaMemcpy(ctx.d_in, ctx.h_in_pinned, ctx.bytes, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    int radius = (filter == F_GAUSSIAN) ? 2 : 1;
    int tileW = block.x + 2 * radius, tileH = block.y + 2 * radius;
    size_t shmem = (size_t)tileW * tileH;

    if (backend == B_GPU_NAIVE) {
        if (filter == F_SOBEL)         sobelNaive << <grid, block >> > (ctx.d_in, ctx.d_out, W, H);
        else if (filter == F_GAUSSIAN) gaussianNaive << <grid, block >> > (ctx.d_in, ctx.d_out, W, H);
        else                           medianNaive << <grid, block >> > (ctx.d_in, ctx.d_out, W, H);
    }
    else { // tiled
        if (filter == F_SOBEL)         sobelTiled << <grid, block, shmem >> > (ctx.d_in, ctx.d_out, W, H);
        else if (filter == F_GAUSSIAN) gaussianTiled << <grid, block, shmem >> > (ctx.d_in, ctx.d_out, W, H);
        else                           medianTiled << <grid, block, shmem >> > (ctx.d_in, ctx.d_out, W, H);
    }
    CUDA_CHECK(cudaMemcpy(ctx.h_out_pinned, ctx.d_out, ctx.bytes, cudaMemcpyDeviceToHost));
    std::memcpy(h_out, ctx.h_out_pinned, ctx.bytes);
}

// Stream-pipelined: copies into pinned, runs strips across N streams.
static void runGpuStream(GpuCtx& ctx, FilterMode filter,
    const unsigned char* h_in, unsigned char* h_out) {
    int W = ctx.W, H = ctx.H;
    std::memcpy(ctx.h_in_pinned, h_in, ctx.bytes);
    int nStreams = (int)ctx.streams.size();
    int nTiles = 8;
    int radius = (filter == F_GAUSSIAN) ? 2 : 1;
    int rowsPerStrip = (H + nTiles - 1) / nTiles;
    dim3 block(16, 16);
    int tileW = block.x + 2 * radius, tileH = block.y + 2 * radius;
    size_t shmem = (size_t)tileW * tileH;

    for (int strip = 0; strip < nTiles; ++strip) {
        cudaStream_t st = ctx.streams[strip % nStreams];
        int rowStart = strip * rowsPerStrip;
        int rowEnd = std::min(rowStart + rowsPerStrip, H);
        if (rowStart >= rowEnd) continue;
        int srcStart = std::max(rowStart - radius, 0);
        int srcEnd = std::min(rowEnd + radius, H);
        int stripH = srcEnd - srcStart;
        size_t stripBytes = (size_t)W * stripH;
        unsigned char* dIn = ctx.d_inS[strip % nStreams];
        unsigned char* dOut = ctx.d_outS[strip % nStreams];

        CUDA_CHECK(cudaMemcpyAsync(dIn, ctx.h_in_pinned + (size_t)srcStart * W,
            stripBytes, cudaMemcpyHostToDevice, st));
        dim3 grid((W + 15) / 16, (stripH + 15) / 16);
        if (filter == F_SOBEL)         sobelTiled << <grid, block, shmem, st >> > (dIn, dOut, W, stripH);
        else if (filter == F_GAUSSIAN) gaussianTiled << <grid, block, shmem, st >> > (dIn, dOut, W, stripH);
        else                           medianTiled << <grid, block, shmem, st >> > (dIn, dOut, W, stripH);

        int coreOff = rowStart - srcStart;
        int coreRows = rowEnd - rowStart;
        CUDA_CHECK(cudaMemcpyAsync(ctx.h_out_pinned + (size_t)rowStart * W,
            dOut + (size_t)coreOff * W, (size_t)W * coreRows,
            cudaMemcpyDeviceToHost, st));
    }
    for (auto st : ctx.streams) CUDA_CHECK(cudaStreamSynchronize(st));
    std::memcpy(h_out, ctx.h_out_pinned, ctx.bytes);
}

// ============================================================================
// MAIN LOOP: webcam/video -> filter -> overlay -> display
// ============================================================================
int main(int argc, char** argv) {
    initGaussianWeights();

    cv::VideoCapture cap;
    if (argc >= 2) {
        cap.open(argv[1]);
        printf("Opened video file: %s\n", argv[1]);
    }
    else {
        cap.open(0);                                       // default webcam
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        printf("Opened default webcam at 1280x720\n");
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "Cannot open video source\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Compute %d.%d\n",
        prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    GpuCtx ctx;
    Backend backend = B_GPU_TILED;
    FilterMode filter = F_SOBEL;

    // Rolling FPS window (last 30 frames).
    std::deque<double> frameMs;
    auto pushMs = [&](double ms) {
        frameMs.push_back(ms);
        if (frameMs.size() > 30) frameMs.pop_front();
        };
    auto avgMs = [&]() {
        if (frameMs.empty()) return 0.0;
        return std::accumulate(frameMs.begin(), frameMs.end(), 0.0) / frameMs.size();
        };

    // 'side' artik dongu disinda; create() ayni boyutta no-op olur, realloc yok.
    cv::Mat frame, gray, out, side;
    const std::string winName = "CENG479 Live Demo - press C/G/T/S, 1/2/3, ESC";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            // Loop a video file rather than ending the demo.
            if (argc >= 2) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); continue; }
            break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        ctx.allocate(gray.cols, gray.rows);
        out.create(gray.size(), CV_8UC1);

        // ---- Timed processing ----
        auto t0 = std::chrono::high_resolution_clock::now();
        if (backend == B_CPU) {
            if (filter == F_SOBEL)    cpuSobel(gray.data, out.data, gray.cols, gray.rows);
            else if (filter == F_GAUSSIAN) cpuGaussian(gray.data, out.data, gray.cols, gray.rows);
            else                       cpuMedian(gray.data, out.data, gray.cols, gray.rows);
        }
        else if (backend == B_GPU_STREAM) {
            runGpuStream(ctx, filter, gray.data, out.data);
        }
        else {
            runGpuSimple(ctx, backend, filter, gray.data, out.data);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        pushMs(ms);

        // ---- Overlay info ----
        cv::Mat disp;
        cv::cvtColor(out, disp, cv::COLOR_GRAY2BGR);

        // Side-by-side: original on left, filtered on right.
        side.create(disp.rows, disp.cols * 2, CV_8UC3);
        frame.copyTo(side(cv::Rect(0, 0, disp.cols, disp.rows)));
        disp.copyTo(side(cv::Rect(disp.cols, 0, disp.cols, disp.rows)));

        double avg = avgMs();
        double fps = avg > 0 ? 1000.0 / avg : 0;

        // Background strip behind text so it's always readable.
        cv::rectangle(side, cv::Rect(0, 0, side.cols, 110),
            cv::Scalar(0, 0, 0), cv::FILLED);
        cv::rectangle(side, cv::Rect(0, 0, side.cols, 110),
            cv::Scalar(40, 40, 40), 1);

        char line1[256], line2[256], line3[256];
        std::snprintf(line1, sizeof(line1), "%s  |  %s",
            backendName(backend), filterModeName(filter));
        std::snprintf(line2, sizeof(line2), "Per-frame: %.2f ms   FPS: %.1f",
            avg, fps);
        std::snprintf(line3, sizeof(line3), "%d x %d  ( %.2f MPix/frame )",
            gray.cols, gray.rows, gray.cols * gray.rows / 1.0e6);

        cv::putText(side, line1, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX,
            0.9, cv::Scalar(0, 255, 0), 2);
        cv::putText(side, line2, cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(side, line3, cv::Point(20, 92), cv::FONT_HERSHEY_SIMPLEX,
            0.5, cv::Scalar(180, 180, 180), 1);

        // Labels above each pane.
        cv::putText(side, "INPUT", cv::Point(20, side.rows - 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(side, "FILTERED", cv::Point(disp.cols + 20, side.rows - 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        cv::imshow(winName, side);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break;
        else if (key == 'c' || key == 'C') { backend = B_CPU;        frameMs.clear(); }
        else if (key == 'g' || key == 'G') { backend = B_GPU_NAIVE;  frameMs.clear(); }
        else if (key == 't' || key == 'T') { backend = B_GPU_TILED;  frameMs.clear(); }
        else if (key == 's' || key == 'S') { backend = B_GPU_STREAM; frameMs.clear(); }
        else if (key == '1') { filter = F_SOBEL;      frameMs.clear(); }
        else if (key == '2') { filter = F_GAUSSIAN;   frameMs.clear(); }
        else if (key == '3') { filter = F_MEDIAN;     frameMs.clear(); }
    }
    return 0;
}