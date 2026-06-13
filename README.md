# CUDA Image Processing

**GPU-Accelerated Image Processing: Parallel Implementation of Gaussian Blur, Sobel Edge Detection, and Median Filtering Using CUDA**

A CENG-479 *Parallel Programming* final project at Gazi University, Department of Computer Engineering. This repository contains a sequential C++ baseline and four progressively-optimized CUDA variants of three classic image-processing filters, plus a real-time webcam demo that lets you toggle between CPU and GPU backends at runtime.

| Filter | Sequential (CPU) | Best GPU Kernel | Kernel Speedup | End-to-End Speedup |
|---|---|---|---|---|
| Gaussian Blur (K=5) | 168.4 ms | 0.281 ms (separable) | **598×** | 136× |
| Sobel Edge Detection (K=3) | 70.8 ms | 0.107 ms (naive) | **665×** | 67× |
| Median Filter (K=3) | 137.9 ms | 1.119 ms (tiled) | **123×** | 65× |

*Measured on a 2048×2048 image with a 16×16 CUDA block on an NVIDIA RTX 4060 Laptop GPU.*

---

## Authors

- **Ömer Faruk İlhan** — 21118080023
- **Eren Diken** — 21118080077

Instructor: *Asst. Prof. Dr. Hüseyin Temuçin*

---

## Repository Layout

```
CUDAImageProcessing/
├── CudaImageProcessing/   # Benchmark suite (sequential baseline + 4 CUDA variants)
├── LiveDemo/              # Real-time webcam demo with runtime backend switching
└── .gitignore
```

- **`CudaImageProcessing/`** — Sweeps every filter, variant, and block size on a synthetic 2048×2048 test image and prints a table of CPU time, kernel time, end-to-end time, throughput (MPix/s, GFLOPS), arithmetic intensity, and a PASS/FAIL validation flag against the CPU baseline.
- **`LiveDemo/`** — Reads a 1280×720 webcam stream and runs the Sobel filter through the selected backend with an on-screen FPS counter.

---

## Implementation Overview

### Sequential Baseline (CPU)

Single-threaded C++ compiled with `/O2`. Deliberately avoids SIMD intrinsics and multithreading so that the speedup numbers reflect a genuine single-threaded reference. Used both as the denominator for every speedup figure and as the ground-truth output for pixel-by-pixel correctness validation.

### Parallel Variants (CUDA)

Each variant maps **one CUDA thread to one output pixel**, exploiting the embarrassingly-parallel nature of pixel-wise convolution.

| # | Variant | Key Idea |
|---|---|---|
| 1 | **Naive** | Direct global-memory reads of the K×K neighborhood. Filter weights live in constant memory. |
| 2 | **Tiled (Shared Memory)** | Block cooperatively loads an image tile plus a radius-wide halo into shared memory, then reads from shared memory. Amortizes global-memory traffic across the block. |
| 3 | **Separable Gaussian** | Decomposes the 2-D Gaussian into a horizontal pass followed by a vertical pass, reducing per-pixel arithmetic from O(K²) to O(2K). Uses a float intermediate buffer to avoid double quantization. |
| 4 | **Stream-Pipelined** | Splits the image into horizontal strips processed on four CUDA streams with pinned host memory. Overlaps H2D transfer, kernel execution, and D2H transfer to hide PCIe latency. |

All four variants were validated pixel-by-pixel against the CPU baseline (PASS on every configuration reported).

---

## Requirements

### Hardware
- NVIDIA GPU with **Compute Capability 8.9** (RTX 4060/4070/4080/4090, Ada Lovelace). Other architectures work after adjusting the SM target — see below.
- A webcam (built-in or USB) for the LiveDemo.

### Software
- **Windows 10/11** with Visual Studio 2022 (the `.sln` files target MSVC).
- **CUDA Toolkit 12.x** (developed with 13.2; any 12.x ≥ 12.0 should work).
- **NVIDIA Driver** recent enough for the installed toolkit.
- **OpenCV 4.x** with the world DLL on `PATH` (for the LiveDemo webcam capture and display only).

### Recommended Build Configuration

In Visual Studio, set CUDA C/C++ → Device → **Code Generation = `compute_89,sm_89`**. This emits a native cubin for Ada and removes the PTX-JIT step (which fixed an "unsupported toolchain" launch error during development). For other GPUs:

| GPU Family | SM Target |
|---|---|
| Ada (RTX 40xx) | `compute_89,sm_89` |
| Ampere (RTX 30xx) | `compute_86,sm_86` |
| Turing (RTX 20xx, GTX 16xx) | `compute_75,sm_75` |
| Pascal (GTX 10xx) | `compute_61,sm_61` |

---

## How to Run — Step by Step

> ⚠️ **Important — build order matters.** `LiveDemo` depends on the CUDA kernels and helper code produced by the `CudaImageProcessing` project. You **must build `CudaImageProcessing` first**; otherwise `LiveDemo` will fail to link or load. Run the projects in the order below.

### Step 1 — Clone the repository

```powershell
git clone https://github.com/omerilhNN/CUDAImageProcessing.git
cd CUDAImageProcessing
```

### Step 2 — Prepare the environment (one-time setup)

1. Install **Visual Studio 2022** with the *Desktop development with C++* workload.
2. Install the **CUDA Toolkit 12.x** (the VS integration must be selected during install).
3. Install **OpenCV 4.x** and add its `build/x64/vc16/bin` directory to your system `PATH`. This DLL is required by `LiveDemo` for webcam capture.
4. Verify the install from a terminal:
   ```powershell
   nvcc --version          # should print CUDA 12.x
   nvidia-smi              # should list your GPU
   ```

### Step 3 — Build & run the benchmark suite (`CudaImageProcessing`) — **do this first**

This is the main project: sequential baseline + four CUDA variants. Building it also produces the CUDA object files that `LiveDemo` will link against.

1. Open the solution:
   ```powershell
   start CudaImageProcessing\CudaImageProcessing.sln
   ```
2. In the toolbar, select **Release | x64**.
3. Right-click the `CudaImageProcessing` project → **Properties → CUDA C/C++ → Device → Code Generation** and set it to `compute_89,sm_89` (or the value matching your GPU from the table above).
4. **Build → Build Solution** (`Ctrl + Shift + B`). Wait for `Build: 1 succeeded`.
5. Run with `Ctrl + F5`. A console window opens and the benchmark sweeps every filter, variant, and block size. You should see a table like the one in [Running the Benchmark Suite](#running-the-benchmark-suite) below, with `PASS` on every row. Filter output images are written next to the executable in `CudaImageProcessing\x64\Release\`.
6. Confirm the build artifacts exist:
   ```powershell
   dir CudaImageProcessing\x64\Release\*.exe
   ```
   If `CudaImageProcessing.exe` is there, you're ready for Step 4.

### Step 4 — Build & run the LiveDemo (`LiveDemo`) — **only after Step 3 succeeds**

`LiveDemo` reuses the CUDA filter implementations from Step 3 and adds a webcam capture loop on top.

1. Open the solution:
   ```powershell
   start LiveDemo\LiveDemo.sln
   ```
2. Select **Release | x64** (must match Step 3 — mixing Debug/Release will cause link errors).
3. Make sure the SM target matches what you used in Step 3.
4. Plug in your webcam (or ensure the built-in camera is enabled in Windows privacy settings).
5. **Build → Build Solution** (`Ctrl + Shift + B`).
6. Run with `Ctrl + F5`. The webcam window opens automatically with an on-screen FPS counter.
7. Press `1`/`2`/`3`/`4` to switch backends in real time — see the next section for the key map.

> 💡 If `LiveDemo` reports a missing `opencv_world4xx.dll` at launch, either fix your `PATH` (Step 2.3) or copy the DLL from your OpenCV install into `LiveDemo\x64\Release\` next to `LiveDemo.exe`.

---

## Running the LiveDemo

The LiveDemo is the simplest way to *see* the speedup. It captures from your webcam at 1280×720, runs the Sobel edge filter through the selected backend, and overlays a live FPS counter on the output window.

### Switching backends at runtime

While the demo window is focused, press a key to switch backends. The FPS counter updates immediately so the speedup is visible without restarting:

| Key | Backend | Typical Performance (Sobel, 1280×720, RTX 4060 Laptop) |
|---|---|---|
| `1` | CPU (1 thread)        | ~42 FPS &nbsp;&nbsp;(23.5 ms/frame) — baseline |
| `2` | GPU Naive             | ~1184 FPS &nbsp;(0.84 ms/frame) — ~28× speedup |
| `3` | GPU Tiled (shared memory) | ~1247 FPS &nbsp;(0.80 ms/frame) — ~29× speedup |
| `4` | GPU Stream-Pipelined  | ~695 FPS &nbsp;&nbsp;(1.44 ms/frame) — ~16× speedup |
| `q` / `Esc` | Quit |  |

> The real-time speedups are lower than the 2048×2048 / 4096×4096 benchmark numbers because the 1280×720 frame does not fully saturate the GPU and per-frame PCIe transfer overhead is relatively larger — exactly the end-to-end vs. kernel-only gap that Amdahl's Law predicts.

### Verifying that the GPU is actually doing the work

Open **Task Manager → Performance → GPU**. Switching to a GPU backend should immediately spike the **3D / Compute** utilization on the discrete NVIDIA GPU while the CPU stays near idle. Switching back to CPU mode flips the picture.

---

## Running the Benchmark Suite

Launch the executable from `CudaImageProcessing/`. With no arguments it runs the full sweep (every filter × every variant × block sizes 8×8 / 16×16 / 32×32) on a 2048×2048 synthetic test image containing a smooth gradient, a checkerboard, a circle, and salt-and-pepper noise. Sample output:

```
filter   variant     block   CPU ms  ker ms  tot ms   kSpd    e2e   MPix/s   GF/s    AI    valid
Gaussian naive       16x16   168.4   0.368   1.370    457x    123x  11407    569    12.5  PASS
Gaussian tiled       16x16   168.4   0.293   1.239    575x    136x  14323    717    35.8  PASS
Gaussian separable   16x16   168.4   0.281   1.381    598x    122x  14935    298     8.2  PASS
Gaussian stream      16x16   168.4   1.426   1.426    118x    118x   2942    147     8.2  PASS
Sobel    naive       16x16    70.8   0.107   1.058    665x     67x  39228   1575     2.1  PASS
Sobel    tiled       16x16    70.8   0.136   1.106    520x     64x  30857   1233     5.9  PASS
Sobel    stream      16x16    70.8   1.308   1.308     54x     54x   3206    128     2.1  PASS
Median   naive       16x16   137.9   1.181   2.119    117x     65x   3553     72    47.0  PASS
Median   tiled       16x16   137.9   1.119   2.260    123x     61x   3751     76    47.0  PASS
Median   stream      16x16   137.9   2.617   2.617     53x     53x   1604     33    47.0  PASS
```

- **`CPU`** — sequential baseline time (denominator).
- **`ker`** — pure GPU kernel time.
- **`tot`** — end-to-end GPU time (H2D + kernel + D2H).
- **`kSpd` / `e2e`** — kernel-only and end-to-end speedups vs. CPU.
- **`AI`** — arithmetic intensity (FLOP/byte), used in the roofline analysis.
- **`valid`** — PASS if the GPU output matches the CPU reference pixel-by-pixel within tolerance.

Timing methodology: a warm-up launch primes caches/JIT before timing; each configuration is averaged over 10 iterations; transfers and kernel are measured separately with `cudaEvent_t`.

---

## Key Findings

1. **Shared-memory tiling delivers the highest kernel speedups** — up to 575× for Gaussian and 665× for Sobel at 2048×2048, surpassing 760× at 4K.
2. **Separable Gaussian wins for K ≥ 7.** The curves cross near K=7; at K=15 the tiled 2-D kernel runs ~4× slower than the separable two-pass version.
3. **Stream pipelining wins on end-to-end time for large images.** It hides H2D/D2H transfers behind compute, closing the kernel-vs-end-to-end gap that Amdahl's Law exposes on the other variants.
4. **16×16 (256 threads) is the sweet spot.** 32×32 blocks (1024 threads) limit occupancy to two blocks per SM and consistently lose to 16×16.
5. **Naive kernels are memory-bound; tiled kernels become compute-bound.** The roofline plot shows tiled points crossing the ridge at AI ≈ 45.
6. **Tiling does not always help.** For 3×3 Sobel, the cooperative-load + `__syncthreads()` overhead exceeds the savings from nine reuses, so the naive kernel is actually slightly faster than the tiled one. Tiling pays off for K ≥ 5.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unsupported toolchain` or kernel launch failure | Match the SM target to your GPU (see table above) so `nvcc` emits a native cubin and skips PTX-JIT. Alternatively, update the NVIDIA driver. |
| LiveDemo crashes on startup with a missing DLL | Add the OpenCV `bin/` directory to `PATH` or copy `opencv_world4xx.dll` next to the executable. |
| FPS counter shows 0 / no webcam image | Another application is holding the camera, or the default camera index isn't 0. Close conflicting apps; if needed, edit the `cv::VideoCapture(0)` index in the demo source. |
| Validation reports `FAIL` | Compare on a host with the same compiler flags and SM target; tiny floating-point divergence can cross the tolerance threshold. The separable Gaussian uses a tolerance of 2 to absorb its single extra round-off pass. |

---

## References

1. Amdahl, G. M. (1967). *Validity of the single processor approach to achieving large scale computing capabilities.* AFIPS '67.
2. Canny, J. (1986). *A computational approach to edge detection.* IEEE TPAMI 8(6).
3. Daga, M., Aji, A. M., & Feng, W. C. (2011). *On the efficacy of a fused CPU+GPU processor for parallel computing.* SASP 2011.
4. Fialka, O., & Čadík, M. (2006). *FFT and convolution performance in image filtering on GPU.* IV '06.
5. Jang, B., Schaa, D., Mistry, P., & Kaeli, D. (2010). *Exploiting memory access patterns to improve memory performance in data-parallel architectures.* IEEE TPDS 22(1).
6. NVIDIA Corporation. (2024). *CUDA C++ Programming Guide.* https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

## License

Academic project, released for course evaluation. Feel free to read, fork, and learn from the code.
