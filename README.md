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

## Building

### Benchmark suite

```powershell
# From the repository root, open in Visual Studio
start CudaImageProcessing\CudaImageProcessing.sln
```

1. Select the **Release | x64** configuration.
2. Set the SM target as described above.
3. Build → Build Solution (`Ctrl + Shift + B`).
4. Run (`Ctrl + F5`). The benchmark prints a per-configuration table and writes filter output images next to the executable.

### LiveDemo

```powershell
start LiveDemo\LiveDemo.sln
```

1. Make sure the OpenCV `bin/` directory (containing `opencv_world4xx.dll`) is on your `PATH`, or copy the DLL next to the built `.exe`.
2. Build **Release | x64**.
3. Run — your default webcam opens automatically.

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



## License

Academic project, released for course evaluation. Feel free to read, fork, and learn from the code.
