# Predictive Maintenance Neural Inference Engine

An end-to-end predictive maintenance system combining a custom Python-based neural network training pipeline with a high-performance, multi-threaded C inference engine.

This project tackles anomaly detection in turbofan engines using the NASA CMAPSS dataset. It predicts anomalies based on Remaining Useful Life (RUL) using a sliding window technique.

> **Note:** The neural network training pipeline is built on top of a custom neural network framework developed previously in a separate project:
 [BaranOnal/neural-network-from-scratch](https://github.com/BaranOnal/neural-network-from-scratch)

## Core Features

- **Zero High-Level ML Dependencies:** The C inference engine is a pure native implementation with no external machine learning libraries.
- **Vectorized Preprocessing:** The Python pipeline utilizes NumPy and Pandas for sensor normalization and sliding window generation.
- **Hardware-Level Optimization:** The C engine reconstructs the network utilizing flattened, contiguous memory arrays and Row-Major Order memory access to maximize Cache Hit rates during dot product operations.
- **Windows Native Multithreading:** Parallel inference is achieved using Windows API thread management (CreateThread, CRITICAL_SECTION), ensuring lock-free forward-pass execution.
- **Thread-Safe Synchronization:** Uses atomic operations (InterlockedIncrement) for accurate throughput counting and anomaly tracking without context-switching bottlenecks.

## Dataset and Preprocessing

This project uses the NASA CMAPSS turbofan engine degradation simulation dataset.

- **Target Definition:** Engines with a Remaining Useful Life (RUL) <= 30 cycles are labeled as anomalies (1), others as normal (0).
- **Selected Sensors:** s2, s3, s4, and s7.
- **Sliding Window:** A 50-cycle sliding window is applied to the 4 sensors, generating 200 features per inference step. Boundary leaks between different engine units are strictly prevented at the C-level.

## Neural Network Architecture

```text
Input (200 Features)
   ↓
Dense Layer (128) + ReLU
   ↓
Dense Layer (64) + ReLU
   ↓
Dense Layer (1) + Sigmoid
```

## Build and Run (MSVC)

**Step 1 — Train the model and export weights:**
```cmd
python train_sliding_window.py
```
 
**Step 2 — Compile with optimizations:**
```cmd
cl /O2 /arch:AVX2 /nologo /Fe:inference_engine.exe inference_engine.c cJSON.c
```
 
> `/O2` enables compiler optimizations. `/arch:AVX2` enables SIMD instructions, allowing the CPU to process 8 floats per cycle — matching NumPy-level throughput.
 
**Step 3 — Run benchmark:**
```cmd
.\inference_engine.exe weights.json archive\train_FD001.txt
```

## Benchmark Results
### C vs Python (single sample inference)
 
| Mode | Throughput | Latency |
|------|-----------|---------|
| C — single thread (`/Zi` debug) | 16,856 packets/s | 0.0593 ms |
| Python — NumPy (single sample) | 79,566 packets/s | 0.0126 ms |
| C — single thread (`/O2 AVX2`) | 86,181 packets/s | 0.0116 ms |
| C — 4 threads (`/O2 AVX2`) | **294,871 packets/s** | **0.0034 ms** |
 
**Key insight:** Without compiler optimizations, the naive C loop is ~5x slower than NumPy (which uses BLAS/SIMD internally). With `/O2 /arch:AVX2`, the C engine surpasses NumPy on a single thread and reaches **3.7x NumPy throughput** with 4 threads.
 
### Thread scaling
 
```
>>> [1] Single Thread
  Anomaly    : 2193 / 15731  (13.9%)
  Time       : 0. 825 s
  Throughput : 86,181 packets/s
  Avg. latency: 0.0116 ms/packet
 
>>> [2] Multi Thread (4 threads)
  Anomaly    : 2193 / 15731  (13.9%)
  Time       : 0.0533 s
  Throughput : 294,871 packets/s
  Avg. latency: 0.0034 ms/packet
 
====================================
        BENCHMARK REPORT
====================================
  Single thread time : 0.1825 s
  Multi thread time  : 0.0533 s
  Speedup            : 3.42x
====================================
```

---

## Project Structure

```text
project/
│
├── train_sliding_window.py
├── inference_engine.c
├── cJSON.c
├── cJSON.h
├── weights.json
├── archive/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
└── nn/
    ├── layers.py
    ├── activations.py  
    ├── losses.py  
    ├── optimizers.py     
    └── utils.py 
```

