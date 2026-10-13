# Optimizing E = AᵀB + CDᵀ in CUDA

A step-by-step optimization of a fused "transpose-multiply-add" kernel, from a naive one-thread-per-element version to a shared-memory tiled kernel with fully coalesced global loads and no shared-memory bank conflicts.

Built as part of CS6023 (GPU Programming), IIT Madras.

| Version | Kernel time (input6) | Speedup |
|---|---|---|
| Naive (`src/naive.cu`) | 18.9 ms | 1.0× |
| Tiled + coalesced + padded (`src/tiled.cu`) | 10.4 ms | **1.8×** |

Measured on an NVIDIA T4 (Google Colab), input6: p = 1024, q = 1000, r = 997. Both pass all 6 public test cases.

---

## 1. The problem

Given integer matrices **A (q×p)**, **B (q×r)**, **C (p×q)** and **D (r×q)**, compute

```
E = Aᵀ·B + C·Dᵀ        (E is p×r)
```

All matrices are stored **row-major as 1D arrays**. The transposes Aᵀ and Dᵀ are never stored; they exist only through indexing. That is what makes this problem interesting: transposed access patterns are bad for memory coalescing.

### The formula without transposes

Using `Aᵀ[i][k] = A[k][i]` and `Dᵀ[k][j] = D[j][k]`:

```
E[i][j] = Σk A[k][i]·B[k][j]  +  Σk C[i][k]·D[j][k]
```

In 1D indexing:

```
A[k*p + i]    B[k*r + j]    C[i*q + k]    D[j*q + k]
```

Both sums run over the same `k`, so they can share **one loop** and **one kernel**.

---

## 2. A simple test for any memory access

Write the 1D address and ask two questions:

1. **When `threadIdx.x` goes up by 1** (neighboring threads in a warp), how does the address change?
   - `+1` → coalesced ✅
   - `0` → broadcast (all threads read the same value) ✅
   - big jump (`p`, `q`, …) → not coalesced ❌
2. **When `k` goes up by 1**, how does the address change?
   - `+1` → next value is in the same cache line, cache-friendly ✅
   - big jump → new cache line every iteration

For **shared memory**, the question is different: do the 32 threads of a warp hit **32 different banks**? If not, it's a **bank conflict**.

---

## 3. Step 1: Naive kernel

One thread computes one element `E[i][j]`. `threadIdx.y` → row `i`, `threadIdx.x` → column `j`.

```cpp
int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;
if (i < p && j < r) {
    int sum = 0;
    for (int k = 0; k < q; k++) {
        sum += A[k * p + i] * B[k * r + j];
        sum += C[i * q + k] * D[j * q + k];
    }
    E[i * r + j] = sum;
}
```

### Analysis

Applying the test (`threadIdx.x` controls `j`):

| Matrix | Address | Within a warp | Along k |
|---|---|---|---|
| A | `k*p + i` | broadcast ✅ (no `j`) | jump of p |
| B | `k*r + j` | coalesced ✅ | jump of r |
| C | `i*q + k` | broadcast ✅ (no `j`) | +1, cache-friendly ✅ |
| D | `j*q + k` | **not coalesced ❌** (jump of q) | +1, cache-friendly |

Per warp, per iteration of `k` (worst case, no cache hits): about **1 fetch each for A, B, C, but 32 fetches for D**, since each thread's D value is in a different 128-byte line.

### Two problems

1. **D is not coalesced.** It dominates global memory traffic.
2. **Repeated fetching.** Every thread in row `i` of E needs the same A and C values; every thread in column `j` needs the same B and D values. Different warps and blocks fetch them again and again. Reuse depends on the cache, which is not guaranteed (lines get evicted).

---

## 4. Step 2: Shared-memory tiling

Each **16×16 block** computes a **16×16 tile of E**. Instead of every thread fetching its own data, the block cooperatively loads small tiles into shared memory and reuses them.

The `k` loop is split into chunks of 16. Each iteration loads **four 16×16 tiles**:

| Tile | Holds | Rows | Columns |
|---|---|---|---|
| `As` | piece of Aᵀ | E's rows | k |
| `Bs` | piece of B | k | E's columns |
| `Cs` | piece of C | E's rows | k |
| `Ds` | piece of Dᵀ | k | E's columns |

Each of the 256 threads loads **one element into each tile**, then:

```cpp
__syncthreads();                         // 1. tiles fully loaded
for (int kk = 0; kk < TILE; kk++)
    sum += As[ty][kk] * Bs[kk][tx] + Cs[ty][kk] * Ds[kk][tx];
__syncthreads();                         // 2. done reading before overwrite
```

Why two syncs:
1. **After loading**: no thread reads a tile before all 256 threads have written it.
2. **After computing**: no fast thread overwrites a tile (next iteration) while a slow thread is still reading it.

Out-of-range elements are loaded as `0`, so they add nothing and the compute loop needs no bounds checks. The final write to E happens **once, after** the loop.

**Result:** each global value is loaded once per block and reused 16 times from shared memory, about **16× fewer global loads**.

### The first (direct) version of the loads

```cpp
As[ty][tx] = A[(k + tx) * p + row];    // Aᵀ[row][k+tx]
Bs[ty][tx] = B[(k + ty) * r + col];    // B[k+ty][col]
Cs[ty][tx] = C[row * q + (k + tx)];    // C[row][k+tx]
Ds[ty][tx] = D[col * q + (k + ty)];    // Dᵀ[k+ty][col]
```

(`row = blockRow + ty`, `col = blockCol + tx`.)

This is correct, but re-applying the test shows:

| Load | `tx` + 1 → address | |
|---|---|---|
| As | +p | ❌ |
| Bs | +1 | ✅ |
| Cs | +1 | ✅ |
| Ds | +q | ❌ |

The two **transposed** matrices are uncoalesced. Note that A was a broadcast in the naive kernel but became a problem here, because in a tiled load each thread loads a *different* element. Every load has to be re-checked.

---

## 5. Step 3: Coalescing the transposed loads

**Key idea:** which 256 elements a block loads is fixed; we only choose *which thread loads which element*. So:

- **Read** global memory in the order memory likes: neighboring `tx` → neighboring addresses.
- **Write** into shared memory at the **transposed** position. Shared memory is fast, so the transpose costs almost nothing there.

In practice: unfold `row`/`col`, then swap `tx` and `ty` everywhere in the line.

```cpp
// before: As[ty][tx] = A[(k + tx) * p + (blockRow + ty)];
As[tx][ty] = A[(k + ty) * p + (blockRow + tx)];   // tx walks along a row of A

// before: Ds[ty][tx] = D[(blockCol + tx) * q + (k + ty)];
Ds[tx][ty] = D[(blockCol + ty) * q + (k + tx)];   // tx walks along a row of D
```

Every spot of the tile receives exactly the same value as before (e.g. `As[3][5]` still holds `A[5][3]`); only the thread doing the copy changed. Now neighboring threads read `A[5][3], A[5][4], …`, which are adjacent in memory.

**Bounds checks** follow one rule: find the element's row and column in the original matrix and compare with its size. For A (q×p): `k + ty < q` and `blockRow + tx < p`. For D (r×q): `blockCol + ty < r` and `k + tx < q`.

**Now all four global loads are coalesced.** ✅

---

## 6. Step 4: Removing shared-memory bank conflicts

### Banks

Shared memory is split into **32 banks**. Element `n` (4-byte int) lives in bank `n % 32`. A warp's access takes **as many cycles as its busiest bank**. Different elements in the same bank must be served one after another (a *bank conflict*); the *same* element is a free broadcast.

### Where the conflict is

The transposed writes `As[tx][ty]` and `Ds[tx][ty]`. With a 16-wide tile, thread `(ty, tx)` writes element `tx*16 + ty`, so neighboring threads are **16 elements apart**, and every second thread wraps around to the same bank:

```
ty = 0:  tx = 0, 2, 4, …, 14  →  elements 0, 32, 64, …, 224  →  all bank 0
         tx = 1, 3, 5, …, 15  →  elements 16, 48, …, 240     →  all bank 16
```

A warp (2 rows of 16 threads in a 16×16 block) uses only **4 banks, 8 threads each → 8 cycles** instead of 1.

The reads in the compute loop are fine: `As[ty][kk]` / `Cs[ty][kk]` are broadcasts, and `Bs[kk][tx]` / `Ds[kk][tx]` are side by side.

### Fix: padding

Make each row of `As` and `Ds` longer so that consecutive rows start on different banks:

| Row width | Thread writes element | Cycles per warp |
|---|---|---|
| 16 | `tx*16 + ty` | 8 |
| 17 (textbook +1) | `tx*17 + ty` | 2 |
| **18 (+2)** | `tx*18 + ty` | **1** |

The usual "+1" fix is perfect for 32-wide tiles, where a warp covers one row. With 16-wide tiles a warp covers **two rows** (`ty` and `ty+1`); +1 still leaves one pair colliding (`ty=0,tx=0` and `ty=1,tx=15` both hit bank 0). With +2, the `ty = 0` threads land on all even banks and the `ty = 1` threads on all odd banks, so all 32 are distinct.

```cpp
__shared__ int As[TILE][TILE + 2];
__shared__ int Ds[TILE][TILE + 2];
```

`Bs` and `Cs` are written normally, so they need no padding.

---

## 7. Results

| | Naive | Final tiled |
|---|---|---|
| Kernel time (input6) | 18.9 ms | 10.4 ms |
| Total `compute()` time | ~308 ms | ~299 ms |
| Kernel launches | 1 | 1 |
| Tests passed | 6/6 | 6/6 |

### Why the total time barely changes

The provided timer measures all of `compute()`: CUDA context startup, `cudaMalloc`, and host↔device copies. These take ~290 ms, so the kernel is only ~3% of the total. That is why even the 2×2 test case reports ~300 ms. To compare kernels, I timed the launch alone with CUDA events (removed from the submitted code, since prints are not allowed).

### Why 1.8× and not 16×

The naive kernel was not the worst case: A and C were broadcasts, B was coalesced, and the L1/L2 caches caught much of the reuse. Tiling removed the remaining repeated global traffic and fixed D. The tiled kernel is now limited by **shared-memory reads**: each step of the inner loop reads 4 values from shared memory to do 2 multiply-adds.

---

## 8. What could come next

- **Register tiling / thread coarsening**: each thread computes several outputs (e.g. 2×2 or 4×1) and keeps partial sums in registers, reusing each shared-memory value several times.
- **Tile size experiments**: 32×32 tiles (with +1 padding) increase reuse per load but use 1024 threads per block.
- **Profiling** with Nsight Compute to measure achieved memory bandwidth and confirm zero bank conflicts.

---

## Repository layout

```
src/naive.cu   naive kernel (one thread per E element)
src/tiled.cu   final kernel: tiling + coalesced loads + padding
```

Both files keep the course-provided `main()` unchanged; only the kernel and its launch inside `compute()` differ.

## How to run

Requires an NVIDIA GPU and the CUDA Toolkit. On Google Colab, pick a T4 GPU runtime.

```bash
nvcc -O3 -arch=sm_75 src/tiled.cu -o tiled
./tiled input.txt output.txt
```

**Input format:** the first line is `p q r`, followed by A (q lines of p ints), B (q lines of r ints), C (p lines of q ints), and D (r lines of q ints). The program writes E (p lines of r ints) to the output file and prints the time taken.
