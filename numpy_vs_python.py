"""
numpy_vs_python.py
==================
Side-by-side performance and memory comparison between pure-Python constructs
and their NumPy equivalents — illustrating **why** you should use NumPy for
numerical work.

Run:
    python numpy_vs_python.py
"""

import sys
import time

import numpy as np

SIZE = 1_000_000   # 1 million elements


def timer(label, func):
    """Run func(), print elapsed time, and return the result."""
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    print(f"  {label:<40s}: {elapsed * 1000:8.3f} ms")
    return result


# ---------------------------------------------------------------------------
# 1. Element-wise multiplication
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. ELEMENT-WISE MULTIPLICATION  (n = {:,})".format(SIZE))
print("=" * 60)

py_a  = list(range(SIZE))
py_b  = list(range(SIZE))
np_a  = np.arange(SIZE, dtype=np.int64)
np_b  = np.arange(SIZE, dtype=np.int64)

timer("Python list comprehension", lambda: [a * b for a, b in zip(py_a, py_b)])
timer("NumPy vectorized   (np_a * np_b)",  lambda: np_a * np_b)

# ---------------------------------------------------------------------------
# 2. Sum of elements
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. SUM OF ELEMENTS  (n = {:,})".format(SIZE))
print("=" * 60)

timer("Python built-in sum(list)", lambda: sum(py_a))
timer("NumPy   np.sum(array)",     lambda: np.sum(np_a))

# ---------------------------------------------------------------------------
# 3. Dot product
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DOT PRODUCT  (n = {:,})".format(SIZE))
print("=" * 60)

py_x  = [float(i) for i in range(SIZE)]
np_x  = np.arange(SIZE, dtype=np.float64)

timer("Python loop (sum a*b)",       lambda: sum(a * b for a, b in zip(py_x, py_x)))
timer("NumPy   np.dot(arr, arr)",    lambda: np.dot(np_x, np_x))

# ---------------------------------------------------------------------------
# 4. Boolean filtering
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. BOOLEAN FILTERING  (keep even numbers, n = {:,})".format(SIZE))
print("=" * 60)

timer("Python list comprehension", lambda: [x for x in py_a if x % 2 == 0])
timer("NumPy boolean mask",        lambda: np_a[np_a % 2 == 0])

# ---------------------------------------------------------------------------
# 5. Matrix multiplication  (1000 x 1000)
# ---------------------------------------------------------------------------
MSIZE = 1_000
print("\n" + "=" * 60)
print("5. MATRIX MULTIPLICATION  ({0}x{0})".format(MSIZE))
print("=" * 60)

# Python nested lists
py_mat = [[float(i * MSIZE + j) for j in range(MSIZE)] for i in range(MSIZE)]

def py_matmul(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

np_mat = np.arange(MSIZE * MSIZE, dtype=np.float64).reshape(MSIZE, MSIZE)

# Python nested-loop matmul is extremely slow for 1000x1000 — skip it and
# demonstrate with a smaller size so the comparison is still meaningful.
SMALL = 100
py_s  = [[float(i * SMALL + j) for j in range(SMALL)] for i in range(SMALL)]
np_s  = np.arange(SMALL * SMALL, dtype=np.float64).reshape(SMALL, SMALL)

print(f"  (using {SMALL}x{SMALL} for Python loop to keep runtime reasonable)")
timer(f"Python {SMALL}x{SMALL} triple loop",    lambda: py_matmul(py_s, py_s))
timer(f"NumPy  {SMALL}x{SMALL} np.matmul/@",    lambda: np_s @ np_s)
print(f"  NumPy {MSIZE}x{MSIZE} matmul:")
timer(f"NumPy  {MSIZE}x{MSIZE} np.matmul/@",    lambda: np_mat @ np_mat)

# ---------------------------------------------------------------------------
# 6. Memory usage comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. MEMORY USAGE  (n = {:,})".format(SIZE))
print("=" * 60)

py_list_mem = sys.getsizeof(py_a) + sum(sys.getsizeof(x) for x in py_a)
np_arr_mem  = np_a.nbytes

print(f"  Python list of ints  : {py_list_mem / 1024 / 1024:8.1f} MB")
print(f"  NumPy int64 array    : {np_arr_mem  / 1024 / 1024:8.1f} MB")
print(f"  Memory ratio         :  Python uses ~{py_list_mem / np_arr_mem:.1f}x more memory")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: Why NumPy?")
print("=" * 60)
print("""
  1. SPEED  — NumPy operations are implemented in optimised C/Fortran code.
              Vectorised operations avoid slow Python interpreter loops.

  2. MEMORY — NumPy stores data in typed, contiguous blocks.
              A Python int object alone takes 28 bytes; NumPy int64 takes 8.

  3. SYNTAX — Broadcasting and ufuncs make numerical code concise and readable.

  4. ECOSYSTEM — NumPy is the foundation of Pandas, SciPy, Matplotlib,
                  scikit-learn, TensorFlow, and PyTorch.

  Use NumPy whenever you work with large numerical arrays or matrices.
""")

print("Done — numpy_vs_python.py")
