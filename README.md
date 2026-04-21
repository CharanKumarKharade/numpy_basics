# numpy_basics

This repository deals with samples of NumPy, useful for quick revision and interview prep.

---

## Table of Contents

1. [What is NumPy?](#what-is-numpy)
2. [Why Use NumPy?](#why-use-numpy)
3. [Installation](#installation)
4. [Repository Structure](#repository-structure)
5. [Quick Start](#quick-start)
6. [Topics Covered](#topics-covered)

---

## What is NumPy?

**NumPy** (Numerical Python) is the foundational library for numerical computing in Python.
It provides:

- A powerful **N-dimensional array** object (`ndarray`)
- Vectorized mathematical operations that run on compiled C code
- Tools for integrating C/C++ and Fortran code
- Linear algebra, Fourier transforms, and random-number utilities

NumPy is the backbone of the entire Python scientific ecosystem — SciPy, Pandas, Matplotlib,
scikit-learn, TensorFlow, and PyTorch all build on top of it.

---

## Why Use NumPy?

| Feature | Pure Python Lists | NumPy Arrays |
|---|---|---|
| Speed | Slow (interpreted loops) | **Fast** (C-level vectorized ops) |
| Memory | Higher (object overhead) | **Lower** (typed, contiguous blocks) |
| Syntax | Verbose element-wise loops | **Concise** broadcast expressions |
| Math functions | Manual or `math` module | **Built-in** (`np.sin`, `np.exp`, …) |
| Multi-dim data | Nested lists, tricky to handle | **Native** N-D arrays |

### Performance example

```python
import numpy as np
import time

size = 1_000_000
py_list = list(range(size))
np_arr  = np.arange(size)

# Python list — loop required
t0 = time.time()
result = [x * 2 for x in py_list]
print(f"Python list: {time.time() - t0:.4f}s")

# NumPy — vectorized, no Python loop
t0 = time.time()
result = np_arr * 2
print(f"NumPy array: {time.time() - t0:.4f}s")
```

Typical output (NumPy is **10–100× faster** for large arrays):

```
Python list: 0.0821s
NumPy array: 0.0012s
```

---

## Installation

```bash
pip install numpy
```

Verify:

```python
import numpy as np
print(np.__version__)
```

---

## Repository Structure

```
numpy_basics/
├── README.md                    # This file
├── numpy_basics.py              # Array creation, attributes, data types
├── numpy_array_operations.py   # Indexing, slicing, reshaping, stacking
├── numpy_math_statistics.py    # Math functions, statistics, linear algebra
└── numpy_vs_python.py          # Performance comparison: NumPy vs pure Python
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/CharanKumarKharade/numpy_basics.git
cd numpy_basics

# Install dependency
pip install numpy

# Run any sample file
python numpy_basics.py
python numpy_array_operations.py
python numpy_math_statistics.py
python numpy_vs_python.py
```

---

## Topics Covered

### 1. `numpy_basics.py` — Core Array Concepts
- Creating arrays: `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, `np.random`
- Array attributes: `shape`, `ndim`, `dtype`, `size`, `itemsize`
- Data types and type casting
- Array vs Python list differences

### 2. `numpy_array_operations.py` — Array Manipulation
- Indexing and slicing (1-D, 2-D, N-D)
- Boolean / fancy indexing
- Reshaping: `reshape`, `flatten`, `ravel`
- Stacking: `vstack`, `hstack`, `concatenate`
- Splitting: `split`, `vsplit`, `hsplit`
- Broadcasting rules

### 3. `numpy_math_statistics.py` — Math & Statistics
- Element-wise math: `np.sqrt`, `np.exp`, `np.log`, `np.sin` / `np.cos`
- Aggregations: `sum`, `mean`, `std`, `var`, `min`, `max`
- Axis-wise operations
- Linear algebra: `np.dot`, `np.linalg.inv`, `np.linalg.eig`
- Random number generation: `np.random.rand`, `np.random.randn`, `np.random.randint`

### 4. `numpy_vs_python.py` — Why NumPy Wins
- Head-to-head timing comparisons for common operations
- Memory usage comparison
- Demonstrates vectorization and broadcasting advantages
