"""
numpy_basics.py
===============
Introduction to NumPy arrays: creation, attributes, data types, and the key
differences between NumPy arrays and plain Python lists.

Run:
    python numpy_basics.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Creating arrays
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. CREATING ARRAYS")
print("=" * 60)

# From a Python list
arr_1d = np.array([1, 2, 3, 4, 5])
print("1-D array from list      :", arr_1d)

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
print("2-D array from nested list:\n", arr_2d)

# Built-in constructors
print("\nnp.zeros((2, 3)):\n",   np.zeros((2, 3)))
print("np.ones((2, 4)):\n",    np.ones((2, 4)))
print("np.full((2, 3), 7):\n", np.full((2, 3), 7))
print("np.eye(3):\n",           np.eye(3))

# Ranges
print("\nnp.arange(0, 10, 2)  :", np.arange(0, 10, 2))
print("np.linspace(0, 1, 5) :", np.linspace(0, 1, 5))

# Random arrays (seeded for reproducibility)
rng = np.random.default_rng(seed=42)
print("\nnp.random integers (0-9, shape 2x3):\n", rng.integers(0, 10, size=(2, 3)))
print("np.random uniform  (shape 2x3):\n",        rng.random((2, 3)).round(3))

# ---------------------------------------------------------------------------
# 2. Array attributes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. ARRAY ATTRIBUTES")
print("=" * 60)

sample = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=np.float32)

print("Array:\n", sample)
print("shape    :", sample.shape)     # (2, 3)
print("ndim     :", sample.ndim)      # 2
print("dtype    :", sample.dtype)     # float32
print("size     :", sample.size)      # 6
print("itemsize :", sample.itemsize, "bytes")  # 4 for float32
print("nbytes   :", sample.nbytes,   "bytes")  # 24

# ---------------------------------------------------------------------------
# 3. Data types and type casting
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DATA TYPES AND TYPE CASTING")
print("=" * 60)

int_arr = np.array([1, 2, 3], dtype=np.int32)
print("int32 array :", int_arr, " | dtype:", int_arr.dtype)

float_arr = int_arr.astype(np.float64)
print("Cast to float64:", float_arr, " | dtype:", float_arr.dtype)

bool_arr = np.array([0, 1, 0, 1, 1], dtype=bool)
print("bool array  :", bool_arr)

# ---------------------------------------------------------------------------
# 4. NumPy array vs Python list
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. NUMPY ARRAY vs PYTHON LIST")
print("=" * 60)

py_list = [1, 2, 3, 4]
np_arr  = np.array([1, 2, 3, 4])

# Element-wise multiply — list repeats, array scales
print("Python list * 2       :", py_list * 2)   # [1,2,3,4,1,2,3,4]
print("NumPy array * 2       :", np_arr  * 2)   # [2,4,6,8]

# Element-wise add
print("Python list + [1,2,3,4]:", [a + b for a, b in zip(py_list, [1, 2, 3, 4])])
print("NumPy  array + array   :", np_arr + np.array([1, 2, 3, 4]))

# Type homogeneity
mixed_list = [1, "two", 3.0]
mixed_arr  = np.array([1, "two", 3.0])
print("\nMixed Python list  :", mixed_list, " | types:", [type(x).__name__ for x in mixed_list])
print("Mixed NumPy array  :", mixed_arr,  " | dtype:", mixed_arr.dtype)   # upcasts to string

print("\nDone — numpy_basics.py")
