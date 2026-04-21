"""
numpy_array_operations.py
==========================
Demonstrates indexing, slicing, reshaping, stacking, splitting,
and broadcasting with NumPy arrays.

Run:
    python numpy_array_operations.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Indexing and slicing — 1-D
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. INDEXING AND SLICING — 1-D")
print("=" * 60)

arr = np.arange(10)          # [0 1 2 3 4 5 6 7 8 9]
print("Array       :", arr)
print("arr[3]      :", arr[3])          # 3
print("arr[-1]     :", arr[-1])         # 9
print("arr[2:7]    :", arr[2:7])        # [2 3 4 5 6]
print("arr[::2]    :", arr[::2])        # [0 2 4 6 8]
print("arr[::-1]   :", arr[::-1])       # [9 8 7 6 5 4 3 2 1 0]

# ---------------------------------------------------------------------------
# 2. Indexing and slicing — 2-D
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. INDEXING AND SLICING — 2-D")
print("=" * 60)

mat = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])
print("Matrix:\n", mat)
print("mat[1, 2]      :", mat[1, 2])         # 60
print("mat[0]         :", mat[0])            # first row
print("mat[:, 1]      :", mat[:, 1])         # second column
print("mat[0:2, 1:3]  :\n", mat[0:2, 1:3])  # sub-matrix

# ---------------------------------------------------------------------------
# 3. Boolean (mask) indexing
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. BOOLEAN (MASK) INDEXING")
print("=" * 60)

scores = np.array([45, 82, 67, 91, 55, 78])
print("Scores :", scores)

mask = scores > 70
print("Mask (>70)       :", mask)
print("Filtered scores  :", scores[mask])

# Update in-place: failing scores → 0
scores[scores < 60] = 0
print("After zero-ing failing scores:", scores)

# ---------------------------------------------------------------------------
# 4. Fancy indexing
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. FANCY INDEXING")
print("=" * 60)

data = np.array([10, 20, 30, 40, 50])
idx  = [0, 2, 4]
print("Data :", data)
print("data[[0,2,4]] :", data[idx])   # [10 30 50]

# 2-D fancy indexing
grid = np.arange(1, 10).reshape(3, 3)
print("\nGrid:\n", grid)
rows = [0, 2]
cols = [1, 2]
print("grid[[0,2], [1,2]] :", grid[rows, cols])   # [2 9]

# ---------------------------------------------------------------------------
# 5. Reshaping
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. RESHAPING")
print("=" * 60)

flat = np.arange(1, 13)
print("flat :", flat)

reshaped = flat.reshape(3, 4)
print("reshape(3,4):\n", reshaped)

reshaped_3d = flat.reshape(2, 2, 3)
print("reshape(2,2,3):\n", reshaped_3d)

# flatten vs ravel
print("\nflatten() (always copy) :", reshaped.flatten())
print("ravel()   (view if possible):", reshaped.ravel())

# Transpose
print("\nTranspose of reshape(3,4):\n", reshaped.T)

# ---------------------------------------------------------------------------
# 6. Stacking and concatenation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. STACKING AND CONCATENATION")
print("=" * 60)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("a:\n", a)
print("b:\n", b)
print("vstack(a, b):\n",       np.vstack([a, b]))
print("hstack(a, b):\n",       np.hstack([a, b]))
print("concatenate axis=0:\n", np.concatenate([a, b], axis=0))
print("concatenate axis=1:\n", np.concatenate([a, b], axis=1))
print("dstack (depth):\n",     np.dstack([a, b]))

# ---------------------------------------------------------------------------
# 7. Splitting
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. SPLITTING")
print("=" * 60)

arr = np.arange(12).reshape(4, 3)
print("Array (4x3):\n", arr)

parts = np.vsplit(arr, 2)
print("vsplit into 2:")
for p in parts:
    print(p)

row = np.arange(9)
print("\nRow:", row)
print("split into 3:", np.split(row, 3))

# ---------------------------------------------------------------------------
# 8. Broadcasting
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. BROADCASTING")
print("=" * 60)

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vec = np.array([10, 20, 30])   # shape (3,) broadcasts across rows

print("matrix:\n",       matrix)
print("row_vec :",        row_vec)
print("matrix + row_vec (broadcast):\n", matrix + row_vec)

# Outer product via broadcasting
col = np.array([[1], [2], [3]])    # shape (3,1)
row = np.array([10, 20, 30])       # shape (3,)
print("\nouter product via broadcasting:\n", col * row)

print("\nDone — numpy_array_operations.py")
