"""
numpy_math_statistics.py
=========================
Covers element-wise math functions, aggregations (axis-wise), linear algebra,
and random number generation with NumPy.

Run:
    python numpy_math_statistics.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Element-wise math functions (ufuncs)
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. ELEMENT-WISE MATH FUNCTIONS (ufuncs)")
print("=" * 60)

x = np.array([0, 1, 2, 3, 4], dtype=float)
print("x          :", x)
print("np.sqrt(x) :", np.sqrt(x))
print("np.exp(x)  :", np.exp(x).round(3))
print("np.log1p(x):", np.log1p(x).round(3))   # log(1+x), avoids log(0)
print("np.sin(x)  :", np.sin(x).round(3))
print("np.cos(x)  :", np.cos(x).round(3))
print("np.abs(np.array([-3,-1,0,2])) :", np.abs(np.array([-3, -1, 0, 2])))

# Trigonometric helpers
angles_deg = np.array([0, 30, 45, 60, 90])
angles_rad = np.deg2rad(angles_deg)
print("\nAngles (deg):", angles_deg)
print("sin values  :", np.sin(angles_rad).round(3))
print("cos values  :", np.cos(angles_rad).round(3))

# ---------------------------------------------------------------------------
# 2. Arithmetic operations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. ARITHMETIC OPERATIONS")
print("=" * 60)

a = np.array([10, 20, 30, 40])
b = np.array([1,   2,  3,  4])

print("a       :", a)
print("b       :", b)
print("a + b   :", a + b)
print("a - b   :", a - b)
print("a * b   :", a * b)
print("a / b   :", a / b)
print("a // b  :", a // b)     # floor division
print("a % b   :", a % b)      # modulo
print("a ** 2  :", a ** 2)     # power

# ---------------------------------------------------------------------------
# 3. Aggregations (whole array)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. AGGREGATIONS — WHOLE ARRAY")
print("=" * 60)

data = np.array([4, 7, 2, 9, 1, 5, 8, 3, 6])
print("data   :", data)
print("sum    :", np.sum(data))
print("mean   :", np.mean(data))
print("median :", np.median(data))
print("std    :", np.std(data).round(3))
print("var    :", np.var(data).round(3))
print("min    :", np.min(data),  "  (index:", np.argmin(data), ")")
print("max    :", np.max(data),  "  (index:", np.argmax(data), ")")
print("cumsum :", np.cumsum(data))

# ---------------------------------------------------------------------------
# 4. Axis-wise aggregations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. AXIS-WISE AGGREGATIONS")
print("=" * 60)

mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print("Matrix:\n", mat)
print("sum (whole)  :", mat.sum())
print("sum axis=0 (col totals):", mat.sum(axis=0))   # [12 15 18]
print("sum axis=1 (row totals):", mat.sum(axis=1))   # [ 6 15 24]
print("mean axis=0 :", mat.mean(axis=0))
print("mean axis=1 :", mat.mean(axis=1))
print("max  axis=0 :", mat.max(axis=0))

# ---------------------------------------------------------------------------
# 5. Sorting
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. SORTING")
print("=" * 60)

unsorted = np.array([5, 2, 8, 1, 9, 3])
print("unsorted      :", unsorted)
print("np.sort()     :", np.sort(unsorted))          # returns sorted copy
print("argsort()     :", np.argsort(unsorted))       # indices of sorted order

mat2 = np.array([[3, 1, 2],
                 [9, 7, 8]])
print("\nMatrix:\n", mat2)
print("sort axis=1 (each row):\n", np.sort(mat2, axis=1))

# ---------------------------------------------------------------------------
# 6. Linear algebra
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. LINEAR ALGEBRA")
print("=" * 60)

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("A:\n", A)
print("B:\n", B)
print("np.dot(A, B):\n",      np.dot(A, B))
print("A @ B (matmul):\n",    A @ B)
print("np.linalg.det(A)     :", np.linalg.det(A))
print("np.linalg.inv(A):\n",   np.linalg.inv(A).round(3))

# Eigenvalues / eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues  :", eigenvalues.round(3))
print("Eigenvectors:\n", eigenvectors.round(3))

# Solving Ax = b
b_vec = np.array([5, 11])
x = np.linalg.solve(A, b_vec)
print("\nSolve Ax=b where b=[5,11]: x =", x)   # [1, 2]

# ---------------------------------------------------------------------------
# 7. Random number generation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. RANDOM NUMBER GENERATION")
print("=" * 60)

rng = np.random.default_rng(seed=0)   # reproducible

print("Uniform [0,1) shape (3,)  :", rng.random(3).round(4))
print("Normal  µ=0 σ=1 shape (3,):", rng.standard_normal(3).round(4))
print("Integer [1,7) — dice rolls:", rng.integers(1, 7, size=10))

# Simulating coin flips
flips = rng.choice(["H", "T"], size=20)
heads = np.sum(flips == "H")
print("\n20 coin flips:", flips)
print("Heads:", heads, "  Tails:", 20 - heads)

# Shuffling
arr = np.arange(1, 11)
rng.shuffle(arr)
print("\nShuffled 1–10:", arr)

# ---------------------------------------------------------------------------
# 8. Statistical functions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. STATISTICAL FUNCTIONS")
print("=" * 60)

grades = np.array([72, 85, 90, 68, 95, 88, 76, 82, 91, 79])
print("Grades        :", grades)
print("Mean          :", np.mean(grades))
print("Median        :", np.median(grades))
print("Std dev       :", np.std(grades).round(2))
print("Percentile 25 :", np.percentile(grades, 25))
print("Percentile 75 :", np.percentile(grades, 75))
print("IQR           :", np.percentile(grades, 75) - np.percentile(grades, 25))
print("Correlation with [1..10]:",
      np.corrcoef(grades, np.arange(1, 11))[0, 1].round(3))

print("\nDone — numpy_math_statistics.py")
