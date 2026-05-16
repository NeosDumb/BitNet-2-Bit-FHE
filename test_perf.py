import numpy as np
import time

def original_1(w, g):
    return sum((w[:, :, :, ig] << ig) for ig in range(g))

def optimized_1(w, g):
    w_out = w[..., -1].copy()
    for ig in range(g - 2, -1, -1):
        w_out = (w_out << 1) | w[..., ig]
    return w_out

def original_2(w, ngroups_per_elem, g):
    return sum((w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem))

def optimized_2(w, ngroups_per_elem, g):
    w_out = w[..., -1].copy()
    for ng in range(ngroups_per_elem - 2, -1, -1):
        w_out = (w_out << g) | w[..., ng]
    return w_out

np.random.seed(0)

# Larger arrays to see perf difference
g = 2
w1 = np.random.randint(0, 2, size=(512, 128, 128, g), dtype=np.int32)

t0 = time.time()
r1 = original_1(w1, g)
t1 = time.time()
print(f"Original 1: {t1 - t0:.4f}s")

t0 = time.time()
r2 = optimized_1(w1, g)
t1 = time.time()
print(f"Optimized 1: {t1 - t0:.4f}s")

ngroups_per_elem = 4
w2 = np.random.randint(0, 4, size=(256, 16, 16, 8, 8, ngroups_per_elem), dtype=np.int32)

t0 = time.time()
r3 = original_2(w2, ngroups_per_elem, g)
t1 = time.time()
print(f"Original 2: {t1 - t0:.4f}s")

t0 = time.time()
r4 = optimized_2(w2, ngroups_per_elem, g)
t1 = time.time()
print(f"Optimized 2: {t1 - t0:.4f}s")
