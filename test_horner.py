import numpy as np

def original_1(w, g):
    return sum([(w[:, :, :, ig] << ig) for ig in range(g)])

def optimized_1(w, g):
    w_out = w[..., -1].copy()
    for ig in range(g - 2, -1, -1):
        w_out = (w_out << 1) | w[..., ig]
    return w_out

def original_2(w, ngroups_per_elem, g):
    return sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])

def optimized_2(w, ngroups_per_elem, g):
    w_out = w[..., -1].copy()
    for ng in range(ngroups_per_elem - 2, -1, -1):
        w_out = (w_out << g) | w[..., ng]
    return w_out

np.random.seed(0)
g = 2
w1 = np.random.randint(0, 2, size=(4, 4, 4, g), dtype=np.int32)
np.testing.assert_array_equal(original_1(w1, g), optimized_1(w1, g))

ngroups_per_elem = 4
w2 = np.random.randint(0, 4, size=(2, 2, 2, 2, 2, ngroups_per_elem), dtype=np.int32)
np.testing.assert_array_equal(original_2(w2, ngroups_per_elem, g), optimized_2(w2, ngroups_per_elem, g))

print("All tests passed.")
