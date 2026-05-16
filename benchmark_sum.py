import time
import sys

def benchmark():
    n = 10**7

    # List comprehension
    start_time = time.time()
    res_list = sum([i for i in range(n)])
    list_duration = time.time() - start_time
    print(f"sum([...]) (list comprehension): {list_duration:.4f}s")

    # Generator expression
    start_time = time.time()
    res_gen = sum(i for i in range(n))
    gen_duration = time.time() - start_time
    print(f"sum(...) (generator expression): {gen_duration:.4f}s")

    improvement = (list_duration - gen_duration) / list_duration * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    benchmark()
