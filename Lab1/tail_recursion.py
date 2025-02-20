import timeit
import matplotlib.pyplot as plt
import concurrent.futures
from functools import lru_cache

# 1. Tail Recursion with Memoization
@lru_cache(maxsize=None)
def tail_recursive_fib(n, current=0, next_val=1):
    if n == 0:
        return current
    return tail_recursive_fib(n - 1, next_val, current + next_val)

# 2. Iterative Bitwise Doubling Method
def bitwise_doubling_fib(n):
    a, b = 0, 1
    for bit in range(n.bit_length() - 1, -1, -1):
        temp = a * ((b << 1) - a)
        new_b = a * a + b * b
        a, b = temp, new_b
        if (n >> bit) & 1:
            a, b = b, a + b
    return a

# 3. Parallelized Fibonacci Calculation (Optimized with Memoization)
@lru_cache(maxsize=None)
def parallel_fib(n):
    if n <= 1:
        return n
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(parallel_fib, n - 1)
        future2 = executor.submit(parallel_fib, n - 2)
        return future1.result() + future2.result()

def run_performance_tests():
    test_values = [5, 10, 15, 20, 25, 30, 35, 40]
    timing_results = {
        "Tail Recursive": [],
        "Bitwise Doubling": [],
        "Parallel": []
    }

    for val in test_values:
        timing_results["Tail Recursive"].append(timeit.timeit(lambda: tail_recursive_fib(val), number=5))
        timing_results["Bitwise Doubling"].append(timeit.timeit(lambda: bitwise_doubling_fib(val), number=5))
        timing_results["Parallel"].append(timeit.timeit(lambda: parallel_fib(val), number=5))

    for method, times in timing_results.items():
        plt.plot(test_values, times, label=method)

    plt.xlabel("Fibonacci Term (n)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Fibonacci Algorithm Performance Comparison")
    plt.legend()
    plt.show()

# Execute the performance tests
run_performance_tests()
