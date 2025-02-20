import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper function to compute the nth Fibonacci number with memoization
def compute_fib_memo(n, cache):
    if n <= 1:
        return n
    # Check if result is already computed
    if cache[n] != -1:
        return cache[n]
    # Recursively compute and memoize the Fibonacci number
    cache[n] = compute_fib_memo(n - 1, cache) + compute_fib_memo(n - 2, cache)
    return cache[n]

# Primary function that initializes memoization and returns the Fibonacci number
def memoized_fibonacci(n):
    cache = [-1] * (n + 1)
    return compute_fib_memo(n, cache)

# Fibonacci indices to test
fib_indices = [100, 315, 420, 525, 630, 835]
runtime_measurements = []

# Measure execution time for each Fibonacci computation
for index in fib_indices:
    start = time.time()
    memoized_fibonacci(index)
    end = time.time()
    runtime_measurements.append(end - start)

# Create a table to display the execution times
table_matrix = np.zeros((4, len(fib_indices)))
table_matrix[2, :] = runtime_measurements

df_results = pd.DataFrame(table_matrix, columns=fib_indices)
df_results.index = range(4)

pd.set_option('display.max_columns', None)
print(df_results)

# Plot the execution times
plt.figure(figsize=(10, 5))
plt.plot(fib_indices, runtime_measurements, marker='o', linestyle='-', color='b', label="Memoized Fibonacci Time")
plt.xlabel("Fibonacci Term (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time of Memoization-Based Fibonacci Method")
plt.legend()
plt.grid(True)
plt.show()
