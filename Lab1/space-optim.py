import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_fibonacci(n):
    if n <= 1:
        return n

    fib_current = 0
    fib_prev = 1
    fib_prev_prev = 0

    for _ in range(2, n + 1):
        fib_current = fib_prev + fib_prev_prev
        fib_prev_prev = fib_prev
        fib_prev = fib_current

    return fib_current

# Indices to compute Fibonacci numbers for
fibo_indices = [11000, 22015, 33420, 41325, 55130, 66135]
execution_durations = []

# Measure execution time for each index
for index in fibo_indices:
    start = time.time()
    compute_fibonacci(index)
    end = time.time()
    execution_durations.append(end - start)

# Prepare a table to display execution times
data_matrix = np.zeros((4, len(fibo_indices)))
data_matrix[1, :] = execution_durations

df_results = pd.DataFrame(data_matrix, columns=fibo_indices)
df_results.index = range(4)

pd.set_option('display.max_columns', None)
print(df_results)

# Additionally, store computed Fibonacci numbers and their execution times
fibo_numbers = []
timings = []
for index in fibo_indices:
    start = time.time()
    result = compute_fibonacci(index)
    end = time.time()
    fibo_numbers.append(result)
    timings.append(end - start)

# Plot the execution times for the iterative Fibonacci computation
plt.figure(figsize=(8, 5))
plt.plot(fibo_indices, timings, marker='o', linestyle='-')
plt.xlabel("n-th Fibonacci Term")
plt.ylabel("Time (seconds)")
plt.title("Iterative Space-Optimized Fibonacci Function")
plt.grid()
plt.show()
