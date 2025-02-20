import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Iterative Fibonacci with Memoization (Bottom-Up)
def compute_bottomup_fib(n):
    if n <= 1:
        return n
    fib_cache = [0] * (n + 1)
    fib_cache[1] = 1
    for index in range(2, n + 1):
        fib_cache[index] = fib_cache[index - 1] + fib_cache[index - 2]
    return fib_cache[n]

# Fibonacci indices to test
fibo_indices = [11000, 22015, 33420, 41325, 55130, 66135]

# Measure execution times for each index
times_record = []

for num in fibo_indices:
    start = time.time()
    compute_bottomup_fib(num)
    end = time.time()
    times_record.append(end - start)

# Prepare data for display in a table format
data_matrix = np.zeros((4, len(fibo_indices)))
data_matrix[1, :] = times_record

df_results = pd.DataFrame(data_matrix, columns=fibo_indices)
df_results.index = range(4)

pd.set_option('display.max_columns', None)
print(df_results)

# Plotting execution times
plt.figure(figsize=(10, 5))
plt.plot(fibo_indices, times_record, marker='o', linestyle='-', color='b', label="Bottom-up Fibonacci Time")
plt.xlabel("Fibonacci Term (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time of Bottom-up Fibonacci Method")
plt.legend()
plt.grid(True)
plt.show()
