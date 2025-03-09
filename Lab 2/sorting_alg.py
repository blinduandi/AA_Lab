import random
import time
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Implement Sorting Algorithms
#    (QuickSort with random pivot, MergeSort, HeapSort, InsertionSort)
# ----------------------------------------------------------------------

def quick_sort(arr):
    def partition(a, low, high):
        # Choose a random pivot index in [low, high]
        pivot_index = random.randint(low, high)
        # Swap pivot with the last element
        a[pivot_index], a[high] = a[high], a[pivot_index]

        pivot = a[high]
        i = low - 1
        for j in range(low, high):
            if a[j] <= pivot:
                i += 1
                a[i], a[j] = a[j], a[i]
        a[i + 1], a[high] = a[high], a[i + 1]
        return i + 1

    def _quick_sort(a, low, high):
        if low < high:
            pivot_idx = partition(a, low, high)
            _quick_sort(a, low, pivot_idx - 1)
            _quick_sort(a, pivot_idx + 1, high)

    _quick_sort(arr, 0, len(arr) - 1)
    return arr


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        # Copy any remaining elements of right_half
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr


def heap_sort(arr):
    n = len(arr)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)

    return arr


def heapify(a, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    # If left child is larger than root
    if left < n and a[left] > a[largest]:
        largest = left

    # If right child is larger than largest so far
    if right < n and a[right] > a[largest]:
        largest = right

    # If largest is not root
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        heapify(a, n, largest)


def insertion_sort(arr):
    """
    InsertionSort implementation.
    Time Complexity: O(n^2) average and worst case, O(n) best case (nearly sorted data).
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# ----------------------------------------------------------------------
# 2. Generate Test Data
#    We'll use four types: random, sorted, reversed, nearly sorted
# ----------------------------------------------------------------------

def generate_data(n, data_type='random'):
    """
    Generate a list of size n according to 'data_type':
      - 'random': random integers
      - 'sorted': ascending order
      - 'reversed': descending order
      - 'nearly': nearly sorted with ~10% random swaps
    """
    if data_type == 'random':
        return [random.randint(0, n) for _ in range(n)]
    elif data_type == 'sorted':
        return list(range(n))
    elif data_type == 'reversed':
        return list(range(n, 0, -1))
    elif data_type == 'nearly':
        arr = list(range(n))
        swap_count = max(1, n // 10)  # ~10% random swaps
        for _ in range(swap_count):
            i1 = random.randint(0, n - 1)
            i2 = random.randint(0, n - 1)
            arr[i1], arr[i2] = arr[i2], arr[i1]
        return arr
    else:
        # Default to random data
        return [random.randint(0, n) for _ in range(n)]

# ----------------------------------------------------------------------
# 3. Define a helper to measure time
# ----------------------------------------------------------------------

def measure_time(sort_func, arr):
    """
    Measure the time (in seconds) it takes for sort_func to sort arr.
    Returns the elapsed time as a float.
    """
    start = time.time()
    sort_func(arr)
    end = time.time()
    return end - start

# ----------------------------------------------------------------------
# 4 & 5. Run experiments and plot results
# ----------------------------------------------------------------------

def run_experiments():
    # Sizes for testing
    sizes = [500, 1000, 2000, 5000, 10000]

    # Different input data distributions
    data_types = ['random', 'sorted', 'reversed', 'nearly']

    # Dictionary to store results
    # Each key: sorting algorithm name,
    # Each value: dict of data_type -> list of (n, time)
    results = {
        'quick_sort': {dt: [] for dt in data_types},
        'merge_sort': {dt: [] for dt in data_types},
        'heap_sort': {dt: [] for dt in data_types},
        'insertion_sort': {dt: [] for dt in data_types}
    }

    # For each data type and size, measure the time for each algorithm
    for dt in data_types:
        for n in sizes:
            test_arr = generate_data(n, dt)

            # QuickSort
            arr_copy = test_arr.copy()
            q_time = measure_time(quick_sort, arr_copy)

            # MergeSort
            arr_copy = test_arr.copy()
            m_time = measure_time(merge_sort, arr_copy)

            # HeapSort
            arr_copy = test_arr.copy()
            h_time = measure_time(heap_sort, arr_copy)

            # InsertionSort
            arr_copy = test_arr.copy()
            i_time = measure_time(insertion_sort, arr_copy)

            # Store the times in our dictionary
            results['quick_sort'][dt].append((n, q_time))
            results['merge_sort'][dt].append((n, m_time))
            results['heap_sort'][dt].append((n, h_time))
            results['insertion_sort'][dt].append((n, i_time))

    # ------------------------------------------------------------------
    # Instead of plotting by data type, we now plot by sorting algorithm.
    # For each sorting algorithm, we'll have one figure with four lines
    # (one line per data distribution).
    # ------------------------------------------------------------------
    for algo_name in results:  # 'quick_sort', 'merge_sort', 'heap_sort', 'insertion_sort'
        plt.figure()
        for dt in data_types:
            # Extract lists of (n, time) for this data type
            size_time_pairs = results[algo_name][dt]
            x_sizes = [pair[0] for pair in size_time_pairs]
            y_times = [pair[1] for pair in size_time_pairs]

            # Plot one line per data type
            plt.plot(x_sizes, y_times, label=dt)

        plt.xlabel("Input size (n)")
        plt.ylabel("Execution time (seconds)")
        plt.title(f"{algo_name} performance on different data distributions")
        plt.legend()
        plt.tight_layout()

    # Show all figures
    plt.show()


if __name__ == "__main__":
    run_experiments()
