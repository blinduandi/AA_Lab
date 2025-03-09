import tkinter as tk
from tkinter import ttk, messagebox
import random
import time

# ----------------------------
# Instrumented Sorting Algorithms with Highlighting
# ----------------------------

def quick_sort_gen(array):
    def _quick_sort(arr, low, high):
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                yield (arr.copy(), [j, high])
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield (arr.copy(), [i, j])
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield (arr.copy(), [i+1, high])
            yield from _quick_sort(arr, low, i)
            yield from _quick_sort(arr, i+2, high)
    yield from _quick_sort(array, 0, len(array) - 1)

def quick_sort_opt_gen(array):
    def _quick_sort(arr, low, high):
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                yield (arr.copy(), [j, high])
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield (arr.copy(), [i, j])
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield (arr.copy(), [i+1, high])
            yield from _quick_sort(arr, low, i)
            yield from _quick_sort(arr, i+2, high)
    yield from _quick_sort(array, 0, len(array) - 1)

def merge_sort_gen(array):
    def _merge_sort(arr, l, r):
        if l < r:
            m = (l + r) // 2
            yield from _merge_sort(arr, l, m)
            yield from _merge_sort(arr, m+1, r)
            yield from merge(arr, l, m, r)
    def merge(arr, l, m, r):
        temp = arr[l:r+1]
        i, j, k = 0, m - l + 1, l
        while i < m - l + 1 and j < r - l + 1:
            yield (arr.copy(), [l+i, l+j])
            if temp[i] <= temp[j]:
                arr[k] = temp[i]
                i += 1
            else:
                arr[k] = temp[j]
                j += 1
            k += 1
            yield (arr.copy(), [k-1])
        while i < m - l + 1:
            arr[k] = temp[i]
            i += 1
            k += 1
            yield (arr.copy(), [k-1])
        while j < r - l + 1:
            arr[k] = temp[j]
            j += 1
            k += 1
            yield (arr.copy(), [k-1])
    yield from _merge_sort(array, 0, len(array)-1)

def merge_sort_opt_gen(array):
    def _merge_sort(arr, l, r):
        if l < r:
            if arr[l:r+1] == sorted(arr[l:r+1]):
                yield (arr.copy(), [])
                return
            m = (l + r) // 2
            yield from _merge_sort(arr, l, m)
            yield from _merge_sort(arr, m+1, r)
            yield from merge(arr, l, m, r)
    def merge(arr, l, m, r):
        temp = arr[l:r+1]
        i, j, k = 0, m - l + 1, l
        while i < m - l + 1 and j < r - l + 1:
            yield (arr.copy(), [l+i, l+j])
            if temp[i] <= temp[j]:
                arr[k] = temp[i]
                i += 1
            else:
                arr[k] = temp[j]
                j += 1
            k += 1
            yield (arr.copy(), [k-1])
        while i < m - l + 1:
            arr[k] = temp[i]
            i += 1
            k += 1
            yield (arr.copy(), [k-1])
        while j < r - l + 1:
            arr[k] = temp[j]
            j += 1
            k += 1
            yield (arr.copy(), [k-1])
    yield from _merge_sort(array, 0, len(array)-1)

def heap_sort_gen(array):
    def heapify(arr, n, i):
        l = 2*i + 1
        r = 2*i + 2
        if l < n:
            yield (arr.copy(), [i, l])
        if r < n:
            yield (arr.copy(), [i, r])
        largest = i
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            yield (arr.copy(), [i, largest])
            yield from heapify(arr, n, largest)
    n = len(array)
    for i in range(n//2-1, -1, -1):
        yield from heapify(array, n, i)
    for i in range(n-1, 0, -1):
        array[0], array[i] = array[i], array[0]
        yield (array.copy(), [0, i])
        yield from heapify(array, i, 0)

def heap_sort_opt_gen(array):
    def heapify(arr, n, i):
        while True:
            l = 2*i + 1
            r = 2*i + 2
            if l < n:
                yield (arr.copy(), [i, l])
            if r < n:
                yield (arr.copy(), [i, r])
            largest = i
            if l < n and arr[l] > arr[largest]:
                largest = l
            if r < n and arr[r] > arr[largest]:
                largest = r
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                yield (arr.copy(), [i, largest])
                i = largest
            else:
                break
    n = len(array)
    for i in range(n//2-1, -1, -1):
        yield from heapify(array, n, i)
    for i in range(n-1, 0, -1):
        array[0], array[i] = array[i], array[0]
        yield (array.copy(), [0, i])
        yield from heapify(array, i, 0)

def insertion_sort_gen(arr, left, right):
    for i in range(left+1, right+1):
        key = arr[i]
        j = i-1
        yield (arr.copy(), [j, i])
        while j >= left and arr[j] > key:
            yield (arr.copy(), [j, i])
            arr[j+1] = arr[j]
            j -= 1
            yield (arr.copy(), [j+1, i])
        arr[j+1] = key
        yield (arr.copy(), [j+1, i])

def insertion_sort_whole_gen(array):
    yield from insertion_sort_gen(array, 0, len(array)-1)

# ----------------------------
# DARK THEME UI with Neon/Orange Colors & Updated Layout
# (Centered Radio Buttons for Array Size, Algorithm, and Speed)
# ----------------------------

class SortingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sorting Visualizer - Dark Theme")
        self.root.attributes("-fullscreen", True)

        # Color definitions for bars
        self.canvas_bg = "#000000"
        self.bar_color = "#39FF14"         # Neon green for positive bars
        self.negative_bar_color = "#FF4500"  # Neon orange for negative bars
        self.highlight_color = "#FFFF33"     # Bright yellow highlight
        self.baseline_color = "#FFFFFF"      # White baseline

        # Set main window background
        self.root.configure(bg="#121212")

        # ---------- ttk.Style Configuration ----------
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#121212")
        self.style.configure("TLabel", background="#121212", foreground="#FFFFFF", font=("Segoe UI", 11))
        self.style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"), foreground="#FFA500", underline=True)
        self.style.configure("NeonButton.TButton", font=("Segoe UI", 11, "bold"),
                             foreground="#FFFFFF", background="#D35400", padding=8, borderwidth=2)
        self.style.map("NeonButton.TButton", background=[("active", "#E67E22")], foreground=[("active", "#FFFFFF")])
        self.style.configure("TRadiobutton", background="#121212", foreground="#FFFFFF", font=("Segoe UI", 11))
        # Menu Bar styling
        menubar = tk.Menu(self.root, bg="#333333", fg="#FFFFFF", tearoff=0)
        file_menu = tk.Menu(menubar, tearoff=0, bg="#333333", fg="#FFFFFF")
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Data variables
        self.array = []
        self.working_array = []
        self.generator = None
        self.start_time = 0
        self.after_id = None
        self.animation_speed = 50
        self.comparison_count = 0
        self.swap_count = 0
        self.highlight_indices = []

        # ---------- Main Container Frame ----------
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        # Grid rows:
        # 0: Header
        # 1: Input Controls
        # 2: Visualization Canvas (Graph)
        # 3: Action Buttons (Generate, Start, Reset)
        # 4: Stats Panel
        main_frame.grid_rowconfigure(2, weight=6)  # Increase canvas space
        main_frame.grid_rowconfigure(4, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # ---------- Header ----------
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0,15))
        header_frame.grid_columnconfigure(0, weight=1)
        title_label = ttk.Label(header_frame, text="Sorting Visualizer", style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky="w")
        info_btn = ttk.Button(header_frame, text="ⓘ", width=3, style="NeonButton.TButton", command=self.show_info)
        info_btn.grid(row=0, column=1, sticky="e")

        # ---------- Input Controls (Centered Radio Buttons) ----------
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(0,15))
        controls_frame.grid_columnconfigure(0, weight=1)
        # Array Size
        size_frame = ttk.Frame(controls_frame)
        size_frame.grid(row=0, column=0, sticky="ew", pady=5)
        self.size_var = tk.IntVar(value=50)
        rb_small = ttk.Radiobutton(size_frame, text="Small (20)", variable=self.size_var, value=20)
        rb_medium = ttk.Radiobutton(size_frame, text="Medium (50)", variable=self.size_var, value=50)
        rb_large = ttk.Radiobutton(size_frame, text="Large (100)", variable=self.size_var, value=100)
        rb_small.pack(side="left", padx=10)
        rb_medium.pack(side="left", padx=10)
        rb_large.pack(side="left", padx=10)
        # Algorithm Selection
        algo_frame = ttk.Frame(controls_frame)
        algo_frame.grid(row=1, column=0, sticky="ew", pady=5)
        self.algo_var = tk.StringVar(value="Quick Sort")
        algorithms = [
            "Quick Sort", "Quick Sort Optimised",
            "Merge Sort", "Merge Sort Optimised",
            "Heap Sort", "Heap Sort Optimised",
            "Insertion Sort"
        ]
        for algo in algorithms:
            rb = ttk.Radiobutton(algo_frame, text=algo, variable=self.algo_var, value=algo)
            rb.pack(side="left", padx=5)
        # Speed Selection
        speed_frame = ttk.Frame(controls_frame)
        speed_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.speed_var = tk.StringVar(value="Normal")
        rb_slow = ttk.Radiobutton(speed_frame, text="Slow", variable=self.speed_var, value="Slow")
        rb_normal = ttk.Radiobutton(speed_frame, text="Normal", variable=self.speed_var, value="Normal")
        rb_fast = ttk.Radiobutton(speed_frame, text="Fast", variable=self.speed_var, value="Fast")
        rb_super = ttk.Radiobutton(speed_frame, text="Super Fast", variable=self.speed_var, value="Super Fast")
        rb_slow.pack(side="left", padx=5)
        rb_normal.pack(side="left", padx=5)
        rb_fast.pack(side="left", padx=5)
        rb_super.pack(side="left", padx=5)

        # ---------- Visualization Canvas (Graph) ----------
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=2, column=0, sticky="nsew", pady=(0,15))
        canvas_border = tk.Frame(canvas_frame, bg="#333333", bd=2, relief="ridge")
        canvas_border.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_border, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

        # ---------- Action Buttons (Under the Graph) ----------
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, sticky="ew", pady=(0,15))
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        buttons_frame.grid_columnconfigure(2, weight=1)
        gen_btn = ttk.Button(buttons_frame, text="Generate Array", style="NeonButton.TButton", command=self.generate_array)
        gen_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        start_btn = ttk.Button(buttons_frame, text="Start Sorting", style="NeonButton.TButton", command=self.start_sorting)
        start_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        reset_btn = ttk.Button(buttons_frame, text="Reset", style="NeonButton.TButton", command=self.reset)
        reset_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # ---------- Stats Panel ----------
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=4, column=0, sticky="ew", pady=(0,15))
        for i in range(4):
            stats_frame.grid_columnconfigure(i, weight=1)
        self.time_label = ttk.Label(stats_frame, text="Time: 0.000 sec", anchor="center")
        self.time_label.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.comp_label = ttk.Label(stats_frame, text="Comparisons: 0", anchor="center")
        self.comp_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.swap_label = ttk.Label(stats_frame, text="Swaps: 0", anchor="center")
        self.swap_label.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.array_size_label = ttk.Label(stats_frame, text="Array Size: 0", anchor="center")
        self.array_size_label.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

    # ---------------------------
    # GUI Helper Methods
    # ---------------------------
    def show_info(self):
        info_text = (
            "Dark-Themed Sorting Visualizer\n\n"
            "Visualize various sorting algorithms with highlighted comparisons.\n"
            "Controls are centered for a clean, modern look.\n\n"
            "• Select array size, algorithm, and speed using the radio buttons above.\n"
            "• Click 'Generate Array' to create a new array.\n"
            "• Click 'Start Sorting' to begin visualization.\n"
            "• Click 'Reset' to clear the visualization.\n"
            "• Press ESC to exit full screen, or use File > Exit to close."
        )
        messagebox.showinfo("About Sorting Visualizer", info_text)

    def generate_array(self):
        try:
            size = int(self.size_var.get())
            if size < 1 or size > 500:
                messagebox.showerror("Error", "Array size must be between 1 and 500")
                return
            self.array = [random.randint(-100, 100) for _ in range(size)]
            self.array_size_label.config(text=f"Array Size: {size}")
            self.draw_array()
            self.reset_stats()
        except ValueError:
            messagebox.showerror("Error", "Invalid array size")

    def reset_stats(self):
        self.comparison_count = 0
        self.swap_count = 0
        self.time_label.config(text="Time: 0.000 sec")
        self.comp_label.config(text="Comparisons: 0")
        self.swap_label.config(text="Swaps: 0")

    def draw_array(self):
        self.canvas.delete("all")
        if not self.array:
            return
        n = len(self.array)
        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 400
        bar_width = width / n
        max_val = max(abs(x) for x in self.array) or 1
        baseline = height // 2
        self.canvas.create_line(0, baseline, width, baseline, fill=self.baseline_color, width=1)
        for i, num in enumerate(self.array):
            x0 = i * bar_width + 1
            x1 = (i+1) * bar_width - 1
            bar_height = (abs(num) / max_val) * (baseline - 20)
            if num >= 0:
                y0 = baseline - bar_height
                y1 = baseline
            else:
                y0 = baseline
                y1 = baseline + bar_height
            fill_color = self.highlight_color if i in self.highlight_indices else (self.bar_color if num >= 0 else self.negative_bar_color)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="#222222")
            if bar_width > 15:
                text_y = y0 - 10 if num >= 0 else y1 + 10
                self.canvas.create_text((x0+x1)/2, text_y, text=str(num),
                                        font=("Segoe UI", 8), fill="#FFFFFF")

    def start_sorting(self):
        if not self.array:
            messagebox.showinfo("Info", "Please generate an array first.")
            return
        if self.generator:
            messagebox.showinfo("Info", "Sorting is already in progress.")
            return
        self.working_array = self.array.copy()
        self.start_time = time.time()
        self.reset_stats()
        speed_choice = self.speed_var.get()
        if speed_choice == "Slow":
            self.animation_speed = 100
        elif speed_choice == "Normal":
            self.animation_speed = 50
        elif speed_choice == "Fast":
            self.animation_speed = 10
        elif speed_choice == "Super Fast":
            self.animation_speed = 1
        sort_type = self.algo_var.get()
        if sort_type == "Quick Sort":
            self.generator = quick_sort_gen(self.working_array)
        elif sort_type == "Quick Sort Optimised":
            self.generator = quick_sort_opt_gen(self.working_array)
        elif sort_type == "Merge Sort":
            self.generator = merge_sort_gen(self.working_array)
        elif sort_type == "Merge Sort Optimised":
            self.generator = merge_sort_opt_gen(self.working_array)
        elif sort_type == "Heap Sort":
            self.generator = heap_sort_gen(self.working_array)
        elif sort_type == "Heap Sort Optimised":
            self.generator = heap_sort_opt_gen(self.working_array)
        elif sort_type == "Insertion Sort":
            self.generator = insertion_sort_whole_gen(self.working_array)
        self.animate()

    def animate(self):
        try:
            result = next(self.generator)
            if isinstance(result, tuple):
                self.array = result[0].copy()
                self.highlight_indices = result[1]
                self.comparison_count += 1
                self.swap_count += 1
            else:
                self.array = result.copy()
                self.highlight_indices = []
            elapsed = time.time() - self.start_time
            self.time_label.config(text=f"Time: {elapsed:.3f} sec")
            self.comp_label.config(text=f"Comparisons: {self.comparison_count}")
            self.swap_label.config(text=f"Swaps: {self.swap_count}")
            self.draw_array()
            self.after_id = self.root.after(self.animation_speed, self.animate)
        except StopIteration:
            self.generator = None
            self.after_id = None
            elapsed = time.time() - self.start_time
            messagebox.showinfo("Sort Complete",
                                f"Algorithm: {self.algo_var.get()}\n"
                                f"Time: {elapsed:.3f} sec\n"
                                f"Comparisons: {self.comparison_count}\n"
                                f"Swaps: {self.swap_count}")

    def reset(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.array = []
        self.working_array = []
        self.generator = None
        self.highlight_indices = []
        self.canvas.delete("all")
        self.reset_stats()
        self.array_size_label.config(text="Array Size: 0")


if __name__ == "__main__":
    root = tk.Tk()
    app = SortingVisualizer(root)
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
    root.mainloop()
