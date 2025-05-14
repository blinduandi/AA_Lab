#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Algorithm Visualizer –  versiune completă funcțională
Autor: ChatGPT • aprilie 2025
Revizuit și îmbunătățit: Gemini • mai 2025

• Suportă algoritmi: BFS, DFS, Dijkstra, Prim, Kruskal, Floyd-Warshall
• Afişează în timp real coada/stiva, nodurile vizitate şi muchiile active
• Rezultatul final (ordine parcurgere, distanţe minime sau arbore parţial minim, APSP)
  este afişat în panoul din dreapta după terminarea execuţiei
• Include măsurarea timpului de execuție al algoritmului
• Adaugă tab de comparație a performanței algoritmilor cu grafic și tabel.
"""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from math import inf, log
from typing import Dict, Generator, List, Tuple, Optional, Set, Any
import time
import threading
import copy  # For deep copying matrices

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------- Configurări grafice (păstrate) ---------- #
BACKGROUND_COLOR = "#2b2b2b"
TEXT_COLOR = "#ffffff"
FONT_FAMILY = "Segoe UI"
NODE_COLOR = "#4e5d6c"
VISITED_COLOR = "#ffc107"  # Also for K-node in Floyd
CURRENT_COLOR = "#9ccc65"  # Also for I-node in Floyd
QUEUE_COLOR = "#81c784"  # Also for J-node in Floyd
STACK_COLOR = "#ef5350"
EDGE_COLOR = "#616161"
ACTIVE_EDGE_COLOR = "#00e676"
HIGHLIGHT_EDGE_COLOR = "#ffee58"
PATH_EDGE_COLOR = "#18ffff"

BUTTON_COLOR = "#3c3f41"
BUTTON_HOVER_COLOR = "#4a5358"
BUTTON_ACTIVE_COLOR = "#33383b"
SPINBOX_BG_COLOR = "#3c3f41"
SPINBOX_FG_COLOR = "#ffffff"
COMBOBOX_BG_COLOR = "#3c3f41"
COMBOBOX_FG_COLOR = "#ffffff"

INFO_PANEL_WIDTH = 370  # Slightly wider for potential matrix info
NODE_SIZE = 2300
DEFAULT_GRAPH_SIZE = 10  # Reduced default for potentially slower Floyd
DEFAULT_DELAY_MS = 300

PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
               '#17becf']


# ---------------------------------------------------------------------------- #
#                               Generare graf (păstrat)                       #
# ---------------------------------------------------------------------------- #
def create_connected_graph(n: int, seed: Optional[int] = None) -> nx.Graph:
    if seed is not None:
        random.seed(seed)
    G = nx.Graph()
    if n == 0: return G
    if n == 1: G.add_node(0); return G
    G.add_nodes_from(range(n))
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(n - 1):
        G.add_edge(nodes[i], nodes[i + 1], weight=random.randint(1, 10))
    num_extra_edges = n
    for _ in range(num_extra_edges):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=random.randint(1, 10))
    return G


# ---------------------------------------------------------------------------- #
#                Generatoare algoritmi (BFS, DFS, Dijkstra, Prim, Kruskal - păstrate) #
# ---------------------------------------------------------------------------- #
# These functions (bfs_gen, dfs_gen, dijkstra_gen, prim_gen, kruskal_gen, UnionFind)
# are assumed to be present from the previous version and are not repeated for brevity.
# (The placeholder functions from the problem description are used below)

def bfs_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    visited: Set[int] = set()
    order: List[int] = []
    queue: List[int] = []
    if not G.nodes: yield ("result", []); return
    if start not in G: yield ("error", f"Nod start {start} invalid."); yield ("result", []); return
    queue.append(start)
    yield ("queue", queue.copy())
    while queue:
        u = queue.pop(0)
        if u in visited: continue
        visited.add(u);
        order.append(u)
        yield ("visit", u);
        yield ("queue", queue.copy())
        for v in sorted(list(G.neighbors(u))):
            if v not in visited and v not in queue: queue.append(v); yield ("queue", queue.copy())
    yield ("result", order)


def dfs_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    visited: Set[int] = set()
    order: List[int] = []
    stack: List[int] = []
    if not G.nodes: yield ("result", []); return
    if start not in G: yield ("error", f"Nod start {start} invalid."); yield ("result", []); return
    stack.append(start)
    yield ("stack", stack.copy())
    while stack:
        u = stack.pop()
        if u in visited: yield ("stack", stack.copy()); continue
        visited.add(u);
        order.append(u)
        yield ("visit", u);
        yield ("stack", stack.copy())
        for v in sorted(list(G.neighbors(u)), reverse=True):
            if v not in visited: stack.append(v); yield ("stack", stack.copy())
    yield ("result", order)


def dijkstra_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    if not G.nodes: yield ("result", {}); return
    if start not in G: yield ("error", f"Nod start {start} invalid."); yield ("result", {}); return
    dist: Dict[int, float] = {v: inf for v in G.nodes};
    pred: Dict[int, Optional[int]] = {v: None for v in G.nodes}
    dist[start] = 0;
    pq: List[Tuple[float, int]] = [(0, start)];
    processed: Set[int] = set()
    yield ("queue", [v for _, v in pq])
    while pq:
        pq.sort();
        d, u = pq.pop(0)
        yield ("queue", [v for _, v in pq])
        if u in processed: continue
        processed.add(u);
        yield ("visit", u)
        if pred[u] is not None: yield ("edge", (pred[u], u))
        for v in G.neighbors(u):
            if v not in processed:
                weight = G.edges[u, v].get("weight", 1);
                nd = d + weight
                if nd < dist[v]:
                    pq = [(cost, node) for cost, node in pq if node != v]
                    dist[v] = nd;
                    pred[v] = u;
                    pq.append((nd, v))
                    yield ("highlight_edge_temp", (u, v));
                    yield ("queue", [x for _, x in pq])
    final_paths_edges = [tuple(sorted((p_node, node))) for node, p_node in pred.items() if p_node is not None]
    yield ("path_edges", list(set(final_paths_edges)))
    yield ("result", {"distances": dist, "predecessors": pred})


def prim_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    if not G.nodes: yield ("result", []); return
    if start not in G: start = list(G.nodes())[0] if G.nodes() else 0
    visited: Set[int] = {start};
    mst_edges: List[Tuple[int, int]] = []
    candidate_edges: List[Tuple[int, int, int]] = []

    def add_edges_from_node(u_node):
        for v_node in G.neighbors(u_node):
            if v_node not in visited:
                candidate_edges.append((G.edges[u_node, v_node].get("weight", 1), u_node, v_node))

    add_edges_from_node(start)
    yield ("visit", start);
    yield ("edge_candidates", [(u, v) for _, u, v in candidate_edges])
    while candidate_edges and len(visited) < G.number_of_nodes():
        candidate_edges.sort();
        w, u, v = candidate_edges.pop(0)
        yield ("edge_candidates", [(n1, n2) for _, n1, n2 in candidate_edges])
        if v in visited: yield ("highlight_edge_skip", (u, v)); continue
        visited.add(v);
        mst_edges.append(tuple(sorted((u, v))))
        yield ("visit", v);
        yield ("edge", (u, v))
        add_edges_from_node(v)
        yield ("edge_candidates", [(n1, n2) for _, n1, n2 in candidate_edges])
    yield ("result", mst_edges)


class UnionFind:
    def __init__(self, nodes: List[int]):  # Changed to accept node list for mapping
        self.node_to_idx = {node_val: i for i, node_val in enumerate(nodes)}
        self.idx_to_node = {i: node_val for i, node_val in enumerate(nodes)}
        n_indices = len(nodes)
        self.parent = list(range(n_indices))
        self.num_sets = n_indices

    def find(self, node_val: int) -> int:  # Operates on node values, translates to indices
        x = self.node_to_idx[node_val]
        if self.parent[x] == x: return x
        self.parent[x] = self.find(self.idx_to_node[self.parent[
            x]])  # Path compression needs original node value for recursive call if find expects node_val
        # Simpler: parent stores indices. find operates on index.
        # Let's simplify: find operates on index. The class maps initially.
        # The user of UnionFind will pass indices if needed, or the class handles it fully.
        # For Kruskal, it's easier if uf.union(u,v) takes actual node values.

        # Reverted to simpler index-based internal parent array for now
        # find_idx(x_idx)
        if self.parent[x] == x:
            return x  # Return index
        self.parent[x] = self._find_idx(self.parent[x])
        return self.parent[x]

    def _find_idx(self, x_idx: int) -> int:  # Internal helper for indices
        if self.parent[x_idx] == x_idx:
            return x_idx
        self.parent[x_idx] = self._find_idx(self.parent[x_idx])
        return self.parent[x_idx]

    def union(self, node_a_val: int, node_b_val: int) -> bool:
        idx_a = self.node_to_idx[node_a_val]
        idx_b = self.node_to_idx[node_b_val]
        root_a_idx = self._find_idx(idx_a)
        root_b_idx = self._find_idx(idx_b)
        if root_a_idx == root_b_idx: return False
        self.parent[root_b_idx] = root_a_idx
        self.num_sets -= 1
        return True


def kruskal_gen(G: nx.Graph) -> Generator[Tuple[str, Any], None, None]:
    if not G.nodes: yield ("result", []); return
    uf = UnionFind(list(G.nodes()))  # Pass actual node values
    mst_edges: List[Tuple[int, int]] = []
    all_edges = sorted([(G.edges[u, v].get("weight", 1), u, v) for u, v in G.edges()], key=lambda e: e[0])
    yield ("edge_candidates", [(u, v) for _, u, v in all_edges])
    for w, u, v in all_edges:
        yield ("highlight_edge_temp", (u, v))
        if uf.union(u, v):  # Use actual node values
            mst_edges.append(tuple(sorted((u, v))))
            yield ("edge", (u, v))
            if len(mst_edges) == G.number_of_nodes() - 1 and G.number_of_nodes() > 0: break
        else:
            yield ("highlight_edge_skip", (u, v))
    yield ("result", mst_edges)


# ---------------------------------------------------------------------------- #
#                           Floyd-Warshall Algorithm                           #
# ---------------------------------------------------------------------------- #
def floyd_warshall_gen(G: nx.Graph) -> Generator[Tuple[str, Any], None, None]:
    if not G.nodes:
        yield ("result", {"distances": [], "predecessors": []})
        return

    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    n = len(nodes)

    # Initialize distance matrix (dist_matrix)
    dist_matrix = [[inf] * n for _ in range(n)]
    # Initialize predecessor matrix (pred_matrix) for path reconstruction
    # pred_matrix[i][j] stores the predecessor of j on the path from i to j
    pred_matrix = [[None] * n for _ in range(n)]

    for i_idx in range(n):
        dist_matrix[i_idx][i_idx] = 0
        pred_matrix[i_idx][i_idx] = i_idx  # Predecessor of i from i is i itself (or None, depending on convention)

    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        weight = data.get('weight', 1)
        dist_matrix[u_idx][v_idx] = weight
        dist_matrix[v_idx][u_idx] = weight  # Assuming undirected graph for now
        pred_matrix[u_idx][v_idx] = u_idx
        pred_matrix[v_idx][u_idx] = v_idx

    yield ("matrix_state", {
        "dist": copy.deepcopy(dist_matrix),
        "pred": copy.deepcopy(pred_matrix),
        "node_map": idx_to_node  # Send node map for display purposes
    })

    for k_idx in range(n):
        k_node = idx_to_node[k_idx]
        yield ("k_iteration", {"k_node": k_node, "k_idx": k_idx})
        for i_idx in range(n):
            i_node = idx_to_node[i_idx]
            for j_idx in range(n):
                j_node = idx_to_node[j_idx]

                # Yield before potential update to show what's being checked
                yield ("check_path", {
                    "i_node": i_node, "j_node": j_node, "k_node": k_node,
                    "i_idx": i_idx, "j_idx": j_idx, "k_idx": k_idx,
                    "current_dist_ij": dist_matrix[i_idx][j_idx],
                    "dist_ik": dist_matrix[i_idx][k_idx],
                    "dist_kj": dist_matrix[k_idx][j_idx]
                })

                if dist_matrix[i_idx][k_idx] != inf and \
                        dist_matrix[k_idx][j_idx] != inf and \
                        dist_matrix[i_idx][k_idx] + dist_matrix[k_idx][j_idx] < dist_matrix[i_idx][j_idx]:
                    new_dist = dist_matrix[i_idx][k_idx] + dist_matrix[k_idx][j_idx]
                    dist_matrix[i_idx][j_idx] = new_dist
                    pred_matrix[i_idx][j_idx] = pred_matrix[k_idx][
                        j_idx]  # Predecessor of j via k is k's predecessor for j

                    yield ("path_updated", {
                        "i_node": i_node, "j_node": j_node, "k_node": k_node,
                        "i_idx": i_idx, "j_idx": j_idx,
                        "new_dist": new_dist
                    })
                    yield ("matrix_state", {  # Send updated matrices
                        "dist": copy.deepcopy(dist_matrix),
                        "pred": copy.deepcopy(pred_matrix),
                        "node_map": idx_to_node
                    })

    # Check for negative cycles
    for i_idx in range(n):
        if dist_matrix[i_idx][i_idx] < 0:
            yield ("negative_cycle_detected", {"node_idx": i_idx, "node": idx_to_node[i_idx]})
            # Result might be invalid in presence of negative cycles
            # For simplicity, we still yield the matrices. A more robust solution
            # might alter the result format or stop.

    # Convert matrices to use original node labels in keys for the final result if desired
    # For now, the generator result will use the idx_to_node mapping provided earlier.
    final_dist_dict = {idx_to_node[r]: {idx_to_node[c]: dist_matrix[r][c] for c in range(n)} for r in range(n)}
    final_pred_dict = {
        idx_to_node[r]: {idx_to_node[c]: (idx_to_node[pred_matrix[r][c]] if pred_matrix[r][c] is not None else None) for
                         c in range(n)} for r in range(n)}

    yield ("result", {"distances": final_dist_dict, "predecessors": final_pred_dict, "node_map": idx_to_node})


_ALGOS = {
    "BFS": bfs_gen,
    "DFS": dfs_gen,
    "Dijkstra": dijkstra_gen,
    "Prim": prim_gen,
    "Kruskal": kruskal_gen,
    "Floyd-Warshall": floyd_warshall_gen,
}


# ---------------------------------------------------------------------------- #
#                                Stare vizuală                                #
# ---------------------------------------------------------------------------- #
@dataclass
class VisualState:
    visited: Set[int] = field(default_factory=set)  # General visited nodes
    # Specific for Floyd-Warshall visualization
    fw_k_node: Optional[int] = None
    fw_i_node: Optional[int] = None
    fw_j_node: Optional[int] = None
    fw_dist_matrix_display: Optional[List[List[Any]]] = None  # For displaying a part or summary
    fw_node_map: Optional[Dict[int, Any]] = None

    queue: List[int] = field(default_factory=list)
    stack: List[int] = field(default_factory=list)
    edge_colors: Dict[Tuple[int, int], str] = field(default_factory=dict)
    solution_edges: Set[Tuple[int, int]] = field(default_factory=set)
    candidate_edges: Set[Tuple[int, int]] = field(default_factory=set)
    highlighted_edge_temp: Optional[Tuple[int, int]] = None
    highlighted_edge_skip: Optional[Tuple[int, int]] = None
    step: int = 0
    last_action: str = ""
    current_message: str = ""
    result: Optional[Any] = None
    execution_time_ms: Optional[float] = None


# ---------------------------------------------------------------------------- #
#                           Clasa principală de GUI                            #
# ---------------------------------------------------------------------------- #

class GraphVisualizer:  # Structure from previous version
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Graph Algorithm Visualizer & Comparator")
        self.root.geometry("1450x950")
        self.root.configure(bg=BACKGROUND_COLOR)
        self.root.minsize(1200, 700)
        self._setup_style()
        self.G: Optional[nx.Graph] = None;
        self.pos: Dict[int, Tuple[float, float]] = {}
        self.state: VisualState = VisualState()
        self.step_iter: Optional[Generator] = None;
        self.running_visualization: bool = False
        self.visualization_start_time: float = 0.0
        self.comparison_results_data: Dict[str, List[Tuple[int, float]]] = {}
        self.comparison_thread: Optional[threading.Thread] = None
        self.notebook = ttk.Notebook(self.root)
        self.visualizer_tab = ttk.Frame(self.notebook, style="TFrame")
        self._init_vars_visualizer();
        self._build_visualizer_tab()
        self.notebook.add(self.visualizer_tab, text=" Algorithm Visualizer ")
        self.comparator_tab = ttk.Frame(self.notebook, style="TFrame")
        self._init_vars_comparator();
        self._build_comparator_tab()
        self.notebook.add(self.comparator_tab, text=" Performance Comparison ")
        self.notebook.pack(expand=True, fill="both", padx=5, pady=5)
        self.generate_graph_for_visualizer()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _on_closing(self):  # As before
        if self.comparison_thread and self.comparison_thread.is_alive():
            if messagebox.askyesno("Confirm Quit", "Comparison is running. Quit?"):
                self.root.destroy()
            else:
                return
        self.root.destroy()

    def _setup_style(self):  # As before
        style = ttk.Style(self.root);
        style.theme_use("clam")
        style.configure(".", font=(FONT_FAMILY, 10), background=BACKGROUND_COLOR, foreground=TEXT_COLOR)
        style.configure("TFrame", background=BACKGROUND_COLOR)
        style.configure("TLabel", background=BACKGROUND_COLOR, foreground=TEXT_COLOR, padding=(2, 2),
                        font=(FONT_FAMILY, 10))
        style.configure("Bold.TLabel", font=(FONT_FAMILY, 11, "bold"))
        style.configure("Title.TLabel", font=(FONT_FAMILY, 14, "bold"), foreground=VISITED_COLOR)
        style.configure("TButton", background=BUTTON_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, 10, "bold"),
                        padding=(8, 4), relief=tk.FLAT, borderwidth=1)
        style.map("TButton", background=[("pressed", BUTTON_ACTIVE_COLOR), ("active", BUTTON_HOVER_COLOR),
                                         ("disabled", "#555555")], foreground=[("disabled", "#999999")],
                  relief=[("pressed", tk.SUNKEN), ("!pressed", tk.FLAT)])
        style.configure("TSpinbox", arrowsize=12, background=SPINBOX_BG_COLOR, foreground=SPINBOX_FG_COLOR,
                        font=(FONT_FAMILY, 10), relief=tk.FLAT, borderwidth=1, fieldbackground=SPINBOX_BG_COLOR,
                        selectbackground="#555555", selectforeground=TEXT_COLOR)
        style.configure("TCombobox", background=COMBOBOX_BG_COLOR, foreground=COMBOBOX_FG_COLOR, font=(FONT_FAMILY, 10),
                        relief=tk.FLAT, padding=(4, 3), fieldbackground=COMBOBOX_BG_COLOR, selectbackground="#555555",
                        selectforeground=TEXT_COLOR)
        self.root.option_add('*TCombobox*Listbox.background', COMBOBOX_BG_COLOR);
        self.root.option_add('*TCombobox*Listbox.foreground', COMBOBOX_FG_COLOR)
        self.root.option_add('*TCombobox*Listbox.selectBackground', BUTTON_HOVER_COLOR);
        self.root.option_add('*TCombobox*Listbox.selectForeground', TEXT_COLOR)
        self.root.option_add('*TCombobox*Listbox.font', (FONT_FAMILY, 10))
        style.configure("Horizontal.TPanedwindow", background=BACKGROUND_COLOR);
        style.configure("TSeparator", background="#555555")
        style.configure("Treeview", fieldbackground=BACKGROUND_COLOR, background=BACKGROUND_COLOR,
                        foreground=TEXT_COLOR)
        style.configure("Treeview.Heading", background=BUTTON_COLOR, foreground=TEXT_COLOR,
                        font=(FONT_FAMILY, 10, "bold"));
        style.map("Treeview.Heading", background=[('active', BUTTON_HOVER_COLOR)])
        style.configure("TNotebook.Tab", font=(FONT_FAMILY, 10, "bold"), padding=[5, 2], background=BACKGROUND_COLOR)
        style.map("TNotebook.Tab", background=[("selected", BUTTON_HOVER_COLOR), ("!selected", BUTTON_COLOR)],
                  foreground=[("selected", TEXT_COLOR), ("!selected", "#cccccc")])
        style.configure("TCheckbutton", background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, 10),
                        indicatorcolor=TEXT_COLOR, selectcolor=BACKGROUND_COLOR)
        style.map("TCheckbutton", indicatorcolor=[('selected', VISITED_COLOR), ('!selected', TEXT_COLOR)],
                  foreground=[('disabled', '#777777')])
        style.configure("Status.TLabel", font=(FONT_FAMILY, 9), foreground="#cccccc")

    def _init_vars_visualizer(self):  # As before
        self.vis_alg_var = tk.StringVar(value="BFS")
        self.vis_size_var = tk.IntVar(value=DEFAULT_GRAPH_SIZE)
        self.vis_delay_var = tk.IntVar(value=DEFAULT_DELAY_MS)
        self.vis_start_var = tk.IntVar(value=0)

    def _build_visualizer_tab(self):  # As before, with minor adjustments if needed for Floyd info
        vis_controls_frame = ttk.Frame(self.visualizer_tab, padding=(10, 5));
        vis_controls_frame.pack(fill=tk.X, side=tk.TOP)
        ttk.Label(vis_controls_frame, text="Algoritm:").pack(side=tk.LEFT, padx=(0, 5))
        self.vis_alg_combo = ttk.Combobox(vis_controls_frame, textvariable=self.vis_alg_var, values=list(_ALGOS.keys()),
                                          width=12, state="readonly")  # Wider for Floyd-Warshall
        self.vis_alg_combo.pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(vis_controls_frame, text="Noduri:").pack(side=tk.LEFT, padx=(0, 5))
        self.vis_size_spin = ttk.Spinbox(vis_controls_frame, from_=3, to=25, textvariable=self.vis_size_var,
                                         width=5)  # Max 25 for Floyd visualizer
        self.vis_size_spin.pack(side=tk.LEFT, padx=(0, 15))
        self.vis_size_var.trace_add("write", self._update_vis_start_node_max)
        self.vis_start_node_label = ttk.Label(vis_controls_frame, text="Nod start:")  # Store to hide/show
        self.vis_start_node_label.pack(side=tk.LEFT, padx=(0, 5))
        self.vis_start_spin = ttk.Spinbox(vis_controls_frame, from_=0, to=self.vis_size_var.get() - 1,
                                          textvariable=self.vis_start_var, width=5)
        self.vis_start_spin.pack(side=tk.LEFT, padx=(0, 15))
        self.vis_alg_var.trace_add("write", self._toggle_start_node_for_visualizer)  # Hide start for Floyd

        ttk.Label(vis_controls_frame, text="Întârziere (ms):").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(vis_controls_frame, from_=50, to=3000, increment=50, textvariable=self.vis_delay_var, width=7).pack(
            side=tk.LEFT, padx=(0, 20))
        self.vis_btn_gen = ttk.Button(vis_controls_frame, text="Generează Graf Nou",
                                      command=self.generate_graph_for_visualizer);
        self.vis_btn_gen.pack(side=tk.LEFT, padx=5)
        self.vis_btn_run = ttk.Button(vis_controls_frame, text="Rulează Algoritm", command=self.run_visualization,
                                      state="disabled");
        self.vis_btn_run.pack(side=tk.LEFT, padx=5)

        vis_pw = ttk.Panedwindow(self.visualizer_tab, orient=tk.HORIZONTAL, style="Horizontal.TPanedwindow");
        vis_pw.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        vis_canvas_outer_frame = ttk.Frame(vis_pw, padding=0)
        self.vis_fig = Figure(figsize=(8, 8), facecolor=BACKGROUND_COLOR, constrained_layout=True);
        self.vis_fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.vis_ax = self.vis_fig.add_subplot(111);
        self.vis_ax.axis("off")
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, master=vis_canvas_outer_frame);
        self.vis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        vis_pw.add(vis_canvas_outer_frame, weight=3)
        vis_info_outer_frame = ttk.Frame(vis_pw, width=INFO_PANEL_WIDTH, style="TFrame")
        vis_info_scroll_frame = ttk.Frame(vis_info_outer_frame, padding=(15, 10));
        vis_info_scroll_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(vis_info_scroll_frame, text="Stare Algoritm", style="Title.TLabel").pack(anchor=tk.NW, pady=(0, 10))
        self.vis_label_algo = ttk.Label(vis_info_scroll_frame, text="Algoritm: -");
        self.vis_label_algo.pack(anchor=tk.NW, pady=2)
        self.vis_label_step = ttk.Label(vis_info_scroll_frame, text="Pas: 0");
        self.vis_label_step.pack(anchor=tk.NW, pady=2)
        self.vis_label_action = ttk.Label(vis_info_scroll_frame, text="Acțiune: -");
        self.vis_label_action.pack(anchor=tk.NW, pady=2)
        self.vis_label_fw_kij = ttk.Label(vis_info_scroll_frame, text="k,i,j: -");  # For Floyd
        self.vis_label_fw_kij.pack(anchor=tk.NW, pady=2)
        self.vis_label_message = ttk.Label(vis_info_scroll_frame, text="", foreground="yellow",
                                           wraplength=INFO_PANEL_WIDTH - 30);
        self.vis_label_message.pack(anchor=tk.NW, pady=2)
        self.vis_label_queue = ttk.Label(vis_info_scroll_frame, text="Coadă: []", wraplength=INFO_PANEL_WIDTH - 30);
        self.vis_label_queue.pack(anchor=tk.NW, pady=2)
        self.vis_label_stack = ttk.Label(vis_info_scroll_frame, text="Stivă: []", wraplength=INFO_PANEL_WIDTH - 30);
        self.vis_label_stack.pack(anchor=tk.NW, pady=2)
        self.vis_label_edge_info = ttk.Label(vis_info_scroll_frame, text="Muchie curentă: -");
        self.vis_label_edge_info.pack(anchor=tk.NW, pady=2)
        ttk.Separator(vis_info_scroll_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(vis_info_scroll_frame, text="Rezultat & Analiză", style="Title.TLabel").pack(anchor=tk.NW,
                                                                                               pady=(5, 10))
        self.vis_label_exec_time = ttk.Label(vis_info_scroll_frame, text="Timp execuție: -");
        self.vis_label_exec_time.pack(anchor=tk.NW, pady=2)
        self.vis_label_result = ttk.Label(vis_info_scroll_frame, text="Rezultat: -", wraplength=INFO_PANEL_WIDTH - 30,
                                          justify=tk.LEFT);
        self.vis_label_result.pack(anchor=tk.NW, pady=2)
        vis_pw.add(vis_info_outer_frame, weight=1)
        self._toggle_start_node_for_visualizer()  # Initial check

    def _toggle_start_node_for_visualizer(self, *args):
        algo = self.vis_alg_var.get()
        if algo == "Floyd-Warshall" or algo == "Kruskal":
            self.vis_start_node_label.pack_forget()
            self.vis_start_spin.pack_forget()
        else:
            self.vis_start_node_label.pack(side=tk.LEFT, padx=(0, 5))  # Re-pack if forgotten
            self.vis_start_spin.pack(side=tk.LEFT, padx=(0, 15))

    def _update_vis_start_node_max(self, *args):  # As before
        try:
            max_node_val = self.vis_size_var.get() - 1;
            max_node_val = max(0, max_node_val)
            self.vis_start_spin.config(to=max_node_val)
            if self.vis_start_var.get() > max_node_val: self.vis_start_var.set(max_node_val)
        except tk.TclError:
            pass

    def generate_graph_for_visualizer(self):  # As before
        if self.running_visualization: return
        n = self.vis_size_var.get()
        self.G = create_connected_graph(n, seed=random.randint(0, 10000))
        self.pos = nx.spring_layout(self.G, seed=42, k=0.7 / ((n) ** 0.5) if n > 1 else 0.7,
                                    iterations=70 if n < 30 else 50) if n > 0 else {}
        self._update_vis_start_node_max()
        self.vis_start_var.set(min(self.vis_start_var.get(), n - 1 if n > 0 else 0))
        self.state = VisualState(edge_colors={tuple(sorted((u, v))): EDGE_COLOR for u, v in self.G.edges()})
        self.vis_btn_run.config(state=tk.NORMAL if self.G and self.G.number_of_nodes() > 0 else tk.DISABLED)
        self.vis_label_message.config(text="");
        self.vis_label_exec_time.config(text="Timp execuție: -")
        self.vis_label_result.config(text="Rezultat: -")
        self._draw_visualizer_graph()

    def run_visualization(self):  # Modified for Floyd-Warshall
        if not self.G or self.running_visualization or not self.G.nodes():
            if not self.G or not self.G.nodes(): self.state.current_message = "Generați un graf întâi!"; self._update_visualizer_info_panel()
            return
        start_node_vis = self.vis_start_var.get();
        algo_name_vis = self.vis_alg_var.get();
        gen_func = _ALGOS[algo_name_vis]
        self.state = VisualState(edge_colors={tuple(sorted((u, v))): EDGE_COLOR for u, v in self.G.edges()})

        if algo_name_vis in ("Prim", "BFS", "DFS", "Dijkstra"):
            if start_node_vis >= self.G.number_of_nodes() or start_node_vis < 0:
                self.state.current_message = f"Nod start {start_node_vis} invalid.";
                self._update_visualizer_info_panel();
                return
            self.step_iter = gen_func(self.G, start_node_vis)
        elif algo_name_vis in ("Kruskal", "Floyd-Warshall"):
            self.step_iter = gen_func(self.G)
        else:  # Should not happen
            self.state.current_message = "Algoritm necunoscut.";
            self._update_visualizer_info_panel();
            return

        self.running_visualization = True;
        self.visualization_start_time = time.perf_counter()
        self.vis_btn_gen.config(state=tk.DISABLED);
        self.vis_btn_run.config(state=tk.DISABLED)
        self.vis_alg_combo.config(state=tk.DISABLED);
        self.vis_size_spin.config(state=tk.DISABLED)
        if algo_name_vis not in ("Kruskal", "Floyd-Warshall"): self.vis_start_spin.config(state=tk.DISABLED)
        self.state.current_message = ""
        self._advance_visualization()

    def _advance_visualization(self):  # Modified for Floyd-Warshall events
        if not self.running_visualization or not self.step_iter: return
        try:
            typ, val = next(self.step_iter)
            self.state.step += 1;
            self.state.last_action = typ
            self.state.highlighted_edge_temp = None;
            self.state.highlighted_edge_skip = None
            # Reset Floyd-Warshall specific highlights each step unless overwritten
            self.state.fw_k_node = None;
            self.state.fw_i_node = None;
            self.state.fw_j_node = None

            if typ == "queue":
                self.state.queue = list(val)
            elif typ == "stack":
                self.state.stack = list(val)
            elif typ == "visit":
                self.state.visited.add(val)
            elif typ == "edge":
                self.state.solution_edges.add(tuple(sorted(val)))
            elif typ == "path_edges":
                self.state.solution_edges.update(val)
            elif typ == "highlight_edge_temp":
                self.state.highlighted_edge_temp = tuple(sorted(val))
            elif typ == "highlight_edge_skip":
                self.state.highlighted_edge_skip = tuple(sorted(val))
            elif typ == "edge_candidates":
                self.state.candidate_edges = {tuple(sorted(e)) for e in val}
            # Floyd-Warshall specific events
            elif typ == "k_iteration":
                self.state.fw_k_node = val["k_node"]
                self.state.visited.clear()  # Clear general visited for FW iteration focus
            elif typ == "check_path":
                self.state.fw_k_node = val["k_node"]
                self.state.fw_i_node = val["i_node"]
                self.state.fw_j_node = val["j_node"]
            elif typ == "path_updated":
                # Could add temporary edge highlight for i-k, k-j path here
                self.state.fw_i_node = val["i_node"];
                self.state.fw_j_node = val["j_node"]
                # Path update itself is reflected in matrix, graph highlights i,j,k
            elif typ == "matrix_state":  # For Floyd-Warshall
                self.state.fw_dist_matrix_display = val["dist"]  # Store for potential display
                self.state.fw_node_map = val["node_map"]
            elif typ == "negative_cycle_detected":
                self.state.current_message = f"Ciclu negativ detectat la nodul {val['node']}!"
            elif typ == "result":
                self.state.result = val;
                self.running_visualization = False
            elif typ == "error":
                self.state.current_message = str(val);
                self.running_visualization = False

            self._draw_visualizer_graph()
            if self.running_visualization:
                self.root.after(self.vis_delay_var.get(), self._advance_visualization)
            else:
                self._finalize_visualization_run()
        except StopIteration:
            self._finalize_visualization_run()
        except Exception as e:
            self.state.current_message = f"Eroare: {e}"; self._finalize_visualization_run(error_occurred=True)

    def _finalize_visualization_run(self, error_occurred=False):  # As before
        self.running_visualization = False;
        end_time = time.perf_counter()
        self.state.execution_time_ms = (end_time - self.visualization_start_time) * 1000
        self.vis_btn_gen.config(state=tk.NORMAL)
        self.vis_btn_run.config(state=tk.NORMAL if self.G and self.G.number_of_nodes() > 0 else tk.DISABLED)
        self.vis_alg_combo.config(state="readonly");
        self.vis_size_spin.config(state=tk.NORMAL)
        if self.vis_alg_var.get() not in ("Kruskal", "Floyd-Warshall"): self.vis_start_spin.config(state=tk.NORMAL)
        self.state.last_action = "finalizat" if not error_occurred else "eroare"
        self.state.highlighted_edge_temp = None;
        self.state.highlighted_edge_skip = None
        self._draw_visualizer_graph(final=True)

    def _node_color_visualizer(self, v: int) -> str:  # Modified for Floyd-Warshall highlights
        if not self.state: return NODE_COLOR
        # Floyd-Warshall specific highlighting takes precedence
        if self.vis_alg_var.get() == "Floyd-Warshall":
            if v == self.state.fw_k_node: return VISITED_COLOR  # K node
            if v == self.state.fw_i_node: return CURRENT_COLOR  # I node
            if v == self.state.fw_j_node: return QUEUE_COLOR  # J node
            # Fall through to general visited if no specific FW role

        if v in self.state.visited: return VISITED_COLOR
        if v in self.state.queue: return QUEUE_COLOR
        if v in self.state.stack: return STACK_COLOR
        return NODE_COLOR

    def _edge_color_and_width_visualizer(self, u: int, v: int) -> Tuple[str, float, int]:  # As before
        edge = tuple(sorted((u, v)));
        zorder, width = 1, 1.5
        if self.state.highlighted_edge_temp == edge: return HIGHLIGHT_EDGE_COLOR, 3.0, 3
        if self.state.highlighted_edge_skip == edge: return STACK_COLOR, 2.0, 2
        if edge in self.state.solution_edges:
            algo = self.vis_alg_var.get()
            return PATH_EDGE_COLOR if algo == "Dijkstra" else ACTIVE_EDGE_COLOR, 2.5, 4
        if edge in self.state.candidate_edges: return VISITED_COLOR, 1.5, 0
        return self.state.edge_colors.get(edge, EDGE_COLOR), width, zorder

    def _draw_visualizer_graph(self, final: bool = False):  # As before
        self.vis_ax.clear();
        self.vis_ax.set_facecolor(BACKGROUND_COLOR);
        self.vis_ax.axis("off")
        if not self.G or not self.pos: self.vis_canvas.draw_idle(); return
        for u, v_neighbor in self.G.edges():
            color, width, z = self._edge_color_and_width_visualizer(u, v_neighbor)
            self.vis_ax.plot(
                [self.pos[u][0], self.pos[v_neighbor][0]], [self.pos[u][1], self.pos[v_neighbor][1]],
                color=color, linewidth=width, zorder=z, alpha=0.7 if z <= 1 else 1.0
            )
            weight = self.G.edges[u, v_neighbor].get("weight")
            if weight is not None:
                mid_x = (self.pos[u][0] + self.pos[v_neighbor][0]) / 2;
                mid_y = (self.pos[u][1] + self.pos[v_neighbor][1]) / 2
                self.vis_ax.text(mid_x, mid_y, str(weight), color="#bbbbbb", fontsize=8, fontfamily=FONT_FAMILY,
                                 ha='center', va='center', zorder=5,
                                 bbox=dict(facecolor=BACKGROUND_COLOR, alpha=0.6, pad=1, edgecolor='none'))
        node_colors = [self._node_color_visualizer(node) for node in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.vis_ax, node_color=node_colors, node_size=NODE_SIZE,
                               edgecolors="#dddddd", linewidths=0.5, alpha=0.9)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.vis_ax, font_size=10, font_color=TEXT_COLOR,
                                font_family=FONT_FAMILY, font_weight='bold')

        handles = [
            mpatches.Patch(color=NODE_COLOR, label="Nod inactiv"),
            mpatches.Patch(color=VISITED_COLOR, label="Vizitat/K-Nod(FW)"),  # Updated legend for FW
            mpatches.Patch(color=CURRENT_COLOR, label="Curent/I-Nod(FW)"),  # Updated legend for FW
            mpatches.Patch(color=QUEUE_COLOR, label="Coadă/J-Nod(FW)"),  # Updated legend for FW
            mpatches.Patch(color=STACK_COLOR, label="Stivă/Sărit"),
            mpatches.Patch(color=ACTIVE_EDGE_COLOR, label="Muchie Soluție (MST)"),
            mpatches.Patch(color=PATH_EDGE_COLOR, label="Muchie Cale (Dijkstra)"),
        ]
        self.vis_ax.legend(handles=handles, loc="lower left", fontsize="small", facecolor="#444444",
                           labelcolor=TEXT_COLOR, framealpha=0.8, ncol=2)
        self.vis_canvas.draw_idle()
        self._update_visualizer_info_panel(final=final)

    def _update_visualizer_info_panel(self, final: bool = False):  # Modified for Floyd-Warshall
        if not self.state: return
        self.vis_label_algo.config(text=f"Algoritm: {self.vis_alg_var.get()}")
        self.vis_label_step.config(text=f"Pas: {self.state.step}")
        self.vis_label_action.config(text=f"Acțiune: {self.state.last_action}")
        self.vis_label_message.config(text=self.state.current_message)

        if self.vis_alg_var.get() == "Floyd-Warshall":
            k_str = f"k={self.state.fw_k_node}" if self.state.fw_k_node is not None else "k=-"
            i_str = f"i={self.state.fw_i_node}" if self.state.fw_i_node is not None else "i=-"
            j_str = f"j={self.state.fw_j_node}" if self.state.fw_j_node is not None else "j=-"
            self.vis_label_fw_kij.config(text=f"{k_str}, {i_str}, {j_str}")
            self.vis_label_fw_kij.pack(anchor=tk.NW, pady=2)  # Ensure visible
            self.vis_label_queue.pack_forget();
            self.vis_label_stack.pack_forget()  # Hide Q/S
        else:
            self.vis_label_fw_kij.pack_forget()  # Hide FW info
            q_text = f"Coadă: {self.state.queue if len(self.state.queue) < 20 else str(self.state.queue[:18]) + '...'}"
            s_text = f"Stivă: {self.state.stack if len(self.state.stack) < 20 else str(self.state.stack[:18]) + '...'}"
            self.vis_label_queue.config(text=q_text);
            self.vis_label_stack.config(text=s_text)
            self.vis_label_queue.pack(anchor=tk.NW, pady=2);
            self.vis_label_stack.pack(anchor=tk.NW, pady=2)

        edge_text = "-"
        if self.state.highlighted_edge_temp:
            edge_text = f"Consideră: {self.state.highlighted_edge_temp}"
        elif self.state.highlighted_edge_skip:
            edge_text = f"Sare: {self.state.highlighted_edge_skip}"
        self.vis_label_edge_info.config(text=f"Muchie: {edge_text}")

        if final or self.state.execution_time_ms is not None:
            time_str = f"{self.state.execution_time_ms:.2f} ms" if self.state.execution_time_ms is not None else "-"
            self.vis_label_exec_time.config(text=f"Timp execuție: {time_str}")
            self._show_visualizer_result()

    def _show_visualizer_result(self):  # Modified for Floyd-Warshall
        if self.state.result is None and not self.state.current_message: self.vis_label_result.config(
            text="Rezultat: Algoritmul nu a produs un rezultat final."); return
        if self.state.current_message and "eroare" in self.state.last_action.lower(): self.vis_label_result.config(
            text=f"Rezultat: EROARE - {self.state.current_message}"); return

        algo = self.vis_alg_var.get();
        result_data = self.state.result;
        result_text = "Rezultat:\n"
        if algo in ("BFS", "DFS"):
            order = result_data if isinstance(result_data, list) else [];
            result_text += f"Ordine vizitare: {order}"
        elif algo == "Dijkstra":
            if isinstance(result_data, dict) and "distances" in result_data:
                dist = result_data["distances"];
                dist_text = ", ".join(f"{k}: {int(v) if v != inf else '∞'}" for k, v in sorted(dist.items()))
                result_text += f"Distanțe de la {self.vis_start_var.get()} → {dist_text}"
            else:
                result_text += "Structură rezultat Dijkstra neașteptată."
        elif algo in ("Prim", "Kruskal"):
            if isinstance(result_data, list):
                mst_edges_list = result_data;
                total_w = 0
                if self.G:
                    try:
                        total_w = sum(self.G.edges[u, v]["weight"] for u, v in mst_edges_list)
                    except KeyError:
                        result_text += "Eroare calcul cost MST."; self.vis_label_result.config(text=result_text); return
                result_text += f"MST (cost {total_w}):\n{mst_edges_list}"
            else:
                result_text += "Structură rezultat MST neașteptată."
        elif algo == "Floyd-Warshall":
            if isinstance(result_data, dict) and "distances" in result_data and "node_map" in result_data:
                dist_matrix_dict = result_data["distances"]
                # node_map_fw = result_data["node_map"] # idx -> node_label
                # For brevity, show a small part or summary.
                # Example: Distances from node 0 (if exists)
                result_text += "Matrice Distanțe (All-Pairs Shortest Paths):\n"
                if not dist_matrix_dict:
                    result_text += "Matrice goală."
                else:
                    # Display first few rows/cols or specific queries for brevity in UI
                    # Example: show distances from first node in the map
                    first_node_key = list(dist_matrix_dict.keys())[0] if dist_matrix_dict else None
                    if first_node_key is not None:
                        result_text += f"De la nodul {first_node_key}:\n"
                        for target_node, d_val in sorted(dist_matrix_dict[first_node_key].items()):
                            result_text += f"  → {target_node}: {int(d_val) if d_val != inf else '∞'}\n"
                        if len(dist_matrix_dict) > 1: result_text += "(și alte surse...)"
                    else:
                        result_text += "Nu s-au putut afișa distanțele."


            else:
                result_text += "Structură rezultat Floyd-Warshall neașteptată."

        else:
            result_text += str(result_data)
        self.vis_label_result.config(text=result_text)

    # --- Metode pentru Tab-ul Comparator (modificate pentru Floyd-Warshall) ---
    def _init_vars_comparator(self):  # As before
        self.comp_min_nodes_var = tk.IntVar(value=5)
        self.comp_max_nodes_var = tk.IntVar(value=20)  # Reduced max for Floyd in comparator
        self.comp_step_nodes_var = tk.IntVar(value=5)
        self.comp_num_runs_var = tk.IntVar(value=1)
        self.comp_selected_algos_vars: Dict[str, tk.BooleanVar] = {name: tk.BooleanVar(value=True) for name in
                                                                   _ALGOS.keys()}
        self.comp_status_var = tk.StringVar(value="Gata pentru comparație.")

    def _build_comparator_tab(self):  # As before, checkbuttons will include Floyd
        main_frame = ttk.Frame(self.comparator_tab, padding=10);
        main_frame.pack(fill=tk.BOTH, expand=True)
        controls_lf = ttk.LabelFrame(main_frame, text=" Setări Comparație ", padding=10);
        controls_lf.pack(fill=tk.X, pady=(0, 10))
        node_settings_frame = ttk.Frame(controls_lf);
        node_settings_frame.pack(fill=tk.X, pady=5)
        ttk.Label(node_settings_frame, text="Min Noduri:").pack(side=tk.LEFT, padx=5);
        ttk.Spinbox(node_settings_frame, from_=3, to=50, textvariable=self.comp_min_nodes_var, width=5).pack(
            side=tk.LEFT, padx=5)  # Max 50 for non-FW
        ttk.Label(node_settings_frame, text="Max Noduri:").pack(side=tk.LEFT, padx=5);
        ttk.Spinbox(node_settings_frame, from_=5, to=100, textvariable=self.comp_max_nodes_var, width=5).pack(
            side=tk.LEFT, padx=5)  # Max 100
        ttk.Label(node_settings_frame, text="Pas:").pack(side=tk.LEFT, padx=5);
        ttk.Spinbox(node_settings_frame, from_=1, to=50, textvariable=self.comp_step_nodes_var, width=5).pack(
            side=tk.LEFT, padx=5)
        ttk.Label(node_settings_frame, text="Rulări/dim:").pack(side=tk.LEFT, padx=5);
        ttk.Spinbox(node_settings_frame, from_=1, to=10, textvariable=self.comp_num_runs_var, width=5).pack(
            side=tk.LEFT, padx=5)

        algo_select_lf = ttk.LabelFrame(main_frame, text=" Selectează Algoritmi ", padding=10);
        algo_select_lf.pack(fill=tk.X, pady=10)
        col1 = ttk.Frame(algo_select_lf);
        col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        col2 = ttk.Frame(algo_select_lf);
        col2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        col3 = ttk.Frame(algo_select_lf);
        col3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)  # Third column if many algos

        algo_keys = list(_ALGOS.keys())
        items_per_col = (len(algo_keys) + 2) // 3  # Distribute to 3 columns

        for i, algo_name in enumerate(algo_keys):
            parent_col = col1
            if i >= items_per_col * 2:
                parent_col = col3
            elif i >= items_per_col:
                parent_col = col2
            cb = ttk.Checkbutton(parent_col, text=algo_name, variable=self.comp_selected_algos_vars[algo_name])
            cb.pack(anchor=tk.W, padx=5, pady=2)

        run_status_frame = ttk.Frame(main_frame);
        run_status_frame.pack(fill=tk.X, pady=10)
        self.comp_btn_run = ttk.Button(run_status_frame, text="Rulează Comparația",
                                       command=self._start_comparison_thread);
        self.comp_btn_run.pack(side=tk.LEFT, padx=5)
        self.comp_progressbar = ttk.Progressbar(run_status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate');
        self.comp_progressbar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.comp_status_label = ttk.Label(main_frame, textvariable=self.comp_status_var, style="Status.TLabel");
        self.comp_status_label.pack(fill=tk.X, pady=(5, 0))

        results_pw = ttk.Panedwindow(main_frame, orient=tk.VERTICAL, style="Vertical.TPanedwindow");
        results_pw.pack(fill=tk.BOTH, expand=True, pady=10)
        plot_frame = ttk.Frame(results_pw, padding=0)
        self.comp_fig = Figure(figsize=(7, 5), facecolor=BACKGROUND_COLOR, constrained_layout=True)
        self.comp_ax = self.comp_fig.add_subplot(111);
        self.comp_ax.set_facecolor(BACKGROUND_COLOR)
        self.comp_ax.tick_params(axis='x', colors=TEXT_COLOR, labelcolor=TEXT_COLOR);
        self.comp_ax.tick_params(axis='y', colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        self.comp_ax.spines['bottom'].set_color(TEXT_COLOR);
        self.comp_ax.spines['left'].set_color(TEXT_COLOR)
        self.comp_ax.spines['top'].set_color(BACKGROUND_COLOR);
        self.comp_ax.spines['right'].set_color(BACKGROUND_COLOR)
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, master=plot_frame);
        self.comp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        results_pw.add(plot_frame, weight=2)
        table_frame = ttk.Frame(results_pw);
        columns = ("algoritm", "noduri", "timp_ms")
        self.comp_table = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
        self.comp_table.heading("algoritm", text="Algoritm");
        self.comp_table.heading("noduri", text="Nr. Noduri");
        self.comp_table.heading("timp_ms", text="Timp (ms)")
        self.comp_table.column("algoritm", width=120, anchor=tk.W);
        self.comp_table.column("noduri", width=80, anchor=tk.CENTER);
        self.comp_table.column("timp_ms", width=100, anchor=tk.E)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.comp_table.yview);
        self.comp_table.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y);
        self.comp_table.pack(fill=tk.BOTH, expand=True)
        results_pw.add(table_frame, weight=1)

    def _run_single_algorithm_for_timing(self, algo_key: str, G: nx.Graph, start_node_val: Optional[int] = 0) -> Tuple[
        Optional[float], Any]:  # Modified for Floyd
        gen_func = _ALGOS[algo_key];
        current_start_node = start_node_val
        if algo_key in ("BFS", "DFS", "Dijkstra", "Prim"):  # These need a start node
            if not G.nodes(): return None, "Graph has no nodes"
            if current_start_node is None or current_start_node not in G.nodes():
                current_start_node = list(G.nodes())[0] if G.nodes() else 0

        iterator: Optional[Generator] = None
        if algo_key in ("BFS", "DFS", "Dijkstra", "Prim"):
            iterator = gen_func(G, current_start_node)
        elif algo_key in ("Kruskal", "Floyd-Warshall"):
            iterator = gen_func(G)  # No start node
        else:
            return None, f"Unknown algorithm key: {algo_key}"

        start_time = time.perf_counter();
        final_result = None
        try:
            while iterator:
                event_type, event_data = next(iterator)
                if event_type == "result":
                    final_result = event_data; break
                elif event_type == "error":
                    return None, f"Error: {event_data}"
        except StopIteration:
            pass
        except Exception as e:
            return None, f"Exception: {e}"
        end_time = time.perf_counter();
        exec_time_ms = (end_time - start_time) * 1000
        return exec_time_ms, final_result

    def _comparison_task(self):  # As before
        min_n = self.comp_min_nodes_var.get();
        max_n = self.comp_max_nodes_var.get()
        step_n = self.comp_step_nodes_var.get();
        num_runs = self.comp_num_runs_var.get()
        if min_n > max_n: self.comp_status_var.set("Eroare: Min Noduri > Max Noduri."); self.root.after(10,
                                                                                                        self._comparison_finished); return
        if step_n <= 0: self.comp_status_var.set("Eroare: Pasul trebuie să fie > 0."); self.root.after(10,
                                                                                                       self._comparison_finished); return
        selected_algos = [name for name, var in self.comp_selected_algos_vars.items() if var.get()]
        if not selected_algos: self.comp_status_var.set("Selectați cel puțin un algoritm."); self.root.after(10,
                                                                                                             self._comparison_finished); return
        self.comparison_results_data.clear()
        for item in self.comp_table.get_children(): self.comp_table.delete(item)
        node_counts = list(range(min_n, max_n + 1, step_n));
        total_steps = len(node_counts) * len(selected_algos) * num_runs
        current_step = 0;
        self.comp_progressbar['maximum'] = total_steps;
        self.comp_progressbar['value'] = 0
        for n_nodes in node_counts:
            self.comp_status_var.set(f"Procesare grafuri cu {n_nodes} noduri...")
            for algo_name in selected_algos:
                times_for_current_config = []
                # Reduce max_n for Floyd-Warshall if it's selected to prevent very long runs
                if algo_name == "Floyd-Warshall" and n_nodes > 30:  # Heuristic limit for comparator speed
                    self.comp_status_var.set(f"Floyd-Warshall sărit pentru N={n_nodes} (prea mare).")
                    # Still need to increment progress bar as if it ran
                    current_step += num_runs
                    self.comp_progressbar['value'] = min(current_step, total_steps)
                    self.root.update_idletasks()
                    continue

                for run_idx in range(num_runs):
                    graph_seed = hash((n_nodes, algo_name, run_idx));
                    G_comp = create_connected_graph(n_nodes, seed=graph_seed)
                    start_node_comp = 0 if G_comp.has_node(0) else (list(G_comp.nodes())[0] if G_comp.nodes() else 0)
                    time_taken, _ = self._run_single_algorithm_for_timing(algo_name, G_comp, start_node_comp)
                    if time_taken is not None: times_for_current_config.append(time_taken)
                    current_step += 1;
                    self.comp_progressbar['value'] = min(current_step, total_steps)
                    self.root.update_idletasks()
                if times_for_current_config:
                    avg_time = sum(times_for_current_config) / len(times_for_current_config)
                    if algo_name not in self.comparison_results_data: self.comparison_results_data[algo_name] = []
                    self.comparison_results_data[algo_name].append((n_nodes, avg_time))
                    self.comp_table.insert("", tk.END, values=(algo_name, n_nodes, f"{avg_time:.3f}"))
        self.root.after(10, self._plot_comparison_results);
        self.root.after(20, self._comparison_finished)

    def _start_comparison_thread(self):  # As before
        if self.comparison_thread and self.comparison_thread.is_alive(): messagebox.showwarning("Comparație în Curs",
                                                                                                "O altă comparație este deja în desfășurare."); return
        self.comp_btn_run.config(state=tk.DISABLED)
        for child in self.comparator_tab.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.Frame):
                        for item in grandchild.winfo_children():
                            if hasattr(item, 'config'): item.config(state=tk.DISABLED)
                    elif hasattr(grandchild, 'config'):
                        grandchild.config(state=tk.DISABLED)
        self.comp_status_var.set("Rulează comparația...");
        self.comparison_thread = threading.Thread(target=self._comparison_task, daemon=True);
        self.comparison_thread.start()

    def _comparison_finished(self):  # As before
        self.comp_status_var.set("Comparație finalizată.");
        self.comp_btn_run.config(state=tk.NORMAL)
        for child in self.comparator_tab.winfo_children():  # Re-enable controls
            if isinstance(child, ttk.LabelFrame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.Frame):
                        for item in grandchild.winfo_children():
                            if hasattr(item, 'config'): item.config(state=tk.NORMAL)
                    elif hasattr(grandchild, 'config'):
                        grandchild.config(state=tk.NORMAL)

    def _plot_comparison_results(self):  # As before
        self.comp_ax.clear();
        self.comp_ax.set_facecolor(BACKGROUND_COLOR)
        self.comp_ax.tick_params(axis='x', colors=TEXT_COLOR, labelcolor=TEXT_COLOR);
        self.comp_ax.tick_params(axis='y', colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        self.comp_ax.spines['bottom'].set_color(TEXT_COLOR);
        self.comp_ax.spines['left'].set_color(TEXT_COLOR)
        self.comp_ax.spines['top'].set_color(BACKGROUND_COLOR);
        self.comp_ax.spines['right'].set_color(BACKGROUND_COLOR)
        self.comp_ax.set_xlabel("Număr de Noduri", color=TEXT_COLOR);
        self.comp_ax.set_ylabel("Timp Mediu (ms)", color=TEXT_COLOR)
        self.comp_ax.set_title("Comparație Performanță Algoritmi", color=TEXT_COLOR)
        if not self.comparison_results_data: self.comp_ax.text(0.5, 0.5, "Nicio dată.", color=TEXT_COLOR, ha='center',
                                                               va='center',
                                                               transform=self.comp_ax.transAxes); self.comp_canvas.draw(); return
        for i, (algo_name, data_points) in enumerate(self.comparison_results_data.items()):
            if data_points:
                nodes, times = zip(*sorted(data_points))
                self.comp_ax.plot(nodes, times, marker='o', linestyle='-', label=algo_name,
                                  color=PLOT_COLORS[i % len(PLOT_COLORS)])
        legend = self.comp_ax.legend(facecolor="#333333", edgecolor="#555555", labelcolor=TEXT_COLOR)
        for text in legend.get_texts(): text.set_color(TEXT_COLOR)
        self.comp_canvas.draw()


if __name__ == "__main__":
    app = GraphVisualizer()