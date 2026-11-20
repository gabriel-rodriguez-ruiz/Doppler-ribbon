# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:25:40 2026

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

data_dir = Path("Data")
files = sorted(data_dir.glob("*.npz"))

# Opcional: si solo quieres ciertos valores de N_x_sites, ponlos aquí (o déjalo en None para todos los que tengan N_x_sites)
ALLOWED_NX = None  # p. ej. {100, 200, 300}

grouped = defaultdict(list)

for f in files:
    with np.load(f, allow_pickle=True) as d:
        if "is_merged" in d:
            continue

        # Excluir archivos que no tengan N_x_sites explícito
        if "N_x_sites" not in d:
            continue

        nx = int(np.asarray(d["N_x_sites"]).item())
        if ALLOWED_NX is not None and nx not in ALLOWED_NX:
            continue

        if "B_value" in d:
            B_arr = np.atleast_1d(d["B_value"]).astype(float)
        elif "B_values" in d:
            B_arr = np.atleast_1d(d["B_values"]).astype(float)
        else:
            continue

        nyy_arr = np.atleast_1d(d["n_yy"]).astype(float)
        E0_arr = np.atleast_1d(d["E0"]).astype(float)
        delta = float(np.asarray(d["Delta"]).item()) if "Delta" in d else 1.0

        for b_val, nyy_val, e0_val in zip(B_arr, nyy_arr, E0_arr):
            grouped[nx].append((b_val, nyy_val, e0_val, delta))

if not grouped:
    raise RuntimeError("No se encontraron archivos .npz con 'N_x_sites' válido en Data/.")

fig, ax = plt.subplots(1, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

for nx in sorted(grouped.keys()):
    records = np.array(grouped[nx], dtype=float)
    records = records[np.argsort(records[:, 0])]

    _, unique_idx = np.unique(np.round(records[:, 0], decimals=12), return_index=True)
    records = records[unique_idx]

    B_vals = records[:, 0]
    nyy_vals = records[:, 1]
    E0_vals = records[:, 2]
    delta = records[0, 3]
    B_norm = B_vals / delta

    # Con pocos puntos en B (p. ej. solo B = 2*Delta), usar marcador sin línea o línea tenue ayuda a leer mejor
    linestyle = "-" if len(B_vals) > 1 else "None"
    ax.plot(B_norm, nyy_vals, marker="o", markersize=6, linestyle=linestyle, linewidth=1.5, label=f"$N_x = {nx}$")

    np.savez_compressed(
        data_dir / f"merged_stiffness_Nx={nx}.npz",
        B_values=B_vals,
        n_yy=nyy_vals,
        E0=E0_vals,
        N_x_sites=nx,
        Delta=delta,
        is_merged=True,
    )

ax.set_ylabel(r"$n_{yy}$")
ax.grid(True, alpha=0.3)
ax.legend()



plt.savefig(data_dir / "stiffness_vs_B_nx_only.png", dpi=300)
plt.show()