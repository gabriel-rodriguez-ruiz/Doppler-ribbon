# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:19:24 2026

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

data_dir = Path("Data")
files = sorted(data_dir.glob("*.npz"))

# Opcional: filtra por valores específicos si lo deseas (o déjalo en None para incluir todos)
ALLOWED_LX = None       # p. ej. {750.0, 1500.0, 2250.0}
ALLOWED_NX = None       # p. ej. {100, 200, 300}

# Agrupa por clave (L_x, N_x_sites)
grouped = defaultdict(list)

for f in files:
    with np.load(f, allow_pickle=True) as d:
        if "is_merged" in d:
            continue
        if "N_x_sites" not in d or "L_x" not in d:
            continue

        nx = int(np.asarray(d["N_x_sites"]).item())
        lx = float(np.asarray(d["L_x"]).item())

        if ALLOWED_LX is not None and lx not in ALLOWED_LX:
            continue
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
            grouped[(lx, nx)].append((b_val, nyy_val, e0_val, delta))

if not grouped:
    raise RuntimeError("No se encontraron archivos .npz con 'L_x' y 'N_x_sites' válidos en Data/.")

fig, ax = plt.subplots(1, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)

for lx, nx in sorted(grouped.keys()):
    records = np.array(grouped[(lx, nx)], dtype=float)
    records = records[np.argsort(records[:, 0])]

    _, unique_idx = np.unique(np.round(records[:, 0], decimals=12), return_index=True)
    records = records[unique_idx]

    B_vals = records[:, 0]
    nyy_vals = records[:, 1]
    E0_vals = records[:, 2]
    delta = records[0, 3]
    B_norm = B_vals / delta
    a_x = lx / (nx + 1)

    linestyle = "-" if len(B_vals) > 1 else "None"
    label = rf"$L_x = {lx:.0f}\text{{ nm}},\ N_x = {nx}\ (a_x = {a_x:.2f}\text{{ nm}})$"

    ax.plot(B_norm, nyy_vals, marker="o", markersize=6, linestyle=linestyle, linewidth=1.5, label=label)

    np.savez_compressed(
        data_dir / f"merged_stiffness_Lx={lx:.0f}_Nx={nx}.npz",
        B_values=B_vals,
        n_yy=nyy_vals,
        E0=E0_vals,
        L_x=lx,
        N_x_sites=nx,
        a_x=a_x,
        Delta=delta,
        is_merged=True,
    )

ax.set_ylabel(r"$n_{yy}$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax.set_xlabel(r"$B / \Delta$")


plt.savefig(data_dir / "stiffness_vs_B_by_Lx_Nx.png", dpi=300)

data = np.load("Data/compare_2d_vs_obc_Lambda=15.0_Lx=500.0_Nx=200.npz")
nyy_2d = data["nyy_2d"]
B_values = data["B_values"]
ax.plot(B_values/0.4, nyy_2d*2*np.pi)

plt.show()
