# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:21:25 2026

@author: Gabriel
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import scipy.sparse as sp
import multiprocessing
from threadpoolctl import threadpool_limits

# ==============================================================================
# 1. MATRICES DE PAULI Y DE NAMBU
# ==============================================================================
tau_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
tau_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
tau_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

sigma_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)

# ==============================================================================
# 2. PARÁMETROS (LAMBDA = 0, B = 0, L_x = 750 nm)
# ==============================================================================
c = 3e17                      # nm/s
m_e = 5.1e8 / c**2            # meV s²/(nm)²
m = 0.0403 * m_e              # meV s²/(nm)²
hbar = 6.58e-13               # meV s
gamma = hbar**2 / (2 * m)     # meV (nm)²
E_F = 50.6                    # meV
k_F = np.sqrt(E_F / gamma)    # 1/nm

Delta = 0.4                   # meV
mu = E_F                      # meV
Lambda = 0.0                  # Sin acoplamiento Rashba
B_val = 0.0                   # Campo cero
theta = np.pi / 2
phi_y = 0.0
h = 2.5e-4
T = True
beta = 125.0
Zeeman = True
Z_RATIO = 1.0
q_B_constant = 0.0

L_x = 750.0                   # nm
NX_LIST = [100, 300]
n_cores = min(8, os.cpu_count() or 1)

# Grilla compacta en k_y para prueba rápida
N_KY_INNER = 250
KY_CHUNK_SIZE = 64
cut_off = 2.0 * k_F

k_limit = min(k_F + 0.2 * k_F, 0.9 * cut_off)
k_y_half = np.unique(np.concatenate([
    np.linspace(0.0, k_limit, N_KY_INNER),
    np.linspace(k_limit, cut_off, 100)
]))
k_y_values = np.unique(np.concatenate([-k_y_half[::-1], k_y_half]))
W_KY = np.gradient(k_y_values)


def eval_thermodynamic_potential(eigvals):
    if T:
        stable_term = np.minimum(0.0, eigvals)
        log_term = -(1.0 / beta) * np.log1p(np.exp(-beta * np.abs(eigvals)))
        return stable_term + log_term
    return np.minimum(0.0, eigvals)


_STATE = {}


def init_worker(L_x_val, nx, bx, by, px):
    ax = L_x_val / (nx + 1)
    t_hop = gamma / (ax**2)
    t_R = Lambda / (2 * ax)

    P_e = (tau_0 + tau_z) / 2.0
    P_h = (tau_0 - tau_z) / 2.0
    I_nx = np.eye(nx, dtype=np.complex128)
    off = np.ones(nx - 1, dtype=np.complex128)

    kron_y = np.kron(np.kron(tau_0, sigma_y), I_nx)
    kron_x = np.kron(np.kron(tau_0, sigma_x), I_nx)
    kron_x0 = np.kron(np.kron(tau_x, sigma_0), I_nx)
    n_y = np.kron(sigma_y, I_nx)
    n_x = np.kron(sigma_x, I_nx)

    kin_e0 = sp.diags([-t_hop * off, -t_hop * off], [1, -1], shape=(nx, nx), format="csr")
    kin_h0 = sp.diags([+t_hop * off, +t_hop * off], [1, -1], shape=(nx, nx), format="csr")
    h_ne_base = sp.kron(sigma_0, kin_e0, format="csr").toarray()
    h_nh_base = sp.kron(sigma_0, kin_h0, format="csr").toarray()

    th_x = px * ax
    pp, pm = np.exp(1j * th_x), np.exp(-1j * th_x)
    kin_e = sp.diags([-t_hop * pp * off, -t_hop * pm * off], [1, -1], shape=(nx, nx), format="csr")
    kin_h = sp.diags([+t_hop * pm * off, +t_hop * pp * off], [1, -1], shape=(nx, nx), format="csr")
    h_kin = (
        sp.kron(np.kron(P_e, sigma_0), kin_e, format="csr")
        + sp.kron(np.kron(P_h, sigma_0), kin_h, format="csr")
    ).toarray()

    hx_const = h_kin - Delta * kron_x0 - by * kron_y - bx * kron_x
    he_const = h_ne_base - by * n_y - bx * n_x
    hh_const = h_nh_base - by * n_y - bx * n_x

    _STATE.clear()
    _STATE["L_x"] = L_x_val
    _STATE["nx"] = nx
    _STATE["t_hop"] = t_hop
    _STATE["t_R"] = t_R
    _STATE["hx_const"] = hx_const
    _STATE["he_const"] = he_const
    _STATE["hh_const"] = hh_const
    _STATE["idx_eu"] = np.arange(0, nx)
    _STATE["idx_ed"] = np.arange(nx, 2 * nx)
    _STATE["idx_hu"] = np.arange(2 * nx, 3 * nx)
    _STATE["idx_hd"] = np.arange(3 * nx, 4 * nx)


def eval_block(task):
    start_idx, end_idx, py_list = task
    with threadpool_limits(limits=1):
        L_x_val = _STATE["L_x"]
        nx = _STATE["nx"]
        t_hop = _STATE["t_hop"]
        t_R = _STATE["t_R"]
        hx_const = _STATE["hx_const"]
        he_const = _STATE["he_const"]
        hh_const = _STATE["hh_const"]
        idx_eu = _STATE["idx_eu"]
        idx_ed = _STATE["idx_ed"]
        idx_hu = _STATE["idx_hu"]
        idx_hd = _STATE["idx_hd"]

        ky_sub = k_y_values[start_idx:end_idx]
        w_sub = W_KY[start_idx:end_idx]
        n_sub = len(ky_sub)

        # Devuelve [E_sc, E_norm] para cada py en py_list
        out = np.zeros((len(py_list), 2), dtype=np.float64)

        for p_idx, py in enumerate(py_list):
            acc_sc = 0.0
            acc_nm = 0.0
            for s in range(0, n_sub, KY_CHUNK_SIZE):
                e = min(s + KY_CHUNK_SIZE, n_sub)
                M = e - s
                ky_s = ky_sub[s:e]
                w_s = w_sub[s:e]

                kp = ky_s + py
                km = -ky_s + py
                chip = (2.0 * t_hop + gamma * (kp**2) - mu)[:, None]
                chim = (2.0 * t_hop + gamma * (km**2) - mu)[:, None]

                hsc = np.broadcast_to(hx_const, (M, 4 * nx, 4 * nx)).copy()
                hsc[:, idx_eu, idx_eu] += chip
                hsc[:, idx_ed, idx_ed] += chip
                hsc[:, idx_hu, idx_hu] -= chim
                hsc[:, idx_hd, idx_hd] -= chim

                he = np.broadcast_to(he_const, (M, 2 * nx, 2 * nx)).copy()
                hh = np.broadcast_to(hh_const, (M, 2 * nx, 2 * nx)).copy()
                he[:, idx_eu, idx_eu] += chip
                he[:, idx_ed, idx_ed] += chip
                hh[:, idx_eu, idx_eu] -= chim
                hh[:, idx_ed, idx_ed] -= chim

                if Lambda != 0:
                    ryp = (-Lambda * kp)[:, None]
                    rym = (-Lambda * km)[:, None]
                    hsc[:, idx_eu, idx_ed] += ryp
                    hsc[:, idx_ed, idx_eu] += ryp
                    hsc[:, idx_hu, idx_hd] += rym
                    hsc[:, idx_hd, idx_hu] += rym
                    he[:, idx_eu, idx_ed] += ryp
                    he[:, idx_ed, idx_eu] += ryp
                    hh[:, idx_eu, idx_ed] += rym
                    hh[:, idx_ed, idx_eu] += rym

                sc_prof = np.sum(eval_thermodynamic_potential(np.linalg.eigvalsh(hsc)), axis=-1)
                nm_prof = (
                    np.sum(eval_thermodynamic_potential(np.linalg.eigvalsh(he)), axis=-1)
                    + np.sum(eval_thermodynamic_potential(np.linalg.eigvalsh(hh)), axis=-1)
                )

                acc_sc += sc_prof @ w_s
                acc_nm += nm_prof @ w_s

            out[p_idx, 0] = 0.5 * acc_sc / L_x_val
            out[p_idx, 1] = 0.5 * acc_nm / L_x_val

        return out


def run_for_nx(nx):
    t0 = time.perf_counter()
    ax = L_x / (nx + 1)
    bx = float(B_val * np.cos(theta) * Zeeman * Z_RATIO)
    by = float(B_val * np.sin(theta) * Zeeman * Z_RATIO)
    px = float(q_B_constant * B_val * np.sin(theta))
    py0 = float(q_B_constant * B_val * (-np.cos(theta)) + phi_y)

    use_mirror = np.isclose(bx, 0.0) and np.isclose(py0, 0.0)
    py_list = [py0, py0 - h] if use_mirror else [py0, py0 - h, py0 + h]

    n_tasks = max(n_cores * 2, 16)
    edges = np.linspace(0, len(k_y_values), n_tasks + 1, dtype=int)
    tasks = [(int(edges[i]), int(edges[i + 1]), py_list) for i in range(n_tasks)]

    with multiprocessing.Pool(
        processes=n_cores,
        initializer=init_worker,
        initargs=(L_x, nx, bx, by, px),
    ) as pool:
        parts = pool.map(eval_block, tasks)

    arr = np.sum(parts, axis=0)  # shape (len(py_list), 2): [E_sc, E_norm]
    Esc0, En0 = arr[0, 0], arr[0, 1]
    Esc_m, En_m = arr[1, 0], arr[1, 1]
    if use_mirror:
        Esc_p, En_p = Esc_m, En_m
    else:
        Esc_p, En_p = arr[2, 0], arr[2, 1]

    nyy_sc = (Esc_p - 2.0 * Esc0 + Esc_m) / (h**2)
    nyy_norm = (En_p - 2.0 * En0 + En_m) / (h**2)
    nyy_tot = nyy_sc - nyy_norm
    E0_tot = Esc0 - En0
    dt = time.perf_counter() - t0

    return {
        "nx": nx,
        "ax": ax,
        "nyy_sc": nyy_sc,
        "nyy_norm": nyy_norm,
        "nyy_tot": nyy_tot,
        "E0_tot": E0_tot,
        "dt": dt,
    }


if __name__ == "__main__":
    t_all = time.perf_counter()
    ref_2d = E_F / np.pi
    print(f"Prueba sin Rashba (Lambda = 0, B = 0, L_x = {L_x} nm, workers = {n_cores})")
    print(f"Referencia continua 2D limpia (E_F / pi): {ref_2d:.6f}\n")

    for nx in NX_LIST:
        r = run_for_nx(nx)
        ratio = r["nyy_tot"] / ref_2d
        print(
            f"N_x={r['nx']:3d} (a_x={r['ax']:5.2f} nm) | "
            f"nyy_sc={r['nyy_sc']:10.4f} | "
            f"nyy_norm={r['nyy_norm']:10.4f} | "
            f"nyy_tot={r['nyy_tot']:10.4f} | "
            f"nyy_tot/(E_F/pi)={ratio:6.3f} | "
            f"E0_tot={r['E0_tot']:10.6f} | "
            f"t={r['dt']:5.2f} s"
        )

    print(f"\nTiempo total: {time.perf_counter() - t_all:.2f} s")