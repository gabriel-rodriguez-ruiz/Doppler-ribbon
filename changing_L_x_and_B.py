# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:36:39 2026

@author: Gabriel
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import gc
import numpy as np
import scipy.sparse as sp
import multiprocessing
from pathlib import Path
from threadpoolctl import threadpool_limits

# ==============================================================================
# 1. MATRICES DE PAULI Y DE NAMBU
# ==============================================================================
tau_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
tau_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
tau_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
tau_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

sigma_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

# ==============================================================================
# 2. PARÁMETROS FÍSICOS Y DE LA SIMULACIÓN
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
Lambda = 15.0                 # meV nm

theta = np.pi / 2
cut_off = 2 * k_F
phi_y = 0.0
h = 2.5e-4

T = True
beta = 125.0
Zeeman = True
n_cores = 8                   # 8 núcleos físicos del Ryzen 7 5700U

N_X_SITES = 400
L_X_LIST = [3000]
KY_CHUNK_SIZE = 8 #16 #32

q_B_constant = 0.0
Z_RATIO = 1.0
B_TARGETS = [1.75*Delta, 2.5*Delta, 3.5*Delta] #[0 * Delta, 1.0 * Delta, 1.25 * Delta, 3.0 * Delta]
SAVE_NPZ = True               # Guarda .npz tras cada L_x y tras cada campo B

# --- GRILLA ADAPTATIVA EN k_y (PBC) ---
N = 1000
k_so = Lambda / (2 * gamma)
k_F_eff = np.sqrt(k_F**2 + k_so**2)
k_F_out = k_F_eff + k_so
k_limit = k_F_out + 0.1 * k_F

k_y_half = np.unique(np.concatenate([
    np.linspace(0.0, k_limit, N),
    np.linspace(k_limit, cut_off, 300)
]))
k_y_values = np.unique(np.concatenate([-k_y_half[::-1], k_y_half]))
W_KY = np.gradient(k_y_values)


def eval_thermodynamic_potential(eigvals):
    if T:
        stable_term = np.minimum(0.0, eigvals)
        log_term = -(1.0 / beta) * np.log1p(np.exp(-beta * np.abs(eigvals)))
        return stable_term + log_term
    return np.minimum(0.0, eigvals)


def build_system_operators(L_x, N_x_sites):
    a_x = L_x / (N_x_sites + 1)
    t_hop = gamma / (a_x**2)
    t_R = Lambda / (2 * a_x)

    P_e = (tau_0 + tau_z) / 2.0
    P_h = (tau_0 - tau_z) / 2.0
    I_Nx_dense = np.eye(N_x_sites, dtype=np.complex128)

    KRON_Y = np.kron(np.kron(tau_0, sigma_y), I_Nx_dense)
    KRON_X = np.kron(np.kron(tau_0, sigma_x), I_Nx_dense)
    KRON_X_0 = np.kron(np.kron(tau_x, sigma_0), I_Nx_dense)

    N_Y = np.kron(sigma_y, I_Nx_dense)
    N_X = np.kron(sigma_x, I_Nx_dense)

    off_ones = np.ones(N_x_sites - 1, dtype=np.complex128)
    kin_e_zero = sp.diags([-t_hop * off_ones, -t_hop * off_ones],
                          offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
    kin_h_zero = sp.diags([+t_hop * off_ones, +t_hop * off_ones],
                          offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')

    rashba_e_zero = sp.diags([-1j * t_R * off_ones, +1j * t_R * off_ones],
                             offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
    rashba_h_zero = sp.diags([+1j * t_R * off_ones, -1j * t_R * off_ones],
                             offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')

    H_NORM_E_BASE = (
        sp.kron(sigma_0, kin_e_zero, format='csr')
        + (sp.kron(sigma_y, rashba_e_zero, format='csr') if Lambda != 0 else 0)
    ).toarray()

    H_NORM_H_BASE = (
        sp.kron(sigma_0, kin_h_zero, format='csr')
        + (sp.kron(sigma_y, rashba_h_zero, format='csr') if Lambda != 0 else 0)
    ).toarray()

    return {
        "L_x": L_x,
        "N_x_sites": N_x_sites,
        "a_x": a_x,
        "t_hop": t_hop,
        "t_R": t_R,
        "P_e": P_e,
        "P_h": P_h,
        "off_ones": off_ones,
        "KRON_Y": KRON_Y,
        "KRON_X": KRON_X,
        "KRON_X_0": KRON_X_0,
        "N_Y": N_Y,
        "N_X": N_X,
        "H_NORM_E_BASE": H_NORM_E_BASE,
        "H_NORM_H_BASE": H_NORM_H_BASE,
    }


def build_x_hopping_matrix(ops, px, delta_val):
    N_x_sites = ops["N_x_sites"]
    a_x = ops["a_x"]
    t_hop = ops["t_hop"]
    t_R = ops["t_R"]
    off_ones = ops["off_ones"]
    P_e = ops["P_e"]
    P_h = ops["P_h"]

    theta_x = px * a_x
    phase_plus = np.exp(1j * theta_x)
    phase_minus = np.exp(-1j * theta_x)

    kin_e = sp.diags([-t_hop * phase_plus * off_ones, -t_hop * phase_minus * off_ones],
                     offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
    kin_h = sp.diags([+t_hop * phase_minus * off_ones, +t_hop * phase_plus * off_ones],
                     offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')

    H_kin_off = (
        sp.kron(np.kron(P_e, sigma_0), kin_e, format='csr') +
        sp.kron(np.kron(P_h, sigma_0), kin_h, format='csr')
    ).toarray()

    if Lambda != 0:
        rashba_e = sp.diags([-1j * t_R * phase_plus * off_ones, +1j * t_R * phase_minus * off_ones],
                            offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
        rashba_h = sp.diags([+1j * t_R * phase_minus * off_ones, -1j * t_R * phase_plus * off_ones],
                            offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
        H_rashba_x = (
            sp.kron(np.kron(P_e, sigma_y), rashba_e, format='csr') +
            sp.kron(np.kron(P_h, sigma_y), rashba_h, format='csr')
        ).toarray()
    else:
        H_rashba_x = 0.0

    return H_kin_off + H_rashba_x - delta_val * ops["KRON_X_0"]


_WORKER_STATE = {}


def init_worker(L_x, N_x_sites, bx, by, px, chunk_size):
    ops = build_system_operators(L_x, N_x_sites)
    H_x_const = build_x_hopping_matrix(ops, px, Delta) - by * ops["KRON_Y"] - bx * ops["KRON_X"]
    H_e_const = ops["H_NORM_E_BASE"] - by * ops["N_Y"] - bx * ops["N_X"]
    H_h_const = ops["H_NORM_H_BASE"] - by * ops["N_Y"] - bx * ops["N_X"]

    _WORKER_STATE.clear()
    _WORKER_STATE["L_x"] = L_x
    _WORKER_STATE["N_x_sites"] = N_x_sites
    _WORKER_STATE["t_hop"] = ops["t_hop"]
    _WORKER_STATE["H_x_const"] = H_x_const
    _WORKER_STATE["H_e_const"] = H_e_const
    _WORKER_STATE["H_h_const"] = H_h_const
    _WORKER_STATE["chunk_size"] = chunk_size
    _WORKER_STATE["IDX_E_UP"] = np.arange(0, N_x_sites)
    _WORKER_STATE["IDX_E_DN"] = np.arange(N_x_sites, 2 * N_x_sites)
    _WORKER_STATE["IDX_H_UP"] = np.arange(2 * N_x_sites, 3 * N_x_sites)
    _WORKER_STATE["IDX_H_DN"] = np.arange(3 * N_x_sites, 4 * N_x_sites)


def eval_ky_block(args):
    start_idx, end_idx, py_list = args
    with threadpool_limits(limits=1):
        L_x = _WORKER_STATE["L_x"]
        N_x_sites = _WORKER_STATE["N_x_sites"]
        t_hop = _WORKER_STATE["t_hop"]
        H_x_const = _WORKER_STATE["H_x_const"]
        H_e_const = _WORKER_STATE["H_e_const"]
        H_h_const = _WORKER_STATE["H_h_const"]
        chunk_size = _WORKER_STATE["chunk_size"]

        IDX_E_UP = _WORKER_STATE["IDX_E_UP"]
        IDX_E_DN = _WORKER_STATE["IDX_E_DN"]
        IDX_H_UP = _WORKER_STATE["IDX_H_UP"]
        IDX_H_DN = _WORKER_STATE["IDX_H_DN"]

        ky_sub = k_y_values[start_idx:end_idx]
        w_sub = W_KY[start_idx:end_idx]
        n_sub = len(ky_sub)

        partial_E = np.zeros(len(py_list), dtype=np.float64)

        for p_idx, py in enumerate(py_list):
            acc = 0.0
            for s in range(0, n_sub, chunk_size):
                e = min(s + chunk_size, n_sub)
                M = e - s
                ky_slice = ky_sub[s:e]
                w_slice = w_sub[s:e]

                kp = ky_slice + py
                km = -ky_slice + py

                chi_p = (2.0 * t_hop + gamma * (kp**2) - mu)[:, np.newaxis]
                chi_m = (2.0 * t_hop + gamma * (km**2) - mu)[:, np.newaxis]

                # 1) Superconductor
                H_sc = np.broadcast_to(H_x_const, (M, 4 * N_x_sites, 4 * N_x_sites)).copy()
                H_sc[:, IDX_E_UP, IDX_E_UP] += chi_p
                H_sc[:, IDX_E_DN, IDX_E_DN] += chi_p
                H_sc[:, IDX_H_UP, IDX_H_UP] -= chi_m
                H_sc[:, IDX_H_DN, IDX_H_DN] -= chi_m

                if Lambda != 0:
                    ry_p = (-Lambda * kp)[:, np.newaxis]
                    ry_m = (-Lambda * km)[:, np.newaxis]
                    H_sc[:, IDX_E_UP, IDX_E_DN] += ry_p
                    H_sc[:, IDX_E_DN, IDX_E_UP] += ry_p
                    H_sc[:, IDX_H_UP, IDX_H_DN] += ry_m
                    H_sc[:, IDX_H_DN, IDX_H_UP] += ry_m

                eig_sc = np.linalg.eigvalsh(H_sc)
                sc_prof = np.sum(eval_thermodynamic_potential(eig_sc), axis=-1)
                del H_sc, eig_sc

                # 2) Normal
                H_e = np.broadcast_to(H_e_const, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()
                H_h = np.broadcast_to(H_h_const, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()
                H_e[:, IDX_E_UP, IDX_E_UP] += chi_p
                H_e[:, IDX_E_DN, IDX_E_DN] += chi_p
                H_h[:, IDX_E_UP, IDX_E_UP] -= chi_m
                H_h[:, IDX_E_DN, IDX_E_DN] -= chi_m

                if Lambda != 0:
                    H_e[:, IDX_E_UP, IDX_E_DN] += ry_p
                    H_e[:, IDX_E_DN, IDX_E_UP] += ry_p
                    H_h[:, IDX_E_UP, IDX_E_DN] += ry_m
                    H_h[:, IDX_E_DN, IDX_E_UP] += ry_m

                eig_e = np.linalg.eigvalsh(H_e)
                eig_h = np.linalg.eigvalsh(H_h)
                norm_prof = (
                    np.sum(eval_thermodynamic_potential(eig_e), axis=-1)
                    + np.sum(eval_thermodynamic_potential(eig_h), axis=-1)
                )
                del H_e, H_h, eig_e, eig_h

                acc += (sc_prof - norm_prof) @ w_slice

            partial_E[p_idx] = 0.5 * acc / L_x

        return partial_E


def save_lx_npz(output_dir, B_val, L_x, a_x, N_x_sites, E0, n_yy, elapsed):
    np.savez_compressed(
        output_dir / (
            f"sweep_Lx={L_x:.0f}_Nx={N_x_sites}_B={B_val/Delta:.2f}Delta_"
            f"theta={theta:.3f}_N={N}_beta={beta}_Lambda={Lambda}_Delta={Delta}.npz"
        ),
        B_value=B_val,
        L_x=L_x,
        a_x=a_x,
        N_x_sites=N_x_sites,
        E0=E0,
        n_yy=n_yy,
        elapsed_seconds=elapsed,
        h=h,
        beta=beta,
        Lambda=Lambda,
        Delta=Delta,
        theta=theta,
    )


def save_b_summary_npz(output_dir, B_val, N_x_sites, rows, elapsed_b):
    lx_arr = np.array([r[0] for r in rows], dtype=np.float64)
    ax_arr = np.array([r[1] for r in rows], dtype=np.float64)
    e0_arr = np.array([r[2] for r in rows], dtype=np.float64)
    nyy_arr = np.array([r[3] for r in rows], dtype=np.float64)
    dt_arr = np.array([r[4] for r in rows], dtype=np.float64)

    np.savez_compressed(
        output_dir / (
            f"summary_B={B_val/Delta:.2f}Delta_Nx={N_x_sites}_"
            f"theta={theta:.3f}_N={N}_beta={beta}_Lambda={Lambda}_Delta={Delta}.npz"
        ),
        B_value=B_val,
        B_over_Delta=B_val / Delta,
        N_x_sites=N_x_sites,
        L_x_values=lx_arr,
        a_x_values=ax_arr,
        E0_values=e0_arr,
        n_yy_values=nyy_arr,
        elapsed_per_Lx=dt_arr,
        elapsed_B_total=elapsed_b,
        h=h,
        beta=beta,
        Lambda=Lambda,
        Delta=Delta,
        theta=theta,
    )


def run_Lx_sweep_for_B(B_val, L_x_list, N_x_sites=400, save_npz=True, output_dir=None):
    if output_dir is None:
        output_dir = Path("Data")
    output_dir.mkdir(exist_ok=True)

    bx = float(B_val * np.cos(theta) * Zeeman * Z_RATIO)
    by = float(B_val * np.sin(theta) * Zeeman * Z_RATIO)
    px = float(q_B_constant * B_val * np.sin(theta))
    py0 = float(q_B_constant * B_val * (-np.cos(theta)) + phi_y)

    use_mirror = np.isclose(bx, 0.0) and np.isclose(py0, 0.0)
    py_list = [py0, py0 - h] if use_mirror else [py0, py0 - h, py0 + h]

    n_ky_total = len(k_y_values)
    n_tasks = 32
    edges = np.linspace(0, n_ky_total, n_tasks + 1, dtype=int)
    tasks = [(int(edges[i]), int(edges[i + 1]), py_list) for i in range(n_tasks)]

    t_b = time.perf_counter()
    rows = []

    for L_x in L_x_list:
        t_lx = time.perf_counter()
        a_x = L_x / (N_x_sites + 1)
        print(
            f"\n--- [B = {B_val/Delta:.2f} Delta] Iniciando L_x = {L_x:.1f} nm "
            f"(N_x_sites = {N_x_sites}, a_x = {a_x:.2f} nm, workers = {n_cores}) ---"
        )

        with multiprocessing.Pool(
            processes=n_cores,
            initializer=init_worker,
            initargs=(L_x, N_x_sites, bx, by, px, KY_CHUNK_SIZE),
        ) as pool:
            partials = pool.map(eval_ky_block, tasks)

        E_vals = np.sum(partials, axis=0)
        E0 = E_vals[0]
        E_minus = E_vals[1]
        E_plus = E_minus if use_mirror else E_vals[2]
        n_yy = (E_plus - 2.0 * E0 + E_minus) / (h**2)

        elapsed_lx = time.perf_counter() - t_lx
        print(
            f"L_x = {L_x:6.1f} nm | a_x = {a_x:5.2f} nm | "
            f"E0 = {E0:.10e} | n_yy = {n_yy:.10e} | Tiempo = {elapsed_lx:.2f} s"
        )

        rows.append((L_x, a_x, E0, n_yy, elapsed_lx))

        if save_npz:
            save_lx_npz(output_dir, B_val, L_x, a_x, N_x_sites, E0, n_yy, elapsed_lx)

        del partials, E_vals
        gc.collect()

    elapsed_b = time.perf_counter() - t_b
    if save_npz:
        save_b_summary_npz(output_dir, B_val, N_x_sites, rows, elapsed_b)

    return rows, elapsed_b


def run_B_targets_sequential(B_targets, L_x_list, N_x_sites=400, save_npz=True):
    output_dir = Path("Data")
    output_dir.mkdir(exist_ok=True)
    all_summaries = []

    for idx, B_val in enumerate(B_targets, start=1):
        print(f"\n==================== CAMPO {idx}/{len(B_targets)}: B = {B_val:.6g} meV ({B_val/Delta:.2f} Delta) ====================")
        rows, elapsed_b = run_Lx_sweep_for_B(
            B_val=float(B_val),
            L_x_list=L_x_list,
            N_x_sites=N_x_sites,
            save_npz=save_npz,
            output_dir=output_dir,
        )
        all_summaries.append((float(B_val), rows, elapsed_b))
        gc.collect()

    return all_summaries


if __name__ == "__main__":
    t_total = time.perf_counter()
    summaries = run_B_targets_sequential(
        B_targets=B_TARGETS,
        L_x_list=L_X_LIST,
        N_x_sites=N_X_SITES,
        save_npz=SAVE_NPZ,
    )

    print(f"\n=== RESUMEN GLOBAL POR CAMPO B (N_x_sites = {N_X_SITES} FIJO) ===")
    for b_val, rows, dt_b in summaries:
        print(f"B = {b_val/Delta:.2f} Delta ({b_val:.6g} meV) | Tiempo B: {dt_b:.2f} s")
        for lx_val, ax_val, e0_val, nyy_val, dt_lx in rows:
            print(
                f"   L_x = {lx_val:6.1f} nm (a_x = {ax_val:5.2f} nm) -> "
                f"n_yy = {nyy_val:.10e}, E0 = {e0_val:.10e} ({dt_lx:.1f} s)"
            )
    print(f"Tiempo total: {time.perf_counter() - t_total:.2f} s")