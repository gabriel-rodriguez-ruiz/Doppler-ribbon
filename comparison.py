# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:49:19 2026

@author: Gabriel
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import scipy.sparse as sp
import scipy.integrate
import multiprocessing
from pathlib import Path
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
# 2. PARÁMETROS ALINEADOS Y MALLA RÁPIDA
# ==============================================================================
c = 3e17                      # nm/s
m_e = 5.1e8 / c**2            # meV s²/(nm)²
m = 0.0403 * m_e              # meV s²/(nm)²
hbar = 6.58e-13               # meV s
gamma = hbar**2 / (2 * m)     # meV (nm)²
E_F = 50.6                    # meV
k_F = np.sqrt(E_F / gamma)    # 1/nm

Delta = 0.4                   # meV (alineado)
mu = E_F                      # meV
Lambda = 15.0                 # meV nm (cambia a 0.0 para ver límite limpio)
theta = np.pi / 2
cut_off = 2.0 * k_F
phi_y = 0.0
h = 2e-4                      # paso derivada alineado

T = True
beta = 125.0                  # temperatura inversa alineada
Zeeman = True
Z_RATIO = 1.0
q_B_constant = 0.0

n_cores = 8
points = 16
B_values = np.linspace(0.0 * Delta, 4.0 * Delta, points)

# --- Parámetros 2D continuo ---
N_2D = 200
k_1 = (-Lambda + np.sqrt(Lambda**2 + 4 * gamma * mu)) / (2 * gamma)
k_2 = (Lambda + np.sqrt(Lambda**2 + 4 * gamma * mu)) / (2 * gamma)
k_values_2d = np.unique(np.concatenate([
    np.linspace(0.0, 0.99 * k_1, N_2D),
    np.linspace(0.99 * k_1, 1.01 * k_1, N_2D),
    np.linspace(1.01 * k_1, 0.99 * k_2, N_2D),
    np.linspace(0.99 * k_2, 1.01 * k_2, N_2D),
    np.linspace(1.01 * k_2, cut_off, N_2D),
]))
theta_values_2d = np.linspace(0.0, 2.0 * np.pi, N_2D)

# --- Parámetros cinta OBC ---
L_x = 500.0                   # nm
N_x_sites = 200               # a_x ≈ 2.49 nm
a_x = L_x / (N_x_sites + 1)
t_hop = gamma / (a_x**2)
t_R = Lambda / (2 * a_x)

N_KY_INNER = 500
KY_CHUNK_SIZE = 16 #64
k_so = Lambda / (2 * gamma)
k_F_eff = np.sqrt(k_F**2 + k_so**2)
k_F_out = k_F_eff + k_so
k_limit = k_F_out + 0.1 * k_F
k_y_half = np.unique(np.concatenate([
    np.linspace(0.0, k_limit, N_KY_INNER),
    np.linspace(k_limit, cut_off, 150),
]))
k_y_values = np.unique(np.concatenate([-k_y_half[::-1], k_y_half]))
W_KY = np.gradient(k_y_values)


def omega(eigvals):
    if T:
        return np.minimum(0.0, eigvals) - (1.0 / beta) * np.log1p(np.exp(-beta * np.abs(eigvals)))
    return np.minimum(0.0, eigvals)


# ==============================================================================
# 3. MODELO 2D CONTINUO (PBC / PLANO INFINITO)
# ==============================================================================
KRON_PLUS_2D = np.kron((tau_0 + tau_z) / 2.0, sigma_0)
KRON_MINUS_2D = np.kron((tau_0 - tau_z) / 2.0, sigma_0)
KRON_Y_2D = np.kron(tau_0, sigma_y)
KRON_X_2D = np.kron(tau_0, sigma_x)
KRON_X0_2D = np.kron(tau_x, sigma_0)
KRON_PLUS_Y_2D = np.kron((tau_0 + tau_z) / 2.0, sigma_y)
KRON_MINUS_Y_2D = np.kron((tau_0 - tau_z) / 2.0, sigma_y)
KRON_PLUS_X_2D = np.kron((tau_0 + tau_z) / 2.0, sigma_x)
KRON_MINUS_X_2D = np.kron((tau_0 - tau_z) / 2.0, sigma_x)


def worker_2d(B_chunk):
    with threadpool_limits(limits=1):
        nb = len(B_chunk)
        bx_v = B_chunk * np.cos(theta) * Zeeman * Z_RATIO
        by_v = B_chunk * np.sin(theta) * Zeeman * Z_RATIO
        qx_v = q_B_constant * B_chunk * np.sin(theta)
        qy_v = q_B_constant * B_chunk * (-np.cos(theta)) + phi_y

        # Solo 3 puntos longitudinales en py: [py0, py0 - h, py0 + h]
        px_t = np.repeat(qx_v[:, None], 3, axis=1)
        py_t = np.stack([qy_v, qy_v - h, qy_v + h], axis=1)

        q2 = px_t**2 + py_t**2
        surf = (np.pi / 2.0) * (cut_off**2) * (2.0 * gamma * q2 - 2.0 * mu + gamma * (cut_off**2))

        ang_acc = np.zeros((nb, 3, len(theta_values_2d)), dtype=np.float64)
        px = px_t[:, :, None]
        py = py_t[:, :, None]
        bx = bx_v[:, None, None, None, None]
        by = by_v[:, None, None, None, None]
        const = -by * KRON_Y_2D - bx * KRON_X_2D - Delta * KRON_X0_2D

        for t_idx, t_val in enumerate(theta_values_2d):
            kx = (k_values_2d * np.cos(t_val))[None, None, :]
            ky = (k_values_2d * np.sin(t_val))[None, None, :]

            chip = gamma * ((kx + px)**2 + (ky + py)**2) - mu
            chim = gamma * ((-kx + px)**2 + (-ky + py)**2) - mu

            H = (
                chip[..., None, None] * KRON_PLUS_2D
                - chim[..., None, None] * KRON_MINUS_2D
                + const
            )
            if Lambda != 0:
                H += (
                    Lambda * (kx + px)[..., None, None] * KRON_PLUS_Y_2D
                    + Lambda * (-kx + px)[..., None, None] * KRON_MINUS_Y_2D
                    - Lambda * (ky + py)[..., None, None] * KRON_PLUS_X_2D
                    - Lambda * (-ky + py)[..., None, None] * KRON_MINUS_X_2D
                )
            eigs = np.linalg.eigvalsh(H)
            rad_prof = np.sum(omega(eigs), axis=-1) * k_values_2d[None, None, :]
            ang_acc[:, :, t_idx] = scipy.integrate.trapezoid(rad_prof, k_values_2d, axis=-1)

        tot_int = scipy.integrate.trapezoid(ang_acc, theta_values_2d, axis=-1)
        # Normalización física 2D: 1 / (2*pi)^2
        E = (0.5 * tot_int + surf) / ((2.0 * np.pi) ** 2)
        nyy = (E[:, 2] - 2.0 * E[:, 0] + E[:, 1]) / (h**2)
        return list(zip(nyy, E[:, 0]))


# ==============================================================================
# 4. MODELO CINTA OBC EN x (DISCRETIZADO EN x, CONTINUO EN k_y)
# ==============================================================================
P_e = (tau_0 + tau_z) / 2.0
P_h = (tau_0 - tau_z) / 2.0
I_Nx = np.eye(N_x_sites, dtype=np.complex128)
off_ones = np.ones(N_x_sites - 1, dtype=np.complex128)

KRON_Y_OBC = np.kron(np.kron(tau_0, sigma_y), I_Nx)
KRON_X_OBC = np.kron(np.kron(tau_0, sigma_x), I_Nx)
KRON_X0_OBC = np.kron(np.kron(tau_x, sigma_0), I_Nx)
N_Y_OBC = np.kron(sigma_y, I_Nx)
N_X_OBC = np.kron(sigma_x, I_Nx)

kin_e0 = sp.diags([-t_hop * off_ones, -t_hop * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
kin_h0 = sp.diags([+t_hop * off_ones, +t_hop * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
r_e0 = sp.diags([-1j * t_R * off_ones, +1j * t_R * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
r_h0 = sp.diags([+1j * t_R * off_ones, -1j * t_R * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")

H_NORM_E_BASE = (
    sp.kron(sigma_0, kin_e0, format="csr")
    + (sp.kron(sigma_y, r_e0, format="csr") if Lambda != 0 else 0)
).toarray()
H_NORM_H_BASE = (
    sp.kron(sigma_0, kin_h0, format="csr")
    + (sp.kron(sigma_y, r_h0, format="csr") if Lambda != 0 else 0)
).toarray()

IDX_E_UP = np.arange(0, N_x_sites)
IDX_E_DN = np.arange(N_x_sites, 2 * N_x_sites)
IDX_H_UP = np.arange(2 * N_x_sites, 3 * N_x_sites)
IDX_H_DN = np.arange(3 * N_x_sites, 4 * N_x_sites)


def build_x_hopping_matrix(px, delta_val):
    th_x = px * a_x
    pp, pm = np.exp(1j * th_x), np.exp(-1j * th_x)
    kin_e = sp.diags([-t_hop * pp * off_ones, -t_hop * pm * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
    kin_h = sp.diags([+t_hop * pm * off_ones, +t_hop * pp * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
    h_kin = (
        sp.kron(np.kron(P_e, sigma_0), kin_e, format="csr")
        + sp.kron(np.kron(P_h, sigma_0), kin_h, format="csr")
    ).toarray()
    if Lambda != 0:
        re_m = sp.diags([-1j * t_R * pp * off_ones, +1j * t_R * pm * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
        rh_m = sp.diags([+1j * t_R * pm * off_ones, -1j * t_R * pp * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
        h_r = (
            sp.kron(np.kron(P_e, sigma_y), re_m, format="csr")
            + sp.kron(np.kron(P_h, sigma_y), rh_m, format="csr")
        ).toarray()
    else:
        h_r = 0.0
    return h_kin + h_r - delta_val * KRON_X0_OBC


def eval_obc_energy(hx_const, he_const, hh_const, py):
    sc_acc = 0.0
    nm_acc = 0.0
    n_ky = len(k_y_values)
    for s in range(0, n_ky, KY_CHUNK_SIZE):
        e = min(s + KY_CHUNK_SIZE, n_ky)
        M = e - s
        ky_s = k_y_values[s:e]
        w_s = W_KY[s:e]

        kp = ky_s + py
        km = -ky_s + py
        chip = (2.0 * t_hop + gamma * (kp**2) - mu)[:, None]
        chim = (2.0 * t_hop + gamma * (km**2) - mu)[:, None]

        hsc = np.broadcast_to(hx_const, (M, 4 * N_x_sites, 4 * N_x_sites)).copy()
        hsc[:, IDX_E_UP, IDX_E_UP] += chip
        hsc[:, IDX_E_DN, IDX_E_DN] += chip
        hsc[:, IDX_H_UP, IDX_H_UP] -= chim
        hsc[:, IDX_H_DN, IDX_H_DN] -= chim

        he = np.broadcast_to(he_const, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()
        hh = np.broadcast_to(hh_const, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()
        he[:, IDX_E_UP, IDX_E_UP] += chip
        he[:, IDX_E_DN, IDX_E_DN] += chip
        hh[:, IDX_E_UP, IDX_E_UP] -= chim
        hh[:, IDX_E_DN, IDX_E_DN] -= chim

        if Lambda != 0:
            ryp = (-Lambda * kp)[:, None]
            rym = (-Lambda * km)[:, None]
            hsc[:, IDX_E_UP, IDX_E_DN] += ryp
            hsc[:, IDX_E_DN, IDX_E_UP] += ryp
            hsc[:, IDX_H_UP, IDX_H_DN] += rym
            hsc[:, IDX_H_DN, IDX_H_UP] += rym
            he[:, IDX_E_UP, IDX_E_DN] += ryp
            he[:, IDX_E_DN, IDX_E_UP] += ryp
            hh[:, IDX_E_UP, IDX_E_DN] += rym
            hh[:, IDX_E_DN, IDX_E_UP] += rym

        sc_prof = np.sum(omega(np.linalg.eigvalsh(hsc)), axis=-1)
        nm_prof = (
            np.sum(omega(np.linalg.eigvalsh(he)), axis=-1)
            + np.sum(omega(np.linalg.eigvalsh(hh)), axis=-1)
        )
        sc_acc += sc_prof @ w_s
        nm_acc += nm_prof @ w_s

    # Normalización física OBC por área: 1 / (2*pi * L_x)
    return 0.5 * (sc_acc - nm_acc) / (2.0 * np.pi * L_x)


def worker_obc(B_chunk):
    with threadpool_limits(limits=1):
        out = []
        for B_val in B_chunk:
            bx = float(B_val * np.cos(theta) * Zeeman * Z_RATIO)
            by = float(B_val * np.sin(theta) * Zeeman * Z_RATIO)
            px = float(q_B_constant * B_val * np.sin(theta))
            py0 = float(q_B_constant * B_val * (-np.cos(theta)) + phi_y)

            hx_const = build_x_hopping_matrix(px, Delta) - by * KRON_Y_OBC - bx * KRON_X_OBC
            he_const = H_NORM_E_BASE - by * N_Y_OBC - bx * N_X_OBC
            hh_const = H_NORM_H_BASE - by * N_Y_OBC - bx * N_X_OBC

            use_mirror = np.isclose(bx, 0.0) and np.isclose(py0, 0.0)
            E0 = eval_obc_energy(hx_const, he_const, hh_const, py0)
            Em = eval_obc_energy(hx_const, he_const, hh_const, py0 - h)
            Ep = Em if use_mirror else eval_obc_energy(hx_const, he_const, hh_const, py0 + h)

            nyy = (Ep - 2.0 * E0 + Em) / (h**2)
            out.append((nyy, E0))
        return out


# ==============================================================================
# 5. EJECUCIÓN PARALELA Y COMPARACIÓN
# ==============================================================================
if __name__ == "__main__":
    print(f"Comparación 2D continuo vs. Cinta OBC (Delta={Delta} meV, Lambda={Lambda} meV*nm, beta={beta})")
    print(f"Cinta OBC: L_x={L_x} nm, N_x={N_x_sites}, a_x={a_x:.2f} nm | Workers: {n_cores}\n")

    chunks = np.array([[b] for b in B_values])

    t0 = time.perf_counter()
    with multiprocessing.Pool(n_cores) as pool:
        res_2d = [item for sub in pool.map(worker_2d, chunks) for item in sub]
    nyy_2d, E0_2d = map(np.array, zip(*res_2d))
    t_2d = time.perf_counter() - t0

    t1 = time.perf_counter()
    with multiprocessing.Pool(n_cores) as pool:
        res_obc = [item for sub in pool.map(worker_obc, chunks) for item in sub]
    nyy_obc, E0_obc = map(np.array, zip(*res_obc))
    t_obc = time.perf_counter() - t1

    print(f"Tiempo 2D: {t_2d:.2f} s | Tiempo OBC: {t_obc:.2f} s | Ref limpia E_F/pi = {E_F/np.pi:.4f}\n")
    print(f"{'B/Delta':>8} | {'n_yy (2D)':>12} | {'n_yy (OBC)':>12} | {'Dif %':>9} | {'E0 (2D)':>12} | {'E0 (OBC)':>12}")
    print("-" * 78)
    for b, n2, no, e2, eo in zip(B_values / Delta, nyy_2d, nyy_obc, E0_2d, E0_obc):
        rel = 100.0 * (no - n2) / max(abs(n2), 1e-12)
        print(f"{b:8.3f} | {n2:12.4f} | {no:12.4f} | {rel:8.2f}% | {e2:12.6f} | {eo:12.6f}")

    out_dir = Path("Data")
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out_dir / f"compare_2d_vs_obc_Lambda={Lambda}_Lx={L_x}_Nx={N_x_sites}.npz",
        B_values=B_values,
        nyy_2d=nyy_2d,
        nyy_obc=nyy_obc,
        E0_2d=E0_2d,
        E0_obc=E0_obc,
    )

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(B_values / Delta, nyy_2d, "o-", label="2D Continuo (PBC)")
        ax.plot(B_values / Delta, nyy_obc, "s--", label=f"Cinta OBC ($L_x={L_x}$ nm, $N_x={N_x_sites}$)")
        ax.axhline(E_F / np.pi, color="gray", linestyle=":", label=r"$E_F/\pi$ (2D limpio $B=0$)")
        ax.set_xlabel(r"$B / \Delta$")
        ax.set_ylabel(r"$n_{yy}$ [meV]")
        ax.set_title(rf"Comparación $n_{{yy}}(B)$ ($\Delta={Delta}$ meV, $\Lambda={Lambda}$ meV$\cdot$nm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_2d_vs_obc_Lambda={Lambda}_Lx={L_x}_Nx={N_x_sites}.png", dpi=150)
        plt.show()
    except Exception as exc:
        print(f"Gráfica no generada: {exc}")