# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:36:34 2026

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
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import gc

# ==============================================================================
# 1. MATRICES DE PAULI Y PARÁMETROS FÍSICOS
# ==============================================================================
tau_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
tau_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
tau_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

sigma_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)

c = 3e17                      # nm/s
m_e = 5.1e8 / c**2            # meV s²/(nm)²
m = 0.0403 * m_e              # meV s²/(nm)²
hbar = 6.58e-13               # meV s
gamma = hbar**2 / (2 * m)     # meV (nm)²
E_F = 50.6                    # meV
k_F = np.sqrt(E_F / gamma)    # 1/nm

Delta = 0.40                  # meV
mu = E_F                      # meV
Lambda = 15.0                 # meV nm

theta = np.pi / 2
cut_off = 2.0 * k_F
phi_y = 0.0
h = 2.5e-4

T = True
beta = 125.0
Zeeman = True
Z_RATIO = 1.0
n_cores = 15
points = 3 * n_cores

L_x = 750.0                   # nm
N_x_sites = 100
a_x = L_x / (N_x_sites + 1)
t_hop = gamma / (a_x**2)
t_R = Lambda / (2 * a_x)

# Primer mínimo de Fraunhofer ajustado en B/Delta = 6.2
q_B_constant = 0  #(np.pi / L_x) / (6.2 * Delta)

# Grilla adaptativa en k_y
N = 900
KY_CHUNK_SIZE = 75

k_so = Lambda / (2 * gamma)
k_F_eff = np.sqrt(k_F**2 + k_so**2)
k_limit = (k_F_eff + k_so) + 0.1 * k_F

k_y_half = np.unique(np.concatenate([
    np.linspace(0.0, k_limit, N),
    np.linspace(k_limit, cut_off, 200)
]))
k_y_values = np.unique(np.concatenate([-k_y_half[::-1], k_y_half]))
W_KY = np.gradient(k_y_values)

# ==============================================================================
# 2. OPERADORES BASE Y PROYECTOR DE CANALES TRANSVERSALES
# ==============================================================================
P_e = (tau_0 + tau_z) / 2.0
P_h = (tau_0 - tau_z) / 2.0
I_Nx = np.eye(N_x_sites, dtype=np.complex128)

KRON_Y = np.kron(np.kron(tau_0, sigma_y), I_Nx)
KRON_X = np.kron(np.kron(tau_0, sigma_x), I_Nx)
KRON_X_0 = np.kron(np.kron(tau_x, sigma_0), I_Nx)
N_Y = np.kron(sigma_y, I_Nx)
N_X = np.kron(sigma_x, I_Nx)

off_ones = np.ones(N_x_sites - 1, dtype=np.complex128)
kin_e_0 = sp.diags([-t_hop * off_ones, -t_hop * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
kin_h_0 = sp.diags([+t_hop * off_ones, +t_hop * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
rashba_e_0 = sp.diags([-1j * t_R * off_ones, +1j * t_R * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
rashba_h_0 = sp.diags([+1j * t_R * off_ones, -1j * t_R * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')

H_NORM_E_BASE = (sp.kron(sigma_0, kin_e_0) + sp.kron(sigma_y, rashba_e_0)).toarray()
H_NORM_H_BASE = (sp.kron(sigma_0, kin_h_0) + sp.kron(sigma_y, rashba_h_0)).toarray()

IDX_E_UP = np.arange(0, N_x_sites)
IDX_E_DN = np.arange(N_x_sites, 2 * N_x_sites)
IDX_H_UP = np.arange(2 * N_x_sites, 3 * N_x_sites)
IDX_H_DN = np.arange(3 * N_x_sites, 4 * N_x_sites)

# Matriz ortogonal U_trans[j, n_x] de ondas estacionarias en x
j_sites = np.arange(1, N_x_sites + 1)[:, np.newaxis]
n_modes = np.arange(1, N_x_sites + 1)[np.newaxis, :]
U_TRANS = np.sqrt(2.0 / (N_x_sites + 1)) * np.sin(j_sites * n_modes * np.pi / (N_x_sites + 1))


def eval_thermodynamic_potential(eigvals):
    if T:
        stable_term = np.minimum(0.0, eigvals)
        log_term = -(1.0 / beta) * np.log1p(np.exp(-beta * np.abs(eigvals)))
        return stable_term + log_term
    return np.minimum(0.0, eigvals)


def build_x_hopping_matrix(px):
    theta_x = px * a_x
    p_plus, p_minus = np.exp(1j * theta_x), np.exp(-1j * theta_x)
    kin_e = sp.diags([-t_hop * p_plus * off_ones, -t_hop * p_minus * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
    kin_h = sp.diags([+t_hop * p_minus * off_ones, +t_hop * p_plus * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
    rashba_e = sp.diags([-1j * t_R * p_plus * off_ones, +1j * t_R * p_minus * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
    rashba_h = sp.diags([+1j * t_R * p_minus * off_ones, -1j * t_R * p_plus * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
    return (
        sp.kron(np.kron(P_e, sigma_0), kin_e) + sp.kron(np.kron(P_h, sigma_0), kin_h)
        + sp.kron(np.kron(P_e, sigma_y), rashba_e) + sp.kron(np.kron(P_h, sigma_y), rashba_h)
    ).toarray() - Delta * KRON_X_0


def compute_channel_E_point(H_x_sc, H_e_norm, H_h_norm, py):
    """Versión de memoria constante (< 100 MB/proceso) sin tensores 4D de einsum."""
    tot_channels = np.zeros(N_x_sites, dtype=np.float64)
    U_T = U_TRANS.T  # (N_x_sites, N_x_sites)

    for s_idx in range(0, len(k_y_values), KY_CHUNK_SIZE):
        e_idx = min(s_idx + KY_CHUNK_SIZE, len(k_y_values))
        M = e_idx - s_idx
        ky_s = k_y_values[s_idx:e_idx]
        w_s = W_KY[s_idx:e_idx]

        kp, km = ky_s + py, -ky_s + py
        chi_p = (2.0 * t_hop + gamma * (kp**2) - mu)[:, np.newaxis]
        chi_m = (2.0 * t_hop + gamma * (km**2) - mu)[:, np.newaxis]
        ry_p, ry_m = (-Lambda * kp)[:, np.newaxis], (-Lambda * km)[:, np.newaxis]

        # 1. Sector Superconductor
        H_sc = np.broadcast_to(H_x_sc, (M, 4 * N_x_sites, 4 * N_x_sites)).copy()
        H_sc[:, IDX_E_UP, IDX_E_UP] += chi_p
        H_sc[:, IDX_E_DN, IDX_E_DN] += chi_p
        H_sc[:, IDX_H_UP, IDX_H_UP] -= chi_m
        H_sc[:, IDX_H_DN, IDX_H_DN] -= chi_m
        H_sc[:, IDX_E_UP, IDX_E_DN] += ry_p
        H_sc[:, IDX_E_DN, IDX_E_UP] += ry_p
        H_sc[:, IDX_H_UP, IDX_H_DN] += ry_m
        H_sc[:, IDX_H_DN, IDX_H_UP] += ry_m

        evals_sc, evecs_sc = np.linalg.eigh(H_sc)
        del H_sc

        # Proyección directa por bloques sin crear tensores 4D temporales
        weights_sc = np.zeros((M, N_x_sites, 4 * N_x_sites), dtype=np.float64)
        for sec in range(4):
            sl = slice(sec * N_x_sites, (sec + 1) * N_x_sites)
            proj = U_T @ evecs_sc[:, sl, :]
            weights_sc += (proj.real**2 + proj.imag**2)
        del evecs_sc, proj

        pot_sc = eval_thermodynamic_potential(evals_sc)
        prof_sc = np.einsum("mna,ma->mn", weights_sc, pot_sc)
        del evals_sc, weights_sc, pot_sc

        # 2. Sector Normal (Electrones + Huecos)
        He = np.broadcast_to(H_e_norm, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()
        Hh = np.broadcast_to(H_h_norm, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()
        He[:, IDX_E_UP, IDX_E_UP] += chi_p
        He[:, IDX_E_DN, IDX_E_DN] += chi_p
        Hh[:, IDX_E_UP, IDX_E_UP] -= chi_m
        Hh[:, IDX_E_DN, IDX_E_DN] -= chi_m
        He[:, IDX_E_UP, IDX_E_DN] += ry_p
        He[:, IDX_E_DN, IDX_E_UP] += ry_p
        Hh[:, IDX_E_UP, IDX_E_DN] += ry_m
        Hh[:, IDX_E_DN, IDX_E_UP] += ry_m   

        ev_e, vec_e = np.linalg.eigh(He)
        ev_h, vec_h = np.linalg.eigh(Hh)
        del He, Hh

        w_e = np.zeros((M, N_x_sites, 2 * N_x_sites), dtype=np.float64)
        w_h = np.zeros((M, N_x_sites, 2 * N_x_sites), dtype=np.float64)
        for sec in range(2):
            sl = slice(sec * N_x_sites, (sec + 1) * N_x_sites)
            pe = U_T @ vec_e[:, sl, :]
            ph = U_T @ vec_h[:, sl, :]
            w_e += (pe.real**2 + pe.imag**2)
            w_h += (ph.real**2 + ph.imag**2)
        del vec_e, vec_h, pe, ph

        prof_norm = (
            np.einsum("mna,ma->mn", w_e, eval_thermodynamic_potential(ev_e))
            + np.einsum("mna,ma->mn", w_h, eval_thermodynamic_potential(ev_h))
        )
        del ev_e, ev_h, w_e, w_h

        tot_channels += w_s @ (prof_sc - prof_norm)

    gc.collect()
    return 0.5 * tot_channels / L_x


def worker_channel_nyy(B_val):
    with threadpool_limits(limits=1):
        bx = B_val * np.cos(theta) * Zeeman * Z_RATIO
        by = B_val * np.sin(theta) * Zeeman * Z_RATIO
        px = q_B_constant * B_val * np.sin(theta)
        py0 = q_B_constant * B_val * (-np.cos(theta)) + phi_y

        H_x_sc = build_x_hopping_matrix(px) - by * KRON_Y - bx * KRON_X
        H_e_norm = H_NORM_E_BASE - by * N_Y - bx * N_X
        H_h_norm = H_NORM_H_BASE - by * N_Y - bx * N_X

        E0_ch = compute_channel_E_point(H_x_sc, H_e_norm, H_h_norm, py0)
        Em_ch = compute_channel_E_point(H_x_sc, H_e_norm, H_h_norm, py0 - h)
        Ep_ch = Em_ch if (np.isclose(bx, 0.0) and np.isclose(py0, 0.0)) else compute_channel_E_point(H_x_sc, H_e_norm, H_h_norm, py0 + h)

        return (Ep_ch - 2.0 * E0_ch + Em_ch) / (h**2)


# ==============================================================================
# 3. EJECUCIÓN Y VISUALIZACIÓN POR CANAL
# ==============================================================================
if __name__ == "__main__":
    t0 = time.perf_counter()
    B_values = np.linspace(0.0 * Delta, 8.0 * Delta, points)

    with multiprocessing.Pool(n_cores, maxtasksperchild=1) as pool:
        # Matriz de forma (points, N_x_sites)
        nyy_channels = np.array(pool.map(worker_channel_nyy, B_values))

    print(f"Cálculo resuelto por canal completado en {time.perf_counter() - t0:.2f} s.")

    # Verificación exacta: la suma de todos los canales recupera n_yy total
    nyy_total = np.sum(nyy_channels, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)

    # Panel 1: Agrupación por familias de canales (bajos, intermedios y altos cerca de E_F)
    ax1.plot(B_values / Delta, nyy_total, 'k-', lw=2.2, label=r'Total $n_{yy} = \sum_{n_x} n_{yy}^{(n_x)}$')
    ax1.plot(B_values / Delta, np.sum(nyy_channels[:, :20], axis=1), '--', lw=1.7, label=r'Canales profundos ($n_x = 1\text{--}20$, $v_y$ muy alta)')
    ax1.plot(B_values / Delta, np.sum(nyy_channels[:, 20:42], axis=1), '-.', lw=1.7, label=r'Canales intermedios ($n_x = 21\text{--}42$, caída en $B/\Delta \approx 4$)')
    ax1.plot(B_values / Delta, np.sum(nyy_channels[:, 42:60], axis=1), ':', lw=1.9, label=r'Canales altos ($n_x = 43\text{--}60$, cerca de $k_y \approx 0$)')
    ax1.set_ylabel(r'Rigidez $n_{yy}$ (meV)')
    ax1.set_title(r'Descomposición de $n_{yy}(B)$ por familias de canales transversales ($L_x = 750$ nm)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    # Panel 2: Mapa de calor de n_yy^{(n_x)} para cada canal ocupado (n_x = 1..60)
    im = ax2.pcolormesh(
    B_values / Delta, np.arange(1, 61), nyy_channels[:, :60].T,
    shading='auto', cmap='RdBu_r', vmin=-1.5, vmax=3.5
    )
    ax2.set_xlabel(r'$B / \Delta$')
    ax2.set_ylabel(r'Índice de canal transversal $n_x$')
    ax2.set_title(r'Contribución individual $n_{yy}^{(n_x)}(B)$ (azul = resta paramagnética por BFS)')
    cbar = plt.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label(r'$n_{yy}^{(n_x)}$ (meV)')
    plt.tight_layout()
    plt.show()