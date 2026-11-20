# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:11:06 2026

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
mu_B = 5.79e-2                # meV/T

Delta = 0.4 #0.08                  # meV
mu = E_F                      # meV
Lambda = 15.0          # meV nm

theta = np.pi / 2
N_phi = 3
cut_off = 2*k_F     #1.35 * k_F          # Recorte optimizado: cancela colas UV y acelera ~35%

phi_y = 0.0
h = 2.5e-4 #5e-5

T = True
beta = 125 #125 #64.0
Zeeman = True
n_cores = 15
points = 3 * n_cores

L_x = 750 #2500                  # nm (OBC en x)
B_star = 4 * Delta
q_B_constant = 0 #0.0039 #(np.pi / (2 * L_x)) / B_star
Z_RATIO = 1.0

# --- DISCRETIZACIÓN ESPACIAL EN x (OBC) ---
N_x_sites = 100 #100 #80   #80
a_x = L_x / (N_x_sites + 1)
t_hop = gamma / (a_x**2)
t_R = Lambda / (2 * a_x)

# --- GRILLA ADAPTATIVA EN k_y (PBC) ---
N = 1000 #1200  #1500
KY_CHUNK_SIZE = 32 #125 #125 #256

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

parameters = {
    "gamma": gamma, "points": points, "k_F": k_F,
    "mu": mu, "Delta": Delta,
    "N_phi": N_phi, "Lambda": Lambda, "L_x": L_x, "N_x_sites": N_x_sites,
    "N": N, "KY_CHUNK_SIZE": KY_CHUNK_SIZE,
    "cut_off": cut_off, "theta": theta,
    "phi_y": phi_y,
    "T": T, "beta": beta,
    "k_y_values": k_y_values,
    "q_B_constant": q_B_constant,
    "Z_RATIO": Z_RATIO
}

# ==============================================================================
# 3. OPERADORES BASE Y ÍNDICES DIRECTOS
# ==============================================================================
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

IDX_E_UP = np.arange(0, N_x_sites)
IDX_E_DN = np.arange(N_x_sites, 2 * N_x_sites)
IDX_H_UP = np.arange(2 * N_x_sites, 3 * N_x_sites)
IDX_H_DN = np.arange(3 * N_x_sites, 4 * N_x_sites)


def eval_thermodynamic_potential(eigvals, T, beta):
    if T:
        stable_term = np.minimum(0.0, eigvals)
        log_term = -(1.0 / beta) * np.log1p(np.exp(-beta * np.abs(eigvals)))
        return stable_term + log_term
    else:
        return np.minimum(0.0, eigvals)


def compute_normal_E_point(py, bx, by):
    """Cálculo rápido del estado normal (2*Nx x 2*Nx) consistente con BdG."""
    H_e_const = H_NORM_E_BASE - by * N_Y - bx * N_X
    H_h_const = H_NORM_H_BASE - by * N_Y - bx * N_X

    total_integral = 0.0
    n_ky_total = len(k_y_values)

    for start_idx in range(0, n_ky_total, KY_CHUNK_SIZE):
        end_idx = min(start_idx + KY_CHUNK_SIZE, n_ky_total)
        M = end_idx - start_idx
        ky_slice = k_y_values[start_idx:end_idx]
        w_slice = W_KY[start_idx:end_idx]

        kp = ky_slice + py
        km = -ky_slice + py

        chi_p = (2.0 * t_hop + gamma * (kp**2) - mu)[:, np.newaxis]
        chi_m = (2.0 * t_hop + gamma * (km**2) - mu)[:, np.newaxis]

        H_e_tensor = np.broadcast_to(H_e_const, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()
        H_h_tensor = np.broadcast_to(H_h_const, (M, 2 * N_x_sites, 2 * N_x_sites)).copy()

        H_e_tensor[:, IDX_E_UP, IDX_E_UP] += chi_p
        H_e_tensor[:, IDX_E_DN, IDX_E_DN] += chi_p
        H_h_tensor[:, IDX_E_UP, IDX_E_UP] -= chi_m
        H_h_tensor[:, IDX_E_DN, IDX_E_DN] -= chi_m

        if Lambda != 0:
            ry_p = (-Lambda * kp)[:, np.newaxis]
            ry_m = (-Lambda * km)[:, np.newaxis]
            H_e_tensor[:, IDX_E_UP, IDX_E_DN] += ry_p
            H_e_tensor[:, IDX_E_DN, IDX_E_UP] += ry_p
            H_h_tensor[:, IDX_E_UP, IDX_E_DN] += ry_m
            H_h_tensor[:, IDX_E_DN, IDX_E_UP] += ry_m

        eig_e = np.linalg.eigvalsh(H_e_tensor)
        eig_h = np.linalg.eigvalsh(H_h_tensor)

        ky_profiles = (
            np.sum(eval_thermodynamic_potential(eig_e, T, beta), axis=-1)
            + np.sum(eval_thermodynamic_potential(eig_h, T, beta), axis=-1)
        )
        total_integral += ky_profiles @ w_slice

    return 0.5 * total_integral / L_x


def build_x_hopping_matrix(px, delta_val):
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

    return H_kin_off + H_rashba_x - delta_val * KRON_X_0


def compute_sc_E_point(H_x_const, py):
    """Cálculo rápido superconductor (4*Nx x 4*Nx) sin multiplicaciones 3D."""
    total_integral = 0.0
    n_ky_total = len(k_y_values)

    for start_idx in range(0, n_ky_total, KY_CHUNK_SIZE):
        end_idx = min(start_idx + KY_CHUNK_SIZE, n_ky_total)
        M = end_idx - start_idx
        ky_slice = k_y_values[start_idx:end_idx]
        w_slice = W_KY[start_idx:end_idx]

        ky_p = ky_slice + py
        ky_m = -ky_slice + py

        chi_plus = (2.0 * t_hop + gamma * (ky_p**2) - mu)[:, np.newaxis]
        chi_minus = (2.0 * t_hop + gamma * (ky_m**2) - mu)[:, np.newaxis]

        H_tensor = np.broadcast_to(H_x_const, (M, 4 * N_x_sites, 4 * N_x_sites)).copy()

        H_tensor[:, IDX_E_UP, IDX_E_UP] += chi_plus
        H_tensor[:, IDX_E_DN, IDX_E_DN] += chi_plus
        H_tensor[:, IDX_H_UP, IDX_H_UP] -= chi_minus
        H_tensor[:, IDX_H_DN, IDX_H_DN] -= chi_minus

        if Lambda != 0:
            ry_p = (-Lambda * ky_p)[:, np.newaxis]
            ry_m = (-Lambda * ky_m)[:, np.newaxis]
            H_tensor[:, IDX_E_UP, IDX_E_DN] += ry_p
            H_tensor[:, IDX_E_DN, IDX_E_UP] += ry_p
            H_tensor[:, IDX_H_UP, IDX_H_DN] += ry_m
            H_tensor[:, IDX_H_DN, IDX_H_UP] += ry_m

        Z_tensor = np.linalg.eigvalsh(H_tensor)
        ky_profiles = np.sum(eval_thermodynamic_potential(Z_tensor, T, beta), axis=-1)
        total_integral += ky_profiles @ w_slice

    return 0.5 * total_integral / L_x


def process_B_chunk(B_chunk):
    """Calcula únicamente n_yy y E_0 usando solo el stencil longitudinal en py."""
    with threadpool_limits(limits=1):
        N_B_chunk = len(B_chunk)

        B_x_vec = B_chunk * np.cos(theta) * Zeeman * Z_RATIO
        B_y_vec = B_chunk * np.sin(theta) * Zeeman * Z_RATIO
        q_D_x_vec = q_B_constant * B_chunk * np.sin(theta)
        q_D_y_vec = q_B_constant * B_chunk * (-np.cos(theta)) + phi_y

        n_yy_chunk = np.empty(N_B_chunk, dtype=np.float64)

        E0_chunk = np.empty(N_B_chunk, dtype=np.float64)
        for b in range(N_B_chunk):
            bx = B_x_vec[b]
            by = B_y_vec[b]
            px = q_D_x_vec[b]
            py0 = q_D_y_vec[b]
            # Única matriz espacial en x necesaria (px = q_D_x)
            H_x_const = build_x_hopping_matrix(px, Delta) - by * KRON_Y - bx * KRON_X
            # Punto central py = py0
            E0 = compute_sc_E_point(H_x_const, py0) - compute_normal_E_point(py0, bx, by)
            # Punto desplazado py = py0 - h
            E_minus = compute_sc_E_point(H_x_const, py0 - h) - compute_normal_E_point(py0 - h, bx, by)
            # Si bx == 0 y py0 == 0, E(py0 + h) == E(py0 - h) por simetría espejo en y
            if np.isclose(bx, 0.0) and np.isclose(py0, 0.0):
                E_plus = E_minus
            else:
                E_plus = compute_sc_E_point(H_x_const, py0 + h) - compute_normal_E_point(py0 + h, bx, by)
            n_yy_chunk[b] = (E_plus - 2.0 * E0 + E_minus) / (h**2)
            E0_chunk[b] = E0
    return [(n_yy_chunk[b], E0_chunk[b]) for b in range(N_B_chunk)]
    
if __name__ == "__main__":
    t0 = time.perf_counter()
    B_values = np.linspace(0 * Delta, 12 * Delta, points)
    CHUNKS_OF_B = 3
    B_batched_chunks = [B_values[i:i + CHUNKS_OF_B] for i in range(0, len(B_values), CHUNKS_OF_B)]
    with multiprocessing.Pool(n_cores) as pool:
        chunk_results = pool.map(process_B_chunk, B_batched_chunks)
    flat_results = [item for sublist in chunk_results for item in sublist]
    n_yy, E0 = map(np.array, zip(*flat_results))
    output_dir = Path("Data")
    output_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        output_dir / f"obc_stiffness_theta={theta:.3f}_L_x={L_x:.0f}_N_x_sites={N_x_sites}_N={N}_q_B_constant={q_B_constant}_beta={beta}_Lambda={Lambda}_Delta={Delta}.npz",
        B_values=B_values,
        n_yy=n_yy,
        E0=E0,
        parameters=parameters,
        Delta=Delta
        )
    
    print(f"Completado en {time.perf_counter() - t0:.2f} segundos.")
    print("Done! \007")