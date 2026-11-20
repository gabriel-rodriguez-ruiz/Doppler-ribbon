# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:51:16 2026

@author: Gabriel
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import scipy.sparse as sp
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
# 2. PARÁMETROS BASE (CONSISTENTES CON TU SCRIPT)
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
phi_y = 0.0
h = 2.5e-4
T = True
beta = 125.0
Zeeman = True
Z_RATIO = 1.0
q_B_constant = 0.0

L_x = 750.0                   # nm
N_x_sites = 100
a_x = L_x / (N_x_sites + 1)
t_hop = gamma / (a_x**2)
t_R = Lambda / (2 * a_x)

N_KY_INNER = 1000
KY_CHUNK_SIZE = 32

# ==============================================================================
# 3. OPERADORES BASE
# ==============================================================================
P_e = (tau_0 + tau_z) / 2.0
P_h = (tau_0 - tau_z) / 2.0
I_Nx = np.eye(N_x_sites, dtype=np.complex128)
off_ones = np.ones(N_x_sites - 1, dtype=np.complex128)

KRON_Y = np.kron(np.kron(tau_0, sigma_y), I_Nx)
KRON_X = np.kron(np.kron(tau_0, sigma_x), I_Nx)
KRON_X_0 = np.kron(np.kron(tau_x, sigma_0), I_Nx)

N_Y = np.kron(sigma_y, I_Nx)
N_X = np.kron(sigma_x, I_Nx)

kin_e_zero = sp.diags([-t_hop * off_ones, -t_hop * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
kin_h_zero = sp.diags([+t_hop * off_ones, +t_hop * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
rashba_e_zero = sp.diags([-1j * t_R * off_ones, +1j * t_R * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")
rashba_h_zero = sp.diags([+1j * t_R * off_ones, -1j * t_R * off_ones], [1, -1], shape=(N_x_sites, N_x_sites), format="csr")

H_NORM_E_BASE = (
    sp.kron(sigma_0, kin_e_zero, format="csr")
    + (sp.kron(sigma_y, rashba_e_zero, format="csr") if Lambda != 0 else 0)
).toarray()

H_NORM_H_BASE = (
    sp.kron(sigma_0, kin_h_zero, format="csr")
    + (sp.kron(sigma_y, rashba_h_zero, format="csr") if Lambda != 0 else 0)
).toarray()

IDX_E_UP = np.arange(0, N_x_sites)
IDX_E_DN = np.arange(N_x_sites, 2 * N_x_sites)
IDX_H_UP = np.arange(2 * N_x_sites, 3 * N_x_sites)
IDX_H_DN = np.arange(3 * N_x_sites, 4 * N_x_sites)


def build_ky_grid(cut_off_val):
    k_so = Lambda / (2 * gamma)
    k_F_eff = np.sqrt(k_F**2 + k_so**2)
    k_F_out = k_F_eff + k_so
    k_limit = min(k_F_out + 0.1 * k_F, 0.9 * cut_off_val)
    n_outer = max(300, int(round(300 * (cut_off_val - k_limit) / max(k_F, 1e-9))))

    k_y_half = np.unique(np.concatenate([
        np.linspace(0.0, k_limit, N_KY_INNER),
        np.linspace(k_limit, cut_off_val, n_outer)
    ]))
    k_y_values = np.unique(np.concatenate([-k_y_half[::-1], k_y_half]))
    w_ky = np.gradient(k_y_values)
    return k_y_values, w_ky


def eval_thermodynamic_potential(eigvals):
    if T:
        stable_term = np.minimum(0.0, eigvals)
        log_term = -(1.0 / beta) * np.log1p(np.exp(-beta * np.abs(eigvals)))
        return stable_term + log_term
    return np.minimum(0.0, eigvals)


def build_x_hopping_matrix(px, delta_val):
    theta_x = px * a_x
    phase_plus = np.exp(1j * theta_x)
    phase_minus = np.exp(-1j * theta_x)

    kin_e = sp.diags([-t_hop * phase_plus * off_ones, -t_hop * phase_minus * off_ones],
                     offsets=[1, -1], shape=(N_x_sites, N_x_sites), format="csr")
    kin_h = sp.diags([+t_hop * phase_minus * off_ones, +t_hop * phase_plus * off_ones],
                     offsets=[1, -1], shape=(N_x_sites, N_x_sites), format="csr")

    H_kin_off = (
        sp.kron(np.kron(P_e, sigma_0), kin_e, format="csr") +
        sp.kron(np.kron(P_h, sigma_0), kin_h, format="csr")
    ).toarray()

    if Lambda != 0:
        rashba_e = sp.diags([-1j * t_R * phase_plus * off_ones, +1j * t_R * phase_minus * off_ones],
                            offsets=[1, -1], shape=(N_x_sites, N_x_sites), format="csr")
        rashba_h = sp.diags([+1j * t_R * phase_minus * off_ones, -1j * t_R * phase_plus * off_ones],
                            offsets=[1, -1], shape=(N_x_sites, N_x_sites), format="csr")
        H_rashba_x = (
            sp.kron(np.kron(P_e, sigma_y), rashba_e, format="csr") +
            sp.kron(np.kron(P_h, sigma_y), rashba_h, format="csr")
        ).toarray()
    else:
        H_rashba_x = 0.0

    return H_kin_off + H_rashba_x - delta_val * KRON_X_0


def compute_normal_E_point(py, bx, by, k_y_values, w_ky):
    H_e_const = H_NORM_E_BASE - by * N_Y - bx * N_X
    H_h_const = H_NORM_H_BASE - by * N_Y - bx * N_X

    total_integral = 0.0
    n_ky_total = len(k_y_values)

    for start_idx in range(0, n_ky_total, KY_CHUNK_SIZE):
        end_idx = min(start_idx + KY_CHUNK_SIZE, n_ky_total)
        M = end_idx - start_idx
        ky_slice = k_y_values[start_idx:end_idx]
        w_slice = w_ky[start_idx:end_idx]

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
            np.sum(eval_thermodynamic_potential(eig_e), axis=-1)
            + np.sum(eval_thermodynamic_potential(eig_h), axis=-1)
        )
        total_integral += ky_profiles @ w_slice

    return 0.5 * total_integral / L_x


def compute_sc_E_point(H_x_const, py, k_y_values, w_ky):
    total_integral = 0.0
    n_ky_total = len(k_y_values)

    for start_idx in range(0, n_ky_total, KY_CHUNK_SIZE):
        end_idx = min(start_idx + KY_CHUNK_SIZE, n_ky_total)
        M = end_idx - start_idx
        ky_slice = k_y_values[start_idx:end_idx]
        w_slice = w_ky[start_idx:end_idx]

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
        ky_profiles = np.sum(eval_thermodynamic_potential(Z_tensor), axis=-1)
        total_integral += ky_profiles @ w_slice

    return 0.5 * total_integral / L_x


def evaluate_stiffness_components(B_val, delta_val, cut_off_val):
    k_y_values, w_ky = build_ky_grid(cut_off_val)

    bx = float(B_val * np.cos(theta) * Zeeman * Z_RATIO)
    by = float(B_val * np.sin(theta) * Zeeman * Z_RATIO)
    px = float(q_B_constant * B_val * np.sin(theta))
    py0 = float(q_B_constant * B_val * (-np.cos(theta)) + phi_y)

    H_x_const = build_x_hopping_matrix(px, delta_val) - by * KRON_Y - bx * KRON_X
    use_mirror = np.isclose(bx, 0.0) and np.isclose(py0, 0.0)

    Esc0 = compute_sc_E_point(H_x_const, py0, k_y_values, w_ky)
    Esc_m = compute_sc_E_point(H_x_const, py0 - h, k_y_values, w_ky)
    Esc_p = Esc_m if use_mirror else compute_sc_E_point(H_x_const, py0 + h, k_y_values, w_ky)

    En0 = compute_normal_E_point(py0, bx, by, k_y_values, w_ky)
    En_m = compute_normal_E_point(py0 - h, bx, by, k_y_values, w_ky)
    En_p = En_m if use_mirror else compute_normal_E_point(py0 + h, bx, by, k_y_values, w_ky)

    nyy_sc = (Esc_p - 2.0 * Esc0 + Esc_m) / (h**2)
    nyy_norm = (En_p - 2.0 * En0 + En_m) / (h**2)
    nyy_tot = nyy_sc - nyy_norm
    E0_tot = Esc0 - En0

    return {
        "E0_sc": Esc0,
        "E0_norm": En0,
        "E0_tot": E0_tot,
        "nyy_sc": nyy_sc,
        "nyy_norm": nyy_norm,
        "nyy_tot": nyy_tot,
    }


if __name__ == "__main__":
    t_start = time.perf_counter()
    with threadpool_limits(limits=4):
        print(f"=== PRUEBAS DE CONSISTENCIA (L_x = {L_x} nm, N_x = {N_x_sites}, a_x = {a_x:.2f} nm) ===\n")

        # 1. Prueba exacta Delta = 0 en B = 2*Delta
        print("--- 1. Prueba identidad Delta = 0 (B = 2*Delta, cut_off = 2*k_F) ---")
        res_d0 = evaluate_stiffness_components(B_val=2.0 * Delta, delta_val=0.0, cut_off_val=2.0 * k_F)
        print(f"E0_sc - E0_norm (debe ser ~0):   {res_d0['E0_tot']:.3e}")
        print(f"nyy_sc - nyy_norm (debe ser ~0): {res_d0['nyy_tot']:.3e}")
        print(f"(Valores individuales: nyy_sc = {res_d0['nyy_sc']:.6f}, nyy_norm = {res_d0['nyy_norm']:.6f})\n")

        # 2. Sensibilidad al cut_off en B = 2*Delta
        print("--- 2. Sensibilidad a cut_off en B = 2*Delta ---")
        for mult in [1.5, 2.0, 3.0, 4.0]:
            t_c = time.perf_counter()
            r = evaluate_stiffness_components(B_val=2.0 * Delta, delta_val=Delta, cut_off_val=mult * k_F)
            dt = time.perf_counter() - t_c
            print(
                f"cut_off = {mult:3.1f}*k_F | "
                f"nyy_sc = {r['nyy_sc']:11.4f} | "
                f"nyy_norm = {r['nyy_norm']:11.4f} | "
                f"nyy_tot = {r['nyy_tot']:10.4f} | "
                f"E0_tot = {r['E0_tot']:11.6f} | ({dt:.2f} s)"
            )

        # 3. Referencia en B = 0 vs B = 2*Delta
        print("\n--- 3. Comparación de escala B = 0 vs B = 2*Delta (cut_off = 2*k_F) ---")
        r_b0 = evaluate_stiffness_components(B_val=0.0, delta_val=Delta, cut_off_val=2.0 * k_F)
        r_b2 = evaluate_stiffness_components(B_val=2.0 * Delta, delta_val=Delta, cut_off_val=2.0 * k_F)
        print(f"Referencia analítica 2D limpia (sin Rashba, B=0): E_F / pi = {E_F / np.pi:.4f}")
        print(f"B = 0.0*Delta -> nyy_sc = {r_b0['nyy_sc']:10.4f}, nyy_norm = {r_b0['nyy_norm']:10.4f}, nyy_tot = {r_b0['nyy_tot']:10.4f}")
        print(f"B = 2.0*Delta -> nyy_sc = {r_b2['nyy_sc']:10.4f}, nyy_norm = {r_b2['nyy_norm']:10.4f}, nyy_tot = {r_b2['nyy_tot']:10.4f}")

    print(f"\nTiempo total de pruebas: {time.perf_counter() - t_start:.2f} s")