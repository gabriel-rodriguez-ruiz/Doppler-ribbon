# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:59:13 2026

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

tau_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
tau_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
tau_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
sigma_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)

c = 3e17
m_e = 5.1e8 / c**2
m = 0.0403 * m_e
hbar = 6.58e-13
gamma = hbar**2 / (2 * m)
E_F = 50.6
k_F = np.sqrt(E_F / gamma)

Delta = 0.4
mu = E_F
Lambda = 15.0
theta = np.pi / 2
cut_off = 2 * k_F
phi_y = 0.0
h = 2.5e-4
beta = 125.0
Zeeman = True
Z_RATIO = 1.0
q_B_constant = 0.0
B_val = 2.0 * Delta

# Short test ribbon so each eigvalsh call is fast
L_X_BENCH = 300.0
TARGET_AX_LIST = [5.0, 2.5, 1.5, 1.0]

# Compact k_y grid for quick benchmarking
k_so = Lambda / (2 * gamma)
k_F_eff = np.sqrt(k_F**2 + k_so**2)
k_F_out = k_F_eff + k_so
k_limit = k_F_out + 0.1 * k_F
k_y_half = np.unique(np.concatenate([
    np.linspace(0.0, k_limit, 250),
    np.linspace(k_limit, cut_off, 80),
]))
k_y_values = np.unique(np.concatenate([-k_y_half[::-1], k_y_half]))
W_KY = np.gradient(k_y_values)


def omega(eigvals):
    m_neg = np.minimum(0.0, eigvals)
    return m_neg - (1.0 / beta) * np.log1p(np.exp(-beta * np.abs(eigvals)))


def benchmark_ax(L_x, target_ax):
    nx = max(10, int(round(L_x / target_ax)) - 1)
    ax = L_x / (nx + 1)
    t_hop = gamma / (ax**2)
    t_R = Lambda / (2 * ax)

    # Continuum vs. lattice dispersion error at k_F and cut_off
    kin_rel_err_kF = abs((2 * t_hop * (1 - np.cos(k_F * ax)) - gamma * k_F**2) / (gamma * k_F**2))
    kin_rel_err_cut = abs((2 * t_hop * (1 - np.cos(cut_off * ax)) - gamma * cut_off**2) / (gamma * cut_off**2))
    rashba_rel_err_kF = abs((2 * t_R * np.sin(k_F * ax) - Lambda * k_F) / (Lambda * k_F)) if Lambda != 0 else 0.0

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
    r_e0 = sp.diags([-1j * t_R * off, +1j * t_R * off], [1, -1], shape=(nx, nx), format="csr")
    r_h0 = sp.diags([+1j * t_R * off, -1j * t_R * off], [1, -1], shape=(nx, nx), format="csr")

    h_ne_base = (sp.kron(sigma_0, kin_e0, format="csr") + (sp.kron(sigma_y, r_e0, format="csr") if Lambda != 0 else 0)).toarray()
    h_nh_base = (sp.kron(sigma_0, kin_h0, format="csr") + (sp.kron(sigma_y, r_h0, format="csr") if Lambda != 0 else 0)).toarray()

    bx = float(B_val * np.cos(theta) * Zeeman * Z_RATIO)
    by = float(B_val * np.sin(theta) * Zeeman * Z_RATIO)
    px = float(q_B_constant * B_val * np.sin(theta))
    py0 = float(q_B_constant * B_val * (-np.cos(theta)) + phi_y)

    th_x = px * ax
    pp, pm = np.exp(1j * th_x), np.exp(-1j * th_x)
    kin_e = sp.diags([-t_hop * pp * off, -t_hop * pm * off], [1, -1], shape=(nx, nx), format="csr")
    kin_h = sp.diags([+t_hop * pm * off, +t_hop * pp * off], [1, -1], shape=(nx, nx), format="csr")
    h_kin = (sp.kron(np.kron(P_e, sigma_0), kin_e, format="csr") + sp.kron(np.kron(P_h, sigma_0), kin_h, format="csr")).toarray()
    if Lambda != 0:
        re_m = sp.diags([-1j * t_R * pp * off, +1j * t_R * pm * off], [1, -1], shape=(nx, nx), format="csr")
        rh_m = sp.diags([+1j * t_R * pm * off, -1j * t_R * pp * off], [1, -1], shape=(nx, nx), format="csr")
        h_r = (sp.kron(np.kron(P_e, sigma_y), re_m, format="csr") + sp.kron(np.kron(P_h, sigma_y), rh_m, format="csr")).toarray()
    else:
        h_r = 0.0

    hx_const = h_kin + h_r - Delta * kron_x0 - by * kron_y - bx * kron_x
    he_const = h_ne_base - by * n_y - bx * n_x
    hh_const = h_nh_base - by * n_y - bx * n_x

    idx_eu = np.arange(0, nx)
    idx_ed = np.arange(nx, 2 * nx)
    idx_hu = np.arange(2 * nx, 3 * nx)
    idx_hd = np.arange(3 * nx, 4 * nx)

    def energy_at_py(py):
        kp = k_y_values + py
        km = -k_y_values + py
        chip = (2.0 * t_hop + gamma * (kp**2) - mu)[:, None]
        chim = (2.0 * t_hop + gamma * (km**2) - mu)[:, None]

        hsc = np.broadcast_to(hx_const, (len(k_y_values), 4 * nx, 4 * nx)).copy()
        hsc[:, idx_eu, idx_eu] += chip
        hsc[:, idx_ed, idx_ed] += chip
        hsc[:, idx_hu, idx_hu] -= chim
        hsc[:, idx_hd, idx_hd] -= chim

        he = np.broadcast_to(he_const, (len(k_y_values), 2 * nx, 2 * nx)).copy()
        hh = np.broadcast_to(hh_const, (len(k_y_values), 2 * nx, 2 * nx)).copy()
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

        sc_prof = np.sum(omega(np.linalg.eigvalsh(hsc)), axis=-1)
        nm_prof = (
            np.sum(omega(np.linalg.eigvalsh(he)), axis=-1)
            + np.sum(omega(np.linalg.eigvalsh(hh)), axis=-1)
        )
        return 0.5 * ((sc_prof - nm_prof) @ W_KY) / L_x

    t0 = time.perf_counter()
    use_mirror = np.isclose(bx, 0.0) and np.isclose(py0, 0.0)
    e0 = energy_at_py(py0)
    em = energy_at_py(py0 - h)
    ep = em if use_mirror else energy_at_py(py0 + h)
    nyy = (ep - 2.0 * e0 + em) / (h**2)
    dt = time.perf_counter() - t0

    return {
        "nx": nx,
        "ax": ax,
        "kin_err_kF": kin_rel_err_kF,
        "kin_err_cut": kin_rel_err_cut,
        "rashba_err_kF": rashba_rel_err_kF,
        "nyy": nyy,
        "dt": dt,
    }


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        print(f"Benchmark on L_x = {L_X_BENCH} nm, B = {B_val/Delta:.1f}*Delta, k_F = {k_F:.4f} nm^-1\n")
        prev_nyy = None
        for target_ax in TARGET_AX_LIST:
            r = benchmark_ax(L_X_BENCH, target_ax)
            rel_shift = abs(r["nyy"] - prev_nyy) / abs(r["nyy"]) if prev_nyy is not None else np.nan
            shift_str = f"{rel_shift*100:6.2f}%" if np.isfinite(rel_shift) else "   N/A "
            print(
                f"a_x={r['ax']:5.2f} nm (N_x={r['nx']:3d}) | "
                f"kin_err(k_F)={r['kin_err_kF']*100:5.2f}% | "
                f"kin_err(cut)={r['kin_err_cut']*100:5.2f}% | "
                f"R_err(k_F)={r['rashba_err_kF']*100:5.2f}% | "
                f"n_yy={r['nyy']:10.4f} | "
                f"d_nyy={shift_str} | "
                f"t={r['dt']:5.2f} s"
            )
            prev_nyy = r["nyy"]