
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp

hbar2_over_2m = 948.0
hbar = 0.6582119
m_eff = (hbar**2) / (2.0 * hbar2_over_2m)

Lx = 200.0
Ly = 2000.0
Nx = 100
a = Lx / (Nx + 1)

mu = 50.6
Delta0 = 0.08
alpha_R = 25.0
Ez = 0.05

v_fraunhofer = (np.pi * hbar) / (m_eff * Lx)
area = Lx * Ly

t = hbar2_over_2m / (a**2)
t_R = alpha_R / (2.0 * a)
V_x = np.zeros(Nx)

kF = np.sqrt(mu / hbar2_over_2m)
ky_cutoff = 1.35 * kF
ny_max = int(np.floor(ky_cutoff * Ly / (2.0 * np.pi)))

ny_full = np.arange(-ny_max, ny_max + 1)
ky_full = 2.0 * np.pi * ny_full / Ly

ny_half = np.arange(0, ny_max + 1)
ky_half = 2.0 * np.pi * ny_half / Ly
weights_half = np.where(ny_half == 0, 1.0, 2.0)

s0 = sp.eye(2, format='csr', dtype=complex)
sx = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
sy = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
sz = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)

P_e = sp.csr_matrix([[1, 0], [0, 0]], dtype=complex)
P_h = sp.csr_matrix([[0, 0], [0, 1]], dtype=complex)
I_Nx = sp.eye(Nx, format='csr', dtype=complex)

def compute_Egs_2D_shift(vx, vy, delta_val, use_half_grid=False):
    theta_x = (m_eff * vx * a) / hbar
    phase_plus = np.exp(1j * theta_x)
    phase_minus = np.exp(-1j * theta_x)
    ky_shift = (m_eff * vy) / hbar

    H_kin_off_e = sp.diags(
        [-t * phase_plus * np.ones(Nx - 1), -t * phase_minus * np.ones(Nx - 1)],
        offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
    )
    H_kin_off_h = sp.diags(
        [+t * phase_minus * np.ones(Nx - 1), +t * phase_plus * np.ones(Nx - 1)],
        offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
    )
    H_kin_off = (
        sp.kron(sp.kron(P_e, s0), H_kin_off_e, format='csr') +
        sp.kron(sp.kron(P_h, s0), H_kin_off_h, format='csr')
    )

    D_rashba_e = sp.diags(
        [-1j * t_R * phase_plus * np.ones(Nx - 1), +1j * t_R * phase_minus * np.ones(Nx - 1)],
        offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
    )
    D_rashba_h = sp.diags(
        [+1j * t_R * phase_minus * np.ones(Nx - 1), -1j * t_R * phase_plus * np.ones(Nx - 1)],
        offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
    )
    H_rashba_x = (
        sp.kron(sp.kron(P_e, sy), D_rashba_e, format='csr') +
        sp.kron(sp.kron(P_h, sy), D_rashba_h, format='csr')
    )

    Delta_mat = sp.diags(np.full(Nx, delta_val), offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
    H_base = (
        H_kin_off + H_rashba_x +
        sp.kron(sp.kron(sz, sy), Ez * I_Nx, format='csr') +
        sp.kron(sp.kron(sx, s0), Delta_mat, format='csr')
    ).toarray()

    E_gs_total = 0.0
    idx_e_up = np.arange(0, Nx)
    idx_e_dn = np.arange(Nx, 2 * Nx)
    idx_h_dn = np.arange(2 * Nx, 3 * Nx)
    idx_h_up = np.arange(3 * Nx, 4 * Nx)

    H_BdG = np.empty_like(H_base)
    k_iter = zip(ky_half, weights_half) if use_half_grid else zip(ky_full, np.ones_like(ky_full))

    for ky, w in k_iter:
        np.copyto(H_BdG, H_base)
        ky_e = ky - ky_shift
        ky_h = ky + ky_shift

        diag_e = 2.0 * t + hbar2_over_2m * (ky_e**2) + V_x - mu
        diag_h = -(2.0 * t + hbar2_over_2m * (ky_h**2) + V_x - mu)

        H_BdG[idx_e_up, idx_e_up] += diag_e
        H_BdG[idx_e_dn, idx_e_dn] += diag_e
        H_BdG[idx_h_dn, idx_h_dn] += diag_h
        H_BdG[idx_h_up, idx_h_up] += diag_h

        ry_e = -alpha_R * ky_e
        ry_h = +alpha_R * ky_h
        H_BdG[idx_e_up, idx_e_dn] += ry_e
        H_BdG[idx_e_dn, idx_e_up] += ry_e
        H_BdG[idx_h_dn, idx_h_up] += ry_h
        H_BdG[idx_h_up, idx_h_dn] += ry_h

        evals = np.linalg.eigvalsh(H_BdG)
        tr_He = 2.0 * np.sum(diag_e)
        neg_sum = np.sum(evals[evals < 0.0])
        E_gs_total += w * 0.5 * (neg_sum + tr_He)

    return E_gs_total

def worker_flux_point(idx, vx, dvy, factor_q):
    E_center = compute_Egs_2D_shift(vx, 0.0, Delta0, use_half_grid=True)
    E_plus   = compute_Egs_2D_shift(vx, +dvy, Delta0, use_half_grid=False)
    E_minus  = compute_Egs_2D_shift(vx, -dvy, Delta0, use_half_grid=False)
    Ds_yy_sc = (factor_q / area) * ((E_plus - 2.0 * E_center + E_minus) / (dvy**2))
    return idx, E_center, Ds_yy_sc
