# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:11:06 2026

@author: Gabriel
"""

import os
import time
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# 1. CREAR MÓDULO WORKER AISLADO (Evita deadlocks de Windows spawn y MKL)
# ==============================================================================
WORKER_CODE = textwrap.dedent("""
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
""")

with open("_bdg_worker.py", "w", encoding="utf-8") as f:
    f.write(WORKER_CODE)

import _bdg_worker

# ==============================================================================
# 2. EJECUCIÓN PARALELA EN WINDOWS Y EXPORTACIÓN
# ==============================================================================
if __name__ == "__main__":
    flux_vals = np.linspace(0.0, 3.5, 31)
    vx_vals = flux_vals * _bdg_worker.v_fraunhofer

    dvy = 1e-3 * _bdg_worker.v_fraunhofer
    factor_q = (_bdg_worker.hbar / (2.0 * _bdg_worker.m_eff))**2

    E0_sc = np.zeros_like(vx_vals)
    Ds_yy_sc = np.zeros_like(vx_vals)

    num_workers = min(os.cpu_count() or 8, 16)
    print(f"Lanzando {len(vx_vals)} puntos de flujo en {num_workers} procesos independientes...")
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_bdg_worker.worker_flux_point, i, vx_vals[i], dvy, factor_q)
            for i in range(len(vx_vals))
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Flux Sweep", unit="pt"):
            idx, E_center, ds_yy = future.result()
            E0_sc[idx] = E_center
            Ds_yy_sc[idx] = ds_yy

    elapsed = time.perf_counter() - t0
    print(f"Barrido paralelo completado en {elapsed:.2f} segundos.")

    d2E_dvx2 = np.gradient(np.gradient(E0_sc, vx_vals), vx_vals)
    Ds_xx_sc = (factor_q / _bdg_worker.area) * d2E_dvx2

    output_filename = f"superfluid_stiffness_alphaR_{_bdg_worker.alpha_R}_Ez_{_bdg_worker.Ez}.npz"
    np.savez_compressed(
        output_filename,
        flux_vals=flux_vals,
        vx_vals=vx_vals,
        Ds_yy_sc=Ds_yy_sc,
        Ds_xx_sc=Ds_xx_sc,
        E0_sc=E0_sc,
        Lx=_bdg_worker.Lx,
        Ly=_bdg_worker.Ly,
        Nx=_bdg_worker.Nx,
        mu=_bdg_worker.mu,
        Delta0=_bdg_worker.Delta0,
        alpha_R=_bdg_worker.alpha_R,
        Ez=_bdg_worker.Ez,
        hbar2_over_2m=_bdg_worker.hbar2_over_2m,
    )
    print(f"Datos exportados a '{output_filename}'.")

    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

    axs.plot(flux_vals, Ds_yy_sc, 'o-', color='#1f77b4', lw=2, markersize=4)
    for n in range(1, 4):
        axs.axvline(n, color='gray', ls='--', lw=0.8, alpha=0.6)
    axs.axhline(0, color='black', lw=0.8, alpha=0.5)
    axs.set_title(r'Longitudinal Superfluid Stiffness $D_{s,yy}(\Phi_{\mathrm{eff}}/\Phi_0)$')
    axs.set_ylabel(r'$D_{s,yy}$ (meV)')
    axs.grid(True, alpha=0.3)

    axs.plot(flux_vals, Ds_xx_sc, 's-', color='#d62728', lw=2, markersize=4)
    for n in range(1, 4):
        axs.axvline(n, color='gray', ls='--', lw=0.8, alpha=0.6)
    axs.axhline(0, color='black', lw=0.8, alpha=0.5)
    axs.set_title(r'Transverse Stiffness $D_{s,xx} \propto \partial^2 E_{\mathrm{GS}} / \partial v_x^2$')
    axs.set_xlabel(r'Effective Flux $\Phi_{\mathrm{eff}} / \Phi_0 = q_x L_x / (2\pi)$')
    axs.set_ylabel(r'$D_{s,xx}$ (meV)')
    axs.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
