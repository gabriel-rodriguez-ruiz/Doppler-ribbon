# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:11:03 2026

@author: Gabriel
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARÁMETROS IDÉNTICOS A TU SIMULACIÓN CONVERGIDA (N_x = 100)
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

L_x = 750.0                   # nm
N_x_sites = 100               # Grilla convergida
a_x = L_x / (N_x_sites + 1)
t_hop = gamma / (a_x**2)
t_R = Lambda / (2 * a_x)

# ==============================================================================
# 2. CONSTRUCCIÓN DE OPERADORES Y CÁLCULO DE BANDAS E_n(k_y)
# ==============================================================================
P_e = (tau_0 + tau_z) / 2.0
P_h = (tau_0 - tau_z) / 2.0
I_Nx = np.eye(N_x_sites, dtype=np.complex128)

KRON_Y = np.kron(np.kron(tau_0, sigma_y), I_Nx)
KRON_X_0 = np.kron(np.kron(tau_x, sigma_0), I_Nx)

off_ones = np.ones(N_x_sites - 1, dtype=np.complex128)
kin_e = sp.diags([-t_hop * off_ones, -t_hop * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
kin_h = sp.diags([+t_hop * off_ones, +t_hop * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
rashba_e = sp.diags([-1j * t_R * off_ones, +1j * t_R * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')
rashba_h = sp.diags([+1j * t_R * off_ones, -1j * t_R * off_ones], offsets=[1, -1], shape=(N_x_sites, N_x_sites), format='csr')

H_X_0 = (
    sp.kron(np.kron(P_e, sigma_0), kin_e) + sp.kron(np.kron(P_h, sigma_0), kin_h)
    + sp.kron(np.kron(P_e, sigma_y), rashba_e) + sp.kron(np.kron(P_h, sigma_y), rashba_h)
).toarray() - Delta * KRON_X_0

IDX_E_UP = np.arange(0, N_x_sites)
IDX_E_DN = np.arange(N_x_sites, 2 * N_x_sites)
IDX_H_UP = np.arange(2 * N_x_sites, 3 * N_x_sites)
IDX_H_DN = np.arange(3 * N_x_sites, 4 * N_x_sites)


def compute_spectrum_ky(B_over_Delta, ky_grid):
    """Devuelve los autovalores E_n(k_y) para un valor dado de B / Delta (theta = pi/2)."""
    by = B_over_Delta * Delta
    H_base = H_X_0 - by * KRON_Y

    M = len(ky_grid)
    chi = (2.0 * t_hop + gamma * (ky_grid**2) - mu)[:, np.newaxis]
    ry = (-Lambda * ky_grid)[:, np.newaxis]

    H_tensor = np.broadcast_to(H_base, (M, 4 * N_x_sites, 4 * N_x_sites)).copy()
    H_tensor[:, IDX_E_UP, IDX_E_UP] += chi
    H_tensor[:, IDX_E_DN, IDX_E_DN] += chi
    H_tensor[:, IDX_H_UP, IDX_H_UP] -= chi
    H_tensor[:, IDX_H_DN, IDX_H_DN] -= chi

    H_tensor[:, IDX_E_UP, IDX_E_DN] += ry
    H_tensor[:, IDX_E_DN, IDX_E_UP] += ry
    H_tensor[:, IDX_H_UP, IDX_H_DN] -= ry  # -Lambda * (-ky) = +Lambda * ky
    H_tensor[:, IDX_H_DN, IDX_H_UP] -= ry

    return np.linalg.eigvalsh(H_tensor)


if __name__ == "__main__":
    # Ventana enfocada en k_y alrededor de 0 para ver los fondos de sub-banda cruzando E = 0
    ky_vals = np.linspace(-0.08, 0.08, 240)
    B_targets = [6.2, 8.0, 10]  #[4.0, 6.2, 8.0]
    titles = [
        r"$B/\Delta = 6.2$ ",
        r"$B/\Delta = 8$ ",
        r"$B/\Delta = 10.0$ "
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, b_ratio, title in zip(axs, B_targets, titles):
        evals = compute_spectrum_ky(b_ratio, ky_vals)
        # Graficamos las 28 bandas más cercanas a E = 0
        mid = evals.shape[1] // 2
        for band_idx in range(mid - 45, mid + 45):
            ax.plot(ky_vals, evals[:, band_idx] / Delta, color="#1f77b4", lw=1.1, alpha=0.8)

        ax.axhline(0.0, color="#d62728", ls="--", lw=1.3, label=r"$E = 0$ (Nivel de Fermi)")
        ax.axvline(0.0, color="gray", ls=":", lw=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$k_y\ (\mathrm{nm}^{-1})$")
        ax.set_ylim(-3.5, 3.5)
        ax.grid(True, alpha=0.25)

    axs[0].set_ylabel(r"Energía de cuasipartícula $E_n(k_y) / \Delta$")
    axs[0].legend(loc="upper right")
    plt.tight_layout()
    plt.show()
