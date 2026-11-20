# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:09:00 2026

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

