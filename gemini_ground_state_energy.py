# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:13 2026

@author: Gabriel
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

# ==============================================================================
# 1. PARÁMETROS DEL SISTEMA (Unidades: meV y nm)
# ==============================================================================
hbar2_over_2m = 948  #38.0998 / 0.067  # meV*nm^2 (ej. GaAs)
hbar = 0.6582119                 # meV*ps

Lx = 150.0        # Ancho en x (nm) - Condiciones Abiertas (OBC)
Ly = 1000.0       # Largo en y (nm) - Condiciones Periódicas (PBC)
Nx = 100          # Sitios interiores en x (matriz total: 400 x 400)
a = Lx / (Nx + 1)

mu = 50.6          # Potencial químico (meV)
Delta0 = 0.08      # Gap tipo s (meV)
Ez = 0          # Energía Zeeman por By (meV)
vx = 10.0         # Velocidad Doppler en x (nm/ps)

t = hbar2_over_2m / (a**2)
t_D = (hbar * vx) / (2.0 * a)
assert abs(t_D) < t, "Aumenta Nx para evitar inestabilidad numérica en el término Doppler."

# ==============================================================================
# 2. CONSTRUCCIÓN DE BLOQUES SPARSE
# ==============================================================================
s0 = sp.eye(2, format='csr', dtype=complex)
sx = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
sy = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
sz = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)
I_Nx = sp.eye(Nx, format='csr', dtype=complex)

V_x = np.zeros(Nx)
Delta_x = np.full(Nx, Delta0)

# Matriz Doppler en x
H_Doppler = sp.diags(
    [1j * t_D * np.ones(Nx - 1), -1j * t_D * np.ones(Nx - 1)],
    offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
)

H_Zeeman_space = Ez * I_Nx
Delta_mat = sp.diags(Delta_x, offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)

# Parte independiente de ky en la base: (c_{k,up}, c_{k,down}, c^\dagger_{-k,down}, -c^\dagger_{-k,up})^T
H_const = (
    sp.kron(sp.kron(s0, s0), H_Doppler, format='csr') +
    sp.kron(sp.kron(sz, sy), H_Zeeman_space, format='csr') +
    sp.kron(sp.kron(sx, s0), Delta_mat, format='csr')
)

def build_H_BdG_and_He_trace(ky):
    """Construye H_BdG(ky) y devuelve también Tr[H_e(ky)] para regularizar la suma."""
    ky_term = hbar2_over_2m * (ky**2)
    diag_kin = 2.0 * t + ky_term + V_x - mu
    off_kin = -t * np.ones(Nx - 1)
    
    H_kin = sp.diags([diag_kin, off_kin, off_kin], offsets=[0, 1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    H_BdG = sp.kron(sp.kron(sz, s0), H_kin, format='csr') + H_const
    
    # Traza del bloque de electrones H_e(ky) (tamaño 2*Nx x 2*Nx):
    # Como H_Doppler y sigma_y tienen traza cero, Tr[H_e] = 2 * Tr[H_kin]
    tr_He = 2.0 * np.sum(diag_kin)
    
    return H_BdG.toarray(), tr_He

# ==============================================================================
# 3. CÁLCULO DE LA ENERGÍA DEL ESTADO FUNDAMENTAL (SUMA SOBRE ky)
# ==============================================================================
def compute_ground_state_energy(ky_cutoff):
    """
    Suma sobre los modos discretos ky = 2*pi*ny / Ly hasta un cutoff |ky| <= ky_cutoff.
    """
    ny_max = int(np.floor(ky_cutoff * Ly / (2.0 * np.pi)))
    ny_vals = np.arange(-ny_max, ny_max + 1)
    ky_vals = 2.0 * np.pi * ny_vals / Ly
    
    E_gs_total = 0.0
    
    for ky in ky_vals:
        H_dense, tr_He = build_H_BdG_and_He_trace(ky)
        
        # Calculamos TODOS los 4*Nx autovalores reales usando LAPACK (eigh)
        evals = eigh(H_dense, eigvals_only=True)
        
        # Sumamos solo los autovalores negativos + la traza de regularización
        neg_evals_sum = np.sum(evals[evals < 0.0])
        E_gs_total += 0.5 * (neg_evals_sum + tr_He)
        
    return E_gs_total, len(ky_vals)

# ==============================================================================
# 4. PRUEBA DE CONVERGENCIA EN EL CUTOFF DE ky
# ==============================================================================
kF = np.sqrt(mu / hbar2_over_2m)

for factor in [2.0, 4.0, 6.0]:
    cutoff = factor * kF
    E_gs, n_modes = compute_ground_state_energy(cutoff)
    # Energía por unidad de área (meV / nm^2)
    energy_density = E_gs / (Lx * Ly)
    print(f"Cutoff = {factor}*kF ({n_modes} modos ky) -> E_GS = {E_gs:.4f} meV | Densidad = {energy_density:.6e} meV/nm^2")
