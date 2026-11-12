# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:32:54 2026

@author: Gabriel
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARÁMETROS DEL SISTEMA (Unidades: meV y nm)
# ==============================================================================
hbar2_over_2m = 948  # meV*nm^2 (ej. GaAs: m* = 0.067 m_e)
hbar = 0.6582119                 # meV*ps
m_eff = (hbar**2) / (2.0 * hbar2_over_2m)  # Masa efectiva en meV*ps^2/nm^2

Lx = 150.0        # Ancho en x (nm) - Condiciones Abiertas (OBC)
Ly = 800.0        # Largo en y (nm) - Condiciones Periódicas (PBC)
Nx = 120          # Sitios interiores en x (matriz total: 480 x 480)
a = Lx / (Nx + 1)

mu = 50.6         # Potencial químico (meV)
Delta0 = 0.08      # Gap superconductor tipo s (meV)
Ez = 0.0          # Energía Zeeman por By (meV) - puedes cambiarlo a ej. 0.2

# Velocidad de Fermi y velocidad crítica de Landau (referencia física)
kF = np.sqrt(mu / hbar2_over_2m)
vF = (hbar * kF) / m_eff
vc_landau = Delta0 / (hbar * kF)  # Velocidad crítica de despareamiento (bulk)

# ==============================================================================
# 2. MATRICES DE PAULI Y OPERADORES BASE
# ==============================================================================
s0 = sp.eye(2, format='csr', dtype=complex)
sx = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
sy = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
sz = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)
I_Nx = sp.eye(Nx, format='csr', dtype=complex)

t = hbar2_over_2m / (a**2)
V_x = np.zeros(Nx)

# Grilla discreta de momentos positivos ky >= 0 (aprovechamos simetría ky <-> -ky)
ky_cutoff = 4.0 * kF
ny_max = int(np.floor(ky_cutoff * Ly / (2.0 * np.pi)))
ny_vals = np.arange(0, ny_max + 1)
ky_vals = 2.0 * np.pi * ny_vals / Ly
weights = np.where(ny_vals == 0, 1.0, 2.0)  # Peso 1 para ky=0, peso 2 para ky>0

# ==============================================================================
# 3. FUNCIÓN PARA CALCULAR E_GS DADOS (vx, Delta0)
# ==============================================================================
def compute_Egs(vx, delta_val, include_galilean_shift=True):
    t_D = (hbar * vx) / (2.0 * a)
    if abs(t_D) >= t:
        raise ValueError(f"Inestabilidad numérica para vx={vx:.2f}: |t_D| >= t. Aumenta Nx.")
    
    # Energía cinética extra (1/2) m vx^2 proveniente de la transformación de calibre
    galilean_energy = 0.5 * m_eff * (vx**2) if include_galilean_shift else 0.0
    
    # Matriz Doppler: i * hbar * vx * d/dx
    H_Doppler = sp.diags(
        [1j * t_D * np.ones(Nx - 1), -1j * t_D * np.ones(Nx - 1)],
        offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
    )
    
    Delta_mat = sp.diags(np.full(Nx, delta_val), offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
    H_Zeeman_space = Ez * I_Nx
    
    # Parte independiente de ky en la base (c_{k,up}, c_{k,down}, c^\dagger_{-k,down}, -c^\dagger_{-k,up})^T
    H_const = (
        sp.kron(sp.kron(s0, s0), H_Doppler, format='csr') +
        sp.kron(sp.kron(sz, sy), H_Zeeman_space, format='csr') +
        sp.kron(sp.kron(sx, s0), Delta_mat, format='csr')
    )
    
    off_kin = -t * np.ones(Nx - 1)
    E_gs_total = 0.0
    
    for ky, w in zip(ky_vals, weights):
        ky_term = hbar2_over_2m * (ky**2)
        diag_kin = 2.0 * t + ky_term + V_x - mu + galilean_energy
        
        H_kin = sp.diags([diag_kin, off_kin, off_kin], offsets=[0, 1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
        H_BdG = (sp.kron(sp.kron(sz, s0), H_kin, format='csr') + H_const).toarray()
        
        # Todos los autovalores reales
        evals = eigh(H_BdG, eigvals_only=True)
        
        # Regularización con Tr[H_e] = 2 * Tr[H_kin]
        tr_He = 2.0 * np.sum(diag_kin)
        neg_sum = np.sum(evals[evals < 0.0])
        
        E_gs_total += w * 0.5 * (neg_sum + tr_He)
        
    return E_gs_total

# ==============================================================================
# 4. BARRIDO EN VELOCIDAD DOPPLER vx
# ==============================================================================
vx_vals = np.linspace(0.0, 2.0 * vc_landau, 35)
E_sc = np.zeros_like(vx_vals)
E_norm = np.zeros_like(vx_vals)

print(f"Velocidad crítica de Landau estimada: vc = {vc_landau:.2f} nm/ps")
print("Calculando barrido en vx...")

for i, vx in enumerate(vx_vals):
    E_sc[i] = compute_Egs(vx, Delta0, include_galilean_shift=True)
    E_norm[i] = compute_Egs(vx, 0.0, include_galilean_shift=True)

# Energía de condensación relativa al estado normal en reposo (meV / nm^2)
area = Lx * Ly
E_cond_sc = (E_sc - E_norm[0]) / area
E_cond_norm = (E_norm - E_norm[0]) / area

# ==============================================================================
# 5. GRÁFICA DE RESULTADOS
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(vx_vals / vc_landau, E_cond_sc, 'o-', color='#1f77b4', lw=2, markersize=4, label=r'Superconductor ($\Delta_0 > 0$)')
plt.plot(vx_vals / vc_landau, E_cond_norm, '--', color='#7f7f7f', lw=1.5, label=r'Estado Normal ($\Delta_0 = 0$)')

plt.axvline(1.0, color='red', ls=':', lw=1.2, label=r'Límite de Landau ($v_x = v_c$)')
plt.axhline(0.0, color='black', lw=0.8, alpha=0.5)

plt.title(f'Energía del Estado Fundamental vs. Doppler Shift ($E_Z = {Ez}$ meV)')
plt.xlabel(r'Velocidad Doppler normalizada $v_x / v_c$')
plt.ylabel(r'Densidad de energía $(E_{\mathrm{GS}} - E_N(0)) / A$ ($\mathrm{meV/nm}^2$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
