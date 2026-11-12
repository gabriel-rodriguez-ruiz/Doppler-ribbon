# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:03:30 2026

@author: Gabriel
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARÁMETROS DEL SISTEMA (Unidades: meV y nm)
# ==============================================================================
hbar2_over_2m = 948.0            # meV*nm^2
hbar = 0.6582119                 # meV*ps
m_eff = (hbar**2) / (2.0 * hbar2_over_2m)

Lx = 200.0        # Ancho en x (nm) - Condiciones Abiertas (OBC)
Ly = 800.0        # Largo en y (nm) - Condiciones Periódicas (PBC)
Nx = 150          # Sitios interiores en x
a = Lx / (Nx + 1)

mu = 50.6         # Potencial químico (meV)
Delta0 = 0.08     # Gap superconductor tipo s (meV)
alpha_R = 1 #25.0    # Acoplamiento espín-órbita de Rashba (meV*nm)
Ez = 0.0          # Energía Zeeman por By (meV)

# Velocidad crítica efectiva donde el Zeeman inducido por Rashba iguala a Delta0:
vx_c_rashba = (Delta0 * hbar) / (alpha_R * m_eff)

# ==============================================================================
# 2. MATRICES DE PAULI, PROYECTORES Y OPERADORES BASE
# ==============================================================================
s0 = sp.eye(2, format='csr', dtype=complex)
sx = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
sy = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
sz = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)

P_e = sp.csr_matrix([[1, 0], [0, 0]], dtype=complex)  # Proyector electrones
P_h = sp.csr_matrix([[0, 0], [0, 1]], dtype=complex)  # Proyector huecos

I_Nx = sp.eye(Nx, format='csr', dtype=complex)

t = hbar2_over_2m / (a**2)
t_R = alpha_R / (2.0 * a)
V_x = np.zeros(Nx)

# Grilla discreta de momentos positivos ky >= 0 (simetría ky <-> -ky)
kF = np.sqrt(mu / hbar2_over_2m)
ky_cutoff = 3.0 * kF
ny_max = int(np.floor(ky_cutoff * Ly / (2.0 * np.pi)))
ny_vals = np.arange(0, ny_max + 1)
ky_vals = 2.0 * np.pi * ny_vals / Ly
weights = np.where(ny_vals == 0, 1.0, 2.0)

# ==============================================================================
# 3. FUNCIÓN PARA CALCULAR E_GS CON SUSTITUCIÓN DE PEIERLS EXACTA
# ==============================================================================
def compute_Egs_peierls(vx, delta_val):
    theta = (m_eff * vx * a) / hbar
    phase_plus = np.exp(1j * theta)
    phase_minus = np.exp(-1j * theta)
    
    # --- A. Saltos cinéticos con fase de Peierls ---
    hop_e_right = -t * phase_plus * np.ones(Nx - 1)
    hop_e_left  = -t * phase_minus * np.ones(Nx - 1)
    
    hop_h_right = +t * phase_minus * np.ones(Nx - 1)
    hop_h_left  = +t * phase_plus * np.ones(Nx - 1)
    
    H_kin_off_e = sp.diags([hop_e_right, hop_e_left], offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    H_kin_off_h = sp.diags([hop_h_right, hop_h_left], offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    
    H_kin_off_total = (
        sp.kron(sp.kron(P_e, s0), H_kin_off_e, format='csr') +
        sp.kron(sp.kron(P_h, s0), H_kin_off_h, format='csr')
    )
    
    # --- B. Rashba en x con fase de Peierls ---
    r_e_right = -1j * t_R * phase_plus * np.ones(Nx - 1)
    r_e_left  = +1j * t_R * phase_minus * np.ones(Nx - 1)
    
    r_h_right = +1j * t_R * phase_minus * np.ones(Nx - 1)
    r_h_left  = -1j * t_R * phase_plus * np.ones(Nx - 1)
    
    D_rashba_e = sp.diags([r_e_right, r_e_left], offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    D_rashba_h = sp.diags([r_h_right, r_h_left], offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    
    H_rashba_x_total = (
        sp.kron(sp.kron(P_e, sy), D_rashba_e, format='csr') +
        sp.kron(sp.kron(P_h, sy), D_rashba_h, format='csr')
    )
    
    # --- C. Zeeman y Apareamiento tipo s ---
    Delta_mat = sp.diags(np.full(Nx, delta_val), offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
    H_const = (
        H_kin_off_total +
        H_rashba_x_total +
        sp.kron(sp.kron(sz, sy), Ez * I_Nx, format='csr') +
        sp.kron(sp.kron(sx, s0), Delta_mat, format='csr')
    )
    
    E_gs_total = 0.0
    
    for ky, w in zip(ky_vals, weights):
        ky_term = hbar2_over_2m * (ky**2)
        diag_kin = 2.0 * t + ky_term + V_x - mu
        
        H_diag_kin = sp.diags(diag_kin, offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
        H_rashba_y = sp.kron(sp.kron(sz, sx), (-alpha_R * ky) * I_Nx, format='csr')
        
        H_BdG = (sp.kron(sp.kron(sz, s0), H_diag_kin, format='csr') + H_rashba_y + H_const).toarray()
        
        evals = eigh(H_BdG, eigvals_only=True)
        
        tr_He = 2.0 * np.sum(diag_kin)
        neg_sum = np.sum(evals[evals < 0.0])
        
        E_gs_total += w * 0.5 * (neg_sum + tr_He)
        
    return E_gs_total

# ==============================================================================
# 4. BARRIDO EN VELOCIDAD DOPPLER vx
# ==============================================================================
vx_vals = np.linspace(0.0, 2.2 * vx_c_rashba, 45)
E_sc = np.zeros_like(vx_vals)
E_norm = np.zeros_like(vx_vals)

print(f"Velocidad crítica inducida por Rashba: vx_c = {vx_c_rashba:.2f} nm/ps")
print("Calculando barrido en vx con Peierls exacto...")

for i, vx in enumerate(vx_vals):
    E_sc[i] = compute_Egs_peierls(vx, Delta0)
    E_norm[i] = compute_Egs_peierls(vx, 0.0)

area = Lx * Ly
E_norm_ref = E_norm[0]
E_cond_sc = (E_sc - E_norm_ref) / area
E_cond_norm = (E_norm - E_norm_ref) / area

# ==============================================================================
# 5. GRÁFICA DE RESULTADOS
# ==============================================================================
plt.figure(figsize=(8.5, 5.2))
plt.plot(vx_vals / vx_c_rashba, E_cond_sc, 'o-', color='#1f77b4', lw=2, markersize=4, label=r'Superconductor ($\Delta_0 = 0.08$ meV)')
plt.plot(vx_vals / vx_c_rashba, E_cond_norm, '--', color='#7f7f7f', lw=1.5, label=r'Estado Normal ($\Delta_0 = 0$)')

plt.axvline(1.0, color='red', ls=':', lw=1.2, label=r'Velocidad crítica ($v_x = v_{x,c}$)')
plt.title(f'Energía del Estado Fundamental (Peierls exacto) | $\\mu={mu}$ meV, $\\Delta_0={Delta0}$ meV')
plt.xlabel(r'Velocidad Doppler normalizada $v_x / v_{x,c}$')
plt.ylabel(r'Densidad de energía $(E_{\mathrm{GS}} - E_N(0)) / A$ ($\mathrm{meV/nm}^2$)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# Calcular primera y segunda derivada numérica de E_cond_sc respecto a vx
dE_dv = np.gradient(E_cond_sc, vx_vals)
d2E_dv2 = np.gradient(dE_dv, vx_vals)

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# 1. Energía de condensación
axs[0].plot(vx_vals / vx_c_rashba, E_cond_sc, 'o-', color='#1f77b4', markersize=3)
axs[0].axvline(1.0, color='red', ls=':')
axs[0].set_ylabel(r'$(E_{GS} - E_N)/A$')
axs[0].set_title('Energía y sus derivadas alrededor de $v_{x,c}$')
axs[0].grid(True, alpha=0.3)

# 2. Primera derivada
axs[1].plot(vx_vals / vx_c_rashba, dE_dv, 'o-', color='#2ca02c', markersize=3)
axs[1].axvline(1.0, color='red', ls=':')
axs[1].set_ylabel(r'$\partial E_{GS} / \partial v_x$')
axs[1].grid(True, alpha=0.3)

# 3. Segunda derivada (curvatura)
axs[2].plot(vx_vals / vx_c_rashba, d2E_dv2, 'o-', color='#d62728', markersize=3)
axs[2].axvline(1.0, color='red', ls=':', label=r'$v_x = v_{x,c}$')
axs[2].set_ylabel(r'$\partial^2 E_{GS} / \partial v_x^2$ (Curvatura)')
axs[2].set_xlabel(r'$v_x / v_{x,c}$')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
