# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:43:03 2026

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
Ly = 2000.0       # Largo en y (nm) - Condiciones Periódicas (PBC)
Nx = 140          # Sitios interiores en x
a = Lx / (Nx + 1)

mu = 50.6         # Potencial químico (meV)
Delta0 = 0.08     # Gap superconductor tipo s (meV)
alpha_R = 25.0    # Acoplamiento espín-órbita de Rashba (meV*nm)
Ez = 0.05         # Campo Zeeman en y (meV)

v_fraunhofer = (np.pi * hbar) / (m_eff * Lx)
area = Lx * Ly

# ==============================================================================
# 2. MATRICES DE PAULI, PROYECTORES Y OPERADORES BASE
# ==============================================================================
s0 = sp.eye(2, format='csr', dtype=complex)
sx = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
sy = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
sz = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)

P_e = sp.csr_matrix([[1, 0], [0, 0]], dtype=complex)
P_h = sp.csr_matrix([[0, 0], [0, 1]], dtype=complex)
I_Nx = sp.eye(Nx, format='csr', dtype=complex)

t = hbar2_over_2m / (a**2)
t_R = alpha_R / (2.0 * a)
V_x = np.zeros(Nx)

# Grilla completa de ky (necesaria porque vy != 0 rompe simetría ky <-> -ky)
kF = np.sqrt(mu / hbar2_over_2m)
ky_cutoff = 2.2 * kF
ny_max = int(np.floor(ky_cutoff * Ly / (2.0 * np.pi)))
ny_vals = np.arange(-ny_max, ny_max + 1)
ky_vals = 2.0 * np.pi * ny_vals / Ly

# ==============================================================================
# 3. FUNCIÓN PARA CALCULAR E_GS(vx, vy) CON PEIERLS EN x Y DESPLAZAMIENTO EN y
# ==============================================================================
def compute_Egs_2D_shift(vx, vy, delta_val):
    theta_x = (m_eff * vx * a) / hbar
    phase_plus = np.exp(1j * theta_x)
    phase_minus = np.exp(-1j * theta_x)
    
    # Momento de par en y: ky_shift = m * vy / hbar
    ky_shift = (m_eff * vy) / hbar
    
    # Saltos cinéticos en x con Peierls
    H_kin_off_e = sp.diags([-t * phase_plus * np.ones(Nx - 1), -t * phase_minus * np.ones(Nx - 1)],
                           offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    H_kin_off_h = sp.diags([+t * phase_minus * np.ones(Nx - 1), +t * phase_plus * np.ones(Nx - 1)],
                           offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    
    H_kin_off = sp.kron(sp.kron(P_e, s0), H_kin_off_e, format='csr') + \
                sp.kron(sp.kron(P_h, s0), H_kin_off_h, format='csr')
    
    # Rashba en x con Peierls
    D_rashba_e = sp.diags([-1j * t_R * phase_plus * np.ones(Nx - 1), +1j * t_R * phase_minus * np.ones(Nx - 1)],
                          offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    D_rashba_h = sp.diags([+1j * t_R * phase_minus * np.ones(Nx - 1), -1j * t_R * phase_plus * np.ones(Nx - 1)],
                          offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex)
    
    H_rashba_x = sp.kron(sp.kron(P_e, sy), D_rashba_e, format='csr') + \
                 sp.kron(sp.kron(P_h, sy), D_rashba_h, format='csr')
    
    Delta_mat = sp.diags(np.full(Nx, delta_val), offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
    H_const = (
        H_kin_off + H_rashba_x +
        sp.kron(sp.kron(sz, sy), Ez * I_Nx, format='csr') +
        sp.kron(sp.kron(sx, s0), Delta_mat, format='csr')
    )
    
    E_gs_total = 0.0
    
    for ky in ky_vals:
        # Electrones tienen momento (ky - ky_shift); Huecos tienen -(-ky - ky_shift) = (ky + ky_shift)
        ky_e = ky - ky_shift
        ky_h = ky + ky_shift
        
        diag_e = 2.0 * t + hbar2_over_2m * (ky_e**2) + V_x - mu
        diag_h = -(2.0 * t + hbar2_over_2m * (ky_h**2) + V_x - mu)
        
        H_diag_e = sp.diags(diag_e, offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
        H_diag_h = sp.diags(diag_h, offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
        H_diag_kin = sp.kron(sp.kron(P_e, s0), H_diag_e, format='csr') + \
                     sp.kron(sp.kron(P_h, s0), H_diag_h, format='csr')
        
        # Rashba en y: -alpha_R * ky_e para electrones, +alpha_R * ky_h para huecos
        H_rashba_y = sp.kron(sp.kron(P_e, sx), (-alpha_R * ky_e) * I_Nx, format='csr') + \
                     sp.kron(sp.kron(P_h, sx), (+alpha_R * ky_h) * I_Nx, format='csr')
        
        H_BdG = (H_diag_kin + H_rashba_y + H_const).toarray()
        evals = eigh(H_BdG, eigvals_only=True)
        
        # Regularización con Tr[H_e(ky)]
        tr_He = 2.0 * np.sum(diag_e)
        neg_sum = np.sum(evals[evals < 0.0])
        
        E_gs_total += 0.5 * (neg_sum + tr_He)
        
    return E_gs_total

# ==============================================================================
# 4. CÁLCULO DE LA SUPERFLUID STIFFNESS Ds_xx(vx) Y Ds_yy(vx)
# ==============================================================================
flux_vals = np.linspace(0.0, 3.5, 2)    #np.linspace(0.0, 3.5, 55)
vx_vals = flux_vals * v_fraunhofer

# Paso pequeño dvy para la derivada segunda en y
dvy = 1e-3 * v_fraunhofer
factor_q = (hbar / (2.0 * m_eff))**2  # Conversión de d^2E/dv^2 a d^2E/dq^2 (meV*nm^2)

E0_sc = np.zeros_like(vx_vals)
Ds_yy_sc = np.zeros_like(vx_vals)
Ds_yy_norm = np.zeros_like(vx_vals)

print("Calculando E_GS y Superfluid Stiffness Ds_yy(vx)...")
for i, vx in enumerate(vx_vals):
    E_center = compute_Egs_2D_shift(vx, 0.0, Delta0)
    E_plus   = compute_Egs_2D_shift(vx, +dvy, Delta0)
    E_minus  = compute_Egs_2D_shift(vx, -dvy, Delta0)
    
    E0_sc[i] = E_center
    d2E_dvy2 = (E_plus - 2.0 * E_center + E_minus) / (dvy**2)
    Ds_yy_sc[i] = (factor_q / area) * d2E_dvy2
    
    # Referencia del estado normal (Delta0 = 0) para restar rigidez diamagnética de red
    En_c = compute_Egs_2D_shift(vx, 0.0, 0.0)
    En_p = compute_Egs_2D_shift(vx, +dvy, 0.0)
    En_m = compute_Egs_2D_shift(vx, -dvy, 0.0)
    Ds_yy_norm[i] = (factor_q / area) * ((En_p - 2.0 * En_c + En_m) / (dvy**2))

# Rigidez superconductora neta en y (restando contribución normal de fondo)
Ds_yy_net = Ds_yy_sc - Ds_yy_norm

# Rigidez transversal Ds_xx(vx) mediante segunda derivada numérica respecto a vx
d2E_dvx2 = np.gradient(np.gradient(E0_sc, vx_vals), vx_vals)
Ds_xx_sc = (factor_q / area) * d2E_dvx2

# ==============================================================================
# 5. GRÁFICA DE LAS DOS COMPONENTES DE SUPERFLUID STIFFNESS
# ==============================================================================
fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

# Panel 1: Ds_yy(vx) (Rigidez longitudinal a lo largo del canal y)
axs[0].plot(flux_vals, Ds_yy_net, 'o-', color='#1f77b4', lw=2, markersize=4)
for n in range(1, 4):
    axs[0].axvline(n, color='gray', ls='--', lw=0.8, alpha=0.6)
axs[0].axhline(0, color='black', lw=0.8, alpha=0.5)
axs[0].set_title(r'Superfluid Stiffness Longitudinal $D_{s,yy}(\Phi_{\mathrm{eff}}/\Phi_0)$ (Oscilaciones tipo Fraunhofer)')
axs[0].set_ylabel(r'$D_{s,yy}$ (meV)')
axs[0].grid(True, alpha=0.3)

# Panel 2: Ds_xx(vx) (Segunda derivada transversal respecto a vx)
axs[1].plot(flux_vals, Ds_xx_sc, 's-', color='#d62728', lw=2, markersize=4)
for n in range(1, 4):
    axs[1].axvline(n, color='gray', ls='--', lw=0.8, alpha=0.6)
axs[1].axhline(0, color='black', lw=0.8, alpha=0.5)
axs[1].set_title(r'Rigidez Transversal $D_{s,xx} \propto \partial^2 E_{\mathrm{GS}} / \partial v_x^2$')
axs[1].set_xlabel(r'Flujo efectivo $\Phi_{\mathrm{eff}} / \Phi_0 = q_x L_x / (2\pi)$')
axs[1].set_ylabel(r'$D_{s,xx}$ (meV)')
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
