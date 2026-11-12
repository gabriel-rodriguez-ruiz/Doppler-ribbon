# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:41:50 2026

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
Ly = 2000.0       # Largo en y (nm) - Aumentado para mejor resolución en ky
Nx = 160          # Sitios interiores en x
a = Lx / (Nx + 1)

mu = 50.6         # Potencial químico (meV)
Delta0 = 0.08     # Gap superconductor tipo s (meV)
alpha_R = 0 #25.0    # Acoplamiento espín-órbita de Rashba (meV*nm)
Ez = 0.05         # Campo Zeeman en y (meV)

# Escala de velocidad de Fraunhofer: q_x * L_x = 2 * pi  =>  Phi_eff / Phi_0 = 1
v_fraunhofer = (np.pi * hbar) / (m_eff * Lx)

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
x_grid = np.linspace(a, Lx - a, Nx)

# Grilla de momentos ky >= 0
kF = np.sqrt(mu / hbar2_over_2m)
ky_cutoff = 2.0 * kF
ny_max = int(np.floor(ky_cutoff * Ly / (2.0 * np.pi)))
ny_vals = np.arange(0, ny_max + 1)
ky_vals = 2.0 * np.pi * ny_vals / Ly
weights = np.where(ny_vals == 0, 1.0, 2.0)

# ==============================================================================
# 3. FUNCIÓN PARA CONSTRUIR H_BdG(ky, vx)
# ==============================================================================
def build_H_BdG_peierls(ky, vx, delta_val):
    theta = (m_eff * vx * a) / hbar
    phase_plus = np.exp(1j * theta)
    phase_minus = np.exp(-1j * theta)
    
    # Saltos cinéticos con Peierls
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
    
    # Diagonal cinética + Rashba en y + Zeeman + Apareamiento
    diag_kin = 2.0 * t + hbar2_over_2m * (ky**2) + V_x - mu
    H_diag_kin = sp.diags(diag_kin, offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
    H_rashba_y = sp.kron(sp.kron(sz, sx), (-alpha_R * ky) * I_Nx, format='csr')
    Delta_mat = sp.diags(np.full(Nx, delta_val), offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)
    
    H_BdG = (
        sp.kron(sp.kron(sz, s0), H_diag_kin, format='csr') +
        H_kin_off + H_rashba_x + H_rashba_y +
        sp.kron(sp.kron(sz, sy), Ez * I_Nx, format='csr') +
        sp.kron(sp.kron(sx, s0), Delta_mat, format='csr')
    )
    return H_BdG.toarray(), 2.0 * np.sum(diag_kin)

# ==============================================================================
# 4. BARRIDO EN FLUJO EFECTIVO (Phi_eff / Phi_0) PARA VER OSCILACIONES FRAUNHOFER
# ==============================================================================
flux_vals = np.linspace(0.0, 4.0, 60)  # De 0 a 4 cuantos de flujo efectivos en Lx
vx_vals = flux_vals * v_fraunhofer

E_sc = np.zeros_like(vx_vals)
min_gap = np.full_like(vx_vals, np.inf)

print(f"Velocidad para 1 cuanto de flujo efectivo: v_Fraunhofer = {v_fraunhofer:.2f} nm/ps")
print("Calculando oscilaciones de tamaño finito y estados de Andreev...")

for i, vx in enumerate(vx_vals):
    E_tot = 0.0
    for ky, w in zip(ky_vals, weights):
        H_mat, tr_He = build_H_BdG_peierls(ky, vx, Delta0)
        evals = eigh(H_mat, eigvals_only=True)
        
        min_gap[i] = min(min_gap[i], np.min(np.abs(evals)))
        E_tot += w * 0.5 * (np.sum(evals[evals < 0.0]) + tr_He)
    E_sc[i] = E_tot / (Lx * Ly)

# Extraer la parte oscilatoria restando un ajuste polinómico suave (fondo)
poly_bg = np.polyval(np.polyfit(flux_vals, E_sc, deg=4), flux_vals)
E_osc = E_sc - poly_bg

"""
# ==============================================================================
# 5. CALCULAR PERFIL ESPACIAL |psi(x)|^2 DE LOS ESTADOS DE ANDREEV EN LOS BORDES
# ==============================================================================
# Evaluamos en Phi_eff / Phi_0 = 1.5 (donde los estados de borde están bien desarrollados) y ky = 0
vx_edge = 1.5 * v_fraunhofer
H_edge, _ = build_H_BdG_peierls(ky=0.0, vx=vx_edge, delta_val=Delta0)
evals_edge, evecs_edge = eigh(H_edge)

# Seleccionamos los 2 estados más cercanos a E = 0
idx_zero = np.argsort(np.abs(evals_edge))[:2]
densities = []
for idx in idx_zero:
    vec = evecs_edge[:, idx].reshape(4, Nx)  # 4 componentes de Nambu-espín x Nx sitios
    prob_x = np.sum(np.abs(vec)**2, axis=0)  # Densidad total en cada punto x_j
    densities.append((evals_edge[idx], prob_x))
"""

# ==============================================================================
# 6. GRÁFICAS: OSCILACIONES TIPO FRAUNHOFER + LOCALIZACIÓN EN LOS BORDES
# ==============================================================================
fig, axs = plt.subplots(2, 1, figsize=(8.5, 10))

# Panel 1: Oscilaciones tipo Fraunhofer en E_GS
axs[0].plot(flux_vals, E_osc * 1e6, 'o-', color='#1f77b4', lw=1.8, markersize=4)
for n in range(1, 5):
    axs[0].axvline(n, color='gray', ls='--', lw=0.8, alpha=0.6)
axs[0].set_title(r'Oscilaciones tipo Fraunhofer en $E_{\mathrm{GS}}$ (parte oscilatoria)')
axs[0].set_xlabel(r'Flujo efectivo $\Phi_{\mathrm{eff}} / \Phi_0 = q_x L_x / (2\pi)$')
axs[0].set_ylabel(r'$\Delta E_{\mathrm{osc}} / A$ ($\times 10^{-6}\ \mathrm{meV/nm}^2$)')
axs[0].grid(True, alpha=0.3)

# Panel 2: Gap mínimo de cuasipartículas (modulación de estados de Andreev)
axs[1].plot(flux_vals, min_gap, 's-', color='#d62728', lw=1.8, markersize=4)
axs[1].axhline(Delta0, color='black', ls=':', lw=1.0, label=r'$\Delta_0$')
axs[1].set_title('Modulación del estado de Andreev de menor energía')
axs[1].set_xlabel(r'Flujo efectivo $\Phi_{\mathrm{eff}} / \Phi_0$')
axs[1].set_ylabel(r'$\min |E_n|$ (meV)')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
"""
# Panel 3: Perfil espacial |psi(x)|^2 mostrando localización en x=0 y x=Lx
for E_val, prob_x in densities:
    axs[2].plot(x_grid, prob_x, lw=2, label=f'Estado de borde ($E = {E_val:.4f}$ meV)')
axs[2].set_title(r'Localización espacial $|\psi(x)|^2$ en los bordes abiertos ($\Phi_{{\mathrm{{eff}}}}/\Phi_0 = 1.5, k_y = 0$)')
axs[2].set_xlabel(r'Posición transversal $x$ (nm)')
axs[2].set_ylabel(r'Densidad de probabilidad $|\psi(x)|^2$')
axs[2].legend()
axs[2].grid(True, alpha=0.3)
"""
plt.tight_layout()
plt.show()
