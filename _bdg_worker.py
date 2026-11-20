
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp

# --- PARÁMETROS FÍSICOS IDÉNTICOS A TU MODELO ---
c = 3e17                      # nm/s
m_e = 5.1e8 / c**2            # meV s^2 / nm^2
m = 0.0403 * m_e              # meV s^2 / nm^2
hbar = 6.58e-13               # meV s
gamma = hbar**2 / (2.0 * m)   # meV nm^2  (~1661.2 meV nm^2)
E_F = 50.6                    # meV
k_F = np.sqrt(E_F / gamma)    # 1/nm      (~0.1745 nm^-1)

Delta = 0.08                  # meV
mu = E_F                      # meV
Lambda = 15.0 * 4.0 / 3.0     # meV nm    (20.0 meV nm)

theta = np.pi / 2.0
cut_off = 2.0 * k_F
phi_y = 0.0
h = 5e-5                      # Paso del stencil en 1/nm

# --- PARÁMETROS DE TEMPERATURA Y ZEEMAN ---
T = False
beta = 64.0
Zeeman = True
Z_RATIO = 1.0                 # E_Z = Z_RATIO * B

# --- GEOMETRÍA OBC EN x ---
L_x = 2500.0 / 2.0            # 1250.0 nm
B_star = 4.0 * Delta          # Campo de 1/4 de cuanto de flujo: q_D_x(B*) = pi / (2 * L_x)
q_B_constant = (np.pi / (2.0 * L_x)) / B_star

# Discretización en x: con L_x = 1250 nm y k_F ~ 0.1745 nm^-1 hay ~69 sub-bandas ocupadas.
# Nx = 280 da un paso a ~ 4.45 nm (Nx > 4 * N_sub), manteniendo buen balance velocidad/precisión.
Nx = 160 #280
a = L_x / (Nx + 1)
t_hop = gamma / (a**2)
t_R = Lambda / (2.0 * a)

# --- GRILLA ADAPTATIVA EN ky (Concentrada alrededor de las superficies de Fermi) ---
k_so = Lambda / (2.0 * gamma)
k_F_eff = np.sqrt(k_F**2 + k_so**2)
k_F_in = max(0.0, k_F_eff - k_so)
k_F_out = k_F_eff + k_so
k_limit = k_F_out + 0.1 * k_F

# Regla del trapecio / punto medio adaptativo en ky
N_ky_inner = 180 #320
N_ky_outer = 60
k_y_half = np.unique(np.concatenate([
    np.linspace(0.0, k_limit, N_ky_inner),
    np.linspace(k_limit, cut_off, N_ky_outer)
]))
k_y_values = np.unique(np.concatenate([-k_y_half[::-1], k_y_half]))

dk_y = np.diff(k_y_values)
W_KY = np.empty_like(k_y_values)
W_KY[0] = 0.5 * dk_y[0]
W_KY[-1] = 0.5 * dk_y[-1]
W_KY[1:-1] = 0.5 * (dk_y[:-1] + dk_y[1:])

s0 = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex))
sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sy = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
P_e = sp.csr_matrix(np.array([[1, 0], [0, 0]], dtype=complex))
P_h = sp.csr_matrix(np.array([[0, 0], [0, 1]], dtype=complex))
I_Nx = sp.eye(Nx, format='csr', dtype=complex)

def eval_thermodynamic_potential(eigvals, T_flag, beta_val):
    if T_flag:
        stable_term = np.minimum(0.0, eigvals)
        log_term = -(1.0 / beta_val) * np.log1p(np.exp(-beta_val * np.abs(eigvals)))
        return stable_term + log_term
    else:
        return np.minimum(0.0, eigvals)

def compute_E_single_point(px, py, bx, by, delta_val):
    '''Calcula E(px, py) en OBC en x para un punto dado del stencil.'''
    theta_x = px * a
    phase_plus = np.exp(1j * theta_x)
    phase_minus = np.exp(-1j * theta_x)

    # Saltos cinéticos en x con fase px
    H_kin_off_e = sp.diags(
        [-t_hop * phase_plus * np.ones(Nx - 1), -t_hop * phase_minus * np.ones(Nx - 1)],
        offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
    )
    H_kin_off_h = sp.diags(
        [+t_hop * phase_minus * np.ones(Nx - 1), +t_hop * phase_plus * np.ones(Nx - 1)],
        offsets=[1, -1], shape=(Nx, Nx), format='csr', dtype=complex
    )
    H_kin_off = (
        sp.kron(sp.kron(P_e, s0), H_kin_off_e, format='csr') +
        sp.kron(sp.kron(P_h, s0), H_kin_off_h, format='csr')
    )

    # Rashba en x: +Lambda * (kx + px) * sigma_y para e, +Lambda * (-kx + px) * sigma_y para h
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

    # Convención de signos idéntica a tu tensor: -by * KRON_Y - bx * KRON_X - Delta * KRON_X_0
    H_zeeman = (
        sp.kron(sp.kron(s0, sx), (-bx) * I_Nx, format='csr') +
        sp.kron(sp.kron(s0, sy), (-by) * I_Nx, format='csr')
    )
    Delta_mat = sp.diags(np.full(Nx, -delta_val), offsets=0, shape=(Nx, Nx), format='csr', dtype=complex)

    H_base = (
        H_kin_off + H_rashba_x + H_zeeman +
        sp.kron(sp.kron(sx, s0), Delta_mat, format='csr')
    ).toarray()

    idx_e_up = np.arange(0, Nx)
    idx_e_dn = np.arange(Nx, 2 * Nx)
    idx_h_dn = np.arange(2 * Nx, 3 * Nx)
    idx_h_up = np.arange(3 * Nx, 4 * Nx)

    H_BdG = np.empty_like(H_base)
    ky_integrand = np.empty(len(k_y_values), dtype=np.float64)

    for j, ky in enumerate(k_y_values):
        np.copyto(H_BdG, H_base)
        ky_p = ky + py
        ky_m = -ky + py

        diag_e = 2.0 * t_hop + gamma * (ky_p**2) - mu
        diag_h = -(2.0 * t_hop + gamma * (ky_m**2) - mu)

        H_BdG[idx_e_up, idx_e_up] += diag_e
        H_BdG[idx_e_dn, idx_e_dn] += diag_e
        H_BdG[idx_h_dn, idx_h_dn] += diag_h
        H_BdG[idx_h_up, idx_h_up] += diag_h

        # Rashba en y: -Lambda * (ky + py) * sigma_x para e, -Lambda * (-ky + py) * sigma_x para h
        ry_e = -Lambda * ky_p
        ry_h = -Lambda * ky_m
        H_BdG[idx_e_up, idx_e_dn] += ry_e
        H_BdG[idx_e_dn, idx_e_up] += ry_e
        H_BdG[idx_h_dn, idx_h_up] += ry_h
        H_BdG[idx_h_up, idx_h_dn] += ry_h

        evals = np.linalg.eigvalsh(H_BdG)
        ky_integrand[j] = np.sum(eval_thermodynamic_potential(evals, T, beta))

    # Normalización idéntica a tu código: divide por L_x para obtener densidad por área
    return 0.5 * (ky_integrand @ W_KY) / L_x

def worker_B_point(idx, B_val):
    bx = B_val * np.cos(theta) * Zeeman * Z_RATIO
    by = B_val * np.sin(theta) * Zeeman * Z_RATIO
    q_D_x = q_B_constant * B_val * np.sin(theta)
    q_D_y = q_B_constant * B_val * (-np.cos(theta)) + phi_y

    # Stencil de 9 puntos idéntico al tuyo
    offsets_x = np.array([0.0, -h, h, 0.0, 0.0, h, -h, h, -h])
    offsets_y = np.array([0.0, 0.0, 0.0, -h, h, h, -h, -h, h])

    E_stencil = np.empty(9, dtype=np.float64)
    for s in range(9):
        px = q_D_x + offsets_x[s]
        py = q_D_y + offsets_y[s]
        E_sc = compute_E_single_point(px, py, bx, by, Delta)
        E_norm = compute_E_single_point(px, py, bx, by, 0.0)
        E_stencil[s] = E_sc - E_norm

    n_xx = (E_stencil[2] - 2.0 * E_stencil[0] + E_stencil[1]) / (h**2)
    n_yy = (E_stencil[4] - 2.0 * E_stencil[0] + E_stencil[3]) / (h**2)
    n_xy = (E_stencil[5] - E_stencil[7] - E_stencil[8] + E_stencil[6]) / (4.0 * h**2)

    return idx, n_xx, n_yy, n_xy, E_stencil[0]
