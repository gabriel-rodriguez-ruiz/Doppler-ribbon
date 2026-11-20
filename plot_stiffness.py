# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:24:27 2026

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_folder = Path(r"./Data")

file_to_open = data_folder / "conv_B=2Delta_theta=1.571_Lx=750_Nx=100_N=1000_beta=125.0_Lambda=15.0_Delta=0.4.npz"

data = np.load(file_to_open)

# B_values = data["B_values"]
n_yy = data["n_yy"]


fig, ax = plt.subplots()

ax.plot(B_values/2, n_yy)
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$n_{yy}$")
# ax.set_xlim([0,4*0.4])
# ax.set_ylim([110, 120])

#%%

file_to_open = data_folder / "C:/Users/Gabriel/Python/Doppler-ribbon/Data/conv_B=2Delta_theta=1.571_Lx=750_Nx=200_N=1000_beta=125.0_Lambda=15.0_Delta=0.4.npz"

data = np.load(file_to_open)

# B_values = data["B_values"]
n_yy = data["n_yy"]

# fig, ax = plt.subplots()

ax.plot(B_values/0.4, n_yy)

#%%

file_to_open = data_folder / "obc_stiffness_theta=1.571_L_x=750_N_x_sites=100_N=1000_q_B_constant=0_beta=125_Lambda=20.0.npz"

data = np.load(file_to_open)

B_values = data["B_values"]
n_yy = data["n_yy"]

# fig, ax = plt.subplots()

ax.plot(B_values/0.4, n_yy)

#%%

file_to_open = data_folder / "obc_stiffness_theta=1.571_L_x=750_N_x_sites=100_N=1000_q_B_constant=0_beta=125_Lambda=0.npz"

data = np.load(file_to_open)

B_values = data["B_values"]
n_yy = data["n_yy"]

# fig, ax = plt.subplots()

ax.plot(B_values/0.4, n_yy)

