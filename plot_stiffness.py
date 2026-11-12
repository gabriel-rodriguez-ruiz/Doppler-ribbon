# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:24:27 2026

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load("superfluid_stiffness_alphaR_25.0_Ez_0.05.npz")

flux = data["flux_vals"]
Ds_yy_sc = data["Ds_yy_sc"]
Ds_xx_sc = data["Ds_xx_sc"]

# Access saved scalar parameters using .item()
print(f"Loaded run for alpha_R = {data['alpha_R'].item()} meV*nm, Ez = {data['Ez'].item()} meV")

fig, ax = plt.subplots()

ax.plot(flux, Ds_yy_sc)
ax.set_xlabel("Flux")
ax.set_ylabel("Dyy")
fig, ax = plt.subplots()

ax.plot(flux, Ds_xx_sc)
ax.set_xlabel("Flux")
ax.set_ylabel("Dxx")