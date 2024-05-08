import pipic
from pipic import consts, types
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numba import cfunc, carray, types as nbt
import os
import ctypes
import sys
from pipic.extensions import qed_volokitin2023

'''Laser & PIC domain parameters'''
######################Laser Settings############################################################
wavelength = 1e-4  # One mictron in CGS units.
omega = 2.0 * np.pi * consts.light_velocity / wavelength  # Angular frequency of EM-wave.
amplitude_a0 = 3  # Normalized amplitude of EM-wave.
a0 = consts.electron_mass * consts.light_velocity * omega / (-consts.electron_charge)
field_amplitude = a0 * amplitude_a0  # Corresponding field amplitude in CGS units.

pulsewidth = 12e-4

######################Simulation Dimensions and discretization###################################
L = 8 * pulsewidth  # Simulation box dimension.
xmin, xmax = -L, L  # Box endpoints in x and y.
ymin, ymax = -L, L
nx, ny = 256*4, 256*4  # Number of cells in x and y.
dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny  # Corresponding resolution along x and y (usually equal).
time_step = 0.1 * dx / consts.light_velocity  # The time step for one iteration.
T_sim = 2.7 * pulsewidth / consts.light_velocity  # Duration of the simulation.
nt = int(T_sim / time_step)  # Corresponding amount of time steps based on the time_step and duration.
collision_length = 2.4 * pulsewidth

######################Electron initilazation settings############################################
n_electrons_macro = 100  # Alternative way to write large numbers for readability. These are the number of macro particles for the electrons.
gamma = 1000  # Lorentz factor for the electrons.
n_crit = consts.electron_mass * omega ** 2 / (4 * np.pi * consts.electron_charge ** 2)  # Plasma critical density.
density = 1000  # Density of electrons used in this simulation.
n_bins = 100  # The number of bins to use when plotting the photon and electron energy spectra.
n_scatteringBins = 1000  # The number of bins to use when plotting the photon scattering angles.

print(T_sim)

print(density)



@cfunc(nbt.double(nbt.double))
def gauss2(x):
    sigmaX = pulsewidth / (np.sqrt(2 * np.log(2)))
    return np.exp(-(x / sigmaX) ** 2)

print(pulsewidth/consts.light_velocity)
print(gauss2(pulsewidth/2))

xRange = np.linspace(-pulsewidth, pulsewidth)

# plt.plot(xRange/consts.light_velocity, gauss2(xRange))
# plt.gca().ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
# plt.show()