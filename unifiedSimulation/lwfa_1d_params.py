import numpy as np
from pipic import consts
import scipy



# ===========================SIMULATION INITIALIZATION===========================
# Electron number density
n0 = 8e19 #8e18  # 1e18 # [1/cm^3]
plasmaRegionFaktor = 4e-3
# plasma frequency, [e] = statC, [me] = g, [n0] = 1/cm^3, [c] = cm/s
omega_p = np.sqrt(np.pi * 4 * consts.electron_charge ** 2 * n0 / consts.electron_mass)  # [1/s]
# plasma wavelength
wp = 2 * np.pi * consts.light_velocity / omega_p  # [cm]
# laser wavelength
wl = 1e-4  # [cm]
# laser pulse width (spatial in z and radially in focus)
pulseDuration_FWHM = 16e-15

pulseWidth_z = (pulseDuration_FWHM / 2.355) * consts.light_velocity  # [cm]
spotsize = 2 * pulseWidth_z
nz, zmin, zmax = 2 ** 8, -15 * pulseWidth_z, 5 * pulseWidth_z#2 ** 8, -15 * pulseWidth_z, 5 * pulseWidth_z
nx, xmin, xmax = 4, -2, 2
ny, ymin, ymax = 4, -2, 2
dz = (zmax - zmin) / nz
dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny
print(dx/(2*consts.light_velocity))


# 10 timesteps per laser cycle
timestep = 1e-1 / (2 * np.pi * consts.light_velocity / wl)
print(timestep)
thickness = 10  # thickness (in dz) of the area where the density and field is restored/removed 

# ---------------------------setting field of the pulse--------------------------
# laser (radial) frequency
omega = 2 * np.pi * consts.light_velocity / wl  # [1/s]
a0 = 100  # [unitless]
# Field amplitude

focusPosition = wp * 2

k = 2 * np.pi / wl
Zr = np.pi * omega ** 1 / wl
R = focusPosition * (1 + (Zr / focusPosition) ** 2)

# laser wave number
k = 2 * np.pi / wl  # [1/cm]
# Critical density
N_cr = consts.electron_mass * omega ** 2 / (4 * np.pi * consts.electron_charge ** 2)

debye_length = 1e-2  # [cm] 
temperature = 0  # 4 * np.pi * n0 * consts.electron_charge ** 2 * debye_length ** 2 # [erg/kB] (?)

particles_per_cell = 1 * 13

Nr_iterations = 500
checkpoint = 100


# Plasma profile
upramp = 0.001  # cm
plasma_end = 0.90 * Nr_iterations * consts.light_velocity * timestep  # cm



maximal_binnable_energy = 40000 # i 1/mc**2

nrElectronBins = int(maximal_binnable_energy/4)

nrMacroPasses = 10

a0 = 100
omega_p_Calc = np.sqrt(
        np.pi * 4 * consts.electron_charge ** 2 * (n0 * plasmaRegionFaktor) / consts.electron_mass)  # [1/s]
wp_calc = 2 * np.pi * consts.light_velocity / omega_p_Calc
print(wp_calc**3 / (2*wl**2) * np.sqrt(a0) / np.pi)
print(wp_calc**3 / (wl**2)* np.sqrt(a0) / np.pi)
