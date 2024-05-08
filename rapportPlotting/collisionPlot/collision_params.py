from pipic import consts
import numpy as np


'''Laser & PIC domain parameters'''


pulsewidth = 9e-4
# Laser
wavelength = 1e-4 # One mictron in CGS units.
omega = 2.0 * np.pi * consts.light_velocity / wavelength  # Angular frequency of EM-wave.
amplitude_a0 = 5  # Normalized amplitude of EM-wave.
a0 = consts.electron_mass * consts.light_velocity * omega / (-consts.electron_charge)
field_amplitude = a0 * amplitude_a0  # Corresponding field amplitude in CGS units.
emWaveOffset = 3*pulsewidth


######################Simulation Dimensions and discretization###################################
L = 4 * pulsewidth  # Simulation box dimension.
# xmin, xmax = -L, L  # Box endpoints in x and y.
# ymin, ymax = -L, L
# nx, ny = 256, 1  # Number of cells in x and y.
# dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny  # Corresponding resolution along x and y (usually equal).

pulseDuration_FWHM = 16e-15
LWFA_pulseWidth_z = (pulseDuration_FWHM / 2.355) * consts.light_velocity  # [cm]
LWFA_nz, LWFA_zmin, LWFA_zmax = 2 ** 8, -15 * LWFA_pulseWidth_z, 5 * LWFA_pulseWidth_z
LWFA_dz = (LWFA_zmax - LWFA_zmin) / LWFA_nz

dz = LWFA_dz

zmin, zmax = -L, L

nz = int(np.round((zmax - zmin) / dz)) ##Detta ger att LWFA och collsion har samma cell storlek.

nx, xmin, xmax = 32, -2, 2
ny, ymin, ymax = 32, -2, 2

dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny

print(nz)
print((zmax - zmin) / dz)
print(LWFA_pulseWidth_z/dz)


time_step = 0.25 * dz / consts.light_velocity  # The time step for one iteration.
T_sim = 2 * pulsewidth / consts.light_velocity  # Duration of the simulation.
nt = int(T_sim / time_step)  # Corresponding amount of time steps based on the time_step and duration.



######################Electron initilazation settings############################################
nr_electrons_macro = 500_000  # Alternative way to write large numbers for readability. These are the number of macro particles for the electrons.
gamma = 4000
density = 1e9 # Need an density to initalize electrons, it will later be adjusted when partilces from lwfa is loaded.
maxEnergy = gamma*1.1
collision_length = 2.5*pulsewidth

#Data aqusition settings.
n_bins = 700#250  # The number of bins to use when plotting the photon and electron energy spectra.
n_scatteringBins = 30000  # The number of bins to use when plotting the photon scattering angles.

checkpoint = 10
