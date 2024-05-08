from pipic import consts
import numpy as np


'''Laser & PIC domain parameters'''


pulsewidth = 9e-4

######################Simulation Dimensions and discretization###################################
L = 8 * pulsewidth  # Simulation box dimension.
# xmin, xmax = -L, L  # Box endpoints in x and y.
# ymin, ymax = -L, L
# nx, ny = 256, 1  # Number of cells in x and y.
# dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny  # Corresponding resolution along x and y (usually equal).

pulseDuration_FWHM = 16e-15
LWFA_pulseWidth_z = (pulseDuration_FWHM / 2.355) * consts.light_velocity  # [cm]
LWFA_nz, LWFA_zmin, LWFA_zmax = 2 ** 8, -15 * LWFA_pulseWidth_z, 5 * LWFA_pulseWidth_z
LWFA_dz = (LWFA_zmax - LWFA_zmin) / LWFA_nz

dz = LWFA_dz
zmin, zmax = -12*LWFA_pulseWidth_z, 12*LWFA_pulseWidth_z

nz = int(np.round((zmax - zmin) / dz)) ##Detta ger att LWFA och collsion har samma cell storlek.

nx, xmin, xmax = 4, -2, 2
ny, ymin, ymax = 4, -2, 2

dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny

print(nz)
print((zmax - zmin) / dz)
print(LWFA_pulseWidth_z/dz)


time_step = 0.25 * dz / consts.light_velocity  # The time step for one iteration.
T_sim = 10 * LWFA_pulseWidth_z / consts.light_velocity  # Duration of the simulation.
nt = int(T_sim / time_step)  # Corresponding amount of time steps based on the time_step and duration.
collision_length = 8*LWFA_pulseWidth_z

######################Electron initilazation settings############################################
nr_electrons_macro = 50_000  # Alternative way to write large numbers for readability. These are the number of macro particles for the electrons.

density = 10e5 # Need an density to initalize electrons, it will later be adjusted when partilces from lwfa is loaded.


#Data aqusition settings.
n_bins = 100  # The number of bins to use when plotting the photon and electron energy spectra.
n_scatteringBins = 40000  # The number of bins to use when plotting the photon scattering angles.

checkpoint = 10
