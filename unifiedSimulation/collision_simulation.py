# Basic setup for a laser pulse interation with a solid-density plasma layer 
# for results see Sec. 8 in arXiv:2302.01893
from pipic.consts import electron_mass, electron_charge
import pipic
import CollisionData
import plotting_utilities
from pipic import types
from collision_params import *
import sys
import numba
from numba import cfunc, carray
import os
from os import path
import numpy as np
from pipic.extensions import qed_volokitin2023


def getTopEnergy(macroParticles: np.ndarray):
    Px = macroParticles[:, 3]
    Py = macroParticles[:, 4]
    Pz = macroParticles[:, 5]
    p_squared = Px * Px + Py * Py + Pz * Pz
    maxP = p_squared.max()
    m2c2 = (consts.electron_mass * consts.light_velocity) ** 2
    electron_energy = np.sqrt(1.0 + maxP / m2c2)  ##Electron energi i termer av gamma
    return electron_energy


print("nt:" + str(nt))  # Print the total number of time steps.

temporaryFolder = "_temporaryFiles"  # Folder where collision_data that is passed between simulations are kept

# Retrieves arguments

if len(sys.argv) > 1:
    nr_electrons_macro = int(sys.argv[1])

# Test med detta.
maxEnergy = getTopEnergy(np.load(path.join(temporaryFolder, "lwfa_MacroPass.npy"))) * 1.25  # För att veta hur stor energy man ska binna på.
n_bins = int(maxEnergy / 2)  # 2 gamma per bin.
if n_bins == 0:  ##Fallet då det är låg energi, löser lite error som uppstår i det fallet.
    maxEnergy = 10
    n_bins = int(maxEnergy / 2)

'''PiPic main module initialization and callback definitions'''
# The simulation is *almost* 1D, nx = ny = 4.
sim = pipic.init(solver='fourier_boris', nx=nx, xmin=xmin, xmax=xmax, ny=ny, ymin=ymin, ymax=ymax, nz=nz, zmin=zmin, zmax=zmax)  # Init the PiPic main module.
sim.fourier_solver_settings(divergence_cleaning=True)  # Enable divergence cleaning.


@cfunc(types.particle_loop_callback)  # A Particle loop callback that is frequenctly used in the 'collision_utilities.py' addon.
def density_cb(r, p, w, id, data_double, data_int):
    ix = int(nx * (r[0] - xmin) / (xmax - xmin))
    iy = int(ny * (r[1] - ymin) / (ymax - ymin))
    iz = int(nz * (r[2] - zmin) / (zmax - zmin))
    data = carray(data_double, (nx, ny, nz), dtype=np.double)
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        data[ix, iy, iz] += w[0]


simulationData = CollisionData.CollisionData(sim=sim,  # Helper object to encapsulate some useful code.
                                             densityCallBack=density_cb,
                                             nrIterations=nt,
                                             checkPoint=checkpoint,
                                             stopTime=nt * time_step,
                                             NrOfPhotonBins=n_bins,
                                             nrScatteringBins=n_scatteringBins
                                             )


@cfunc(types.add_particles_callback)  # Callback function for placing the number density of electrons.
def density_callback(r, data_double, data_int):
    return density * (abs(r[2] - collision_length) < (dz * 30))  # * (abs(r[1]) < (dy * 60)) ##3.5 från dz faktor ökningen mellan simuleringarna.


# Adding electrons, positrons and photons to the PIC domain.
sim.add_particles(name='electron', number=int(nr_electrons_macro * 1.2),  # Some margin on the macro particles, beacuse unexplained reasons
                  charge=electron_charge, mass=electron_mass,  # The leftover particles will be removed in the following function.
                  temperature=0, density=density_callback.address)

# Adds saved particles to the simulation
simulationData.Add_lwfa_Elektrons(np.load(path.join(temporaryFolder, "lwfa_MacroPass.npy")), nr_electrons_macro)

# Adds saved EM field to the simulation
simulationData.Add_lwfa_Field(np.load(path.join(temporaryFolder, "lwfa_EM_pulse_pass.npy")))
maxField = np.load(path.join(temporaryFolder, "lwfa_EM_pulse_pass.npy")).max()


# Callback that can initalize particles with 0 macroparticles
@cfunc(types.add_particles_callback)
def null_callback(r, data_double, data_int):
    return 0


# Initializing particles.
# sim.add_particles(name='electron', number=0, charge=0, mass=0,
#                   temperature=0, density=null_callback.address)

sim.add_particles(name='photon', number=0, charge=0, mass=0,
                  temperature=0, density=null_callback.address)

sim.add_particles(name='positron', number=0,
                  charge=-electron_charge, mass=electron_mass,
                  temperature=0, density=null_callback.address)

# Adding the fast QED extension

sim.set_rng_seed(2)  # Setting a random seed. Choosen by fair dice roll.

# Initializing the fast QED handler.
qed_handler = qed_volokitin2023.handler(electron_type=sim.get_type_index('electron'),
                                        positron_type=sim.get_type_index('positron'),
                                        photon_type=sim.get_type_index('photon'))
# Adding the handler.
sim.add_handler(name=qed_volokitin2023.name, subject='electron, positron, photon',
                handler=qed_handler)

for i in range(nt):

    # simulationData.UpdateMovement(i)
    if i % checkpoint == 0:
        simulationData.UpdateDensity()
        simulationData.Get_Complete_FieldChanges()
        plotting_utilities.FieldPlotAndElektronDensity(simulationData, "out", 2*maxField, i)

        # Fetch the weight of positrons and electrons that do exceed a certain threshold. Momentum unit in cgs.
        simulationData.UpdateNrElectronsPositronsPhotons(i, positronElectronCutoffMomentum=0.0,
                                                         photonCutoffMomentum=0.0)
        print(i, '/', nt, "\n",
              f"Electrons: {simulationData.Get_Latest_Nr_Electrons()}\t\t",
              f"Positrons: {simulationData.Get_Latest_Nr_Positrons()}\t\t",
              f"Photons: {simulationData.Get_Latest_Nr_Photons()}",
              f"\t\tMacroParticles: {sim.get_number_of_particles()}")
    sim.advance(time_step=time_step)  # Advancing the simulation one time step.

# The helper class sets the binning of electron and photon energies and then plots them. Figure is stored in the output folder.
# In unit "gamma". gamma/0.511 gives energy in KeV (0.511 because cgs....)

simulationData.PhotonBinning(maxEnergy)
simulationData.ElectronBinning(maxEnergy)
simulationData.PhotonCumulativeBinning(maxEnergy)
simulationData.PhotonScatteringBinning()

np.save(path.join(temporaryFolder, "photonBins"), simulationData.photonBins)
np.save(path.join(temporaryFolder, "photonCumulativeBins"), simulationData.photonCumulativeBins)
np.save(path.join(temporaryFolder, "photonScatteringBins"), simulationData.photonScatteringBins)
np.save(path.join(temporaryFolder, "electronBins"), simulationData.ElectronsBins)
np.save(path.join(temporaryFolder, "nrElectrons"), simulationData.nrElectrons)
np.save(path.join(temporaryFolder, "nrPositrons"), simulationData.nrPositrons)
np.save(path.join(temporaryFolder, "nrPhotons"), simulationData.nrPhotons)
