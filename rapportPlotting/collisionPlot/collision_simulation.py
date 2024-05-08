# Basic setup for a laser pulse interation with a solid-density plasma layer 
# for results see Sec. 8 in arXiv:2302.01893
from pipic.consts import electron_mass, electron_charge
import pipic
import sys
sys.path.append("../../unifiedSimulation")
from unifiedSimulation import CollisionData
from collision_params import *
import sys
import numba
from numba import cfunc, carray
from pipic import types
import os
import numba.types as nbt
from os import path
import numpy as np
from pipic.extensions import qed_volokitin2023
from unifiedSimulation import plotting_utilities

print("nt:" + str(nt))  # Print the total number of time steps.

temporaryFolder = "collision_data" # Folder where collision_data that is passed between simulations are kept



'''PiPic main module initialization and callback definitions'''
# The simulation is *almost* 1D, nx = ny = 4.
sim = pipic.init(solver='fourier_boris', nx=nx, xmin=xmin, xmax=xmax, ny=ny, ymin=ymin, ymax=ymax, nz=nz, zmin=zmin, zmax=zmax)  # Init the PiPic main module.
sim.fourier_solver_settings(divergence_cleaning=True)  # Enable divergence cleaning.



@cfunc(types.particle_loop_callback)  # A Particle loop callback that is frequenctly used in the 'collision_utilities.py' addon.
def density_cb(r, p, w, id, data_double, data_int):
    ix = int(nx * (r[0] - xmin) / (xmax - xmin))
    iy = int(ny * (r[1] - ymin) / (ymax - ymin))
    data = carray(data_double, (ny, nx), dtype=np.double)
    if 0 <= iy < ny and 0 <= ix < nx:
        data[iy, ix] += w[0] / (dx * dy * 2 * density)

@cfunc(nbt.double(nbt.double))
def gauss(x):
    sigmaX = pulsewidth / (np.sqrt(2 * np.log(2)))
    return np.exp(-(x / sigmaX) ** 2) * np.sin(2 * np.pi * x / wavelength)

@cfunc(nbt.double(nbt.double))
def gauss2(x):
    sigmaX = pulsewidth / (np.sqrt(2 * np.log(2)))
    return np.exp(-(x / sigmaX) ** 2) * np.cos(2 * np.pi * x / wavelength)



@cfunc(types.field_loop_callback) #Field callback that defines a plane wave.
def field_callback(ind, r, E, B, data_double, data_int):
    #Computing both field components of the CP pulse
    f_1 = field_amplitude * gauss(r[2])
    f_2 = field_amplitude * gauss2(r[2])
    E[0] = f_1
    E[1] = f_2
    E[2] = 0.0
    B[0] = -f_2
    B[1] = f_1
    B[2] = 0

sim.field_loop(handler=field_callback.address, use_omp=True) #Setting the above defined field.

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



# Callback that can initalize particles with 0 macroparticles
@cfunc(types.add_particles_callback)
def null_callback(r, data_double, data_int):
    return 0


#Initializing particles.
sim.add_particles(name='electron', number=nr_electrons_macro,
                  charge=electron_charge, mass=electron_mass,
                  temperature=0, density=density_callback.address)

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

@cfunc(types.particle_loop_callback)
def set_p_callback(r, p, w, id, data_double, data_int):
    p[0] = 0
    p[1] = 0
    p[2] = -gamma * electron_mass * consts.light_velocity #* np.random.normal(loc=gamma, scale=0.05 * gamma)

sim.particle_loop(name='electron', handler=set_p_callback.address) #Calling the callback to actually set their momenta.


simulationData.PhotonBinning(maxEnergy)
simulationData.ElectronBinning(maxEnergy*2)
simulationData.PhotonCumulativeBinning(maxEnergy)
simulationData.PhotonScatteringBinning()
np.save(path.join(temporaryFolder, "preCollision_photonBins"), simulationData.photonBins)
np.save(path.join(temporaryFolder, "preCollision_photonCumulativeBins"), simulationData.photonCumulativeBins)
np.save(path.join(temporaryFolder, "preCollision_photonScatteringBins"), simulationData.photonScatteringBins)
np.save(path.join(temporaryFolder, "preCollision_electronBins"), simulationData.ElectronsBins)




for i in range(nt):
    
    
    simulationData.UpdateMovement(i)
    if i % checkpoint == 0:
        simulationData.UpdateNrElectronsPositronsPhotons(i, positronElectronCutoffMomentum=0.0,
                                                         photonCutoffMomentum=0.0)
        simulationData.Get_Complete_FieldChanges()
        
        # Fetch the weight of positrons and electrons that do exceed a certain threshold. Momentum unit in cgs.
        plotting_utilities.FieldPlotAndElektronDensity(simulationData, "emField", 2*field_amplitude, i)
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

np.save(path.join(temporaryFolder, "postCollision_photonBins"), simulationData.photonBins)
np.save(path.join(temporaryFolder, "postCollision_photonCumulativeBins"), simulationData.photonCumulativeBins)
np.save(path.join(temporaryFolder, "postCollision_photonScatteringBins"), simulationData.photonScatteringBins)
np.save(path.join(temporaryFolder, "postCollision_electronBins"), simulationData.ElectronsBins)
np.save(path.join(temporaryFolder, "nrElectrons"), simulationData.nrElectrons)
np.save(path.join(temporaryFolder, "nrPositrons"), simulationData.nrPositrons)
np.save(path.join(temporaryFolder, "nrPhotons"), simulationData.nrPhotons)
