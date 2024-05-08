# Basic setup for a laser pulse interation with a solid-density plasma layer 
# for results see Sec. 8 in arXiv:2302.01893
import pipic
from pipic import types
from pipic.consts import light_velocity, electron_mass, electron_charge
import numpy as np
from math import sin, pi
from numba import cfunc, carray
import sys

sys.path.append("..")
from rapportPlotting import RapportUtilites
from pipic.extensions import qed_volokitin2023
from elektronMomentumParams import *

output_folder = 'våg_output'  # Name of folder to store pictures.
print("nt:" + str(nt))  # Print the total number of time steps.

'''PiPic main module initialization and callback definitions'''
# Observera 1D.
sim = pipic.init(solver='fourier_boris', nx=nx, xmin=xmin, xmax=xmax, ny=ny, ymin=ymin, ymax=ymax)  # Init the PiPic main module.
sim.fourier_solver_settings(divergence_cleaning=True)  # Enable divergence cleaning.


@cfunc(nbt.double(nbt.double))
def gauss(x):
    sigmaX = pulsewidth / (np.sqrt(2 * np.log(2)))
    return np.exp(-(x / sigmaX) ** 2) * np.sin(2 * np.pi * x / wavelength)

@cfunc(nbt.double(nbt.double))
def gauss2(x):
    sigmaX = pulsewidth / (np.sqrt(2 * np.log(2)))
    return np.exp(-(x / sigmaX) ** 2) * np.cos(2 * np.pi * x / wavelength)




@cfunc(types.field_loop_callback)  # Field callback that defines a plane wave.
def field_callback(ind, r, E, B, data_double, data_int):
    # Computing both field components of the CP pulse
    f_1 = field_amplitude * gauss(r[0])
    f_2 = field_amplitude * gauss2(r[0])
    E[0] = 0.0
    E[1] = f_1
    E[2] = f_2
    B[0] = 0.0
    B[1] = -f_2
    B[2] = f_1


@cfunc(types.particle_loop_callback)  # A Particle loop callback that is frequenctly used in the 'utilities.py' addon.
def density_cb(r, p, w, id, data_double, data_int):
    ix = int(nx * (r[0] - xmin) / (xmax - xmin))
    iy = int(ny * (r[1] - ymin) / (ymax - ymin))
    data = carray(data_double, (ny, nx), dtype=np.double)
    if 0 <= iy < ny and 0 <= ix < nx:
        data[iy, ix] += w[0] / (dx * dy * 2 * density)


@cfunc(types.add_particles_callback)  # Callback function for placing the number density of electrons.
def density_callback(r, data_double, data_int):
    return density * (abs(r[0] - collision_length) < dx * 30) * (abs(r[1]) < (dy * 30))



'''Initializing fields and particles onto the PIC grid'''
sim.field_loop(handler=field_callback.address, use_omp=True)  # Setting the above defined field.
sim.fourier_solver_settings(divergence_cleaning=False)  # Disable divergence cleaning

# Adding electrons, positrons and photons to the PIC domain.
sim.add_particles(name='electron', number=n_electrons_macro,
                  charge=electron_charge, mass=electron_mass,
                  temperature=0, density=density_callback.address)

# Adding the fast QED extension
sim.set_rng_seed(2)  # Setting a random seed.


# Initializing the fast QED handler.
# qed_handler = qed_volokitin2023.handler(electron_type=sim.get_type_index('electron'),
#                                         positron_type=sim.get_type_index('positron'),
#                                         photon_type=sim.get_type_index('photon'))
# # Adding the handler.
# sim.add_handler(name=qed_volokitin2023.name, subject='electron, positron, photon',
#                 handler=qed_handler)


##Setting momentum of electrons. They will propagate along the x-axis.
@cfunc(types.particle_loop_callback)
def set_p_callback(r, p, w, id, data_double, data_int):
    p[0] = -1 * electron_mass * light_velocity*gamma #* np.random.normal(loc=gamma, scale=0.05 * gamma)  ##Std 0.1?
    p[1] = 0
    p[2] = 0


sim.particle_loop(name='electron',
                  handler=set_p_callback.address)  # Calling the callback to actually set their momenta.

'''Setting up collision_data allocations and running the actual simulation'''
# Helper class that encapsulates some collision_data recording and plotting.
FieldInfo = RapportUtilites.FieldInfo(sim=sim,
                                      densityCallBack=density_cb,
                                      timeSteps=nt,
                                      stopTime=nt * time_step,
                                      NrOfPhotonBins=n_bins,
                                      nrScatteringBins=n_scatteringBins)  #
# Place the output folder if it exists. Otherwise, create it.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(nt):
    # Fetch the weight of positrons and electrons that do not exceed a certain threshold.
    
    FieldInfo.UpdateMovement(i)
    
    if i % 10 == 0:
        print(i)
        #FieldInfo.GetFieldChanges()
        #FieldInfo.FieldPlotAndElektronDensity(output_folder, field_amplitude, i)
        #FieldInfo.UpdateDensity()
        #FieldInfo.PlotDensityDistrobution(output_folder,i)
        # FieldInfo.UpdateNrElectronsPositronsPhotons(positronElectronCutoffMomentum=0.0,
        #                                             photonCutoffMomentum=0.0)
        # print(i, '/', nt, "\n",
        #       f"Electrons: {FieldInfo.Get_Nr_Electrons()}\t\t",
        #       f"Positrons: {FieldInfo.Get_Nr_Positrons()}\t\t",
        #       f"Photons: {FieldInfo.Get_Nr_Photons()}",
        #       f"\t\tMacroParticles: {sim.get_number_of_particles()}")
    sim.advance(time_step=time_step)  # Advancing the simulation one time step.



np.save("./data/momentumData", FieldInfo.ParticleMovementData)

print(gamma)
print(amplitude_a0)
print(field_amplitude)

print("x")
print(amplitude_a0**2 / (4*gamma) * consts.electron_mass*consts.light_velocity)
print("y")
print(amplitude_a0 * consts.electron_mass*consts.light_velocity)

