# Basic setup for a laser pulse interation with a solid-density plasma layer 
# for results see fig. 6 in arXiv:2302.01893
import sys
from os import path
import pipic
import Lwfa_1d_Data
from pipic import consts, types
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numba import cfunc, carray, types as nbt
from pipic.extensions import moving_window
import h5py
import pickle
from lwfa_1d_params import *




if __name__ == '__main__':
    
    

    # Various constants are defined in lwfa_1d_params.

    # ---------------------setting solver and simulation region----------------------
    sim = pipic.init(solver='ec', nx=nx, xmin=xmin, xmax=xmax, ny=ny, ymin=ymin, ymax=ymax, nz=nz, zmin=zmin, zmax=zmax)

    output_folder = "_temporaryFiles"
    if len(sys.argv) > 1:
        a0 = float(sys.argv[1])
        # Tänk på hur spotsize är definierat
        spotsize = float(sys.argv[2]) * pulseWidth_z
        Nr_iterations = int(sys.argv[3])
        

    # Variabler som beror på sys.argv

    omega_p_Calc = np.sqrt(np.pi * 4 * consts.electron_charge ** 2 * (n0 * 2e-1) / consts.electron_mass)  # [1/s]
    wp_calc = 2 * np.pi * consts.light_velocity / omega_p_Calc

    focusPosition = wp * 2
    E0 = -a0 * consts.electron_mass * consts.light_velocity * omega / consts.electron_charge  # [statV/cm] 
    spotsize_init = spotsize * np.sqrt(1 + (focusPosition / Zr) ** 2)


    @cfunc(types.field_loop_callback)
    def initiate_field_callback(ind, r, E, B, data_double, data_int):

        if data_int[0] == 0:
            z = r[2]

            k = 2 * np.pi / wl
            # Rayleigh length
            Zr = np.pi * spotsize ** 2 / wl
            # curvature 
            R = focusPosition * (1 + (Zr / focusPosition) ** 2)
            spotsize_init = spotsize * np.sqrt(1 + (focusPosition / Zr) ** 2)
            phase = np.arctan(focusPosition / Zr)
            amp = E0 * (spotsize / spotsize_init)
            gp = np.real(amp * np.exp(-1j * (k * z + phase)) * np.exp(-z ** 2 / (2 * pulseWidth_z ** 2)))
            gp2 = np.real(amp * np.exp(-1j * (k * z + phase + np.pi / 2)) * np.exp(-z ** 2 / (2 * pulseWidth_z ** 2)))

            # circular polarized
            E[0] = gp
            B[1] = gp

            E[1] = gp2
            B[0] = -gp2


    upramp = upramp * 10


    @cfunc(types.add_particles_callback)
    def density_profile(r, data_double, data_int):
        # r is the position in the 'lab frame'  
        pos = r[2] - (zmax - thickness * dz)
        if pos > 0 and pos < upramp:
            return n0 * pos / upramp
        elif pos > upramp and pos < plasma_end:
            return n0 * plasmaRegionFaktor
        else:
            return 0


    @cfunc(types.field_loop_callback)
    def remove_field(ind, r, E, B, data_double, data_int):
        rollback = np.floor(data_int[0] * timestep * consts.light_velocity / dz)
        r_rel = zmin + dz * (rollback % nz)

        r_min = r_rel - thickness * dz
        r_max = r_rel  # + thickness*dx
        if (r[2] > r_min and r[2] < r_max) or (r[2] > zmax - (zmin - r_min)) or (r[2] < zmin + (r_max - zmax)):
            E[1] = 0
            B[2] = 0
            E[2] = 0
            B[1] = 0
            E[0] = 0
            B[0] = 0


    # ===============================SIMULATION======================================

    data_int = np.zeros((1,), dtype=np.intc)  # collision_data for passing the iteration number
    window_speed = consts.light_velocity  # speed of moving window

    # -----------------------adding the handler of extension-------------------------

    density_handler_adress = moving_window.handler(thickness=thickness,
                                                   particles_per_cell=particles_per_cell,
                                                   temperature=temperature,
                                                   density=density_profile.address, )
    sim.add_handler(name=moving_window.name,
                    subject='electron,cells',
                    handler=density_handler_adress,
                    data_int=pipic.addressof(data_int), )

    # -----------------------initiate field and plasma-------------------------
    sim.field_loop(handler=initiate_field_callback.address, data_int=pipic.addressof(data_int),
                   use_omp=True)

    # This part is just for initiating the electron species, 
    # so that the algorithm knows that there is a species called electron
    # it is therefore not important where the electrons are or what density they have
    sim.add_particles(name='electron', number=int(nz * nx * ny * particles_per_cell),
                      charge=consts.electron_charge, mass=consts.electron_mass,
                      temperature=temperature, density=density_profile.address,
                      data_int=pipic.addressof(data_int))
    # -----------------------run simulation-------------------------

    omega_p_Calc = np.sqrt(
        np.pi * 4 * consts.electron_charge ** 2 * (n0 * plasmaRegionFaktor) / consts.electron_mass)  # [1/s]
    wp_calc = 2 * np.pi * consts.light_velocity / omega_p_Calc
    print(f'Dephasing length: {2 * np.pi * omega ** 2 * consts.light_velocity / omega_p_Calc ** 3:0.2}' + ' cm')
    print(f'Pump depletion length: {np.sqrt(2) * a0 * wp_calc ** 3 / (np.pi * wl ** 2):0.2}' + ' cm')
    print(f"Propagation length: {Nr_iterations * timestep * consts.light_velocity:0.2} cm")
    print(f"Nr macro-particles: {nz * nx * ny * particles_per_cell}")
    print(f"Actuall nr sim macro particles: {sim.get_number_of_particles()}")
    lwfaParam = {"Dephasing": 2 * np.pi * omega ** 2 * consts.light_velocity / omega_p_Calc ** 3,
                 "Pump": np.sqrt(2) * a0 * wp_calc ** 3 / (np.pi * wl ** 2),
                 "Propagation": Nr_iterations * timestep * consts.light_velocity,
                 "InjectionDensity": n0, "PlasmaDensity": n0 * plasmaRegionFaktor,
                 "Resolution": nz}

    with open(path.join(output_folder, "lwfaParam.pickle"), "wb") as file:
        pickle.dump(lwfaParam, file, protocol=4)

    electronData = Lwfa_1d_Data.lwfaData(sim=sim, nrElectronBins=nrElectronBins, nrSimIterations=Nr_iterations,
                                         CheckPoint=checkpoint)

    electronData.Get_Complete_FieldChanges()  ##Retrieve the EM field of the entire vollume
    electronData.PackageCompleteFieldAndSave(output_folder, "lwfa_EM_pulse_passStart")

    savedMacroParticles = dict()

    passes_ = (Nr_iterations // nrMacroPasses)
    
    for i in range(Nr_iterations):

        if i % passes_ == 0:
            electronData.electronSaving()  ##Save electron macroParticles
            propagatedDistance = i*consts.light_velocity*timestep
            savedMacroParticles.update({propagatedDistance: electronData.electronMacroParticleData})



        data_int[0] = i

        sim.advance(time_step=timestep, number_of_iterations=1, use_omp=True)

        sim.field_loop(handler=remove_field.address, data_int=pipic.addressof(data_int),
                       use_omp=False)

        if i % checkpoint == 0:
            if i % (checkpoint * 5) == 0:
                print(i)
                print(f"Nr Electrons: {electronData.nrElectrons[1]}")
                print(f"Nr macroParticles: {sim.get_number_of_particles()}")
                print(f"Propagation Length: {np.round(timestep * consts.light_velocity * i, 5)} cm")

            electronData.UpdateNrElectrons(0)
            electronData.ElectronBinning(maximal_binnable_energy)
            electronData.SaveElectronBinnning(i)

    print(f"Actuall nr sim macro particles 2: {sim.get_number_of_particles()}")
    ##Save collision_data to pass to collision Simulation


    with open(path.join(output_folder, "lwfaMacroPasses.pickle"), "wb") as file:
        pickle.dump(savedMacroParticles, file, protocol=4)

    electronData.Get_Complete_FieldChanges()  ##Retrieve the EM field of the entire vollume
    electronData.PackageCompleteFieldAndSave(output_folder, "lwfa_EM_pulse_passFinish")
    electronData.WriteSavedElectronDataToFile(output_folder, "savedElectronBins")  # Save electronbinning
