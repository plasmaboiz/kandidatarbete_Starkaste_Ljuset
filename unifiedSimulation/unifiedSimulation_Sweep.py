import os
import string
import subprocess
import sys
import random

from pipic import consts, types
import numpy as np
import time
from os import path
import unifiedSimulationData
import dill as pickle

from lwfa_1d_params import *


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def spotsizeFromA0(a0, Energy):
    P = Energy / pulseDuration_FWHM
    spotsizeSquare = 4 * P * 1 ** 2 / (1.37e18 * a0 ** 2)
    return np.sqrt(spotsizeSquare)


def Nriteration_Calculator(plasma_Lengt):
    iterations = (upramp + plasma_Lengt) / (timestep * consts.light_velocity)
    return int(iterations / 0.90)  # /0.90 to take in to account that 10% is propagation through free space.


lwfa_script = "./lwfa_1d_simulation.py"
collision_script = "./collision_simulation.py"

temporaryFolder = "_temporaryFiles"

if not os.path.exists(temporaryFolder):
    os.makedirs(temporaryFolder)

if not os.path.exists("./lwfa_data"):
    os.makedirs("./lwfa_data")

beamEnergy = 10  # Mycket energi?# Joule

plasma_length = 5  # cm.
lwfa_a0 = [100, 125, 150]  # np.linspace(100, 10, 1)  # Normalized amplitude factor.
counter = 1
totalTime = 0
for a0 in lwfa_a0:
    spotsizeFaktor = spotsizeFromA0(a0, beamEnergy) / pulseWidth_z
    start_time = time.time()

    print(f"Running with a0={np.round(a0, 2)}, plasma length = {np.round(plasma_length, 5)} cm,"
          f" SpotsizeFaktor = {np.round(spotsizeFaktor, 2)},\niterations = {Nriteration_Calculator(plasma_length)}")
    # Tänk på hur spotsize är definierat

    # lwfa simulation
    lwfa_arguments = [str(a0), str(spotsizeFaktor), str(Nriteration_Calculator(plasma_length))]

    subprocess.run([sys.executable, lwfa_script, *lwfa_arguments])

    print(f"LWFA simulation {counter} took: {np.round(time.time() - start_time)} s")
    totalTime += np.round(time.time() - start_time)
    print(f"Total time at: {totalTime}")

    lwfaData = unifiedSimulationData.unifiedSimulationData(a0, plasma_length, spotsizeFaktor, beamEnergy,
                                                           np.load(path.join(temporaryFolder, "savedElectronBins.npy")),
                                                           np.load(path.join(temporaryFolder, "savedElectronBins_edges.npy")),
                                                           np.load(path.join(temporaryFolder, "lwfa_EM_pulse_passStart.npy")),
                                                           np.load(path.join(temporaryFolder, "lwfa_EM_pulse_passFinish.npy")),
                                                           checkpoint)

    with open(path.join(temporaryFolder, "lwfaParam.pickle"), "rb") as file:
        lwfaData.lwfaParam = pickle.load(file)

    with open(path.join(temporaryFolder, "lwfaMacroPasses.pickle"), "rb") as file:
        lwfaData.macroPasses = pickle.load(file)

    with open(f"./lwfa_data/a0_{np.round(a0, 2)}_length_{np.round(plasma_length, 6)}_{id_generator()}.pickle", "wb") as file:
        pickle.dump(lwfaData, file, protocol=4)

    # Cleanup
    os.remove(path.join(temporaryFolder, "lwfaParam.pickle"))
    os.remove(path.join(temporaryFolder, "lwfa_EM_pulse_passStart.npy"))
    os.remove(path.join(temporaryFolder, "lwfa_EM_pulse_passFinish.npy"))
    os.remove(path.join(temporaryFolder, "savedElectronBins.npy"))
    os.remove(path.join(temporaryFolder, "savedElectronBins_edges.npy"))
    counter += 1
    
    