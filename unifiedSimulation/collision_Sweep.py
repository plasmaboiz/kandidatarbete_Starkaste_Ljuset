import pickle
import time
import sys
import subprocess
from os import path
import os
import string
import random

import unifiedSimulationData
import numpy as np
from pipic import consts

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def loadObj(filePath) -> unifiedSimulationData.unifiedSimulationData:
    with open(filePath, "rb") as file:
        return pickle.load(file)


temporaryFolder = "_temporaryFiles"
collision_script = "./collision_simulation.py"

if not os.path.exists(temporaryFolder):
    os.makedirs(temporaryFolder)

if not os.path.exists("collision_data"):
    os.makedirs("collision_data")


fileName = "./lwfa_data/a0_150_length_5_PZW5QE.pickle"
#fileName = "./collision_data/a0_50_length_0.001_KVMVDE.pickle"

obj = loadObj(fileName)
np.save(path.join(temporaryFolder, "lwfa_EM_pulse_pass.npy"), obj.lwfa_PackagedFieldEnd)

macroPassses: dict = obj.macroPasses


total_time = 0
counter = 1
for prop_Dist, macroParticles in macroPassses.items():
    np.save(path.join(temporaryFolder, "lwfa_MacroPass"), macroParticles)
    if prop_Dist < 0.001:
        continue
    

    start_time = time.time()
    print(f"Running collision simulation. Nr: {counter}, distance: {prop_Dist}")
    
    nrMacroParticles = 50_000
    collision_arguments = [str(nrMacroParticles)]
    subprocess.run([sys.executable, collision_script, *collision_arguments])
    print(f"Collision simulation took: {np.round(time.time() - start_time)} s")
    total_time += np.round(time.time() - start_time)
    print(f"Total runtime at: {total_time} s")
    
    currentCollisionData = unifiedSimulationData.collisionData(temporaryFolder,
                                                               fileName,
                                                               obj.lwfaParam,
                                                               obj.lwfa_a0,
                                                               obj.lwfa_spotsize,
                                                               obj.lwfa_energy,
                                                               prop_Dist,
                                                               macroParticles)

    with open(f"collision_data/a0_{np.round(obj.lwfa_a0, 2)}_length_{np.round(prop_Dist, 6)}_{id_generator()}.pickle", "wb") as file:
        pickle.dump(currentCollisionData, file, protocol=4)

    with open(f"/mnt/e/\"Min enhet\"/pickles2/a0_{np.round(obj.lwfa_a0, 2)}_length_{np.round(prop_Dist, 6)}_{id_generator()}.pickle", "wb") as file:
        pickle.dump(currentCollisionData, file, protocol=4)

    os.remove(path.join(temporaryFolder, "photonBins.npy"))
    os.remove(path.join(temporaryFolder, "photonCumulativeBins.npy"))
    os.remove(path.join(temporaryFolder, "photonScatteringBins.npy"))
    os.remove(path.join(temporaryFolder, "electronBins.npy"))
    os.remove(path.join(temporaryFolder, "nrElectrons.npy"))
    os.remove(path.join(temporaryFolder, "nrPositrons.npy"))
    os.remove(path.join(temporaryFolder, "nrPhotons.npy"))
    os.remove(path.join(temporaryFolder, "lwfa_MacroPass.npy"))
    counter += 1
    
os.remove(path.join(temporaryFolder, "lwfa_EM_pulse_pass.npy"))
    