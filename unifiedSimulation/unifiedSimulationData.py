import numpy as np
import scipy
from lwfa_1d_params import pulseWidth_z, pulseDuration_FWHM, omega
from os import path
from pipic import consts



class unifiedSimulationData:
    
    def __init__(self, a0, plasma_distance, _spotsize, energy, lwfa_saved_electron_bins, edges,  lwfa_PackagedFieldStart=None, lwfa_PackagedFieldEnd=None,  checkpoint=None):
        self.lwfa_a0 = a0
        self.lwfa_spotsize = _spotsize
        self.lwfa_plasma_length = plasma_distance
        self.lwfa_energy = energy
        self.lwfaParam = None
        self.macroPasses = None
        
        self.lwfa_PackagedFieldStart = lwfa_PackagedFieldStart
        self.lwfa_PackagedFieldEnd = lwfa_PackagedFieldEnd
        self.lwfa_saved_electron_bins = lwfa_saved_electron_bins
        self.lwfa_saved_electron_bins_edges = edges
        self.checkpoint = checkpoint



def getAverageGamma(macroParticles: np.ndarray):
    Px = macroParticles[:, 3]
    Py = macroParticles[:, 4]
    Pz = macroParticles[:, 5]
    p_squared = Px * Px + Py * Py + Pz * Pz
    meanP = p_squared.mean()
    m2c2 = (consts.electron_mass * consts.light_velocity) ** 2
    electron_energy = np.sqrt(1.0 + meanP / m2c2) ##Electron energi i termer av gamma
    return electron_energy

class collisionData:
    def __init__(self, tempFolder, lwfaFile, lwfaParam, _a0, _spotsize, energy, distance, macroParticles):
        self.lwfaPickleFile = lwfaFile
        self.lwfa_Params = lwfaParam
        self.a0 = _a0
        self.spotsize = _spotsize
        self.lwfa_Energy = energy
        self.propagated_Distance = distance
        self.averageGamma = getAverageGamma(macroParticles)

        self.photonBins = np.load(path.join(tempFolder, "photonBins.npy"))
        self.photonCumulativeBins = np.load(path.join(tempFolder, "photonCumulativeBins.npy"))
        self.photonScatteringBins = np.load(path.join(tempFolder, "photonScatteringBins.npy"))
        self.electronBins= np.load(path.join(tempFolder, "electronBins.npy"))
        self.nrElectronDict = np.load(path.join(tempFolder, "nrElectrons.npy"))
        self.nrPositrons = np.load(path.join(tempFolder, "nrPositrons.npy"))
        self.nrPhotons = np.load(path.join(tempFolder, "nrPhotons.npy"))
        
        