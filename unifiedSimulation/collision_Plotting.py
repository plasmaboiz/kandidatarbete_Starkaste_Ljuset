import pickle
import time
import sys
import subprocess
from pathlib import Path
import matplotlib.ticker
from matplotlib.ticker import LogFormatterMathtext, LogFormatterExponent, LogFormatterSciNotation
import re
import os
import string
import random

import matplotlib.pyplot

import unifiedSimulationData
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 34})


def loadObj(filePath) -> unifiedSimulationData.collisionData:
    with open(filePath, "rb") as file:
        return pickle.load(file)


def dictionaryGetter(xArr, yArr, dict):
    res = np.zeros(shape=(len(yArr), len(xArr)))
    for indy, y in enumerate(yArr):
        for indx, x in enumerate(xArr):
            res[len(yArr) - 1 - indy, indx] = dict[(x, y)]
    return res


def getAverageEnergy(photonBins, cumuPhotonBins):
    totalEnergy = np.sum(cumuPhotonBins[1:]) * 0.511  ## Enhet MeV
    totalNrPhotons = np.sum([photonBins[1:]])
    averageEnergy = totalEnergy / totalNrPhotons
    return averageEnergy

def getTotalEnergy(photonBins, cumuPhotonBins):
    totalEnergy = np.sum(cumuPhotonBins[1:]) * 0.511  ## Enhet MeV
    return totalEnergy



folder = "collision_data"

# f"a0_{np.round(a0,2)}_spot_{np.round(spotsize/pulseWidth_z,2)}
electronAverageGammaDict = {}
photonTotalEnergy = {}
photonAverageEnergy = {}
positronCount = {}
a0_set = set()
distance_set = set()

for filePath in Path(folder).glob("*"):
    obj = loadObj(str(filePath))
    electronAverageGammaDict[(obj.propagated_Distance, obj.a0)] = obj.averageGamma
    photonTotalEnergy[(obj.propagated_Distance, obj.a0)] = getTotalEnergy(obj.photonBins, obj.photonCumulativeBins)
    photonAverageEnergy[(obj.propagated_Distance, obj.a0)] = getAverageEnergy(obj.photonBins, obj.photonCumulativeBins)
    positronCount[(obj.propagated_Distance, obj.a0)] = obj.nrPositrons[-1]

    distance_set.add(obj.propagated_Distance)
    a0_set.add(obj.a0)

distance_set = sorted(distance_set)
a0_set = sorted(a0_set)



plt.figure(figsize=(12.8, 9.6))
valueArray = dictionaryGetter(distance_set, a0_set, electronAverageGammaDict)
plt.imshow(valueArray, extent=(min(distance_set), max(distance_set), min(a0_set), max(a0_set)), aspect="auto", interpolation="none")
plt.yticks([50+10, 75+5, 100, 125-5, 150-10], [f"{a0}" for a0 in a0_set])
plt.xticks(np.arange(1,6)*1+0.25, np.arange(1,6)*1)
plt.xlabel("$L_p$ [cm]")
plt.ylabel("$a_0$")
plt.colorbar(label="$\gamma_{e^-}$")

plt.tight_layout()
plt.savefig("averageGamma", dpi=300)

plt.close()

plt.figure(figsize=(12.8, 9.6))
valueArray = dictionaryGetter(distance_set, a0_set, photonTotalEnergy)
plt.imshow(valueArray, extent=(min(distance_set), max(distance_set), min(a0_set), max(a0_set)), aspect="auto", interpolation="none")
plt.yticks([50+10, 75+5, 100, 125-5, 150-10], [f"{a0}" for a0 in a0_set])
plt.xticks(np.arange(1,6)*1+0.25, np.arange(1,6)*1)
plt.xlabel("$L_p$ [cm]")
plt.ylabel("$a_0$")
plt.colorbar(label="$E_f^{\\text{Tot}}$ [MeV]")

plt.tight_layout()
plt.savefig("photonTotalEnergy", dpi=300)


valueArray = dictionaryGetter(distance_set, a0_set, photonAverageEnergy)

plt.close()
plt.figure(figsize=(12.8, 9.6))
plt.imshow(valueArray, extent=(min(distance_set), max(distance_set), min(a0_set), max(a0_set)), aspect="auto", interpolation="none")
plt.yticks([50+10, 75+5, 100, 125-5, 150-10], [f"{a0}" for a0 in a0_set])
plt.xticks(np.arange(1,6)*1+0.25, np.arange(1,6)*1)
plt.xlabel("$L_p$ [cm]")
plt.ylabel("$a_0$")
plt.colorbar(label="$E_f^{\\text{Avg}}$ [MeV]")

plt.tight_layout()
plt.savefig("photonAverageEnergy", dpi=300)


valueArray = dictionaryGetter(distance_set, a0_set, positronCount)
plt.close()
plt.figure(figsize=(12.8, 9.6))
plt.imshow(valueArray, extent=(min(distance_set), max(distance_set), min(a0_set), max(a0_set)), aspect="auto", interpolation="none")
plt.yticks([50+10, 75+5, 100, 125-5, 150-10], [f"{a0}" for a0 in a0_set])
plt.xticks(np.arange(1,6)*1+0.25, np.arange(1,6)*1)
plt.xlabel("$L_p$ [cm]")
plt.ylabel("$a_0$")

plt.colorbar(label="$n_{e^+}$")

plt.tight_layout()
plt.savefig("positronCount", dpi=300)

plt.show()
