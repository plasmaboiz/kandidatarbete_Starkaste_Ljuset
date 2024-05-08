import sys

import matplotlib
import matplotlib.pyplot as plt
from pipic import consts
import numpy as np
from matplotlib import ticker
import scipy

plt.rcParams.update({'font.size': 20})

# sys.path.append("../../../unifiedSimulation")
sys.path.append("../../rapportPlotting")
from unifiedSimulation.lwfa_1d_params import *
import pickleLoad

obj = pickleLoad.loadObj("../unifiedSimulationData/a0_50_length_0.001_8NVNP0.pickle")
saved_electron_bins = obj.lwfa_saved_electron_bins

sampleElectrons: np.ndarray = saved_electron_bins[saved_electron_bins.shape[0] // 2, 1:] * 0.511  # MeV
min = np.min(sampleElectrons[np.nonzero(sampleElectrons)])
sampleElectrons = sampleElectrons / min

nrSavedBins = saved_electron_bins.shape[0]

xRange = np.arange(nrSavedBins) * consts.light_velocity * timestep * checkpoint
# CutOff = saved_electron_bins.shape[2]

yRange = obj.lwfa_saved_electron_bins_edges * 0.511  # MeV

Z_data: np.ndarray = saved_electron_bins[:, 1:]

Z_data = np.ma.masked_where(Z_data <= 0, Z_data)
fig, ax = plt.subplots(1, 1, sharex="all")

im = ax.imshow(Z_data.T, cmap="viridis", extent=(xRange[0], xRange[-1], yRange[0], yRange[-1]), norm="log", origin="lower",
               aspect="auto", interpolation="none")
# plt.gca().set_facecolor("black")

# fig.colorbar(im)

ax.set_ylabel("$E$ [MeV]")
ax.set_xlabel("z [cm]")
ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

fig.set_tight_layout(True)
plt.savefig("test", dpi=300)
