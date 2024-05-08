import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 20})
import os
from elektronMomentumParams import *


outputFolder = "../_plots"

momentumData = np.load("./data/momentumData.npy")

time = momentumData[6, :]
Px = momentumData[3, :]
Py = momentumData[4, :]
Pz = momentumData[5, :]


if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

# Px som funktion av tid
fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.plot(time, Px-Px[0])
#ax.set_title(r"$\Delta P_x(t)$")
ax.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_x$ [g cm/s]')
ax.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
ax.set_xlim(0, time[-1])
fig.savefig(outputFolder + f"/imPx.png", dpi=300)

# Py som funktion av tid
fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.plot(time, Py-Py[0])
#ax.set_title(r"$\Delta P_y(t)$")
ax.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_y$ [g cm/s]')
ax.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
ax.set_xlim(0, time[-1])
fig.savefig(outputFolder + f"/imPy.png", dpi=300)

fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.plot(time, Pz - Pz[0])
#ax.set_title(r"$\Delta P_z(t)$")
ax.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_z$ [g cm/s]')
ax.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
ax.set_xlim(0, time[-1])
fig.savefig(outputFolder + f"/imPz.png", dpi=300)