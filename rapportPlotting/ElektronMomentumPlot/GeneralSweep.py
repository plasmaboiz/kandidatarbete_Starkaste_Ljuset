import glob
import os
import subprocess
from pathlib import Path
import sys
from os import path
import pipic
from pipic import consts, types
import numpy as np
from numba import cfunc, carray, types as nbt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
sys.path.append("..")
from rapportPlotting import RapportUtilites
from elektronMomentumParams import *


pipicScript = "elektronMomentum_simulation.py"


for a0_norm in [1,2,3]:
    arguments = [str(a0_norm)]
    subprocess.run([sys.executable, pipicScript, *arguments])

outputFolder = "plotSweepOutput"

fig_X, ax_X = plt.subplots()
fig_Y, ax_Y = plt.subplots()

for filePath in Path(outputFolder).glob("collision_data*.npy"):
    label = filePath.name.replace("collision_data", "").replace(".npy", "")
    
    #collision_data = np.vstack(self.Get_Time(), self.Get_Px_TimeSeries(), self.Get_Py_TimeSeries())
    data = np.load(str(filePath))
    time = data[0]
    Px = data[1]
    Py = data[2]

    # Px som funktion av tid
    ax_X.plot(time, Px - Px[0], label=label)
    ax_X.set_title(r"$\Delta P_x(t)$")
    ax_X.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_x$ [g cm/s]')
    ax_X.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
    ax_X.set_xlim(0, time[-1])
    
    
    # Py som funktion av tid
    ax_Y.plot(time, Py - Py[0], label=label)
    ax_Y.set_title(r"$\Delta P_y(t)$")
    ax_Y.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_y$ [g cm/s]')
    ax_Y.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
    ax_Y.set_xlim(0, time[-1])
    
    os.remove(str(filePath))

ax_X.legend()
ax_Y.legend()
fig_X.savefig(outputFolder + f"/imPx.png", dpi=300)
fig_Y.savefig(outputFolder + f"/imPy.png", dpi=300)
    

