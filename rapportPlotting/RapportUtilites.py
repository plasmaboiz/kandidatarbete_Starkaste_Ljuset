######################OBS##################################
###Kanske har andra enheter jämfört utilites därför ny fil.
###########################################################

import glob
import os
import subprocess
from os import path
import numba
import numpy
import pipic
from pipic import consts, types
import numpy as np
from numba import cfunc, carray, types as nbt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 20})

@cfunc(types.field_loop_callback)
def FieldRetrieverCallback(ind, r, E, B, data_double, data_int):
    # Fields innehållar alla fält m.h.a en 3D array. Indexerad enligt 0-6, B-xyz, E-xyz
    Fields = carray(data_double, (data_int[0], data_int[1], 6), dtype=np.double)  ##Shape från data_int, 
    Fields[ind[1], ind[0], 0] = B[0]
    Fields[ind[1], ind[0], 1] = B[1]
    Fields[ind[1], ind[0], 2] = B[2]
    Fields[ind[1], ind[0], 3] = E[0]
    Fields[ind[1], ind[0], 4] = E[1]
    Fields[ind[1], ind[0], 5] = E[2]


@cfunc(types.particle_loop_callback)
def PositionMomentumRetriverCallback(r, p, w, id, data_double, data_int):
    movData = carray(data_double, (6, data_int[1]), dtype=np.double)
    if id[0] == 0:
        movData[0, data_int[0]] = r[0]  # x
        movData[1, data_int[0]] = r[1]  # y
        movData[2, data_int[0]] = r[2]  # z
        movData[3, data_int[0]] = p[0]  # Px
        movData[4, data_int[0]] = p[1]  # Py
        movData[5, data_int[0]] = p[2]  # Pz


@cfunc(types.particle_loop_callback)
def N_e_ep_cb(r, p, w, id, data_double, data_int):
    if np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) > data_double[0]:
        data_double[1] += w[0]
    else:
        w[0] = 0


@cfunc(types.particle_loop_callback)
def photonCounter_cb(r, p, w, id, data_double, data_int):
    if np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) > data_double[0]:
        data_double[1] += w[0]
    else:
        w[0] = 0


@cfunc(types.particle_loop_callback)
def photonBinning_Callback(r, p, w, id, data_double, data_int):
    nrBins = data_int[0] - 1
    binWidth = data_double[0] / nrBins
    # PhotonEnergy i E/mc**2
    photonEnergy = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) * consts.light_velocity / (
            consts.light_velocity ** 2 * consts.electron_mass)  ##Fotonenergi i 1/mc**2

    indx = int(photonEnergy / binWidth)
    if (indx < nrBins):
        data_double[indx + 1] += w[0]


@cfunc(types.particle_loop_callback)
def electronBinning_Callback(r, p, w, id, data_double, data_int):
    nrBins = data_int[0] - 1
    binWidth = data_double[0] / nrBins

    p_squared = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
    m2c2 = (consts.electron_mass * consts.light_velocity) ** 2
    electron_energy = np.sqrt(1.0 + p_squared / m2c2)
    # electronEnergy = np.sqrt((p[0] ** 2 + p[1] ** 2 + p[2] ** 2) * consts.light_velocity ** 2 + (consts.electron_mass * consts.light_velocity ** 2) ** 2) / (consts.light_velocity ** 2 * consts.electron_mass)

    indx = int(electron_energy / binWidth)
    if (indx < nrBins):
        data_double[indx + 1] += w[0]


@cfunc(types.particle_loop_callback)
def photonCumulative_Callback(r, p, w, id, data_double, data_int):
    nrBins = data_int[0] - 1
    binWidth = data_double[0] / nrBins
    # PhotonEnergy i E/mc**2
    photonEnergy = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) * consts.light_velocity / (
            consts.light_velocity ** 2 * consts.electron_mass)  ## Energi i gamma

    indx = int(photonEnergy / binWidth)
    if (indx < nrBins):
        data_double[indx + 1] += w[0] * photonEnergy


@cfunc(types.particle_loop_callback)
def photonScattering_Callback(r, p, w, id, data_double, data_int):  ##OBS scattering i x-y planet.
    nrBins = data_int[0]
    binWidth = np.pi / nrBins  ## 180 grader.
    thetha = np.arctan(p[1] / abs(p[0]))  # Py bestämmer tecken
    indx = int(thetha / binWidth) + int(nrBins / 2)  ##indx = 0 motsvara -90 grader, indx = nrBins = +90 grader.
    data_double[indx] += w[0]



class FieldInfo:
    def __init__(self, sim: pipic.init, densityCallBack, timeSteps=0, stopTime=0, NrOfPhotonBins=0,
                 nrScatteringBins=0, nrIterations=0, checkPoint=0):  # För långa ardument, kör drop-down.
        self.pipicSim = sim
        self.Fields = np.zeros((sim.ny, sim.nx, 6), dtype=np.double)  # Indexerad enligt 0-5, B-xyz, E-xyz
        if timeSteps != 0:
            self.ParticleMovementData = np.zeros((7, timeSteps),
                                                 dtype=np.double)  # Indexerad enligt 0-5 xyz, P-xyz för id = 0.

            self.ParticleMovementData[6] = np.linspace(0, stopTime, timeSteps)

        self.shape = np.array([sim.ny, sim.nx], dtype=np.intc)  # Index 6 är tid.
        self.fig, self.axs = None, None
        self.movementTimeStep = np.array([0, timeSteps], dtype=np.intc)  ##[CurrentStep, maxSteps]
        self.densityData = np.zeros((sim.ny, sim.nx), dtype=np.double)
        self.densityCallBack = densityCallBack
        self.nrElectrons = np.zeros((2, 1), dtype=np.double)
        self.nrPositrons = np.zeros((2, 1), dtype=np.double)
        self.nrPhotons = np.zeros((2, 1), dtype=np.double)

        self.photonBins = np.zeros((1 + NrOfPhotonBins, 1), dtype=np.double)  ##[0] innehåler max energy som uppmätes.
        self.photonCumulativeBins = np.zeros((1 + NrOfPhotonBins, 1), dtype=np.double)
        self.ElectronsBins = np.zeros((1 + NrOfPhotonBins, 1),
                                      dtype=np.double)  ##[0] innehåler max energy som uppmätes.
        self.photonScatteringBins = np.zeros((nrScatteringBins, 1), dtype=np.double)
    

    def Get_Nr_Electrons(self):
        return self.nrElectrons[1]

    def Get_Nr_Positrons(self):
        return self.nrPositrons[1]

    def Get_Nr_Photons(self):
        return self.nrPhotons[1]

    def Get_Density(self):
        return self.densityData

    def Get_Bx(self):
        return self.Fields[:, :, 0]

    def Get_By(self):
        return self.Fields[:, :, 1]

    def Get_Bz(self):
        return self.Fields[:, :, 2]

    def Get_Ex(self):
        return self.Fields[:, :, 3]

    def Get_Ey(self):
        return self.Fields[:, :, 4]

    def Get_Ez(self):
        return self.Fields[:, :, 5]

    def Get_X_TimeSeries(self):
        return self.ParticleMovementData[0, :]

    def Get_Y_TimeSeries(self):
        return self.ParticleMovementData[1, :]

    def Get_Z_TimeSeries(self):
        return self.ParticleMovementData[2, :]

    def Get_Px_TimeSeries(self):
        return self.ParticleMovementData[3, :]

    def Get_Py_TimeSeries(self):
        return self.ParticleMovementData[4, :]

    def Get_Pz_TimeSeries(self):
        return self.ParticleMovementData[5, :]

    def Get_Time(self):
        return self.ParticleMovementData[6, :]

    def UpdateMovement(self, currentTimeStep):
        self.movementTimeStep[0] = currentTimeStep
        self.pipicSim.particle_loop(name="electron", handler=PositionMomentumRetriverCallback.address,
                                    data_int=pipic.addressof(self.movementTimeStep),
                                    data_double=pipic.addressof(self.ParticleMovementData))
    

    def GetFieldChanges(self):
        self.pipicSim.field_loop(handler=FieldRetrieverCallback.address, data_int=pipic.addressof(self.shape),
                                 data_double=pipic.addressof(self.Fields))

    def UpdateDensity(self):
        self.densityData.fill(0)
        self.pipicSim.particle_loop(name='electron', handler=self.densityCallBack.address,
                                    data_double=pipic.addressof(self.densityData))

    def UpdateNrElectronsPositronsPhotons(self, positronElectronCutoffMomentum, photonCutoffMomentum):
        self.nrElectrons[1] = 0
        self.nrPositrons[1] = 0
        self.nrPhotons[1] = 0

        self.nrElectrons[0] = positronElectronCutoffMomentum
        self.nrPositrons[0] = positronElectronCutoffMomentum
        self.nrPhotons[0] = photonCutoffMomentum
        self.pipicSim.particle_loop(name='electron', handler=N_e_ep_cb.address,
                                    data_double=pipic.addressof(self.nrElectrons))
        self.pipicSim.particle_loop(name='positron', handler=N_e_ep_cb.address,
                                    data_double=pipic.addressof(self.nrPositrons))
        self.pipicSim.particle_loop(name='photon', handler=photonCounter_cb.address,
                                    data_double=pipic.addressof(self.nrPhotons))

    def PhotonBinning(self, maximumEnergy):
        self.photonBins = self.photonBins * 0
        self.photonBins[0] = maximumEnergy
        shapePass = np.array(self.photonBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="photon", handler=photonBinning_Callback.address,
                                    data_int=pipic.addressof(shapePass), data_double=pipic.addressof(self.photonBins))

    def ElectronBinning(self, maximumEnergy):
        self.ElectronsBins = self.ElectronsBins * 0
        self.ElectronsBins[0] = maximumEnergy
        shapePass = np.array(self.ElectronsBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="electron", handler=electronBinning_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.ElectronsBins))

    def PhotonCumulativeBinning(self, maximumEnergy):
        self.photonCumulativeBins[0] = maximumEnergy
        shapePass = np.array(self.photonCumulativeBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="photon", handler=photonCumulative_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.photonCumulativeBins))

    def PhotonScatteringBinning(self):
        shapePass = np.array(self.photonScatteringBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="photon", handler=photonScattering_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.photonScatteringBins))




    def plotElektronMomentumDataPass(self, extraString):
        data = np.vstack((self.Get_Time(), self.Get_Px_TimeSeries(), self.Get_Py_TimeSeries()))
        np.save(f"plotSweepOutput/data{extraString}", data)
        
        
        
    def ElektronMomentumPlot(self, outputFolder, extraString=""):
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        _xmin = self.pipicSim.xmin
        _xmax = self.pipicSim.xmax
        _ymin = self.pipicSim.ymin
        _ymax = self.pipicSim.ymax
        # Px som funktion av tid
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.plot(self.Get_Time(), self.Get_Px_TimeSeries()-self.Get_Px_TimeSeries()[0])
        #ax.set_title(r"$\Delta P_x(t)$")
        ax.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_x$ [g cm/s]')
        ax.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
        ax.set_xlim(0, self.Get_Time()[-1])
        fig.savefig(outputFolder + f"/imPx{extraString}.png", dpi=300)

        # Py som funktion av tid
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.plot(self.Get_Time(), self.Get_Py_TimeSeries()-self.Get_Py_TimeSeries()[0])
        #ax.set_title(r"$\Delta P_y(t)$")
        ax.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_y$ [g cm/s]')
        ax.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
        ax.set_xlim(0, self.Get_Time()[-1])
        fig.savefig(outputFolder + f"/imPy{extraString}.png", dpi=300)

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.plot(self.Get_Time(), self.Get_Pz_TimeSeries() - self.Get_Pz_TimeSeries()[0])
        #ax.set_title(r"$\Delta P_z(t)$")
        ax.set(xlabel='$t$ [s]', ylabel=r'$\Delta P_z$ [g cm/s]')
        ax.ticklabel_format(axis='both', scilimits=(0, 0), useMathText=True)
        ax.set_xlim(0, self.Get_Time()[-1])
        fig.savefig(outputFolder + f"/imPz{extraString}.png", dpi=300)
        
    def FieldPlotAndElektronDensity(self, outputFolder, maxField, i):
        maxField = 2*maxField
        _xmin = self.pipicSim.xmin
        _xmax = self.pipicSim.xmax
        _ymin = self.pipicSim.ymin
        _ymax = self.pipicSim.ymax
        plt.imshow(self.Get_Ey(), vmin=-maxField, vmax=maxField,
                   extent=(_xmin, _xmax, _ymin, _ymax), interpolation='none',
                   aspect='equal', cmap='RdBu', origin='lower')
        plt.scatter(self.Get_X_TimeSeries()[i], self.Get_Y_TimeSeries()[i])
        plt.savefig(outputFolder + f"/Field{i}.png", dpi=300)
        plt.close()

    def PlotDensityDistrobution(self, outputFolder, i):
        _xmin = self.pipicSim.xmin
        _xmax = self.pipicSim.xmax
        _ymin = self.pipicSim.ymin
        _ymax = self.pipicSim.ymax
        plt.imshow(self.densityData, # vmin=-maxField, vmax=maxField,
                   extent=(_xmin, _xmax, _ymin, _ymax), interpolation='none',
                   aspect='equal', cmap='RdBu', origin='lower')
        plt.savefig(outputFolder + f"/density{i}.png", dpi=300)
        plt.close()