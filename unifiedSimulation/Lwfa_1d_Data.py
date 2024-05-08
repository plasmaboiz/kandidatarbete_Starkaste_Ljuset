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


@cfunc(types.particle_loop_callback)
def SaveParticleMomentumAndPosition_Callback(r, p, w, id, data_double, data_int):
    electronData = carray(data_double, (data_int[0], data_int[1]), dtype=np.double)
    electronData[data_int[2], 0] = r[0]
    electronData[data_int[2], 1] = r[1]
    electronData[data_int[2], 2] = r[2]
    electronData[data_int[2], 3] = p[0]
    electronData[data_int[2], 4] = p[1]
    electronData[data_int[2], 5] = p[2]
    electronData[data_int[2], 6] = w[0]
    data_int[2] += 1

@cfunc(types.particle_loop_callback)
def Particle_Number_Counter_Callback(r, p, w, id, data_double, data_int):
    """Counts the weight of an arbitrary particle"""
    #data_int[0] carries array position.
    if np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) > data_double[0]:
        data_double[1] += w[0]
    else:
        w[0] = 0

@cfunc(types.field_loop_callback)
def FieldRetriever_Complete_Callback(ind, r, E, B, data_double, data_int):
    # Fields innehållar alla fält m.h.a en 2D array. Indexerad enligt 0-6, B-xyz, E-xyz
    # Plockar enbart ut fältet i Z-rummet.
    Fields = carray(data_double, (data_int[0], data_int[1], data_int[2], data_int[3]), dtype=np.double)  ##Shape från data_int, 
    Fields[ind[0], ind[1], ind[2], 0] = B[0]
    Fields[ind[0], ind[1], ind[2], 1] = B[1]
    Fields[ind[0], ind[1], ind[2], 2] = B[2]
    Fields[ind[0], ind[1], ind[2], 3] = E[0]
    Fields[ind[0], ind[1], ind[2], 4] = E[1]
    Fields[ind[0], ind[1], ind[2], 5] = E[2]


@cfunc(types.particle_loop_callback)
def N_e_ep_cb(r, p, w, id, data_double, data_int):
    if np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) > data_double[0]:
        data_double[1] += w[0]
    else:
        w[0] = 0


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


class lwfaData:
    def __init__(self, sim: pipic.init, nrElectronBins, nrSimIterations, CheckPoint):
        self._Checkpoint = CheckPoint
        self.pipicSim = sim
        self.Field_Complete = np.zeros((sim.nx, sim.ny, sim.nz, 6), dtype=np.double)
        self.ElectronsBins = np.zeros((1 + nrElectronBins, 1), dtype=np.double)
        self.SavedElectronBins = np.zeros((int(nrSimIterations / CheckPoint)+1, 1 + nrElectronBins))
        self.electronMacroParticleData = np.zeros((int(sim.get_number_of_particles() * 1.2), 7), dtype=np.double)
        self.nrElectrons = np.zeros((2, 1), dtype=np.double)
    """Sparar elektroner för att införas i nästa simulering"""


    def UpdateNrElectrons(self, electronCutoffMomentum):
        self.nrElectrons[0] = electronCutoffMomentum
        self.nrElectrons[1] = 0
        self.pipicSim.particle_loop(name='electron', handler=Particle_Number_Counter_Callback.address,
                                    data_double=pipic.addressof(self.nrElectrons))
    def Get_Complete_FieldChanges(self):
        self.pipicSim.field_loop(handler=FieldRetriever_Complete_Callback.address, data_int=pipic.addressof(np.array(self.Field_Complete.shape, dtype=np.intc)),
                                 data_double=pipic.addressof(self.Field_Complete))




        ##Glöm ej get comlete field changes

    def electornCounting(self):
        result = np.zeros(2)

        self.pipicSim.particle_loop(name='electron', handler=N_e_ep_cb.address,
                                    data_double=pipic.addressof(result))
        return result[1]

    def FieldPlotAndElektronDensity(self, outputFolder, maxField, i):
        maxField = 2 * maxField
        _zmin = self.pipicSim.zmin
        _zmax = self.pipicSim.zmax
        _ymin = self.pipicSim.ymin
        _ymax = self.pipicSim.ymax
        plt.imshow(self.Field_Complete[:, 2, :, 0],  # vmin=-maxField, vmax=maxField,
                   extent=(_zmin, _zmax, _ymin, _ymax), interpolation='none',
                   aspect='auto', cmap='RdBu', origin='lower')
        # plt.scatter(self.Get_Z_TimeSeries()[i], self.Get_Y_TimeSeries()[i])
        plt.colorbar()
        plt.savefig(outputFolder + f"/Field{i}.png", dpi=300)
        plt.close()

    def PackageCompleteFieldAndSave(self, outputFolder: str, filename: str):
        maxLocation = np.argmax(self.Field_Complete[:, :, :, 0],
                                axis=2).max()  ##Hittar största värdet längs z-axeln, vilket vi vill nu ha i mitten av "rolledField"
        shift = int(self.Field_Complete.shape[2] / 2) - maxLocation  ##Avståndet som max ska flyttas för att få i mitten av array.
        rolledField = np.roll(self.Field_Complete, shift, axis=2)  ##Rolla längs z axeln.
        np.save(path.join(outputFolder, filename), rolledField)
        return rolledField

    def electronSaving(self):
        shapePass = np.array((*self.electronMacroParticleData.shape, 0), dtype=np.intc) #The last 0 is an index
        self.electronMacroParticleData = np.zeros((int(self.pipicSim.get_number_of_particles() * 1.2), 7), dtype=np.double)
        self.pipicSim.particle_loop(name='electron', handler=SaveParticleMomentumAndPosition_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.electronMacroParticleData))
        self.electronMacroParticleData = self.electronMacroParticleData[(self.electronMacroParticleData[:, 6] > 0), :]
        #np.save(path.join(outputFolder, name), self.electronMacroParticleData)

    def ElectronBinning(self, maximumEnergy):
        self.ElectronsBins = self.ElectronsBins * 0
        self.ElectronsBins[0] = maximumEnergy
        shapePass = np.array(self.ElectronsBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="electron", handler=electronBinning_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.ElectronsBins))

    def SaveElectronBinnning(self, i):
        
        self.SavedElectronBins[int(i / self._Checkpoint)] = self.ElectronsBins[:, 0]

    def WriteSavedElectronDataToFile(self, outputFolder, name: str):
        binWidth = (self.ElectronsBins[0] / self.ElectronsBins.shape[0])
        edges = np.arange(0, self.ElectronsBins.shape[0]) * binWidth
        np.save(path.join(outputFolder, name), self.SavedElectronBins)
        np.save(path.join(outputFolder, name+"_edges"), edges)
