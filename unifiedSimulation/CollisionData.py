import pipic
from pipic import consts, types
import numpy as np
from numba import cfunc, carray



def Expand_lwfa_Data(lwfaArr: np.ndarray, nrMacroParticles):
    """Cool funktion som kan förläng en array till en annan storlek och lite extra.
   Används för att öka antalet makropartiklar i en simulering utan att
   påverka densiteten."""
    if nrMacroParticles < lwfaArr.shape[0]:
        nrMacroParticles = lwfaArr.shape[0]
    makroParticleFaktor = nrMacroParticles // lwfaArr.shape[0]
    expanded_lwfaArr = np.zeros((makroParticleFaktor * lwfaArr.shape[0], lwfaArr.shape[1]))
    for i in range(lwfaArr.shape[0]):
        tmp = lwfaArr[i][:]
        tmp[6] = tmp[6] / makroParticleFaktor
        for j in range(makroParticleFaktor):
            ##För att se till att totala elektrondensiteten är detsamma.
            expanded_lwfaArr[makroParticleFaktor * i + j][:] = tmp
    return expanded_lwfaArr


@cfunc(types.field_loop_callback)
def Field_Retriever_Complete_Callback(ind, r, E, B, data_double, data_int):
    """Fields innehållar alla fält m.h.a en 2D array. Indexerad enligt 0-6, B-xyz, E-xyz
       Plockar enbart ut fältet i Z-rummet."""
    
    Fields = carray(data_double, (data_int[0], data_int[1], data_int[2], data_int[3]), dtype=np.double)  ##Shape från data_int, 
    Fields[ind[0], ind[1], ind[2], 0] = B[0]
    Fields[ind[0], ind[1], ind[2], 1] = B[1]
    Fields[ind[0], ind[1], ind[2], 2] = B[2]
    Fields[ind[0], ind[1], ind[2], 3] = E[0]
    Fields[ind[0], ind[1], ind[2], 4] = E[1]
    Fields[ind[0], ind[1], ind[2], 5] = E[2]


@cfunc(types.field_loop_callback)
def Add_LWFA_EM_Field_Callback(ind, r, E, B, data_double, data_int):
    """"""
    Fields = carray(data_double, (data_int[0], data_int[1], data_int[2], data_int[3]), dtype=np.double)
    if ind[2] < data_int[2]:  ##läser in en hel array med pulsdata längs z-axeln.
        B[0] = Fields[ind[0], ind[1], ind[2], 0]
        B[1] = Fields[ind[0], ind[1], ind[2], 1]
        B[2] = Fields[ind[0], ind[1], ind[2], 2]
        E[0] = Fields[ind[0], ind[1], ind[2], 3]
        E[1] = Fields[ind[0], ind[1], ind[2], 4]
        E[2] = Fields[ind[0], ind[1], ind[2], 5]


@cfunc(types.particle_loop_callback)
def Add_LWFA_Elektrons_callback(r, p, w, id, data_double, data_int):
    """Adds an saved EM field to the simulation volume. nx, ny, and dx,dy,dz MUST be the same across the simultions"""
    electronData = carray(data_double, (data_int[0], data_int[1]), dtype=np.double)
    if data_int[2] < data_int[0]:  # Applies the values from our saved macro-particles array to as many particles as needed.
        # r[0] = electronData[data_int[2], 0]
        # r[1] = electronData[data_int[2], 1]
        # r[2] = electronData[data_int[2], 2] #r är read only i "2"types.particle_loop_callback"
        p[0] = electronData[data_int[2], 3]
        p[1] = electronData[data_int[2], 4]
        p[2] = -electronData[data_int[2], 5]  # Negative to emulate that the EM field has been mirrored.
        w[0] = electronData[data_int[2], 6]
        data_int[2] += 1
    else:  # When all our saved marco-particle collision_data is added, we remove the left over macro-particles.
        w[0] = 0


@cfunc(types.particle_loop_callback)
def Position_Momentum_Retriver_Callback(r, p, w, id, data_double, data_int):
    movData = carray(data_double, (6, data_int[1]), dtype=np.double)
    if id[0] == 0:  # To track a single electron, maybe should average or something?
        movData[0, data_int[0]] = r[0]  # x
        movData[1, data_int[0]] = r[1]  # y
        movData[2, data_int[0]] = r[2]  # z
        movData[3, data_int[0]] = p[0]  # Px
        movData[4, data_int[0]] = p[1]  # Py
        movData[5, data_int[0]] = p[2]  # Pz


@cfunc(types.particle_loop_callback)
def Particle_Number_Counter_Callback(r, p, w, id, data_double, data_int):
    """Counts the weight of an arbitrary particle"""
    #data_int[0] carries array position.
    if np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) > data_double[0]:
        data_double[1+data_int[0]] += w[0]
    else:
        w[0] = 0



@cfunc(types.particle_loop_callback)
def Photon_Binning_Callback(r, p, w, id, data_double, data_int):
    """Binnning process for determining the energy distribution of photons"""
    nrBins = data_int[0] - 1
    binWidth = data_double[0] / nrBins
    # PhotonEnergy i E/mc**2
    photonEnergy = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) * consts.light_velocity / (
            consts.light_velocity ** 2 * consts.electron_mass)  ##Fotonenergi i 1/mc**2

    indx = int(photonEnergy / binWidth)
    if (indx < nrBins):
        data_double[indx + 1] += w[0]


@cfunc(types.particle_loop_callback)
def Electron_Binning_Callback(r, p, w, id, data_double, data_int):
    """Binnning process for determining the energy distribution of electrons"""
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
def Photon_Cumulative_Binning_Callback(r, p, w, id, data_double, data_int):
    """Binnning process for determining the cumulative distribution of photons
    (photonEnergy * nr of photons)"""
    nrBins = data_int[0] - 1
    binWidth = data_double[0] / nrBins
    # PhotonEnergy i E/mc**2
    photonEnergy = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) * consts.light_velocity / (
            consts.light_velocity ** 2 * consts.electron_mass)  ## Energi i gamma

    indx = int(photonEnergy / binWidth)
    if (indx < nrBins):
        data_double[indx + 1] += w[0] * photonEnergy


@cfunc(types.particle_loop_callback)
def Photon_Scattering_Callback(r, p, w, id, data_double, data_int):  ##OBS scattering i y-z planet.
    """Binning process for determining the scattering angles of created photons."""
    nrBins = data_int[0]
    binWidth = np.pi / nrBins  ## 180 grader.
    thetha = np.arctan(p[0] / np.abs(p[2]))  # Py bestämmer tecken
    indx = int(thetha / binWidth) + int(nrBins / 2)  ##indx = 0 motsvara -90 grader, indx = nrBins = +90 grader.
    data_double[indx] += w[0]


class CollisionData:
    def __init__(self, sim: pipic.init, densityCallBack, nrIterations=0, checkPoint=0, stopTime=0, NrOfPhotonBins=0,
                 nrScatteringBins=0):  # För långa ardument, kör drop-down.
        
        self.pipicSim = sim
        self.CheckPoint_iterationCounter = np.zeros((1,1), dtype=np.intc)
        self.checkPoint = checkPoint
        
        if nrIterations != 0:
            self.ParticleMovementData = np.zeros((7, nrIterations),
                                                 dtype=np.double)  # Indexerad enligt 0-5 xyz, P-xyz för id = 0.

            self.ParticleMovementData[6] = np.linspace(0, stopTime, nrIterations)

        self.movementTimeStep = np.array([0, nrIterations], dtype=np.intc)  ##[CurrentStep, maxSteps]
        self.densityData = np.zeros((sim.nx, sim.ny, sim.nz), dtype=np.double)
        self.densityCallBack = densityCallBack

        self.nrElectrons = np.zeros((1+nrIterations//checkPoint+1, 1), dtype=np.double)
        self.nrPositrons = np.zeros((1+nrIterations//checkPoint+1, 1), dtype=np.double)
        self.nrPhotons = np.zeros((1+nrIterations//checkPoint+1, 1), dtype=np.double)

        self.photonBins = np.zeros((1 + NrOfPhotonBins, 1), dtype=np.double)  ##[0] innehåler max energy som uppmätes.
        self.photonCumulativeBins = np.zeros((1 + NrOfPhotonBins, 1), dtype=np.double)
        self.ElectronsBins = np.zeros((1 + NrOfPhotonBins, 1), dtype=np.double)  ##[0] innehåler max energy som uppmätes.
        self.photonScatteringBins = np.zeros((nrScatteringBins, 1), dtype=np.double)

        self.Field_Complete = np.zeros((sim.nx, sim.ny, sim.nz, 6), dtype=np.double)

    def Add_lwfa_Elektrons(self, data, nrMacroParticles):
        """Adds electrons to the simulation given an appropriate collision_data array.
        It supports creating more particles than the collision_data has, while having the same total electron weight."""
        print(f"Macro collision_data shape: {data.shape}")
        expanded_lwfa_data = Expand_lwfa_Data(data, nrMacroParticles)
        shapePass = np.array((*expanded_lwfa_data.shape, 0), dtype=np.intc)
        self.pipicSim.particle_loop(name='electron', handler=Add_LWFA_Elektrons_callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(expanded_lwfa_data))

    def Add_lwfa_Field(self, data: np.ndarray):
        """Adds an EM field given an appropriate collision_data array."""
        self.pipicSim.field_loop(handler=Add_LWFA_EM_Field_Callback.address, data_int=pipic.addressof(np.array(data.shape, dtype=np.intc)),
                                 data_double=pipic.addressof(data))
    # Getter functions.
    
    def Get_Latest_Nr_Electrons(self):
        return self.nrElectrons[1+self.CheckPoint_iterationCounter,0]

    def Get_Latest_Nr_Positrons(self):
        return self.nrPositrons[1+self.CheckPoint_iterationCounter,0]

    def Get_Latest_Nr_Photons(self):
        return self.nrPhotons[1+self.CheckPoint_iterationCounter,0]

    def Get_Density(self):
        return self.densityData

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


    def Get_Complete_FieldChanges(self):
        self.pipicSim.field_loop(handler=Field_Retriever_Complete_Callback.address, data_int=pipic.addressof(np.array(self.Field_Complete.shape, dtype=np.intc)),
                                 data_double=pipic.addressof(self.Field_Complete))

    def UpdateMovement(self, currentTimeStep):
        self.movementTimeStep[0] = currentTimeStep
        self.pipicSim.particle_loop(name="electron", handler=Position_Momentum_Retriver_Callback.address,
                                    data_int=pipic.addressof(self.movementTimeStep),
                                    data_double=pipic.addressof(self.ParticleMovementData))

    def UpdateDensity(self):
        self.densityData.fill(0)
        self.pipicSim.particle_loop(name='electron', handler=self.densityCallBack.address,
                                    data_double=pipic.addressof(self.densityData))

    def UpdateNrElectronsPositronsPhotons(self, iteration, positronElectronCutoffMomentum, photonCutoffMomentum):
        self.nrElectrons[0] = positronElectronCutoffMomentum
        self.nrPositrons[0] = positronElectronCutoffMomentum
        self.nrPhotons[0] = photonCutoffMomentum
        self.CheckPoint_iterationCounter[0] = iteration//self.checkPoint
        
        self.pipicSim.particle_loop(name='electron', handler=Particle_Number_Counter_Callback.address,
                                    data_double=pipic.addressof(self.nrElectrons), data_int=pipic.addressof(self.CheckPoint_iterationCounter))
        self.pipicSim.particle_loop(name='positron', handler=Particle_Number_Counter_Callback.address,
                                    data_double=pipic.addressof(self.nrPositrons), data_int=pipic.addressof(self.CheckPoint_iterationCounter))
        self.pipicSim.particle_loop(name='photon', handler=Particle_Number_Counter_Callback.address,
                                    data_double=pipic.addressof(self.nrPhotons), data_int=pipic.addressof(self.CheckPoint_iterationCounter))

    def PhotonBinning(self, maximumEnergy):
        self.photonBins = self.photonBins * 0
        self.photonBins[0] = maximumEnergy
        shapePass = np.array(self.photonBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="photon", handler=Photon_Binning_Callback.address,
                                    data_int=pipic.addressof(shapePass), data_double=pipic.addressof(self.photonBins))

    def ElectronBinning(self, maximumEnergy):
        self.ElectronsBins = self.ElectronsBins * 0
        self.ElectronsBins[0] = maximumEnergy
        shapePass = np.array(self.ElectronsBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="electron", handler=Electron_Binning_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.ElectronsBins))

    def PhotonCumulativeBinning(self, maximumEnergy):
        self.photonCumulativeBins = self.photonCumulativeBins * 0
        self.photonCumulativeBins[0] = maximumEnergy
        shapePass = np.array(self.photonCumulativeBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="photon", handler=Photon_Cumulative_Binning_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.photonCumulativeBins))

    def PhotonScatteringBinning(self):
        self.photonScatteringBins = self.photonScatteringBins * 0
        shapePass = np.array(self.photonScatteringBins.shape, dtype=np.intc)
        self.pipicSim.particle_loop(name="photon", handler=Photon_Scattering_Callback.address,
                                    data_int=pipic.addressof(shapePass),
                                    data_double=pipic.addressof(self.photonScatteringBins))
