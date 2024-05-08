import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 20})


fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
# PhotonSpektra
# collision_data: np.ndarray = np.load("./collision_data/preCollision_photonBins.npy")
# n_bins_ph = collision_data.shape[0] - 1
# binWidth = (collision_data[0] / n_bins_ph) * 0.511  # Gamma till MeV, ty E/(mc**2) = gamma. Detta är dividerat med n_bins+1
# edges = np.arange(0, collision_data.shape[0]) * binWidth
# normedPhotonsBins = (collision_data[1:, 0] / binWidth) / np.sum(collision_data[1:, 0])
# 
# ax.stairs(normedPhotonsBins, edges, fill=False)

data: np.ndarray = np.load("./data/postCollision_photonBins.npy")
n_bins_ph = data.shape[0] - 1
binWidth = (data[0] / n_bins_ph) * 0.511  # Gamma till MeV, ty E/(mc**2) = gamma. Detta är dividerat med n_bins+1
edges = np.arange(0, data.shape[0]) * binWidth
normedPhotonsBins = (data[1:, 0] / binWidth) / np.sum(data[1:, 0])
ax.stairs(normedPhotonsBins, edges, fill=False)
ax.set_yscale("log")
ax.set_xlim(0,400)

ax.set(xlabel='$E$ [MeV]', ylabel=r'$dN_f/dE$ [$\text{MeV}^{-1}$]')
fig.savefig("../_plots/photonSpectra.png", dpi=300)


fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
# ElectronSpektra
# collision_data: np.ndarray = np.load("./collision_data/preCollision_electronBins.npy")
# n_bins_ph = collision_data.shape[0] - 1
# binWidth = (collision_data[0] / n_bins_ph) * 0.511  # Gamma till MeV, ty E/(mc**2) = gamma. Detta är dividerat med n_bins+1
# edges = np.arange(0, collision_data.shape[0]) * binWidth
# normedElectronBins = (collision_data[1:, 0] / binWidth) / np.sum(collision_data[1:, 0])
# ax.stairs(normedElectronBins, edges, fill=False)

data: np.ndarray = np.load("./data/postCollision_electronBins.npy")
n_bins_ph = data.shape[0] - 1
binWidth = (data[0] / n_bins_ph) * 0.511  # Gamma till MeV, ty E/(mc**2) = gamma. Detta är dividerat med n_bins+1
edges = np.arange(0, data.shape[0]) * binWidth
normedElectronBins = (data[1:, 0] / binWidth) / np.sum(data[1:, 0])

ax.stairs(normedElectronBins, edges, fill=False)
ax.set_yscale("log")
ax.set(xlabel='$E$ [MeV]', ylabel=r'$dN_e/dE$ [$\text{MeV}^{-1}$]')
ax.set_xlim(1300, 2070)

fig.savefig("../_plots/electronSpectra.png", dpi=300)

# Scattering
data = np.load("./data/postCollision_photonScatteringBins.npy")

fig, ax = plt.subplots(1,1)

fig.set_tight_layout(True)

n_bins = data.shape[0]
binWidth = np.pi / n_bins
edges = np.arange(0, data.shape[0] + 1) * binWidth * (
        180 / np.pi) - 90  # Edges i grader. -90 till + 90

ax.stairs(data[:, 0]/binWidth/np.sum(data[:,0]), edges, fill=False, color='blue')
ax.set(xlabel=r'$\theta$', ylabel=r'$dN_f/d\theta$')
ax.set_xlim(-0.5,0.5)
fig.savefig('../_plots/scatteringAngle.png', dpi=300)

