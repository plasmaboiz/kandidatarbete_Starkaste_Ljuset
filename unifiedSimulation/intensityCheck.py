import matplotlib.pyplot as plt
import numpy as np
import unifiedSimulationData
import pickle

plt.rcParams.update({'font.size': 18})
def loadObj(filePath) -> unifiedSimulationData.unifiedSimulationData:
    with open(filePath, "rb") as file:
        return pickle.load(file)

obj = loadObj("./lwfa_data/a0_100_length_5_0GNMOY.pickle")

print(obj.lwfaParam)

emStart = obj.lwfa_PackagedFieldStart[2, 2, :, :] #B_xyz, E_xyz

emEnd = obj.lwfa_PackagedFieldEnd[2, 2, :, :]



Ex = emStart[:, 3]
By = emStart[:, 1]
Ey = emStart[:, 4]
Bx = emStart[:, 0]

S_start = (Ex * By - Ey * Bx)

startMaxIndice = np.argmax(S_start)

Ex = emEnd[:, 3]
By = emEnd[:, 1]
Ey = emEnd[:, 4]
Bx = emEnd[:, 0]


S_End = (Ex * By - Ey * Bx)

endMaxIndice = np.argmax(S_End)
S_End = np.roll(S_End, -(endMaxIndice-startMaxIndice))

integral1 = np.trapz(S_start)
integral2 = np.trapz(S_End)
print(integral2/integral1)

print(integral1)
print(integral2)

print(np.max(S_start))
print(np.max(S_End))


normConst = np.max(S_start)
S_End = S_End/normConst
S_start = S_start/normConst



plt.plot(np.linspace(-1/2,1/2, len(S_start)),S_start, label="Start")
plt.plot(np.linspace(-1/2,1/2, len(S_start)),S_End, label="Slut")

plt.ylabel("Intensitet [a.u.]")
plt.xlabel("$x$ [a.u]")
plt.legend()
plt.tight_layout()
plt.savefig("intensityComp", dpi=300)
plt.show()




