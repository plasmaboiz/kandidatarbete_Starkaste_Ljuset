from pipic import consts
import numpy as np

import matplotlib.pyplot as plt

def spotsizeFromA0(a0, Energy):
    P = Energy / 16e-15
    spotsizeSquare = 4 * P * 1 ** 2 / (1.37e18 * a0 ** 2)
    return np.sqrt(spotsizeSquare)

n0 = 8e19 #8e18  # 1e18 # [1/cm^3]
plasmaRegionFaktor = 2e-1
# plasma frequency, [e] = statC, [me] = g, [n0] = 1/cm^3, [c] = cm/s
omega_p = np.sqrt(np.pi * 4 * consts.electron_charge ** 2 * n0*plasmaRegionFaktor / consts.electron_mass)  # [1/s]
# plasma wavelength



a0 = np.linspace(1,50, 1000)


optimumBubbleDiameter =4 * np.sqrt(a0)*consts.light_velocity/omega_p

plt.plot(a0, optimumBubbleDiameter)
plt.plot(a0, spotsizeFromA0(a0, 10))
plt.show()