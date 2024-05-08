import CollisionData
import Lwfa_1d_Data
import matplotlib.pyplot as plt
import numpy as np
import CollisionData


def FieldPlotAndElektronDensity(collsionData: CollisionData.CollisionData, outputFolder, maxField, i):
    maxField = 2 * maxField
    _zmin = collsionData.pipicSim.zmin
    _zmax = collsionData.pipicSim.zmax
    _ymin = collsionData.pipicSim.ymin
    _ymax = collsionData.pipicSim.ymax
    fig, ax = plt.subplots(2,1)
    
    im = ax[0].imshow(collsionData.Field_Complete[:, 2, :, 1], vmin=-maxField, vmax=maxField,
                    extent=(_zmin, _zmax, _ymin, _ymax), interpolation='none',
                    aspect='auto', cmap='RdBu', origin='lower')
    #plt.colorbar(im)
    ax[1].imshow(collsionData.densityData[:, 2, :], extent=(_zmin, _zmax, _ymin, _ymax), aspect="auto", origin="lower")
    fig.savefig(outputFolder + f"/Field{i}.png", dpi=300)
    plt.close(fig)
