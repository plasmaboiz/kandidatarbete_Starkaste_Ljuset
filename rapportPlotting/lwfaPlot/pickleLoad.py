import dill as pickle
import sys
sys.path.append("../../unifiedSimulation")
from unifiedSimulation import unifiedSimulationData
import numpy as np


def loadObj(filePath) -> unifiedSimulationData.unifiedSimulationData:
    with open(filePath, "rb") as file:
        return pickle.load(file)
