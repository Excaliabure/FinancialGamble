# region imports
from AlgorithmImports import *
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# endregion


# ==============================
#   PPO Agent (Conv1D Model)
# ==============================


# =======================================
#   QuantConnect Algorithm (PPO trader)
# =======================================
class SquareFluorescentPinkLemur(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2017, 4, 11)
        self.SetEndDate(2018, 4, 11)
        self.SetCash(1_000_000)

        self.add_forex("EURUSD", Resolution.HOUR)

    # ------------------------------------------------------
    #  Handle incoming market data
    # ------------------------------------------------------
    def OnData(self, data: Slice):

        self.debug(data)
        pass
    # ------------------------------------------------------
