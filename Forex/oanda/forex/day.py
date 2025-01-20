from .utilities._utils import get_day
import numpy as np
import pandas as pd
import datetime



class hr:
    def __init__(self, pair):
        pairs = []
        self.data = []


        if type(pair) == str:
            pairs = [pair]
        else:
            pairs = pair
        for p in pairs:

            self.data.append(get_day(p))
        


    def to_numpy(self):
        a = []
        for i in range(len(self.data)):
            if type(self.data[i]) == pd.DataFrame:
                a.append(self.data[i].to_numpy())
        return a
 

    def info(self):
        r = []
        for i in self.data:
            q = i
            if type(i) == pd.DataFrame:
                q["Datetime"] = i["Datetime"].apply(lambda x: datetime.datetime.fromtimestamp(x))
                r.append(q)

        return r
    


