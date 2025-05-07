from .utilities._utils import get_hr
import numpy as np
import pandas as pd
import datetime



class hr:
    def __init__(self, pair):
        pairs = []
        self.data = []
        self.isNone = False


        if type(pair) == str:
            pairs = [pair]
        else:
            pairs = pair
        for p in pairs:

            self.data.append(get_hr(p))

        self.chlo = self.to_numpy()[0][:,1:5]

        # can delete if want to later
        self.y = self.data


    def to_numpy(self):
        r = []
        if self.data == None:
            return None
        for i in range(len(self.data)):
            if type(self.data[i]) == pd.DataFrame:
                r.append(self.data[i].to_numpy())
            
        return r
    
    def info(self):
        r = []
        for i in self.data:
            q = i
            if type(i) == pd.DataFrame:
                q["Datetime"] = i["Datetime"].apply(lambda x: datetime.datetime.fromtimestamp(x))
                r.append(q)

        return r
    

 


    


