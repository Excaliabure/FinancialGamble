from .utilities._utils import get_day, check_database
import numpy as np
import pandas as pd
import datetime
import os



class day:
    def __init__(self, pair, database=True):
        pairs = []
        self.data = []
        self.database_path = os.path.join(os.path.dirname(__file__), "database"),pair

        if type(pair) == str:
            pair = pair.replace("_","")
            pairs = [pair]
        else:
            pairs = pair
        for p in pairs:

            # Checks database to see if pair has been recently updated
            # check_database(self.database_path[0],p)
            
            # Checks & Updates pair if necessary
           
            r = get_day(p.replace("_",""))
            self.data.append(r)


        self.chlo = self.to_numpy()[0][:,1:5]

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
    


