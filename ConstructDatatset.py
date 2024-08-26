import pandas as pd
import numpy as np
import os 
class Builddataset:
    def __init__(self):
        pass        
    def concateDataset(self,folder_name):
        data_list = []
        for filename in os.listdir(folder_name):
            name = os.path.join(folder_name,filename)
            data_list.append(pd.read_csv(name,index_col='Unnamed: 0'))
        return pd.concat(data_list)