"""[summary]
"""
import os

import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        self._data_dir = data_dir
        self._data = {}
        
        self.load_data()
        
    def load_data(self):
        experiments = ['hubble']
        
        for experiment in experiments:
            path = os.path.join(self._data_dir, experiment) + '.txt'
            data = np.loadtxt(path, comments='#', unpack=True)
            self._data[experiment] = data
                
        
    