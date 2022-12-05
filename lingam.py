# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:21:59 2022

@author: molae
"""

# pip install lingam
# pip install igraph
# pip install pygam
# pip install factor_analyzer

import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot, print_causal_directions, print_dagc

import warnings

# from lingam.utils import make_dot, print_causal_directions, print_dagc

# import warnings
data=pd.read_csv('D:/projects/Seminar/Code/Blockchain/crypto_forcasting/data/features.csv')
data.set_index("time_stamp",inplace=True,drop=True)


#%%
np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
causal_order = [2, 0, 1, 3, 4]
# data generated from psi0 and phi1 and theta1, causal_order
!kaggle kernels output singh2299/varmalingam-causal -p ./data

X = np.loadtxt('data/sample_data_varma_lingam.csv', delimiter=',')
model = lingam.VARMALiNGAM(order=(1, 1), criterion=None)
model.fit(X)