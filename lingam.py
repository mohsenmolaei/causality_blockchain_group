# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:21:59 2022

@author: molae
"""

!pip install lingam
!pip install igraph
!pip install pygam
!pip install factor_analyzer
!pip install yfinance

import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot, print_causal_directions, print_dagc
import matplotlib.pyplot as plt
import seaborn as sns


# print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
# np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
#%%
B0 = [
[0,-0.12,0,0,0],
[0,0,0,0,0],
[-0.41,0.01,0,-0.02,0],
[0.04,-0.22,0,0,0],
[0.15,0,-0.03,0,0],
]
B1 = [
[-0.32,0,0.12,0.32,0],
[0,-0.35,-0.1,-0.46,0.4],
[0,0,0.37,0,0.46],
[-0.38,-0.1,-0.24,0,-0.13],
[0,0,0,0,0],
]
causal_order = [1, 0, 3, 2, 4]
# data generated from B0 and B1
X = pd.read_csv('https://raw.githubusercontent.com/cdt15/lingam/master/examples/data/sample_data_var_lingam.csv')

#%%
import yfinance as yf
try : #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    data = yf.download("BTC-USD ETH-USD BNB-USD XRP-USD ADA-USD LTC-USD",start="2019-1-10", end="2022-12-10" , interval ="1d") #start="2022-11-06", end="2022-12-06" 
except Exception as e:
    print("error")
        
print(data.Close)

X = np.array(data.Close)

#%%
model = lingam.VARLiNGAM(lags= 5)
model.fit(X)

#%%
model.causal_order_
print(model.causal_order_)

print(model.adjacency_matrices_.shape)
#%%
model.adjacency_matrices_[0]

model.adjacency_matrices_[1]

print(model.adjacency_matrices_.shape)
model.residuals_

dlingam = lingam.DirectLiNGAM()
dlingam.fit(model.residuals_)
dlingam.adjacency_matrix_

#%%
labels = ['x0(t)', 'x1(t)', 'x2(t)', 'x3(t)', 'x4(t)', 'x5(t)', 'x0(t-1)', 'x1(t-1)', 'x2(t-1)', 'x3(t-1)', 'x4(t-1)' , 'x5(t-1)' ]
make_dot(np.hstack(model.adjacency_matrices_), ignore_shape=True, lower_limit=0.05,labels=labels)


#%%
p_values = model.get_error_independence_p_values()
print(p_values)

#%%
model = lingam.VARLiNGAM()
result = model.bootstrap(X, n_sampling=100)

#%%
cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.3, split_by_causal_effect_sign=True)

#%%
print_causal_directions(cdc, 100, labels=labels)

#%%
dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.2,split_by_causal_effect_sign=True)

#%%
print_dagc(dagc, 100, labels=labels)

#%%
prob = result.get_probabilities(min_causal_effect=0.1)
print('Probability of B0:\n', prob[0])
print('Probability of B1:\n', prob[1])

#%%
causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)
df = pd.DataFrame(causal_effects)
df['from'] = df['from'].apply(lambda x : labels[x])
df['to'] = df['to'].apply(lambda x : labels[x])
df

#%%
df.sort_values('effect', ascending=False).head()
df[df['to']=='x1(t)'].head()

#%%

sns.set()
# %matplotlib inline
from_index = 7 # index of x2(t-1). (index:2)+(n_features:5)*(lag:1) = 7
to_index = 2 # index of x2(t). (index:2)+(n_features:5)*(lag:0) = 2
plt.hist(result.total_effects_[:, to_index, from_index])



