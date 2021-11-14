# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 10:01:58 2021

@author: asus
"""
#%%
import numpy as np
from neupy import algorithms

def draw_bin_image(image_matrix):
    for row in image_matrix.tolist():
        print('| ' + ' '.join(' *'[val] for val in row))
#%%
zero = np.matrix([
0, 1, 1, 1, 0,
1, 0, 0, 0, 1,
1, 0, 0, 0, 1,
1, 0, 0, 0, 1,
1, 0, 0, 0, 1,
0, 1, 1, 1, 0
])

one = np.matrix([
0, 1, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0
])

two = np.matrix([
1, 1, 1, 0, 0,
0, 0, 0, 1, 0,
0, 0, 0, 1, 0,
0, 1, 1, 0, 0,
1, 0, 0, 0, 0,
1, 1, 1, 1, 1,
])

data = np.concatenate([zero, one, two], axis=0)
dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
dhnet.train(data)

#%%
# Test set #

half_zero = np.matrix([
0, 1, 1, 1, 0,
1, 0, 0, 0, 1,
1, 0, 0, 0, 1,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
])

half_two = np.matrix([
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 1, 1, 0, 0,
1, 0, 0, 0, 0,
1, 1, 1, 1, 1,
])

result = dhnet.predict(half_zero)
draw_bin_image(result.reshape((6, 5)))

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))