#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:32:53 2023

@author: aisulu
"""

import sys
sys.path.insert(0, '/Users/aisulu/Desktop/CandN')
from Complexity import Oslo
import matplotlib.pyplot as plt
import numpy as np
#%%

L4 = Oslo (4, [1, 2], [0.5, 0.5]) 
L4.initialise()
L4_transient_heights = L4._transient_heights
L4_transient_time = np.arange(L4._number_of_grains)
L4_recurrent_heights = []
L4_recurrent_time = []

L8 = Oslo (8, [1, 2], [0.5, 0.5]) 
L8.initialise()
L8_transient_heights = L8._transient_heights
L8_transient_time = np.arange(L8._number_of_grains)
L8_recurrent_heights = []
L8_recurrent_time = []

for i in range (50+L8._number_of_grains):
    L4.start()
    L4_recurrent_heights.append (L4._sites[0]._site_height)
    L4_recurrent_time.append(L4._number_of_grains + i -1)
    
    L8.start()
    L8_recurrent_heights.append (L8._sites[0]._site_height)
    L8_recurrent_time.append(L8._number_of_grains + i -1)
    
    if i == 0:
        L4_transient_heights.append(L4._sites[0]._site_height)
        L4_transient_time = np.append(L4_transient_time, L4._number_of_grains + i -1)
        
        L8_transient_heights.append(L8._sites[0]._site_height)
        L8_transient_time = np.append(L8_transient_time, L8._number_of_grains + i -1)

plt.plot (L8_transient_time ,L8_transient_heights, color = 'magenta', linestyle = '--', label = 'L = 8, transient')
plt.plot (L8_recurrent_time,L8_recurrent_heights, color = 'magenta' , label = 'L = 8, recurrent')

plt.plot (L4_transient_time ,L4_transient_heights, color = 'purple', linestyle = '--', label = 'L = 4, transient')
plt.plot (L4_recurrent_time,L4_recurrent_heights, color = 'purple' , label = 'L = 4, recurrent') 

plt.xlim([0,50+L8._number_of_grains])
plt.legend()
plt.title('The total height of piles over time')
plt.xlabel('Time (number of grains added)')
plt.ylabel('Total height (number of grains)')
#%%
L4 = Oslo (4, [1, 2], [0.5, 0.5]) 
L4_heights = []

L8 = Oslo (8, [1, 2], [0.5, 0.5]) 
L8_heights = []

L16 = Oslo (16, [1, 2], [0.5, 0.5]) 
L16_heights = []

L32 = Oslo (32, [1, 2], [0.5, 0.5]) 
L32_heights = []

L64 = Oslo (64, [1, 2], [0.5, 0.5]) 
L64_heights = []

L128 = Oslo (128, [1, 2], [0.5, 0.5]) 
L128_heights = []

L256 = Oslo (256, [1, 2], [0.5, 0.5]) 
L256_heights = []

n = 70000

time = np.arange(n)

for i in range (n):
    L4.start()
    L4_heights.append(L4._sites[0]._site_height)
    
    L8.start()
    L8_heights.append(L8._sites[0]._site_height)
    
    L16.start()
    L16_heights.append(L16._sites[0]._site_height)
    
    L32.start()
    L32_heights.append(L32._sites[0]._site_height)
    
    L64.start()
    L64_heights.append(L64._sites[0]._site_height)
    
    L128.start()
    L128_heights.append(L128._sites[0]._site_height)
    
    L256.start()
    L256_heights.append(L256._sites[0]._site_height)
    
    print(i)
    
plt.plot(time, L256_heights, color = 'red', label = 'L = 256')
plt.plot(time, L128_heights, color = 'orange', label = 'L = 128')
plt.plot(time, L64_heights, color = 'green', label = 'L = 64')
plt.plot(time, L32_heights, color = 'dodgerblue', label = 'L = 32')
plt.plot(time, L16_heights, color = 'blue', label = 'L = 16')
plt.plot(time, L8_heights, color = 'magenta', label = 'L = 8')
plt.plot(time, L4_heights, color = 'purple', label = 'L = 4')

plt.legend()
plt.xlabel('Time (number of added grains)')
plt.ylabel('Total height (number of grains)')
plt.title('The total height of piles over time')


#%%




