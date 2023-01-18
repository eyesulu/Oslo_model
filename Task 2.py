#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:38:31 2023

@author: aisulu
"""

import Oslo_model_optimised
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import statistics as stat

red = list(plt.rcParams['axes.prop_cycle'])[0]['color']
blue = list(plt.rcParams['axes.prop_cycle'])[1]['color']
purple = list(plt.rcParams['axes.prop_cycle'])[2]['color']
grey = list(plt.rcParams['axes.prop_cycle'])[3]['color']
yellow = list(plt.rcParams['axes.prop_cycle'])[4]['color']
green = list(plt.rcParams['axes.prop_cycle'])[5]['color']
pink = list(plt.rcParams['axes.prop_cycle'])[6]['color']
#%%
#The following section shows Task 2a
#But only for L = 4 and L = 8
#Produces a plot that shows the transients and recurrent configs

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

for i in range (100):
    L4.start()
    L4_recurrent_heights.append (L4._sites[0])
    L4_recurrent_time.append(L4._number_of_grains + i -1)
    
    L8.start()
    L8_recurrent_heights.append (L8._sites[0])
    L8_recurrent_time.append(L8._number_of_grains + i -1)
    
    if i == 0:
        L4_transient_heights.append(L4._sites[0])
        L4_transient_time = np.append(L4_transient_time, L4._number_of_grains + i -1)
        
        L8_transient_heights.append(L8._sites[0])
        L8_transient_time = np.append(L8_transient_time, L8._number_of_grains + i -1)

plt.plot (L8_transient_time ,L8_transient_heights, color=green, linestyle = '--', label = 'L = 8, transient')
plt.plot (L8_recurrent_time,L8_recurrent_heights ,color = green, label = 'L = 8, recurrent')

plt.plot (L4_transient_time ,L4_transient_heights,color = blue, linestyle = '--', label = 'L = 4, transient')
plt.plot (L4_recurrent_time,L4_recurrent_heights , color = blue , label = 'L = 4, recurrent') 

plt.xlim([0,100])
plt.legend()
plt.title('The total height of piles over time')
plt.xlabel('Time (number of grains added)')
plt.ylabel('Total height (number of grains)')
#%%
#The follwing section does Task 2a for all system sizes required
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

n = 80000

time = np.arange(n)

for i in range (n):
    L4.start()
    L4_heights.append(L4._sites[0])
    
    L8.start()
    L8_heights.append(L8._sites[0])
    
    L16.start()
    L16_heights.append(L16._sites[0])
    
    L32.start()
    L32_heights.append(L32._sites[0])
    
    L64.start()
    L64_heights.append(L64._sites[0])
    
    L128.start()
    L128_heights.append(L128._sites[0])
    
    L256.start()
    L256_heights.append(L256._sites[0])
    
    print(i)
    
plt.plot(time, L256_heights, label = 'L = 256')
plt.plot(time, L128_heights, label = 'L = 128')
plt.plot(time, L64_heights, label = 'L = 64')
plt.plot(time, L32_heights, label = 'L = 32')
plt.plot(time, L16_heights, label = 'L = 16')
plt.plot(time, L8_heights, label = 'L = 8')
plt.plot(time, L4_heights, label = 'L = 4')

plt.legend()
plt.xlabel('Time (number of grains added)')
plt.ylabel('Total height (number of grains)')
plt.title('The total height of piles over time')


#%%
#The follwoing section does Task 2b

sizes = np.arange(1, 16)
tc = [] 
for i in range (16):
    if i != 0:
        tci = []
        for j in range (100):
            test_tc = Oslo (i)
            test_tc.initialise()
            tci.append (test_tc._number_of_grains)
        tc.append(np.average(tci))

plt.scatter(sizes, tc, label ='data')

fit = np.poly1d(np.polyfit(sizes, tc, 2))
xs = np.linspace(1, 16, 50)
plt.plot(xs, fit(xs), label = 'Fit function')
plt.legend()
plt.xlabel('System size, L (number of grains')
plt.ylabel ('Average height (number of grains)')
plt.title ('Size of the system vs average height')

#%%
#This section is Task 2d 


L4 = Oslo (4, [1, 2], [0.5, 0.5])
L4h = []
L8 = Oslo (8, [1, 2], [0.5, 0.5])
L8h = []
L16 = Oslo (16, [1, 2], [0.5, 0.5])
L16h = []
L32 = Oslo (32, [1, 2], [0.5, 0.5])
L32h = []
L64 = Oslo (64, [1, 2], [0.5, 0.5])
L64h = []

times = np.arange (10000)

for i in range (10000):
    L4.start()
    L4h.append(L4._sites[0]/4)
    L8.start()
    L8h.append(L8._sites[0]/8)
    L16.start()
    L16h.append(L16._sites[0]/16)
    L32.start()
    L32h.append(L32._sites[0]/32)
    L64.start()
    L64h.append(L64._sites[0]/64)
#%%
plt.plot (times, L4h, label = 'L = 4')
plt.plot (times, L8h, label = 'L = 8')
plt.plot (times, L16h, label = 'L = 16')
plt.plot (times, L32h, label = 'L = 32')
plt.plot (times, L64h, label = 'L = 64')
plt.legend()
plt.rc('text', usetex=False)
plt.xlabel ('Time, t')
plt.ylabel ('h(t; L)/L')
plt.title ('Collapsed height versus time')

#%%
    
plt.plot (np.divide(times, 4**2), L4h, label = 'L = 4')
plt.plot (np.divide(times, 8**2), L8h, label = 'L = 8')
plt.plot (np.divide(times, 16**2), L16h, label = 'L = 16')
plt.plot (np.divide(times, 32**2), L32h, label = 'L = 32')
plt.plot (np.divide(times, 64**2), L64h, label = 'L = 64')
plt.legend()
plt.xlim ([0, 2])
plt.rc('text', usetex=False)
plt.xlabel ('\frac{t}{L^2}')
plt.ylabel ('\frac{\langle h (t;L)\rangle}{L}')
plt.title ('Collapsed height versus time')

#%%

L4 = Oslo (4, [1, 2], [0.5, 0.5]) 
L4_heights = []
L4.initialise()

L8 = Oslo (8, [1, 2], [0.5, 0.5]) 
L8_heights = []
L8.initialise()

L16 = Oslo (16, [1, 2], [0.5, 0.5]) 
L16_heights = []
L16.initialise()

L32 = Oslo (32, [1, 2], [0.5, 0.5]) 
L32_heights = []
L32.initialise()

L64 = Oslo (64, [1, 2], [0.5, 0.5]) 
L64_heights = []
L64.initialise()

L80 = Oslo (80, [1, 2], [0.5, 0.5]) 
L80_heights = []
L80.initialise()

L100 = Oslo (100, [1, 2], [0.5, 0.5]) 
L100_heights = []
L100.initialise()

L128 = Oslo (128, [1, 2], [0.5, 0.5]) 
L128_heights = []
L128.initialise()

for j in range (200):
    L4.start()
    L8.start()
    L16.start()
    L32.start()
    L64.start()
    L80.start()
    L100.start()
    L128.start()

for i in range (2000):
    L4.start()
    L4_heights.append(L4._sites[0])
    
    L8.start()
    L8_heights.append(L8._sites[0])
    
    L16.start()
    L16_heights.append(L16._sites[0])
    
    L32.start()
    L32_heights.append(L32._sites[0])
    
    L64.start()
    L64_heights.append(L64._sites[0])
    
    L80.start()
    L80_heights.append(L80._sites[0])
    
    L100.start()
    L100_heights.append(L100._sites[0])
    
    L128.start()
    L128_heights.append(L128._sites[0])
    
    print (i)

L4_aver = np.average(L4_heights)
L8_aver = np.average(L8_heights)
L16_aver = np.average(L16_heights)
L32_aver = np.average(L32_heights)
L64_aver = np.average(L64_heights)
L80_aver = np.average(L80_heights)
L100_aver = np.average(L100_heights)
L128_aver = np.average(L128_heights)
#%%
ys = [L4_aver,L8_aver,L16_aver,L32_aver,L64_aver,L80_aver, L100_aver, L128_aver ]
xs = [4, 8, 16, 32, 64, 80, 100, 128]

fit = np.poly1d(np.polyfit(xs, ys, 1))
xs_fit = np.linspace(1, 128, 50)
plt.plot(xs_fit, fit(xs_fit), label = 'Fit function (1d polynomial)', color = purple)
plt.scatter(xs, ys , label = 'Data')
plt.legend()
plt.xlabel (' L (number of grains)')
plt.ylabel(' Average height')
plt.title ('Average height versus system size')
print (fit)

#%%

def fit_func(x, a0, a1, w1):
    func = a0*x*(1-a1*x**(-w1))
    return func

po,po_cov=sp.optimize.curve_fit(fit_func,xs,ys)

y_newfit = []
for i in range (len(xs)):
    y_newfit.append( fit_func(xs[i],po[0],po[1],po[2] ))

print (po)
plt.plot(xs,y_newfit,label='Fit function (1d polynomial + scaling corrections)', color = purple)
plt.scatter(xs, ys , label = 'Data') 
plt.xlabel (' L (number of grains)')
plt.ylabel(' Average height')
plt.title ('Average height versus system size')
plt.legend()
#%%
res = ys - fit(xs)
plt.scatter(xs, res, label = '1d polynomial')
plt.title ('Residuals')
res2 = np.subtract(ys, y_newfit)
plt.scatter(xs, res2, label = '1d polynomial + scaling corrections')
plt.xlabel ('System size, L (number of grains)')
plt.ylabel('\langle h (t;L)')
plt.legend()
for i in range (len(xs)):
    plt.arrow(xs[i], res[i], 0, res2[i]-res[i], color ='grey', head_width=3, head_length=.02, length_includes_head=True, fill = False, lw=1)

#%%
#Task 2f
L4_std = stat.stdev(L4_heights)
L8_std = stat.stdev(L8_heights)
L16_std = stat.stdev(L16_heights)
L32_std = stat.stdev(L32_heights)
L64_std = stat.stdev(L64_heights)
L80_std = stat.stdev(L80_heights)
L100_std = stat.stdev(L100_heights)
L128_std = stat.stdev(L128_heights)
    
ys = [L4_std,L8_std,L16_std,L32_std,L64_std,L80_std, L100_std, L128_std ]
xs = [4, 8, 16, 32, 64, 80, 100, 128]

plt.scatter(xs, ys)
















