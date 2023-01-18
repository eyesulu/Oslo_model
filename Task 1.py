#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:37:12 2023

@author: aisulu
"""

import Oslo_model_optimised
import matplotlib.pylab as plt
import numpy as np
import time
import pylab as pl
import matplotlib.font_manager as font_manager
plt.style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"


#%%

#The following section tests the average height of L=16 and L=32 ricepiles

a = Oslo (16, [1, 2], [0.5, 0.5])
a.initialise()

heights_a =[]
for i in range (20000):
    a.start()
    heights_a.append(a._sites[0])
print ('The average height of the ricepile for L=16 is %.1f' %(np.average(heights_a)))
print ('Number of grains needed for the system to reach its first steady state = %.0f' %(a.get_number_of_grains()))
print()

b = Oslo (32, [1, 2], [0.5, 0.5])
b.initialise()
heights_b =[]
for i in range (20000):
    b.start()
    heights_b.append(b._sites[0])
print ('The average height of the ricepile for L=32 is %.1f' %(np.average(heights_b)))
print ('Number of grains needed for the system to reach its first steady state = %.0f' %(b.get_number_of_grains()))
print()
print ('Both average heights are equal to the ones given in the Project Notes Task 1')

#%%

#The following test compares the original Oslo model and a BTW model

BTW_model = Oslo (16, [1, 1], [0.5, 0.5])
Oslo_model = Oslo (16, [1, 2], [0.5, 0.5])

heights_BTW =[]
heights_Oslo = []
test_index = np.arange(500)

for i in range (500):
    BTW_model.start()
    heights_BTW.append(BTW_model._sites[0])
    Oslo_model.start()
    heights_Oslo.append(Oslo_model._sites[0])
    
plt.plot(test_index,heights_BTW, label = 'BTW model')
plt.plot(test_index,heights_Oslo, label = 'Oslo model')
plt.legend()
plt.xlabel('Time (number of grains added)')
plt.ylabel('The heiight of the ricepile (number of grains)')
plt.title('The height of the ricepile for BTW and Oslo model with 500 dropped grains')
plt.ylim(0, 30)
plt.show()

#%%

#The following test checks the number of slope threshold values of all sites of the ricepile
#The ricepile has L = 100

threshold_vals =[]
test_threshold = Oslo (100, [1, 2], [0.5, 0.5])
for k in range (test_threshold._L):
    threshold_vals.append(test_threshold._sites_zth[k])
    
plt.hist(threshold_vals, bins=[0.5, 1.5, 2.5], edgecolor = 'white', linewidth=3)
plt.title('The threshold values for the initial setup')
plt.xlabel('The threshold slope value')
plt.ylabel('Frequency')

#%%

for i in range (10000):
    for j in range (test_threshold._L):
        threshold_vals.append(test_threshold._sites_zth[j])
    test_threshold.start()

plt.hist(threshold_vals, bins=[0.5, 1.5, 2.5], edgecolor = 'white', linewidth=3)
plt.title('The threshold values for each site for 10000 added grains')
plt.xlabel('The threshold slope value')
plt.ylabel('Frequency')   

#%%

#The following test demonstrates the avalanche sizes for L = 50 and L = 100 ricepiles

test_aval_size_50 = Oslo (50, [1, 2], [0.5, 0.5])
test_aval_size_50.initialise()
aval_sizes_50=[]
test_index_av_50 = np.arange(1000)
for i in range(1000):
    test_aval_size_50.start()
    aval_sizes_50.append(test_aval_size_50._avalanche_size)
plt.plot(test_index_av_50, aval_sizes_50)
plt.xlabel('Time (number of grains added)')
plt.ylabel('The avalanche size (number of toppled grains)')
plt.title('The avalance sizes for an Oslo model with L = 50')

#%%

test_aval_size_100 = Oslo (100, [1, 2], [0.5, 0.5])
test_aval_size_100.initialise()
aval_sizes_100=[]
test_index_av_100 = np.arange(1000)
for i in range(1000):
    test_aval_size_100.start()
    aval_sizes_100.append(test_aval_size_100._avalanche_size)
plt.plot(test_index_av_100, aval_sizes_100)
plt.xlabel('Time (number of grains added)')
plt.ylabel('The avalanche size (number of toppled grains)')
plt.title('The avalance sizes for an Oslo model with L = 100')

#%%

#The following test shows how much time is required for the initialisatoin to execute
#L = 256

test_big = Oslo (256, [1, 2], [0.5, 0.5])
#The next lines contain a function to check the time of code execution
#The function was found at : https://www.geeksforgeeks.org/how-to-check-the-execution-time-of-python-script/
start = time.time()
test_big.initialise()
end = time.time()
print ('The L = 256 initialisation was completed in %.1f minutes' %((end-start)/60))

#%%

#The following test demonstrates the animation for L = 4 

test_anim = Oslo (4)

test_anim.animate(n=30)
