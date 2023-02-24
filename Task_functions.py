#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:28:20 2023

@author: aisulu
"""

import Oslo_model_optimised
import matplotlib.pylab as plt
import numpy as np
import time
import scipy as sp
from scipy.optimize import curve_fit
import statistics as stat
from scipy.signal import savgol_filter
import logbin as lb
from scipy.stats import linregress
import pandas as pd

plt.style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"
red = list(plt.rcParams['axes.prop_cycle'])[0]['color']
blue = list(plt.rcParams['axes.prop_cycle'])[1]['color']
purple = list(plt.rcParams['axes.prop_cycle'])[2]['color']
grey = list(plt.rcParams['axes.prop_cycle'])[3]['color']
yellow = list(plt.rcParams['axes.prop_cycle'])[4]['color']
green = list(plt.rcParams['axes.prop_cycle'])[5]['color']
pink = list(plt.rcParams['axes.prop_cycle'])[6]['color']

def power_func(x, A, p):
    return A*x**p
def fit_func(x, a0, a1, w1):
    return a0*(1-a1*x**(-w1))
def gaus(x, a, mu, std):
    return a * np.exp(-(x - mu)**2.0 / (2 * std**2))

def Task_1():
    print ('Checking the average heights for L = 16 and 32:')
    a = Oslo(16)
    a.initialise()
    b = Oslo(32)
    b.initialise()
    a_heights = []
    b_heights = []
    for i in range (20000):
        a.start()
        a_heights.append(a._sites[0])
        b.start()
        b_heights.append(b._sites[0])
    print ('The average pile height of L = 16 system is %.1f' %(np.average(a_heights)))
    print ('The average pile height of L = 32 system is %.1f' %(np.average(b_heights)))
    
def Test():
    print ('Checking the height of each pile for L = 16 system size:')
    a = Oslo (16)
    a.initialise ()
    print (a._sites)
    print ('The difference in neighboring piles heights does not exceed 2, hence the model works correclty')
    
def Test_1():
    print ('Comparing the original Oslo model and a BTW model:')
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
    plt.ylabel('The height of the ricepile (number of grains)')
    #plt.title('The height of the ricepile for BTW and Oslo model with 500 dropped grains')
    plt.ylim(0, 30)
    plt.show()
    
def Test_2():
    print ('Checking the threshold slope values with p = 1/2')
    threshold_vals =[]
    test_threshold = Oslo (100)
    for k in range (test_threshold._L):
        threshold_vals.append(test_threshold._sites_zth[k])
    y = []
    for i in range (10):
        y.append(50)
    threshold_vals_after = []
    for i in range (10000):
        for j in range (test_threshold._L):
            threshold_vals_after.append(test_threshold._sites_zth[j])
        test_threshold.start()
    figure, axis = plt.subplots(1, 2, figsize = (10, 5))
    axis[0].hist(threshold_vals, bins=[0.5, 1.5, 2.5], edgecolor = 'white', linewidth=3, color = blue)
    axis[0].set_title('Initial setup')
    axis[0].plot(np.linspace(0, 3, 10), y, '--', color = red, label = 'Expected frequency')
    axis[0].set_xticks([0 , 1, 2, 3])
    axis[0].set_xlim([0.5, 2.5])
    axis[1].hist(threshold_vals_after, bins=[0.5, 1.5, 2.5], edgecolor = 'white', linewidth=3, color = blue)
    axis[1].set_title('After 10000 added grains')
    axis[1].plot(np.linspace(0, 3, 10), np.multiply(y, 10000), '--', color = red, label = 'Expected frequency')
    axis[1].set_xticks([0 , 1, 2, 3])
    axis[1].set_xlim([0.5, 2.5])
    figure.supxlabel('The threshold slope value')
    figure.supylabel('Frequency')

def Test_3():
    print ('Comparing avalanche sizes for L = 50 and 100 system sizes:')
    test_aval_size_50 = Oslo (50)
    test_aval_size_50.initialise()
    aval_sizes_50=[]
    test_aval_size_100 = Oslo (100)
    test_aval_size_100.initialise()
    aval_sizes_100=[]
    test_index_av_50 = np.arange(1000)
    test_index_av_100 = np.arange(1000)
    for i in range(1000):
        test_aval_size_50.start()
        aval_sizes_50.append(test_aval_size_50._aval)
        test_aval_size_100.start()
        aval_sizes_100.append(test_aval_size_100._aval)
    indicies_100 = []
    indicies_50 = []
    for i in range (len (aval_sizes_100)):
        if aval_sizes_100[i] == 0 :
            indicies_100.append (i)
        if aval_sizes_50[i] == 0 :
            indicies_50.append (i)
    aval_sizes_100 = np.delete(aval_sizes_100, indicies_100)
    test_index_av_100 = np.delete(test_index_av_100, indicies_100)
    aval_sizes_50 = np.delete (aval_sizes_50, indicies_50)
    test_index_av_50 = np.delete(test_index_av_50, indicies_50)
    plt.plot(test_index_av_100, aval_sizes_100, color = green, label = 'L = 100')
    plt.plot(test_index_av_50, aval_sizes_50, color = blue, label = 'L = 50')
    plt.xlabel('Time (number of grains added)')
    plt.ylabel('The avalanche size (number of toppled grains)')
    plt.title('The avalance sizes for an Oslo model with L = 50 and 100')
    plt.legend()
    
def Test_4():
    print ('Checking the time required to initialise (the system to enter steady state) L = 256 system size:')
    test_big = Oslo (256)
    #The next lines contain a function to check the time of code execution
    #The function was found at : https://www.geeksforgeeks.org/how-to-check-the-execution-time-of-python-script/
    start = time.time()
    test_big.initialise()
    end = time.time()
    print ('The L = 256 initialisation was completed in %.1f minutes' %((end-start)/60))

def Test_5():
    print ('Checking the animation of the Oslo model thus allowing to check the implementation visually:')
    test_anim = Oslo (16)
    test_anim.animate(n=40)

def Task_2a_1config():
    print ('Investigating transient and recurrent configurations:')
    L4 = pd.read_csv ('Data_files/L4')
    L8 = pd.read_csv ('Data_files/L8')
    L16 = pd.read_csv ('Data_files/L16')
    L32 = pd.read_csv ('Data_files/L32')
    L64 = pd.read_csv ('Data_files/L64')
    L128 = pd.read_csv ('Data_files/L128')
    L256 = pd.read_csv ('Data_files/L256')
    
    L4_transient_heights = []
    L4_transient_time = []
    L4_recurrent_heights = []
    L4_recurrent_time = []
    
    L8_transient_heights = []
    L8_transient_time = []
    L8_recurrent_heights = []
    L8_recurrent_time = []
    
    rec_time = 0
    for i in range (len(L4)):
        if L4.at[i,'Critical?'] == False:
            L4_transient_heights.append (L4.at[i, 'Height'])
            L4_transient_time.append (i)
        else:
            L4_recurrent_heights.append (L4.at[i, 'Height'])
            L4_recurrent_time.append (i)
            
        if L8.at[i,'Critical?'] == False:
            L8_transient_heights.append (L8.at[i, 'Height'])
            L8_transient_time.append (i)
        else:
            L8_recurrent_heights.append (L8.at[i, 'Height'])
            L8_recurrent_time.append (i)
    
    L4_transient_heights.append (L4_recurrent_heights[0])
    L4_transient_time.append (L4_transient_time[-1]+1)
    
    L8_transient_heights.append (L8_recurrent_heights[0])
    L8_transient_time.append (L8_transient_time[-1]+1)
    
    plt.plot (L8_transient_time ,L8_transient_heights, color=green, linestyle = '--', label = 'L = 8, transient')
    plt.plot (L8_recurrent_time,L8_recurrent_heights ,color = green, label = 'L = 8, recurrent')

    plt.plot (L4_transient_time ,L4_transient_heights,color = blue, linestyle = '--', label = 'L = 4, transient')
    plt.plot (L4_recurrent_time,L4_recurrent_heights , color = blue , label = 'L = 4, recurrent') 

    plt.xlim([0,100])
    plt.legend()
    #plt.title('The total height of piles over time')
    plt.xlabel('Time (number of grains added)')
    plt.ylabel('Total height (number of grains)')

def Task_2a():
    L4 = pd.read_csv ('Data_files/L4')['Height']
    L8 = pd.read_csv ('Data_files/L8')['Height']
    L16 = pd.read_csv ('Data_files/L16')['Height']
    L32 = pd.read_csv ('Data_files/L32')['Height']
    L64 = pd.read_csv ('Data_files/L64')['Height']
    L128 = pd.read_csv ('Data_files/L128')['Height']
    L256 = pd.read_csv ('Data_files/L256')['Height']
    
    time = np.arange(len(L4))
    
    plt.plot(time, L256, label = 'L = 256')
    plt.plot(time, L128, label = 'L = 128')
    plt.plot(time, L64, label = 'L = 64')
    plt.plot(time, L32, label = 'L = 32')
    plt.plot(time, L16, label = 'L = 16')
    plt.plot(time, L8, label = 'L = 8')
    plt.plot(time, L4, label = 'L = 4')

    plt.legend()
    plt.xlabel('Time (number of grains added)')
    plt.ylabel('Total height (number of grains)')
    #plt.title('The total height of piles over time')

def Task_2a_1():
    #The data is smoothed using scipy.signal.savgol_filter()
    #The code can be found at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    L4_smooth = savgol_filter(pd.read_csv ('Data_files/L4')['Height'], 50, 3)
    L8_smooth = savgol_filter(pd.read_csv ('Data_files/L8')['Height'], 50, 3)
    L16_smooth = savgol_filter(pd.read_csv ('Data_files/L16')['Height'], 50, 3)
    L32_smooth = savgol_filter(pd.read_csv ('Data_files/L32')['Height'], 50, 3)
    L64_smooth = savgol_filter(pd.read_csv ('Data_files/L64')['Height'], 50, 3)
    L128_smooth = savgol_filter(pd.read_csv ('Data_files/L128')['Height'], 50, 3)
    L256_smooth = savgol_filter(pd.read_csv ('Data_files/L256')['Height'], 50, 3)
    
    time = np.arange (len(pd.read_csv ('Data_files/L4')['Height']))
    
    plt.loglog(time, L4_smooth, label = 'L = 4')
    plt.loglog(time, L8_smooth, label = 'L = 8')
    plt.loglog(time, L16_smooth, label = 'L = 16')
    plt.loglog(time, L32_smooth, label = 'L = 32')
    plt.loglog(time, L64_smooth, label = 'L = 64')
    plt.loglog(time, L128_smooth, label = 'L = 128')
    plt.loglog(time, L256_smooth, label = 'L = 256')
    
    plt.legend()
    plt.xlabel('Time (number of grains added)')
    plt.ylabel('Smoothed total height (number of grains)')
    #plt.title('The smoothed total height of piles over time')
    plt.xlim([1, 100000])
 
def Task_2b():
    sizes = [4, 6, 8, 16,32, 64, 128, 256]
    #sizes = [4, 6, 8, 16]
    tc = []
    tc_err = []
    for i in sizes:
        tci = []
        for j in range (10):
            test_tc = Oslo (i)
            test_tc.initialise()
            tci.append (test_tc._number_of_grains)
        print (i)
        tc.append(np.average(tci))
        tc_err.append(np.std(tci))

    #plt.scatter(sizes, tc, label ='data')
    plt.errorbar(sizes, tc, yerr = tc_err, fmt = '.')

    fit, cov = np.polyfit(sizes, tc, 2, cov = True)
    xs = np.linspace(1, 256, 1000)
    plt.plot(xs, np.poly1d(fit)(xs), label = 'Quadratic fit function')
    plt.legend()
    plt.xlabel('System size, L (number of grains')
    plt.ylabel ('Average cross-over time')
    print ('The fit function :', fit)
    print ('Uncertainties: ', cov)
    #plt.title ('Size of the system vs average height')
'''
The fit function : [ 0.86882138 -3.06430999 34.74255826]
Uncertainties:  [[ 7.04011440e-06 -1.75498962e-03  3.58342730e-02]
 [-1.75498962e-03  4.69387453e-01 -1.09822498e+01]
 [ 3.58342730e-02 -1.09822498e+01  5.30910598e+02]]
'''
def Task_2d():
    M = 5
    time = 100000
    #time = 1000
    h4 = np.zeros(time)
    h8 = np.zeros(time)
    h16 = np.zeros(time)
    h32 = np.zeros(time)
    h64 = np.zeros(time)
    h128 = np.zeros(time)
    h256 = np.zeros(time)

    for i in range (M):
        L4 = Oslo (4)
        L8 = Oslo (8)
        L16 = Oslo (16)
        L32 = Oslo (32)
        L64 = Oslo (64)
        L128 = Oslo (128)
        L256 = Oslo (256)
        heights4 = L4.get_heights (time)
        heights8 = L8.get_heights (time)
        heights16 = L16.get_heights (time)
        heights32 = L32.get_heights (time)
        heights64 = L64.get_heights (time)
        heights128 = L128.get_heights (time)
        heights256 = L256.get_heights (time)
        print (i)
        for j in range (time):
            h4[j] += heights4[j]
            h8[j] += heights8[j]
            h16[j] += heights16[j]
            h32[j] += heights32[j]
            h64[j] += heights64[j]
            h128[j] += heights128[j]
            h256[j] += heights256[j]
            
    h_collapse4 = savgol_filter(h4/4, 50, 3)
    h_collapse8 = savgol_filter(h8/8, 50, 3)
    h_collapse16 = savgol_filter(h16/16, 50, 3)
    h_collapse32 = savgol_filter(h32/32, 50, 3)
    h_collapse64 = savgol_filter(h64/64, 50, 3)
    h_collapse128 = savgol_filter(h128/128, 50, 3)
    h_collapse256 = savgol_filter(h256/256, 50, 3)
    times = np.arange (time)

    plt.loglog (np.divide(times, 4**2), h_collapse4, label = 'L = 4')
    plt.loglog (np.divide(times, 8**2), h_collapse8, label = 'L = 8')
    plt.loglog (np.divide(times, 16**2), h_collapse16, label = 'L = 16')
    plt.loglog (np.divide(times, 32**2), h_collapse32, label = 'L = 32')  
    plt.loglog (np.divide(times, 64**2), h_collapse64, label = 'L = 64')
    plt.loglog (np.divide(times, 128**2), h_collapse128, label = 'L = 128')
    plt.loglog (np.divide(times, 256**2), h_collapse256, label = 'L = 256')
     
    po,po_cov=sp.optimize.curve_fit(power_func,np.divide(times, 256**2)[1:][:10000],h_collapse256[1:][:10000])
       
    #po,po_cov=sp.optimize.curve_fit(power_func,np.divide(times, 16**2)[1:][:10000],h_collapse16[1:][:500])
    print ('The exponential scaling parameter = ', po[1], '+-', np.sqrt(po_cov[1, 1]))
    
    plt.legend()
    plt.xlabel("$t/L^2$")
    plt.ylabel("$\widetilde h/L$")
    plt.title('Collapsed processed height')

L4_aver = np.average(pd.read_csv('Data_files/L4')['Height'][100000:])
L8_aver = np.average(pd.read_csv('Data_files/L8')['Height'][100000:])
L16_aver = np.average(pd.read_csv('Data_files/L16')['Height'][100000:])
L32_aver = np.average(pd.read_csv('Data_files/L32')['Height'][100000:])
L64_aver = np.average(pd.read_csv('Data_files/L64')['Height'][100000:])
L128_aver = np.average(pd.read_csv('Data_files/L128')['Height'][100000:])
L256_aver = np.average(pd.read_csv('Data_files/L256')['Height'][100000:])

L4_std = stat.stdev(pd.read_csv('Data_files/L4')['Height'][100000:])
L8_std = stat.stdev(pd.read_csv('Data_files/L8')['Height'][100000:])
L16_std = stat.stdev(pd.read_csv('Data_files/L16')['Height'][100000:])
L32_std = stat.stdev(pd.read_csv('Data_files/L32')['Height'][100000:])
L64_std = stat.stdev(pd.read_csv('Data_files/L64')['Height'][100000:])
L128_std = stat.stdev(pd.read_csv('Data_files/L128')['Height'][100000:])
L256_std = stat.stdev(pd.read_csv('Data_files/L256')['Height'][100000:])

ys_std = [L4_std,L8_std,L16_std,L32_std,L64_std, L128_std, L256_std ]
xs_std = [4, 8, 16, 32, 64, 128, 256]

ys_L = [L4_aver/4,L8_aver/8,L16_aver/16,L32_aver/32,L64_aver/64, L128_aver/128, L256_aver/256 ]
xs = [4, 8, 16, 32, 64, 128, 256]
ys = [L4_aver,L8_aver,L16_aver,L32_aver,L64_aver, L128_aver, L256_aver]

po,po_cov=sp.optimize.curve_fit(fit_func,xs,ys_L, p0 = [1.7, 0.22, 0.58])

xs_fit = np.linspace(4, 256, 1000)
y_newfit = []
for i in range (len(xs_fit)):
    y_newfit.append( fit_func(xs_fit[i],po[0],po[1],po[2] ))
    
def Task_2e():
    plt.plot(xs, ys_L, 'o')
    plt.xlabel (' L (number of grains)')
    plt.ylabel(' Average height/L')
    plt.plot(xs_fit,y_newfit,label='Fit function (1d polynomial + scaling corrections)', color = purple)
def Task_2e_2():
    plt.plot(xs_fit,np.multiply(y_newfit, xs_fit),label='Fit function (1d polynomial + scaling corrections)', color = purple)
    plt.scatter(xs, ys , label = 'Data') 
    plt.xlabel (' L (number of grains)')
    plt.ylabel(' Average height')
    #plt.title ('Average height versus system size')
    plt.legend()
    
    print (po)
    print (po_cov[0,0],po_cov[1,1],po_cov[2,2])
    return po
    
def Task_2f():
    plt.scatter(xs_std, ys_std)
    plt.xlabel (' L (number of grains)')
    plt.ylabel('Standard deviation of average height')
    
def Task_2f_2():
    po,po_cov=sp.optimize.curve_fit(power_func,xs_std,ys_std)
    
    fit_xs = np.linspace (4, np.max(xs_std), 1000)
    fit_ys = []
    for i in range (len(fit_xs)):
        fit_ys.append( power_func(fit_xs[i],po[0],po[1] ))
        
    plt.loglog(xs_std, ys_std, 'o')
    plt.loglog(fit_xs, fit_ys, label = 'Fit function')
    plt.legend()
    plt.ylabel("Standard deviation") 
    plt.xlabel("$L$")
    #plt.title ('Fitting the standard deviation')
    
    print ('Coefficients are: ')
    print ('Amplitude = ', po[0], '+-', po_cov[0,0])
    print ('Power = ', po[1], '+-', po_cov[1,1])
    return po
'''
Coefficients are: 
Amplitude =  0.5821387033158694 +- sqrt 1.5416345452716284e-05
Power =  0.24007217560530403 +- sqrt 2.264618326549138e-06
'''
def Task_2f_3():
    Ls = [4,8, 16, 32, 64, 128, 256]
    expected = [1, 1, 1, 1, 1, 1, 1]
    po,po_cov=sp.optimize.curve_fit(power_func,xs_std,ys_std)
    fit_ys = []
    for i in range (len(Ls)):
        fit_ys.append( power_func(Ls[i],0.5821387033158694,0.24007217560530403 ))
        
    plt.plot(xs_std,np.divide(ys_std,fit_ys), 'o')
    plt.plot(Ls, expected, label = 'Fit function')
    plt.legend()
    plt.ylabel(r"$\frac{\sigma}{\alpha L^{\beta}}$") 
    plt.xlabel("$L$")
    #plt.title ('Fitting the standard deviation')

def Task_2g():
    plt.hist(pd.read_csv('Data_files/L4')['Height'][100000:], bins = 5, density = True)
    plt.hist(pd.read_csv('Data_files/L8')['Height'][100000:], bins = 5, density = True)
    plt.hist(pd.read_csv('Data_files/L16')['Height'][100000:], bins = 5, density = True)
    plt.hist(pd.read_csv('Data_files/L32')['Height'][100000:], bins = 5, density = True)
    plt.hist(pd.read_csv('Data_files/L64')['Height'][100000:], bins = 5, density = True)
    plt.hist(pd.read_csv('Data_files/L128')['Height'][100000:], bins = 5, density = True)
    plt.hist(pd.read_csv('Data_files/L256')['Height'][100000:], bins = 5, density = True)

vals4, edges4 = np.histogram(pd.read_csv('Data_files/L4')['Height'][100000:], bins = range(3, 12), density = True)
xs4 = []
for i in range(len(vals4)):
    xs4.append((edges4[i]+edges4[i+1])*0.5)
   
initial_g4 = [1, 6, 3]
po4,po_cov4=sp.optimize.curve_fit(gaus,xs4,vals4, p0=initial_g4)
x_fit4 = np.linspace(3,14,1000)
y_fit4 = []
for i in range (len(x_fit4)):
    y_fit4.append( gaus(x_fit4[i],po4[0],po4[1],po4[2] ))
    
vals8, edges8 = np.histogram(pd.read_csv('Data_files/L8')['Height'][100000:], bins= range (9, 25), density = True)
xs8 = []
for i in range(len(vals8)):
    xs8.append((edges8[i]+edges8[i+1])*0.5)
   
initial_g8 = [1, 13, 2]
po8,po_cov8=sp.optimize.curve_fit(gaus,xs8,vals8, p0=initial_g8)
x_fit8 = np.linspace(9,17,1000)
y_fit8 = []
for i in range (len(x_fit8)):
    y_fit8.append( gaus(x_fit8[i],po8[0],po8[1],po8[2] ))
 
vals16, edges16 = np.histogram(pd.read_csv('Data_files/L16')['Height'][100000:], bins=range(20,40), density = True)
xs16 = []
for i in range(len(vals16)):
    xs16.append((edges16[i]+edges16[i+1])*0.5)
   
initial_g = [1, 26, 3]
po2,po_cov2=sp.optimize.curve_fit(gaus,xs16,vals16, p0 = initial_g)
x_fit16 = np.linspace(20,31,1000)
y_fit16 = []
for i in range (len(x_fit16)):
    y_fit16.append( gaus(x_fit16[i],po2[0],po2[1],po2[2] ))
    
vals32, edges32 = np.histogram(pd.read_csv('Data_files/L32')['Height'][100000:], bins=range(48,60), density = True)
xs32 = []
for i in range(len(vals32)):
    xs32.append((edges32[i]+edges32[i+1])*0.5)
   
initial_g32 = [1, 32, 4]
po32,po_cov32=sp.optimize.curve_fit(gaus,xs32,vals32, p0=initial_g32)
x_fit32 = np.linspace(49,60,1000)
y_fit32 = []
for i in range (len(x_fit32)):
    y_fit32.append( gaus(x_fit32[i],po32[0],po32[1],po32[2] ))
 
    
vals64, edges64 = np.histogram(pd.read_csv('Data_files/L64')['Height'][100000:], bins=range(100,120), density = True)
xs64 = []
for i in range(len(vals64)):
    xs64.append((edges64[i]+edges64[i+1])*0.5)
   
initial_g64 = [1, 64, 8]
po64,po_cov64=sp.optimize.curve_fit(gaus,xs64,vals64, p0=initial_g64)
x_fit64 = np.linspace(100,117,1000)
y_fit64 = []
for i in range (len(x_fit64)):
    y_fit64.append( gaus(x_fit64[i],po64[0],po64[1],po64[2] ))
  
vals128, edges128 = np.histogram(pd.read_csv('Data_files/L128')['Height'][100000:], bins=range(213,228), density = True)
xs128 = []
for i in range(len(vals128)):
    xs128.append((edges128[i]+edges128[i+1])*0.5)
   
initial_g128 = [1, 218, 5]
po128,po_cov128=sp.optimize.curve_fit(gaus,xs128,vals128, p0 =initial_g128 )
x_fit128 = np.linspace(213,228,1000)
y_fit128 = []
for i in range (len(x_fit128)):
    y_fit128.append( gaus(x_fit128[i],po128[0],po128[1],po128[2] ))
   
vals256, edges256 = np.histogram(pd.read_csv('Data_files/L256')['Height'][100000:], bins=range(431, 447), density = True)
xs256 = []
for i in range(len(vals256)):
    xs256.append((edges256[i]+edges256[i+1])*0.5)
   
initial_g256 = [1, 440, 10]
po256,po_cov256=sp.optimize.curve_fit(gaus,xs256,vals256, p0 = initial_g256)
x_fit256 = np.linspace(430,450,1000)
y_fit256 = []
for i in range (len(x_fit256)):
    y_fit256.append( gaus(x_fit256[i],po256[0],po256[1],po256[2] ))

def Task_2g_2():
    plt.plot(x_fit256,y_fit256,  label = 'L = 256')            
    plt.plot(x_fit128,y_fit128,  label = 'L = 128') 
    plt.plot(x_fit64,y_fit64,  label = 'L = 64')   
    plt.plot(x_fit32,y_fit32, label = 'L = 32')   
    plt.plot(x_fit16,y_fit16,label = 'L = 16')
    plt.plot(x_fit8,y_fit8,label = 'L = 8')
    plt.plot(x_fit4,y_fit4,label = 'L = 4') 
    plt.plot(xs256, vals256, '.' , color = 'black')
    plt.plot(xs128, vals128, '.' , color = 'black')
    plt.plot(xs64, vals64, '.' , color = 'black' )
    plt.plot(xs32, vals32, '.' , color = 'black' )
    plt.plot(xs16, vals16, '.' , color = 'black')
    plt.plot(xs8, vals8, '.' , color = 'black')
    plt.plot(xs4, vals4, '.' , color = 'black' )
    
    plt.xlabel ('Height of the pile')
    plt.ylabel (r'$P(h; L)$')
    plt.legend()
    plt.show()

def Task_2g_3():
    plt.plot(np.multiply(np.subtract(xs256, L256_aver), L256_std**-1),np.multiply(vals256, L256_std),  label = 'L = 256') 
    plt.plot(np.multiply(np.subtract(xs128, L128_aver), L128_std**-1),np.multiply(vals128, L128_std),  label = 'L = 128') 
    plt.plot(np.multiply(np.subtract(xs64, L64_aver), L64_std**-1),np.multiply(vals64, L64_std),  label = 'L = 64') 
    plt.plot(np.multiply(np.subtract(xs32, L32_aver), L32_std**-1),np.multiply(vals32, L32_std),  label = 'L = 32') 
    plt.plot(np.multiply(np.subtract(xs16, L16_aver), L16_std**-1),np.multiply(vals16, L16_std),  label = 'L = 16') 
    plt.plot(np.multiply(np.subtract(xs8, L8_aver), L8_std**-1),np.multiply(vals8, L8_std),  label = 'L = 8') 
    plt.plot(np.multiply(np.subtract(xs4, L4_aver), L4_std**-1),np.multiply(vals4, L4_std),  label = 'L = 4') 
    plt.legend()
    plt.xlabel (r"$(h - \langle h \rangle) /  \sigma_h$")
    plt.ylabel ("$\sigma_h P(h; L)$")
    
x4, y4 = lb.logbin(pd.read_csv('Data_files/L4')['Avalanche size'][100000:], scale = 1.3)
x8, y8 = lb.logbin(pd.read_csv('Data_files/L8')['Avalanche size'][100000:], scale = 1.3)
x16, y16 = lb.logbin(pd.read_csv('Data_files/L16')['Avalanche size'][100000:], scale = 1.3)
x32, y32 = lb.logbin(pd.read_csv('Data_files/L32')['Avalanche size'][100000:], scale = 1.3)
x64, y64 = lb.logbin(pd.read_csv('Data_files/L64')['Avalanche size'][100000:], scale = 1.3)
x128, y128 = lb.logbin(pd.read_csv('Data_files/L128')['Avalanche size'][100000:], scale = 1.3)
x256, y256 = lb.logbin(pd.read_csv('Data_files/L256')['Avalanche size'][100000:], scale = 1.3)

def getting_prob (data):
    #the counter method was taken from :https://realpython.com/python-counter/
    #it creates a dict of different values in data and their frequency 
    counter = {}
    for i in data:
        counter[i] = counter.get(i, 0) + 1

    values = []
    probs = []

    for i in counter:
        values.append (i)
        probs.append (counter[i])
        
    probs = np.divide (probs, len(data))
    return values, probs

def Task_3a():
    plt.loglog(x4, y4, label = 'L = 4')
    plt.loglog(x8, y8, label = 'L = 8')
    plt.loglog(x16, y16, label = 'L = 16')
    plt.loglog(x32, y32, label = 'L = 32')
    plt.loglog(x64, y64, label = 'L = 64')
    plt.loglog(x128, y128, label = 'L = 128')
    plt.loglog(x256, y256, label = 'L = 256')
    plt.xlabel('$s$')
    plt.ylabel (r"$\widetilde P_N(s;L)$")
    #plt.title('Avalanche size probability')
    plt.legend()
    
def Task_3aa():
    plt.loglog(x256, y256, label = 'L = 256')
    vals, probs = getting_prob(pd.read_csv('Data_files/L256')['Avalanche size'][100000:])
    plt.loglog(vals, probs, '.')
    plt.ylabel("$P_N(s;L)$")
    plt.xlabel ('$s$')
    
def Task_3a_2():
    xs = x256[:-12][5:]
    ys = y256[:-12][5:]
    
    print (len(ys))
    
    plt.loglog(x256, y256, '.')
    
    fit_xs = np.linspace (np.min(x256), np.max(x256), 1000)
    fit, res = np.polyfit (np.log(xs), np.log(ys), 1, cov=True)
    plt.loglog (fit_xs, power_func (fit_xs, np.exp(fit[1]), fit[0]), c = 'black', linewidth = 0.8, label = 'Fit function')
        
    plt.legend()
    plt.xlabel('$s$')
    plt.ylabel (r"$\widetilde P_N(s;L)$")
    #plt.title('Avalanche size probability')
    
    print ('Coefficients are: ')
    print ('Amplitude = ', np.exp(fit[1]), '+-', np.sqrt(res[0][0]))
    print ('Power = ', fit[0], '+-', np.sqrt(res[1][1]) )
    
    '''
    Coefficients are: 
    Amplitude =  0.35278101990120475 +- 0.0031118604981898853
    Power =  -1.5458985504413807 +- 0.019109322178242457
    '''
def Task_3a_22():
    Ls = [4, 8, 16, 32, 64, 128, 256]
    sc = []
    for i in Ls:
        sc.append(max(pd.read_csv(f'Data_files/L{i}')['Avalanche size'][100000:]))
        
    xs = Ls
    ys = sc
    
    print (len(ys))
    
    plt.loglog(xs, ys, '.')
    
    fit_xs = np.linspace (np.min(xs), np.max(xs), 1000)
    fit, res = np.polyfit (np.log(xs), np.log(ys), 1, cov=True)
    plt.loglog (fit_xs, power_func (fit_xs, np.exp(fit[1]), fit[0]), c = 'black', linewidth = 0.8, label = 'Fit function')
        
    plt.legend()
    plt.xlabel('$L$')
    plt.ylabel (r"$s_c (L)$")
    #plt.title('Cutoff avalnche size')
    
    print ('Coefficients are: ')
    print ('Amplitude = ', np.exp(fit[1]), '+-', np.sqrt(res[0][0]))
    print ('Power = ', fit[0], '+-', np.sqrt(res[1][1]) )
    
    '''
    Coefficients are: 
    Coefficients are: 
    Amplitude =  1.5550414057418998 +- 0.021876958252207028
    Power =  2.1750186329843024 +- 0.08166038027991207
    '''
def Task_3a_31():
    tau = 1.5458985504413807
    plt.loglog(x4, y4*(x4**tau), label = 'L = 4')
    plt.loglog(x8, y8*(x8**tau), label = 'L = 8')
    plt.loglog(x16, y16*(x16**tau), label = 'L = 16')
    plt.loglog(x32, y32*(x32**tau), label = 'L = 32')
    plt.loglog(x64, y64*(x64**tau), label = 'L = 64')
    plt.loglog(x128, y128*(x128**tau), label = 'L = 128')
    plt.loglog(x256, y256*(x256**tau), label = 'L = 256')
    plt.xlabel('$s$')
    plt.ylabel (r"$ s^{\tau_s} \widetilde P_N(s;L)$")
    #plt.title('Avalanche size probability')
    plt.legend()
    
def Task_3a_3():
    tau = 1.5458985504413807
    D = 2.1750186329843024
    
    plt.loglog(x4/(4**D), y4*(x4**tau), label = 'L = 4')
    plt.loglog(x8/(8**D), y8*(x8**tau), label = 'L = 8')
    plt.loglog(x16/(16**D), y16*(x16**tau), label = 'L = 16')
    plt.loglog(x32/(32**D), y32*(x32**tau), label = 'L = 32')
    plt.loglog(x64/(64**D), y64*(x64**tau), label = 'L = 64')
    plt.loglog(x128/(128**D), y128*(x128**tau), label = 'L = 128')
    plt.loglog(x256/(256**D), y256*(x256**tau), label = 'L = 256')
    plt.xlabel('$s / L^D$')
    plt.ylabel (r"$ s^{\tau_s} \widetilde P_N(s;L)$")
    #plt.title('Avalanche size probability')
    plt.legend()

T = 500000
l4_aval = pd.read_csv('Data_files/L4')['Avalanche size'][100000:]
l8_aval = pd.read_csv('Data_files/L8')['Avalanche size'][100000:]
l16_aval = pd.read_csv('Data_files/L16')['Avalanche size'][100000:]
l32_aval = pd.read_csv('Data_files/L32')['Avalanche size'][100000:]
l64_aval = pd.read_csv('Data_files/L64')['Avalanche size'][100000:]
l128_aval = pd.read_csv('Data_files/L128')['Avalanche size'][100000:]
l256_aval = pd.read_csv('Data_files/L256')['Avalanche size'][100000:]

sk14 = (1/T) * np.sum(l4_aval)
sk18 = (1/T) * np.sum(l8_aval)
sk116 = (1/T) * np.sum(l16_aval)
sk132 = (1/T) * np.sum(l32_aval)
sk164 = (1/T) * np.sum(l64_aval)
sk1128 = (1/T) * np.sum(l128_aval)
sk1256 = (1/T) * np.sum(l256_aval)

sk1 = [sk14, sk18, sk116, sk132, sk164, sk1128, sk1256]

sk24 = (1/T) * np.sum(np.power(l4_aval, 2.0))
sk28 = (1/T) * np.sum(np.power(l8_aval, 2.0))
sk216 = (1/T) * np.sum(np.power(l16_aval, 2.0))
sk232 = (1/T) * np.sum(np.power(l32_aval, 2.0))
sk264 = (1/T) * np.sum(np.power(l64_aval, 2.0))
sk2128 = (1/T) * np.sum(np.power(l128_aval, 2.0))
sk2256 = (1/T) * np.sum(np.power(l256_aval, 2.0))

sk2 = [sk24, sk28, sk216, sk232, sk264, sk2128, sk2256]

sk34 = (1/T) * np.sum(np.power(l4_aval, 3.0))
sk38 = (1/T) * np.sum(np.power(l8_aval, 3.0))
sk316 = (1/T) * np.sum(np.power(l16_aval, 3.0))
sk332 = (1/T) * np.sum(np.power(l32_aval, 3.0))
sk364 = (1/T) * np.sum(np.power(l64_aval, 3.0))
sk3128 = (1/T) * np.sum(np.power(l128_aval, 3.0))
sk3256 = (1/T) * np.sum(np.power(l256_aval, 3.0))

sk3 = [sk34, sk38, sk316, sk332, sk364, sk3128, sk3256]

sk44 = (1/T) * np.sum(np.power(l4_aval, 4.0))
sk48 = (1/T) * np.sum(np.power(l8_aval, 4.0))
sk416 = (1/T) * np.sum(np.power(l16_aval, 4.0))
sk432 = (1/T) * np.sum(np.power(l32_aval, 4.0))
sk464 = (1/T) * np.sum(np.power(l64_aval, 4.0))
sk4128 = (1/T) * np.sum(np.power(l128_aval, 4.0))
sk4256 = (1/T) * np.sum(np.power(l256_aval, 4.0))

sk4 = [sk44, sk48, sk416, sk432, sk464, sk4128, sk4256]

L = [4, 8, 16, 32, 64, 128, 256]

def Task_3b():
    plt.loglog(L, sk1, '.', label = 'k = 1', color = blue)
    plt.loglog(L, sk2, '.', label = 'k = 2', color = green)
    plt.loglog(L, sk3, '.', label = 'k = 3', color = red)
    plt.loglog(L, sk4, '.', label = 'k = 4', color = purple)
    plt.ylabel(r'$\langle s^k \rangle$')
    plt.xlabel ('$L$')
    plt.legend()
    
    xs = np.linspace (4, 256, 1000)
    fitsk1, cov1 = np.polyfit (np.log(L), np.log(sk1), 1, cov = True)
    fitsk2, cov2 = np.polyfit (np.log(L), np.log(sk2), 1, cov = True)
    fitsk3, cov3 = np.polyfit (np.log(L), np.log(sk3), 1, cov = True)
    fitsk4, cov4 = np.polyfit (np.log(L), np.log(sk4), 1, cov = True)
    
    plt.loglog (xs, power_func (xs, np.exp(fitsk1[1]), fitsk1[0]), c = blue, linewidth = 0.8)
    plt.loglog (xs, power_func (xs, np.exp(fitsk2[1]), fitsk2[0]), c = green, linewidth = 0.8)
    plt.loglog (xs, power_func (xs, np.exp(fitsk3[1]), fitsk3[0]), c = red, linewidth = 0.8)
    plt.loglog (xs, power_func (xs, np.exp(fitsk4[1]), fitsk4[0]), c = purple, linewidth = 0.8)
    
    k_slopes = [fitsk1[0], fitsk2[0], fitsk3[0], fitsk4[0]]
    k_error = [np.sqrt(cov1[0, 0]), np.sqrt(cov2[0, 0]), np.sqrt(cov3[0, 0]),np.sqrt(cov4[0, 0]) ]
    return k_slopes, k_error

def Task_3b_2(k_slopes):
    k = [1, 2, 3, 4]
    plt.plot (k , k_slopes, '.')
    fitk, cov = np.polyfit (k, k_slopes, 1, cov = True)
    ks = np.linspace (0.5, 4.5, 1000)
    plt.plot(ks, np.poly1d(fitk)(ks), color = 'black', linewidth = 0.8)
    
    tau_s = 1 -fitk[1]/fitk[0]
    
    plt.xlabel('$k$')
    plt.ylabel(r'$D(1+k-\tau_s)$')
            
    print ('D = ' ,fitk[0], '+-' , np.sqrt(cov[0,0]))
    print ('tau_s = ' ,tau_s, '+-' , np.sqrt(cov[0,0])+np.sqrt(cov[1,1]))




