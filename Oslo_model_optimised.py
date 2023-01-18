#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:53:48 2023

@author: aisulu
"""

import numpy as np
from numpy.random import choice
import pylab as pl 


        
#This class is a Ricepile class
#It initialises a lattice of piles/sites
#Number of sites is given by L
#Possible threshold slope values are given by zth
#Probabilities of zth are given by prob
#Threshold value for a site is chosen by 'choice' function
 
class Oslo:
    def __init__(self, L, zth = [1, 2], prob = [0.5, 0.5]):
        self._L = L
        self._zth = zth
        self._prob = prob
        #The next line includes a 'choice' which returns an array 
        #based on a given array with possible outcomes and probabilities
        #The example code and description can be found at:
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        self._sites = list(0 for i in range(self._L))
        self._sites_zth = list (choice(self._zth, p=self._prob) for i in range(self._L))
        self._possible_x_positions = np.arange(self._L)
        self._critical = False
    def get_heights(self):
        return self._sites
    def get_threshold_vals(self):
        return self._sites_zth
    def relax(self, index):
        if index == self._L-1:
            self._sites[index] -= 1
        else:
            self._sites[index] -= 1
            self._sites[index+1] += 1
        self._sites_zth[index] = choice(self._zth, p=self._prob) 
    def drive (self):
        self._sites[0] += 1
    def steady_state(self):
        self._steady_sites=[]
        for i in range (self._L-1):
            difference = self._sites[i]-self._sites[i+1]
            if difference > self._sites_zth[i]:
                self._steady_sites.append(False)
            else:
                self._steady_sites.append(True)
        if self._sites[self._L-1] > self._sites_zth[self._L-1]:
            self._steady_sites.append(False)
        else:
            self._steady_sites.append(True)
        #The next line creates a variable which deteermines if the system is unstable
        #The system is unstable if any element of self._steady_sites is False
        #If the system is unstable the self._unstable = True
        #The code was taken from https://www.geeksforgeeks.org/python-check-if-any-element-in-list-satisfies-a-condition/
        self._unsteady = True in (sites == False for sites in self._steady_sites)
    def get_steady_sites(self):
        return self._steady_sites
    def start(self):
        self.drive()
        self.steady_state()
        self._avalanche_size = 0
        while self._unsteady == True:
            for i in range(self._L):
                if self._steady_sites[i] == False:
                    self.relax(index=i)
                    self.steady_state()
                    self._avalanche_size += 1
                    if i == self._L-1:
                        self._critical = True
    def initialise(self):
        self._number_of_grains = 0
        self._transient_heights = []
        while self._critical != True:
            self.start()
            self._transient_heights.append(self._sites[0])
            self._number_of_grains += 1
    def get_number_of_grains(self):
        return self._number_of_grains
    def animate(self, n):
        for i in range (n):
            pl.pause(0.1)
            ax = pl.axes (xlim = (0, self._L), ylim = (0,self._L*2))
            for x in self._possible_x_positions:
                if self._sites[x] != 0:
                    possible_y_positions = np.arange(self._sites[x])
                    for y in possible_y_positions:
                        patch = pl.Rectangle([x, y], 1, 1, color = 'black' , linewidth = 3, ec = 'white')
                        ax.add_patch(patch)
            ax.set_xticks(np.arange(self._L+1))
            pl.xlabel('Position')
            pl.ylabel('Height')
            pl.show()
            self.start()
        pl.close('all')