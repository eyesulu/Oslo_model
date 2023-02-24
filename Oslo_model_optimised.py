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
        self._aval = 0
    def get_threshold_vals(self):
        return self._sites_zth
    def relax(self):
        unsteady = True
        self._aval = 0
        while unsteady == True:
            unsteady_sites = []
            for i in range (self._L-1):
                difference = self._sites[i]-self._sites[i+1]
                if difference > self._sites_zth[i]:
                    unsteady_sites.append(i)
            if self._sites[self._L-1] > self._sites_zth[self._L-1]:
                unsteady_sites.append(self._L-1)
            if len(unsteady_sites) != 0:
                unsteady = True
                for index in unsteady_sites:
                    self._aval += 1
                    if index == self._L-1:
                        self._sites[index] -= 1
                        self._critical = True
                    else:
                        self._sites[index] -= 1
                        self._sites[index+1] += 1
                    self._sites_zth[index] = choice(self._zth, p=self._prob)
            else:
                unsteady = False      
    def drive (self):
        self._sites[0] += 1
    def start(self):
        self.drive()
        self.relax()
    def initialise(self):
        self._transient_heights = []
        for i in range (self._L**2):
            self.start()
            if self._critical == False:
                self._transient_heights.append (self._sites[0])
                self._number_of_grains = i
            
    def animate(self, n):
        for i in range (n):
            pl.pause(0.1)
            ax = pl.axes (xlim = (0, self._L), ylim = (0,self._L*2))
            for x in self._possible_x_positions:
                if self._sites[x] != 0:
                    possible_y_positions = np.arange(self._sites[x])
                    for y in possible_y_positions:
                        patch = pl.Rectangle([x, y], 1, 1, color = 'grey' , linewidth = 1, ec = 'black')
                        ax.add_patch(patch)
            ax.set_xticks(np.arange(self._L+1))
            pl.xlabel('Position')
            pl.ylabel('Height')
            pl.show()
            self.start()
        pl.close('all')
        
    def get_heights(self, grains):
        h = []
        for i in range (grains):
            h.append(self._sites[0])
            self.start()
        return h
    
    