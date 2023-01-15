#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:21:53 2023

Complexity Project 

@author: aisulu
"""
import numpy as np
from numpy.random import choice
import pylab as pl 


#This class is a Site class
#It initialises a site with given threshold slope value
#Also contains two functions : adding a grain to the site and substracting a grain


class Site:
    def __init__(self, threshold, height=0):
        self._site_height = height
        self._threshold = threshold
        self._possible_y_positions = np.arange(self._site_height)
    def add(self):
        self._site_height = self._site_height + 1
        self._possible_y_positions = np.arange(self._site_height)
    def sub(self):
        self._site_height = self._site_height - 1
        self._possible_y_positions = np.arange(self._site_height)
        
#This class is a Ricepile class
#It initialises a lattice of piles/sites
#Number of sites is given by L
#Possible threshold slope values are given by zth
#Probabilities of zth are given by prob
#Threshold value for a site is chosen by 'choice' function
 
class Oslo:
    def hi(self):
        return
    def __init__(self, L, zth, prob):
        self._L = L
        self._zth = zth
        self._prob = prob
        #The next line includes a 'choice' which returns an array 
        #based on a given array with possible outcomes and probabilities
        #The example code and description can be found at:
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        self._sites = list([Site(choice(self._zth, p=self._prob)) for i in range(self._L)])
        self._possible_x_positions = np.arange(self._L)
        self._critical = False
    def get_heights(self):
        return list([self._sites[i]._site_height for i in range (self._L)])
    def get_threshold_vals(self):
        return list([self._sites[i]._threshold for i in range (self._L)])
    def relax(self, index):
        if index == self._L-1:
            self._sites[index].sub()
        else:
            self._sites[index].sub()
            self._sites[index+1].add()
        self._sites[index]._threshold = choice(self._zth, p=self._prob) 
    def drive (self):
        self._sites[0].add()
    def steady_state(self):
        self._steady_sites=[]
        for i in range (self._L-1):
            difference = self._sites[i]._site_height-self._sites[i+1]._site_height
            if difference > self._sites[i]._threshold:
                self._steady_sites.append(False)
            else:
                self._steady_sites.append(True)
        if self._sites[self._L-1]._site_height > self._sites[self._L-1]._threshold:
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
                    self._avalanche_size = self._avalanche_size + 1
                    if i == self._L-1:
                        self._critical = True
    def initialise(self):
        self._number_of_grains = 0
        self._transient_heights = []
        while self._critical != True:
            self.start()
            self._transient_heights.append(self._sites[0]._site_height)
            self._number_of_grains = self._number_of_grains + 1
    def get_number_of_grains(self):
        return self._number_of_grains
    def animate(self):
        f = pl.figure()
        ax = pl.axes (xlim = (0, self._L), ylim = (0,9))
        for x in self._possible_x_positions:
            if self._sites[x]._site_height != 0:
                for y in self._sites[x]._possible_y_positions:
                    patch = pl.Rectangle([x, y], 1, 1, color = 'black' , linewidth = 3, ec = 'white')
                    ax.add_patch(patch)
        pl.show()
        
 #%%       
        
            
                
                