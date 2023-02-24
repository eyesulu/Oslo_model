#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:21:01 2023

@author: aisulu
"""
import Oslo_model_optimised
import csv

#For L = 4
L4 = Oslo (4)
L8 = Oslo (8)
L16 = Oslo(16)
L32 = Oslo (32)
L64 = Oslo (64)
L128 = Oslo (128)
L256 = Oslo (256)
Column_names = ['Height', 'Avalanche size', 'Critical?']


rows_L4 =[]
rows_L8 =[]
rows_L16 =[]
rows_L32 =[]
rows_L64 =[]
rows_L128 =[]
rows_L256 =[]

for i in range (10):
    rows_L4.append([L4._sites[0], L4._aval, L4._critical])
    L4.start()
    
    rows_L8.append([L8._sites[0], L8._aval, L8._critical])
    L8.start()
    

    rows_L16.append([L16._sites[0], L16._aval, L16._critical])
    L16.start()
    
    rows_L32.append([L32._sites[0], L32._aval, L32._critical])
    L32.start()
    
    rows_L64.append([L64._sites[0], L64._aval, L64._critical])
    L64.start()
    
    rows_L128.append([L128._sites[0], L128._aval, L128._critical])
    L128.start()
    
    rows_L256.append([L256._sites[0], L256._aval, L256._critical])
    L256.start()
    
    print (i)
  
#The procedure to save data was taken from: https://www.geeksforgeeks.org/python-save-list-to-csv/
with open('Data_files/L4_test', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(Column_names)
    write.writerows(rows_L4)
    
with open('Data_files/L8', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(Column_names)
    write.writerows(rows_L8)
    
with open('Data_files/L16', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(Column_names)
    write.writerows(rows_L16)
    
with open('Data_files/L32', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(Column_names)
    write.writerows(rows_L32)
    
with open('Data_files/L64', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(Column_names)
    write.writerows(rows_L64)
    
with open('Data_files/L128', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(Column_names)
    write.writerows(rows_L128)
    
with open('Data_files/L256', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(Column_names)
    write.writerows(rows_L256)
    




    

    
