#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 19:43:06 2024

@author: wfw23
"""

import redback
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import corner
from bilby.core.prior import Constraint, PriorDict, Uniform, Sine
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
#frequencies=[5e9, 2e17]
frequencies=[]
bandfreqs = (redback.utils.bands_to_frequency(bands))
print(bandfreqs)
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

ag_data= pd.read_csv('/home/wfw23/Mphys_proj/regime_of_contamination/onaxis_samples_redback.csv')
ag_data.drop('Unnamed: 0', axis=1, inplace=True)

time= np.logspace(3.6,6.65,60)/86400

lowerag=[]
medianag=[]
upperag=[]

for j in range(1): #loop each freqeuncy
    #maximum= []
    #minimum= []
    lower=[]
    median=[]
    upper=[]
    
    for i in range(60): #loop each point in time
        pointintime=[]
    
        for k in range(5000): #loop each flux
            offflux = redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, av=0.5,
                base_model='tophat_from_emulator', output_format='flux_density', frequency=frequencies[j], **ag_data.iloc[k])
            pointintime.append(offflux[i])
        #maximum.append(max(pointintime))
        #minimum.append(min(pointintime))
        lower.append(np.percentile(pointintime,5))
        median.append(np.percentile(pointintime,50))
        upper.append(np.percentile(pointintime,95))
        
    #maximumon.append(maximum)
    #minimumon.append(minimum)
    lowerag.append(lower)
    medianag.append(median)
    upperag.append(upper)

low1 = np.load('loweron.npy')
low2 = np.array(low1).tolist()
low2.extend(lowerag) 
med1 = np.load('medianon.npy')
med2 = np.array(med1).tolist()
med2.extend(medianag) 
up1 = np.load('upperron.npy')
up2 = np.array(up1).tolist()
up2.extend(upperag)  

np.save('loweron.npy', low2)
np.save('medianon.npy', med2)
np.save('upperon.npy', up2)


