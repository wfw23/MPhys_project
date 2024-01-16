#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:02:24 2023

@author: wfw23
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
from redback.transient_models.afterglow_models import tophat_redback

def new_tophat_func(time,  thv, loge0 , thc, logn0, p, logepse, logepsb, g0,frequency):
    flux= tophat_redback(time=time, output_format='flux_density', redshift=0.01, steps=1000, thv=thv, loge0=loge0 , thc=thc, logn0=logn0, p=p, logepse=logepse,
        logepsb=logepsb, xiN=1, g0=g0, frequency= frequency)
    return flux

samples = pd.read_csv('/emulator_samples.csv') #add path to file
logtime= np.logspace(2.94,7.41,100)/86400 #about 0.1 to 297 days
num=len(samples)

ys = np.zeros((num, len(logtime)))

#ax= plt.subplot()
for i in range(num):
    ys[i]= new_tophat_func(logtime, **samples.iloc[i])
    #ax.loglog(logtime, ys[i], color='c', alpha=0.05)
plt.show()

np.save('emulator_yvals.npy', ys) #save yvals