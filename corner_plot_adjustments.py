#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 21:43:53 2024

@author: wfw23
"""

import redback
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from bilby.core.prior import PriorDict, Uniform, Sine, Constraint
import corner


result= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_kilonova_base_model/GRBsig_off_knonly_result.json')
defaults_kwargs = dict(
            bins=50, smooth=0.9,
            title_kwargs=dict(fontsize=20),
            label_kwargs=dict(fontsize=18), color='#0072C1',
            truth_color='tab:orange' ,quantiles=[0.16, 0.84], show_titles=True, title_quantiles=[0.16,0.5,0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=3)
'''
figure=corner.corner(result.samples, **defaults_kwargs)
axes = np.array(figure.axes).reshape((9, 9))

labels=result.parameter_labels

for xi in range(9):
    ax = axes[8, xi]
    ax.tick_params(labelsize=16)
    ax.set_xlabel(labels[xi], fontsize=18, labelpad=10)
for xi in range(1,9):
    ax =axes[xi,0]
    ax.tick_params(labelsize=16)
    ax.set_ylabel(labels[7-xi], fontsize=18, labelpad=10)
'''
labels=[
 '$M_{\\mathrm{ej}}[M_{\\odot}]$',
 '$v_{\\mathrm{ej}-1} [c]$',
 '$v_{\\mathrm{ej}-2} [c]$',
 '$\\kappa [cm^{2}/g]$',
 '$\\beta$','$A_{v}$']

figure=result.plot_corner(title_kwargs=dict(fontsize=21),labels=labels, label_kwargs=dict(fontsize=23, labelpad=100))
axes = np.array(figure.axes).reshape((6, 6))
for xi in range(6):
    ax = axes[5, xi]
    ax.tick_params(labelsize=14)
    #ax.set_xlabel(labels[xi], fontsize=18, labelpad=10)
for xi in range(1,6):
    ax =axes[xi,0]
    ax.tick_params(labelsize=14)
    #ax.set_ylabel(labels[7-xi], fontsize=18, labelpad=10)

plt.show()