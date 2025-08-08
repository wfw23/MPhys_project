#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:37:55 2024

@author: wfw23
"""

import redback
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

kn = redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_kilonova_base_model/GRBnew_knonly_offax_result.json')
#kn_sub = redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_kilonova_base_model/GRBnew_knonly_onax_subtracted_2_result.json')
no_err = redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_kilonova_base_model/GRBnew_knonly_offax_subtracted_noerr_result.json')
ag = redback.result.read_in_result('//home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_afterglow_base_model/GRBnew_agonly_offax_result.json')
joint = redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/afterglow_and_optical/GRBnew_both_offax_result.json')
#on = redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_afterglow_base_model/GRBnew_agonly_onax_missing_result.json')
#jon= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/afterglow_and_optical/GRBnew_both_onax_missing_result.json')
#joff= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/afterglow_and_optical/GRBnew_both_offax_missing_result.json')
#off= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_afterglow_base_model/GRBnew_agonly_offax_missing_result.json')
#sort parameters to plot from result file posteriors
def kn_params(posterior_df):
    kn_df = posterior_df['mej']*10
    kn_df = pd.concat([kn_df,posterior_df['vej_1']], axis=1)
    kn_df = pd.concat([kn_df, posterior_df['vej_2']],axis=1)
    kn_df = pd.concat([kn_df, posterior_df['kappa']/10],axis=1)
    kn_df = pd.concat([kn_df, posterior_df['beta']/10], axis=1)
    return kn_df

def ag_params(posterior_df):
    ag_df = posterior_df['thv']
    ag_df = pd.concat([ag_df, posterior_df['loge0'] / 100], axis=1)
    ag_df = pd.concat([ag_df, posterior_df['thc']*10], axis=1)
    ag_df = pd.concat([ag_df, posterior_df['logn0']], axis=1)
    return ag_df

#kn_df= kn.posterior.drop('Unnamed: 0', axis=1, inplace=True)
kn_df= kn_params(kn.posterior)
ag_df= ag_params(ag.posterior)
kn_sub = kn_params(no_err.posterior)
ag_joint = ag_params(joint.posterior)
kn_joint = kn_params(joint.posterior)

'''
kn_df.drop('log_likelihood', axis=1, inplace=True)
kn_df.drop('log_prior', axis=1, inplace=True)
kn_df.drop('redshift', axis=1, inplace=True)
kn_df.drop('av', axis=1, inplace=True)
kn_df['mej']=kn_df['mej']*10
kn_df['kappa']=kn_df['kappa']/10
kn_df['beta']=kn_df['beta']/10



#kn_sub_df= kn_subtract.posterior.drop('Unnamed: 0', axis=1, inplace=True)
kn_sub_df= kn_sub.posterior
kn_sub_df.drop('log_likelihood', axis=1, inplace=True)
kn_sub_df.drop('log_prior', axis=1, inplace=True)
kn_sub_df.drop('redshift', axis=1, inplace=True)
kn_sub_df.drop('av', axis=1, inplace=True)
kn_sub_df['mej']=kn_sub_df['mej']*10
kn_sub_df['kappa']=kn_sub_df['kappa']/10
kn_sub_df['beta']=kn_sub_df['beta']/10

no_err_df= no_err.posterior
no_err_df.drop('log_likelihood', axis=1, inplace=True)
no_err_df.drop('log_prior', axis=1, inplace=True)
no_err_df.drop('redshift', axis=1, inplace=True)
no_err_df.drop('av', axis=1, inplace=True)
no_err_df['mej']=no_err_df['mej']*10
no_err_df['kappa']=no_err_df['kappa']/10
no_err_df['beta']=no_err_df['beta']/10

ag_df= ag.posterior
ag_df.drop('log_likelihood', axis=1, inplace=True)
ag_df.drop('log_prior', axis=1, inplace=True)
ag_df.drop('redshift', axis=1, inplace=True)
ag_df.drop('av', axis=1, inplace=True)
ag_df.drop('xiN', axis=1, inplace=True)
ag_df.drop('p', axis=1, inplace=True)
ag_df.drop('g0', axis=1, inplace=True)
ag_df.drop('logepse', axis=1, inplace=True)
ag_df.drop('logepsb', axis=1, inplace=True)
ag_df['loge0']=ag_df['loge0']/100
ag_df['thc']=ag_df['thc']*10
'''
fig, ax = plt.subplots(figsize=(7,10),ncols=1,nrows=2)
#kilonova
#ag_on=ag_params(on.posterior)
#joint_on=ag_params(jon.posterior)
kn_data = [kn_df,kn_sub, kn_joint]
colors = ['steelblue', 'violet', 'limegreen']
#colors=['orangered', 'limegreen']
for i in range(3):
    violin_1 = ax[1].violinplot(kn_data[i],vert=False, showmeans=True, showextrema=True)
    # Make all the violin statistics marks red:
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violin_1[partname]
        vp.set_edgecolor(colors[i])
        vp.set_linewidth(1)

    # Make the violin body blue with a red border:
    for vp in violin_1['bodies']:
        vp.set_facecolor(colors[i])
        vp.set_edgecolor(colors[i])
        vp.set_linewidth(1)
        vp.set_alpha(0.5)

#afterglow
#ag_off=ag_params(off.posterior)
#joint_off=ag_params(joff.posterior)
ag_data =[ag_df, ag_joint]
colors=['orangered', 'limegreen']
for i in range(2):
    violin_2 = ax[0].violinplot(ag_data[i], vert=False,showmeans=True, showextrema=True)
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violin_2[partname]
        vp.set_edgecolor(colors[i])
        vp.set_linewidth(1)

    # Make the violin body blue with a red border:
    for vp in violin_2['bodies']:
        vp.set_facecolor(colors[i])
        vp.set_edgecolor(colors[i])
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
'''
violin_3 = ax.violinplot(no_err_df, vert=False,showmeans=True, showextrema=True)
for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = violin_3[partname]
    vp.set_edgecolor('y')
    vp.set_linewidth(1)

# Make the violin body blue with a red border:
for vp in violin_3['bodies']:
    vp.set_facecolor('y')
    vp.set_edgecolor('y')
    vp.set_linewidth(1)
    vp.set_alpha(0.4)
'''

injection_values=[0.03*10, 0.1, 0.4, 5/10, 4/10]
y_values=np.linspace(1,5,5)
ax[1].plot(injection_values, y_values, marker='x', color='k', ls='None',markersize=8)
ax[1].set_yticks(y_values)
param_labels=('$M_{\\mathrm{ej}}\\times 10~(M_\\odot)$', '$v_{\\mathrm{ej}~1}~(c)$',
              '$v_{\\mathrm{ej}~2}~(c)$', '$\\kappa\\times 0.1~(\\mathrm{cm}^{2}/\\mathrm{g})$', '$\\beta\\times 0.1$')
ax[1].set_yticklabels(param_labels, fontsize=14)
x_vals= [0.0,0.15,0.30,0.45,0.60,0.75]
ax[1].set_xticks(x_vals)
ax[1].set_xticklabels(x_vals,fontsize=12)
'''
injection_values=[ 0.03, 51/100, 0.07*10, 0.5]
y_values=np.linspace(1,4,4)
ax[1].plot(injection_values, y_values, marker='x', color='k', ls='None',markersize=8)
param_labels=('$\\theta_{\\mathrm{observer}}~(\\mathrm{rad})$',
 '$\\log_{10}~E_{0}\\times 0.01/{\\mathrm{erg}}$',
 '$\\theta_{\\mathrm{core}}\\times 10~({\\mathrm{rad}})$',
 '$\\log_{10}~n_{\\mathrm{ism}}/{\\mathrm{cm}}^{-3}$')
ax[1].set_yticks(y_values)
ax[1].set_yticklabels(param_labels, fontsize=14)
x_vals= [-6.0, -4.0, -2.0, 0.0, 2.0]
ax[1].set_xticks(x_vals)
ax[1].set_xticklabels(x_vals,fontsize=12)
'''

injection_values=[ 0.5, 51.5/100, 0.07*10, 1]
y_values=np.linspace(1,4,4)
ax[0].plot(injection_values, y_values, marker='x', color='k', ls='None',markersize=8)
param_labels=('$\\theta_{\\mathrm{observer}}~(\\mathrm{rad})$',
 '$\\log_{10}~E_{0}\\times 0.01/{\\mathrm{erg}}$',
 '$\\theta_{\\mathrm{core}}\\times 10~({\\mathrm{rad}})$',
 '$\\log_{10}~n_{\\mathrm{ism}}/{\\mathrm{cm}}^{-3}$')
ax[0].set_yticks(y_values)
ax[0].set_yticklabels(param_labels, fontsize=14)
x_vals= [-2.0, -1.0, 0.0,1.0, 2.0]
ax[0].set_xticks(x_vals)
ax[0].set_xticklabels(x_vals,fontsize=12)
plt.tight_layout()
plt.show()
