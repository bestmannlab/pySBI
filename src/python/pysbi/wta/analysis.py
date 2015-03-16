import os
from scipy.optimize import curve_fit
import subprocess
from brian import Hz, ms, nA, mA
import traceback
from brian.units import second, farad, siemens, volt, amp
import scipy
from scipy.signal import *
import h5py
import math
from jinja2 import Environment, FileSystemLoader
from matplotlib import cm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pylab
from scikits.learn.linear_model import LinearRegression
import sys
from pysbi import  voxel
from pysbi.config import DATA_DIR, TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.util.utils import Struct, save_to_png, save_to_eps, weibull, rt_function, get_response_time, FitWeibull, FitRT
from pysbi.wta import network
from pysbi.wta.network import run_wta


def parse_output_file_name(output_file):
    strs=output_file.split('.')
    num_groups=int(strs[2])
    input_pattern=strs[4]
    duration=float(strs[6]+'.'+strs[7])
    p_b_e=float(strs[9]+'.'+strs[10])
    p_x_e=float(strs[12]+'.'+strs[13])
    p_e_e=float(strs[15]+'.'+strs[16])
    p_e_i=float(strs[18]+'.'+strs[19])
    p_i_i=float(strs[21]+'.'+strs[22])
    p_i_e=float(strs[24]+'.'+strs[25])
    p_dcs=float(strs[26]+'.'+strs[27])
    i_dcs=float(strs[28]+'.'+strs[29])
    return num_groups,input_pattern,duration,p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e,p_dcs,i_dcs

class FileInfo():
    def __init__(self, file_name, upper_resp_threshold=30, lower_resp_threshold=None, dt=.1*ms):
        self.file_name=file_name
        f = h5py.File(file_name)

        self.num_groups=int(f.attrs['num_groups'])
        self.input_freq=np.array(f.attrs['input_freq'])
        self.trial_duration=float(f.attrs['trial_duration'])*second
        self.background_freq=float(f.attrs['background_freq'])
        self.stim_start_time=float(f.attrs['stim_start_time'])*second
        self.stim_end_time=float(f.attrs['stim_end_time'])*second
        self.network_group_size=int(f.attrs['network_group_size'])
        self.background_input_size=int(f.attrs['background_input_size'])
        self.task_input_size=int(f.attrs['task_input_size'])
        self.muscimol_amount=float(f.attrs['muscimol_amount'])*siemens
        self.injection_site=int(f.attrs['injection_site'])
        if 'p_dcs' in f.attrs:
            self.p_dcs=float(f.attrs['p_dcs'])*amp
        if 'i_dcs' in f.attrs:
            self.i_dcs=float(f.attrs['i_dcs'])*amp

        f_network_params=f['network_params']
        self.wta_params=Struct()
        self.wta_params.C=float(f_network_params.attrs['C'])*farad
        self.wta_params.gL=float(f_network_params.attrs['gL'])*siemens
        self.wta_params.EL=float(f_network_params.attrs['EL'])*volt
        self.wta_params.VT=float(f_network_params.attrs['VT'])*volt
        self.wta_params.DeltaT=float(f_network_params.attrs['DeltaT'])*volt
        self.wta_params.Vr=float(f_network_params.attrs['Vr'])*volt
        self.wta_params.Mg=float(f_network_params.attrs['Mg'])
        self.wta_params.E_ampa=float(f_network_params.attrs['E_ampa'])*volt
        self.wta_params.E_nmda=float(f_network_params.attrs['E_nmda'])*volt
        self.wta_params.E_gaba_a=float(f_network_params.attrs['E_gaba_a'])*volt
        self.wta_params.tau_ampa=float(f_network_params.attrs['tau_ampa'])*second
        self.wta_params.tau1_nmda=float(f_network_params.attrs['tau1_nmda'])*second
        self.wta_params.tau2_nmda=float(f_network_params.attrs['tau2_nmda'])*second
        self.wta_params.tau_gaba_a=float(f_network_params.attrs['tau_gaba_a'])*second
        self.wta_params.p_e_e=float(f_network_params.attrs['p_e_e'])
        self.wta_params.p_e_i=float(f_network_params.attrs['p_e_i'])
        self.wta_params.p_i_i=float(f_network_params.attrs['p_i_i'])
        self.wta_params.p_i_e=float(f_network_params.attrs['p_i_e'])

        f_pyr_params=f['pyr_params']
        self.pyr_params=Struct()
        self.pyr_params.C=float(f_pyr_params.attrs['C'])*farad
        self.pyr_params.gL=float(f_pyr_params.attrs['gL'])*siemens
        self.pyr_params.refractory=float(f_pyr_params.attrs['refractory'])*second
        self.pyr_params.w_nmda=float(f_pyr_params.attrs['w_nmda'])*siemens
        self.pyr_params.w_ampa_ext=float(f_pyr_params.attrs['w_ampa_ext'])*siemens
        self.pyr_params.w_ampa_rec=float(f_pyr_params.attrs['w_ampa_rec'])*siemens
        self.pyr_params.w_gaba=float(f_pyr_params.attrs['w_gaba'])*siemens

        f_inh_params=f['inh_params']
        self.inh_params=Struct()
        self.inh_params.C=float(f_inh_params.attrs['C'])*farad
        self.inh_params.gL=float(f_inh_params.attrs['gL'])*siemens
        self.inh_params.refractory=float(f_inh_params.attrs['refractory'])*second
        self.inh_params.w_nmda=float(f_inh_params.attrs['w_nmda'])*siemens
        self.inh_params.w_ampa_ext=float(f_inh_params.attrs['w_ampa_ext'])*siemens
        self.inh_params.w_ampa_rec=float(f_inh_params.attrs['w_ampa_rec'])*siemens
        self.inh_params.w_gaba=float(f_inh_params.attrs['w_gaba'])*siemens
        
        self.lfp_rec=None
        if 'lfp' in f:
            f_lfp=f['lfp']
            self.lfp_rec={'lfp': np.array(f_lfp['lfp'])}

        if 'voxel' in f:
            f_vox=f['voxel']
            self.voxel_params=voxel.default_params()
            self.voxel_params.eta=float(f_vox.attrs['eta'])*second
            self.voxel_params.G_base=float(f_vox.attrs['G_base'])*amp
            self.voxel_params.tau_f=float(f_vox.attrs['tau_f'])*second
            self.voxel_params.tau_s=float(f_vox.attrs['tau_s'])*second
            self.voxel_params.tau_o=float(f_vox.attrs['tau_o'])*second
            self.voxel_params.e_base=float(f_vox.attrs['e_base'])
            self.voxel_params.v_base=float(f_vox.attrs['v_base'])
            self.voxel_params.alpha=float(f_vox.attrs['alpha'])
            if 'T_2E' in f_vox.attrs:
                self.voxel_params.T_2E=float(f_vox.attrs['T_2E'])
            if 'T_2I' in f_vox.attrs:
                self.voxel_params.T_2I=float(f_vox.attrs['T_2I'])
            if 's_e_0' in f_vox.attrs:
                self.voxel_params.s_e_0=float(f_vox.attrs['s_e_0'])
            if 's_i_0' in f_vox.attrs:
                self.voxel_params.s_i_0=float(f_vox.attrs['s_i_0'])
            if 'B0' in f_vox.attrs:
                self.voxel_params.B0=float(f_vox.attrs['B0'])
            if 'TE' in f_vox.attrs:
                self.voxel_params.TE=float(f_vox.attrs['TE'])
            if 's_e' in f_vox.attrs:
                self.voxel_params.s_e=float(f_vox.attrs['s_e'])
            if 's_i' in f_vox.attrs:
                self.voxel_params.s_i=float(f_vox.attrs['s_i'])
            if 'beta' in f_vox.attrs:
                self.voxel_params.beta=float(f_vox.attrs['beta'])
            if 'k2' in f_vox.attrs:
                self.voxel_params.k2=float(f_vox.attrs['k2'])
            if 'k3' in f_vox.attrs:
                self.voxel_params.k3=float(f_vox.attrs['k3'])

            self.voxel_rec={}
            if 'total_syn' in f_vox:
                total_vox=f_vox['total_syn']
                if 'G_total' in total_vox:
                    self.voxel_rec['G_total']=np.array(total_vox['G_total'])
                if 's' in total_vox:
                    self.voxel_rec['s']=np.array(total_vox['s'])
                if 'f_in' in total_vox:
                    self.voxel_rec['f_in']=np.array(total_vox['f_in'])
                if 'v' in total_vox:
                    self.voxel_rec['v']=np.array(total_vox['v'])
                if 'q' in total_vox:
                    self.voxel_rec['q']=np.array(total_vox['q'])
                if 'y' in total_vox:
                    self.voxel_rec['y']=np.array(total_vox['y'])

            self.voxel_exc_rec={}
            if 'exc_syn' in f_vox:
                exc_vox=f_vox['exc_syn']
                if 'G_total' in exc_vox:
                    self.voxel_exc_rec['G_total']=np.array(exc_vox['G_total'])
                if 's' in exc_vox:
                    self.voxel_exc_rec['s']=np.array(exc_vox['s'])
                if 'f_in' in exc_vox:
                    self.voxel_exc_rec['f_in']=np.array(exc_vox['f_in'])
                if 'v' in exc_vox:
                    self.voxel_exc_rec['v']=np.array(exc_vox['v'])
                if 'q' in exc_vox:
                    self.voxel_exc_rec['q']=np.array(exc_vox['q'])
                if 'y' in exc_vox:
                    self.voxel_exc_rec['y']=np.array(exc_vox['y'])

        self.neural_state_rec=None
        if 'neuron_state' in f:
            f_state=f['neuron_state']
            self.neural_state_rec={'g_ampa_r': np.array(f_state['g_ampa_r']),
                                   'g_ampa_x': np.array(f_state['g_ampa_x']),
                                   'g_ampa_b': np.array(f_state['g_ampa_b']),
                                   'g_nmda':   np.array(f_state['g_nmda']),
                                   'g_gaba_a': np.array(f_state['g_gaba_a']),
                                   #'g_gaba_b': np.array(f_state['g_gaba_b']),
                                   'vm':       np.array(f_state['vm']),
                                   'record_idx': np.array(f_state['record_idx'])}

        self.e_firing_rates=None
        self.i_firing_rates=None
        self.rt=None
        self.choice=None
        if 'firing_rates' in f:
            f_rates=f['firing_rates']
            self.e_firing_rates=np.array(f_rates['e_rates'])
            self.i_firing_rates=np.array(f_rates['i_rates'])
            self.rt,self.choice=get_response_time(self.e_firing_rates, self.stim_start_time, self.stim_end_time,
                upper_threshold=upper_resp_threshold, lower_threshold=lower_resp_threshold, dt=dt)

        self.background_rate=None
        if 'background_rate' in f:
            b_rate=f['background_rate']
            self.background_rate=np.array(b_rate['firing_rate'])

        self.task_rates=None
        if 'task_rates' in f:
            t_rates=f['task_rates']
            self.task_rates=np.array(t_rates['firing_rates'])

        self.e_spike_neurons=None
        self.e_spike_times=None
        self.i_spike_neurons=None
        self.i_spike_times=None
        if 'spikes' in f:
            f_spikes=f['spikes']
            self.e_spike_neurons=[]
            self.e_spike_times=[]
            self.i_spike_neurons=[]
            self.i_spike_times=[]
            for idx in range(self.num_groups):
                if ('e.%d.spike_neurons' % idx) in f_spikes:
                    self.e_spike_neurons.append(np.array(f_spikes['e.%d.spike_neurons' % idx]))
                    self.e_spike_times.append(np.array(f_spikes['e.%d.spike_times' % idx]))
                if ('i.%d.spike_neurons' % idx) in f_spikes:
                    self.i_spike_neurons.append(np.array(f_spikes['i.%d.spike_neurons' % idx]))
                    self.i_spike_times.append(np.array(f_spikes['i.%d.spike_times' % idx]))
            if 'i.spike_neurons' in f_spikes:
                self.i_spike_neurons.append(np.array(f_spikes['i.spike_neurons']))
                self.i_spike_times.append(np.array(f_spikes['i.spike_times']))

        self.summary_data=Struct()
        if 'summary' in f:
            f_summary=f['summary']
            self.summary_data.e_mean=np.array(f_summary['e_mean'])
            self.summary_data.e_max=np.array(f_summary['e_max'])
            self.summary_data.i_mean=np.array(f_summary['i_mean'])
            self.summary_data.i_max=np.array(f_summary['i_max'])
            self.summary_data.bold_max=np.array(f_summary['bold_max'])
            self.summary_data.bold_exc_max=np.array(f_summary['bold_exc_max'])
        else:
            end_idx=int(self.stim_end_time/dt)
            start_idx=end_idx-1000
            e_mean_final=[]
            e_max=[]
            for i in range(self.e_firing_rates.shape[0]):
                e_rate=self.e_firing_rates[i,:]
                e_mean_final.append(np.mean(e_rate[start_idx:end_idx]))
                e_max.append(np.max(e_rate))
            i_mean_final=[]
            i_max=[]
            for i in range(self.i_firing_rates.shape[0]):
                i_rate=self.i_firing_rates[i,:]
                i_mean_final.append(np.mean(i_rate[start_idx:end_idx]))
                i_max.append(np.max(i_rate))
            self.summary_data.e_mean=np.array(e_mean_final)
            self.summary_data.e_max=np.array(e_max)
            self.summary_data.i_mean=np.array(i_mean_final)
            self.summary_data.i_max=np.array(i_max)
            if hasattr(self,'voxel_rec'):
                self.summary_data.bold_max=np.max(self.voxel_rec['y'])
            if hasattr(self,'voxel_exc_rec'):
                self.summary_data.bold_exc_max=np.max(self.voxel_exc_rec['y'])

        f.close()


def is_valid(high_contrast_e_rates, low_contrast_e_rates):
    rate_1=high_contrast_e_rates[0]
    rate_2=high_contrast_e_rates[1]
    rate_3=high_contrast_e_rates[2]
    high_contrast_max=max(rate_3)
    if max(rate_1)+20 > high_contrast_max < max(rate_2)+20:
        return False

    rate_1=low_contrast_e_rates[0]
    rate_2=low_contrast_e_rates[1]
    rate_3=low_contrast_e_rates[2]
    low_contrast_maxes=[max(rate_1), max(rate_2), max(rate_3)]
    maxIdx=-1
    maxRate=0
    for i,low_contrast_max in enumerate(low_contrast_maxes):
        if low_contrast_max>maxRate:
            maxRate=low_contrast_max
            maxIdx=i
    for i,low_contrast_max in enumerate(low_contrast_maxes):
        if not i==maxIdx and maxRate<low_contrast_max+20:
            return False

    return True


def run_posthoc_bayes_analysis(summary_file_name, perf_threshold, r_sqr_threshold):
    f=h5py.File(summary_file_name)
    num_groups=int(f.attrs['num_groups'])
    num_trials=int(f.attrs['num_trials'])
    trial_duration=float(f.attrs['trial_duration'])
    p_b_e_range=np.array(f['p_b_e_range'])
    p_x_e_range=np.array(f['p_x_e_range'])
    p_e_e_range=np.array(f['p_e_e_range'])
    p_e_i_range=np.array(f['p_e_i_range'])
    p_i_i_range=np.array(f['p_i_i_range'])
    p_i_e_range=np.array(f['p_i_e_range'])
    bc_slope=np.array(f['bold_contrast_slope'])
    bc_intercept=np.array(f['bold_contrast_intercept'])
    bc_r_sqr=np.array(f['bold_contrast_r_sqr'])
    auc=np.array(f['auc'])
    f.close()

    return run_bayesian_analysis(auc, bc_slope, bc_intercept, bc_r_sqr, num_trials, p_b_e_range, p_e_e_range, p_e_i_range,
        p_i_e_range, p_i_i_range, p_x_e_range, perf_threshold, r_sqr_threshold)


def run_bayesian_analysis(auc, slope, intercept, r_sqr, num_trials, p_b_e_range, p_e_e_range, p_e_i_range, p_i_e_range,
                          p_i_i_range, p_x_e_range, perf_threshold=0.75, r_sqr_threshold=0.2):
    bayes_analysis = Struct()
    # p(AUC | p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, M)
    # p(AUC | theta, M)
    bayes_analysis.l1_pos_likelihood = np.zeros(auc.shape)
    bayes_analysis.l1_neg_likelihood = np.zeros(auc.shape)
    bayes_analysis.l1_dist=[]
    bayes_analysis.l1_pos_dist=[]
    bayes_analysis.l1_neg_dist=[]
    for i, p_b_e in enumerate(p_b_e_range):
        for j, p_x_e in enumerate(p_x_e_range):
            for k, p_e_e in enumerate(p_e_e_range):
                for l, p_e_i in enumerate(p_e_i_range):
                    for m, p_i_i in enumerate(p_i_i_range):
                        for n, p_i_e in enumerate(p_i_e_range):
                            if not math.isnan(slope[i,j,k,l,m,n]):
                                bayes_analysis.l1_dist.append(slope[i,j,k,l,m,n])
                            if auc[i, j, k, l, m, n] >= perf_threshold:
                                if not math.isnan(slope[i,j,k,l,m,n]):
                                    bayes_analysis.l1_pos_dist.append(slope[i,j,k,l,m,n])
                                bayes_analysis.l1_pos_likelihood[i, j, k, l, m, n] = 1.0
                            else:
                                if not math.isnan(slope[i,j,k,l,m,n]):
                                    bayes_analysis.l1_neg_dist.append(slope[i,j,k,l,m,n])
                                bayes_analysis.l1_neg_likelihood[i, j, k, l, m, n] = 1.0

    # Number of parameter values tested
    n_param_vals = len(p_b_e_range) * len(p_x_e_range) * len(p_e_e_range) * len(p_e_i_range) * len(p_i_i_range) * len(
        p_i_e_range)
    # Priors are uniform
    # p(p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e | M)
    # p(theta | M)
    bayes_analysis.l1_pos_priors = np.ones([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range),
                                            len(p_i_i_range), len(p_i_e_range)]) * 1.0 / float(n_param_vals)
    bayes_analysis.l1_neg_priors = np.ones([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range),
                                            len(p_i_i_range), len(p_i_e_range)]) * 1.0 / float(n_param_vals)
    # p(AUC | M) = INT( p(AUC | theta, M)*p(theta | M) d theta
    bayes_analysis.l1_pos_evidence = np.sum(bayes_analysis.l1_pos_likelihood * bayes_analysis.l1_pos_priors)
    bayes_analysis.l1_neg_evidence = np.sum(bayes_analysis.l1_neg_likelihood * bayes_analysis.l1_neg_priors)
    # p(p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e | AUC, M)
    bayes_analysis.l1_pos_posterior = (bayes_analysis.l1_pos_likelihood * bayes_analysis.l1_pos_priors) / bayes_analysis.l1_pos_evidence
    bayes_analysis.l1_neg_posterior = (bayes_analysis.l1_neg_likelihood * bayes_analysis.l1_neg_priors) / bayes_analysis.l1_neg_evidence
    bayes_analysis.l1_pos_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_pos_priors, bayes_analysis.l1_pos_likelihood,
        bayes_analysis.l1_pos_posterior, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range)
    bayes_analysis.l1_neg_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_neg_priors, bayes_analysis.l1_neg_likelihood,
        bayes_analysis.l1_neg_posterior, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range)

    bayes_analysis.l1_pos_l2_priors = bayes_analysis.l1_pos_posterior
    bayes_analysis.l1_neg_l2_priors = bayes_analysis.l1_neg_posterior

    bayes_analysis.l1_pos_l2_neg_likelihood = np.zeros(bayes_analysis.l1_pos_likelihood.shape)
    bayes_analysis.l1_pos_l2_pos_likelihood = np.zeros(bayes_analysis.l1_pos_likelihood.shape)
    bayes_analysis.l1_pos_l2_zero_likelihood = np.zeros(bayes_analysis.l1_pos_likelihood.shape)
    bayes_analysis.l1_neg_l2_neg_likelihood = np.zeros(bayes_analysis.l1_neg_likelihood.shape)
    bayes_analysis.l1_neg_l2_pos_likelihood = np.zeros(bayes_analysis.l1_neg_likelihood.shape)
    bayes_analysis.l1_neg_l2_zero_likelihood = np.zeros(bayes_analysis.l1_neg_likelihood.shape)

    for i, p_b_e in enumerate(p_b_e_range):
        for j, p_x_e in enumerate(p_x_e_range):
            for k, p_e_e in enumerate(p_e_e_range):
                for l, p_e_i in enumerate(p_e_i_range):
                    for m, p_i_i in enumerate(p_i_i_range):
                        for n, p_i_e in enumerate(p_i_e_range):
                            if r_sqr[i,j,k,l,m,n] > r_sqr_threshold:
                                if slope[i,j,k,l,m,n] > 0.0:
                                    bayes_analysis.l1_pos_l2_pos_likelihood[i, j, k, l, m, n] = 1.0
                                    bayes_analysis.l1_neg_l2_pos_likelihood[i, j, k, l, m, n] = 1.0
                                elif slope[i,j,k,l,m,n] < 0.0:
                                    bayes_analysis.l1_pos_l2_neg_likelihood[i, j, k, l, m, n] = 1.0
                                    bayes_analysis.l1_neg_l2_neg_likelihood[i, j, k, l, m, n] = 1.0
                                else:
                                    bayes_analysis.l1_pos_l2_zero_likelihood[i, j, k, l, m, n] = 1.0
                                    bayes_analysis.l1_neg_l2_zero_likelihood[i, j, k, l, m, n] = 1.0
                            else:
                                bayes_analysis.l1_pos_l2_zero_likelihood[i, j, k, l, m, n] = 1.0
                                bayes_analysis.l1_neg_l2_zero_likelihood[i, j, k, l, m, n] = 1.0

    bayes_analysis.l1_pos_l2_pos_evidence = np.sum(bayes_analysis.l1_pos_l2_pos_likelihood * bayes_analysis.l1_pos_l2_priors)
    bayes_analysis.l1_pos_l2_neg_evidence = np.sum(bayes_analysis.l1_pos_l2_neg_likelihood * bayes_analysis.l1_pos_l2_priors)
    bayes_analysis.l1_pos_l2_zero_evidence = np.sum(bayes_analysis.l1_pos_l2_zero_likelihood * bayes_analysis.l1_pos_l2_priors)
    bayes_analysis.l1_neg_l2_pos_evidence = np.sum(bayes_analysis.l1_neg_l2_pos_likelihood * bayes_analysis.l1_neg_l2_priors)
    bayes_analysis.l1_neg_l2_neg_evidence = np.sum(bayes_analysis.l1_neg_l2_neg_likelihood * bayes_analysis.l1_neg_l2_priors)
    bayes_analysis.l1_neg_l2_zero_evidence = np.sum(bayes_analysis.l1_neg_l2_zero_likelihood * bayes_analysis.l1_neg_l2_priors)

    bayes_analysis.l1_pos_l2_pos_posterior = np.zeros(bayes_analysis.l1_pos_l2_pos_likelihood.shape)
    if bayes_analysis.l1_pos_l2_pos_evidence>0:
        bayes_analysis.l1_pos_l2_pos_posterior = (bayes_analysis.l1_pos_l2_pos_likelihood * bayes_analysis.l1_pos_l2_priors) / bayes_analysis.l1_pos_l2_pos_evidence
    bayes_analysis.l1_pos_l2_neg_posterior = np.zeros(bayes_analysis.l1_pos_l2_neg_likelihood.shape)
    if bayes_analysis.l1_pos_l2_neg_evidence>0:
        bayes_analysis.l1_pos_l2_neg_posterior = (bayes_analysis.l1_pos_l2_neg_likelihood * bayes_analysis.l1_pos_l2_priors) / bayes_analysis.l1_pos_l2_neg_evidence
    bayes_analysis.l1_pos_l2_zero_posterior = np.zeros(bayes_analysis.l1_pos_l2_zero_likelihood.shape)
    if bayes_analysis.l1_pos_l2_zero_evidence>0:
        bayes_analysis.l1_pos_l2_zero_posterior = (bayes_analysis.l1_pos_l2_zero_likelihood * bayes_analysis.l1_pos_l2_priors) / bayes_analysis.l1_pos_l2_zero_evidence
    bayes_analysis.l1_neg_l2_pos_posterior = np.zeros(bayes_analysis.l1_neg_l2_pos_likelihood.shape)
    if bayes_analysis.l1_neg_l2_pos_evidence>0:
        bayes_analysis.l1_neg_l2_pos_posterior = (bayes_analysis.l1_neg_l2_pos_likelihood * bayes_analysis.l1_neg_l2_priors) / bayes_analysis.l1_neg_l2_pos_evidence
    bayes_analysis.l1_neg_l2_neg_posterior = np.zeros(bayes_analysis.l1_neg_l2_neg_likelihood.shape)
    if bayes_analysis.l1_neg_l2_neg_evidence>0:
        bayes_analysis.l1_neg_l2_neg_posterior = (bayes_analysis.l1_neg_l2_neg_likelihood * bayes_analysis.l1_neg_l2_priors) / bayes_analysis.l1_neg_l2_neg_evidence
    bayes_analysis.l1_neg_l2_zero_posterior = np.zeros(bayes_analysis.l1_neg_l2_zero_likelihood.shape)
    if bayes_analysis.l1_neg_l2_zero_evidence>0:
        bayes_analysis.l1_neg_l2_zero_posterior = (bayes_analysis.l1_neg_l2_zero_likelihood * bayes_analysis.l1_neg_l2_priors) / bayes_analysis.l1_neg_l2_zero_evidence

    bayes_analysis.l1_pos_l2_neg_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_pos_l2_priors,
        bayes_analysis.l1_pos_l2_neg_likelihood, bayes_analysis.l1_pos_l2_neg_posterior, p_b_e_range, p_x_e_range, p_e_e_range,
        p_e_i_range, p_i_i_range, p_i_e_range)
    bayes_analysis.l1_pos_l2_pos_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_pos_l2_priors,
        bayes_analysis.l1_pos_l2_pos_likelihood, bayes_analysis.l1_pos_l2_pos_posterior, p_b_e_range, p_x_e_range, p_e_e_range,
        p_e_i_range, p_i_i_range, p_i_e_range)
    bayes_analysis.l1_pos_l2_zero_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_pos_l2_priors,
        bayes_analysis.l1_pos_l2_zero_likelihood, bayes_analysis.l1_pos_l2_zero_posterior, p_b_e_range, p_x_e_range, p_e_e_range,
        p_e_i_range, p_i_i_range, p_i_e_range)
    bayes_analysis.l1_neg_l2_neg_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_neg_l2_priors,
        bayes_analysis.l1_neg_l2_neg_likelihood, bayes_analysis.l1_neg_l2_neg_posterior, p_b_e_range, p_x_e_range, p_e_e_range,
        p_e_i_range, p_i_i_range, p_i_e_range)
    bayes_analysis.l1_neg_l2_pos_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_neg_l2_priors,
        bayes_analysis.l1_neg_l2_pos_likelihood, bayes_analysis.l1_neg_l2_pos_posterior, p_b_e_range, p_x_e_range, p_e_e_range,
        p_e_i_range, p_i_i_range, p_i_e_range)
    bayes_analysis.l1_neg_l2_zero_marginals = run_bayesian_marginal_analysis(bayes_analysis.l1_neg_l2_priors,
        bayes_analysis.l1_neg_l2_zero_likelihood, bayes_analysis.l1_neg_l2_zero_posterior, p_b_e_range, p_x_e_range, p_e_e_range,
        p_e_i_range, p_i_i_range, p_i_e_range)
    return bayes_analysis


def run_bayesian_marginal_analysis(priors, likelihood, posterior, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range,
                                   p_i_i_range, p_i_e_range):
    marginal_analysis=Struct()

    marginal_analysis.posterior_p_b_e=np.zeros([len(p_b_e_range)])
    marginal_analysis.posterior_p_x_e=np.zeros([len(p_x_e_range)])
    marginal_analysis.posterior_p_e_e=np.zeros([len(p_e_e_range)])
    marginal_analysis.posterior_p_e_i=np.zeros([len(p_e_i_range)])
    marginal_analysis.posterior_p_i_i=np.zeros([len(p_i_i_range)])
    marginal_analysis.posterior_p_i_e=np.zeros([len(p_i_e_range)])
    marginal_analysis.prior_p_b_e=np.zeros([len(p_b_e_range)])
    marginal_analysis.prior_p_x_e=np.zeros([len(p_x_e_range)])
    marginal_analysis.prior_p_e_e=np.zeros([len(p_e_e_range)])
    marginal_analysis.prior_p_e_i=np.zeros([len(p_e_i_range)])
    marginal_analysis.prior_p_i_i=np.zeros([len(p_i_i_range)])
    marginal_analysis.prior_p_i_e=np.zeros([len(p_i_e_range)])
    marginal_analysis.likelihood_p_b_e=np.zeros([len(p_b_e_range)])
    marginal_analysis.likelihood_p_x_e=np.zeros([len(p_x_e_range)])
    marginal_analysis.likelihood_p_e_e=np.zeros([len(p_e_e_range)])
    marginal_analysis.likelihood_p_e_i=np.zeros([len(p_e_i_range)])
    marginal_analysis.likelihood_p_i_i=np.zeros([len(p_i_i_range)])
    marginal_analysis.likelihood_p_i_e=np.zeros([len(p_i_e_range)])
    for j,p_x_e in enumerate(p_x_e_range):
        for k,p_e_e in enumerate(p_e_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        marginal_analysis.posterior_p_b_e+=posterior[:,j,k,l,m,n]
                        marginal_analysis.prior_p_b_e+=priors[:,j,k,l,m,n]
                        marginal_analysis.likelihood_p_b_e+=likelihood[:,j,k,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for k,p_e_e in enumerate(p_e_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        marginal_analysis.posterior_p_x_e+=posterior[i,:,k,l,m,n]
                        marginal_analysis.prior_p_x_e+=priors[i,:,k,l,m,n]
                        marginal_analysis.likelihood_p_x_e+=likelihood[i,:,k,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        marginal_analysis.posterior_p_e_e+=posterior[i,j,:,l,m,n]
                        marginal_analysis.prior_p_e_e+=priors[i,j,:,l,m,n]
                        marginal_analysis.likelihood_p_e_e+=likelihood[i,j,:,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for m,p_i_i in enumerate(p_i_i_range):
                        marginal_analysis.posterior_p_i_e+=posterior[i,j,k,l,m,:]
                        marginal_analysis.prior_p_i_e+=priors[i,j,k,l,m,:]
                        marginal_analysis.likelihood_p_i_e+=likelihood[i,j,k,l,m,:]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        marginal_analysis.posterior_p_i_i+=posterior[i,j,k,l,:,n]
                        marginal_analysis.prior_p_i_i+=priors[i,j,k,l,:,n]
                        marginal_analysis.likelihood_p_i_i+=likelihood[i,j,k,l,:,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        marginal_analysis.posterior_p_e_i+=posterior[i,j,k,:,m,n]
                        marginal_analysis.prior_p_e_i+=priors[i,j,k,:,m,n]
                        marginal_analysis.likelihood_p_e_i+=likelihood[i,j,k,:,m,n]

    marginal_analysis.posterior_p_b_e_p_x_e=np.zeros([len(p_b_e_range), len(p_x_e_range)])
    marginal_analysis.posterior_p_e_e_p_e_i=np.zeros([len(p_e_e_range), len(p_e_i_range)])
    marginal_analysis.posterior_p_e_e_p_i_i=np.zeros([len(p_e_e_range), len(p_i_i_range)])
    marginal_analysis.posterior_p_e_e_p_i_e=np.zeros([len(p_e_e_range), len(p_i_e_range)])
    marginal_analysis.posterior_p_e_i_p_i_i=np.zeros([len(p_e_i_range), len(p_i_i_range)])
    marginal_analysis.posterior_p_e_i_p_i_e=np.zeros([len(p_e_i_range), len(p_i_e_range)])
    marginal_analysis.posterior_p_i_i_p_i_e=np.zeros([len(p_i_i_range), len(p_i_e_range)])
    for k,p_e_e in enumerate(p_e_e_range):
        for l,p_e_i in enumerate(p_e_i_range):
            for m,p_i_i in enumerate(p_i_i_range):
                for n,p_i_e in enumerate(p_i_e_range):
                    marginal_analysis.posterior_p_b_e_p_x_e+=posterior[:,:,k,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for m,p_i_i in enumerate(p_i_i_range):
                for n,p_i_e in enumerate(p_i_e_range):
                    marginal_analysis.posterior_p_e_e_p_e_i+=posterior[i,j,:,:,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for n,p_i_e in enumerate(p_i_e_range):
                    marginal_analysis.posterior_p_e_e_p_i_i+=posterior[i,j,:,l,:,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    marginal_analysis.posterior_p_e_e_p_i_e+=posterior[i,j,:,l,m,:]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for n,p_i_e in enumerate(p_i_e_range):
                    marginal_analysis.posterior_p_e_i_p_i_i+=posterior[i,j,k,:,:,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    marginal_analysis.posterior_p_e_i_p_i_e+=posterior[i,j,k,:,m,:]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    marginal_analysis.posterior_p_i_i_p_i_e+=posterior[i,j,k,l,:,:]

    return marginal_analysis


def get_lfp_signal(wta_data, plot=False):
    [b,a]=butter(4.0,1.0/500.0,'high')
    dur=len(wta_data.lfp_rec['lfp'][0])
    resam_dur=int(dur*.1)
    lfp_res=scipy.signal.resample(wta_data.lfp_rec['lfp'][0],resam_dur)
    lfp=lfilter(b,a,lfp_res)
    if plot:
        plt.plot(lfp)
        plt.show()
    return lfp

def get_roc_init(contrast_range, num_trials, num_extra_trials, option_idx, prefix, dt=.1*ms):
    l = []
    p = 0
    n = 0
    for contrast in contrast_range:
        for trial in range(num_trials):
            data_path = '%s.contrast.%0.4f.trial.%d.h5' % (prefix, contrast, trial)
            print(data_path)
            data = FileInfo(data_path)
            stim_end_idx=(data.stim_end_time*dt)/second
            example = 0
            if data.input_freq[option_idx] > data.input_freq[1 - option_idx]:
                example = 1
                for j in range(num_extra_trials):
                    p += 1
            elif data.input_freq[option_idx] < data.input_freq[1-option_idx]:
                for j in range(num_extra_trials):
                    n += 1
            else:
                if np.max(data.e_firing_rates[option_idx, stim_end_idx-1000:stim_end_idx])>\
                   np.max(data.e_firing_rates[1 - option_idx, stim_end_idx-1000:stim_end_idx]):
                    example=1
                    for j in range(num_extra_trials):
                        p += 1
                else:
                    for j in range(num_extra_trials):
                        n += 1

            # Get mean rate of pop 1 for last 100ms
            pop_mean = np.mean(data.e_firing_rates[option_idx, stim_end_idx-1000:stim_end_idx])
            other_pop_mean = np.mean(data.e_firing_rates[1 - option_idx, stim_end_idx-1000:stim_end_idx])
            max_rate=max([np.max(data.e_firing_rates[option_idx, stim_end_idx-1000:stim_end_idx]),
                          np.max(data.e_firing_rates[1 - option_idx, stim_end_idx-1000:stim_end_idx])])
            pop_mean+=.25*max_rate*np.random.randn()
            other_pop_mean+=.25*max_rate*np.random.randn()
            for j in range(num_extra_trials):
                f_score=0
                if float(pop_mean+other_pop_mean):
                    f_score+=float(pop_mean)/float(pop_mean+other_pop_mean)
                l.append((example, f_score))
    l_sorted = sorted(l, key=lambda example: example[1], reverse=True)
    return l_sorted, n, p

def get_auc(prefix, contrast_range, num_trials, num_extra_trials, num_groups, dt=.1*ms):
    total_auc=0
    total_p=0
    single_auc=[]
    single_p=[]
    for i in range(num_groups):
        l_sorted, n, p = get_roc_init(contrast_range, num_trials, num_extra_trials, i, prefix, dt=dt)
        single_auc.append(get_auc_single_option(prefix, contrast_range, num_trials, num_extra_trials, i))
        single_p.append(p)
        total_p+=p
    for i in range(num_groups):
        total_auc+=float(single_auc[i])*(float(single_p[i])/float(total_p))

    return total_auc

def get_auc_single_option(prefix, contrast_range, num_trials, num_extra_trials, option_idx, dt=.1*ms):
    l_sorted, n, p = get_roc_init(contrast_range, num_trials, num_extra_trials, option_idx, prefix, dt=dt)
    fp=0
    tp=0
    fp_prev=0
    tp_prev=0
    a=0
    f_prev=float('-inf')
    for (example_i,f_i) in l_sorted:
        if not f_i==f_prev:
            a+=trapezoid_area(float(fp),float(fp_prev),float(tp),float(tp_prev))
            f_prev=f_i
            fp_prev=fp
            tp_prev=tp
        if example_i>0:
            tp+=1
        else:
            fp+=1
    a+=trapezoid_area(float(fp),float(fp_prev),float(tp),float(tp_prev))
    a=float(a)/(float(max(p,.001))*float(max(n,.001)))
    return a

def get_roc_single_option(prefix, contrast_range, num_trials, num_extra_trials, option_idx, dt=.1*ms):

    l_sorted, n, p = get_roc_init(contrast_range, num_trials, num_extra_trials, option_idx, prefix, dt=dt)

    fp=0
    tp=0
    roc=[]
    f_prev=float('-inf')
    for (example_i,f_i) in l_sorted:
        if not f_i==f_prev:
            roc.append([float(fp)/float(n),float(tp)/float(p)])
            f_prev=f_i
        if example_i>0:
            tp+=1
        else:
            fp+=1
    roc.append([float(fp)/float(n),float(tp)/float(p)])
    roc=np.array(roc)
    return roc

def trapezoid_area(x1,x2,y1,y2):
    base=float(abs(x1-x2))
    height_avg=float(y1+y2)/2.0
    return base*height_avg

def get_roc(prefix, contrast_range, num_trials, num_extra_trials):
    roc1=get_roc_single_option(prefix, contrast_range, num_trials, num_extra_trials, 0)
    roc2=get_roc_single_option(prefix, contrast_range, num_trials, num_extra_trials, 1)

    plt.figure()
    plt.plot(roc1[:,0],roc1[:,1],'x-',label='option 1')
    plt.plot(roc2[:,0],roc2[:,1],'x-',label='option 2')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def test_roc(wta_params, prefix, num_trials):
    inputs=[]
    input_sum=40.0
    for i in range(num_trials):
        input=np.zeros(2)
        input[0]=np.random.rand()*input_sum
        input[1]=input_sum-input[0]
        inputs.append(input)

    for i,input in enumerate(inputs):
        run_wta(wta_params, 2, input*Hz, 1*second, record_lfp=True, record_voxel=True,
            record_neuron_state=True, record_spikes=True, record_firing_rate=True, plot_output=False,
            output_file=os.path.join(DATA_DIR,'%s.trial.%d.h5' % (prefix, i)))


def plot_regression(x, y, a, b, x_max, x_min, dot_style, line_style, label_str):
    plt.plot(x, y, dot_style)
    plt.plot([x_min, x_max], [a * x_min + b, a * x_max + b], line_style, label=label_str)

def compute_auc(l_sorted, n, p):
    fp=0
    tp=0
    fp_prev=0
    tp_prev=0
    a=0
    f_prev=float('-inf')
    for (example_i,f_i) in l_sorted:
        if not f_i==f_prev:
            a+=trapezoid_area(float(fp),float(fp_prev),float(tp),float(tp_prev))
            f_prev=f_i
            fp_prev=fp
            tp_prev=tp
        if example_i>0:
            tp+=1.0
        else:
            fp+=1.0
    a+=trapezoid_area(float(fp),float(fp_prev),float(tp),float(tp_prev))
    a=float(a)/(float(max(p,.001))*float(max(n,.001)))
    return a

class TrialSummary:
    """
    Represents summary of data from a single trial
    """

    def __init__(self, contrast, trial_idx, data, dt):
        """
        Initialize summary data from full data object
        contrast = input contrast
        trial_idx = trial index
        data = full data obejct
        """
        self.data=data
        self.contrast=contrast
        self.trial_idx=trial_idx

        endIdx=int(data.stim_end_time/dt)
        startIdx=endIdx-500
        e_mean=np.mean(data.e_firing_rates[:,startIdx:endIdx],axis=1)

        # Index of max responding population
        self.decision_idx=0
        if e_mean[1]>e_mean[0]:
            self.decision_idx=1
        # Index of maximum input
        self.max_in_idx=0
        if data.input_freq[1] > data.input_freq[0]:
            self.max_in_idx=1
        # Whether or not selection is correct
        self.correct=False
        if data.rt is not None and (data.input_freq[1]==data.input_freq[0] or self.decision_idx==self.max_in_idx):
            self.correct=True
        self.max_rate=np.max(data.summary_data.e_max)

    def get_f_score(self, idx, alpha=2.0):
        """
        Get F-score for this trial
        """
        if np.sum(self.data.summary_data.e_mean):
            e_mean_rand=self.data.summary_data.e_mean+alpha*np.random.randn(len(self.data.summary_data.e_mean))*self.data.summary_data.e_mean
            return float(e_mean_rand[idx])/float(np.sum(e_mean_rand))
            #return float(self.data.summary_data.e_mean[idx])/float(np.sum(self.data.summary_data.e_mean))
        return 0.0


class MaxBOLDContrastRegression:
    def __init__(self, trials_max_bold, trials_contrast):
        self.trials_max_bold=np.array(trials_max_bold)
        self.trials_max_bold_contrast=np.array(trials_contrast)
        self.trials_max_bold_contrast=np.reshape(self.trials_max_bold_contrast,(-1,1))

        self.bold_contrast_a=float('NaN')
        self.bold_contrast_b=0
        if len(self.trials_max_bold_contrast):
            clf = LinearRegression()
            clf.fit(self.trials_max_bold_contrast, self.trials_max_bold)
            self.bold_contrast_a = clf.coef_[0]
            self.bold_contrast_b = clf.intercept_
            self.bold_contrast_r_sqr=clf.score(self.trials_max_bold_contrast, self.trials_max_bold)

    def plot(self, x_max, x_min, dot_style, line_style, label_str):
        plt.plot(self.trials_max_bold_contrast, self.trials_max_bold, dot_style)
        plt.plot([x_min, x_max], [self.bold_contrast_a * x_min + self.bold_contrast_b,
                                  self.bold_contrast_a * x_max + self.bold_contrast_b], line_style, label=label_str)






class TrialSeries:
    """
    Represents a series of trials with varying input contrast levels
    """

    def __init__(self, dir, prefix, num_trials, contrast_range=(0.0, 0.0625, 0.125, 0.25, 0.5, 1.0),
                 upper_resp_threshold=30, lower_resp_threshold=None, dt=.1*ms):
        """
        Load trial data files
        dir = directory to load files from
        prefix = file prefix
        num_trials = number of trials per contrast level
        contrast_range = range of contrast values tested
        """
        self.contrast_range=contrast_range
        self.num_trials=num_trials

        self.trial_summaries=[]

        # Load each trial data file and store TrialSummary object
        for i,contrast in enumerate(self.contrast_range):
            print('loading contrast %.4f' % contrast)
            for trial_idx in range(num_trials):

                file_name=os.path.join(dir,'%s.contrast.%0.4f.trial.%d.h5' % (prefix, contrast, trial_idx))
                trial_summary=None
                if not os.path.exists(file_name):
                    print('file does not exist: %s' % file_name)
                else:
                    try:
                        trial_summary=TrialSummary(contrast, trial_idx,
                            FileInfo(file_name,upper_resp_threshold=upper_resp_threshold,
                                lower_resp_threshold=lower_resp_threshold,  dt=dt), dt)
                    except:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_exception(exc_type, exc_value, exc_traceback,
                            limit=2, file=sys.stdout)
                        print('cannot load file %s' % file_name)

                if not trial_summary is None:
                    self.trial_summaries.append(trial_summary)

        self.compute_muticlass_auc(dt)
        self.compute_bold_contrast_regression()


    def compute_muticlass_auc(self, dt, n_options=2):
        """
        Compute the multi-option AUC for this series
        """
        self.response_option_aucs=[]
        for i in range(n_options):
            auc=Struct()
            auc.option_idx=i
            auc.l = []
            auc.p = 0
            auc.n = 0
            self.response_option_aucs.append(auc)

        # Get F-scores for each class for each trial
        for trial_summary in self.trial_summaries:
            if trial_summary.contrast>0.0:
                for i in range(20):
                    self.response_option_aucs[trial_summary.max_in_idx].p+=1.0
                    self.response_option_aucs[1-trial_summary.max_in_idx].n+=1.0

                    self.response_option_aucs[0].l.append((1-trial_summary.max_in_idx, trial_summary.get_f_score(0)))
                    self.response_option_aucs[1].l.append((trial_summary.max_in_idx, trial_summary.get_f_score(1)))

        # Compute AUC of each response option
        for auc in self.response_option_aucs:
            auc.l_sorted = sorted(auc.l, key=lambda example: example[1], reverse=True)
            auc.auc=compute_auc(auc.l_sorted,auc.n,auc.p)

        # Compute weighted total AUC
        self.total_auc=0.0
        sub_total_auc=0
        for auc in self.response_option_aucs:
            sub_total_auc+=auc.p
        for auc in self.response_option_aucs:
            self.total_auc+=auc.auc*(auc.p/sub_total_auc)

    def plot_multiclass_roc(self, filename=None):
        fig=plt.figure()

        for idx,auc in enumerate(self.response_option_aucs):
            fp=0
            tp=0
            auc.roc=[]
            f_prev=float('-inf')
            for (example_i,f_i) in auc.l_sorted:
                if not f_i==f_prev:
                    auc.roc.append([float(fp)/float(auc.n),float(tp)/float(auc.p)])
                    f_prev=f_i
                if example_i>0:
                    tp+=1
                else:
                    fp+=1
            auc.roc.append([float(fp)/float(auc.n),float(tp)/float(auc.p)])
            auc.roc=np.array(auc.roc)

            plt.plot(auc.roc[:,0],auc.roc[:,1],'x-',label='option %d' % idx)

        plt.plot([0,1],[0,1],'--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        if filename is None:
            plt.show()
        else:
            save_to_png(fig, '%s.png' % filename)
            save_to_eps(fig, '%s.eps' % filename)
            plt.close(fig)

    def compute_bold_contrast_regression(self):
        """
        Perform regression - max bold by input contrast
        """
        trials_max_bold=[]
        trials_max_bold_contrast=[]
        trials_max_exc_bold=[]
        trials_max_exc_bold_contrast=[]
        for trial_summary in self.trial_summaries:
            if not math.isnan(trial_summary.data.summary_data.bold_max):
                trials_max_bold.append(trial_summary.data.summary_data.bold_max)
                trials_max_bold_contrast.append(trial_summary.contrast)
            if not math.isnan(trial_summary.data.summary_data.bold_exc_max):
                trials_max_exc_bold.append(trial_summary.data.summary_data.bold_exc_max)
                trials_max_exc_bold_contrast.append(trial_summary.contrast)

        self.max_bold_regression=MaxBOLDContrastRegression(trials_max_bold, trials_max_bold_contrast)
        self.max_exc_bold_regression=MaxBOLDContrastRegression(trials_max_exc_bold, trials_max_exc_bold_contrast)

    def get_contrast_perc_correct_stats(self):
        contrast_correct = {}
        for trial_summary in self.trial_summaries:
            if trial_summary.contrast > 0:
                if not trial_summary.contrast in contrast_correct:
                    contrast_correct[trial_summary.contrast] = []
                if trial_summary.correct:
                    contrast_correct[trial_summary.contrast].append(1.0)
                else:
                    contrast_correct[trial_summary.contrast].append(0.0)
        contrast = []
        perc_correct = []
        for c in sorted(contrast_correct.keys()):
            correct = contrast_correct[c]
            contrast.append(c)
            perc_correct.append(np.sum(correct) / len(correct))
        contrast = np.array(contrast)
        #contrast.reshape([1,len(contrast)])
        perc_correct = np.array(perc_correct)
        #perc_correct.reshape([1,len(perc_correct)])
        return contrast, perc_correct

    def plot_perc_correct(self, filename=None):
        contrast, perc_correct = self.get_contrast_perc_correct_stats()

        fig=plt.figure()
        acc_fit=FitWeibull(contrast, perc_correct, guess=[0.2, 0.5])
        thresh = np.max([0,acc_fit.inverse(0.8)])
        smoothInt = pylab.arange(0.0, max(contrast), 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        plt.plot(smoothInt, smoothResp,'k')
        plt.plot(contrast, perc_correct, 'ok')
        plt.plot([thresh,thresh],[0.4,1.0],'k')
        #plt.ylim([0.4,1])
        plt.xlabel('Contrast')
        plt.ylabel('% correct')
        plt.xscale('log')
        if filename is None:
            plt.show()
        else:
            save_to_png(fig, '%s.png' % filename)
            save_to_eps(fig, '%s.eps' % filename)
            plt.close(fig)

    def get_contrast_rt_stats(self):
        contrast_rt = {}
        for trial_summary in self.trial_summaries:
            if not trial_summary.data.rt is None:
                if not trial_summary.contrast in contrast_rt:
                    contrast_rt[trial_summary.contrast] = []
                contrast_rt[trial_summary.contrast].append(trial_summary.data.rt)
        contrast = []
        mean_rt = []
        std_rt = []
        for c in sorted(contrast_rt.keys()):
            rts = contrast_rt[c]
            contrast.append(c)
            mean_rt.append(np.mean(rts))
            std_rt.append(np.std(rts)/len(rts))
        return contrast, mean_rt, std_rt

    def plot_rt(self, filename=None):
        contrast, mean_rt, std_rt = self.get_contrast_rt_stats()

        fig=plt.figure()
        if len(mean_rt):
            rt_fit = FitRT(np.array(contrast), mean_rt, guess=[1,1,1])
            smoothInt = pylab.arange(0.01, max(contrast), 0.001)
            smoothResp = rt_fit.eval(smoothInt)
            plt.errorbar(contrast, mean_rt,yerr=std_rt,fmt='ok')
            plt.plot(smoothInt, smoothResp, 'k')
        plt.xlabel('Contrast')
        plt.ylabel('Decision time (ms)')
        plt.xscale('log')
        if filename is None:
            plt.show()
        else:
            save_to_png(fig, '%s.png' % filename)
            save_to_eps(fig, '%s.eps' % filename)
            plt.close(fig)

    def sort_by_correct(self):
        """
        Sorts correct/incorrect trials and separately performs regression - max bold by input contrast
        """
        trials_max_bold_correct=[]
        trials_max_bold_correct_contrast=[]
        trials_max_bold_incorrect=[]
        trials_max_bold_incorrect_contrast=[]
        trials_max_exc_bold_correct=[]
        trials_max_exc_bold_correct_contrast=[]
        trials_max_exc_bold_incorrect=[]
        trials_max_exc_bold_incorrect_contrast=[]

        for trial_summary in self.trial_summaries:
            if trial_summary.correct:
                if not math.isnan(trial_summary.data.summary_data.bold_max):
                    trials_max_bold_correct.append(trial_summary.data.summary_data.bold_max)
                    trials_max_bold_correct_contrast.append(trial_summary.contrast)
                if not math.isnan(trial_summary.data.summary_data.bold_exc_max):
                    trials_max_exc_bold_correct.append(trial_summary.data.summary_data.bold_exc_max)
                    trials_max_exc_bold_correct_contrast.append(trial_summary.contrast)
            else:
                if not math.isnan(trial_summary.data.summary_data.bold_max):
                    trials_max_bold_incorrect.append(trial_summary.data.summary_data.bold_max)
                    trials_max_bold_incorrect_contrast.append(trial_summary.contrast)
                if not math.isnan(trial_summary.data.summary_data.bold_exc_max):
                    trials_max_exc_bold_incorrect.append(trial_summary.data.summary_data.bold_exc_max)
                    trials_max_exc_bold_incorrect_contrast.append(trial_summary.contrast)

        self.correct_max_bold_regression=MaxBOLDContrastRegression(trials_max_bold_correct, trials_max_bold_correct_contrast)
        self.incorrect_max_bold_regression=MaxBOLDContrastRegression(trials_max_bold_incorrect, trials_max_bold_incorrect_contrast)
        self.correct_max_exc_bold_regression=MaxBOLDContrastRegression(trials_max_exc_bold_correct, trials_max_exc_bold_correct_contrast)
        self.incorrect_max_exc_bold_regression=MaxBOLDContrastRegression(trials_max_exc_bold_incorrect, trials_max_exc_bold_incorrect_contrast)


    def sort_by_correct_lesioned(self):
        """
        Sorts affect/intact correct/incorrect trials and separately performs regression - max bold by input contrast
        """
        trials_max_bold_affected_correct=[]
        trials_max_bold_affected_incorrect=[]
        trials_max_bold_intact_correct=[]
        trials_max_bold_intact_incorrect=[]
        trials_max_exc_bold_affected_correct=[]
        trials_max_exc_bold_affected_incorrect=[]
        trials_max_exc_bold_intact_correct=[]
        trials_max_exc_bold_intact_incorrect=[]

        trials_max_bold_affected_correct_contrast=[]
        trials_max_bold_affected_incorrect_contrast=[]
        trials_max_bold_intact_correct_contrast=[]
        trials_max_bold_intact_incorrect_contrast=[]
        trials_max_exc_bold_affected_correct_contrast=[]
        trials_max_exc_bold_affected_incorrect_contrast=[]
        trials_max_exc_bold_intact_correct_contrast=[]
        trials_max_exc_bold_intact_incorrect_contrast=[]

        for trial_summary in self.trial_summaries:
            if trial_summary.decision_idx==trial_summary.injection_site and trial_summary.muscimol_amount>0:
                if trial_summary.correct:
                    if not math.isnan(trial_summary.bold_max):
                        trials_max_bold_affected_correct.append(trial_summary.bold_max)
                        trials_max_bold_affected_correct_contrast.append(trial_summary.contrast)
                    if not math.isnan(trial_summary.bold_exc_max):
                        trials_max_exc_bold_affected_correct.append(trial_summary.bold_exc_max)
                        trials_max_exc_bold_affected_correct_contrast.append(trial_summary.contrast)

                else:
                    if not math.isnan(trial_summary.bold_max):
                        trials_max_bold_affected_incorrect.append(trial_summary.bold_max)
                        trials_max_bold_affected_incorrect_contrast.append(trial_summary.contrast)
                    if not math.isnan(trial_summary.bold_exc_max):
                        trials_max_exc_bold_affected_incorrect.append(trial_summary.bold_exc_max)
                        trials_max_exc_bold_affected_incorrect_contrast.append(trial_summary.contrast)

            else:
                if trial_summary.correct:
                    if not math.isnan(trial_summary.bold_max):
                        trials_max_bold_intact_correct.append(trial_summary.bold_max)
                        trials_max_bold_intact_correct_contrast.append(trial_summary.contrast)
                    if not math.isnan(trial_summary.bold_exc_max):
                        trials_max_exc_bold_intact_correct.append(trial_summary.bold_exc_max)
                        trials_max_exc_bold_intact_correct_contrast.append(trial_summary.contrast)
                else:
                    if not math.isnan(trial_summary.bold_max):
                        trials_max_bold_intact_incorrect.append(trial_summary.bold_max)
                        trials_max_bold_intact_incorrect_contrast.append(trial_summary.contrast)
                    if not math.isnan(trial_summary.bold_exc_max):
                        trials_max_exc_bold_intact_incorrect.append(trial_summary.bold_exc_max)
                        trials_max_exc_bold_intact_incorrect_contrast.append(trial_summary.contrast)

        self.affected_correct_max_bold_regression=MaxBOLDContrastRegression(trials_max_bold_affected_correct,
            trials_max_bold_affected_correct_contrast)
        self.affected_incorrect_max_bold_regression=MaxBOLDContrastRegression(trials_max_bold_affected_incorrect,
            trials_max_bold_affected_incorrect_contrast)
        self.intact_correct_max_bold_regression=MaxBOLDContrastRegression(trials_max_bold_intact_correct,
            trials_max_bold_intact_correct_contrast)
        self.intact_incorrect_max_bold_regression=MaxBOLDContrastRegression(trials_max_bold_intact_incorrect,
            trials_max_bold_intact_incorrect_contrast)

        self.affected_correct_max_exc_bold_regression=MaxBOLDContrastRegression(trials_max_exc_bold_affected_correct,
            trials_max_exc_bold_affected_correct_contrast)
        self.affected_incorrect_max_exc_bold_regression=MaxBOLDContrastRegression(trials_max_exc_bold_affected_incorrect,
            trials_max_exc_bold_affected_incorrect_contrast)
        self.intact_correct_max_exc_bold_regression=MaxBOLDContrastRegression(trials_max_exc_bold_intact_correct,
            trials_max_exc_bold_intact_correct_contrast)
        self.intact_incorrect_max_exc_bold_regression=MaxBOLDContrastRegression(trials_max_exc_bold_intact_incorrect,
            trials_max_exc_bold_intact_incorrect_contrast)


def plot_auc_one_param_lesioned(base_dir, param_range, file_prefix, num_trials):
    p_auc=[]
    lesioned_p_auc=[]
    p_bc=[]
    p_exc_bc=[]
    for p in param_range:
        prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p,p,p,p,'control')
        p_series=TrialSeries(base_dir,prefix,num_trials)
        prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p,p,p,p,'lesioned')
        lesioned_p_series=TrialSeries(base_dir,prefix, num_trials)
        p_auc.append(p_series.total_auc)
        lesioned_p_auc.append(lesioned_p_series.total_auc)
        p_bc.append(p_series.max_bold_regression.bold_contrast_a)
        p_exc_bc.append(p_series.max_exc_bold_regression.bold_contrast_a)

    fig=plt.figure()
    plt.plot(param_range,p_auc,'b')
    plt.plot(param_range,lesioned_p_auc,'r')
    plt.xlabel('Param value')
    plt.ylabel('AUC')
    plt.show()

    fig=plt.figure()
    plt.hist(p_bc)
    plt.xlabel('BOLD - Contrast slope')
    plt.show()

    fig=plt.figure()
    plt.hist(p_exc_bc)
    plt.xlabel('Exc BOLD - Contrast slope')
    plt.show()

def open_two_param_range_files(base_dir, param_range, file_prefix, num_trials):
    param_files=[]
    for i,p_intra in enumerate(param_range):
        for j,p_inter in enumerate(param_range):
            prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p_intra,p_inter,p_intra,p_inter,'control')
            p_series=TrialSeries(base_dir,prefix,num_trials)
            p_series.sort_by_correct()
            param_files.append(p_series)
    return param_files

def open_lesioned_param_range_files(base_dir, param_range, file_prefix, num_trials):
    param_files=[]
    for i,p_intra in enumerate(param_range):
        for j,p_inter in enumerate(param_range):
            prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p_intra,p_inter,p_intra,p_inter,'lesioned')
            p_series=TrialSeries(base_dir,prefix,num_trials)
            p_series.sort_by_correct_lesioned()
            param_files.append(p_series)
    return param_files

def plot_bc_slope(base_dir, param_range, file_prefix, num_trials, perf_thresh=.90):
    param_files=open_two_param_range_files(base_dir, param_range, file_prefix, num_trials)
    bc_slope_data=compute_bc_slope(param_files, perf_thresh=perf_thresh)
    fig=plt.figure()
    ax=plt.subplot(211)
    if len(bc_slope_data.p_pos_bc):
        plt.hist(bc_slope_data.p_pos_bc)
        plt.xlabel('BOLD - Contrast slope')
        plt.ylabel('Positive')
    ax=plt.subplot(212)
    if len(bc_slope_data.p_neg_bc):
        plt.hist(bc_slope_data.p_neg_bc)
        plt.xlabel('BOLD - Contrast slope')
        plt.ylabel('Negative')
    plt.show()

    fig=plt.figure()
    ax=plt.subplot(221)
    if len(bc_slope_data.p_pos_bc_correct):
        plt.hist(bc_slope_data.p_pos_bc_correct)
        plt.xlabel('BOLD - Contrast slope')
        plt.ylabel('Positive - Correct')
    ax=plt.subplot(222)
    if len(bc_slope_data.p_pos_bc_incorrect):
        plt.hist(bc_slope_data.p_pos_bc_incorrect)
        plt.xlabel('BOLD - Contrast slope')
        plt.ylabel('Positive - Incorrect')
    ax=plt.subplot(223)
    if len(bc_slope_data.p_neg_bc_correct):
        plt.hist(bc_slope_data.p_neg_bc_correct)
        plt.xlabel('BOLD - Contrast slope')
        plt.ylabel('Negative - Correct')
    ax=plt.subplot(224)
    if len(bc_slope_data.p_neg_bc_incorrect):
        plt.hist(bc_slope_data.p_neg_bc_incorrect)
        plt.xlabel('BOLD - Contrast slope')
        plt.ylabel('Negative - Incorrect')
    plt.show()

    fig=plt.figure()
    ax=plt.subplot(211)
    if len(bc_slope_data.p_pos_exc_bc):
        plt.hist(bc_slope_data.p_pos_exc_bc)
        plt.xlabel('Exc BOLD - Contrast slope')
        plt.ylabel('Positive')
    ax=plt.subplot(212)
    if len(bc_slope_data.p_neg_exc_bc):
        plt.hist(bc_slope_data.p_neg_exc_bc)
        plt.xlabel('Exc BOLD - Contrast slope')
        plt.ylabel('Negative')
    plt.show()

    fig=plt.figure()
    ax=plt.subplot(221)
    if len(bc_slope_data.p_pos_exc_bc_correct):
        plt.hist(bc_slope_data.p_pos_exc_bc_correct)
        plt.xlabel('Exc BOLD - Contrast slope')
        plt.ylabel('Exc Positive - Correct')
    ax=plt.subplot(222)
    if len(bc_slope_data.p_pos_exc_bc_incorrect):
        plt.hist(bc_slope_data.p_pos_exc_bc_incorrect)
        plt.xlabel('Exc BOLD - Contrast slope')
        plt.ylabel('Exc Positive - Incorrect')
    ax=plt.subplot(223)
    if len(bc_slope_data.p_neg_exc_bc_correct):
        plt.hist(bc_slope_data.p_neg_exc_bc_correct)
        plt.xlabel('Exc BOLD - Contrast slope')
        plt.ylabel('Exc Negative - Correct')
    ax=plt.subplot(224)
    if len(bc_slope_data.p_neg_exc_bc_incorrect):
        plt.hist(bc_slope_data.p_neg_exc_bc_incorrect)
        plt.xlabel('Exc BOLD - Contrast slope')
        plt.ylabel('Exc Negative - Incorrect')
    plt.show()

def plot_perf_slope_analysis(base_dir, param_range, file_prefix, num_trials):
    num_bins=20
    min_perf=.5
    perf_range=min_perf+np.array(range(40))*(1.0-min_perf)/40.0
    param_files=open_two_param_range_files(base_dir, param_range, file_prefix, num_trials)

    bc_data=[]
    for perf_thresh in perf_range:
        bc_slope_data=compute_bc_slope(param_files, perf_thresh=perf_thresh)
        bc_data.append(bc_slope_data)

    pos_perf_bc_slope=np.zeros([num_bins,len(perf_range)])
    bins=None
    for i,bc_slope_data in enumerate(bc_data):
        if len(bc_slope_data.p_pos_bc):
            [bc_hist, bins]=np.histogram(bc_slope_data.p_pos_bc,bins=num_bins,range=(-.035,.035))
            pos_perf_bc_slope[:,i]=bc_hist/float(len(bc_slope_data.p_pos_bc))
    fig=plt.figure()
    plt.imshow(pos_perf_bc_slope, interpolation='nearest', origin='lower')
    plt.xlabel('Performance Threshold')
    plt.xticks(range(0,len(perf_range),2),['%.2f' % x for x in perf_range[range(0,len(perf_range),2)]])
    plt.ylabel('Slope')
    plt.yticks(range(0,len(bins)-1,2),['%.3f' % x for x in bins[range(0,len(bins)-1,2)]])
    plt.colorbar()

    fig=plt.figure()
    plt.imshow(pos_perf_bc_slope, interpolation='nearest', origin='lower', cmap = cm.Greys_r)
    plt.xlabel('Performance Threshold')
    plt.xticks(range(0,len(perf_range),2),['%.2f' % x for x in perf_range[range(0,len(perf_range),2)]])
    plt.ylabel('Slope')
    plt.yticks(range(0,len(bins)-1,2),['%.3f' % x for x in bins[range(0,len(bins)-1,2)]])
    plt.colorbar()

    pos_perf_correct_bc_slope=np.zeros([num_bins,len(perf_range)])
    bins=None
    for i,bc_slope_data in enumerate(bc_data):
        if len(bc_slope_data.p_pos_bc_correct):
            [bc_hist, bins]=np.histogram(bc_slope_data.p_pos_bc_correct,bins=num_bins,range=(-.035,.035))
            pos_perf_correct_bc_slope[:,i]=bc_hist/float(len(bc_slope_data.p_pos_bc_correct))
    fig=plt.figure()
    plt.imshow(pos_perf_correct_bc_slope, interpolation='nearest', origin='lower')
    plt.xlabel('Performance Threshold')
    plt.xticks(range(0,len(perf_range),2),['%.2f' % x for x in perf_range[range(0,len(perf_range),2)]])
    plt.ylabel('Slope')
    plt.yticks(range(0,len(bins)-1,2),['%.3f' % x for x in bins[range(0,len(bins)-1,2)]])
    plt.colorbar()

    fig=plt.figure()
    plt.imshow(pos_perf_correct_bc_slope, interpolation='nearest', origin='lower', cmap = cm.Greys_r)
    plt.xlabel('Performance Threshold')
    plt.xticks(range(0,len(perf_range),2),['%.2f' % x for x in perf_range[range(0,len(perf_range),2)]])
    plt.ylabel('Slope')
    plt.yticks(range(0,len(bins)-1,2),['%.3f' % x for x in bins[range(0,len(bins)-1,2)]])
    plt.colorbar()

    plt.show()


def compute_bc_slope(param_files, perf_thresh=.90):
    bc_slope_data=Struct()
    bc_slope_data.p_pos_bc=[]
    bc_slope_data.p_neg_bc=[]
    bc_slope_data.p_pos_bc_correct=[]
    bc_slope_data.p_pos_bc_incorrect=[]
    bc_slope_data.p_neg_bc_correct=[]
    bc_slope_data.p_neg_bc_incorrect=[]
    bc_slope_data.p_pos_exc_bc=[]
    bc_slope_data.p_neg_exc_bc=[]
    bc_slope_data.p_pos_exc_bc_correct=[]
    bc_slope_data.p_pos_exc_bc_incorrect=[]
    bc_slope_data.p_neg_exc_bc_correct=[]
    bc_slope_data.p_neg_exc_bc_incorrect=[]

    for p_series in param_files:
        if p_series.total_auc>=perf_thresh:
            if not math.isnan(p_series.max_bold_regression.bold_contrast_a):
                bc_slope_data.p_pos_bc.append(p_series.max_bold_regression.bold_contrast_a)
            if not math.isnan(p_series.max_exc_bold_regression.bold_contrast_a):
                bc_slope_data.p_pos_exc_bc.append(p_series.max_exc_bold_regression.bold_contrast_a)

            if not math.isnan(p_series.correct_max_bold_regression.bold_contrast_a):
                bc_slope_data.p_pos_bc_correct.append(p_series.correct_max_bold_regression.bold_contrast_a)
            if not math.isnan(p_series.correct_max_exc_bold_regression.bold_contrast_a):
                bc_slope_data.p_pos_exc_bc_correct.append(p_series.correct_max_exc_bold_regression.bold_contrast_a)

            if not math.isnan(p_series.incorrect_max_bold_regression.bold_contrast_a):
                bc_slope_data.p_pos_bc_incorrect.append(p_series.incorrect_max_bold_regression.bold_contrast_a)
            if not math.isnan(p_series.incorrect_max_exc_bold_regression.bold_contrast_a):
                bc_slope_data.p_pos_exc_bc_incorrect.append(p_series.incorrect_max_exc_bold_regression.bold_contrast_a)
        else:
            if not math.isnan(p_series.max_bold_regression.bold_contrast_a):
                bc_slope_data.p_neg_bc.append(p_series.max_bold_regression.bold_contrast_a)
            if not math.isnan(p_series.max_exc_bold_regression.bold_contrast_a):
                bc_slope_data.p_neg_exc_bc.append(p_series.max_exc_bold_regression.bold_contrast_a)

            if not math.isnan(p_series.correct_max_bold_regression.bold_contrast_a):
                bc_slope_data.p_neg_bc_correct.append(p_series.correct_max_bold_regression.bold_contrast_a)
            if not math.isnan(p_series.correct_max_exc_bold_regression.bold_contrast_a):
                bc_slope_data.p_neg_exc_bc_correct.append(p_series.correct_max_exc_bold_regression.bold_contrast_a)

            if not math.isnan(p_series.incorrect_max_bold_regression.bold_contrast_a):
                bc_slope_data.p_neg_bc_incorrect.append(p_series.incorrect_max_bold_regression.bold_contrast_a)
            if not math.isnan(p_series.incorrect_max_exc_bold_regression.bold_contrast_a):
                bc_slope_data.p_neg_exc_bc_incorrect.append(p_series.incorrect_max_exc_bold_regression.bold_contrast_a)

    return bc_slope_data

def plot_auc_two_param(base_dir, param_range, file_prefix, num_trials):
    p_auc=np.zeros([len(param_range), len(param_range)])
    lesioned_p_auc=np.zeros([len(param_range), len(param_range)])
    for i,p_intra in enumerate(param_range):
        for j,p_inter in enumerate(param_range):
            prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p_intra,p_inter,p_intra,p_inter,'control')
            p_series=TrialSeries(base_dir,prefix,num_trials)
            p_series.sort_by_correct()
            prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p_intra,p_inter,p_intra,p_inter,'lesioned')
            lesioned_p_series=TrialSeries(base_dir,prefix, num_trials)
            p_auc[i,j]=p_series.total_auc
            lesioned_p_auc[i,j]=lesioned_p_series.total_auc

    fig=plt.figure()
    im = plt.imshow(p_auc, extent=[min(param_range), max(param_range), min(param_range),
                                   max(param_range)], interpolation='nearest', origin='lower')
    fig.colorbar(im)
    plt.xlabel('p_intra')
    plt.ylabel('p_inter')
    plt.title('Control')
    plt.show()

    fig=plt.figure()
    im = plt.imshow(lesioned_p_auc, extent=[min(param_range), max(param_range), min(param_range),
                                            max(param_range)], interpolation='nearest', origin='lower')
    fig.colorbar(im)
    plt.xlabel('p_intra')
    plt.ylabel('p_inter')
    plt.title('Lesioned')
    plt.show()



def plot_bold_contrast_lesion_two_param(base_dir, param_range, file_prefix, num_trials):
    bc_slope=np.zeros([len(param_range), len(param_range)])
    lesioned_bc_slope=np.zeros([len(param_range), len(param_range)])
    bc_exc_slope=np.zeros([len(param_range), len(param_range)])
    lesioned_bc_exc_slope=np.zeros([len(param_range), len(param_range)])
    for i,p_intra in enumerate(param_range):
        for j,p_inter in enumerate(param_range):
            prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p_intra,p_inter,p_intra,p_inter,'control')
            control_data=TrialSeries(base_dir,prefix,num_trials)
            control_data.sort_by_correct()
            prefix='%s.p_e_e.%.3f.p_e_i.%.3f.p_i_i.%.3f.p_i_e.%.3f.%s' % (file_prefix,p_intra,p_inter,p_intra,p_inter,'lesioned')
            lesioned_data=TrialSeries(base_dir,prefix, num_trials)
            lesioned_data.sort_by_correct_lesioned()
            bc_slope[i,j]=control_data.max_bold_regression.bold_contrast_a
            lesioned_bc_slope[i,j]=lesioned_data.max_bold_regression.bold_contrast_a
            bc_exc_slope[i,j]=control_data.max_exc_bold_regression.bold_contrast_a
            lesioned_bc_exc_slope[i,j]=lesioned_data.max_exc_bold_regression.bold_contrast_a

    fig=plt.figure()
    im = plt.imshow(bc_slope, extent=[min(param_range), max(param_range), min(param_range),
                                      max(param_range)], interpolation='nearest', origin='lower')
    fig.colorbar(im)
    plt.xlabel('p_intra')
    plt.ylabel('p_inter')
    plt.title('Control')
    plt.show()

    fig=plt.figure()
    im = plt.imshow(lesioned_bc_slope, extent=[min(param_range), max(param_range), min(param_range),
                                               max(param_range)], interpolation='nearest', origin='lower')
    fig.colorbar(im)
    plt.xlabel('p_intra')
    plt.ylabel('p_inter')
    plt.title('Lesioned')
    plt.show()

    fig=plt.figure()
    im = plt.imshow(bc_exc_slope, extent=[min(param_range), max(param_range), min(param_range),
                                      max(param_range)], interpolation='nearest', origin='lower')
    fig.colorbar(im)
    plt.xlabel('p_intra')
    plt.ylabel('p_inter')
    plt.title('Control - Exc only')
    plt.show()

    fig=plt.figure()
    im = plt.imshow(lesioned_bc_exc_slope, extent=[min(param_range), max(param_range), min(param_range),
                                               max(param_range)], interpolation='nearest', origin='lower')
    fig.colorbar(im)
    plt.xlabel('p_intra')
    plt.ylabel('p_inter')
    plt.title('Lesioned - Exc only')
    plt.show()


def plot_bold_contrast_lesion_one_param(output_dir, file_prefix, num_trials):
    
    control_data=TrialSeries(output_dir, '%s.control' % file_prefix, num_trials)
    control_data.sort_by_correct()
    lesioned_data=TrialSeries(output_dir,'%s.lesioned' % file_prefix, num_trials)
    lesioned_data.sort_by_correct_lesioned()
    print('control AUC=%.4f' % control_data.total_auc)
    print('lesioned AUC=%.4f' % lesioned_data.total_auc)
            
    x_min=np.min(control_data.contrast_range)
    x_max=np.max(control_data.contrast_range)

    fig=plt.figure()
    if len(control_data.correct_max_bold_regression.trials_max_bold):
        control_data.max_bold_regression.plot(x_max, x_min,'ok','k','Control Correct')
        
    if len(control_data.incorrect_max_bold_regression.trials_max_bold):
        control_data.incorrect_max_bold_regression.plot(x_max, x_min, 'xk', '--k', 'Control Incorrect')

    if len(lesioned_data.affected_correct_max_bold_regression.trials_max_bold):
        lesioned_data.affected_correct_max_bold_regression.plot(x_max,x_min,'xb','--b','Lesioned Affected Correct')

    if len(lesioned_data.affected_incorrect_max_bold_regression.trials_max_bold):
        lesioned_data.affected_incorrect_max_bold_regression.plot(x_max,x_min,'xr','--r','Lesioned Affected Incorrect')

    if len(lesioned_data.intact_correct_max_bold_regression.trials_max_bold):
        lesioned_data.intact_correct_max_bold_regression.plot(x_max, x_min, 'ob','b','Lesioned Intact Correct')

    if len(lesioned_data.intact_incorrect_max_bold_regression.trials_max_bold):
        lesioned_data.intact_incorrect_max_bold_regression.plot(x_max, x_min, 'or', 'r','Lesioned Intact Incorrect')

    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD')
    plt.legend(loc='best')
    plt.show()

    fig=plt.figure()
    if len(control_data.correct_max_exc_bold_regression.trials_max_bold):
        control_data.correct_max_exc_bold_regression.plot(x_max, x_min,'ok','k','Control Correct')

    if len(control_data.incorrect_max_exc_bold_regression.trials_max_bold):
        control_data.incorrect_max_exc_bold_regression.plot(x_max, x_min, 'xk', '--k','Control Incorrect')

    if len(lesioned_data.affected_correct_max_exc_bold_regression.trials_max_bold):
        lesioned_data.affected_correct_max_exc_bold_regression.plot(x_max,x_min,'xb','--b','Lesioned Affected Correct')

    if len(lesioned_data.affected_incorrect_max_exc_bold_regression.trials_max_bold):
        lesioned_data.affected_incorrect_max_exc_bold_regression.plot(x_max,x_min,'xr','--r','Lesioned Affected Incorrect')

    if len(lesioned_data.intact_correct_max_exc_bold_regression.trials_max_bold):
        lesioned_data.intact_correct_max_exc_bold_regression.plot(x_max, x_min, 'ob','b','Lesioned Intact Correct')

    if len(lesioned_data.intact_incorrect_max_exc_bold_regression.trials_max_bold):
        lesioned_data.intact_incorrect_max_exc_bold_regression.plot(x_max, x_min, 'or', 'r','Lesioned Intact Incorrect')
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD (exc only)')
    plt.legend(loc='best')
    plt.show()


def create_dcs_comparison_report(data_dir, file_prefix, stim_levels, num_trials, reports_dir, edesc):
    """
    Create report for DCS simulations
    data_dir=directory where datafiles are stored
    file_prefix=prefix of data files (before p_dcs param)
    stim_levels= dict of conditions - key is name, value is tuple of stimulation levels in pA of pyramidal and interneurons
        i.e. {'control':(0,0),'anode':(4,-2),'cathode':(-4,2)}
    num_trials=number of trials in each condition
    reports_dir=directory to put reports in
    edesc=extra description
    """
    make_report_dirs(reports_dir)

    report_info=Struct()
    report_info.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    report_info.edesc=edesc

    report_info.urls={}
    series={}
    for stim_level in stim_levels:
        p_dcs=stim_levels[stim_level][0]
        i_dcs=stim_levels[stim_level][1]
        stim_report_dir=os.path.join(reports_dir,stim_level)
        prefix='%s.p_dcs.%.4f.i_dcs.%.4f.control' % (file_prefix,p_dcs,i_dcs)
        stim_report=create_network_report(data_dir,prefix,num_trials,stim_report_dir,'', version=report_info.version)
        report_info.urls[stim_level]=os.path.join(stim_level,'wta_network.%s.html' % prefix)
        series[stim_level]=stim_report.series
        if stim_level=='control':
            report_info.wta_params=stim_report.series.trial_summaries[0].data.wta_params
            report_info.voxel_params=stim_report.series.trial_summaries[0].data.voxel_params
            report_info.num_groups=stim_report.series.trial_summaries[0].data.num_groups
            report_info.trial_duration=stim_report.series.trial_summaries[0].data.trial_duration
            report_info.background_freq=stim_report.series.trial_summaries[0].data.background_freq
            report_info.stim_start_time=stim_report.series.trial_summaries[0].data.stim_start_time
            report_info.stim_end_time=stim_report.series.trial_summaries[0].data.stim_end_time
            report_info.network_group_size=stim_report.series.trial_summaries[0].data.network_group_size
            report_info.background_input_size=stim_report.series.trial_summaries[0].data.background_input_size
            report_info.task_input_size=stim_report.series.trial_summaries[0].data.task_input_size
            report_info.muscimol_amount=stim_report.series.trial_summaries[0].data.muscimol_amount
            report_info.injection_site=stim_report.series.trial_summaries[0].data.injection_site
        del stim_report
    colors={'anode':'g','cathode':'b'}

    furl='img/rt'
    fname=os.path.join(reports_dir, furl)
    report_info.rt_url='%s.png' % furl
    fig=plt.figure()
    contrast, mean_rt, std_rt = series['control'].get_contrast_rt_stats()
    #plt.errorbar(contrast,mean_rt,yerr=std_rt,fmt='ko-',label='control')
    plt.plot(contrast,mean_rt,'ko',label='control')
    try:
        popt,pcov=curve_fit(rt_function, contrast, mean_rt)
        plt.plot(np.array(range(101))*.01,rt_function(np.array(range(101))*.01,*popt),'k')
    except:
        print('error fitting RT data')
    for stim_level in stim_levels:
        if not stim_level=='control':
            contrast, mean_rt, std_rt = series[stim_level].get_contrast_rt_stats()
            #plt.errorbar(contrast,mean_rt,yerr=std_rt,fmt='o-'+colors[color_idx],label=stim_level)
            plt.plot(contrast,mean_rt,'o'+colors[stim_level])
            try:
                popt,pcov=curve_fit(rt_function, contrast, mean_rt)
                plt.plot(np.array(range(101))*.01,rt_function(np.array(range(101))*.01,*popt),colors[stim_level])
            except:
                print('error fitting RT data')
    plt.xlabel('Contrast')
    plt.ylabel('Decision time (ms)')
    plt.xscale('log')
    plt.legend()
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

    furl='img/perc_correct'
    fname=os.path.join(reports_dir, furl)
    report_info.perc_correct_url='%s.png' % furl
    fig=plt.figure()
    contrast, perc_correct = series['control'].get_contrast_perc_correct_stats()
    plt.plot(contrast,perc_correct,'ok',label='control')
    try:
        popt, pcov = curve_fit(weibull, contrast, perc_correct)
        plt.plot(np.array(range(101))*.01,weibull(np.array(range(101))*.01,*popt),'k')
    except:
        print('error fitting performance data')
    for stim_level in stim_levels:
        if not stim_level=='control':
            contrast, perc_correct = series[stim_level].get_contrast_perc_correct_stats()
            plt.plot(contrast,perc_correct,'o'+colors[stim_level],label=stim_level)
            try:
                popt, pcov = curve_fit(weibull, contrast, perc_correct, maxfev=3000)
                plt.plot(np.array(range(101))*.01,weibull(np.array(range(101))*.01,*popt),colors[stim_level])
            except:
                print('error fitting performance data')
    plt.xlabel('Contrast')
    plt.ylabel('% correct')
    plt.legend()
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

    furl='img/bold_contrast_regression'
    fname=os.path.join(reports_dir,furl)
    report_info.bold_contrast_regression_url='%s.png' % furl
    x_min=np.min(series['control'].contrast_range)
    x_max=np.max(series['control'].contrast_range)
    fig=plt.figure()
    series['control'].max_bold_regression.plot(x_max, x_min,'ok','k','control')
    for stim_level in stim_levels:
        if not stim_level=='control':
            series[stim_level].max_bold_regression.plot(x_max, x_min,'o'+colors[stim_level],colors[stim_level],stim_level)
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD')
    plt.legend(loc='best')
    plt.xscale('log')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

    #create report
    template_file='wta_dcs_comparison.html'
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template=env.get_template(template_file)

    output_file='dcs_comparison.%s.html' % file_prefix
    fname=os.path.join(reports_dir,output_file)
    stream=template.stream(rinfo=report_info)
    stream.dump(fname)

def create_network_report(data_dir, file_prefix, num_trials, reports_dir, edesc, version=None):
    make_report_dirs(reports_dir)

    report_info=Struct()
    if version is None:
        version=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    report_info.version = version
    report_info.edesc=edesc

    report_info.series=TrialSeries(data_dir, file_prefix, num_trials)
    report_info.series.sort_by_correct()

    report_info.wta_params=report_info.series.trial_summaries[0].data.wta_params
    report_info.voxel_params=report_info.series.trial_summaries[0].data.voxel_params
    report_info.num_groups=report_info.series.trial_summaries[0].data.num_groups
    report_info.trial_duration=report_info.series.trial_summaries[0].data.trial_duration
    report_info.background_freq=report_info.series.trial_summaries[0].data.background_freq
    report_info.stim_start_time=report_info.series.trial_summaries[0].data.stim_start_time
    report_info.stim_end_time=report_info.series.trial_summaries[0].data.stim_end_time
    report_info.network_group_size=report_info.series.trial_summaries[0].data.network_group_size
    report_info.background_input_size=report_info.series.trial_summaries[0].data.background_input_size
    report_info.task_input_size=report_info.series.trial_summaries[0].data.task_input_size
    report_info.muscimol_amount=report_info.series.trial_summaries[0].data.muscimol_amount
    report_info.injection_site=report_info.series.trial_summaries[0].data.injection_site
    report_info.p_dcs=report_info.series.trial_summaries[0].data.p_dcs
    report_info.i_dcs=report_info.series.trial_summaries[0].data.i_dcs

    furl='img/roc'
    fname=os.path.join(reports_dir, furl)
    report_info.roc_url='%s.png' % furl
    report_info.series.plot_multiclass_roc(filename=fname)

    furl='img/rt'
    fname=os.path.join(reports_dir, furl)
    report_info.rt_url='%s.png' % furl
    report_info.series.plot_rt(filename=fname)

    furl='img/perc_correct'
    fname=os.path.join(reports_dir, furl)
    report_info.perc_correct_url='%s.png' % furl
    report_info.series.plot_perc_correct(filename=fname)

    furl='img/bold_contrast_regression'
    fname=os.path.join(reports_dir,furl)
    report_info.bold_contrast_regression_url='%s.png' % furl
    x_min=np.min(report_info.series.contrast_range)
    x_max=np.max(report_info.series.contrast_range)
    fig=plt.figure()
    report_info.series.max_bold_regression.plot(x_max, x_min,'ok','k','Fit')
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD')
    plt.legend(loc='best')
    plt.xscale('log')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

    report_info.trial_reports=[]
    for trial_summary in report_info.series.trial_summaries:
        report_info.trial_reports.append(create_trial_report(trial_summary, reports_dir))

    #create report
    template_file='wta_network_instance_new.html'
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template=env.get_template(template_file)

    output_file='wta_network.%s.html' % file_prefix
    fname=os.path.join(reports_dir,output_file)
    stream=template.stream(rinfo=report_info)
    stream.dump(fname)

    return report_info

def create_trial_report(trial_summary, reports_dir):
    trial_report=Struct()
    trial_report.trial_idx=trial_summary.trial_idx
    trial_report.contrast=trial_summary.contrast
    trial_report.input_freq=trial_summary.data.input_freq
    trial_report.correct=trial_summary.correct
    trial_report.rt=trial_summary.data.rt
    trial_report.max_rate=trial_summary.max_rate
    trial_report.max_bold=trial_summary.data.summary_data.bold_max

    trial_report.firing_rate_url = None
    if trial_summary.data.e_firing_rates is not None and trial_summary.data.i_firing_rates is not None:
        furl = 'img/firing_rate.contrast.%0.4f.trial.%d' % (trial_summary.contrast, trial_summary.trial_idx)
        fname = os.path.join(reports_dir, furl)
        trial_report.firing_rate_url = '%s.png' % furl

        # figure out max firing rate of all neurons (pyramidal and interneuron)
        max_pop_rate=0
        for i, pop_rate in enumerate(trial_summary.data.e_firing_rates):
            max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])
        for i, pop_rate in enumerate(trial_summary.data.i_firing_rates):
            max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])

        #fig = plt.figure()
        fig=Figure()

        # Plot pyramidal neuron firing rate
        #ax = plt.subplot(211)
        ax=fig.add_subplot(2,1,1)
        for i, pop_rate in enumerate(trial_summary.data.e_firing_rates):
            ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
        # Plot line showing RT
        if trial_report.rt:
            rt_idx=(trial_summary.data.stim_start_time+trial_report.rt)/ms
            ax.plot([rt_idx,rt_idx],[0,max_pop_rate])
        plt.ylim([0,10+max_pop_rate])
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')

        # Plot interneuron firing rate
        #ax = plt.subplot(212)
        ax = fig.add_subplot(2,1,2)
        for i, pop_rate in enumerate(trial_summary.data.i_firing_rates):
            ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
        # Plot line showing RT
        if trial_report.rt:
            rt_idx=(trial_summary.data.stim_start_time+trial_report.rt)/ms
            ax.plot([rt_idx,rt_idx],[0,max_pop_rate])
        plt.ylim([0,10+max_pop_rate])
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    trial_report.neural_state_url=None
    if trial_summary.data.neural_state_rec is not None:
        furl = 'img/neural_state.contrast.%0.4f.trial.%d' % (trial_summary.contrast, trial_summary.trial_idx)
        fname = os.path.join(reports_dir, furl)
        trial_report.neural_state_url = '%s.png' % furl
        fig = plt.figure()
        for i in range(trial_summary.data.num_groups):
            times=np.array(range(len(trial_summary.data.neural_state_rec['g_ampa_r'][i*2])))*.1
            ax = plt.subplot(trial_summary.data.num_groups * 100 + 20 + (i * 2 + 1))
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_r'][i * 2] / nA, label='AMPA-recurrent')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_x'][i * 2] / nA, label='AMPA-task')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_b'][i * 2] / nA, label='AMPA-backgrnd')
            ax.plot(times, trial_summary.data.neural_state_rec['g_nmda'][i * 2] / nA, label='NMDA')
            ax.plot(times, trial_summary.data.neural_state_rec['g_gaba_a'][i * 2] / nA, label='GABA_A')
            plt.xlabel('Time (ms)')
            plt.ylabel('Conductance (nA)')
            ax = plt.subplot(trial_summary.data.num_groups * 100 + 20 + (i * 2 + 2))
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_r'][i * 2 + 1] / nA, label='AMPA-recurrent')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_x'][i * 2 + 1] / nA, label='AMPA-task')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_b'][i * 2 + 1] / nA, label='AMPA-backgrnd')
            ax.plot(times, trial_summary.data.neural_state_rec['g_nmda'][i * 2 + 1] / nA, label='NMDA')
            ax.plot(times, trial_summary.data.neural_state_rec['g_gaba_a'][i * 2 + 1] / nA, label='GABA_A')
            plt.xlabel('Time (ms)')
            plt.ylabel('Conductance (nA)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    trial_report.lfp_url = None
    if trial_summary.data.lfp_rec is not None:
        furl = 'img/lfp.contrast.%0.4f.trial.%d' % (trial_summary.contrast, trial_summary.trial_idx)
        fname = os.path.join(reports_dir, furl)
        trial_report.lfp_url = '%s.png' % furl
        fig = plt.figure()
        ax = plt.subplot(111)
        lfp=get_lfp_signal(trial_summary.data)
        ax.plot(np.array(range(len(lfp))), lfp / mA)
        plt.xlabel('Time (ms)')
        plt.ylabel('LFP (mA)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    trial_report.voxel_url = None
    if trial_summary.data.voxel_rec is not None:
        furl = 'img/voxel.contrast.%0.4f.trial.%d' % (trial_summary.contrast, trial_summary.trial_idx)
        fname = os.path.join(reports_dir, furl)
        trial_report.voxel_url = '%s.png' % furl
        end_idx=int(trial_summary.data.trial_duration/ms/.1)
        fig = plt.figure()
        ax = plt.subplot(211)
        ax.plot(np.array(range(end_idx))*.1, trial_summary.data.voxel_rec['G_total'][0][:end_idx] / nA)
        plt.xlabel('Time (ms)')
        plt.ylabel('Total Synaptic Activity (nA)')
        ax = plt.subplot(212)
        ax.plot(np.array(range(len(trial_summary.data.voxel_rec['y'][0])))*.1*ms, trial_summary.data.voxel_rec['y'][0])
        plt.xlabel('Time (s)')
        plt.ylabel('BOLD')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    return trial_report


class TrialReport:
    def __init__(self, data_dir, file_prefix, reports_dir, trial_idx):
        self.data_dir=data_dir
        self.reports_dir=reports_dir
        self.file_prefix=file_prefix
        self.trial_idx=trial_idx

    def create_report(self):
        data=FileInfo(os.path.join(self.data_dir,'%s.h5' % self.file_prefix))
        self.input_freq=data.input_freq
        self.rt=data.rt

        self.firing_rate_url = None
        if data.e_firing_rates is not None and data.i_firing_rates is not None:
            furl = 'img/firing_rate.%s' % self.file_prefix
            fname = os.path.join(self.reports_dir, furl)
            self.firing_rate_url = '%s.png' % furl

            # figure out max firing rate of all neurons (pyramidal and interneuron)
            max_pop_rate=0
            for i, pop_rate in enumerate(data.e_firing_rates):
                max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])
            for i, pop_rate in enumerate(data.i_firing_rates):
                max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])

            #fig = plt.figure()
            fig=Figure()

            # Plot pyramidal neuron firing rate
            #ax = plt.subplot(211)
            ax=fig.add_subplot(2,1,1)
            for i, pop_rate in enumerate(data.e_firing_rates):
                ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
                # Plot line showing RT
            if self.rt:
                rt_idx=(data.stim_start_time+self.rt)/ms
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate])
            plt.ylim([0,10+max_pop_rate])
            plt.legend()
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing Rate (Hz)')

            # Plot interneuron firing rate
            #ax = plt.subplot(212)
            ax = fig.add_subplot(2,1,2)
            for i, pop_rate in enumerate(data.i_firing_rates):
                ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
                # Plot line showing RT
            if self.rt:
                rt_idx=(data.stim_start_time+self.rt)/ms
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate])
            plt.ylim([0,10+max_pop_rate])
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing Rate (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        self.neural_state_url=None
        if data.neural_state_rec is not None:
            furl = 'img/neural_state.%s' % self.file_prefix
            fname = os.path.join(self.reports_dir, furl)
            self.neural_state_url = '%s.png' % furl
            fig = plt.figure()
            for i in range(data.num_groups):
                times=np.array(range(len(data.neural_state_rec['g_ampa_r'][i*2])))*.1
                ax = plt.subplot(data.num_groups * 100 + 20 + (i * 2 + 1))
                ax.plot(times, data.neural_state_rec['g_ampa_r'][i * 2] / nA, label='AMPA-recurrent')
                ax.plot(times, data.neural_state_rec['g_ampa_x'][i * 2] / nA, label='AMPA-task')
                ax.plot(times, data.neural_state_rec['g_ampa_b'][i * 2] / nA, label='AMPA-backgrnd')
                ax.plot(times, data.neural_state_rec['g_nmda'][i * 2] / nA, label='NMDA')
                ax.plot(times, data.neural_state_rec['g_gaba_a'][i * 2] / nA, label='GABA_A')
                plt.xlabel('Time (ms)')
                plt.ylabel('Conductance (nA)')
                ax = plt.subplot(data.num_groups * 100 + 20 + (i * 2 + 2))
                ax.plot(times, data.neural_state_rec['g_ampa_r'][i * 2 + 1] / nA, label='AMPA-recurrent')
                ax.plot(times, data.neural_state_rec['g_ampa_x'][i * 2 + 1] / nA, label='AMPA-task')
                ax.plot(times, data.neural_state_rec['g_ampa_b'][i * 2 + 1] / nA, label='AMPA-backgrnd')
                ax.plot(times, data.neural_state_rec['g_nmda'][i * 2 + 1] / nA, label='NMDA')
                ax.plot(times, data.neural_state_rec['g_gaba_a'][i * 2 + 1] / nA, label='GABA_A')
                plt.xlabel('Time (ms)')
                plt.ylabel('Conductance (nA)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        self.lfp_url = None
        if data.lfp_rec is not None:
            furl = 'img/lfp.%s' % self.file_prefix
            fname = os.path.join(self.reports_dir, furl)
            self.lfp_url = '%s.png' % furl
            fig = plt.figure()
            ax = plt.subplot(111)
            lfp=get_lfp_signal(data)
            ax.plot(np.array(range(len(lfp))), lfp / mA)
            plt.xlabel('Time (ms)')
            plt.ylabel('LFP (mA)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        self.voxel_url = None
        if data.voxel_rec is not None:
            furl = 'img/voxel.%s' % self.file_prefix
            fname = os.path.join(self.reports_dir, furl)
            self.voxel_url = '%s.png' % furl
            end_idx=int(data.trial_duration/ms/.1)
            fig = plt.figure()
            ax = plt.subplot(211)
            ax.plot(np.array(range(end_idx))*.1, data.voxel_rec['G_total'][0][:end_idx] / nA)
            plt.xlabel('Time (ms)')
            plt.ylabel('Total Synaptic Activity (nA)')
            ax = plt.subplot(212)
            ax.plot(np.array(range(len(data.voxel_rec['y'][0])))*.1*ms, data.voxel_rec['y'][0])
            plt.xlabel('Time (s)')
            plt.ylabel('BOLD')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)


def get_fanos(spike_times, spike_neurons, num_neurons, start_time, end_time):
    fano_factors=np.zeros(num_neurons)
    for i in range(num_neurons):
        idx=np.where((spike_times>=start_time) & (spike_times<end_time) & (spike_neurons==i))[0]
        isis=np.array(np.diff([spike_times[j] for j in idx]))
        if len(isis)>1:
            fano_factors[i]=float(isis.std())/abs(float(isis.mean()))
    return fano_factors

def test_fano3(file_name, time_window):
    data=FileInfo(file_name)
    e_size=int(data.network_group_size*.8/2)

    idx=0
    start_time=0*second
    end_time=time_window
    slide_width=time_window/10.0
    e0_fanos=np.zeros((data.trial_duration/slide_width,e_size))
    e1_fanos=np.zeros((data.trial_duration/slide_width,e_size))

    while end_time<data.trial_duration:
        print(end_time)
        e0_fanos[idx,:]=get_fanos(data.e_spike_times[0],data.e_spike_neurons[0],e_size,start_time,end_time)

        e1_fanos[idx,:]=get_fanos(data.e_spike_times[1],data.e_spike_neurons[1],e_size,start_time,end_time)

        idx+=1
        start_time+=slide_width
        end_time+=slide_width

    e0_mean_fano=np.mean(e0_fanos,axis=1)
    e0_std_fano=np.std(e0_fanos,axis=1)
    e1_mean_fano=np.mean(e1_fanos,axis=1)
    e1_std_fano=np.std(e1_fanos,axis=1)

    fig=plt.figure()
    ax=plt.subplot(211)
    plt.plot(data.e_firing_rates[0])
    plt.plot(data.e_firing_rates[1])
    #plt.plot(data.i_firing_rates[0])
    ax=plt.subplot(212)
    plt.plot(e0_mean_fano,'b')
    plt.fill_between(range(len(e0_mean_fano)),e0_mean_fano-e0_std_fano,e0_mean_fano+e0_std_fano,alpha=0.5)
    plt.plot(e1_mean_fano,'g')
    plt.fill_between(range(len(e1_mean_fano)),e1_mean_fano-e1_std_fano,e1_mean_fano+e1_std_fano,alpha=0.5)
    #plt.plot(i_fano)
    plt.show()

def test_fano2(file_name, time_window):
    data=FileInfo(file_name)
    e_size=int(data.network_group_size*.8/2)

    idx=0
    start_time=0*second
    end_time=time_window
    bins=10
    #slide_width=time_window/bins
    slide_width=10*ms
    e_0_fano=np.zeros((data.trial_duration/slide_width))
    e_1_fano=np.zeros((data.trial_duration/slide_width))
    start_idx_1=0
    start_idx_2=0

    while end_time<data.trial_duration:
        print(end_time)
        # Analyze data for each specific neuron:
        e0_timesDict = {}
        e1_timesDict = {}
        for neuron in range(e_size):
            e0_timesDict[neuron] = []
            e1_timesDict[neuron] = []
        for bin in range(bins):
            # We iterate over the bins to calculate the spike counts
            lower_time = start_time+time_window/bins*bin
            upper_time = start_time+time_window/bins*(bin+1)
            for j in range(start_idx_1,len(data.e_spike_times[0])):
                if lower_time > data.e_spike_times[0][j]*second:
                    start_idx_1=j
                if lower_time <= data.e_spike_times[0][j] * second < upper_time:
                    e0_timesDict[data.e_spike_neurons[0][j]].append(data.e_spike_times[0][j])
                elif data.e_spike_times[0][j]*second>upper_time:
                    break

            for j in range(start_idx_2,len(data.e_spike_times[1])):
                if lower_time > data.e_spike_times[1][j]*second:
                    start_idx_2=j
                if lower_time <= data.e_spike_times[1][j] * second < upper_time:
                    e1_timesDict[data.e_spike_neurons[1][j]].append(data.e_spike_times[1][j])
                elif data.e_spike_times[1][j]*second>upper_time:
                    break

        e0_isis=[]
        e1_isis=[]
        for neuron in range(e_size):
            e0_isis.extend(np.diff(e0_timesDict[neuron]))
            e1_isis.extend(np.diff(e1_timesDict[neuron]))
        e0_isis=np.array(e0_isis)
        e1_isis=np.array(e1_isis)
        if len(e0_isis)>1:
            e_0_fano[idx]=float(e0_isis.var())/abs(float(e0_isis.mean()))
        if len(e1_isis)>1:
            e_1_fano[idx]=float(e1_isis.var())/abs(float(e1_isis.mean()))

        idx+=1
        start_time+=slide_width
        end_time+=slide_width

    #fig=plt.figure()
    #ax=plt.subplot(211)
    #plt.plot(data.e_firing_rates[0])
    #plt.plot(data.e_firing_rates[1])
    #plt.plot(data.i_firing_rates[0])
    #ax=plt.subplot(212)
    #plt.plot(np.mean(e_0_fano,axis=1))
    #plt.plot(np.mean(e_1_fano,axis=1))
    #plt.plot(np.mean(i_fano,axis=1))

    fig=plt.figure()
    ax=plt.subplot(211)
    plt.plot(data.e_firing_rates[0])
    plt.plot(data.e_firing_rates[1])
    #plt.plot(data.i_firing_rates[0])
    ax=plt.subplot(212)
    plt.plot(e_0_fano)
    plt.plot(e_1_fano)
    #plt.plot(i_fano)
    plt.show()

def test_fano(file_name, time_window):
    data=FileInfo(file_name)

    idx=0
    start_time=0*second
    end_time=time_window
    bins=5
    slide_width=time_window/bins
    e_0_fano=np.zeros((data.trial_duration/slide_width,int(data.network_group_size*.8/2)))
    e_1_fano=np.zeros((data.trial_duration/slide_width,int(data.network_group_size*.8/2)))
    i_fano=np.zeros((data.trial_duration/slide_width,int(data.network_group_size*.2)))

    start_idx_1=0
    start_idx_2=0
    start_idx_3=0

    while end_time<data.trial_duration:
        print(end_time)
        #for i in range(data.network_group_size):
        # Arrays for binning of spike counts
        e0_binned_spikes = np.zeros((int(data.network_group_size*.8/2),bins))
        e1_binned_spikes = np.zeros((int(data.network_group_size*.8/2),bins))
        i_binned_spikes = np.zeros((int(data.network_group_size*.2),bins))
        for bin in range(bins):
            # We iterate over the bins to calculate the spike counts
            lower_time = start_time+time_window/bins*bin
            upper_time = start_time+time_window/bins*(bin+1)
            for j in range(start_idx_1,len(data.e_spike_times[0])):
                if lower_time > data.e_spike_times[0][j]*second:
                    start_idx_1=j
                if lower_time <= data.e_spike_times[0][j] * second < upper_time:
                    e0_binned_spikes[data.e_spike_neurons[0][j],bin]+=1.0
                elif data.e_spike_times[0][j]*second>upper_time:
                    break

            for j in range(start_idx_2,len(data.e_spike_times[1])):
                if lower_time > data.e_spike_times[1][j]*second:
                    start_idx_2=j
                if lower_time <= data.e_spike_times[1][j] * second < upper_time:
                    e1_binned_spikes[data.e_spike_neurons[1][j],bin]+=1.0
                elif data.e_spike_times[1][j]*second>upper_time:
                    break

            for j in range(start_idx_3,len(data.i_spike_times[0])):
                if lower_time > data.i_spike_times[0][j]*second:
                    start_idx_3=j
                if lower_time <= data.i_spike_times[0][j] * second < upper_time:
                    i_binned_spikes[data.i_spike_neurons[0][j],bin]+=1.0
                elif data.i_spike_times[0][j]*second>upper_time:
                    break

        var = np.var(e0_binned_spikes,axis=1)
        avg = np.mean(e0_binned_spikes,axis=1)
        nz_idx=np.where(avg>0)
        e_0_fano[idx,nz_idx[0]]=var[nz_idx[0]]/avg[nz_idx[0]]

        var = np.var(e1_binned_spikes,axis=1)
        avg = np.mean(e1_binned_spikes,axis=1)
        nz_idx=np.where(avg>0)
        e_1_fano[idx,nz_idx[0]]=var[nz_idx[0]]/avg[nz_idx[0]]

        var = np.var(i_binned_spikes,axis=1)
        avg = np.mean(i_binned_spikes,axis=1)
        nz_idx=np.where(avg>0)
        i_fano[idx,nz_idx[0]]=var[nz_idx[0]]/avg[nz_idx[0]]

        #print(np.mean(e_0_fano,axis=1))
        idx+=1
        start_time+=slide_width
        end_time+=slide_width

    fig=plt.figure()
    ax=plt.subplot(211)
    plt.plot(data.e_firing_rates[0])
    plt.plot(data.e_firing_rates[1])
    plt.plot(data.i_firing_rates[0])
    ax=plt.subplot(212)
    plt.plot(np.mean(e_0_fano,axis=1))
    plt.plot(np.mean(e_1_fano,axis=1))
    plt.plot(np.mean(i_fano,axis=1))

    fig=plt.figure()
    ax=plt.subplot(211)
    plt.plot(data.e_firing_rates[0])
    plt.plot(data.e_firing_rates[1])
    plt.plot(data.i_firing_rates[0])
    ax=plt.subplot(212)
    plt.plot(e_0_fano)
    plt.plot(e_1_fano)
    plt.plot(i_fano)
    plt.show()

def analyze_input_test(data_dir, trials):
    input_diffs=np.zeros(trials)
    choice_made=np.zeros(trials)
    correct_choice=np.zeros(trials)
    for trial in range(trials):
        filename=os.path.join(data_dir,'trial.%d.h5' % trial)
        data=FileInfo(filename, dt=.5*ms)
        input_diffs[trial]=np.abs(data.input_freq[0]-data.input_freq[1])
        if data.choice>-1:
            choice_made[trial]=1.0
            if data.input_freq[data.choice]>data.input_freq[1-data.choice]:
                correct_choice[trial]=1.0
    fig=plt.figure()
    hist,bins=np.histogram(input_diffs, bins=10)
    bin_width=bins[1]-bins[0]
    plt.bar(bins[:-1], hist/float(input_diffs.shape[0]), width=bin_width)
    plt.xlabel('Input Diffs')
    plt.ylabel('% of Trials')

    bin_perc_choice_made=np.zeros(10)
    bin_perc_correct=np.zeros(10)
    for i in range(trials):
        bin_idx=-1
        for j in range(10):
            if bins[j] <= input_diffs[i] < bins[j+1]:
                bin_idx=j
                break
        bin_perc_choice_made[bin_idx]+=choice_made[i]
        bin_perc_correct[bin_idx]+=correct_choice[i]
    bin_perc_choice_made=bin_perc_choice_made/hist
    bin_perc_correct=bin_perc_correct/hist

    fig=plt.figure()
    plt.bar(bins[:-1], bin_perc_choice_made, width=bin_width)
    plt.xlabel('Input Diffs')
    plt.ylabel('% Choice Made')

    fig=plt.figure()
    plt.bar(bins[:-1], bin_perc_correct, width=bin_width)
    plt.xlabel('Input Diffs')
    plt.ylabel('% Correct')
    plt.show()


if __name__=='__main__':
    #p_range=np.array(range(1,11))*.01
    #plot_perf_slope_analysis('/media/data/projects/ezrcluster/data/output',p_range,'wta.groups.2.duration.1.000.p_b_e.0.100.p_x_e.0.100',10)
    #prefix='wta.groups.2.duration.3.000.p_b_e.0.050.p_x_e.0.050.p_e_e.0.025.p_e_i.0.030.p_i_i.0.010.p_i_e.0.060.p_dcs.0.0000.i_dcs.0.0000.control'
    #dir='../../data/dcs'
    #t=TrialSeries(dir,prefix,10)
    #t.plot_rt()
#    create_dcs_comparison_report('/home/jbonaiuto/Projects/pySBI/data/dcs',
#        'wta.groups.2.duration.4.000.p_b_e.0.030.p_x_e.0.010.p_e_e.0.030.p_e_i.0.080.p_i_i.0.200.p_i_e.0.080',
#        {'control':(0,0),'anode':(4,-2),'cathode':(-4,2)},50,
#        '/home/jbonaiuto/Projects/pySBI/data/reports/dcs/comparison_4s','')
    #test_fano('../../data/dcs/wta.groups.2.duration.4.000.p_b_e.0.030.p_x_e.0.010.p_e_e.0.030.p_e_i.0.080.p_i_i.0.200.p_i_e.0.080.p_dcs.0.0000.i_dcs.0.0000.control.contrast.0.0000.trial.0.h5', 50*ms)
    #test_fano2('../../data/dcs/test_dcs_fano.h5', 200*ms)
    #test_fano3('../../data/dcs/test_dcs_fano.h5', 500*ms)
    analyze_input_test('../../data/rerw/input_test',20)