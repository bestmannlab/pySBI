import os
from brian.stdunits import Hz, ms
from brian.units import second, farad, siemens, volt, amp
from scipy.signal import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pysbi import wta, voxel
from pysbi.config import DATA_DIR
from pysbi.util.utils import Struct
from pysbi.wta import  run_wta

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
    return num_groups,input_pattern,duration,p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e

class FileInfo():
    def __init__(self, file_name):
        self.file_name=file_name
        f = h5py.File(file_name)

        self.single_inh_pop=0
        if 'single_inh_pop' in f.attrs:
            self.single_inh_pop=int(f.attrs['single_inh_pop'])
        self.num_groups=int(f.attrs['num_groups'])
        self.input_freq=np.array(f.attrs['input_freq'])
        self.trial_duration=float(f.attrs['trial_duration'])*second
        self.background_rate=float(f.attrs['background_rate'])*second
        self.stim_start_time=float(f.attrs['stim_start_time'])*second
        self.stim_end_time=float(f.attrs['stim_end_time'])*second
        self.network_group_size=int(f.attrs['network_group_size'])
        self.background_input_size=int(f.attrs['background_input_size'])
        self.task_input_size=int(f.attrs['task_input_size'])
        self.muscimol_amount=float(f.attrs['muscimol_amount'])*siemens
        self.injection_site=int(f.attrs['injection_site'])

        self.wta_params=wta.default_params()
        self.wta_params.C=float(f.attrs['C'])*farad
        self.wta_params.gL=float(f.attrs['gL'])*siemens
        self.wta_params.EL=float(f.attrs['EL'])*volt
        self.wta_params.VT=float(f.attrs['VT'])*volt
        self.wta_params.DeltaT=float(f.attrs['DeltaT'])*volt
        if 'Mg' in f.attrs:
            self.wta_params.Mg=float(f.attrs['Mg'])
        self.wta_params.E_ampa=float(f.attrs['E_ampa'])*volt
        self.wta_params.E_nmda=float(f.attrs['E_nmda'])*volt
        self.wta_params.E_gaba_a=float(f.attrs['E_gaba_a'])*volt
        if 'E_gaba_b' in f.attrs:
            self.wta_params.E_gaba_b=float(f.attrs['E_gaba_b'])*volt
        self.wta_params.tau_ampa=float(f.attrs['tau_ampa'])*second
        self.wta_params.tau1_nmda=float(f.attrs['tau1_nmda'])*second
        self.wta_params.tau2_nmda=float(f.attrs['tau2_nmda'])*second
        self.wta_params.tau_gaba_a=float(f.attrs['tau_gaba_a'])*second
        self.wta_params.tau1_gaba_b=float(f.attrs['tau1_gaba_b'])*second
        self.wta_params.tau2_gaba_b=float(f.attrs['tau2_gaba_b'])*second
        self.wta_params.w_ampa_min=float(f.attrs['w_ampa_min'])*siemens
        self.wta_params.w_ampa_max=float(f.attrs['w_ampa_max'])*siemens
        self.wta_params.w_nmda_min=float(f.attrs['w_nmda_min'])*siemens
        self.wta_params.w_nmda_max=float(f.attrs['w_nmda_max'])*siemens
        self.wta_params.w_gaba_a_min=float(f.attrs['w_gaba_a_min'])*siemens
        self.wta_params.w_gaba_a_max=float(f.attrs['w_gaba_a_max'])*siemens
        self.wta_params.w_gaba_b_min=float(f.attrs['w_gaba_b_min'])*siemens
        self.wta_params.w_gaba_b_max=float(f.attrs['w_gaba_b_max'])*siemens
        self.wta_params.p_b_e=float(f.attrs['p_b_e'])
        self.wta_params.p_x_e=float(f.attrs['p_x_e'])
        self.wta_params.p_e_e=float(f.attrs['p_e_e'])
        self.wta_params.p_e_i=float(f.attrs['p_e_i'])
        self.wta_params.p_i_i=float(f.attrs['p_i_i'])
        self.wta_params.p_i_e=float(f.attrs['p_i_e'])

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
        if 'firing_rates' in f:
            f_rates=f['firing_rates']
            self.e_firing_rates=np.array(f_rates['e_rates'])
            self.i_firing_rates=np.array(f_rates['i_rates'])
            self.rt=get_response_time(self.e_firing_rates, self.stim_start_time)

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
                          p_i_i_range, p_x_e_range, perf_threshold=0.9, r_sqr_threshold=0.2):
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
                            bayes_analysis.l1_dist.append(slope[i,j,k,l,m,n])
                            if auc[i, j, k, l, m, n] >= perf_threshold:
                                bayes_analysis.l1_pos_dist.append(slope[i,j,k,l,m,n])
                                bayes_analysis.l1_pos_likelihood[i, j, k, l, m, n] = 1.0
                            else:
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

    bayes_analysis.l1_pos_l2_pos_posterior = 0.0
    if bayes_analysis.l1_pos_l2_pos_evidence>0:
        bayes_analysis.l1_pos_l2_pos_posterior = (bayes_analysis.l1_pos_l2_pos_likelihood * bayes_analysis.l1_pos_l2_priors) / bayes_analysis.l1_pos_l2_pos_evidence
    bayes_analysis.l1_pos_l2_neg_posterior = 0.0
    if bayes_analysis.l1_pos_l2_neg_evidence>0:
        bayes_analysis.l1_pos_l2_neg_posterior = (bayes_analysis.l1_pos_l2_neg_likelihood * bayes_analysis.l1_pos_l2_priors) / bayes_analysis.l1_pos_l2_neg_evidence
    bayes_analysis.l1_pos_l2_zero_posterior = 0.0
    if bayes_analysis.l1_pos_l2_zero_evidence>0:
        bayes_analysis.l1_pos_l2_zero_posterior = (bayes_analysis.l1_pos_l2_zero_likelihood * bayes_analysis.l1_pos_l2_priors) / bayes_analysis.l1_pos_l2_zero_evidence
    bayes_analysis.l1_neg_l2_pos_posterior = 0.0
    if bayes_analysis.l1_neg_l2_pos_evidence>0:
        bayes_analysis.l1_neg_l2_pos_posterior = (bayes_analysis.l1_neg_l2_pos_likelihood * bayes_analysis.l1_neg_l2_priors) / bayes_analysis.l1_neg_l2_pos_evidence
    bayes_analysis.l1_neg_l2_neg_posterior = 0.0
    if bayes_analysis.l1_neg_l2_neg_evidence>0:
        bayes_analysis.l1_neg_l2_neg_posterior = (bayes_analysis.l1_neg_l2_neg_likelihood * bayes_analysis.l1_neg_l2_priors) / bayes_analysis.l1_neg_l2_neg_evidence
    bayes_analysis.l1_neg_l2_zero_posterior = 0.0
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

def get_response_time(e_firing_rates, stim_start_time):
    rate_1=e_firing_rates[0]
    rate_2=e_firing_rates[1]
    times=np.array(range(len(rate_1)))*.1
    rt=None
    winner=None
    for idx,time in enumerate(times):
        time=time*second
        if time>stim_start_time:
            if rt is None:
                if rate_1[idx]>=2*rate_2[idx]:
                    winner=1
                    rt=time-stim_start_time
                if rate_2[idx]>=2*rate_1[idx]:
                    winner=2
                    rt=time-stim_start_time
            else:
                if (winner==1 and rate_1[idx]<2*rate_2[idx]) or (winner==2 and rate_2[idx]<2*rate_1[idx]):
                    winner=None
                    rt=None
    return rt


def get_roc_init(contrast_range, num_trials, num_extra_trials, option_idx, prefix):
    l = []
    p = 0
    n = 0
    for contrast in contrast_range:
        for trial in range(num_trials):
            data_path = '%s.contrast.%0.4f.trial.%d.h5' % (prefix, contrast, trial)
            data = FileInfo(data_path)
            example = 0
            if data.input_freq[option_idx] > data.input_freq[1 - option_idx]:
                example = 1
                for j in range(num_extra_trials):
                    p += 1
            else:
                for j in range(num_extra_trials):
                    n += 1

            # Get mean rate of pop 1 for last 100ms
            pop_mean = np.mean(data.e_firing_rates[option_idx, 6500:7500])
            other_pop_mean = np.mean(data.e_firing_rates[1 - option_idx, 6500:7500])
            for j in range(num_extra_trials):
                f_score=.25*np.random.randn()
                if float(pop_mean+other_pop_mean):
                    f_score+=float(pop_mean)/float(pop_mean+other_pop_mean)
                l.append((example, f_score))
    l_sorted = sorted(l, key=lambda example: example[1], reverse=True)
    return l_sorted, n, p

def get_auc(prefix, contrast_range, num_trials, num_extra_trials, num_groups):
    total_auc=0
    total_p=0
    single_auc=[]
    single_p=[]
    for i in range(num_groups):
        l_sorted, n, p = get_roc_init(contrast_range, num_trials, num_extra_trials, i, prefix)
        single_auc.append(get_auc_single_option(prefix, contrast_range, num_trials, num_extra_trials, i))
        single_p.append(p)
        total_p+=p
    for i in range(num_groups):
        total_auc+=float(single_auc[i])*(float(single_p[i])/float(total_p))

    return total_auc

def get_auc_single_option(prefix, contrast_range, num_trials, num_extra_trials, option_idx):
    l_sorted, n, p = get_roc_init(contrast_range, num_trials, num_extra_trials, option_idx, prefix)
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
    a=float(a)/(float(p)*float(n))
    return a

def get_roc_single_option(prefix, contrast_range, num_trials, num_extra_trials, option_idx):

    l_sorted, n, p = get_roc_init(contrast_range, num_trials, num_extra_trials, option_idx, prefix)

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

def get_roc(prefix, num_trials, num_extra_trials):
    roc1=get_roc_single_option(prefix, num_trials, num_extra_trials, 0)
    roc2=get_roc_single_option(prefix, num_trials, num_extra_trials, 1)

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
