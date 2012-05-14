import os
from brian.clock import reinit_default_clock, defaultclock
from brian.network import network_operation, Network
from brian.stdunits import Hz
from brian.units import second, farad, siemens, volt, amp
from scipy.signal import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pysbi import wta, voxel
from pysbi.config import DATA_DIR
from pysbi.utils import Struct
from pysbi.voxel import Voxel
from pysbi.wta import WTAMonitor, run_wta

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

        self.num_groups=int(f.attrs['num_groups'])
        self.input_freq=np.array(f.attrs['input_freq'])
        self.trial_duration=float(f.attrs['trial_duration'])*second
        self.background_rate=float(f.attrs['background_rate'])*second
        self.stim_start_time=float(f.attrs['stim_start_time'])*second
        self.stim_end_time=float(f.attrs['stim_end_time'])*second
        self.network_group_size=int(f.attrs['network_group_size'])
        self.background_input_size=int(f.attrs['background_input_size'])
        self.task_input_size=int(f.attrs['task_input_size'])

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
        #self.wta_params.E_gaba_b=float(f.attrs['E_gaba_b'])*volt
        self.wta_params.tau_ampa=float(f.attrs['tau_ampa'])*second
        self.wta_params.tau1_nmda=float(f.attrs['tau1_nmda'])*second
        self.wta_params.tau2_nmda=float(f.attrs['tau2_nmda'])*second
        self.wta_params.tau_gaba_a=float(f.attrs['tau_gaba_a'])*second
        #self.wta_params.tau1_gaba_b=float(f.attrs['tau1_gaba_b'])*second
        #self.wta_params.tau2_gaba_b=float(f.attrs['tau2_gaba_b'])*second
        self.wta_params.w_ampa_x=float(f.attrs['w_ampa_x'])*siemens
        self.wta_params.w_ampa_b=float(f.attrs['w_ampa_b'])*siemens
        self.wta_params.w_ampa_r=float(f.attrs['w_ampa_r'])*siemens
        self.wta_params.w_nmda=float(f.attrs['w_nmda'])*siemens
        self.wta_params.w_gaba_a=float(f.attrs['w_gaba_a'])*siemens
        #self.wta_params.w_gaba_b=float(f.attrs['w_gaba_b'])*siemens
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
            if 'G_total' in f_vox:
                self.voxel_rec['G_total']=np.array(f_vox['G_total'])
            if 'G_total' in f_vox:
                self.voxel_rec['s']=np.array(f_vox['s'])
            if 'G_total' in f_vox:
                self.voxel_rec['f_in']=np.array(f_vox['f_in'])
            if 'G_total' in f_vox:
                self.voxel_rec['v']=np.array(f_vox['v'])
            if 'G_total' in f_vox:
                self.voxel_rec['q']=np.array(f_vox['q'])
            if 'G_total' in f_vox:
                self.voxel_rec['y']=np.array(f_vox['y'])

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
        if 'firing_rates' in f:
            f_rates=f['firing_rates']
            self.e_firing_rates=np.array(f_rates['e_rates'])
            self.i_firing_rates=np.array(f_rates['i_rates'])

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

def run_bold_analysis(num_groups, trial_duration, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range,
                      p_i_e_range,priors):
    contrast=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range), len(p_i_i_range),
                       len(p_i_e_range)])
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for m,p_i_i in enumerate(p_i_i_range):
                        for n,p_i_e in enumerate(p_i_e_range):
                            low_contrast_file='wta.groups.%d.input.low.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.' \
                                              '%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.h5' %\
                                              (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e)
                            low_contrast_path=os.path.join(DATA_DIR,'wta-output',low_contrast_file)
                            high_contrast_file='wta.groups.%d.input.high.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.' \
                                               '%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.h5' %\
                                               (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i,p_i_e)
                            high_contrast_path=os.path.join(DATA_DIR,'wta-output',high_contrast_file)
                            try:
                                low_contrast_data=FileInfo(low_contrast_path)
                                low_contrast_bold=get_bold_signal(low_contrast_data, [500,2500])

                                high_contrast_data=FileInfo(high_contrast_path)
                                high_contrast_bold=get_bold_signal(high_contrast_data, [500,2500])

                                contrast[i,j,k,l,m,n]=max(high_contrast_bold)-max(low_contrast_bold)
                            except Exception:
                                pass
    return contrast

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

def get_bold_signal(g_total, voxel_params, baseline_range, plot=False):
    voxel=Voxel(params=voxel_params)
    voxel.G_base=g_total[baseline_range[0]:baseline_range[1]].mean()
    voxel_monitor = WTAMonitor(None, 1, 1, None, voxel, None, None, record_voxel=True, record_firing_rate=False,
        record_neuron_state=False, record_spikes=False, record_lfp=False, record_inputs=False)

    @network_operation(when='start')
    def get_input():
        idx=int(defaultclock.t/defaultclock.dt)
        if idx<baseline_range[0]:
            voxel.G_total=voxel.G_base
        elif idx<len(g_total):
            voxel.G_total=g_total[idx]
        else:
            voxel.G_total=voxel.G_base

    net=Network(voxel, get_input, voxel_monitor.monitors)
    reinit_default_clock()
    net.run(6*second)

    if plot:
        voxel_monitor.plot()

    return voxel_monitor.voxel_monitor


def run_bayesian_analysis(priors, likelihood, evidence, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range,
                          p_i_e_range):
    bayes_analysis=Struct()
    bayes_analysis.posterior=(likelihood*priors)/evidence

    bayes_analysis.marginal_posterior_p_b_e=np.zeros([len(p_b_e_range)])
    bayes_analysis.marginal_posterior_p_x_e=np.zeros([len(p_x_e_range)])
    bayes_analysis.marginal_posterior_p_e_e=np.zeros([len(p_e_e_range)])
    bayes_analysis.marginal_posterior_p_e_i=np.zeros([len(p_e_i_range)])
    bayes_analysis.marginal_posterior_p_i_i=np.zeros([len(p_i_i_range)])
    bayes_analysis.marginal_posterior_p_i_e=np.zeros([len(p_i_e_range)])
    bayes_analysis.marginal_prior_p_b_e=np.zeros([len(p_b_e_range)])
    bayes_analysis.marginal_prior_p_x_e=np.zeros([len(p_x_e_range)])
    bayes_analysis.marginal_prior_p_e_e=np.zeros([len(p_e_e_range)])
    bayes_analysis.marginal_prior_p_e_i=np.zeros([len(p_e_i_range)])
    bayes_analysis.marginal_prior_p_i_i=np.zeros([len(p_i_i_range)])
    bayes_analysis.marginal_prior_p_i_e=np.zeros([len(p_i_e_range)])
    bayes_analysis.marginal_likelihood_p_b_e=np.zeros([len(p_b_e_range)])
    bayes_analysis.marginal_likelihood_p_x_e=np.zeros([len(p_x_e_range)])
    bayes_analysis.marginal_likelihood_p_e_e=np.zeros([len(p_e_e_range)])
    bayes_analysis.marginal_likelihood_p_e_i=np.zeros([len(p_e_i_range)])
    bayes_analysis.marginal_likelihood_p_i_i=np.zeros([len(p_i_i_range)])
    bayes_analysis.marginal_likelihood_p_i_e=np.zeros([len(p_i_e_range)])
    for j,p_x_e in enumerate(p_x_e_range):
        for k,p_e_e in enumerate(p_e_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        bayes_analysis.marginal_posterior_p_b_e+=bayes_analysis.posterior[:,j,k,l,m,n]
                        bayes_analysis.marginal_prior_p_b_e+=priors[:,j,k,l,m,n]
                        bayes_analysis.marginal_likelihood_p_b_e+=likelihood[:,j,k,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for k,p_e_e in enumerate(p_e_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        bayes_analysis.marginal_posterior_p_x_e+=bayes_analysis.posterior[i,:,k,l,m,n]
                        bayes_analysis.marginal_prior_p_x_e+=priors[i,:,k,l,m,n]
                        bayes_analysis.marginal_likelihood_p_x_e+=likelihood[i,:,k,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for l,p_e_i in enumerate(p_e_i_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        bayes_analysis.marginal_posterior_p_e_e+=bayes_analysis.posterior[i,j,:,l,m,n]
                        bayes_analysis.marginal_prior_p_e_e+=priors[i,j,:,l,m,n]
                        bayes_analysis.marginal_likelihood_p_e_e+=likelihood[i,j,:,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for m,p_i_i in enumerate(p_i_i_range):
                        bayes_analysis.marginal_posterior_p_i_e+=bayes_analysis.posterior[i,j,k,l,m,:]
                        bayes_analysis.marginal_prior_p_i_e+=priors[i,j,k,l,m,:]
                        bayes_analysis.marginal_likelihood_p_i_e+=likelihood[i,j,k,l,m,:]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        bayes_analysis.marginal_posterior_p_i_i+=bayes_analysis.posterior[i,j,k,l,:,n]
                        bayes_analysis.marginal_prior_p_i_i+=priors[i,j,k,l,:,n]
                        bayes_analysis.marginal_likelihood_p_i_i+=likelihood[i,j,k,l,:,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        bayes_analysis.marginal_posterior_p_e_i+=bayes_analysis.posterior[i,j,k,:,m,n]
                        bayes_analysis.marginal_prior_p_e_i+=priors[i,j,k,:,m,n]
                        bayes_analysis.marginal_likelihood_p_e_i+=likelihood[i,j,k,:,m,n]

    bayes_analysis.joint_marginal_posterior_p_b_e_p_x_e=np.zeros([len(p_b_e_range), len(p_x_e_range)])
    bayes_analysis.joint_marginal_posterior_p_e_i_p_i_i=np.zeros([len(p_e_i_range), len(p_i_i_range)])
    bayes_analysis.joint_marginal_posterior_p_e_i_p_i_e=np.zeros([len(p_e_i_range), len(p_i_e_range)])
    bayes_analysis.joint_marginal_posterior_p_i_i_p_i_e=np.zeros([len(p_i_i_range), len(p_i_e_range)])
    for k,p_e_e in enumerate(p_e_e_range):
        for l,p_e_i in enumerate(p_e_i_range):
            for m,p_i_i in enumerate(p_i_i_range):
                for n,p_i_e in enumerate(p_i_e_range):
                    bayes_analysis.joint_marginal_posterior_p_b_e_p_x_e+=bayes_analysis.posterior[:,:,k,l,m,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for n,p_i_e in enumerate(p_i_e_range):
                    bayes_analysis.joint_marginal_posterior_p_e_i_p_i_i+=bayes_analysis.posterior[i,j,k,:,:,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    bayes_analysis.joint_marginal_posterior_p_e_i_p_i_e+=bayes_analysis.posterior[i,j,k,:,m,:]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    bayes_analysis.joint_marginal_posterior_p_i_i_p_i_e+=bayes_analysis.posterior[i,j,k,l,:,:]

    return bayes_analysis


def get_roc_init(num_trials, option_idx, prefix):
    l = []
    p = 0
    n = 0
    for trial in range(num_trials):
        data_path = '%s.trial.%d.h5' % (prefix, trial)
        data = FileInfo(data_path)
        example = 0
        if data.input_freq[option_idx] > data.input_freq[1 - option_idx]:
            example = 1
            p += 1
        else:
            n += 1

        # Get mean rate of pop 1 for last 100ms
        #pop_mean = np.mean(data.e_firing_rates[option_idx, 6500:7500])
        #other_pop_mean = np.mean(data.e_firing_rates[1 - option_idx, 6500:7500])
        pop_sum = np.sum(data.e_firing_rates[option_idx, 6500:7500])
        other_pop_sum = np.sum(data.e_firing_rates[1 - option_idx, 6500:7500])
        #f_score = pop_mean - other_pop_mean
        f_score = pop_sum - other_pop_sum
        l.append((example, f_score))
    l_sorted = sorted(l, key=lambda example: example[1], reverse=True)
    return l_sorted, n, p

def get_auc(prefix, num_trials, num_groups):
    total_auc=0
    total_p=0
    single_auc=[]
    single_p=[]
    for i in range(num_groups):
        l_sorted, n, p = get_roc_init(num_trials, i, prefix)
        single_auc.append(get_auc_single_option(prefix, num_trials, i))
        single_p.append(p)
        total_p+=p
    for i in range(num_groups):
        total_auc+=float(single_auc[i])*(float(single_p[i])/float(total_p))

    return total_auc

def get_auc_single_option(prefix, num_trials, option_idx):
    l_sorted, n, p = get_roc_init(num_trials, option_idx, prefix)
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

def get_roc_single_option(prefix, num_trials, option_idx):

    l_sorted, n, p = get_roc_init(num_trials, option_idx, prefix)

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

def get_roc(prefix, num_trials):
    roc1=get_roc_single_option(prefix, num_trials, 0)
    roc2=get_roc_single_option(prefix, num_trials, 1)

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
    for i in range(num_trials):
        input=np.zeros(2)
        input[0]=np.random.rand()*40
        input[1]=40-input[0]
        input+=10
        inputs.append(input)

    for i,input in enumerate(inputs):
        run_wta(wta_params, 2, input*Hz, 1*second, record_lfp=True, record_voxel=True,
            record_neuron_state=True, record_spikes=True, record_firing_rate=True, plot_output=False,
            output_file=os.path.join(DATA_DIR,'%s.trial.%d.h5' % (prefix, i)))
