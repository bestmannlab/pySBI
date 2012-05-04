import os
from brian.clock import reinit_default_clock, defaultclock
from brian.network import network_operation, Network
from brian.units import second, farad, siemens, volt, amp
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pysbi import wta, voxel
from pysbi.config import DATA_DIR
from pysbi.voxel import Voxel
from pysbi.wta import WTAMonitor

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
        self.input_freq=str(f.attrs['input_freq'])
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
        self.wta_params.w_ampa_e=float(f.attrs['w_ampa_e'])*siemens
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
            try:
                self.voxel_params.T_2E=float(f_vox.attrs['T_2E'])
                self.voxel_params.T_2I=float(f_vox.attrs['T_2I'])
                self.voxel_params.s_e_0=float(f_vox.attrs['s_e_0'])
                self.voxel_params.s_i_0=float(f_vox.attrs['s_i_0'])
                self.voxel_params.B0=float(f_vox.attrs['B0'])
                self.voxel_params.TE=float(f_vox.attrs['TE'])
                self.voxel_params.s_e=float(f_vox.attrs['s_e'])
                self.voxel_params.s_i=float(f_vox.attrs['s_i'])
                self.voxel_params.beta=float(f_vox.attrs['beta'])
                self.voxel_params.k2=float(f_vox.attrs['k2'])
                self.voxel_params.k3=float(f_vox.attrs['k3'])
                self.voxel_rec={'G_total': np.array(f_vox['G_total']),
                                's':       np.array(f_vox['s']),
                                'f_in':    np.array(f_vox['f_in']),
                                'v':       np.array(f_vox['v']),
                                'q':       np.array(f_vox['q']),
                                'y':       np.array(f_vox['y'])}
            except Exception:
                pass

        if 'neuron_state' in f:
            f_state=f['neuron_state']
            self.neural_state_rec={'g_ampa':   np.array(f_state['g_ampa']),
                                   'g_nmda':   np.array(f_state['g_nmda']),
                                   'g_gaba_a': np.array(f_state['g_gaba_a']),
                                   'g_gaba_b': np.array(f_state['g_gaba_b']),
                                   'vm':       np.array(f_state['vm'])}

        if 'firing_rates' in f:
            f_rates=f['firing_rates']
            self.e_firing_rates=np.array(f_rates['e_rates'])
            self.i_firing_rates=np.array(f_rates['i_rates'])

        if 'spikes' in f:
            f_spikes=f['spikes']
            self.e_spike_neurons=[]
            self.e_spike_times=[]
            self.i_spike_neurons=[]
            self.i_spike_times=[]
            for idx in range(self.num_groups):
                self.e_spike_neurons.append(np.array(f_spikes['e.%d.spike_neurons' % idx]))
                self.e_spike_times.append(np.array(f_spikes['e.%d.spike_times' % idx]))
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

def analyze_wta(num_groups, trial_duration, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range,
                p_i_e_range):
    posterior,marginals=run_bayesian_analysis(num_groups, trial_duration, p_b_e_range, p_x_e_range, p_e_e_range,
        p_e_i_range, p_i_i_range, p_i_e_range)
    contrast=run_bold_analysis(num_groups, trial_duration, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range,
        p_i_i_range, p_i_e_range, marginals)


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
                                low_contrast_bold=get_bold_signal(low_contrast_data)

                                high_contrast_data=FileInfo(high_contrast_path)
                                high_contrast_bold=get_bold_signal(high_contrast_data)

                                contrast[i,j,k,l,m,n]=max(high_contrast_bold)-max(low_contrast_bold)
                            except Exception:
                                pass
    return contrast

def get_bold_signal(wta_data, plot=False):
    voxel=Voxel(params=wta_data.voxel_params)
    voxel.G_base=wta_data.voxel_rec['G_total'][0][200:800].mean()
    voxel_monitor = WTAMonitor(None, voxel, record_voxel=True, record_firing_rate=False, record_neuron_state=False,
        record_spikes=False)

    @network_operation(when='start')
    def get_input():
        idx=int(defaultclock.t/defaultclock.dt)
        if idx<len(wta_data.voxel_rec['G_total'][0]):
            voxel.G_total=wta_data.voxel_rec['G_total'][0][idx]
        else:
            voxel.G_total=voxel.G_base

    net=Network(voxel, get_input, voxel_monitor.monitors)
    reinit_default_clock()
    net.run(6*second)

    if plot:
        voxel_monitor.plot()

    return voxel_monitor.voxel_monitor['y'].values


def run_bayesian_analysis(num_groups, trial_duration, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range,
                          p_i_e_range):
    likelihood=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range),
                         len(p_i_i_range), len(p_i_e_range)])
    n_param_vals=len(p_b_e_range)*len(p_x_e_range)*len(p_e_e_range)*len(p_e_i_range)*len(p_i_i_range)*len(p_i_e_range)
    priors=1.0/n_param_vals*np.ones([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range),
                                     len(p_i_i_range), len(p_i_e_range)])
    evidence=0
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
                                high_contrast_data=FileInfo(high_contrast_path)
                                if is_valid(high_contrast_data.e_firing_rates, low_contrast_data.e_firing_rates):
                                    likelihood[i,j,k,l,m,n]=1
                                    evidence+=1/n_param_vals
                            except Exception:
                                print('Error opening files %s and %s' % (low_contrast_path, high_contrast_path))
                                pass


    posterior=(likelihood*priors)/evidence

    marginal_p_e_i=np.zeros([len(p_e_i_range)])
    marginal_p_i_i=np.zeros([len(p_i_i_range)])
    marginal_p_i_e=np.zeros([len(p_i_e_range)])
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for m,p_i_i in enumerate(p_i_i_range):
                        marginal_p_i_e+=posterior[i,j,k,l,m,:]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        marginal_p_i_i+=posterior[i,j,k,l,:,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    for n,p_i_e in enumerate(p_i_e_range):
                        marginal_p_e_i+=posterior[i,j,k,:,m,n]

    marginal_p_e_i_p_i_i=np.zeros([len(p_e_i_range), len(p_i_i_range)])
    marginal_p_e_i_p_i_e=np.zeros([len(p_e_i_range), len(p_i_e_range)])
    marginal_p_i_i_p_i_e=np.zeros([len(p_i_i_range), len(p_i_e_range)])
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for n,p_i_e in enumerate(p_i_e_range):
                    marginal_p_e_i_p_i_i+=posterior[i,j,k,:,:,n]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for m,p_i_i in enumerate(p_i_i_range):
                    marginal_p_e_i_p_i_e+=posterior[i,j,k,:,m,:]
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    marginal_p_i_i_p_i_e+=posterior[i,j,k,l,:,:]

    plt.figure()
    ax=plt.subplot(311)
    ax.bar(np.array(p_i_e_range)-.005,marginal_p_i_e,.01)
    plt.xlabel('p_i_e')
    plt.ylabel('p(WTA)')
    ax=plt.subplot(312)
    ax.bar(np.array(p_i_i_range)-.005,marginal_p_i_i,.01)
    plt.xlabel('p_i_i')
    plt.ylabel('p(WTA)')
    ax=plt.subplot(313)
    ax.bar(np.array(p_e_i_range)-.005,marginal_p_e_i,.01)
    plt.xlabel('p_e_i')
    plt.ylabel('p(WTA)')

    plt.figure()
    ax=plt.subplot(311)
    ax.imshow(marginal_p_e_i_p_i_i, extent=[min(p_i_i_range),max(p_i_i_range),min(p_e_i_range),max(p_e_i_range)])
    plt.xlabel('p_i_i')
    plt.ylabel('p_e_i')
    ax=plt.subplot(312)
    ax.imshow(marginal_p_e_i_p_i_e, extent=[min(p_i_e_range),max(p_i_e_range),min(p_e_i_range),max(p_e_i_range)])
    plt.xlabel('p_i_e')
    plt.ylabel('p_e_i')
    ax=plt.subplot(313)
    ax.imshow(marginal_p_i_i_p_i_e, extent=[min(p_i_i_range),max(p_i_i_range),min(p_i_e_range),max(p_i_e_range)])
    plt.xlabel('p_i_i')
    plt.ylabel('p_e_i')

    plt.show()

    return posterior, {'p_e_i':marginal_p_e_i, 'p_i_i': marginal_p_i_i, 'p_i_e': marginal_p_i_e}
