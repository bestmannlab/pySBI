from brian import *
from brian.library.IF import *
from brian.library.synapses import *
from time import time
import h5py
import argparse
from pysbi.voxel import Voxel, LFPSource

default_params=Parameters(
    # Neuron parameters
    C = 200 * pF,
    gL = 20 * nS,
    EL = -70 * mV,
    VT = -55 * mV,
    DeltaT = 3 * mV,
    # Magnesium concentration
    Mg = 1,
    # Synapse parameters
    E_ampa = 0 * mvolt,
    E_nmda = 0 * mvolt,
    E_gaba_a = -70 * mvolt,
    #E_gaba_b = -95 * mvolt,
    # value from  (Hestrin, Sah, & Nicoll, 1990; Sah, Hestrin, & Nicoll, 1990; Spruston, Jonas, & Sakmann, 1995; Angulo, Rossier, & Audinat, 1999)
    tau_ampa = 2*ms,
    # value from (Hestrin et al., 1990)
    tau1_nmda = 14*ms,
    # value from (Hestrin et al., 1990; Sah et al., 1990)
    tau2_nmda = 100*ms,
    # value from (Salin & Prince, 1996; Xiang, Huguenard, & Prince, 1998; Gupta, Wang, & Markram, 2000)
    tau_gaba_a = 7.5*ms,
    #tau1_gaba_b = 10*ms,
    #tau2_gaba_b =100*ms,
    w_ampa_b = 2.0 * nS,
    w_ampa_x = 2.0 * nS,
    w_ampa_r=1.0*nS,
    w_nmda=0.01*nS,
    w_gaba_a=3.0*nS,
    #w_gaba_b=0.1*nS,
    # Connection probabilities
    p_b_e=0.05,
    p_x_e=0.05,
    p_e_e=0.01,
    p_e_i=0.05,
    p_i_i=0.01,
    p_i_e=0.01)

single_inh_pop_params=Parameters(
    # Neuron parameters
    C = 200 * pF,
    gL = 20 * nS,
    EL = -70 * mV,
    VT = -55 * mV,
    DeltaT = 3 * mV,
    # Magnesium concentration
    Mg = 1,
    # Synapse parameters
    E_ampa = 0 * mvolt,
    E_nmda = 0 * mvolt,
    E_gaba_a = -70 * mvolt,
    #E_gaba_b = -95 * mvolt,
    # value from  (Hestrin, Sah, & Nicoll, 1990; Sah, Hestrin, & Nicoll, 1990; Spruston, Jonas, & Sakmann, 1995; Angulo, Rossier, & Audinat, 1999)
    tau_ampa = 2*ms,
    # value from (Hestrin et al., 1990)
    tau1_nmda = 14*ms,
    # value from (Hestrin et al., 1990; Sah et al., 1990)
    tau2_nmda = 100*ms,
    # value from (Salin & Prince, 1996; Xiang, Huguenard, & Prince, 1998; Gupta, Wang, & Markram, 2000)
    tau_gaba_a = 7.5*ms,
    #tau1_gaba_b = 10*ms,
    #tau2_gaba_b =100*ms,
    w_ampa_b = 1.0 * nS,
    w_ampa_x = 1.0 * nS,
    w_ampa_r=1.0*nS,
    w_nmda=0.01*nS,
    w_gaba_a=1.0*nS,
    #w_gaba_b=0.1*nS,
    # Connection probabilities
    p_b_e=0.04,
    p_x_e=0.05,
    p_e_e=0.01,
    p_e_i=0.1,
    p_i_i=0.0,
    p_i_e=0.03)

class WTANetworkGroup(NeuronGroup):
    def __init__(self, N, num_groups, params=default_params, background_input=None, task_inputs=None, single_inh_pop=False):
        self.num_groups=num_groups
        self.params=params

        # Exponential integrate-and-fire neuron
        eqs = exp_IF(params.C, params.gL, params.EL, params.VT, params.DeltaT)

        # AMPA conductance - recurrent input current
        eqs += exp_synapse('g_ampa_r', params.tau_ampa, siemens)
        eqs += Current('I_ampa_r=g_ampa_r*(E-vm): amp', E=params.E_ampa)

        # AMPA conductance - background input current
        eqs += exp_synapse('g_ampa_b', params.tau_ampa, siemens)
        eqs += Current('I_ampa_b=g_ampa_b*(E-vm): amp', E=params.E_ampa)

        # AMPA conductance - task input current
        eqs += exp_synapse('g_ampa_x', params.tau_ampa, siemens)
        eqs += Current('I_ampa_x=g_ampa_x*(E-vm): amp', E=params.E_ampa)

        # Voltage-dependent NMDA conductance
        #eqs += biexp_conductance('g_nmda', E=params.E_nmda, tau1=params.tau1_nmda, tau2=params.tau2_nmda)
        eqs += biexp_synapse('g_nmda', params.tau1_nmda, params.tau2_nmda, siemens)
        eqs += Equations('dg_V/dt =  (-g_V + 1/(1+(Mg/3.57)*exp(-0.062 *vm/mV)))/second : 1 ', Mg=params.Mg)
        eqs += Current('I_nmda=g_V*g_nmda*(E-vm): amp', E=params.E_nmda)

        # GABA-A conductance
        #eqs += exp_conductance('g_gaba_a', E=params.E_gaba_a, tau=params.tau_gaba_a)
        eqs += exp_synapse('g_gaba_a', params.tau_gaba_a, siemens)
        eqs += Current('I_gaba_a=g_gaba_a*(E-vm): amp', E=params.E_gaba_a)

        # GABA-B conductance
        #eqs += biexp_conductance('g_gaba_b', E=params.E_gaba_b, tau1=params.tau1_gaba_b, tau2=params.tau2_gaba_b)

        # Total synaptic conductance
        eqs += Equations('g_syn=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda+g_gaba_a : siemens', Mg=params.Mg)
        # Total synaptic current
        eqs += Equations('I_abs=abs(I_ampa_r)+abs(I_ampa_b)+abs(I_ampa_x)+abs(I_nmda)+abs(I_gaba_a) : amp')

        NeuronGroup.__init__(self, N*num_groups, model=eqs, threshold=-20*mV, reset=params.EL, compile=True)

        self.init_subpopulations(N, single_inh_pop)

        self.init_connectivity(single_inh_pop, background_input, task_inputs)

    def init_subpopulations(self, N, single_inh_pop):
        self.groups_e=[]
        self.groups_i=[]

        for i in range(self.num_groups):
            group=None
            if not single_inh_pop:
                group=self.subgroup(N)
                group_e=group.subgroup(int(4*N/5))
            else:
                group_e=self.subgroup(int(4*N/5))
                # regular spiking params (from Naud et al., 2008)
            group_e.C=104*pF
            group_e.gL=4.3*nS
            group_e.EL=-65*mV
            group_e.VT=-52*mV
            group_e.DeltaT=0.8*mV
            self.groups_e.append(group_e)

            if not single_inh_pop:
                group_i=group.subgroup(int(N/5))
                # fast-spiking interneuron params (from Naud et al., 2008)
                group_i.C=59*pF
                group_i.gL=2.9*nS
                group_i.EL=-62*mV
                group_i.VT=-42*mV
                group_i.DeltaT=3.0*mV
                self.groups_i.append(group_i)

        if single_inh_pop:
            N_inh=num_groups*int(N/5)
            group_i=self.subgroup(N_inh)
            # fast-spiking interneuron params (from Naud et al., 2008)
            group_i.C=59*pF
            group_i.gL=2.9*nS
            group_i.EL=-62*mV
            group_i.VT=-42*mV
            group_i.DeltaT=3.0*mV
            self.groups_i.append(group_i)

        self.vm = self.params.EL+randn(N*self.num_groups)*10*mV
        self.g_ampa_r = randn(N*self.num_groups)*self.params.w_ampa_r*.1
        self.g_ampa_b = randn(N*self.num_groups)*self.params.w_ampa_b*.1
        self.g_ampa_x = randn(N*self.num_groups)*self.params.w_ampa_x*.1
        self.g_nmda = randn(N*self.num_groups)*self.params.w_nmda*.1
        self.g_gaba_a = randn(N*self.num_groups)*self.params.w_gaba_a*.1


    def init_connectivity(self, single_inh_pop, background_input, task_inputs):
        self.connections=[]
        for i in range(self.num_groups):
            # E population - recurrent connections
            self.connections.append(DelayConnection(self.groups_e[i], self.groups_e[i], 'g_ampa_r',
                sparseness=self.params.p_e_e, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms)))
            self.connections.append(DelayConnection(self.groups_e[i], self.groups_e[i], 'g_nmda',
                sparseness=self.params.p_e_e, weight=self.params.w_nmda, delay=(0*ms, 5*ms)))

            if not single_inh_pop:
                # I population - recurrent connections
                self.connections.append(DelayConnection(self.groups_i[i], self.groups_i[i], 'g_gaba_a',
                    sparseness=self.params.p_i_i, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms)))
                #self.connections.append(DelayConnection(self.input_groups_i[i], self.input_groups_i[i], 'g_gaba_b',
                #    sparseness=params.p_i_i, weight=params.w_gaba_b, delay=(0*ms, 5*ms)))

                # E -> I excitatory connections
                self.connections.append(DelayConnection(self.groups_e[i], self.groups_i[i], 'g_ampa_r',
                    sparseness=self.params.p_e_i, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms)))
                self.connections.append(DelayConnection(self.groups_e[i], self.groups_i[i], 'g_nmda',
                    sparseness=self.params.p_e_i, weight=self.params.w_nmda, delay=(0*ms, 5*ms)))

                # I -> E - inhibitory connections
                for j in range(self.num_groups):
                    if not i==j:
                        self.connections.append(DelayConnection(self.groups_i[i], self.groups_e[j], 'g_gaba_a',
                            sparseness=self.params.p_i_e, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms)))
                        #self.connections.append(DelayConnection(self.input_groups_i[i], self.input_groups_e[j], 'g_gaba_b',
                        #    sparseness=params.p_i_e, weight=params.w_gaba_b, delay=(0*ms, 5*ms)))

            else:
                # E -> I excitatory connections
                self.connections.append(DelayConnection(self.groups_e[i], self.groups_i[0], 'g_ampa_r',
                    sparseness=self.params.p_e_i, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms)))
                self.connections.append(DelayConnection(self.groups_e[i], self.groups_i[0], 'g_nmda',
                    sparseness=self.params.p_e_i, weight=self.params.w_nmda, delay=(0*ms, 5*ms)))

                # I -> E - inhibitory connections
                self.connections.append(DelayConnection(self.groups_i[0], self.groups_e[i], 'g_gaba_a',
                    sparseness=self.params.p_i_e, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms)))

        if single_inh_pop:
            # I population - recurrent connections
            self.connections.append(DelayConnection(self.groups_i[0], self.groups_i[0], 'g_gaba_a',
                sparseness=self.params.p_i_i, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms)))
            #self.connections.append(DelayConnection(self.input_groups_i[0], self.input_groups_i[0], 'g_gaba_b',
            #    sparseness=params.p_i_i, weight=params.w_gaba_b, delay=(0*ms, 5*ms)))

        if background_input is not None:
            # Background -> E+I population connectinos
            self.connections.append(DelayConnection(background_input, self, 'g_ampa_b', sparseness=self.params.p_b_e,
                weight=self.params.w_ampa_b, delay=(0*ms, 5*ms)))

        if task_inputs is not None:
            # Task input -> E population connections
            for i in range(self.num_groups):
                self.connections.append(DelayConnection(task_inputs[i], self.groups_e[i], 'g_ampa_x',
                    sparseness=self.params.p_x_e, weight=self.params.w_ampa_x, delay=(0*ms, 5*ms)))

class WTAMonitor():
    def __init__(self, network, N, num_groups, lfp_source, voxel, record_lfp=True, record_voxel=True, record_neuron_state=False,
                 record_spikes=True, record_firing_rate=True):
        self.num_groups=num_groups
        self.N=N
        self.monitors=[]

        # LFP monitor
        if record_lfp:
            self.lfp_monitor = StateMonitor(lfp_source, 'LFP', record=0)
            self.monitors.append(self.lfp_monitor)
        else:
            self.lfp_monitor=None

        # Voxel monitor
        if record_voxel:
            self.voxel_monitor = MultiStateMonitor(voxel, vars=['G_total','s','f_in','v','f_out','q','y'], record=True)
            self.monitors.append(self.voxel_monitor)
        else:
            self.voxel_monitor=None

        # Network monitor
        if record_neuron_state:
            self.record_idx=[]
            for i in range(num_groups):
                e_idx=i*N
                i_idx=i*N+N-1
                self.record_idx.extend([e_idx, i_idx])
            self.network_monitor = MultiStateMonitor(network, vars=['vm','g_ampa_r','g_ampa_x','g_ampa_b','g_gaba_a','g_nmda'],#,'g_gaba_b'],
                record=self.record_idx)
            self.monitors.append(self.network_monitor)
        else:
            self.network_monitor=None

        # Population rate monitors
        if record_firing_rate:
            self.population_rate_monitors={'excitatory':[], 'inhibitory':[]}
            for group_e in network.groups_e:
                e_rate_monitor=PopulationRateMonitor(group_e)
                self.population_rate_monitors['excitatory'].append(e_rate_monitor)
                self.monitors.append(e_rate_monitor)

            for group_i in network.groups_i:
                i_rate_monitor=PopulationRateMonitor(group_i)
                self.population_rate_monitors['inhibitory'].append(i_rate_monitor)
                self.monitors.append(i_rate_monitor)
        else:
            self.population_rate_monitors=None

        # Spike monitors
        if record_spikes:
            self.spike_monitors={'excitatory':[], 'inhibitory':[]}
            for group_e in network.groups_e:
                e_spike_monitor=SpikeMonitor(group_e)
                self.spike_monitors['excitatory'].append(e_spike_monitor)
                self.monitors.append(e_spike_monitor)

            for group_i in network.groups_i:
                i_spike_monitor=SpikeMonitor(group_i)
                self.spike_monitors['inhibitory'].append(i_spike_monitor)
                self.monitors.append(i_spike_monitor)
        else:
            self.spike_monitors=None

    def plot(self):
        if self.spike_monitors is not None:
            figure()
            subplot(211)
            raster_plot(*self.spike_monitors['excitatory'],newfigure=False)
            subplot(212)
            raster_plot(*self.spike_monitors['inhibitory'],newfigure=False)
        if self.population_rate_monitors is not None:
            figure()
            ax=subplot(211)
            for pop_rate_monitor in self.population_rate_monitors['excitatory']:
                ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz)
            ax=subplot(212)
            for pop_rate_monitor in self.population_rate_monitors['inhibitory']:
                ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz)
        if self.network_monitor is not None:
            figure()
            for i in range(self.num_groups):
                ax=subplot(self.num_groups*100+20+(i*2+1))
                ax.plot(self.network_monitor['g_ampa_r'].times/ms, self.network_monitor['g_ampa_r'][self.record_idx[i*2]]/nA, label='AMPA-recurrent')
                ax.plot(self.network_monitor['g_ampa_x'].times/ms, self.network_monitor['g_ampa_x'][self.record_idx[i*2]]/nA, label='AMPA-task')
                ax.plot(self.network_monitor['g_ampa_b'].times/ms, self.network_monitor['g_ampa_b'][self.record_idx[i*2]]/nA, label='AMPA-backgrnd')
                ax.plot(self.network_monitor['g_nmda'].times/ms, self.network_monitor['g_nmda'][self.record_idx[i*2]]/nA, label='NMDA')
                ax.plot(self.network_monitor['g_gaba_a'].times/ms, self.network_monitor['g_gaba_a'][self.record_idx[i*2]]/nA, label='GABA_A')
                #ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][0]/nA, label='GABA_B')
                xlabel('Time (ms)')
                ylabel('Conductance (nA)')
                legend()
                ax=subplot(self.num_groups*100+20+(i*2+2))
                ax.plot(self.network_monitor['g_ampa_r'].times/ms, self.network_monitor['g_ampa_r'][self.record_idx[i*2+1]]/nA, label='AMPA-recurrent')
                ax.plot(self.network_monitor['g_ampa_x'].times/ms, self.network_monitor['g_ampa_x'][self.record_idx[i*2+1]]/nA, label='AMPA-task')
                ax.plot(self.network_monitor['g_ampa_b'].times/ms, self.network_monitor['g_ampa_b'][self.record_idx[i*2+1]]/nA, label='AMPA-backgrnd')
                ax.plot(self.network_monitor['g_nmda'].times/ms, self.network_monitor['g_nmda'][self.record_idx[i*2+1]]/nA, label='NMDA')
                ax.plot(self.network_monitor['g_gaba_a'].times/ms, self.network_monitor['g_gaba_a'][self.record_idx[i*2+1]]/nA, label='GABA_A')
                #ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][0]/nA, label='GABA_B')
                xlabel('Time (ms)')
                ylabel('Conductance (nA)')
                legend()
        if self.lfp_monitor is not None:
            fig=figure()
            ax=subplot(111)
            ax.plot(self.lfp_monitor.times / ms, self.lfp_monitor[0] / mA)
            xlabel('Time (ms)')
            ylabel('LFP (mA)')
        if self.voxel_monitor is not None:
            figure()
            ax=subplot(211)
            ax.plot(self.voxel_monitor['G_total'].times / ms, self.voxel_monitor['G_total'][0] / nA)
            xlabel('Time (ms)')
            ylabel('Total Synaptic Activity (nA)')
            ax=subplot(212)
            ax.plot(self.voxel_monitor['y'].times / ms, self.voxel_monitor['y'][0])
            xlabel('Time (ms)')
            ylabel('BOLD')
        show()


def write_output(background_input_size, background_rate, input_freq, network_group_size, num_groups, output_file,
                 record_firing_rate, record_neuron_state, record_spikes, record_voxel, record_lfp, stim_end_time,
                 stim_start_time, task_input_size, trial_duration, voxel, wta_monitor, wta_params):
    f = h5py.File(output_file, 'w')
    f.attrs['num_groups'] = num_groups
    f.attrs['input_freq'] = input_freq
    f.attrs['trial_duration'] = trial_duration
    f.attrs['background_rate'] = background_rate
    f.attrs['stim_start_time'] = stim_start_time
    f.attrs['stim_end_time'] = stim_end_time
    f.attrs['network_group_size'] = network_group_size
    f.attrs['background_input_size'] = background_input_size
    f.attrs['task_input_size'] = task_input_size
    f.attrs['C'] = wta_params.C
    f.attrs['gL'] = wta_params.gL
    f.attrs['EL'] = wta_params.EL
    f.attrs['VT'] = wta_params.VT
    f.attrs['Mg'] = wta_params.Mg
    f.attrs['DeltaT'] = wta_params.DeltaT
    f.attrs['E_ampa'] = wta_params.E_ampa
    f.attrs['E_nmda'] = wta_params.E_nmda
    f.attrs['E_gaba_a'] = wta_params.E_gaba_a
    #f.attrs['E_gaba_b'] = wta_params.E_gaba_b
    f.attrs['tau_ampa'] = wta_params.tau_ampa
    f.attrs['tau1_nmda'] = wta_params.tau1_nmda
    f.attrs['tau2_nmda'] = wta_params.tau2_nmda
    f.attrs['tau_gaba_a'] = wta_params.tau_gaba_a
    #f.attrs['tau1_gaba_b'] = wta_params.tau1_gaba_b
    #f.attrs['tau2_gaba_b'] = wta_params.tau2_gaba_b
    f.attrs['w_ampa_b'] = wta_params.w_ampa_b
    f.attrs['w_ampa_x'] = wta_params.w_ampa_x
    f.attrs['w_ampa_r'] = wta_params.w_ampa_r
    f.attrs['w_nmda'] = wta_params.w_nmda
    f.attrs['w_gaba_a'] = wta_params.w_gaba_a
    #f.attrs['w_gaba_b'] = wta_params.w_gaba_b
    f.attrs['p_b_e'] = wta_params.p_b_e
    f.attrs['p_x_e'] = wta_params.p_x_e
    f.attrs['p_e_e'] = wta_params.p_e_e
    f.attrs['p_e_i'] = wta_params.p_e_i
    f.attrs['p_i_i'] = wta_params.p_i_i
    f.attrs['p_i_e'] = wta_params.p_i_e
    if record_lfp:
        f_lfp = f.create_group('lfp')
        f_lfp['lfp']=wta_monitor.lfp_monitor.values
    if record_voxel:
        f_vox = f.create_group('voxel')
        f_vox.attrs['eta'] = voxel.eta
        f_vox.attrs['G_base'] = voxel.G_base
        f_vox.attrs['tau_f'] = voxel.tau_f
        f_vox.attrs['tau_s'] = voxel.tau_s
        f_vox.attrs['tau_o'] = voxel.tau_o
        f_vox.attrs['e_base'] = voxel.e_base
        f_vox.attrs['v_base'] = voxel.v_base
        f_vox.attrs['alpha'] = voxel.alpha
        f_vox.attrs['T_2E'] = voxel.params.T_2E
        f_vox.attrs['T_2I'] = voxel.params.T_2I
        f_vox.attrs['s_e_0'] = voxel.params.s_e_0
        f_vox.attrs['s_i_0'] = voxel.params.s_i_0
        f_vox.attrs['B0'] = voxel.params.B0
        f_vox.attrs['TE'] = voxel.params.TE
        f_vox.attrs['s_e'] = voxel.params.s_e
        f_vox.attrs['s_i'] = voxel.params.s_i
        f_vox.attrs['beta'] = voxel.params.beta
        f_vox.attrs['k2'] = voxel.k2
        f_vox.attrs['k3'] = voxel.k3
        f_vox['G_total'] = wta_monitor.voxel_monitor['G_total'].values
        f_vox['s'] = wta_monitor.voxel_monitor['s'].values
        f_vox['f_in'] = wta_monitor.voxel_monitor['f_in'].values
        f_vox['v'] = wta_monitor.voxel_monitor['v'].values
        f_vox['q'] = wta_monitor.voxel_monitor['q'].values
        f_vox['y'] = wta_monitor.voxel_monitor['y'].values
    if record_neuron_state:
        f_state = f.create_group('neuron_state')
        f_state['g_ampa_r'] = wta_monitor.network_monitor['g_ampa_r'].values
        f_state['g_ampa_x'] = wta_monitor.network_monitor['g_ampa_x'].values
        f_state['g_ampa_b'] = wta_monitor.network_monitor['g_ampa_b'].values
        f_state['g_nmda'] = wta_monitor.network_monitor['g_nmda'].values
        f_state['g_gaba_a'] = wta_monitor.network_monitor['g_gaba_a'].values
        #f_state['g_gaba_b'] = wta_monitor.network_monitor['g_gaba_b'].values
        f_state['vm'] = wta_monitor.network_monitor['vm'].values
    if record_firing_rate:
        f_rates = f.create_group('firing_rates')
        e_rates = []
        for rate_monitor in wta_monitor.population_rate_monitors['excitatory']:
            e_rates.append(rate_monitor.smooth_rate(width=5 * ms, filter='gaussian'))
        f_rates['e_rates'] = np.array(e_rates)

        i_rates = []
        for rate_monitor in wta_monitor.population_rate_monitors['inhibitory']:
            i_rates.append(rate_monitor.smooth_rate(width=5 * ms, filter='gaussian'))
        f_rates['i_rates'] = np.array(i_rates)
    if record_spikes:
        f_spikes = f.create_group('spikes')
        for idx, spike_monitor in enumerate(wta_monitor.spike_monitors['excitatory']):
            if len(spike_monitor.spikes):
                f_spikes['e.%d.spike_neurons' % idx] = np.array([s[0] for s in spike_monitor.spikes])
                f_spikes['e.%d.spike_times' % idx] = np.array([s[1] for s in spike_monitor.spikes])

        for idx, spike_monitor in enumerate(wta_monitor.spike_monitors['inhibitory']):
            if len(spike_monitor.spikes):
                f_spikes['i.%d.spike_neurons' % idx] = np.array([s[0] for s in spike_monitor.spikes])
                f_spikes['i.%d.spike_times' % idx] = np.array([s[1] for s in spike_monitor.spikes])
    f.close()


def run_wta(wta_params, num_groups, input_freq, trial_duration, output_prefix=None, record_lfp=True, record_voxel=True,
            record_neuron_state=False, record_spikes=True, record_firing_rate=True, plot_output=False,
            single_inh_pop=False, num_trials=1):
    background_rate=10*Hz
    #stim_start_time=1*second
    stim_start_time=.25*second
    #stim_end_time=1.5*second
    stim_end_time=.75*second
    network_group_size=2000
    background_input_size=1000
    task_input_size=1000

    # Create network inputs
    background_input=PoissonGroup(background_input_size, rates=background_rate)
    task_inputs=[PoissonGroup(task_input_size, rates=lambda t: (stim_start_time<t<stim_end_time and input_freq[i]) or
                                                                background_rate) for i in range(num_groups)]

    # Create network
    wta_network=WTANetworkGroup(network_group_size, num_groups, params=wta_params, background_input=background_input,
        task_inputs=task_inputs, single_inh_pop=single_inh_pop)

    for i in range(num_trials):
        # LFP source
        lfp_source=LFPSource(wta_network)

        # Create voxel
        voxel=Voxel(network=wta_network)

        wta_monitor=WTAMonitor(wta_network, network_group_size, num_groups, lfp_source, voxel, record_lfp=record_lfp,
            record_voxel=record_voxel, record_neuron_state=record_neuron_state, record_spikes=record_spikes,
            record_firing_rate=record_firing_rate)

        net=Network(background_input, task_inputs, wta_network, lfp_source, voxel, wta_network.connections,
            wta_monitor.monitors)
        reinit_default_clock()

        start_time = time()
        net.run(trial_duration, report='text')
        print "Simulation time:", time() - start_time

        if output_file is not None:
            file='%s.%d.h5' % (output_prefix, i)
            write_output(background_input_size, background_rate, input_freq, network_group_size, num_groups, file,
                record_firing_rate, record_neuron_state, record_spikes, record_voxel, record_lfp, stim_end_time,
                stim_start_time, task_input_size, trial_duration, voxel, wta_monitor, wta_params)

            print 'Wrote output to %s' % output_file

        if plot_output:
            wta_monitor.plot()

if __name__=='__main__':
    ap = argparse.ArgumentParser(description='Run the WTA model')
    ap.add_argument('--num_groups', type=int, default=3, help='Number of input groups')
    ap.add_argument('--input_pattern', type=str, default='low', help='Input pattern (low or high) contrast')
    ap.add_argument('--trial_duration', type=float, default=2.0, help='Trial duration (seconds)')
    ap.add_argument('--p_b_e', type=float, default=0.01, help='Connection prob from background to excitatory neurons')
    ap.add_argument('--p_x_e', type=float, default=0.01, help='Connection prob from task inputs to excitatory neurons')
    ap.add_argument('--p_e_e', type=float, default=0.005, help='Connection prob from excitatory neurons to excitatory ' \
                                                               'neurons in the same group')
    ap.add_argument('--p_e_i', type=float, default=0.1, help='Connection prob from excitatory neurons to inhibitory ' \
                                                             'neurons in the same group')
    ap.add_argument('--p_i_i', type=float, default=0.01, help='Connection prob from inhibitory neurons to inhibitory ' \
                                                              'neurons in the same group')
    ap.add_argument('--p_i_e', type=float, default=0.01, help='Connection prob from inhibitory neurons to excitatory ' \
                                                              'neurons in other groups')
    ap.add_argument('--output_prefix', type=str, default=None, help='HDF5 output file prefix')
    ap.add_argument('--record_lfp', type=int, default=1, help='Record LFP data')
    ap.add_argument('--record_voxel', type=int, default=1, help='Record voxel data')
    ap.add_argument('--record_neuron_state', type=int, default=0, help='Record neuron state data (synaptic conductances, ' \
                                                                       'membrane potential)')
    ap.add_argument('--record_spikes', type=int, default=1, help='Record neuron spikes')
    ap.add_argument('--record_firing_rate', type=int, default=1, help='Record neuron firing rate')

    argvals = ap.parse_args()

    input_freq=np.array([0, 0, 0])*Hz
    if argvals.input_pattern=='low':
        input_freq=np.array([20, 20, 20])*Hz
    elif argvals.input_pattern=='high':
        input_freq=np.array([10, 10, 40])*Hz

    wta_params=default_params()
    wta_params.p_b_e=argvals.p_b_e
    wta_params.p_x_e=argvals.p_x_e
    wta_params.p_e_e=argvals.p_e_e
    wta_params.p_e_i=argvals.p_e_i
    wta_params.p_i_i=argvals.p_i_i
    wta_params.p_i_e=argvals.p_i_e

    run_wta(wta_params, argvals.num_groups, input_freq, argvals.trial_duration*second, output_file=argvals.output_prefix,
        record_lfp=argvals.record_lfp, record_voxel=argvals.record_voxel,
        record_neuron_state=argvals.record_neuron_state, record_spikes=argvals.record_spikes,
        record_firing_rate=argvals.record_firing_rate)
