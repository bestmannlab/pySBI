from time import time
from brian.clock import reinit_default_clock
from brian.directcontrol import PoissonGroup
from brian.equations import Equations
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse
from brian.membrane_equations import Current
from brian.monitor import MultiStateMonitor, StateMonitor, PopulationRateMonitor, SpikeMonitor
from brian.network import Network, network_operation
from brian.neurongroup import NeuronGroup
from brian.plotting import raster_plot
from brian.stdunits import pF, nS, mV, ms, nA, mA, Hz
from brian.tools.parameters import Parameters
from brian.units import siemens, hertz, second
import h5py
import argparse
import numpy as np
from matplotlib.pyplot import figure, subplot, xlabel, ylabel, legend, show, ylim
from numpy.matlib import randn
from pysbi.util.utils import init_connection
from pysbi.voxel import Voxel, LFPSource, get_bold_signal

# Default parameters for a WTA network with multiple inhibitory populations
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
    E_ampa = 0 * mV,
    E_nmda = 0 * mV,
    E_gaba_a = -70 * mV,
    E_gaba_b = -95 * mV,
    tau_ampa = 2.5*ms,
    tau1_nmda = 10*ms,
    tau2_nmda = 100*ms,
    tau_gaba_a = 2.5*ms,
    tau1_gaba_b = 10*ms,
    tau2_gaba_b =100*ms,
    w_ampa_min=0.35*nS,
    w_ampa_max=1.0*nS,
    w_nmda_min=0.01*nS,
    w_nmda_max=0.6*nS,
    w_gaba_a_min=0.25*nS,
    w_gaba_a_max=1.2*nS,
    w_gaba_b_min=0.1*nS,
    w_gaba_b_max=1.0*nS,
    # Connection probabilities
    p_b_e=0.075,
    p_x_e=0.075,
    p_e_e=0.075,
    p_e_i=0.1,
    p_i_i=0.01,
    p_i_e=0.02)

# WTA network class - extends Brian's NeuronGroup
class WTANetworkGroup(NeuronGroup):

    ### Constructor
    #       N = total number of neurons per input group
    #       num_groups = number of input groups
    #       params = network parameters
    #       background_input = background input source
    #       task_inputs = task input sources
    #       single_inh_pop = single inhibitory population if true
    def __init__(self, N, num_groups, params=default_params, background_input=None, task_inputs=None,
                 single_inh_pop=False):
        self.N=N
        self.num_groups=num_groups
        self.params=params
        self.background_input=background_input
        self.task_inputs=task_inputs
        self.single_inh_pop=single_inh_pop

        ## Set up equations

        # Exponential integrate-and-fire neuron
        eqs = exp_IF(params.C, params.gL, params.EL, params.VT, params.DeltaT)

        eqs += Equations('g_muscimol : nS')
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
        eqs += biexp_synapse('g_nmda', params.tau1_nmda, params.tau2_nmda, siemens)
        eqs += Equations('g_V = 1/(1+(Mg/3.57)*exp(-0.062 *vm/mV)) : 1 ', Mg=params.Mg)
        eqs += Current('I_nmda=g_V*g_nmda*(E-vm): amp', E=params.E_nmda)

        # GABA-A conductance
        eqs += exp_synapse('g_gaba_a', params.tau_gaba_a, siemens)
        eqs += Current('I_gaba_a=(g_gaba_a+g_muscimol)*(E-vm): amp', E=params.E_gaba_a)

        # GABA-B conductance
        eqs += biexp_synapse('g_gaba_b', params.tau1_gaba_b, params.tau2_gaba_b, siemens)
        eqs += Current('I_gaba_b=g_gaba_b*(E-vm): amp', E=params.E_gaba_b)

        # Total synaptic conductance
        eqs += Equations('g_syn=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda+g_gaba_a+g_gaba_b : siemens')
        eqs += Equations('g_syn_exc=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda : siemens')
        # Total synaptic current
        eqs += Equations('I_abs=(I_ampa_r**2)**.5+(I_ampa_b**2)**.5+(I_ampa_x**2)**.5+(I_nmda**2)**.5+(I_gaba_a**2)**.5+(I_gaba_b**2)**.5 : amp')

        NeuronGroup.__init__(self, N*num_groups, model=eqs, threshold=-20*mV, reset=params.EL, compile=True,
            freeze=True)

        self.init_subpopulations()

        self.init_connectivity()

#    def inject_muscimol(self, group_idx, amount):
#        self.groups_e[group_idx].gL+=amount
#        print('e %d muscimol=%.4f nS' % (group_idx,self.groups_e[group_idx].gL/nS))
#        self.groups_i[group_idx].gL+=amount
#        print('i %d muscimol=%.4f nS' % (group_idx,self.groups_i[group_idx].gL/nS))

    ## Initialize excitatory and inhibitory subpopulations
    def init_subpopulations(self):
        self.groups_e=[]
        self.groups_i=[]

        # Main excitatory subpopulation
        self.group_e=self.subgroup(int(4*self.N*self.num_groups/5))
        # regular spiking params (from Naud et al., 2008)
        self.group_e.C=104*pF
        self.group_e.gL=4.3*nS
        self.group_e.EL=-65*mV
        self.group_e.VT=-52*mV
        self.group_e.DeltaT=0.8*mV

        # Main inhibitory subpopulation
        self.group_i=self.subgroup(int(self.N*self.num_groups/5))
        # fast-spiking interneuron params (from Naud et al., 2008)
        self.group_i.C=59*pF
        self.group_i.gL=2.9*nS
        self.group_i.EL=-62*mV
        self.group_i.VT=-42*mV
        self.group_i.DeltaT=3.0*mV

        # Input-specific sub-subpopulations
        for i in range(self.num_groups):
            subgroup_e=self.group_e.subgroup(int(4*self.N/5))
            self.groups_e.append(subgroup_e)

            if not self.single_inh_pop:
                subgroup_i=self.group_i.subgroup(int(self.N/5))
                self.groups_i.append(subgroup_i)

        # Initialize state variables
        self.vm = self.params.EL+randn(self.N*self.num_groups)*10*mV
        self.g_ampa_r = self.params.w_ampa_min+randn(self.N*self.num_groups)*(self.params.w_ampa_max-self.params.w_ampa_min)*.1
        self.g_ampa_b = self.params.w_ampa_min+randn(self.N*self.num_groups)*(self.params.w_ampa_max-self.params.w_ampa_min)*.1
        self.g_ampa_x = self.params.w_ampa_min+randn(self.N*self.num_groups)*(self.params.w_ampa_max-self.params.w_ampa_min)*.1
        self.g_nmda = self.params.w_nmda_min+randn(self.N*self.num_groups)*(self.params.w_nmda_max-self.params.w_nmda_min)*.1
        self.g_gaba_a = self.params.w_gaba_a_min+randn(self.N*self.num_groups)*(self.params.w_gaba_a_max-self.params.w_gaba_a_min)*.1
        self.g_gaba_b = self.params.w_gaba_a_min+randn(self.N*self.num_groups)*(self.params.w_gaba_a_max-self.params.w_gaba_a_min)*.1


    ## Initialize network connectivity
    def init_connectivity(self):
        self.connections=[]

        # Iterate over input groups
        for i in range(self.num_groups):

            # E population - recurrent connections
            self.connections.append(init_connection(self.groups_e[i], self.groups_e[i], 'g_ampa_r',
                self.params.w_ampa_min, self.params.w_ampa_max, self.params.p_e_e, (0*ms, 5*ms), allow_self_conn=False))
            self.connections.append(init_connection(self.groups_e[i], self.groups_e[i], 'g_nmda',
                self.params.w_nmda_min, self.params.w_nmda_max, self.params.p_e_e, (0*ms, 5*ms), allow_self_conn=False))

            if not self.single_inh_pop:
                # I -> I population - recurrent connections
                self.connections.append(init_connection(self.groups_i[i], self.groups_i[i], 'g_gaba_a',
                    self.params.w_gaba_a_min, self.params.w_gaba_a_max, self.params.p_i_i, (0*ms, 5*ms),
                    allow_self_conn=False))
                self.connections.append(init_connection(self.groups_i[i], self.groups_i[i], 'g_gaba_b',
                    self.params.w_gaba_b_min, self.params.w_gaba_b_max, self.params.p_i_i, (0*ms, 5*ms),
                    allow_self_conn=False))

                # E -> I excitatory connections
                self.connections.append(init_connection(self.groups_e[i], self.groups_i[i], 'g_ampa_r',
                    self.params.w_ampa_min, self.params.w_ampa_max, self.params.p_e_i, (0*ms, 5*ms)))
                self.connections.append(init_connection(self.groups_e[i], self.groups_i[i], 'g_nmda',
                    self.params.w_nmda_min, self.params.w_nmda_max, self.params.p_e_i, (0*ms, 5*ms)))

                # I -> E - inhibitory connections
                for j in range(self.num_groups):
                    if not i==j:
                        self.connections.append(init_connection(self.groups_i[i], self.groups_e[j], 'g_gaba_a',
                            self.params.w_gaba_a_min, self.params.w_gaba_a_max, self.params.p_i_e, (0*ms, 5*ms)))
                        self.connections.append(init_connection(self.groups_i[i], self.groups_e[j], 'g_gaba_b',
                            self.params.w_gaba_b_min, self.params.w_gaba_b_max, self.params.p_i_e, (0*ms, 5*ms)))

            else:
                # E -> I excitatory connections
                self.connections.append(init_connection(self.groups_e[i], self.group_i, 'g_ampa_r',
                    self.params.w_ampa_min, self.params.w_ampa_max, self.params.p_e_i, (0*ms, 5*ms)))
                self.connections.append(init_connection(self.groups_e[i], self.group_i, 'g_nmda',
                    self.params.w_nmda_min, self.params.w_nmda_max, self.params.p_e_i, (0*ms, 5*ms)))

                # I -> E - inhibitory connections
                self.connections.append(init_connection(self.group_i, self.groups_e[i], 'g_gaba_a',
                    self.params.w_gaba_a_min, self.params.w_gaba_a_max, self.params.p_i_e, (0*ms, 5*ms)))
                self.connections.append(init_connection(self.group_i, self.groups_e[i], 'g_gaba_b',
                    self.params.w_gaba_b_min, self.params.w_gaba_b_max, self.params.p_i_e, (0*ms, 5*ms)))

        if self.single_inh_pop:
            # I population - recurrent connections
            self.connections.append(init_connection(self.group_i, self.group_i, 'g_gaba_a', self.params.w_gaba_a_min,
                self.params.w_gaba_a_max, self.params.p_i_i, (0*ms, 5*ms), allow_self_conn=False))
            self.connections.append(init_connection(self.group_i, self.group_i, 'g_gaba_b', self.params.w_gaba_b_min,
                self.params.w_gaba_b_max, self.params.p_i_i, (0*ms, 5*ms), allow_self_conn=False))

        if self.background_input is not None:
            # Background -> E+I population connectinos
            self.connections.append(init_connection(self.background_input, self, 'g_ampa_b', self.params.w_ampa_min,
                self.params.w_ampa_max, self.params.p_b_e, (0*ms, 5*ms)))

        if self.task_inputs is not None:
            # Task input -> E population connections
            for i in range(self.num_groups):
                self.connections.append(init_connection(self.task_inputs[i], self.groups_e[i], 'g_ampa_x',
                    self.params.w_ampa_min, self.params.w_ampa_max, self.params.p_x_e, (0*ms, 5*ms)))


# Collection of monitors for WTA network
class WTAMonitor():

    ## Constructor
    #       network = network to monitor
    #       lfp_source = LFP source to monitor
    #       voxel = voxel to monitor
    #       record_lfp = record LFP signals if true
    #       record_voxel = record voxel signals if true
    #       record_neuron_state = record neuron state signals if true
    #       record_spikes = record spikes if true
    #       record_firing_rate = record firing rate if true
    #       record_inputs = record inputs if true
    def __init__(self, network, lfp_source, voxel, record_lfp=True, record_voxel=True, record_neuron_state=False,
                 record_spikes=True, record_firing_rate=True, record_inputs=False):
        self.num_groups=network.num_groups
        self.N=network.N
        self.monitors=[]

        # LFP monitor
        if record_lfp:
            self.lfp_monitor = StateMonitor(lfp_source, 'LFP', record=0)
            self.monitors.append(self.lfp_monitor)
        else:
            self.lfp_monitor=None

        # Voxel monitor
        if record_voxel:
            self.voxel_monitor = MultiStateMonitor(voxel, vars=['G_total','G_total_exc','s','f_in','v','f_out','q','y'],
                record=True)
            self.monitors.append(self.voxel_monitor)
        else:
            self.voxel_monitor=None
        self.voxel_exc_monitor=None

        # Network monitor
        if record_neuron_state:
            self.record_idx=[]
            for i in range(self.num_groups):
                e_idx=i*int(4*self.N/5)
                i_idx=int(4*self.N*self.num_groups/5)+i*int(self.N/5)
                self.record_idx.extend([e_idx, i_idx])
            self.network_monitor = MultiStateMonitor(network, vars=['vm','g_ampa_r','g_ampa_x','g_ampa_b','g_gaba_a',
                                                                    'g_gaba_b','g_nmda','I_ampa_r','I_ampa_x',
                                                                    'I_ampa_b','I_gaba_a','I_gaba_b','I_nmda'],
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

            if len(network.groups_i):
                for group_i in network.groups_i:
                    i_rate_monitor=PopulationRateMonitor(group_i)
                    self.population_rate_monitors['inhibitory'].append(i_rate_monitor)
                    self.monitors.append(i_rate_monitor)
            else:
                i_rate_monitor=PopulationRateMonitor(network.group_i)
                self.population_rate_monitors['inhibitory'].append(i_rate_monitor)
                self.monitors.append(i_rate_monitor)
        else:
            self.population_rate_monitors=None

        # Input rate monitors
        if record_inputs:
            self.background_rate_monitor=PopulationRateMonitor(network.background_input)
            self.monitors.append(self.background_rate_monitor)
            self.task_rate_monitors=[]
            for task_input in network.task_inputs:
                task_monitor=PopulationRateMonitor(task_input)
                self.task_rate_monitors.append(task_monitor)
                self.monitors.append(task_monitor)
        else:
            self.background_rate_monitor=None
            self.task_rate_monitors=None

        # Spike monitors
        if record_spikes:
            self.spike_monitors={'excitatory':[], 'inhibitory':[]}
            for group_e in network.groups_e:
                e_spike_monitor=SpikeMonitor(group_e)
                self.spike_monitors['excitatory'].append(e_spike_monitor)
                self.monitors.append(e_spike_monitor)

            if len(network.groups_i):
                for group_i in network.groups_i:
                    i_spike_monitor=SpikeMonitor(group_i)
                    self.spike_monitors['inhibitory'].append(i_spike_monitor)
                    self.monitors.append(i_spike_monitor)
            else:
                i_spike_monitor=SpikeMonitor(network.group_i)
                self.spike_monitors['inhibitory'].append(i_spike_monitor)
                self.monitors.append(i_spike_monitor)
        else:
            self.spike_monitors=None

    # Plot monitor data
    def plot(self):

        # Spike raster plots
        if self.spike_monitors is not None:
            figure()
            subplot(211)
            raster_plot(*self.spike_monitors['excitatory'],newfigure=False)
            subplot(212)
            raster_plot(*self.spike_monitors['inhibitory'],newfigure=False)

        # Network firing rate plots
        if self.population_rate_monitors is not None:
            figure()
            ax=subplot(211)
            max_rates=[np.max(pop_rate_monitor.smooth_rate(width=5*ms)/hertz) for pop_rate_monitor in
                       self.population_rate_monitors['excitatory']]
            max_rates.extend([np.max(pop_rate_monitor.smooth_rate(width=5*ms)/hertz) for pop_rate_monitor in
                              self.population_rate_monitors['inhibitory']])
            max_rate=np.max(max_rates)

            for idx,pop_rate_monitor in enumerate(self.population_rate_monitors['excitatory']):
                ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms)/hertz, label='e %d' % idx)
                ylim(0,max_rate)
            legend()
            ylabel('Firing rate (Hz)')

            ax=subplot(212)
            for idx,pop_rate_monitor in enumerate(self.population_rate_monitors['inhibitory']):
                ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms)/hertz, label='i %d' % idx)
                ylim(0,max_rate)
            legend()
            ylabel('Firing rate (Hz)')
            xlabel('Time (ms)')

        # Input firing rate plots
        if self.background_rate_monitor is not None and self.task_rate_monitors is not None:
            figure()
            max_rate=np.max(self.background_rate_monitor.smooth_rate(width=5*ms)/hertz)
            for task_monitor in self.task_rate_monitors:
                max_rate=np.max(max_rate, np.max(task_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz))
            ax=subplot(111)
            ax.plot(self.background_rate_monitor.times/ms, self.background_rate_monitor.smooth_rate(width=5*ms)/hertz)
            ylim(0,max_rate)
            for task_monitor in self.task_rate_monitors:
                ax.plot(task_monitor.times/ms, task_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz)
                ylim(0,max_rate)

        # Network state plots
        if self.network_monitor is not None:
            figure()
            max_conductances=[]
            for i in range(self.num_groups):
                neuron_idx=self.record_idx[i*2]
                max_conductances.append(np.max(self.network_monitor['g_ampa_r'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_ampa_x'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_ampa_b'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_nmda'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_gaba_a'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_gaba_b'][neuron_idx]/nS))
                neuron_idx=self.record_idx[i*2+1]
                max_conductances.append(np.max(self.network_monitor['g_ampa_r'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_ampa_x'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_ampa_b'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_nmda'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_gaba_a'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_gaba_b'][neuron_idx]/nS))
            max_conductance=np.max(max_conductances)

            for i in range(self.num_groups):
                ax=subplot(self.num_groups*100+20+(i*2+1))
                neuron_idx=self.record_idx[i*2]
                ax.plot(self.network_monitor['g_ampa_r'].times/ms, self.network_monitor['g_ampa_r'][neuron_idx]/nS,
                    label='AMPA-recurrent')
                ax.plot(self.network_monitor['g_ampa_x'].times/ms, self.network_monitor['g_ampa_x'][neuron_idx]/nS,
                    label='AMPA-task')
                ax.plot(self.network_monitor['g_ampa_b'].times/ms, self.network_monitor['g_ampa_b'][neuron_idx]/nS,
                    label='AMPA-backgrnd')
                ax.plot(self.network_monitor['g_nmda'].times/ms, self.network_monitor['g_nmda'][neuron_idx]/nS,
                    label='NMDA')
                ax.plot(self.network_monitor['g_gaba_a'].times/ms, self.network_monitor['g_gaba_a'][neuron_idx]/nS,
                    label='GABA_A')
                ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][neuron_idx]/nS,
                    label='GABA_B')
                ylim(0,max_conductance)
                xlabel('Time (ms)')
                ylabel('Conductance (nS)')
                legend()

                ax=subplot(self.num_groups*100+20+(i*2+2))
                neuron_idx=self.record_idx[i*2+1]
                ax.plot(self.network_monitor['g_ampa_r'].times/ms, self.network_monitor['g_ampa_r'][neuron_idx]/nS,
                    label='AMPA-recurrent')
                ax.plot(self.network_monitor['g_ampa_x'].times/ms, self.network_monitor['g_ampa_x'][neuron_idx]/nS,
                    label='AMPA-task')
                ax.plot(self.network_monitor['g_ampa_b'].times/ms, self.network_monitor['g_ampa_b'][neuron_idx]/nS,
                    label='AMPA-backgrnd')
                ax.plot(self.network_monitor['g_nmda'].times/ms, self.network_monitor['g_nmda'][neuron_idx]/nS,
                    label='NMDA')
                ax.plot(self.network_monitor['g_gaba_a'].times/ms, self.network_monitor['g_gaba_a'][neuron_idx]/nS,
                    label='GABA_A')
                ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][neuron_idx]/nS,
                    label='GABA_B')
                ylim(0,max_conductance)
                xlabel('Time (ms)')
                ylabel('Conductance (nS)')
                legend()

            figure()
            min_currents=[]
            max_currents=[]
            for i in range(self.num_groups):
                neuron_idx=self.record_idx[i*2]
                max_currents.append(np.max(self.network_monitor['I_ampa_r'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_ampa_x'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_ampa_b'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_nmda'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_gaba_a'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_gaba_b'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_r'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_x'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_b'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_nmda'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_gaba_a'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_gaba_b'][neuron_idx]/nS))
                neuron_idx=self.record_idx[i*2+1]
                max_currents.append(np.max(self.network_monitor['I_ampa_r'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_ampa_x'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_ampa_b'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_nmda'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_gaba_a'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_gaba_b'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_r'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_x'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_b'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_nmda'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_gaba_a'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_gaba_b'][neuron_idx]/nS))
            max_current=np.max(max_currents)
            min_current=np.min(min_currents)

            for i in range(self.num_groups):
                ax=subplot(self.num_groups*100+20+(i*2+1))
                neuron_idx=self.record_idx[i*2]
                ax.plot(self.network_monitor['I_ampa_r'].times/ms, self.network_monitor['I_ampa_r'][neuron_idx]/nA,
                    label='AMPA-recurrent')
                ax.plot(self.network_monitor['I_ampa_x'].times/ms, self.network_monitor['I_ampa_x'][neuron_idx]/nA,
                    label='AMPA-task')
                ax.plot(self.network_monitor['I_ampa_b'].times/ms, self.network_monitor['I_ampa_b'][neuron_idx]/nA,
                    label='AMPA-backgrnd')
                ax.plot(self.network_monitor['I_nmda'].times/ms, self.network_monitor['I_nmda'][neuron_idx]/nA,
                    label='NMDA')
                ax.plot(self.network_monitor['I_gaba_a'].times/ms, self.network_monitor['I_gaba_a'][neuron_idx]/nA,
                    label='GABA_A')
                ax.plot(self.network_monitor['I_gaba_b'].times/ms, self.network_monitor['I_gaba_b'][neuron_idx]/nA,
                    label='GABA_B')
                ylim(min_current,max_current)
                xlabel('Time (ms)')
                ylabel('Current (nA)')
                legend()

                ax=subplot(self.num_groups*100+20+(i*2+2))
                neuron_idx=self.record_idx[i*2+1]
                ax.plot(self.network_monitor['I_ampa_r'].times/ms, self.network_monitor['I_ampa_r'][neuron_idx]/nA,
                    label='AMPA-recurrent')
                ax.plot(self.network_monitor['I_ampa_x'].times/ms, self.network_monitor['I_ampa_x'][neuron_idx]/nA,
                    label='AMPA-task')
                ax.plot(self.network_monitor['I_ampa_b'].times/ms, self.network_monitor['I_ampa_b'][neuron_idx]/nA,
                    label='AMPA-backgrnd')
                ax.plot(self.network_monitor['I_nmda'].times/ms, self.network_monitor['I_nmda'][neuron_idx]/nA,
                    label='NMDA')
                ax.plot(self.network_monitor['I_gaba_a'].times/ms, self.network_monitor['I_gaba_a'][neuron_idx]/nA,
                    label='GABA_A')
                ax.plot(self.network_monitor['I_gaba_b'].times/ms, self.network_monitor['I_gaba_b'][neuron_idx]/nA,
                    label='GABA_B')
                ylim(min_current,max_current)
                xlabel('Time (ms)')
                ylabel('Current (nA)')
                legend()

        # LFP plot
        if self.lfp_monitor is not None:
            figure()
            ax=subplot(111)
            ax.plot(self.lfp_monitor.times / ms, self.lfp_monitor[0] / mA)
            xlabel('Time (ms)')
            ylabel('LFP (mA)')

        # Voxel activity plots
        if self.voxel_monitor is not None:
            syn_max=np.max(self.voxel_monitor['G_total'][0] / nS)
            y_max=np.max(self.voxel_monitor['y'][0])
            y_min=np.min(self.voxel_monitor['y'][0])
            figure()
            if self.voxel_exc_monitor is None:
                ax=subplot(211)
            else:
                ax=subplot(221)
                syn_max=np.max([syn_max, np.max(self.voxel_exc_monitor['G_total'][0])])
                y_max=np.max([y_max, np.max(self.voxel_exc_monitor['y'][0])])
                y_min=np.min([y_min, np.min(self.voxel_exc_monitor['y'][0])])
            ax.plot(self.voxel_monitor['G_total'].times / ms, self.voxel_monitor['G_total'][0] / nS)
            xlabel('Time (ms)')
            ylabel('Total Synaptic Activity (nS)')
            ylim(0, syn_max)
            if self.voxel_exc_monitor is None:
                ax=subplot(212)
            else:
                ax=subplot(222)
            ax.plot(self.voxel_monitor['y'].times / ms, self.voxel_monitor['y'][0])
            xlabel('Time (ms)')
            ylabel('BOLD')
            ylim(y_min, y_max)
            if self.voxel_exc_monitor is not None:
                ax=subplot(223)
                ax.plot(self.voxel_exc_monitor['G_total'].times / ms, self.voxel_exc_monitor['G_total'][0] / nS)
                xlabel('Time (ms)')
                ylabel('Total Synaptic Activity (nS)')
                ylim(0, syn_max)
                ax=subplot(224)
                ax.plot(self.voxel_exc_monitor['y'].times / ms, self.voxel_exc_monitor['y'][0])
                xlabel('Time (ms)')
                ylabel('BOLD')
                ylim(y_min, y_max)
        show()

## Write monitor data to HDF5 file
#       background_input_size = number of background inputs
#       bacground rate = background firing rate
#       input_freq = input firing rates
#       network_group_size = number of neurons per input group
#       num_groups = number of input groups
#       single_inh_pop = single inhibitory population
#       output_file = filename to write to
#       record_firing_rate = write network firing rate data when true
#       record_neuron_stae = write neuron state data when true
#       record_spikes = write spike data when true
#       record_voxel = write voxel data when true
#       record_lfp = write LFP data when true
#       record_inputs = write input firing rates when true
#       stim_end_time = stimulation end time
#       stim_start_time = stimulation start time
#       task_input_size = number of neurons in each task input group
#       trial_duration = duration of the trial
#       voxel = voxel for network
#       wta_monitor = network monitor
#       wta_params = network parameters
def write_output(background_input_size, background_rate, input_freq, network_group_size, num_groups, single_inh_pop,
                 output_file, record_firing_rate, record_neuron_state, record_spikes, record_voxel, record_lfp,
                 record_inputs, stim_end_time, stim_start_time, task_input_size, trial_duration, voxel, wta_monitor,
                 wta_params, muscimol_amount, injection_site):

    f = h5py.File(output_file, 'w')

    # Write basic parameters
    f.attrs['single_inh_pop']=single_inh_pop
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
    f.attrs['tau_ampa'] = wta_params.tau_ampa
    f.attrs['tau1_nmda'] = wta_params.tau1_nmda
    f.attrs['tau2_nmda'] = wta_params.tau2_nmda
    f.attrs['tau_gaba_a'] = wta_params.tau_gaba_a
    f.attrs['tau1_gaba_b'] = wta_params.tau1_gaba_b
    f.attrs['tau2_gaba_b'] = wta_params.tau2_gaba_b
    f.attrs['w_ampa_b'] = wta_params.w_ampa_b
    f.attrs['w_ampa_x'] = wta_params.w_ampa_x
    f.attrs['w_ampa_r'] = wta_params.w_ampa_r
    f.attrs['w_nmda'] = wta_params.w_nmda
    f.attrs['w_gaba_a'] = wta_params.w_gaba_a
    f.attrs['p_b_e'] = wta_params.p_b_e
    f.attrs['p_x_e'] = wta_params.p_x_e
    f.attrs['p_e_e'] = wta_params.p_e_e
    f.attrs['p_e_i'] = wta_params.p_e_i
    f.attrs['p_i_i'] = wta_params.p_i_i
    f.attrs['p_i_e'] = wta_params.p_i_e
    f.attrs['muscimol_amount'] = muscimol_amount
    f.attrs['injection_site'] = injection_site

    # Write LFP data
    if record_lfp:
        f_lfp = f.create_group('lfp')
        f_lfp['lfp']=wta_monitor.lfp_monitor.values

    # Write voxel data
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

        f_vox_total=f_vox.create_group('total_syn')
        f_vox_total['G_total'] = wta_monitor.voxel_monitor['G_total'].values
        f_vox_total['s'] = wta_monitor.voxel_monitor['s'].values
        f_vox_total['f_in'] = wta_monitor.voxel_monitor['f_in'].values
        f_vox_total['v'] = wta_monitor.voxel_monitor['v'].values
        f_vox_total['q'] = wta_monitor.voxel_monitor['q'].values
        f_vox_total['y'] = wta_monitor.voxel_monitor['y'].values

        f_vox_exc=f_vox.create_group('exc_syn')
        f_vox_exc['G_total'] = wta_monitor.voxel_exc_monitor['G_total'].values
        f_vox_exc['s'] = wta_monitor.voxel_exc_monitor['s'].values
        f_vox_exc['f_in'] = wta_monitor.voxel_exc_monitor['f_in'].values
        f_vox_exc['v'] = wta_monitor.voxel_exc_monitor['v'].values
        f_vox_exc['q'] = wta_monitor.voxel_exc_monitor['q'].values
        f_vox_exc['y'] = wta_monitor.voxel_exc_monitor['y'].values

    # Write neuron state data
    if record_neuron_state:
        f_state = f.create_group('neuron_state')
        f_state['g_ampa_r'] = wta_monitor.network_monitor['g_ampa_r'].values
        f_state['g_ampa_x'] = wta_monitor.network_monitor['g_ampa_x'].values
        f_state['g_ampa_b'] = wta_monitor.network_monitor['g_ampa_b'].values
        f_state['g_nmda'] = wta_monitor.network_monitor['g_nmda'].values
        f_state['g_gaba_a'] = wta_monitor.network_monitor['g_gaba_a'].values
        f_state['g_gaba_b'] = wta_monitor.network_monitor['g_gaba_b'].values
        f_state['I_ampa_r'] = wta_monitor.network_monitor['I_ampa_r'].values
        f_state['I_ampa_x'] = wta_monitor.network_monitor['I_ampa_x'].values
        f_state['I_ampa_b'] = wta_monitor.network_monitor['I_ampa_b'].values
        f_state['I_nmda'] = wta_monitor.network_monitor['I_nmda'].values
        f_state['I_gaba_a'] = wta_monitor.network_monitor['I_gaba_a'].values
        f_state['I_gaba_b'] = wta_monitor.network_monitor['I_gaba_b'].values
        f_state['vm'] = wta_monitor.network_monitor['vm'].values
        f_state['record_idx'] = np.array(wta_monitor.record_idx)

    # Write network firing rate data
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

    # Write input firing rate data
    if record_inputs:
        back_rate=f.create_group('background_rate')
        back_rate['firing_rate']=wta_monitor.background_rate_monitor.smooth_rate(width=5*ms,filter='gaussian')
        task_rates=f.create_group('task_rates')
        t_rates=[]
        for task_monitor in wta_monitor.task_rate_monitors:
            t_rates.append(task_monitor.smooth_rate(width=5*ms,filter='gaussian'))
        task_rates['firing_rates']=np.array(t_rates)

    # Write spike data
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


## Run WTA network
#       wta_params = network parameters
#       num_groups = number of input groups
#       input_freq = mean firing rate of each input group
#       trial_duration = how long to simulate
#       ouput_file = ouput file to write to
#       record_lfp = record LFP data if true
#       record_voxel = record voxel data if true
#       record_neuron_state = record neuron state data if true
#       record_spikes = record spike data if true
#       record_firing_rate = record network firing rates if true
#       record_inputs = record input firing rates if true
#       plot_output = plot outputs if true
#       single_inh_pop = single inhibitory population if true
def run_wta(wta_params, num_groups, input_freq, trial_duration, output_file=None, record_lfp=True, record_voxel=True,
            record_neuron_state=False, record_spikes=True, record_firing_rate=True, record_inputs=False,
            plot_output=False, single_inh_pop=False, muscimol_amount=0*nS, injection_site=0):

    # Init simulation parameters
    background_rate=10*Hz
    stim_start_time=.25*second
    stim_end_time=.75*second
    network_group_size=2000
    background_input_size=1000
    task_input_size=1000

    # Create network inputs
    background_input=PoissonGroup(background_input_size, rates=background_rate)
    task_inputs=[]
    def make_rate_function(rate):
        return lambda t: ((stim_start_time<t<stim_end_time and (background_rate+rate)) or background_rate)
    for i in range(num_groups):
        rate=input_freq[i]*Hz
        task_inputs.append(PoissonGroup(task_input_size, rates=make_rate_function(rate)))

    # Create WTA network
    wta_network=WTANetworkGroup(network_group_size, num_groups, params=wta_params, background_input=background_input,
        task_inputs=task_inputs, single_inh_pop=single_inh_pop)

    # LFP source
    lfp_source=LFPSource(wta_network.group_e)

    # Create voxel
    voxel=Voxel(network=wta_network)

    # Create network monitor
    wta_monitor=WTAMonitor(wta_network, lfp_source, voxel, record_lfp=record_lfp, record_voxel=record_voxel,
        record_neuron_state=record_neuron_state, record_spikes=record_spikes, record_firing_rate=record_firing_rate,
        record_inputs=record_inputs)

    @network_operation(when='start')
    def inject_muscimol():
        if muscimol_amount>0:
            wta_network.groups_e[injection_site].g_muscimol=muscimol_amount
            wta_network.groups_i[injection_site].g_muscimol=muscimol_amount

    # Create Brian network and reset clock
    net=Network(background_input, task_inputs, wta_network, lfp_source, voxel, wta_network.connections,
        wta_monitor.monitors, inject_muscimol)
    reinit_default_clock()

    # Run simulation
    start_time = time()
    net.run(trial_duration, report='text')
    print "Simulation time:", time() - start_time

    # Compute BOLD signal
    if record_voxel:
        wta_monitor.voxel_exc_monitor=get_bold_signal(wta_monitor.voxel_monitor['G_total_exc'].values[0], voxel.params,
            [500, 2500], trial_duration)
        wta_monitor.voxel_monitor=get_bold_signal(wta_monitor.voxel_monitor['G_total'].values[0], voxel.params,
            [500, 2500], trial_duration)

    # Write output to file
    if output_file is not None:
        write_output(background_input_size, background_rate, input_freq, network_group_size, num_groups, single_inh_pop,
            output_file, record_firing_rate, record_neuron_state, record_spikes, record_voxel, record_lfp, record_inputs,
            stim_end_time, stim_start_time, task_input_size, trial_duration, voxel, wta_monitor, wta_params, muscimol_amount,
            injection_site)

        print 'Wrote output to %s' % output_file

    # Plot outputs
    if plot_output:
        wta_monitor.plot()

    return wta_monitor

if __name__=='__main__':
    ap = argparse.ArgumentParser(description='Run the WTA model')
    ap.add_argument('--num_groups', type=int, default=3, help='Number of input groups')
    ap.add_argument('--inputs', type=str, default='0', help='Input pattern (Hz) - comma-delimited list')
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
    ap.add_argument('--output_file', type=str, default=None, help='HDF5 output file')
    ap.add_argument('--single_inh_pop', type=int, default=0, help='Single inhibitory population')
    ap.add_argument('--muscimol_amount', type=float, default=0.0, help='Amount of muscimol to inject')
    ap.add_argument('--injection_site', type=int, default=0, help='Site of muscimol injection (group index)')
    ap.add_argument('--record_lfp', type=int, default=1, help='Record LFP data')
    ap.add_argument('--record_voxel', type=int, default=1, help='Record voxel data')
    ap.add_argument('--record_neuron_state', type=int, default=0, help='Record neuron state data (synaptic conductances, ' \
                                                                       'membrane potential)')
    ap.add_argument('--record_spikes', type=int, default=1, help='Record neuron spikes')
    ap.add_argument('--record_firing_rate', type=int, default=1, help='Record neuron firing rate')
    ap.add_argument('--record_inputs', type=int, default=0, help='Record network inputs')

    argvals = ap.parse_args()

    input_freq=np.zeros(argvals.num_groups)
    inputs=argvals.inputs.split(',')
    for i in range(argvals.num_groups):
        input_freq[i]=float(inputs[i])*Hz

    wta_params=default_params()
    wta_params.p_b_e=argvals.p_b_e
    wta_params.p_x_e=argvals.p_x_e
    wta_params.p_e_e=argvals.p_e_e
    wta_params.p_e_i=argvals.p_e_i
    wta_params.p_i_i=argvals.p_i_i
    wta_params.p_i_e=argvals.p_i_e

    run_wta(wta_params, argvals.num_groups, input_freq, argvals.trial_duration*second, output_file=argvals.output_file,
        record_lfp=argvals.record_lfp, record_voxel=argvals.record_voxel,
        record_neuron_state=argvals.record_neuron_state, record_spikes=argvals.record_spikes,
        record_firing_rate=argvals.record_firing_rate, record_inputs=argvals.record_inputs,
        single_inh_pop=argvals.single_inh_pop, muscimol_amount=argvals.muscimol_amount*nS,
        injection_site=argvals.injection_site)
