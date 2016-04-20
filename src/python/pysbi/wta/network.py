from brian.stdp import ExponentialSTDP
from numpy.matlib import randn, rand
from time import time
import brian
from brian.clock import defaultclock, Clock
from brian.directcontrol import PoissonGroup
from brian.equations import Equations
from brian.connections import DelayConnection
from brian.experimental.model_documentation import document_network, labels_from_namespace, LaTeXDocumentWriter
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse
from brian.membrane_equations import Current, InjectedCurrent
from brian.network import Network, network_operation
from brian.neurongroup import NeuronGroup
from brian.stdunits import pF, nS, mV, ms, Hz, pA, nF
from brian.tools.parameters import Parameters
from brian.units import siemens, second
import argparse
import numpy as np
from pysbi.util.utils import init_connection
from pysbi.voxel import Voxel, LFPSource, get_bold_signal
from pysbi.wta.monitor import WTAMonitor
brian.set_global_preferences(useweave=True,openmp=True,useweave_linear_diffeq =True,
                             gcc_options = ['-ffast-math','-march=native'],usecodegenweave = True,
                             usecodegenreset = True)

pyr_params=Parameters(
    C=0.5*nF,
    gL=25*nS,
    refractory=2*ms,
    w_nmda = 0.165 * nS,
    w_ampa_ext_correct = 2.1*nS,
    w_ampa_ext_incorrect = 0.0*nS,
    w_ampa_bak = 2.1*nS,
    w_ampa_rec = 0.05*nS,
    w_gaba = 1.3*nS,
)

inh_params=Parameters(
    C=0.2*nF,
    gL=20*nS,
    refractory=1*ms,
    w_nmda = 0.13 * nS,
    w_ampa_ext = 1.62*nS,
    w_ampa_bak = 1.63*nS,
    w_ampa_rec = 0.04*nS,
    w_gaba = 1.0*nS,
)

# Default parameters for a WTA network with multiple inhibitory populations
default_params=Parameters(
    # Neuron parameters
    C = 200 * pF,
    gL = 20 * nS,
    EL = -70 * mV,
    VT = -55 * mV,
    DeltaT = 3 * mV,
    Vr = -53 * mV,
    # Magnesium concentration
    Mg = 1,
    # Synapse parameters
    E_ampa = 0 * mV,
    E_nmda = 0 * mV,
    E_gaba_a = -70 * mV,
    tau_ampa = 2*ms,
    tau1_nmda = 2*ms,
    tau2_nmda = 100*ms,
    tau_gaba_a = 5*ms,
    # Connection probabilities
    p_e_e=0.08,
    p_e_i=0.1,
    p_i_i=0.1,
    p_i_e=0.2,
    # Background firing rate
    background_freq=5*Hz,
    # Input variance
    input_var=4*Hz,
    # Input refresh rate
    refresh_rate=60.0*Hz,
    # Number of response options
    num_groups=2,
    # Total size of the network (excitatory and inhibitory cells)
    network_group_size=2000,
    background_input_size=2000,
    mu_0=40.0,
    # Proportion of pyramidal cells getting task-related input
    f=.15,
    task_input_resting_rate=1*Hz,
    # Response threshold
    resp_threshold=25
)
default_params.p_a=default_params.mu_0/100.0
default_params.p_b=default_params.p_a
default_params.task_input_size=int(default_params.network_group_size*.8*default_params.f)

simulation_params=Parameters(
    trial_duration=4*second,
    stim_start_time=1*second,
    stim_end_time=3*second,
    dt=0.5*ms,
    ntrials=1,
    muscimol_amount=0*nS,
    injection_site=0,
    p_dcs=0*pA,
    i_dcs=0*pA,
    dcs_start_time=0*second,
    dcs_end_time=0*second,
    plasticity=False
)

# Plasticity parameters
plasticity_params=Parameters(
    tau_pre = 20 * ms,
    dA_pre= 0.0005,  #0.0005
    # Maximum synaptic weight
    gmax = 4 * nS   #5 *nS
)
plasticity_params.tau_post=plasticity_params.tau_pre
plasticity_params.dA_post=-plasticity_params.dA_pre*1.1  #1.1

# WTA network class - extends Brian's NeuronGroup
class WTANetworkGroup(NeuronGroup):

    ### Constructor
    #       N = total number of neurons per input group
    #       num_groups = number of input groups
    #       params = network parameters
    #       background_input = background input source
    #       task_inputs = task input sources
    def __init__(self, params=default_params, pyr_params=pyr_params(), inh_params=inh_params(),
                 plasticity_params=plasticity_params(), background_input=None, task_inputs=None, clock=defaultclock):
        self.params=params
        self.pyr_params=pyr_params
        self.inh_params=inh_params
        self.plasticity_params=plasticity_params
        self.background_input=background_input
        self.task_inputs=task_inputs

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

        eqs +=InjectedCurrent('I_dcs: amp')

        # Total synaptic conductance
        eqs += Equations('g_syn=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda+g_gaba_a : siemens')
        eqs += Equations('g_syn_exc=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda : siemens')
        # Total synaptic current
        eqs += Equations('I_abs=(I_ampa_r**2)**.5+(I_ampa_b**2)**.5+(I_ampa_x**2)**.5+(I_nmda**2)**.5+(I_gaba_a**2)**.5 : amp')

        NeuronGroup.__init__(self, params.network_group_size, model=eqs, threshold=-20*mV, refractory=1*ms,
            reset=params.Vr, compile=True, freeze=True, clock=clock)

        self.init_subpopulations()

        self.init_connectivity(clock)

    ## Initialize excitatory and inhibitory subpopulations
    def init_subpopulations(self):
        # Main excitatory subpopulation
        self.e_size=int(self.params.network_group_size*.8)
        self.group_e=self.subgroup(self.e_size)
        self.group_e.C=self.pyr_params.C
        self.group_e.gL=self.pyr_params.gL
        self.group_e._refractory_time=self.pyr_params.refractory

        # Main inhibitory subpopulation
        self.i_size=int(self.params.network_group_size*.2)
        self.group_i=self.subgroup(self.i_size)
        self.group_i.C=self.inh_params.C
        self.group_i.gL=self.inh_params.gL
        self.group_i._refractory_time=self.inh_params.refractory

        # Input-specific sub-subpopulations
        self.groups_e=[]
        for i in range(self.params.num_groups):
            subgroup_e=self.group_e.subgroup(int(self.params.f*self.e_size))
            self.groups_e.append(subgroup_e)

        # Initialize state variables
        self.vm = self.params.EL+randn(self.params.network_group_size)*mV
        self.group_e.g_ampa_b = rand(self.e_size)*self.pyr_params.w_ampa_ext_correct*2.0
        self.group_e.g_nmda = rand(self.e_size)*self.pyr_params.w_nmda*2.0
        self.group_e.g_gaba_a = rand(self.e_size)*self.pyr_params.w_gaba*2.0
        self.group_i.g_ampa_r = rand(self.i_size)*self.inh_params.w_ampa_rec*2.0
        self.group_i.g_ampa_b = rand(self.i_size)*self.inh_params.w_ampa_ext*2.0
        #self.group_i.g_nmda = self.inh_params.w_nmda*100.0+10.0*nS*randn(self.i_size)
        self.group_i.g_nmda = rand(self.i_size)*self.inh_params.w_nmda*2.0
        self.group_i.g_gaba_a = rand(self.i_size)*self.inh_params.w_gaba*2.0


    ## Initialize network connectivity
    def init_connectivity(self, clock):
        self.connections={}
        self.stdp={}

        # Iterate over input groups
        for i in range(self.params.num_groups):

            # E population - recurrent connections
            self.connections['e%d->e%d_ampa' % (i,i)]=init_connection(self.groups_e[i], self.groups_e[i],
                'g_ampa_r', self.pyr_params.w_ampa_rec, self.params.p_e_e, delay=.5*ms, allow_self_conn=False)
            self.connections['e%d->e%d_nmda' % (i,i)]=init_connection(self.groups_e[i], self.groups_e[i],
                'g_nmda', self.pyr_params.w_nmda, self.params.p_e_e, delay=.5*ms, allow_self_conn=False)

        # E -> I excitatory connections
        self.connections['e->i_ampa']=init_connection(self.group_e, self.group_i, 'g_ampa_r', self.inh_params.w_ampa_rec,
            self.params.p_e_i, delay=.5*ms)
        self.connections['e->i_nmda']=init_connection(self.group_e, self.group_i, 'g_nmda', self.inh_params.w_nmda,
            self.params.p_e_i, delay=.5*ms)

        # I -> E - inhibitory connections
        self.connections['i->e_gabaa']=init_connection(self.group_i, self.group_e, 'g_gaba_a', self.pyr_params.w_gaba,
            self.params.p_i_e, delay=.5*ms)

        # I population - recurrent connections
        self.connections['i->i_gabaa']=init_connection(self.group_i, self.group_i, 'g_gaba_a', self.inh_params.w_gaba,
            self.params.p_i_i, delay=.5*ms, allow_self_conn=False)

        if self.background_input is not None:
            # Background -> E+I population connections
            self.connections['b->ampa']=DelayConnection(self.background_input, self, 'g_ampa_b', delay=.5*ms)
            self.connections['b->ampa'][:,:]=0
            for i in xrange(len(self.background_input)):
                if i<self.e_size:
                    self.connections['b->ampa'][i,i]=self.pyr_params.w_ampa_bak
                else:
                    self.connections['b->ampa'][i,i]=self.inh_params.w_ampa_bak

        if self.task_inputs is not None:
            # Task input -> E population connections
            for i in range(self.params.num_groups):
                self.connections['t%d->e%d_ampa' % (i,i)]=DelayConnection(self.task_inputs[i], self.groups_e[i],
                    'g_ampa_x')
                self.connections['t%d->e%d_ampa' % (i,i)].connect_one_to_one(weight=self.pyr_params.w_ampa_ext_correct,
                    delay=.5*ms)
                self.connections['t%d->e%d_ampa' % (i,1-i)]=DelayConnection(self.task_inputs[i], self.groups_e[1-i],
                    'g_ampa_x')
                self.connections['t%d->e%d_ampa' % (i,1-i)].connect_one_to_one(weight=self.pyr_params.w_ampa_ext_incorrect,
                    delay=.5*ms)

                # Input projections plasticity
                self.stdp['stdp%d_%d' % (i,i)] = ExponentialSTDP(self.connections['t%d->e%d_ampa' % (i, i)],
                    self.plasticity_params.tau_pre, self.plasticity_params.tau_post, self.plasticity_params.dA_pre,
                    self.plasticity_params.dA_post, wmax=self.plasticity_params.gmax, update='additive', clock=clock)
                self.stdp['stdp%d_%d' % (i,1-1)] = ExponentialSTDP(self.connections['t%d->e%d_ampa' % (i, 1-i)],
                    self.plasticity_params.tau_pre, self.plasticity_params.tau_post, self.plasticity_params.dA_pre,
                    self.plasticity_params.dA_post, wmax=self.plasticity_params.gmax, update='additive', clock=clock)


def run_wta(wta_params, input_freq, sim_params, pyr_params=pyr_params(), inh_params=inh_params(),
            plasticity_params=plasticity_params(), output_file=None, save_summary_only=False, record_lfp=True,
            record_voxel=True, record_neuron_state=False, record_spikes=True, record_firing_rate=True,
            record_inputs=False, record_connections=None, plot_output=False, report='text'):
    """
    Run WTA network
       wta_params = network parameters
       input_freq = mean firing rate of each input group
       output_file = output file to write to
       save_summary_only = whether or not to save all data or just summary data to file
       record_lfp = record LFP data if true
       record_voxel = record voxel data if true
       record_neuron_state = record neuron state data if true
       record_spikes = record spike data if true
       record_firing_rate = record network firing rates if true
       record_inputs = record input firing rates if true
       plot_output = plot outputs if true
    """

    start_time = time()

    simulation_clock=Clock(dt=sim_params.dt)
    input_update_clock=Clock(dt=1/(wta_params.refresh_rate/Hz)*second)

    background_input=PoissonGroup(wta_params.background_input_size, rates=wta_params.background_freq,
        clock=simulation_clock)
    task_inputs=[]
    for i in range(wta_params.num_groups):
        task_inputs.append(PoissonGroup(wta_params.task_input_size, rates=wta_params.task_input_resting_rate,
                                        clock=simulation_clock))

    # Create WTA network
    wta_network=WTANetworkGroup(params=wta_params, background_input=background_input, task_inputs=task_inputs,
        pyr_params=pyr_params, inh_params=inh_params, plasticity_params=plasticity_params, clock=simulation_clock)

    @network_operation(when='start', clock=input_update_clock)
    def set_task_inputs():
        for idx in range(len(task_inputs)):
            rate=wta_params.task_input_resting_rate
            if sim_params.stim_start_time<=simulation_clock.t<sim_params.stim_end_time:
                rate=input_freq[idx]*Hz+np.random.randn()*wta_params.input_var
                if rate<wta_params.task_input_resting_rate:
                    rate=wta_params.task_input_resting_rate
            task_inputs[idx]._S[0, :]=rate

    @network_operation(clock=simulation_clock)
    def inject_current():
        if simulation_clock.t>sim_params.dcs_start_time:
            wta_network.group_e.I_dcs=sim_params.p_dcs
            wta_network.group_i.I_dcs=sim_params.i_dcs

    # LFP source
    lfp_source=LFPSource(wta_network.group_e, clock=simulation_clock)

    # Create voxel
    voxel=Voxel(simulation_clock, network=wta_network)

    # Create network monitor
    wta_monitor=WTAMonitor(wta_network, lfp_source, voxel, sim_params, record_lfp=record_lfp, record_voxel=record_voxel,
        record_neuron_state=record_neuron_state, record_spikes=record_spikes, record_firing_rate=record_firing_rate,
        record_inputs=record_inputs, record_connections=record_connections, save_summary_only=save_summary_only,
        clock=simulation_clock)

    @network_operation(when='start', clock=simulation_clock)
    def inject_muscimol():
        if sim_params.muscimol_amount>0:
            wta_network.groups_e[sim_params.injection_site].g_muscimol=sim_params.muscimol_amount

    # Create Brian network and reset clock
    net=Network(background_input, task_inputs,set_task_inputs, wta_network, lfp_source, voxel,
        wta_network.connections.values(), wta_monitor.monitors.values(), inject_muscimol, inject_current)
    if sim_params.plasticity:
        net.add(wta_network.stdp.values())
    print "Initialization time: %.2fs" % (time() - start_time)

#    writer=LaTeXDocumentWriter()
#    labels={}
#    labels[voxel]=('v',str(voxel))
#    labels[background_input]=('bi',str(background_input))
#    labels[lfp_source]=('lfp',str(lfp_source))
#    labels[wta_network]=('n',str(wta_network))
#    labels[wta_network.group_e]=('e',str(wta_network.group_e))
#    labels[wta_network.group_i]=('i',str(wta_network.group_i))
#    for i,e_group in enumerate(wta_network.groups_e):
#        labels[e_group]=('e%d' % i,'%s %d' % (str(e_group),i))
#    for i,task_input in enumerate(task_inputs):
#        labels[task_input]=('t%d' % i,'%s %d' % (str(task_input),i))
#    for name,conn in wta_network.connections.iteritems():
#        labels[conn]=(name,str(conn))
#    for name,monitor in wta_monitor.monitors.iteritems():
#        labels[monitor]=(name,str(monitor))
#    writer.document_network(net=net, labels=labels)

    # Run simulation
    start_time = time()
    net.run(sim_params.trial_duration, report=report)
    print "Simulation time: %.2fs" % (time() - start_time)

    # Compute BOLD signal
    if record_voxel:
        start_time = time()
        wta_monitor.monitors['voxel_exc']=get_bold_signal(wta_monitor.monitors['voxel']['G_total_exc'].values[0],
            voxel.params, [500, 2500], sim_params.trial_duration)
        wta_monitor.monitors['voxel']=get_bold_signal(wta_monitor.monitors['voxel']['G_total'].values[0], voxel.params,
            [500, 2500], sim_params.trial_duration)
        print "BOLD generation time: %.2fs" % (time() - start_time)

    # Write output to file
    if output_file is not None:
        start_time = time()
        wta_monitor.write_output(input_freq, output_file)
        print 'Wrote output to %s' % output_file
        print "Write output time: %.2fs" % (time() - start_time)

    # Plot outputs
    if plot_output:
        wta_monitor.plot()

    return wta_monitor

if __name__=='__main__':
    ap = argparse.ArgumentParser(description='Run the WTA model')
    ap.add_argument('--num_groups', type=int, default=2, help='Number of input groups')
    ap.add_argument('--inputs', type=str, default='10,10', help='Input pattern (Hz) - comma-delimited list')
    ap.add_argument('--background', type=float, default=4.0, help='Background firing rate (Hz)')
    ap.add_argument('--trial_duration', type=float, default=4.0, help='Trial duration (seconds)')
    ap.add_argument('--p_e_e', type=float, default=default_params.p_e_e, help='Connection prob from excitatory neurons to excitatory ' \
                                                               'neurons in the same group')
    ap.add_argument('--p_e_i', type=float, default=default_params.p_e_i, help='Connection prob from excitatory neurons to inhibitory ' \
                                                             'neurons in the same group')
    ap.add_argument('--p_i_i', type=float, default=default_params.p_i_i, help='Connection prob from inhibitory neurons to inhibitory ' \
                                                              'neurons in the same group')
    ap.add_argument('--p_i_e', type=float, default=default_params.p_i_e, help='Connection prob from inhibitory neurons to excitatory ' \
                                                              'neurons in other groups')
    ap.add_argument('--output_file', type=str, default=None, help='HDF5 output file')
    ap.add_argument('--muscimol_amount', type=float, default=0.0, help='Amount of muscimol to inject (siemens)')
    ap.add_argument('--injection_site', type=int, default=0, help='Site of muscimol injection (group index)')
    ap.add_argument('--p_dcs', type=float, default=0.0, help='Pyramidal cell DCS (pA)')
    ap.add_argument('--i_dcs', type=float, default=0.0, help='Interneuron cell DCS (pA)')
    ap.add_argument('--refresh_rate', type=float, default=60.0, help='Screen refresh rate (Hz)')
    ap.add_argument('--dcs_start_time', type=float, default=0.0, help='Time to start dcs (s)')
    ap.add_argument('--plasticity', type=int, default=0, help='Include plasticity between inputs and network')
    ap.add_argument('--record_lfp', type=int, default=1, help='Record LFP data')
    ap.add_argument('--record_voxel', type=int, default=1, help='Record voxel data')
    ap.add_argument('--record_neuron_state', type=int, default=0, help='Record neuron state data (synaptic conductances, ' \
                                                                       'membrane potential)')
    ap.add_argument('--record_spikes', type=int, default=1, help='Record neuron spikes')
    ap.add_argument('--record_firing_rate', type=int, default=1, help='Record neuron firing rate')
    ap.add_argument('--record_inputs', type=int, default=0, help='Record network inputs')
    ap.add_argument('--save_summary_only', type=int, default=0, help='Save only summary data')
    ap.add_argument('--plot_output', type=int, default=0, help='Plot data')

    argvals = ap.parse_args()

    input_freq=np.zeros(argvals.num_groups)
    inputs=argvals.inputs.split(',')
    for i in range(argvals.num_groups):
        input_freq[i]=float(inputs[i])

    wta_params=default_params()
    wta_params.p_e_e=argvals.p_e_e
    wta_params.p_e_i=argvals.p_e_i
    wta_params.p_i_i=argvals.p_i_i
    wta_params.p_i_e=argvals.p_i_e
    wta_params.num_groups=argvals.num_groups
    wta_params.background_freq=argvals.background*Hz
    wta_params.refresh_rate=argvals.refresh_rate*Hz

    sim_params=simulation_params()
    sim_params.trial_duration=argvals.trial_duration*second
    sim_params.muscimol_amount=argvals.muscimol_amount*siemens
    sim_params.injection_site=argvals.injection_site
    sim_params.p_dcs=argvals.p_dcs*pA
    sim_params.i_dcs=argvals.i_dcs*pA
    sim_params.dcs_start_time=argvals.dcs_start_time*second

    run_wta(wta_params, input_freq, sim_params, output_file=argvals.output_file, record_lfp=argvals.record_lfp,
        record_voxel=argvals.record_voxel, record_neuron_state=argvals.record_neuron_state,
        record_spikes=argvals.record_spikes, record_firing_rate=argvals.record_firing_rate,
        record_inputs=argvals.record_inputs, record_connections=['t0->e0_ampa'],
        save_summary_only=argvals.save_summary_only, plot_output=argvals.plot_output)
