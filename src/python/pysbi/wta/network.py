from numpy.matlib import randn, rand
from time import time
import brian
from brian.clock import reinit_default_clock
from brian.directcontrol import PoissonGroup
from brian.equations import Equations
from brian.experimental.model_documentation import document_network, labels_from_namespace, LaTeXDocumentWriter
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse
from brian.membrane_equations import Current, InjectedCurrent
from brian.network import Network, network_operation
from brian.neurongroup import NeuronGroup
from brian.stdunits import pF, nS, mV, ms, Hz, pA
from brian.tools.parameters import Parameters
from brian.units import siemens, second
import argparse
import numpy as np
from pysbi.util.utils import init_connection
from pysbi.voxel import Voxel, LFPSource, get_bold_signal

# Default parameters for a WTA network with multiple inhibitory populations
from pysbi.wta.monitor import WTAMonitor, write_output

brian.set_global_preferences(useweave=True,openmp=True,useweave_linear_diffeq =True,
                             gcc_options = ['-ffast-math','-march=native'],usecodegenweave = True,
                             usecodegenreset = True)
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
    #tau1_gaba_b = 10*ms,
    #tau2_gaba_b =100*ms,
    pyr_w_ampa_ext=2.1*nS,
    pyr_w_ampa_rec=0.05*nS,
    int_w_ampa_ext=1.62*nS,
    int_w_ampa_rec=0.04*nS,
    pyr_w_nmda=0.165*nS,
    int_w_nmda=0.13*nS,
    pyr_w_gaba_a=1.3*nS,
    int_w_gaba_a=1.0*nS,
    #w_gaba_b_min=0.1*nS,
    #w_gaba_b_max=0.6*nS,
    # Connection probabilities
    p_b_e=0.03,
    p_x_e=0.01,
    p_e_e=0.03,
    p_e_i=0.08,
    p_i_i=0.2,
    p_i_e=0.08,
)

# WTA network class - extends Brian's NeuronGroup
class WTANetworkGroup(NeuronGroup):

    ### Constructor
    #       N = total number of neurons per input group
    #       num_groups = number of input groups
    #       params = network parameters
    #       background_input = background input source
    #       task_inputs = task input sources
    def __init__(self, N, num_groups, params=default_params, background_input=None, task_inputs=None):
        self.N=N
        self.num_groups=num_groups
        self.params=params
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

        # GABA-B conductance
        #eqs += biexp_synapse('g_gaba_b', params.tau1_gaba_b, params.tau2_gaba_b, siemens)
        #eqs += Current('I_gaba_b=g_gaba_b*(E-vm): amp', E=params.E_gaba_b)

        eqs +=InjectedCurrent('I_dcs: amp')

        # Total synaptic conductance
        #eqs += Equations('g_syn=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda+g_gaba_a+g_gaba_b : siemens')
        eqs += Equations('g_syn=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda+g_gaba_a : siemens')
        eqs += Equations('g_syn_exc=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda : siemens')
        # Total synaptic current
        #eqs += Equations('I_abs=(I_ampa_r**2)**.5+(I_ampa_b**2)**.5+(I_ampa_x**2)**.5+(I_nmda**2)**.5+(I_gaba_a**2)**.5+(I_gaba_b**2)**.5 : amp')
        eqs += Equations('I_abs=(I_ampa_r**2)**.5+(I_ampa_b**2)**.5+(I_ampa_x**2)**.5+(I_nmda**2)**.5+(I_gaba_a**2)**.5 : amp')

        NeuronGroup.__init__(self, N*num_groups, model=eqs, threshold=-20*mV, refractory=1*ms, reset=params.Vr,
            compile=True, freeze=True)

        self.init_subpopulations()

        self.init_connectivity()

    ## Initialize excitatory and inhibitory subpopulations
    def init_subpopulations(self):
        # Main excitatory subpopulation
        e_size=int(self.N*.8)
        self.group_e=self.subgroup(e_size)
        # regular spiking params (from Naud et al., 2008)
        self.group_e.C=104*pF
        self.group_e.gL=4.3*nS
        self.group_e.EL=-65*mV
        self.group_e.VT=-52*mV
        self.group_e.DeltaT=0.8*mV
        #self.group_e.reset=-53*mV
        #self.group_e._refractory_time=2*ms

        # Main inhibitory subpopulation
        i_size=int(self.N*.2)
        self.group_i=self.subgroup(i_size)
        # continuous non-adapting interneuron params (from Naud et al., 2008)
        self.group_i.C=59*pF
        self.group_i.gL=2.9*nS
        self.group_i.EL=-62*mV
        self.group_i.VT=-42*mV
        self.group_i.DeltaT=3.0*mV
        #self.group_i.reset=-54*mV
        #self.group_i._refractory_time=1*ms

        # Input-specific sub-subpopulations
        self.groups_e=[]
        for i in range(self.num_groups):
            subgroup_e=self.group_e.subgroup(int(e_size/self.num_groups))
            self.groups_e.append(subgroup_e)

        # Initialize state variables
        self.vm = self.params.EL+randn(self.N*self.num_groups)*10*mV
        self.group_e.g_ampa_r = rand(e_size)*self.params.pyr_w_ampa_rec*.01
        self.group_e.g_ampa_b = rand(e_size)*self.params.pyr_w_ampa_ext*.01
        self.group_e.g_ampa_x = rand(e_size)*self.params.pyr_w_ampa_ext*.01
        self.group_e.g_nmda = rand(e_size)*self.params.pyr_w_nmda*.01
        self.group_e.g_gaba_a = rand(e_size)*self.params.pyr_w_gaba_a*.01
        self.group_i.g_ampa_r = rand(i_size)*self.params.int_w_ampa_rec*.01
        self.group_i.g_ampa_b = rand(i_size)*self.params.int_w_ampa_ext*.01
        self.group_i.g_ampa_x = rand(i_size)*self.params.int_w_ampa_ext*.01
        self.group_i.g_nmda = rand(i_size)*self.params.int_w_nmda*.01
        self.group_i.g_gaba_a = rand(i_size)*self.params.int_w_gaba_a*.01
#        self.g_gaba_b = self.params.w_gaba_a_min+randn(self.N*self.num_groups)*(self.params.w_gaba_a_max-self.params.w_gaba_a_min)*.1


    ## Initialize network connectivity
    def init_connectivity(self):
        self.connections={}

        # Iterate over input groups
        for i in range(self.num_groups):

            # E population - recurrent connections
            self.connections['e%d->e%d_ampa' % (i,i)]=init_connection(self.groups_e[i], self.groups_e[i],
                'g_ampa_r', self.params.pyr_w_ampa_rec, self.params.p_e_e, .5*ms, allow_self_conn=False)
            self.connections['e%d->e%d_nmda' % (i,i)]=init_connection(self.groups_e[i], self.groups_e[i],
                'g_nmda', self.params.pyr_w_nmda, self.params.p_e_e, .5*ms, allow_self_conn=False)

            # E -> I excitatory connections
            self.connections['e%d->i_ampa' % i]=init_connection(self.groups_e[i], self.group_i, 'g_ampa_r',
                self.params.int_w_ampa_rec, self.params.p_e_i, .5*ms)
            self.connections['e%d->i_nmda' % i]=init_connection(self.groups_e[i], self.group_i, 'g_nmda',
                self.params.int_w_nmda, self.params.p_e_i, .5*ms)

            # I -> E - inhibitory connections
            self.connections['i->e%d_gabaa' % i]=init_connection(self.group_i, self.groups_e[i], 'g_gaba_a',
                self.params.pyr_w_gaba_a, self.params.p_i_e, .5*ms)
            #self.connections.append(init_rand_weight_connection(self.group_i, self.groups_e[i], 'g_gaba_b',
            #    self.params.w_gaba_b_min, self.params.w_gaba_b_max, self.params.p_i_e, (0*ms, 5*ms)))

        # I population - recurrent connections
        self.connections['i->i_gabaa']=init_connection(self.group_i, self.group_i, 'g_gaba_a', self.params.int_w_gaba_a,
            self.params.p_i_i, .5*ms, allow_self_conn=False)
        #self.connections.append(init_rand_weight_connection(self.group_i, self.group_i, 'g_gaba_b', self.params.w_gaba_b_min,
        #    self.params.w_gaba_b_max, self.params.p_i_i, (0*ms, 5*ms), allow_self_conn=False))

        if self.background_input is not None:
            # Background -> E+I population connections
            self.connections['b->e_ampa']=init_connection(self.background_input, self.group_e, 'g_ampa_b',
                self.params.pyr_w_ampa_ext, self.params.p_b_e, .5*ms)
            self.connections['b->i_ampa']=init_connection(self.background_input, self.group_i, 'g_ampa_b',
                self.params.int_w_ampa_ext, self.params.p_b_e, .5*ms)

        if self.task_inputs is not None:
            # Task input -> E population connections
            for i in range(self.num_groups):
                self.connections['t%d->e%d_ampa' % (i,i)]=init_connection(self.task_inputs[i], self.groups_e[i],
                    'g_ampa_x', self.params.pyr_w_ampa_ext, self.params.p_x_e, .5*ms)



def run_wta(wta_params, num_groups, input_freq, trial_duration, background_freq=5, output_file=None,
            save_summary_only=False, record_lfp=True, record_voxel=True, record_neuron_state=False, record_spikes=True,
            record_firing_rate=True, record_inputs=False, plot_output=False, muscimol_amount=0*nS, injection_site=0,
            p_dcs=0*pA, i_dcs=0*pA, report='text'):
    """
    Run WTA network
       wta_params = network parameters
       num_groups = number of input groups
       input_freq = mean firing rate of each input group
       trial_duration = how long to simulate
       output_file = output file to write to
       save_summary_only = whether or not to save all data or just summary data to file
       record_lfp = record LFP data if true
       record_voxel = record voxel data if true
       record_neuron_state = record neuron state data if true
       record_spikes = record spike data if true
       record_firing_rate = record network firing rates if true
       record_inputs = record input firing rates if true
       plot_output = plot outputs if true
       muscimol_amount = amount of muscimol to inject
       injection_site = where to inject muscimol
       p_dcs = DCS to pyramidal cells
       i_dcs = DCS to interneurons
    """

    start_time = time()

    # Init simulation parameters
    stim_start_time=1*second
    stim_end_time=trial_duration-1*second

    # Total size of the network (excitatory and inhibitory cells)
    network_group_size=2000
    background_input_size=5000
    task_input_size=1000

    # Create network inputs
    def make_task_rate_function(rate):
        return lambda t: ((stim_start_time<t<stim_end_time and background_freq*Hz+rate+20*np.random.randn()*Hz) or background_freq*Hz)
    def make_background_rate_function(rate):
        return lambda t: 5*np.random.randn(background_input_size)*Hz+rate
    background_input=PoissonGroup(background_input_size, rates=make_background_rate_function(background_freq*Hz))
    task_inputs=[]
    for i in range(num_groups):
        rate=input_freq[i]*Hz
        task_inputs.append(PoissonGroup(task_input_size, rates=make_task_rate_function(rate)))

    # Create WTA network
    wta_network=WTANetworkGroup(network_group_size, num_groups, params=wta_params, background_input=background_input,
        task_inputs=task_inputs)

    #wta_network.group_e.EL+=p_dcs
    #wta_network.group_i.EL+=i_dcs
    @network_operation
    def inject_current(c):
        wta_network.group_e.I_dcs=p_dcs
        wta_network.group_i.I_dcs=i_dcs

    # LFP source
    lfp_source=LFPSource(wta_network.group_e)

    # Create voxel
    voxel=Voxel(network=wta_network)

    # Create network monitor
    wta_monitor=WTAMonitor(wta_network, lfp_source, voxel, record_lfp=record_lfp, record_voxel=record_voxel,
        record_neuron_state=record_neuron_state, record_spikes=record_spikes, record_firing_rate=record_firing_rate,
        record_inputs=record_inputs, save_summary_only=save_summary_only)

    @network_operation(when='start')
    def inject_muscimol():
        if muscimol_amount>0:
            wta_network.groups_e[injection_site].g_muscimol=muscimol_amount

    # Create Brian network and reset clock
    net=Network(background_input, task_inputs, wta_network, lfp_source, voxel, wta_network.connections.values(),
        wta_monitor.monitors.values(), inject_muscimol, inject_current)
    reinit_default_clock()
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
    net.run(trial_duration, report=report)
    print "Simulation time: %.2fs" % (time() - start_time)

    # Compute BOLD signal
    if record_voxel:
        start_time = time()
        wta_monitor.monitors['voxel_exc']=get_bold_signal(wta_monitor.monitors['voxel']['G_total_exc'].values[0],
            voxel.params, [500, 2500], trial_duration)
        wta_monitor.monitors['voxel']=get_bold_signal(wta_monitor.monitors['voxel']['G_total'].values[0], voxel.params,
            [500, 2500], trial_duration)
        print "BOLD generation time: %.2fs" % (time() - start_time)

    # Write output to file
    if output_file is not None:
        start_time = time()
        write_output(background_input_size, background_freq, input_freq, network_group_size, num_groups, output_file,
            record_firing_rate, record_neuron_state, record_spikes, record_voxel, record_lfp, record_inputs,
            stim_end_time, stim_start_time, task_input_size, trial_duration, voxel, wta_monitor, wta_params,
            muscimol_amount, injection_site, p_dcs, i_dcs)
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
    ap.add_argument('--muscimol_amount', type=float, default=0.0, help='Amount of muscimol to inject')
    ap.add_argument('--injection_site', type=int, default=0, help='Site of muscimol injection (group index)')
    ap.add_argument('--p_dcs', type=float, default=0.0, help='Pyramidal cell DCS')
    ap.add_argument('--i_dcs', type=float, default=0.0, help='Interneuron cell DCS')
    ap.add_argument('--record_lfp', type=int, default=1, help='Record LFP data')
    ap.add_argument('--record_voxel', type=int, default=1, help='Record voxel data')
    ap.add_argument('--record_neuron_state', type=int, default=0, help='Record neuron state data (synaptic conductances, ' \
                                                                       'membrane potential)')
    ap.add_argument('--record_spikes', type=int, default=1, help='Record neuron spikes')
    ap.add_argument('--record_firing_rate', type=int, default=1, help='Record neuron firing rate')
    ap.add_argument('--record_inputs', type=int, default=0, help='Record network inputs')
    ap.add_argument('--save_summary_only', type=int, default=0, help='Save only summary data')

    argvals = ap.parse_args()

    input_freq=np.zeros(argvals.num_groups)
    inputs=argvals.inputs.split(',')
    for i in range(argvals.num_groups):
        input_freq[i]=float(inputs[i])

    wta_params=default_params()
    wta_params.p_b_e=argvals.p_b_e
    wta_params.p_x_e=argvals.p_x_e
    wta_params.p_e_e=argvals.p_e_e
    wta_params.p_e_i=argvals.p_e_i
    wta_params.p_i_i=argvals.p_i_i
    wta_params.p_i_e=argvals.p_i_e

    run_wta(wta_params, argvals.num_groups, input_freq, argvals.trial_duration*second,
        background_freq=argvals.background, output_file=argvals.output_file,
        record_lfp=argvals.record_lfp, record_voxel=argvals.record_voxel,
        record_neuron_state=argvals.record_neuron_state, record_spikes=argvals.record_spikes,
        record_firing_rate=argvals.record_firing_rate, record_inputs=argvals.record_inputs,
        muscimol_amount=argvals.muscimol_amount*siemens, injection_site=argvals.injection_site,
        p_dcs=argvals.p_dcs*pA, i_dcs=argvals.i_dcs*pA, save_summary_only=argvals.save_summary_only)
