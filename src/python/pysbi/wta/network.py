from time import time
from brian.clock import reinit_default_clock
from brian.directcontrol import PoissonGroup
from brian.equations import Equations
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse
from brian.membrane_equations import Current
from brian.network import Network, network_operation
from brian.neurongroup import NeuronGroup
from brian.stdunits import pF, nS, mV, ms, Hz
from brian.tools.parameters import Parameters
from brian.units import siemens, second
import argparse
import numpy as np
from numpy.matlib import randn
from pysbi.util.utils import init_connection
from pysbi.voxel import Voxel, LFPSource, get_bold_signal

# Default parameters for a WTA network with multiple inhibitory populations
from pysbi.wta.monitor import WTAMonitor, write_output

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
    #w_nmda_max=0.6*nS,
    w_nmda_max=0.1*nS,
    w_gaba_a_min=0.25*nS,
    w_gaba_a_max=1.2*nS,
    w_gaba_b_min=0.1*nS,
    #w_gaba_b_max=1.0*nS,
    w_gaba_b_max=0.6*nS,
    # Connection probabilities
    p_b_e=0.1,
    p_x_e=0.05,
    p_e_e=0.0075,
    p_e_i=0.04,
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
def run_wta(wta_params, num_groups, input_freq, trial_duration, output_file=None, save_summary_only=False,
            record_lfp=True, record_voxel=True, record_neuron_state=False, record_spikes=True, record_firing_rate=True,
            record_inputs=False, plot_output=False, single_inh_pop=False, muscimol_amount=0*nS, injection_site=0):

    start_time = time()

    # Init simulation parameters
    background_rate=20*Hz
    stim_start_time=.25*second
    stim_end_time=.75*second
    network_group_size=1000
    background_input_size=500
    task_input_size=500

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
        record_inputs=record_inputs, save_summary_only=save_summary_only)

    @network_operation(when='start')
    def inject_muscimol():
        if muscimol_amount>0:
            wta_network.groups_e[injection_site].g_muscimol=muscimol_amount
        if not single_inh_pop:
            wta_network.groups_i[injection_site].g_muscimol=muscimol_amount

    # Create Brian network and reset clock
    if muscimol_amount>0:
        net=Network(background_input, task_inputs, wta_network, lfp_source, voxel, wta_network.connections,
            wta_monitor.monitors, inject_muscimol)
    else:
        net=Network(background_input, task_inputs, wta_network, lfp_source, voxel, wta_network.connections,
            wta_monitor.monitors)
    reinit_default_clock()
    print "Initialization time: %.2fs" % (time() - start_time)

    # Run simulation
    start_time = time()
    net.run(trial_duration, report='text')
    print "Simulation time: %.2fs" % (time() - start_time)

    # Compute BOLD signal
    if record_voxel:
        start_time = time()
        wta_monitor.voxel_exc_monitor=get_bold_signal(wta_monitor.voxel_monitor['G_total_exc'].values[0], voxel.params,
            [500, 2500], trial_duration)
        wta_monitor.voxel_monitor=get_bold_signal(wta_monitor.voxel_monitor['G_total'].values[0], voxel.params,
            [500, 2500], trial_duration)
        print "BOLD generation time: %.2fs" % (time() - start_time)

    # Write output to file
    if output_file is not None:
        start_time = time()
        write_output(background_input_size, background_rate, input_freq, network_group_size, num_groups, single_inh_pop,
            output_file, record_firing_rate, record_neuron_state, record_spikes, record_voxel, record_lfp, record_inputs,
            stim_end_time, stim_start_time, task_input_size, trial_duration, voxel, wta_monitor, wta_params, muscimol_amount,
            injection_site)
        print 'Wrote output to %s' % output_file
        print "Write output time: %.2fs" % (time() - start_time)

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
    ap.add_argument('--save_summary_only', type=int, default=0, help='Save only summary data')

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
        single_inh_pop=argvals.single_inh_pop, muscimol_amount=argvals.muscimol_amount*siemens,
        injection_site=argvals.injection_site, save_summary_only=argvals.save_summary_only)
