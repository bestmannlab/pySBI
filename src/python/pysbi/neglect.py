from brian.connections.delayconnection import DelayConnection
from brian.equations import Equations
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse
from brian.membrane_equations import Current
from brian.neurongroup import NeuronGroup
from brian.stdunits import nS, ms, mV, pF
from brian.tools.parameters import Parameters

# Default parameters for a WTA network with multiple inhibitory populations
from brian.units import siemens
from numpy.matlib import randn

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
    w_ampa_b = 1.0 * nS,
    w_ampa_x = 1.0 * nS,
    w_ampa_r=1.0*nS,
    w_nmda=0.01*nS,
    w_gaba_a=1.0*nS,
    w_gaba_b=0.01*nS,

    # Connection probabilities
    p_v_ec=0.75,
    p_ec_ec=0.01,
    p_ii_ec=0.02,
    p_ec_ei=0.01,
    p_ei_ei=0.01,
    p_ic_ei=0.02,
    p_ei_ii=0.03,
    p_ec_ii=0.02,
    p_ec_ic=0.03)

# WTA network class - extends Brian's NeuronGroup
class BrainNetworkGroup(NeuronGroup):

    ### Constructor
    #       N = total number of neurons per input group
    #       num_groups = number of input groups
    #       params = network parameters
    #       background_input = background input source
    #       task_inputs = task input sources
    #       single_inh_pop = single inhibitory population if true
    def __init__(self, e_pre, i_pre, e_freq, i_freq, v_size, e_contra_size, params=default_params):
        self.e_pre=e_pre
        self.i_pre=i_pre
        self.e_freq=e_freq
        self.i_freq=i_freq
        self.v_size=v_size
        self.e_contra_size=e_contra_size
        self.e_ipsi_size=int(self.e_contra_size/4)
        self.i_contra_size=int(self.e_contra_size/4)
        self.i_ipsi_size=int(self.e_ipsi_size/4)
        self.lip_size=self.e_contra_size+self.e_ipsi_size+self.i_contra_size+self.i_ipsi_size

        self.params=params

        ## Set up equations

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
        eqs += biexp_synapse('g_nmda', params.tau1_nmda, params.tau2_nmda, siemens)
        eqs += Equations('g_V = 1/(1+(Mg/3.57)*exp(-0.062 *vm/mV)) : 1 ', Mg=params.Mg)
        eqs += Current('I_nmda=g_V*g_nmda*(E-vm): amp', E=params.E_nmda)

        # GABA-A conductance
        eqs += exp_synapse('g_gaba_a', params.tau_gaba_a, siemens)
        eqs += Current('I_gaba_a=g_gaba_a*(E-vm): amp', E=params.E_gaba_a)

        # GABA-B conductance
        eqs += biexp_synapse('g_gaba_b', params.tau1_gaba_b, params.tau2_gaba_b, siemens)
        eqs += Current('I_gaba_b=g_gaba_b*(E-vm): amp', E=params.E_gaba_b)

        # Total synaptic conductance
        eqs += Equations('g_syn=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda+g_gaba_a+g_gaba_b : siemens')
        eqs += Equations('g_syn_exc=g_ampa_r+g_ampa_x+g_ampa_b+g_V*g_nmda : siemens')
        # Total synaptic current
        eqs += Equations('I_abs=abs(I_ampa_r)+abs(I_ampa_b)+abs(I_ampa_x)+abs(I_nmda)+abs(I_gaba_a) : amp')

        NeuronGroup.__init__(self, 2*self.lip_size, model=eqs, threshold=-20*mV, reset=params.EL, compile=True)

        self.init_subpopulations()

        self.connections=[]

        self.init_left_hemisphere_connectivity()
        self.init_right_hemisphere_connectivity()

    ## Initialize excitatory and inhibitory subpopulations
    def init_subpopulations(self):
        # Main excitatory subpopulation
        self.left_lip=self.subgroup(self.lip_size)
        self.left_lip_e=self.left_lip.subgroup(self.e_contra_size+self.e_ipsi_size)
        # regular spiking params (from Naud et al., 2008)
        self.left_lip_e.C=104*pF
        self.left_lip_e.gL=4.3*nS
        self.left_lip_e.EL=-65*mV
        self.left_lip_e.VT=-52*mV
        self.left_lip_e.DeltaT=0.8*mV
        self.left_lip_e_contra=self.left_lip_e.subgroup(self.e_contra_size)
        self.left_lip_e_ipsi=self.left_lip_e.subgroup(self.e_ipsi_size)
        self.left_lip_i=self.left_lip.subgroup(self.i_contra_size+self.i_ipsi_size)
        # fast-spiking interneuron params (from Naud et al., 2008)
        self.left_lip_i.C=59*pF
        self.left_lip_i.gL=2.9*nS
        self.left_lip_i.EL=-62*mV
        self.left_lip_i.VT=-42*mV
        self.left_lip_i.DeltaT=3.0*mV
        self.left_lip.i_contra=self.left_lip_i.subgroup(self.i_contra_size)
        self.left_lip.i_ipsi=self.left_lip_i.subgroup(self.i_ipsi_size)

        self.right_lip=self.subgroup(self.lip_size)
        self.right_lip_e=self.right_lip.subgroup(self.e_contra_size+self.e_ipsi_size)
        # regular spiking params (from Naud et al., 2008)
        self.right_lip_e.C=104*pF
        self.right_lip_e.gL=4.3*nS
        self.right_lip_e.EL=-65*mV
        self.right_lip_e.VT=-52*mV
        self.right_lip_e.DeltaT=0.8*mV
        self.right_lip_e_contra=self.right_lip_e.subgroup(self.e_contra_size)
        self.right_lip_e_ipsi=self.right_lip_e.subgroup(self.e_ipsi_size)
        self.right_lip_i=self.right_lip.subgroup(self.i_contra_size+self.i_ipsi_size)
        # fast-spiking interneuron params (from Naud et al., 2008)
        self.right_lip_i.C=59*pF
        self.right_lip_i.gL=2.9*nS
        self.right_lip_i.EL=-62*mV
        self.right_lip_i.VT=-42*mV
        self.right_lip_i.DeltaT=3.0*mV
        self.right_lip.i_contra=self.right_lip_i.subgroup(self.i_contra_size)
        self.right_lip.i_ipsi=self.right_lip_i.subgroup(self.i_ipsi_size)

        # Initialize state variables
        self.vm = self.params.EL+randn(self.N*self.num_groups)*10*mV
        self.g_ampa_r = randn(self.N*self.num_groups)*self.params.w_ampa_r*.1
        self.g_ampa_b = randn(self.N*self.num_groups)*self.params.w_ampa_b*.1
        self.g_ampa_x = randn(self.N*self.num_groups)*self.params.w_ampa_x*.1
        self.g_nmda = randn(self.N*self.num_groups)*self.params.w_nmda*.1
        self.g_gaba_a = randn(self.N*self.num_groups)*self.params.w_gaba_a*.1
        self.g_gaba_b = randn(self.N*self.num_groups)*self.params.w_gaba_b*.1

    def init_left_hemisphere_connectivity(self):
        # Init connections from contralaterally tuned pyramidal cells to other
        # contralaterally tuned pyramidal cells in the same hemisphere
        left_ec_ec_ampa=DelayConnection(self.left_lip_e_contra, self.left_lip_e_contra, 'g_ampa_r',
            sparseness=self.params.p_ec_ec, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        left_ec_ec_nmda=DelayConnection(self.left_lip_e_contra, self.left_lip_e_contra, 'g_nmda',
            sparseness=self.params.p_ec_ec, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.left_lip_e_contra)):
            left_ec_ec_ampa[j,j]=0.0
            left_ec_ec_ampa.delay[j,j]=0.0
            left_ec_ec_nmda[j,j]=0.0
            left_ec_ec_nmda.delay[j,j]=0.0
        self.connections.append(left_ec_ec_ampa)
        self.connections.append(left_ec_ec_nmda)

        # Init connections from ipsilaterally tuned interneurons to contralaterally
        # tuned pyramidal cells in the same hemisphere
        left_ii_ec_gabaa=DelayConnection(self.left_lip_i_ipsi, self.left_lip_e_contra, 'g_gaba_a',
            sparseness=self.params.p_ii_ec, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms))
        left_ii_ec_gabab=DelayConnection(self.left_lip_i_ipsi, self.left_lip_e_contra, 'g_gaba_b',
            sparseness=self.params.p_ii_ec, weight=self.params.w_gaba_b, delay=(0*ms, 5*ms))
        self.connections.append(left_ii_ec_gabaa)
        self.connections.append(left_ii_ec_gabab)

        # Init connections from contralaterally tuned neurons in the opposite
        # hemisphere to ipsilaterally tuned neurons in this hemisphere
        left_ec_ei_ampa=DelayConnection(self.right_lip_e_contra, self.left_lip_e_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ei, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        left_ec_ei_nmda=DelayConnection(self.right_lip_e_contra, self.left_lip_e_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ei, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(left_ec_ei_ampa)
        self.connections.append(left_ec_ei_nmda)

        # Init connections from ipsilaterally tuned pyramidal cells to other
        # ipsilaterally tuned pyramidal cells in the same hemisphere
        left_ei_ei_ampa=DelayConnection(self.left_lip_e_ipsi, self.left_lip_e_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ei_ei, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        left_ei_ei_nmda=DelayConnection(self.left_lip_e_ipsi, self.left_lip_e_ipsi, 'g_nmda',
            sparseness=self.params.p_ei_ei, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.left_lip_e_ipsi)):
            left_ei_ei_ampa[j,j]=0.0
            left_ei_ei_ampa.delay[j,j]=0.0
            left_ei_ei_nmda[j,j]=0.0
            left_ei_ei_nmda.delay[j,j]=0.0
        self.connections.append(left_ei_ei_ampa)
        self.connections.append(left_ei_ei_nmda)

        # Init connections from contralaterally tuned interneurons to ipsilaterally
        # tuned pyramidal cells in the same hemisphere
        left_ic_ei_gabaa=DelayConnection(self.left_lip_i_contra, self.left_lip_e_ipsi, 'g_gaba_a',
            sparseness=self.params.p_ic_ei, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms))
        left_ic_ei_gabab=DelayConnection(self.left_lip_i_contra, self.left_lip_e_ipsi, 'g_gaba_b',
            sparseness=self.params.p_ic_ei, weight=self.params.w_gaba_b, delay=(0*ms, 5*ms))
        self.connections.append(left_ic_ei_gabaa)
        self.connections.append(left_ic_ei_gabab)

        # Init connections from ipsilaterally tuned pyramidal cells to ipsilaterally
        # tuned interneurons in the same hemisphere
        left_ei_ii_ampa=DelayConnection(self.left_lip_e_ipsi, self.left_lip_i_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ei_ii, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        left_ei_ii_nmda=DelayConnection(self.left_lip_e_ipsi, self.left_lip_i_ipsi, 'g_nmda',
            sparseness=self.params.p_ei_ii, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(left_ei_ii_ampa)
        self.connections.append(left_ei_ii_nmda)

        # Init connections from contralaterally tuned pyramidal cells in the opposite
        # hemisphere to ipsilaterally tuned interneurons in this hemisphere
        left_ec_ii_ampa=DelayConnection(self.right_lip_e_contra, self.left_lip_i_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ii, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        left_ec_ii_nmda=DelayConnection(self.right_lip_e_contra, self.left_lip_i_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ii, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(left_ec_ii_ampa)
        self.connections.append(left_ec_ii_nmda)

        # Init connections from contralaterally tuned pyramidal cells to
        # contralaterally tuned interneurons in the same hemisphere
        left_ec_ic_ampa=DelayConnection(self.left_lip_e_contra, self.left_lip_i_contra, 'g_ampa_r',
            sparseness=self.params.p_ec_ic, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        left_ec_ic_nmda=DelayConnection(self.left_lip_e_contra, self.left_lip_i_contra, 'g_nmda',
            sparseness=self.params.p_ec_ic, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(left_ec_ic_ampa)
        self.connections.append(left_ec_ic_nmda)

    def init_right_hemisphere_connectivity(self):
        # Init connections from contralaterally tuned pyramidal cells to other
        # contralaterally tuned pyramidal cells in the same hemisphere
        right_ec_ec_ampa=DelayConnection(self.right_lip_e_contra, self.right_lip_e_contra, 'g_ampa_r',
            sparseness=self.params.p_ec_ec, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        right_ec_ec_nmda=DelayConnection(self.right_lip_e_contra, self.right_lip_e_contra, 'g_nmda',
            sparseness=self.params.p_ec_ec, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.right_lip_e_contra)):
            right_ec_ec_ampa[j,j]=0.0
            right_ec_ec_ampa.delay[j,j]=0.0
            right_ec_ec_nmda[j,j]=0.0
            right_ec_ec_nmda.delay[j,j]=0.0
        self.connections.append(right_ec_ec_ampa)
        self.connections.append(right_ec_ec_nmda)

        # Init connections from ipsilaterally tuned interneurons to contralaterally
        # tuned pyramidal cells in the same hemisphere
        right_ii_ec_gabaa=DelayConnection(self.right_lip_i_ipsi, self.right_lip_e_contra, 'g_gaba_a',
            sparseness=self.params.p_ii_ec, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms))
        right_ii_ec_gabab=DelayConnection(self.right_lip_i_ipsi, self.right_lip_e_contra, 'g_gaba_b',
            sparseness=self.params.p_ii_ec, weight=self.params.w_gaba_b, delay=(0*ms, 5*ms))
        self.connections.append(right_ii_ec_gabaa)
        self.connections.append(right_ii_ec_gabab)

        # Init connections from contralaterally tuned neurons in the opposite
        # hemisphere to ipsilaterally tuned neurons in this hemisphere
        right_ec_ei_ampa=DelayConnection(self.left_lip_e_contra, self.right_lip_e_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ei, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        right_ec_ei_nmda=DelayConnection(self.left_lip_e_contra, self.right_lip_e_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ei, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(right_ec_ei_ampa)
        self.connections.append(right_ec_ei_nmda)

        # Init connections from ipsilaterally tuned pyramidal cells to other
        # ipsilaterally tuned pyramidal cells in the same hemisphere
        right_ei_ei_ampa=DelayConnection(self.right_lip_e_ipsi, self.right_lip_e_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ei_ei, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        right_ei_ei_nmda=DelayConnection(self.right_lip_e_ipsi, self.right_lip_e_ipsi, 'g_nmda',
            sparseness=self.params.p_ei_ei, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.right_lip_e_ipsi)):
            right_ei_ei_ampa[j,j]=0.0
            right_ei_ei_ampa.delay[j,j]=0.0
            right_ei_ei_nmda[j,j]=0.0
            right_ei_ei_nmda.delay[j,j]=0.0
        self.connections.append(right_ei_ei_ampa)
        self.connections.append(right_ei_ei_nmda)

        # Init connections from contralaterally tuned interneurons to ipsilaterally
        # tuned pyramidal cells in the same hemisphere
        right_ic_ei_gabaa=DelayConnection(self.right_lip_i_contra, self.right_lip_e_ipsi, 'g_gaba_a',
            sparseness=self.params.p_ic_ei, weight=self.params.w_gaba_a, delay=(0*ms, 5*ms))
        right_ic_ei_gabab=DelayConnection(self.right_lip_i_contra, self.right_lip_e_ipsi, 'g_gaba_b',
            sparseness=self.params.p_ic_ei, weight=self.params.w_gaba_b, delay=(0*ms, 5*ms))
        self.connections.append(right_ic_ei_gabaa)
        self.connections.append(right_ic_ei_gabab)

        # Init connections from ipsilaterally tuned pyramidal cells to ipsilaterally
        # tuned interneurons in the same hemisphere
        right_ei_ii_ampa=DelayConnection(self.right_lip_e_ipsi, self.right_lip_i_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ei_ii, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        right_ei_ii_nmda=DelayConnection(self.right_lip_e_ipsi, self.right_lip_i_ipsi, 'g_nmda',
            sparseness=self.params.p_ei_ii, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(right_ei_ii_ampa)
        self.connections.append(right_ei_ii_nmda)

        # Init connections from contralaterally tuned pyramidal cells in the opposite
        # hemisphere to ipsilaterally tuned interneurons in this hemisphere
        right_ec_ii_ampa=DelayConnection(self.left_lip_e_contra, self.right_lip_i_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ii, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        right_ec_ii_nmda=DelayConnection(self.left_lip_e_contra, self.right_lip_i_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ii, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(right_ec_ii_ampa)
        self.connections.append(right_ec_ii_nmda)

        # Init connections from contralaterally tuned pyramidal cells to
        # contralaterally tuned interneurons in the same hemisphere
        right_ec_ic_ampa=DelayConnection(self.right_lip_e_contra, self.right_lip_i_contra, 'g_ampa_r',
            sparseness=self.params.p_ec_ic, weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        right_ec_ic_nmda=DelayConnection(self.right_lip_e_contra, self.right_lip_i_contra, 'g_nmda',
            sparseness=self.params.p_ec_ic, weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(right_ec_ic_ampa)
        self.connections.append(right_ec_ic_nmda)

