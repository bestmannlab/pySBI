from brian import pF, nS, mV, DelayConnection, ms, NeuronGroup, siemens, Current, Equations
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse

# Default parameters
from numpy.matlib import randn

class LIP():
    def __init__(self, neuron_group, params):
        self.neuron_group=neuron_group
        self.N=len(neuron_group)
        self.e_contra_size=int(self.N/1.5625)
        self.e_contra_mem_size=int(self.e_contra_size/4)
        self.e_contra_vis_size=self.e_contra_size-self.e_contra_mem_size
        self.e_ipsi_size=int(self.e_contra_size/4)
        self.e_ipsi_mem_size=int(self.e_ipsi_size/4)
        self.e_ipsi_vis_size=self.e_ipsi_size-self.e_ipsi_mem_size
        self.i_contra_size=int(self.e_contra_size/4)
        self.i_ipsi_size=int(self.e_ipsi_size/4)

        self.params=params

        self.init_subpopulations()

        self.connections=[]

        self.init_connectivity()

    def init_subpopulations(self):
        # Main excitatory subpopulation
        self.e_group=self.neuron_group.subgroup(self.e_contra_size+self.e_ipsi_size)
        # regular spiking params (from Naud et al., 2008)
        self.e_group.C=104*pF
        self.e_group.gL=4.3*nS
        self.e_group.EL=-65*mV
        self.e_group.VT=-52*mV
        self.e_group.DeltaT=0.8*mV

        self.e_contra=self.e_group.subgroup(self.e_contra_size)
        self.e_contra_vis=self.e_contra.subgroup(self.e_contra_vis_size)
        self.e_contra_mem=self.e_contra.subgroup(self.e_contra_mem_size)

        self.e_ipsi=self.e_group.subgroup(self.e_ipsi_size)
        self.e_ipsi_vis=self.e_ipsi.subgroup(self.e_ipsi_vis_size)
        self.e_ipsi_mem=self.e_ipsi.subgroup(self.e_ipsi_mem_size)

        self.i_group=self.neuron_group.subgroup(self.i_contra_size+self.i_ipsi_size)
        # fast-spiking interneuron params (from Naud et al., 2008)
        self.i_group.C=59*pF
        self.i_group.gL=2.9*nS
        self.i_group.EL=-62*mV
        self.i_group.VT=-42*mV
        self.i_group.DeltaT=3.0*mV
        self.i_contra=self.i_group.subgroup(self.i_contra_size)
        self.i_ipsi=self.i_group.subgroup(self.i_ipsi_size)

    def init_connectivity(self):
        # Init connections from contralaterally tuned pyramidal cells to other
        # contralaterally tuned pyramidal cells in the same hemisphere
        ec_vis_ec_vis_ampa=DelayConnection(self.e_contra_vis, self.e_contra_vis, 'g_ampa_r', sparseness=self.params.p_ec_vis_ec_vis,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ec_vis_ec_vis_nmda=DelayConnection(self.e_contra_vis, self.e_contra_vis, 'g_nmda', sparseness=self.params.p_ec_vis_ec_vis,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.e_contra_vis)):
            ec_vis_ec_vis_ampa[j,j]=0.0
            ec_vis_ec_vis_ampa.delay[j,j]=0.0
            ec_vis_ec_vis_nmda[j,j]=0.0
            ec_vis_ec_vis_nmda.delay[j,j]=0.0
        self.connections.append(ec_vis_ec_vis_ampa)
        self.connections.append(ec_vis_ec_vis_nmda)

        ec_mem_ec_mem_ampa=DelayConnection(self.e_contra_mem, self.e_contra_mem, 'g_ampa_r', sparseness=self.params.p_ec_mem_ec_mem,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ec_mem_ec_mem_nmda=DelayConnection(self.e_contra_mem, self.e_contra_mem, 'g_nmda', sparseness=self.params.p_ec_mem_ec_mem,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.e_contra_mem)):
            ec_mem_ec_mem_ampa[j,j]=0.0
            ec_mem_ec_mem_ampa.delay[j,j]=0.0
            ec_mem_ec_mem_nmda[j,j]=0.0
            ec_mem_ec_mem_nmda.delay[j,j]=0.0
        self.connections.append(ec_mem_ec_mem_ampa)
        self.connections.append(ec_mem_ec_mem_nmda)

        ec_vis_ec_mem_ampa=DelayConnection(self.e_contra_vis, self.e_contra_mem, 'g_ampa_r', sparseness=self.params.p_ec_vis_ec_mem,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ec_vis_ec_mem_nmda=DelayConnection(self.e_contra_vis, self.e_contra_mem, 'g_nmda', sparseness=self.params.p_ec_vis_ec_mem,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(ec_vis_ec_mem_ampa)
        self.connections.append(ec_vis_ec_mem_nmda)

        # Init connections from ipsilaterally tuned interneurons to contralaterally
        # tuned pyramidal cells in the same hemisphere
        ii_ec_gabaa=DelayConnection(self.i_ipsi, self.e_contra, 'g_gaba_a', sparseness=self.params.p_ii_ec,
            weight=self.params.w_gaba_a, delay=(0*ms, 5*ms))
        ii_ec_gabab=DelayConnection(self.i_ipsi, self.e_contra, 'g_gaba_b', sparseness=self.params.p_ii_ec,
            weight=self.params.w_gaba_b, delay=(0*ms, 5*ms))
        self.connections.append(ii_ec_gabaa)
        self.connections.append(ii_ec_gabab)

        # Init connections from ipsilaterally tuned pyramidal cells to other
        # ipsilaterally tuned pyramidal cells in the same hemisphere
        ei_vis_ei_vis_ampa=DelayConnection(self.e_ipsi_vis, self.e_ipsi_vis, 'g_ampa_r', sparseness=self.params.p_ei_vis_ei_vis,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ei_vis_ei_vis_nmda=DelayConnection(self.e_ipsi_vis, self.e_ipsi_vis, 'g_nmda', sparseness=self.params.p_ei_vis_ei_vis,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.e_ipsi_vis)):
            ei_vis_ei_vis_ampa[j,j]=0.0
            ei_vis_ei_vis_ampa.delay[j,j]=0.0
            ei_vis_ei_vis_nmda[j,j]=0.0
            ei_vis_ei_vis_nmda.delay[j,j]=0.0
        self.connections.append(ei_vis_ei_vis_ampa)
        self.connections.append(ei_vis_ei_vis_nmda)

        ei_mem_ei_mem_ampa=DelayConnection(self.e_ipsi_mem, self.e_ipsi_mem, 'g_ampa_r', sparseness=self.params.p_ei_mem_ei_mem,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ei_mem_ei_mem_nmda=DelayConnection(self.e_ipsi_mem, self.e_ipsi_mem, 'g_nmda', sparseness=self.params.p_ei_mem_ei_mem,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        for j in xrange(len(self.e_ipsi_mem)):
            ei_mem_ei_mem_ampa[j,j]=0.0
            ei_mem_ei_mem_ampa.delay[j,j]=0.0
            ei_mem_ei_mem_nmda[j,j]=0.0
            ei_mem_ei_mem_nmda.delay[j,j]=0.0
        self.connections.append(ei_mem_ei_mem_ampa)
        self.connections.append(ei_mem_ei_mem_nmda)

        ei_vis_ei_mem_ampa=DelayConnection(self.e_ipsi_vis, self.e_ipsi_mem, 'g_ampa_r', sparseness=self.params.p_ei_vis_ei_mem,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ei_vis_ei_mem_nmda=DelayConnection(self.e_ipsi_vis, self.e_ipsi_mem, 'g_nmda', sparseness=self.params.p_ei_vis_ei_mem,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(ei_vis_ei_mem_ampa)
        self.connections.append(ei_vis_ei_mem_nmda)

        # Init connections from contralaterally tuned interneurons to ipsilaterally
        # tuned pyramidal cells in the same hemisphere
        ic_ei_gabaa=DelayConnection(self.i_contra, self.e_ipsi, 'g_gaba_a', sparseness=self.params.p_ic_ei,
            weight=self.params.w_gaba_a, delay=(0*ms, 5*ms))
        ic_ei_gabab=DelayConnection(self.i_contra, self.e_ipsi, 'g_gaba_b', sparseness=self.params.p_ic_ei,
            weight=self.params.w_gaba_b, delay=(0*ms, 5*ms))
        self.connections.append(ic_ei_gabaa)
        self.connections.append(ic_ei_gabab)

        # Init connections from ipsilaterally tuned pyramidal cells to ipsilaterally
        # tuned interneurons in the same hemisphere
        ei_ii_ampa=DelayConnection(self.e_ipsi, self.i_ipsi, 'g_ampa_r', sparseness=self.params.p_ei_ii,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ei_ii_nmda=DelayConnection(self.e_ipsi, self.i_ipsi, 'g_nmda', sparseness=self.params.p_ei_ii,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(ei_ii_ampa)
        self.connections.append(ei_ii_nmda)

        # Init connections from contralaterally tuned pyramidal cells to
        # contralaterally tuned interneurons in the same hemisphere
        ec_ic_ampa=DelayConnection(self.e_contra, self.i_contra, 'g_ampa_r', sparseness=self.params.p_ec_ic,
            weight=self.params.w_ampa_r, delay=(0*ms, 5*ms))
        ec_ic_nmda=DelayConnection(self.e_contra, self.i_contra, 'g_nmda', sparseness=self.params.p_ec_ic,
            weight=self.params.w_nmda, delay=(0*ms, 5*ms))
        self.connections.append(ec_ic_ampa)
        self.connections.append(ec_ic_nmda)


class BrainNetworkGroup(NeuronGroup):

    ### Constructor
    #       N = total number of neurons per input group
    #       num_groups = number of input groups
    #       params = network parameters
    #       background_input = background input source
    #       task_inputs = task input sources
    #       single_inh_pop = single inhibitory population if true
    def __init__(self, lip_size, params, background_inputs=None, visual_cortex_input=None,
                 go_input=None):
        self.lip_size=lip_size
        self.N=2*self.lip_size

        self.params=params
        self.background_inputs=background_inputs
        self.visual_cortex_input=visual_cortex_input
        self.go_input=go_input

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

        # AMPA conductance - go input current
        eqs += exp_synapse('g_ampa_g', params.tau_ampa, siemens)
        eqs += Current('I_ampa_g=g_ampa_g*(E-vm): amp', E=params.E_ampa)

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
        eqs += Equations('g_syn=g_ampa_r+g_ampa_x+g_ampa_g+g_ampa_b+g_V*g_nmda+g_gaba_a+g_gaba_b : siemens')
        eqs += Equations('g_syn_exc=g_ampa_r+g_ampa_x+g_ampa_g+g_ampa_b+g_V*g_nmda : siemens')
        # Total synaptic current
        eqs += Equations('I_abs=abs(I_ampa_r)+abs(I_ampa_b)+abs(I_ampa_x)+abs(I_ampa_g)+abs(I_nmda)+abs(I_gaba_a) : amp')

        NeuronGroup.__init__(self, self.N, model=eqs, threshold=-20*mV, reset=params.EL, compile=True)

        self.init_subpopulations()

        self.connections=[]

        self.init_connectivity()

        if self.background_inputs is not None:
            # Background -> E+I population connectinos
            self.connections.append(DelayConnection(self.background_inputs[0], self.left_lip.neuron_group, 'g_ampa_b',
                sparseness=self.params.p_b_e, weight=self.params.w_ampa_b, delay=(0*ms, 5*ms)))
            self.connections.append(DelayConnection(self.background_inputs[1], self.right_lip.neuron_group, 'g_ampa_b',
                sparseness=self.params.p_b_e, weight=self.params.w_ampa_b, delay=(0*ms, 5*ms)))

        if self.visual_cortex_input is not None:
            # Task input -> E population connections
            self.connections.append(DelayConnection(self.visual_cortex_input[0], self.left_lip.e_contra_vis, 'g_ampa_x',
                sparseness=self.params.p_v_ec_vis, weight=self.params.w_ampa_x, delay=(50*ms, 270*ms)))

            self.connections.append(DelayConnection(self.visual_cortex_input[1], self.right_lip.e_contra_vis, 'g_ampa_x',
                sparseness=self.params.p_v_ec_vis, weight=self.params.w_ampa_x, delay=(50*ms, 270*ms)))

        if self.go_input is not None:
            self.connections.append(DelayConnection(self.go_input, self.left_lip.i_group, 'g_ampa_g',
                sparseness=self.params.p_g_e, weight=self.params.w_ampa_g, delay=(0*ms, 5*ms)))
            self.connections.append(DelayConnection(self.go_input, self.right_lip.i_group, 'g_ampa_g',
                sparseness=self.params.p_g_e, weight=self.params.w_ampa_g, delay=(0*ms, 5*ms)))

    ## Initialize excitatory and inhibitory subpopulations
    def init_subpopulations(self):
        # Main excitatory subpopulation
        self.left_lip=LIP(self.subgroup(self.lip_size), params=self.params)

        self.right_lip=LIP(self.subgroup(self.lip_size), params=self.params)

        # Initialize state variables
        self.vm = self.params.EL+randn(self.N)*10*mV
        self.g_ampa_r = randn(self.N)*self.params.w_ampa_r*.1
        self.g_ampa_b = randn(self.N)*self.params.w_ampa_b*.1
        self.g_ampa_g = randn(self.N)*self.params.w_ampa_g*.1
        self.g_ampa_x = randn(self.N)*self.params.w_ampa_x*.1
        self.g_nmda = randn(self.N)*self.params.w_nmda*.1
        self.g_gaba_a = randn(self.N)*self.params.w_gaba_a*.1
        self.g_gaba_b = randn(self.N)*self.params.w_gaba_b*.1

    def init_connectivity(self):
        # Init connections from contralaterally tuned neurons in the opposite
        # hemisphere to ipsilaterally tuned neurons in this hemisphere
        left_ec_ei_ampa=DelayConnection(self.right_lip.e_contra, self.left_lip.e_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ei, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        left_ec_ei_nmda=DelayConnection(self.right_lip.e_contra, self.left_lip.e_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ei, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(left_ec_ei_ampa)
        self.connections.append(left_ec_ei_nmda)

        # Init connections from contralaterally tuned pyramidal cells in the opposite
        # hemisphere to ipsilaterally tuned interneurons in this hemisphere
        left_ec_ii_ampa=DelayConnection(self.right_lip.e_contra, self.left_lip.i_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ii, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        left_ec_ii_nmda=DelayConnection(self.right_lip.e_contra, self.left_lip.i_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ii, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(left_ec_ii_ampa)
        self.connections.append(left_ec_ii_nmda)

        # Init connections from contralaterally tuned neurons in the opposite
        # hemisphere to ipsilaterally tuned neurons in this hemisphere
        right_ec_ei_ampa=DelayConnection(self.left_lip.e_contra, self.right_lip.e_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ei, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        right_ec_ei_nmda=DelayConnection(self.left_lip.e_contra, self.right_lip.e_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ei, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(right_ec_ei_ampa)
        self.connections.append(right_ec_ei_nmda)

        # Init connections from contralaterally tuned pyramidal cells in the opposite
        # hemisphere to ipsilaterally tuned interneurons in this hemisphere
        right_ec_ii_ampa=DelayConnection(self.left_lip.e_contra, self.right_lip.i_ipsi, 'g_ampa_r',
            sparseness=self.params.p_ec_ii, weight=self.params.w_ampa_r, delay=(10*ms, 20*ms))
        right_ec_ii_nmda=DelayConnection(self.left_lip.e_contra, self.right_lip.i_ipsi, 'g_nmda',
            sparseness=self.params.p_ec_ii, weight=self.params.w_nmda, delay=(10*ms, 20*ms))
        self.connections.append(right_ec_ii_ampa)
        self.connections.append(right_ec_ii_nmda)

        self.connections.extend(self.left_lip.connections)
        self.connections.extend(self.right_lip.connections)


