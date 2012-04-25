import brian_no_units_no_warnings
from brian import *
from brian.library.IF import *
from brian.library.synapses import *
from time import time

default_params=Parameters(
    # Neuron parameters
    C = 200 * pF,
    gL = 20 * nS,
    EL = -70 * mV,
    VT = -55 * mV,
    DeltaT = 3 * mV,
    # Synapse parameters
    E_ampa = 0 * mvolt,
    E_nmda = 0 * mvolt,
    E_gaba_a = -70 * mvolt,
    E_gaba_b = -95 * mvolt,
    tau_ampa = 5*ms,
    tau1_nmda = 10*ms,
    tau2_nmda = 75*ms,
    tau_gaba_a = 10*ms,
    tau1_gaba_b = 10*ms,
    tau2_gaba_b =100*ms,
    w_ampa_e = 1.5 * nS,
    w_ampa_r=.1*nS,
    w_nmda=0.01*nS,
    w_gaba_a=3.75*nS,
    w_gaba_b=0.01*nS,
    # Connection probabilities
    p_b_e=0.01,
    p_x_e=0.05,
    p_e_e=0.02,
    p_e_i=0.01,
    p_i_i=0.01,
    p_i_e=0.02)

class WTANetworkGroup(NeuronGroup):
    def __init__(self, N, num_groups, params=default_params):
        eqs = exp_IF(params.C, params.gL, params.EL, params.VT, params.DeltaT)
        eqs += exp_conductance('g_ampa', E=params.E_ampa, tau=params.tau_ampa)
        eqs += biexp_conductance('g_nmda', E=params.E_nmda, tau1=params.tau1_nmda, tau2=params.tau2_nmda)
        eqs += exp_conductance('g_gaba_a', E=params.E_gaba_a, tau=params.tau_gaba_a)
        eqs += biexp_conductance('g_gaba_b', E=params.E_gaba_b, tau1=params.tau1_gaba_b, tau2=params.tau2_gaba_b)
        eqs += Equations('g_syn=g_ampa+g_nmda+g_gaba_a+g_gaba_b : siemens')
        #eqs += Equations('g_syn=g_ampa+g_gaba_a : siemens')
        NeuronGroup.__init__(self, N*num_groups, model=eqs, threshold=-20*mV, reset=params.EL, refractory=2*ms,
            compile=True, freeze=True)

        self.input_groups_e=[]
        self.input_groups_i=[]

        for i in range(num_groups):
            group=self.subgroup(N)
            group_e = group.subgroup(int(4*N/5))
            group_i = group.subgroup(int(N/5))
            self.input_groups_e.append(group_e)
            self.input_groups_i.append(group_i)

        self.vm = params.EL
        self.g_ampa_r = 0
        self.g_ampa_e = 0
        self.g_nmda = 0
        self.g_gaba = 0

        self.connections=[]
        for i in range(num_groups):
            self.connections.append(Connection(self.input_groups_e[i], self.input_groups_e[i], 'g_ampa',
                sparseness=params.p_e_e, weight=params.w_ampa_r, delay=(0*ms, 5*ms), max_delay=5*ms))
            #self.connections.append(Connection(self.input_groups_e[i], self.input_groups_e[i], 'g_nmda',
            #    sparseness=params.p_e_e, weight=params.w_nmda, delay=(0*ms, 5*ms), max_delay=5*ms))
            self.connections.append(Connection(self.input_groups_i[i], self.input_groups_i[i], 'g_gaba_a',
                sparseness=params.p_i_i, weight=params.w_gaba_a, delay=(0*ms, 5*ms), max_delay=5*ms))
            #self.connections.append(Connection(self.input_groups_i[i], self.input_groups_i[i], 'g_gaba_b',
            #    sparseness=params.p_i_i, weight=params.w_gaba_b, delay=(0*ms, 5*ms), max_delay=5*ms))
            self.connections.append(Connection(self.input_groups_e[i], self.input_groups_i[i], 'g_ampa',
                sparseness=params.p_e_i, weight=params.w_ampa_r, delay=(0*ms, 5*ms), max_delay=5*ms))
            #self.connections.append(Connection(self.input_groups_e[i], self.input_groups_i[i], 'g_nmda',
            #    sparseness=params.p_e_i, weight=params.w_nmda, delay=(0*ms, 5*ms), max_delay=5*ms))
            for j in range(num_groups):
                if not i==j:
                    self.connections.append(Connection(self.input_groups_i[i], self.input_groups_e[j], 'g_gaba_a',
                        sparseness=params.p_i_e, weight=params.w_gaba_a, delay=(0*ms, 5*ms), max_delay=5*ms))
                    #self.connections.append(Connection(self.input_groups_i[i], self.input_groups_e[j], 'g_gaba_b',
                    #    sparseness=params.p_i_e, weight=params.w_gaba_a, delay=(0*ms, 5*ms), max_delay=5*ms))
        self.contained_objects.extend(self.connections)

def get_voxel(syn_baseline, B0=4.7, TE=.02):
    eqs=Equations('''
        G_total                                                                        : amp
        ds/dt=eta*(G_total-G_base)/G_base-s/tau_s-(f_in-1.0)/tau_f                     : 1
        df_in/dt=s/second                                                              : 1
        dv/dt=1/tau_o*(f_in-f_out)                                                     : 1
        f_out=v**(1.0/alpha)                                                           : 1
        o_e=1-(1-e_base)**(1/f_in)                                                     : 1
        dq/dt=1/tau_o*((f_in*o_e/e_base)-f_out*q/v)                                    : 1
        y=v_base*((k1+k2)*(1-q)-(k2+k3)*(1-v))                                         : 1
        G_base                                                                         : amp
        eta                                                                            : 1/second
        tau_s                                                                          : second
        tau_f                                                                          : second
        alpha                                                                          : 1
        tau_o                                                                          : second
        e_base                                                                         : 1
        v_base                                                                         : 1
        k1                                                                             : 1
        k2                                                                             : 1
        k3                                                                             : 1
        ''')
    #
    voxel=NeuronGroup(1, model=eqs, compile=True, freeze=True)
    voxel.f_in=1
    voxel.G_base=syn_baseline
    # synaptic efficacy (value from Zheng et al., 2002)
    voxel.eta=0.5*second
    # signal decay time constant (value from Zheng et al., 2002)
    voxel.tau_s=0.8*second
    # autoregulatory feedback time constant (value from Zheng et al., 2002)
    voxel.tau_f=0.4*second
    # Grubb's parameter
    voxel.alpha=0.2
    # venous time constant (value from Friston et al., 2000)
    voxel.tau_o=1*second
    # resting net oxygen extraction fraction by capillary bed (value
    # from Friston et al., 2000)
    voxel.e_base=0.8
    # resting blood volume fraction (value from Friston et al., 2000)
    voxel.v_base=0.02

    # MR parameters (from Obata et al, 2004)
    # magnetic field dependent frequency offset (from Behzadi & Liu, 2005)
    freq_offset=40.3*(B0/1.5)
    voxel.k1=4.3*freq_offset*voxel.e_base*TE
    # resting intravascular transverse relaxation time (41.4ms at 4T, from
    # Yacoub et al, 2001)
    T_2E=.0414
    # resting extravascular transverse relaxation time (23.5ms at 4T from
    # Yacoub et al, 2001)
    T_2I=.0235
    # effective intravascular spin density (assumed to be equal to
    # extravascular spin density, from Behzadi & Liu, 2005)
    s_e_0=1
    # effective extravascular spin density (assumed to be equal to
    # intravascular spin density, from Behzadi & Liu, 2005)
    s_i_0=1
    # blood signal
    s_e=s_e_0*exp(-TE/T_2E)
    # tissue signal
    s_i=s_i_0*exp(-TE/T_2I)
    # instrinsic ratio of blood to tissue signals at rest
    beta=s_e/s_i
    # slope of the intravascular relaxation rate versus extraction fraction
    # (from Behzadi & Liu, 2005, 25s^-1 is measured value at 1.5T)
    r_0=25*(B0/1.5)**2
    voxel.k2=beta*r_0*voxel.e_base*TE
    voxel.k3=beta-1

    voxel.s=0
    voxel.f_in=1
    voxel.f_out=1
    voxel.v=1
    voxel.q=1
    voxel.y=0
    return voxel

class WTABOLDResult():
    def __init__(self, contrast, valid):
        self.contrast=contrast
        self.valid=valid

def get_bold(input_freq, network, voxel, wta_params, num_groups):
    background_rate=10*Hz
    trial_duration=2*second
    stim_start_time=1*second
    stim_end_time=1.5*second
    background_input_size=5000
    task_input_size=2000

    connections=[]
    background_input=PoissonGroup(background_input_size, rates=background_rate)
    connections.append(Connection(background_input, network, 'g_ampa', sparseness=wta_params.p_b_e,
        weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms), max_delay=5*ms))

    task_inputs=[]
    for i in range(num_groups):
        task_input=PoissonGroup(task_input_size, rates=lambda t: (stim_start_time<t<stim_end_time and input_freq[i]) or
                                                                 background_rate)
        connections.append(Connection(task_input, network.input_groups_e[i], 'g_ampa', sparseness=wta_params.p_x_e,
            weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms), max_delay=5*ms))
        task_inputs.append(task_input)


    Mb_y = StateMonitor(voxel, 'y', record=0)
    PRe=[]
    for i in range(num_groups):
        PRe.append(PopulationRateMonitor(network.input_groups_e[i]))

    net=Network(background_input, task_inputs, network, voxel, connections, Mb_y, PRe)

    start_time = time()
    net.run(trial_duration, report='text')
    print "Simulation time:", time() - start_time

    return Mb_y[0], PRe

def test(wta_params):
    num_groups=3
    #input_freq=[20, 20, 20]
    input_freq=np.array([10, 10, 40])*Hz
    background_rate=10*Hz
    #trial_duration=6*second
    trial_duration=2*second
    stim_start_time=1*second
    stim_end_time=1.5*second
    network_group_size=4000
    background_input_size=5000
    task_input_size=2000

    task_inputs=[]
    connections=[]
    background_input=PoissonGroup(background_input_size, rates=background_rate)

    for i in range(num_groups):
        task_input=PoissonGroup(task_input_size, rates=lambda t: (stim_start_time<t<stim_end_time and input_freq[i]) or
                                                                 background_rate)
        task_inputs.append(task_input)

    network=WTANetworkGroup(network_group_size, num_groups, params=wta_params)

    connections.append(Connection(background_input, network, 'g_ampa', sparseness=wta_params.p_b_e,
        weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms), max_delay=5*ms))

    for i in range(num_groups):
        connections.append(Connection(task_inputs[i], network.input_groups_e[i], 'g_ampa', sparseness=wta_params.p_x_e,
            weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms), max_delay=5*ms))

    voxel=get_voxel(28670*nA)
    voxel.G_total = linked_var(network, 'g_syn', func=sum)

    monitors=[]
    PRe=[]
    PRi=[]
    Me=[]
    Mi=[]
    for i in range(num_groups):
        PRe.append(PopulationRateMonitor(network.input_groups_e[i]))
        PRi.append(PopulationRateMonitor(network.input_groups_i[i]))
        Me.append(SpikeMonitor(network.input_groups_e[i]))
        Mi.append(SpikeMonitor(network.input_groups_i[i]))
    monitors.extend(PRe)
    monitors.extend(PRi)
    monitors.extend(Me)
    monitors.extend(Mi)

    Mb_i = StateMonitor(voxel, 'G_total', record=0)
    Mb_y = StateMonitor(voxel, 'y', record=0)

    monitors.append(Mb_i)
    monitors.append(Mb_y)

    net=Network(background_input, task_inputs, network, voxel, connections, monitors)

    start_time = time()
    net.run(trial_duration, report='text')
    print "Simulation time:", time() - start_time

    fig1=figure()
    ax=subplot(311)
    raster_plot(Me[0],Mi[0],Me[1],Mi[1],Me[2],Mi[2],newfigure=False)
    ax=subplot(312)
    ax.plot(PRe[0].times / ms, PRe[0].smooth_rate(width=5*ms,filter='gaussian') / hertz)
    ax.plot(PRe[1].times / ms, PRe[1].smooth_rate(width=5*ms,filter='gaussian') / hertz)
    ax.plot(PRe[2].times / ms, PRe[2].smooth_rate(width=5*ms,filter='gaussian') / hertz)
    ax=subplot(313)
    ax.plot(PRi[0].times / ms, PRi[0].smooth_rate(width=5*ms,filter='gaussian') / hertz)
    ax.plot(PRi[1].times / ms, PRi[1].smooth_rate(width=5*ms,filter='gaussian') / hertz)
    ax.plot(PRi[2].times / ms, PRi[2].smooth_rate(width=5*ms,filter='gaussian') / hertz)
    fig1.show()

    fig2=figure()
    ax=subplot(211)
    ax.plot(Mb_i.times / ms, Mb_i[0] / nA)
    xlabel('Time (ms)')
    ylabel('Total Synaptic Activity (nA)')
    ax=subplot(212)
    ax.plot(Mb_y.times / ms, Mb_y[0])
    xlabel('Time (ms)')
    ylabel('BOLD')

    show()

