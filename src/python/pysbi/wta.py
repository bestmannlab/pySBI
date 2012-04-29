import brian_no_units_no_warnings
from brian import *
from brian.library.IF import *
from brian.library.synapses import *
from time import time
import h5py
import argparse

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
    tau_ampa = 10*ms,
    tau1_nmda = 10*ms,
    tau2_nmda = 75*ms,
    tau_gaba_a = 10*ms,
    tau1_gaba_b = 10*ms,
    tau2_gaba_b =100*ms,
    w_ampa_e = 1.5 * nS,
    w_ampa_r=.1*nS,
    w_nmda=0.01*nS,
    w_gaba_a=3.75*nS,
    w_gaba_b=0.1*nS,
    # Connection probabilities
    p_b_e=0.005,
    p_x_e=0.005,
    p_e_e=0.005,#3,
    p_e_i=0.1,
    p_i_i=0.01,#1,
    p_i_e=0.01)#8)

class WTANetworkGroup(NeuronGroup):
    def __init__(self, N, num_groups, params=default_params):
        self.num_groups=num_groups
        eqs = exp_IF(params.C, params.gL, params.EL, params.VT, params.DeltaT)
        eqs += exp_conductance('g_ampa', E=params.E_ampa, tau=params.tau_ampa)
        eqs += biexp_conductance('g_nmda', E=params.E_nmda, tau1=params.tau1_nmda, tau2=params.tau2_nmda)
        eqs += exp_conductance('g_gaba_a', E=params.E_gaba_a, tau=params.tau_gaba_a)
        eqs += biexp_conductance('g_gaba_b', E=params.E_gaba_b, tau1=params.tau1_gaba_b, tau2=params.tau2_gaba_b)
        eqs += Equations('g_syn=g_ampa+g_nmda+g_gaba_a+g_gaba_b : siemens')
        NeuronGroup.__init__(self, N*num_groups, model=eqs, threshold=-20*mV, reset=params.EL, compile=True, freeze=True)

        self.input_groups_e=[]
        self.input_groups_i=[]

        for i in range(num_groups):
            group=self.subgroup(N)
            self.input_groups_e.append(group.subgroup(int(4*N/5)))
            self.input_groups_i.append(group.subgroup(int(N/5)))

        self.vm = params.EL
        self.g_ampa_r = 0
        self.g_ampa_e = 0
        self.g_nmda = 0
        self.g_gaba = 0

        self.connections=[]
        for i in range(num_groups):
            self.connections.append(DelayConnection(self.input_groups_e[i], self.input_groups_e[i], 'g_ampa',
                sparseness=params.p_e_e, weight=params.w_ampa_r, delay=(0*ms, 5*ms)))
            self.connections.append(DelayConnection(self.input_groups_e[i], self.input_groups_e[i], 'g_nmda',
                sparseness=params.p_e_e, weight=params.w_nmda, delay=(0*ms, 5*ms)))
            self.connections.append(DelayConnection(self.input_groups_i[i], self.input_groups_i[i], 'g_gaba_a',
                sparseness=params.p_i_i, weight=params.w_gaba_a, delay=(0*ms, 5*ms)))
            self.connections.append(DelayConnection(self.input_groups_i[i], self.input_groups_i[i], 'g_gaba_b',
                sparseness=params.p_i_i, weight=params.w_gaba_b, delay=(0*ms, 5*ms)))
            self.connections.append(DelayConnection(self.input_groups_e[i], self.input_groups_i[i], 'g_ampa',
                sparseness=params.p_e_i, weight=params.w_ampa_r, delay=(0*ms, 5*ms)))
            for j in range(num_groups):
                if not i==j:
                    self.connections.append(DelayConnection(self.input_groups_i[i], self.input_groups_e[j], 'g_gaba_a',
                        sparseness=params.p_i_e, weight=params.w_gaba_a, delay=(0*ms, 5*ms)))
                    self.connections.append(DelayConnection(self.input_groups_i[i], self.input_groups_e[j], 'g_gaba_b',
                        sparseness=params.p_i_e, weight=params.w_gaba_b, delay=(0*ms, 5*ms)))

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

def get_bold(input_freq, network, voxel, wta_params, num_groups):
    background_rate=10*Hz
    trial_duration=2*second
    stim_start_time=1*second
    stim_end_time=1.5*second
    background_input_size=5000
    task_input_size=2000

    connections=network.connections
    background_input=PoissonGroup(background_input_size, rates=background_rate)
    connections.append(DelayConnection(background_input, network, 'g_ampa', sparseness=wta_params.p_b_e,
        weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms)))

    task_inputs=[]
    for i in range(num_groups):
        task_inputs.append(PoissonGroup(task_input_size, rates=lambda t: (stim_start_time<t<stim_end_time and input_freq[i]) or
                                                                 background_rate))
        connections.append(DelayConnection(task_inputs[i], network.input_groups_e[i], 'g_ampa', sparseness=wta_params.p_x_e,
            weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms)))

    Mb_y = StateMonitor(voxel, 'y', record=0)
    PRe=[]
    for i in range(num_groups):
        PRe.append(PopulationRateMonitor(network.input_groups_e[i]))

    net=Network(background_input, task_inputs, network, voxel, connections, Mb_y, PRe)

    start_time = time()
    net.run(trial_duration, report='text')
    print "Simulation time:", time() - start_time

    return Mb_y[0], PRe

class WTAMonitor():
    def __init__(self, network, voxel, record_voxel=True, record_neuron_state=False, record_spikes=True,
                 record_firing_rate=True):
        self.monitors=[]

        # Voxel monitor
        if record_voxel:
            self.voxel_monitor = MultiStateMonitor(voxel, vars=['s','f_in','v','f_out','q','y'], record=True)
            self.monitors.append(self.voxel_monitor)

        # Network monitor
        if record_neuron_state:
            self.network_monitor = MultiStateMonitor(network, vars=['g_ampa','g_gaba_a','g_nmda','g_gaba_b','vm'], record=True)
            self.monitors.append(self.network_monitor)

        # Population rate monitors
        if record_firing_rate:
            self.population_rate_monitors={'excitatory':[], 'inhibitory':[]}
            for i in range(network.num_groups):
                e_rate_monitor=PopulationRateMonitor(network.input_groups_e[i])
                self.population_rate_monitors['excitatory'].append(e_rate_monitor)
                self.monitors.append(e_rate_monitor)

                i_rate_monitor=PopulationRateMonitor(network.input_groups_i[i])
                self.population_rate_monitors['inhibitory'].append(i_rate_monitor)
                self.monitors.append(i_rate_monitor)

        # Spike monitors
        if record_spikes:
            self.spike_monitors={'excitatory':[], 'inhibitory':[]}
            for i in range(network.num_groups):
                e_spike_monitor=SpikeMonitor(network.input_groups_e[i])
                self.spike_monitors['excitatory'].append(e_spike_monitor)
                self.monitors.append(e_spike_monitor)

                i_spike_monitor=SpikeMonitor(network.input_groups_i[i])
                self.spike_monitors['inhibitory'].append(i_spike_monitor)
                self.monitors.append(i_spike_monitor)

def run_wta(wta_params, num_groups, input_freq, trial_duration, output_file=None, record_voxel=True,
            record_neuron_state=False, record_spikes=True, record_firing_rate=True):
    background_rate=10*Hz
    stim_start_time=1*second
    stim_end_time=1.5*second
    network_group_size=2000
    background_input_size=3000
    task_input_size=1000

    # Create network inputs
    background_input=PoissonGroup(background_input_size, rates=background_rate)
    task_inputs=[]
    for i in range(num_groups):
        task_input=PoissonGroup(task_input_size, rates=lambda t: (stim_start_time<t<stim_end_time and input_freq[i]) or
                                                                 background_rate)
        task_inputs.append(task_input)

    # Create network
    network=WTANetworkGroup(network_group_size, num_groups, params=wta_params)

    # Create connections from input to network
    connections=network.connections
    connections.append(DelayConnection(background_input, network, 'g_ampa', sparseness=wta_params.p_b_e,
        weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms)))
    for i in range(num_groups):
        connections.append(DelayConnection(task_inputs[i], network.input_groups_e[i], 'g_ampa', sparseness=wta_params.p_x_e,
            weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms)))

    # Create voxel
    voxel=get_voxel(28670*nA)
    voxel.G_total = linked_var(network, 'g_syn', func=sum)

    wta_monitor=WTAMonitor(network, voxel, record_voxel=record_voxel, record_neuron_state=record_neuron_state)

    net=Network(background_input, task_inputs, network, voxel, connections, wta_monitor.monitors)

    start_time = time()
    net.run(trial_duration, report='text')
    print "Simulation time:", time() - start_time

    if output_file is not None:
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
        f.attrs['DeltaT'] = wta_params.DeltaT
        f.attrs['E_ampa'] = wta_params.E_ampa
        f.attrs['E_nmda'] = wta_params.E_nmda
        f.attrs['E_gaba_a'] = wta_params.E_gaba_a
        f.attrs['E_gaba_b'] = wta_params.E_gaba_b
        f.attrs['tau_ampa'] = wta_params.tau_ampa
        f.attrs['tau1_nmda'] = wta_params.tau1_nmda
        f.attrs['tau2_nmda'] = wta_params.tau2_nmda
        f.attrs['tau_gaba_a'] = wta_params.tau_gaba_a
        f.attrs['tau1_gaba_b'] = wta_params.tau1_gaba_b
        f.attrs['tau2_gaba_b'] = wta_params.tau2_gaba_b
        f.attrs['w_ampa_e'] = wta_params.w_ampa_e
        f.attrs['w_ampa_r'] = wta_params.w_ampa_r
        f.attrs['w_nmda'] = wta_params.w_nmda
        f.attrs['w_gaba_a'] = wta_params.w_gaba_a
        f.attrs['w_gaba_b'] = wta_params.w_gaba_b
        f.attrs['p_b_e'] = wta_params.p_b_e
        f.attrs['p_x_e'] = wta_params.p_x_e
        f.attrs['p_e_e'] = wta_params.p_e_e
        f.attrs['p_e_i'] = wta_params.p_e_i
        f.attrs['p_i_i'] = wta_params.p_i_i
        f.attrs['p_i_e'] = wta_params.p_i_e

        f_vox=f.create_group('voxel')
        f_vox.attrs['eta']=voxel.eta
        f_vox.attrs['G_base']=voxel.G_base
        f_vox.attrs['tau_f']=voxel.tau_f
        f_vox.attrs['tau_s']=voxel.tau_s
        f_vox.attrs['tau_o']=voxel.tau_o
        f_vox.attrs['e_base']=voxel.e_base
        f_vox.attrs['v_base']=voxel.v_base
        f_vox.attrs['alpha']=voxel.alpha
        f_vox.attrs['k1']=voxel.k1
        f_vox.attrs['k2']=voxel.k2
        f_vox.attrs['k3']=voxel.k3
        if record_voxel:
            f_vox['s']=wta_monitor.voxel_monitor['s'].values
            f_vox['f_in']=wta_monitor.voxel_monitor['f_in'].values
            f_vox['v']=wta_monitor.voxel_monitor['v'].values
            f_vox['q']=wta_monitor.voxel_monitor['q'].values
            f_vox['y']=wta_monitor.voxel_monitor['y'].values


        if record_neuron_state:
            f_state=f.create_group('neuron_state')
            f_state['g_ampa']=wta_monitor.network_monitor['g_ampa'].values
            f_state['g_nmda']=wta_monitor.network_monitor['g_nmda'].values
            f_state['g_gaba_a']=wta_monitor.network_monitor['g_gaba_a'].values
            f_state['g_gaba_b']=wta_monitor.network_monitor['g_gaba_b'].values
            f_state['vm']=wta_monitor.network_monitor['vm'].values

        if record_firing_rate:
            f_rates=f.create_group('firing_rates')
            e_rates=[]
            for rate_monitor in wta_monitor.population_rate_monitors['excitatory']:
                e_rates.append(rate_monitor.smooth_rate(width=5*ms,filter='gaussian'))
            f_rates['e_rates']=np.array(e_rates)

            i_rates=[]
            for rate_monitor in wta_monitor.population_rate_monitors['inhibitory']:
                i_rates.append(rate_monitor.smooth_rate(width=5*ms,filter='gaussian'))
            f_rates['i_rates']=np.array(i_rates)

        if record_spikes:
            f_spikes=f.create_group('spikes')
            for idx,spike_monitor in enumerate(wta_monitor.spike_monitors['excitatory']):
                f_spikes['e.%d.spike_neurons' % idx]=np.array([s[0] for s in spike_monitor.spikes])
                f_spikes['e.%d.spike_times' % idx]=np.array([s[1] for s in spike_monitor.spikes])

            for idx,spike_monitor in enumerate(wta_monitor.spike_monitors['inhibitory']):
                f_spikes['i.%d.spike_neurons' % idx]=np.array([s[0] for s in spike_monitor.spikes])
                f_spikes['i.%d.spike_times' % idx]=np.array([s[1] for s in spike_monitor.spikes])

        f.close()

        print 'Wrote output to %s' % output_file

def test(wta_params):
    num_groups=3
    input_freq=np.array([20, 20, 20])*Hz
    #input_freq=np.array([10, 10, 40])*Hz
    background_rate=10*Hz
    #trial_duration=6*second
    trial_duration=2*second
    stim_start_time=1*second
    stim_end_time=1.5*second
    network_group_size=2000
    background_input_size=3000
    task_input_size=1000

    task_inputs=[]
    background_input=PoissonGroup(background_input_size, rates=background_rate)

    for i in range(num_groups):
        task_input=PoissonGroup(task_input_size, rates=lambda t: (stim_start_time<t<stim_end_time and input_freq[i]) or
                                                                 background_rate)
        task_inputs.append(task_input)

    network=WTANetworkGroup(network_group_size, num_groups, params=wta_params)

    connections=network.connections
    connections.append(DelayConnection(background_input, network, 'g_ampa', sparseness=wta_params.p_b_e,
        weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms)))

    for i in range(num_groups):
        connections.append(DelayConnection(task_inputs[i], network.input_groups_e[i], 'g_ampa', sparseness=wta_params.p_x_e,
            weight=wta_params.w_ampa_e, delay=(0*ms, 5*ms)))

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

    Me_ampa=StateMonitor(network.input_groups_e[0], 'g_ampa', record=0)
    Me_nmda=StateMonitor(network.input_groups_e[0], 'g_nmda', record=0)
    Me_gaba_a=StateMonitor(network.input_groups_e[0], 'g_gaba_a', record=0)
    Me_gaba_b=StateMonitor(network.input_groups_e[0], 'g_gaba_b', record=0)
    monitors.append(Me_ampa)
    monitors.append(Me_nmda)
    monitors.append(Me_gaba_a)
    monitors.append(Me_gaba_b)

    Mi_ampa=StateMonitor(network.input_groups_i[0], 'g_ampa', record=0)
    Mi_nmda=StateMonitor(network.input_groups_i[0], 'g_nmda', record=0)
    Mi_gaba_a=StateMonitor(network.input_groups_i[0], 'g_gaba_a', record=0)
    Mi_gaba_b=StateMonitor(network.input_groups_i[0], 'g_gaba_b', record=0)
    monitors.append(Mi_ampa)
    monitors.append(Mi_nmda)
    monitors.append(Mi_gaba_a)
    monitors.append(Mi_gaba_b)

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
    ax.plot(Me_ampa.times / ms, Me_ampa[0] / nA, label='AMPA')
    ax.plot(Me_nmda.times / ms, Me_nmda[0] / nA, label='NMDA')
    ax.plot(Me_gaba_a.times / ms, Me_gaba_a[0] / nA, label='GABA_A')
    ax.plot(Me_gaba_b.times / ms, Me_gaba_b[0] / nA, label='GABA_B')
    xlabel('Time (ms)')
    ylabel('Conductance (nA)')
    legend()
    ax=subplot(212)
    ax.plot(Mi_ampa.times / ms, Mi_ampa[0] / nA, label='AMPA')
    ax.plot(Mi_nmda.times / ms, Mi_nmda[0] / nA, label='NMDA')
    ax.plot(Mi_gaba_a.times / ms, Mi_gaba_a[0] / nA, label='GABA_A')
    ax.plot(Mi_gaba_b.times / ms, Mi_gaba_b[0] / nA, label='GABA_B')
    xlabel('Time (ms)')
    ylabel('Conductance (nA)')
    legend()

    fig3=figure()
    ax=subplot(211)
    ax.plot(Mb_i.times / ms, Mb_i[0] / nA)
    xlabel('Time (ms)')
    ylabel('Total Synaptic Activity (nA)')
    ax=subplot(212)
    ax.plot(Mb_y.times / ms, Mb_y[0])
    xlabel('Time (ms)')
    ylabel('BOLD')

    show()

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
    ap.add_argument('--output', type=str, default=None, help='HDF5 output file')
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

    run_wta(wta_params, argvals.num_groups, input_freq, argvals.trial_duration, output_file=argvals.output,
        record_voxel=argvals.record_voxel, record_neuron_state=argvals.record_neuron_state,
        record_spikes=argvals.record_spikes, record_firing_rate=argvals.record_firing_rate)
