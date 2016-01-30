import numpy as np
from brian import Clock, ms, second, siemens, Current, Equations, InjectedCurrent, pA, NeuronGroup, mV, network_operation, StateMonitor, Network
from brian.library.IF import exp_IF
from brian.library.synapses import exp_synapse, biexp_synapse
from matplotlib import pyplot as plt
from pysbi.wta.network import default_params, pyr_params, inh_params


def test_stim_pyramidal_impact():
    simulation_clock=Clock(dt=.5*ms)
    trial_duration=1*second
    dcs_start_time=.5*second

    stim_levels=[-8,-6,-4,-2,-1,-.5,-.25,0,.25,.5,1,2,4,6,8]
    voltages = np.zeros(len(stim_levels))
    for idx,stim_level in enumerate(stim_levels):
        print('testing stim_level %.3fpA' % stim_level)
        eqs = exp_IF(default_params.C, default_params.gL, default_params.EL, default_params.VT, default_params.DeltaT)

        # AMPA conductance - recurrent input current
        eqs += exp_synapse('g_ampa_r', default_params.tau_ampa, siemens)
        eqs += Current('I_ampa_r=g_ampa_r*(E-vm): amp', E=default_params.E_ampa)

        # AMPA conductance - background input current
        eqs += exp_synapse('g_ampa_b', default_params.tau_ampa, siemens)
        eqs += Current('I_ampa_b=g_ampa_b*(E-vm): amp', E=default_params.E_ampa)

        # AMPA conductance - task input current
        eqs += exp_synapse('g_ampa_x', default_params.tau_ampa, siemens)
        eqs += Current('I_ampa_x=g_ampa_x*(E-vm): amp', E=default_params.E_ampa)

        # Voltage-dependent NMDA conductance
        eqs += biexp_synapse('g_nmda', default_params.tau1_nmda, default_params.tau2_nmda, siemens)
        eqs += Equations('g_V = 1/(1+(Mg/3.57)*exp(-0.062 *vm/mV)) : 1 ', Mg=default_params.Mg)
        eqs += Current('I_nmda=g_V*g_nmda*(E-vm): amp', E=default_params.E_nmda)

        # GABA-A conductance
        eqs += exp_synapse('g_gaba_a', default_params.tau_gaba_a, siemens)
        eqs += Current('I_gaba_a=g_gaba_a*(E-vm): amp', E=default_params.E_gaba_a)

        eqs +=InjectedCurrent('I_dcs: amp')

        group=NeuronGroup(1, model=eqs, threshold=-20*mV, refractory=pyr_params.refractory, reset=default_params.Vr,
            compile=True, freeze=True, clock=simulation_clock)
        group.C=pyr_params.C
        group.gL=pyr_params.gL

        @network_operation(clock=simulation_clock)
        def inject_current(c):
            if simulation_clock.t>dcs_start_time:
                group.I_dcs=stim_level*pA
        monitor=StateMonitor(group, 'vm', simulation_clock, record=True)
        net=Network(group, monitor, inject_current)
        net.run(trial_duration, report='text')
        voltages[idx]=monitor.values[0,-1]*1000

    voltages=voltages-voltages[7]
    plt.figure()
    plt.plot(stim_levels,voltages)
    plt.xlabel('Stimulation level (pA)')
    plt.ylabel('Voltage Change (mV)')
    plt.show()

def test_stim_interneuron_impact():
    simulation_clock=Clock(dt=.5*ms)
    trial_duration=1*second
    dcs_start_time=.5*second

    stim_levels=[-4,-3,-2,-1,-0.5,-.25,-.125,0,.125,.25,.5,1,2,3,4]
    voltages = np.zeros(len(stim_levels))
    for idx,stim_level in enumerate(stim_levels):
        print('testing stim_level %.3fpA' % stim_level)
        eqs = exp_IF(default_params.C, default_params.gL, default_params.EL, default_params.VT, default_params.DeltaT)

        # AMPA conductance - recurrent input current
        eqs += exp_synapse('g_ampa_r', default_params.tau_ampa, siemens)
        eqs += Current('I_ampa_r=g_ampa_r*(E-vm): amp', E=default_params.E_ampa)

        # AMPA conductance - background input current
        eqs += exp_synapse('g_ampa_b', default_params.tau_ampa, siemens)
        eqs += Current('I_ampa_b=g_ampa_b*(E-vm): amp', E=default_params.E_ampa)

        # AMPA conductance - task input current
        eqs += exp_synapse('g_ampa_x', default_params.tau_ampa, siemens)
        eqs += Current('I_ampa_x=g_ampa_x*(E-vm): amp', E=default_params.E_ampa)

        # Voltage-dependent NMDA conductance
        eqs += biexp_synapse('g_nmda', default_params.tau1_nmda, default_params.tau2_nmda, siemens)
        eqs += Equations('g_V = 1/(1+(Mg/3.57)*exp(-0.062 *vm/mV)) : 1 ', Mg=default_params.Mg)
        eqs += Current('I_nmda=g_V*g_nmda*(E-vm): amp', E=default_params.E_nmda)

        # GABA-A conductance
        eqs += exp_synapse('g_gaba_a', default_params.tau_gaba_a, siemens)
        eqs += Current('I_gaba_a=g_gaba_a*(E-vm): amp', E=default_params.E_gaba_a)

        eqs +=InjectedCurrent('I_dcs: amp')

        group=NeuronGroup(1, model=eqs, threshold=-20*mV, refractory=inh_params.refractory, reset=default_params.Vr,
            compile=True, freeze=True, clock=simulation_clock)
        group.C=inh_params.C
        group.gL=inh_params.gL

        @network_operation(clock=simulation_clock)
        def inject_current(c):
            if simulation_clock.t>dcs_start_time:
                group.I_dcs=stim_level*pA
        monitor=StateMonitor(group, 'vm', simulation_clock, record=True)
        net=Network(group, monitor, inject_current)
        net.run(trial_duration, report='text')
        voltages[idx]=monitor.values[0,-1]*1000

    voltages=voltages-voltages[7]
    plt.figure()
    plt.plot(stim_levels,voltages)
    plt.xlabel('Stimulation level (pA)')
    plt.ylabel('Voltage Change (mV)')
    plt.show()

def plot_coherence_stim(mu_0):
    p_a=mu_0/100.0
    p_b=p_a
    plt.figure()
    plt.plot([0.0, 100.0], [mu_0 + p_a*0, mu_0+p_a*100.0], 'r', label='mu_a')
    plt.plot([0.0, 100.0], [mu_0 - p_b*0, mu_0-p_b*100.0], 'b', label='mu_b')
    plt.legend(loc=0)
    plt.xlabel('Coherence')
    plt.ylabel('Mean stimulus')


if __name__=='__main__':
    test_stim_pyramidal_impact()
    test_stim_interneuron_impact()