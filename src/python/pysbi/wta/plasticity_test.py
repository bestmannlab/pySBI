from brian import Clock, ms, PoissonGroup, Hz, second, network_operation, Network, ExponentialSTDP, nS, Connection, pA
from pysbi.wta.network import WTANetworkGroup
from pysbi.wta.monitor import WTAMonitor
import matplotlib.pyplot as plt
import numpy as np


def test():
    p_dcs = 0 * pA   #2
    i_dcs = 0 * pA  #-1
    tau_pre = 20 * ms
    tau_post = tau_pre
    dA_pre= .005
    dA_post = -dA_pre * 1.05
    gmax = 4 * nS
    ntrials = 40
    sim_clock = Clock(dt= 0.5 * ms)
    trial_duration = 4 * second
    stim_start_time = 1 * second
    stim_end_time = trial_duration - 1 * second
    background_input = PoissonGroup(2000, rates = 900 * Hz, clock = sim_clock)
    task_inputs = [PoissonGroup(240, rates= 1 * Hz, clock = sim_clock), PoissonGroup(240, rates= 1 * Hz, clock = sim_clock)]

    wta_net = WTANetworkGroup(2000, 2, background_input = background_input, task_inputs = task_inputs, clock = sim_clock)
    @network_operation(when = 'start', clock = sim_clock)
    def set_task_inputs():
        if stim_start_time <= sim_clock.t < stim_end_time:
            task_inputs[0]._S[0,:] = 2 * Hz
            task_inputs[1]._S[0,:] = 78 * Hz
        else:
            task_inputs[0]._S[0,:] = 1 * Hz
            task_inputs[1]._S[0,:] = 1 * Hz

    @network_operation(clock = sim_clock)
    def inject_current():
        wta_net.group_e.I_dcs= p_dcs
        wta_net.group_i.I_dcs= i_dcs

    stdp0_0 = ExponentialSTDP(wta_net.connections['t0->e0_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='additive')
    stdp1_1 = ExponentialSTDP(wta_net.connections['t1->e1_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='additive')
    stdp0_1= ExponentialSTDP(wta_net.connections['t0->e1_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='additive')
    stdp1_0 = ExponentialSTDP(wta_net.connections['t1->e0_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='additive')
    wta_monitor = WTAMonitor(wta_net, None, None, record_lfp = False, record_voxel = False, record_connections=['t0->e0_ampa','t1->e1_ampa','t0->e1_ampa','t1->e0_ampa'], clock = sim_clock)
    net = Network(background_input, task_inputs, set_task_inputs, wta_net, wta_net.connections.values(), wta_monitor.monitors.values(), stdp0_0, stdp1_1, stdp1_0, stdp0_1, inject_current)
    trial_weights = np.zeros((4,ntrials))

    for i in range(ntrials):
        net.reinit()
        net.run(trial_duration, report='text')
        #wta_monitor.plot()
        connection_matrix = wta_net.connections['t0->e0_ampa'].W.todense()
        connection_diagonal = np.diagonal(connection_matrix)
        avg_weight = np.mean(connection_diagonal)
        trial_weights[0,i] = avg_weight

        connection_matrix = wta_net.connections['t1->e1_ampa'].W.todense()
        connection_diagonal = np.diagonal(connection_matrix)
        avg_weight = np.mean(connection_diagonal)
        trial_weights[1,i] = avg_weight

        connection_matrix = wta_net.connections['t0->e1_ampa'].W.todense()
        connection_diagonal = np.diagonal(connection_matrix)
        avg_weight = np.mean(connection_diagonal)
        trial_weights[2,i] = avg_weight

        connection_matrix = wta_net.connections['t1->e0_ampa'].W.todense()
        connection_diagonal = np.diagonal(connection_matrix)
        avg_weight = np.mean(connection_diagonal)
        trial_weights[3,i] = avg_weight
        print('trial %d' % i)
    plt.plot(trial_weights[0,:]/nS, label = 't0->e0_ampa')
    plt.plot(trial_weights[1,:]/nS, label = 't1->e1_ampa')
    plt.plot(trial_weights[2,:]/nS,'--', label = 't0->e1_ampa')
    plt.plot(trial_weights[3,:]/nS,'--', label = 't1->e0_ampa')
    plt.legend(loc = 'best')
    plt.ylim(0, gmax/nS)
    plt.xlabel('trial')
    plt.ylabel('average weight')
    plt.show()

if __name__=='__main__':
    test()

