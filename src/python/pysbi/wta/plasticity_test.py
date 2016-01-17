from brian import Clock, ms, PoissonGroup, Hz, second, network_operation, Network, ExponentialSTDP, nS, Connection, pA
from pysbi.wta.network import WTANetworkGroup
from pysbi.wta.monitor import WTAMonitor
import matplotlib.pyplot as plt
import numpy as np
import collections
import random

from pysbi.util.utils import get_response_time


def test():
    p_dcs = 0 * pA   #2
    i_dcs = 0 * pA  #-1
    tau_pre = 20 * ms
    tau_post = tau_pre
    dA_pre= 0.001 #.001
    dA_post = -dA_pre * 1.05
    gmax = 4 * nS
    ntrials = 120
    sim_clock = Clock(dt= 0.5 * ms)
    trial_duration = 4 * second
    stim_start_time = 1 * second
    stim_end_time = trial_duration - 1 * second
    background_input = PoissonGroup(2000, rates = 900 * Hz, clock = sim_clock)
    task_inputs = [PoissonGroup(240, rates= 1 * Hz, clock = sim_clock), PoissonGroup(240, rates= 1 * Hz, clock = sim_clock)]
    window = 10
    wta_net = WTANetworkGroup(2000, 2, background_input = background_input, task_inputs = task_inputs, clock = sim_clock)

    @network_operation(when = 'start', clock = sim_clock)
    def set_task_inputs():
        if stim_start_time <= sim_clock.t < stim_end_time:
            task_inputs[0]._S[0,:] = t0_rate_final * Hz
            task_inputs[1]._S[0,:] = t1_rate_final * Hz

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
    #net = Network(background_input, task_inputs, set_task_inputs, wta_net, wta_net.connections.values(), wta_monitor.monitors.values(), stdp0_0, stdp1_1, stdp1_0, stdp0_1, inject_current)
    net = Network(background_input, task_inputs, set_task_inputs, wta_net, wta_net.connections.values(), wta_monitor.monitors.values())

    trial_weights = np.zeros((4,ntrials))
    response_time = np.zeros((1, ntrials))
    correct_choice = np.zeros((1, ntrials))
    choice_trial = np.zeros((1, ntrials))
    correct_avg = np.zeros((1, ntrials))

    # generate random trials
    t_rate = [19.52, 29.76, 34.88, 37.44, 38.72, 39.36, 40.64, 41.28, 42.56, 45.12, 50.24, 60.48]
    t_rate_low = [19.52, 29.76, 34.88, 37.44, 38.72, 39.36]
    t_rate_high = [40.64, 41.28, 42.56, 45.12, 50.24, 60.48]
    if ntrials>=12:
        t_rate_sample1 = ntrials/12 * t_rate
        t_rate_sample2 = np.random.choice(t_rate_low, size=((ntrials%12)/2), replace=False)
        t_rate_sample3 = np.random.choice(t_rate_high, size=((ntrials%12)/2), replace=False)
        if ((ntrials%12)%2)==0:
            t_rate_sample4 = []
        else:
            t_rate_sample4 = [np.random.choice(t_rate)]
        t_rate_1 = np.array(t_rate_sample1).tolist() + np.array(t_rate_sample2).tolist() + np.array(t_rate_sample3).tolist() + np.array(t_rate_sample4).tolist()
    else:
        t_rate_1 = np.random.choice(t_rate, size=ntrials, replace=False)
    t0_rate = np.random.choice(t_rate_1, size= ntrials, replace=False)
    a80 = np.empty(len(t0_rate)); a80.fill(80)
    t1_rate = a80 - t0_rate
    t0_dom_occur = sum(i > 40.0 for i in t0_rate)
    t1_dom_occur = sum(i > 40.0 for i in t1_rate)
    print 'Task input 0 is dominant = %d Task input 1 is dominant = %d' % (t0_dom_occur, t1_dom_occur)


    for i in range(ntrials):
        net.reinit()
        # if i%2 == 0:
        #     t0_rate1 = 20 * Hz
        #     t1_rate1 = 60 * Hz
        #     correct_input = 1
        # else:
        #     t0_rate0 = 60 * Hz
        #     t1_rate0 = 20 * Hz
        #     correct_input = 0
        #
        # if i%2 == 0:
        #     t0_rate = 20 * Hz
        #     t1_rate = 60 * Hz
        #     correct_input = 1
        # else:
        #     t0_rate = 60 * Hz
        #     t1_rate = 20 * Hz
        #     correct_input = 0
        t0_rate_final = t0_rate[i]
        t1_rate_final = t1_rate[i]
        print 'task input 0 = %.2f  task input 1 = %.2f' % (t0_rate_final, t1_rate_final)
        if t0_rate_final > t1_rate_final:
            correct_input = 0
        elif t0_rate_final == t1_rate_final:
            correct_input = 1 or 0
        else:
            correct_input = 1

        net.run(trial_duration, report='text')
        rate_0 = wta_monitor.monitors['excitatory_rate_0'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        rate_1 = wta_monitor.monitors['excitatory_rate_1'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        rt, choice = get_response_time(np.array([rate_0, rate_1]), stim_start_time, stim_end_time, upper_threshold = 25, dt = 0.5 * ms)
        correct = choice == correct_input
        print 'response time = %.3f correct = %d' % (rt, int(correct))
        #wta_monitor.plot()

        correct_choice[0,i] = correct
        response_time[0,i] = rt
        choice_trial[0,i] = choice
        correct_avg[0,i] = (np.sum(correct_choice))/(i+1)

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

    correct_ma = (np.convolve(correct_choice[0,:], np.ones((window,))/window, mode='valid'))

    print correct_avg
    print correct_ma
    plt.plot(trial_weights[0,:]/nS, label = 't0->e0_ampa')
    plt.plot(trial_weights[1,:]/nS, label = 't1->e1_ampa')
    plt.plot(trial_weights[2,:]/nS,'--', label = 't0->e1_ampa')
    plt.plot(trial_weights[3,:]/nS,'--', label = 't1->e0_ampa')
    plt.legend(loc = 'best')
    plt.ylim(0, gmax/nS)
    plt.xlabel('trial')
    plt.ylabel('average weight')
    plt.show()

    plt.plot(correct_choice[0,:])
    plt.xlabel('trial')
    plt.ylabel('correct choice = 1')
    plt.show()

    plt.plot(correct_avg[0,:], label = 'average')
    plt.plot(correct_ma, label = 'moving avg')
    plt.legend(loc = 'best')
    plt.ylim(0,1)
    plt.xlabel('trial')
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(response_time[0,:])
    plt.ylim(0, 2000)
    plt.xlabel('trial')
    plt.ylabel('response time')
    plt.show()

    plt.plot(choice_trial[0,:])
    plt.xlabel('trial')
    plt.ylabel('choice e0=0 e1=1')
    plt.show()

    # net = Network(background_input, task_inputs, set_task_inputs, wta_net, wta_net.connections.values(), wta_monitor.monitors.values())
    #
    # trial_weights2 = np.zeros((4,ntrials))
    # response_time2 = np.zeros((1, ntrials))
    # correct_choice2 = np.zeros((1, ntrials))
    # choice_trial2 = np.zeros((1, ntrials))
    # correct_avg2 = np.zeros((1, ntrials))
    #
    # for i in range(ntrials):
    #     net.reinit()
    #     if i%2 == 0:
    #         t0_rate = 37 * Hz
    #         t1_rate = 43 * Hz
    #         correct_input = 1
    #     else:
    #         t0_rate = 43 * Hz
    #         t1_rate = 37 * Hz
    #         correct_input = 0
    #
    #     net.run(trial_duration, report='text')
    #     rate_0 = wta_monitor.monitors['excitatory_rate_0'].smooth_rate(width= 5 * ms, filter = 'gaussian')
    #     rate_1 = wta_monitor.monitors['excitatory_rate_1'].smooth_rate(width= 5 * ms, filter = 'gaussian')
    #     rt, choice = get_response_time(np.array([rate_0, rate_1]), stim_start_time, stim_end_time, upper_threshold = 25, dt = 0.5 * ms)
    #     correct = choice == correct_input
    #     print 'response time = %.3f correct = %d' % (rt, int(correct))
    #     # wta_monitor.plot()
    #
    #     correct_choice2[0,i] = correct
    #     response_time2[0,i] = rt
    #     choice_trial2[0,i] = choice
    #     correct_avg2[0,i] = (np.sum(correct_choice2))/(i+1)
    #
    #     connection_matrix = wta_net.connections['t0->e0_ampa'].W.todense()
    #     connection_diagonal = np.diagonal(connection_matrix)
    #     avg_weight = np.mean(connection_diagonal)
    #     trial_weights2[0,i] = avg_weight
    #
    #     connection_matrix = wta_net.connections['t1->e1_ampa'].W.todense()
    #     connection_diagonal = np.diagonal(connection_matrix)
    #     avg_weight = np.mean(connection_diagonal)
    #     trial_weights2[1,i] = avg_weight
    #
    #     connection_matrix = wta_net.connections['t0->e1_ampa'].W.todense()
    #     connection_diagonal = np.diagonal(connection_matrix)
    #     avg_weight = np.mean(connection_diagonal)
    #     trial_weights2[2,i] = avg_weight
    #
    #     connection_matrix = wta_net.connections['t1->e0_ampa'].W.todense()
    #     connection_diagonal = np.diagonal(connection_matrix)
    #     avg_weight = np.mean(connection_diagonal)
    #     trial_weights2[3,i] = avg_weight
    #     print('trial %d' % i)
    #
    # correct_ma2 = (np.convolve(correct_choice2[0,:], np.ones((window,))/window, mode='same'))
    #
    # print correct_avg2
    # print correct_ma2
    # plt.plot(trial_weights2[0,:]/nS, label = 't0->e0_ampa')
    # plt.plot(trial_weights2[1,:]/nS, label = 't1->e1_ampa')
    # plt.plot(trial_weights2[2,:]/nS,'--', label = 't0->e1_ampa')
    # plt.plot(trial_weights2[3,:]/nS,'--', label = 't1->e0_ampa')
    # plt.legend(loc = 'best')
    # plt.ylim(0, gmax/nS)
    # plt.xlabel('trial')
    # plt.ylabel('average weight')
    # plt.show()
    #
    # plt.plot(correct_choice2[0,:])
    # plt.xlabel('trial')
    # plt.ylabel('correct choice = 1')
    # plt.show()
    #
    # plt.plot(correct_avg2[0,:], label = 'average')
    # plt.plot(correct_ma2, label = 'moving avg')
    # plt.legend(loc = 'best')
    # plt.ylim(0,1)
    # plt.xlabel('trial')
    # plt.ylabel('accuracy')
    # plt.show()
    #
    # plt.plot(response_time2[0,:])
    # plt.ylim(0, 4000)
    # plt.xlabel('trial')
    # plt.ylabel('response time')
    # plt.show()
    #
    # plt.plot(choice_trial2[0,:])
    # plt.xlabel('trial')
    # plt.ylabel('choice e0=0 e1=1')
    # plt.show()

if __name__=='__main__':
    test()
