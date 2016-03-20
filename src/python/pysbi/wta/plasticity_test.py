from brian import Clock, ms, PoissonGroup, Hz, network_operation, Network, ExponentialSTDP, nS, pA, DelayConnection, Parameters
from pysbi.wta.network import WTANetworkGroup, pyr_params, simulation_params, default_params
from pysbi.wta.monitor import WTAMonitor, SessionMonitor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, ylim, legend, ylabel, xlabel, show, title

# Plasticity parameters
plasticity_params=Parameters(
    tau_pre = 20 * ms,
    dA_pre= 0.0005,  #0.0005
    # Maximum synaptic weight
    gmax = 4 * nS   #5 *nS
)
plasticity_params.tau_post=plasticity_params.tau_pre
plasticity_params.dA_post=-plasticity_params.dA_pre*1.1  #1.1

contrast_lowdiff = [.256, .512]
contrast_highdiff = [.016, .032]


def test_plasticity(ntrials, conv_window= 10, plasticity=False, p_dcs=0*pA, i_dcs=0*pA, init_weight=1.5*nS, init_incorrect_weight=.81*nS,
                    contrast_range_firsthalf = [.016, .032, .064, .128, .256, .512],
                    contrast_range_secondhalf = [.016, .032, .064, .128, .256, .512] ):

    # Simulation parameters
    sim_params=simulation_params()
    sim_params.ntrials=ntrials
    sim_params.p_dcs=p_dcs
    sim_params.i_dcs=i_dcs
    sim_params.plasticity = plasticity

    # Simulation clock
    sim_clock = Clock(dt=sim_params.dt)

    # Network definition
    wta_params=default_params()
    wta_params.background_freq= 950*Hz

    # Background inputs
    background_input = PoissonGroup(wta_params.background_input_size, rates = wta_params.background_freq, clock = sim_clock)

    # Task-related inputs
    task_inputs = []
    for i in range(wta_params.num_groups):
        task_inputs.append(PoissonGroup(wta_params.task_input_size, rates= wta_params.task_input_resting_rate, clock = sim_clock))

    # WTA network
    # Adjust the strength of inputs
    plasticity_pyr_params=pyr_params()
    #plasticity_pyr_params.w_ampa_ext=4.1*nS # If weights get here, accuracy is worse again
    #plasticity_pyr_params.w_ampa_ext=2.1*nS # If weights get here, accuracy is better
    plasticity_pyr_params.w_ampa_ext=init_weight # Initial weight with bad accuracy

    wta_net = WTANetworkGroup(params=wta_params, pyr_params=plasticity_pyr_params,
        background_input = background_input, task_inputs = task_inputs, clock = sim_clock)
    # Task input -> E population connections
    for i in range(wta_params.num_groups):
        wta_net.connections['t%d->e%d_ampa' % (i,1-i)]=DelayConnection(task_inputs[i], wta_net.groups_e[1-i],
            'g_ampa_x')
        wta_net.connections['t%d->e%d_ampa' % (i,1-i)].connect_one_to_one(weight= init_incorrect_weight,
            delay=.5*ms)

    # Task-related input update function
    @network_operation(when = 'start', clock = sim_clock)
    def set_task_inputs():
        for i in range(wta_params.num_groups):
            task_inputs[i]._S[0,:] = wta_params.task_input_resting_rate
            if sim_params.stim_start_time <= sim_clock.t < sim_params.stim_end_time:
                task_inputs[i]._S[0,:] = task_input_rates[i] * Hz

    # Inject dcs current
    @network_operation(clock = sim_clock)
    def inject_current():
        wta_net.group_e.I_dcs= p_dcs
        wta_net.group_i.I_dcs= i_dcs


    record_connections=[]
    # Create network
    if plasticity:
        record_connections=['t0->e0_ampa','t1->e1_ampa','t0->e1_ampa','t1->e0_ampa']

        # Input projections plasticity
        stdp0_0 = ExponentialSTDP(wta_net.connections['t0->e0_ampa'], plasticity_params.tau_pre,
            plasticity_params.tau_post, plasticity_params.dA_pre, plasticity_params.dA_post,
            wmax=plasticity_params.gmax, update='additive', clock=sim_clock)
        stdp1_1 = ExponentialSTDP(wta_net.connections['t1->e1_ampa'], plasticity_params.tau_pre,
            plasticity_params.tau_post, plasticity_params.dA_pre, plasticity_params.dA_post,
            wmax=plasticity_params.gmax, update='additive', clock=sim_clock)
        stdp0_1= ExponentialSTDP(wta_net.connections['t0->e1_ampa'], plasticity_params.tau_pre,
            plasticity_params.tau_post, plasticity_params.dA_pre, plasticity_params.dA_post,
            wmax=plasticity_params.gmax, update='additive', clock=sim_clock)
        stdp1_0 = ExponentialSTDP(wta_net.connections['t1->e0_ampa'], plasticity_params.tau_pre,
            plasticity_params.tau_post, plasticity_params.dA_pre, plasticity_params.dA_post,
            wmax=plasticity_params.gmax, update='additive', clock=sim_clock)

        # Network monitor
        wta_monitor = WTAMonitor(wta_net, None, None, sim_params, record_lfp = False, record_voxel = False,
            record_connections=record_connections, clock = sim_clock)

        net = Network(background_input, task_inputs, set_task_inputs, inject_current, wta_net, wta_net.connections.values(),
            wta_monitor.monitors.values(), stdp0_0, stdp1_1, stdp1_0, stdp0_1)
    else:
        # Network monitor
        wta_monitor = WTAMonitor(wta_net, None, None, sim_params, record_lfp = False, record_voxel = False,
            clock = sim_clock)

        net = Network(background_input, task_inputs, set_task_inputs, inject_current, wta_net, wta_net.connections.values(),
            wta_monitor.monitors.values())




    session_monitor=SessionMonitor(wta_net, sim_params, plasticity_params, record_connections=record_connections,
        conv_window=conv_window, record_firing_rates=False)

    # Construct list of trials
    trials_1 = []
    trials_2 = []
    trials_per_contrast_firsthalf =np.max([1,np.round(sim_params.ntrials/2/len(contrast_range_firsthalf))])
    #print trials_per_contrast_firsthalf
    trials_per_contrast_secondhalf = np.max([1,np.round(sim_params.ntrials/2/len(contrast_range_secondhalf))])
    #print trials_per_contrast_secondhalf

    for i,contrast in enumerate(contrast_range_firsthalf):
            # Add trials with this contrast value
       for j in range(trials_per_contrast_firsthalf):
            # Compute inputs
          inputs=[wta_params.mu_0+wta_params.p_a*contrast*100.0, wta_params.mu_0-wta_params.p_b*contrast*100.0]
            # Make every other trial have input 1 > input0
          if j%2==0:
              trials_1.append(np.array(inputs))
          else:
              trials_1.append(np.array([inputs[1],inputs[0]]))

    for i,contrast in enumerate(contrast_range_secondhalf):
        # Add trials with this contrast value
        for j in range(trials_per_contrast_secondhalf):
            # Compute inputs
            inputs=[wta_params.mu_0+wta_params.p_a*contrast*100.0, wta_params.mu_0-wta_params.p_b*contrast*100.0]
            # Make every other trial have input 1 > input0
            if j%2==0:
                trials_2.append(np.array(inputs))
            else:
                trials_2.append(np.array([inputs[1],inputs[0]]))
    #Shuffle trial
    np.random.shuffle(trials_1)
    #print trials_1
    #print len(trials_1)
    np.random.shuffle(trials_2)
    #print trials_2
    #print len(trials_2)

    for i in range(sim_params.ntrials):
        # Re-init network
        net.clock=sim_clock
        net.reinit()
        # Get task-related inputs and compute correct response
        if i<(ntrials/2):
            task_input_rates=trials_1[i]
        else:
            task_input_rates=trials_2[i-ntrials/2]
        correct_input=np.where(task_input_rates==np.max(task_input_rates))[0]
        if i >= ntrials/2:
            p_dcs = 0*pA
            i_dcs = 0*pA





        print('trial %d' % i)
        print 'task input 0 = %.2f  task input 1 = %.2f' % (task_input_rates[0], task_input_rates[1])
        print p_dcs
        print i_dcs



        # Run network and get response
        net.run(sim_params.trial_duration, report='text')
        session_monitor.record_trial(i, task_input_rates, correct_input, wta_net, wta_monitor)
        #if sim_params.ntrials==1:
         #   wta_monitor.plot()





    #if sim_params.ntrials>1:
     #   session_monitor.plot()

    correct_ma=session_monitor.get_correct_ma()
    if plasticity:
        trial_diag_weights = session_monitor.get_trial_diag_weights()
    else:
        trial_diag_weights = np.zeros((4, ntrials))
    perc_correct = session_monitor.get_perc_correct()
    perc_correct_training = session_monitor.get_perc_correct_training()
    perc_correct_test = session_monitor.get_perc_correct_test()
    return correct_ma, trial_diag_weights, perc_correct, perc_correct_training, perc_correct_test



if __name__=='__main__':
    nsessions = 10
    ntrials= 240
    conv_window=10
    all_trial_diag_weights=np.zeros((nsessions,4,ntrials))
    all_correct_ma = np.zeros((nsessions,ntrials-conv_window+1))
    all_perc_correct = np.zeros((nsessions,1))
    all_perc_correct_training = np.zeros((nsessions,1))
    all_perc_correct_test = np.zeros((nsessions,1))


    for session in range(nsessions):
        correct_ma, trial_diag_weights, perc_correct, perc_correct_training, perc_correct_test =test_plasticity(ntrials, conv_window= conv_window, plasticity=True, p_dcs=-4*pA, i_dcs=2*pA, init_weight=1.1*nS, init_incorrect_weight=0.6*nS)
        all_trial_diag_weights[session,:,:] = trial_diag_weights
        all_correct_ma[session,:] = correct_ma
        all_perc_correct[session,:] = perc_correct
        all_perc_correct_training[session,:] = perc_correct_training
        all_perc_correct_test[session,:] = perc_correct_test
    avg_all_trial_diag_weights = all_trial_diag_weights.mean(axis=0)
    avg_all_correct_ma = all_correct_ma.mean(axis=0)
    avg_all_perc_correct = all_perc_correct.mean(axis=0)
    avg_all_perc_correct_training = all_perc_correct_training.mean(axis=0)
    avg_all_perc_correct_test = all_perc_correct_test.mean(axis=0)

    print('perc correct (responded)=%.2f' % avg_all_perc_correct)
    print ('perc correct training phase (responded)=%.2f' % avg_all_perc_correct_training)
    print ('perc correct test phase (responded)=%.2f' % avg_all_perc_correct_test)
    print all_perc_correct
    print all_perc_correct_training
    print all_perc_correct_test
    print all_trial_diag_weights[:,:,119]
    print all_trial_diag_weights[:,:,239]


    plt.figure()
    plt.title("Total Moving Average")
    plt.plot(avg_all_correct_ma, label = 'moving avg')
    plt.legend(loc = 'best')
    plt.ylim(0,1)
    plt.xlabel('trial')
    plt.ylabel('accuracy')

    plt.figure()
    plt.title('Total')
    record_connections = ['t0->e0_ampa', 't1->e1_ampa', 't0->e1_ampa', 't1->e0_ampa']
    for i in range(len(record_connections)):
        plt.plot(avg_all_trial_diag_weights[i,:]/nS, label = record_connections[i])
        plt.legend(loc = 'best')
        plt.ylim(0,1.8)
        plt.xlabel('trial')
        plt.ylabel('average weight')

    plt.show()
