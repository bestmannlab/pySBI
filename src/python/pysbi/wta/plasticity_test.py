from brian import Clock, ms, PoissonGroup, Hz, second, network_operation, Network, ExponentialSTDP, nS, pA
from pysbi.wta.network import WTANetworkGroup, pyr_params
from pysbi.wta.monitor import WTAMonitor
import matplotlib.pyplot as plt
import numpy as np

from pysbi.util.utils import get_response_time

def test_plasticity(ntrials, plasticity=False, p_dcs=0*pA, i_dcs=0*pA, init_weight=1.1*nS):
    # Plasticity parameters
    tau_pre = 20 * ms
    tau_post = tau_pre
    dA_pre= 0.001 #.001
    dA_post = -dA_pre * 1.05
    # Maximum synaptic weight
    gmax = 4 * nS

    # Simulation parameters
    # Time step
    dt=0.5*ms
    # Simulation clock
    sim_clock = Clock(dt=dt)
    # Length of each trial
    trial_duration = 4 * second
    # Time inputs start
    input_start_time = 1 * second
    # Time inputs end
    input_end_time = trial_duration - 1 * second
    # Accuracy convolution window size
    conv_window=10

    # Input parameters
    # Strength of each input at contrast=0
    mu_0=40.0
    # Rate of input increase with contrast increase
    p_a=mu_0/100.0
    # Rate of input decrease with contrast increase
    p_b=p_a

    # Network definition
    # Number of response options
    num_options=2
    # Number of neurons in the network (total)
    network_size=2000
    # Number of pyramidal cells
    pyr_size=int(network_size*.8)

    # Background inputs
    background_size=network_size
    background_rate=950*Hz
    background_input = PoissonGroup(background_size, rates = background_rate, clock = sim_clock)

    # Task-related inputs
    task_input_size=int(.15*pyr_size)
    task_input_resting_rate=1*Hz
    task_inputs = []
    for i in range(num_options):
        task_inputs.append(PoissonGroup(task_input_size, rates= task_input_resting_rate, clock = sim_clock))

    # WTA network
    # Adjust the strength of inputs
    plasticity_pyr_params=pyr_params
    #plasticity_pyr_params.w_ampa_ext=4.1*nS # If weights get here, accuracy is worse again
    #plasticity_pyr_params.w_ampa_ext=2.1*nS # If weights get here, accuracy is better
    plasticity_pyr_params.w_ampa_ext=init_weight # Initial weight with bad accuracy
    wta_net = WTANetworkGroup(network_size, num_options, pyr_params=plasticity_pyr_params,
        background_input = background_input, task_inputs = task_inputs, clock = sim_clock)

    # Task-related input update function
    @network_operation(when = 'start', clock = sim_clock)
    def set_task_inputs():
        for i in range(num_options):
            task_inputs[i]._S[0,:] = task_input_resting_rate
            if input_start_time <= sim_clock.t < input_end_time:
                task_inputs[i]._S[0,:] = task_input_rates[i] * Hz

    # Inject dcs current
    @network_operation(clock = sim_clock)
    def inject_current():
        wta_net.group_e.I_dcs= p_dcs
        wta_net.group_i.I_dcs= i_dcs

    # Create network
    if plasticity:
        # Input projections plasticity
        stdp0_0 = ExponentialSTDP(wta_net.connections['t0->e0_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax,
            update='additive')
        stdp1_1 = ExponentialSTDP(wta_net.connections['t1->e1_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax,
            update='additive')
        stdp0_1= ExponentialSTDP(wta_net.connections['t0->e1_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax,
            update='additive')
        stdp1_0 = ExponentialSTDP(wta_net.connections['t1->e0_ampa'], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax,
            update='additive')

        # Network monitor
        wta_monitor = WTAMonitor(wta_net, None, None, record_lfp = False, record_voxel = False,
            record_connections=['t0->e0_ampa','t1->e1_ampa','t0->e1_ampa','t1->e0_ampa'], clock = sim_clock)

        net = Network(background_input, task_inputs, set_task_inputs, wta_net, wta_net.connections.values(),
            wta_monitor.monitors.values(), stdp0_0, stdp1_1, stdp1_0, stdp0_1)
    else:
        # Network monitor
        wta_monitor = WTAMonitor(wta_net, None, None, record_lfp = False, record_voxel = False, clock = sim_clock)

        net = Network(background_input, task_inputs, set_task_inputs, wta_net, wta_net.connections.values(),
            wta_monitor.monitors.values())

    # Construct list of trials
    trials=[]
    contrast_range=[.016, .032, .064, .128, .256, .512]
    # Trials per contrast is at least one, or ntrials / number of contrasts
    trials_per_contrast=np.max([1,np.round(ntrials/len(contrast_range))])
    # For each contrast value
    for i,contrast in enumerate(contrast_range):
        # Add trials with this contrast value
        for j in range(trials_per_contrast):
            # Compute inputs
            inputs=[mu_0+p_a*contrast*100.0, mu_0-p_b*contrast*100.0]
            # Make every other trial have input 1 > input0
            if j%2==0:
                trials.append(np.array(inputs))
            else:
                trials.append(np.array([inputs[1],inputs[0]]))
    # Shuffle trials
    np.random.shuffle(trials)

    # Store weights and behavior
    trial_weights = np.zeros((4,ntrials))
    response_time = np.zeros((1, ntrials))
    correct_choice = np.zeros((1, ntrials))
    choice_trial = np.zeros((1, ntrials))
    correct_avg = np.zeros((1, ntrials))

    num_no_response=0

    for i in range(ntrials):

        # Re-init network
        net.reinit()

        # Get task-related inputs and compute correct response
        task_input_rates=trials[i]
        correct_input=np.where(task_input_rates==np.max(task_input_rates))[0]

        print('trial %d' % i)
        print 'task input 0 = %.2f  task input 1 = %.2f' % (task_input_rates[0], task_input_rates[1])

        # Run network and get response
        net.run(trial_duration, report='text')
        rate_0 = wta_monitor.monitors['excitatory_rate_0'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        rate_1 = wta_monitor.monitors['excitatory_rate_1'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        rt, choice = get_response_time(np.array([rate_0, rate_1]), input_start_time, input_end_time,
            upper_threshold = 25, dt = dt)
        correct = choice == correct_input
        if choice>-1:
            print 'response time = %.3f correct = %d' % (rt, int(correct))
        else:
            print 'no response!'
            num_no_response+=1
        if ntrials==1:
            wta_monitor.plot()

        # Store behavior and weights
        correct_choice[0,i] = correct
        response_time[0,i] = rt
        choice_trial[0,i] = choice
        correct_avg[0,i] = (np.sum(correct_choice))/(i+1)
        trial_weights[0,i] = np.mean(np.diagonal(wta_net.connections['t0->e0_ampa'].W.todense()))
        trial_weights[1,i] = np.mean(np.diagonal(wta_net.connections['t1->e1_ampa'].W.todense()))
        trial_weights[2,i] = np.mean(np.diagonal(wta_net.connections['t0->e1_ampa'].W.todense()))
        trial_weights[3,i] = np.mean(np.diagonal(wta_net.connections['t1->e0_ampa'].W.todense()))

    if ntrials>1:
        # Convolve accuracy
        correct_ma = (np.convolve(correct_choice[0,:], np.ones((conv_window,))/conv_window, mode='valid'))

        resp_trials=np.where(choice_trial[0,:]>-1)[0]
        perc_correct=float(np.sum(correct_choice[0,resp_trials]))/float(len(resp_trials))
        perc_correct_overall=float(np.sum(correct_choice[0,:]))/float(ntrials)
        print('perc correct (overall)=%.2f' % perc_correct_overall)
        print('perc correct (responded)=%.2f' % perc_correct)
        print('no response=%.2f' % (float(num_no_response)/float(ntrials)))
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


if __name__=='__main__':
    test_plasticity(120, plasticity=False, p_dcs=0*pA, i_dcs=0*pA, init_weight=0.55*nS)
