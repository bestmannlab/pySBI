from brian import nS, pA, second
import h5py
from pysbi.wta.network import pyr_params, simulation_params, default_params, plasticity_params
from pysbi.wta.monitor import SessionMonitor
import numpy as np
import matplotlib.pyplot as plt
from pysbi.wta.virtual_subject import VirtualSubject
import os
from collections import OrderedDict


def run_virtual_subjects(subj_ids, conditions, output_dir, behavioral_param_file):
    """
    Runs a set of virtual subjects on the given conditions
    subj_ids = list of subject IDs
    conditions = dictionary: {condition name: (simulation_params, reinit, coherence levels)}, reinit = whether or not
        to reinitialize state variables before running condition
    output_dir = directory to store h5 output files
    behavioral_param_file = h5 file containing softmax-RL parameter distributions, background freq is sampled using
        inverse temp param distribution
    """

    # Load alpha and beta params of control group from behavioral parameter file
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])

    # Run each subject
    for subj_id in subj_ids:
        print('***** Running subject %d *****' % subj_id)

        # Sample beta from subject distribution - don't use subjects with high alpha
        # beta_hist,beta_bins=np.histogram(beta_vals[np.where(alpha_vals<.99)[0]], density=True)
        # bin_width=beta_bins[1]-beta_bins[0]
        # beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        # beta=beta_bin+np.random.rand()*bin_width

        beta = 7.329377505


        # Create virtual subject parameters - background freq from beta dist, resp threshold between 20 and 30Hz
        wta_params=default_params(background_freq=(beta-161.08)/-.17, resp_threshold=25)  #15+np.random.uniform(10)) #20
        # Set initial input weights #2 and 0.85. 1.9 and 0.95
        plasticity_pyr_params=pyr_params(w_nmda=0.145*nS, w_ampa_ext_correct=2.35*nS, w_ampa_ext_incorrect=0.6*nS)
        plas_params=plasticity_params()

        # Create a virtual subject
        subject=VirtualSubject(subj_id, wta_params=wta_params, pyr_params=plasticity_pyr_params,
            plasticity_params=plas_params)

        # Run through each condition
        for condition, (sim_params, reinit, coherence_levels) in conditions.items():
            print condition

            # Reinitialize state variables in subject network
            if reinit:
                subject.net.reinit(states=True)

            # Run session
            run_session(subject, condition, sim_params, coherence_levels,
                output_file=os.path.join(output_dir, 'subject.%d.%s.h5' % (subj_id,condition)))


def run_session(subject, condition, sim_params, coherence_levels, output_file=None, plot=False):
    """
    Run session in subject
    subject = subject object
    sim_params = simulation params
    coherence_levels = coherence levels to test on
    output_file = if not none, writes h5 output to filename
    plot = plots session data if True
    """
    print('** Condition: %s **' % condition)

    # Record input connection weights
    record_connections = ['t0->e0_ampa', 't1->e1_ampa', 't0->e1_ampa', 't1->e0_ampa']
    # Create session monitor
    session_monitor = SessionMonitor(subject.wta_network, sim_params, plasticity_params,
        record_connections=record_connections, conv_window=40, record_firing_rates=True)
    # Trials per coherence level
    trials_per_level = 20
    # Create inputs for each trial
    trial_inputs = np.zeros((trials_per_level * len(coherence_levels), 2))
    # Create left and right directions for each coherence level
    for i in range(len(coherence_levels)):
        coherence = coherence_levels[i]
        # Left
        min_idx=i*trials_per_level
        max_idx=i*trials_per_level+trials_per_level/2
        trial_inputs[min_idx:max_idx, 0] = subject.wta_params.mu_0 + subject.wta_params.p_a * coherence * 100.0
        trial_inputs[min_idx:max_idx, 1] = subject.wta_params.mu_0 - subject.wta_params.p_b * coherence * 100.0

        #Right
        min_idx=i*trials_per_level+trials_per_level/2
        max_idx=i*trials_per_level + trials_per_level
        trial_inputs[min_idx:max_idx, 0] = subject.wta_params.mu_0 - subject.wta_params.p_b * coherence * 100.0
        trial_inputs[min_idx:max_idx, 1] = subject.wta_params.mu_0 + subject.wta_params.p_a * coherence * 100.0

    trial_inputs_difficult = np.repeat(trial_inputs[0:40,:],3,0)
    trial_inputs_easy = np.repeat(trial_inputs[80:120,:],3,0)

    trial_inputs_1 = np.repeat(trial_inputs[0:20,:],6,0)
    trial_inputs_2 = np.repeat(trial_inputs[20:40,:],6,0)
    trial_inputs_3 = np.repeat(trial_inputs[40:60,:],6,0)
    trial_inputs_4 = np.repeat(trial_inputs[60:80,:],6,0)
    trial_inputs_5 = np.repeat(trial_inputs[80:100,:],6,0)
    trial_inputs_6 = np.repeat(trial_inputs[100:120,:],6,0)


    # Shuffle trials
    trial_inputs = np.random.permutation(trial_inputs)

    #trained on easy
    trial_inputs_easy = np.random.permutation(trial_inputs_easy)

    #trained on difficult
    trial_inputs_difficult = np.random.permutation(trial_inputs_difficult)

    trial_inputs_1 = np.random.permutation(trial_inputs_1)
    trial_inputs_2 = np.random.permutation(trial_inputs_2)
    trial_inputs_3 = np.random.permutation(trial_inputs_3)
    trial_inputs_4 = np.random.permutation(trial_inputs_4)
    trial_inputs_5 = np.random.permutation(trial_inputs_5)
    trial_inputs_6 = np.random.permutation(trial_inputs_6)


    # Simulate each trial
    for t in range(sim_params.ntrials):
        print('Trial %d' % t)

        if condition== 'training':
            if training == 'control':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == 'easy':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_easy[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == 'diff':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_difficult[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == '1':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_1[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == '2':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_2[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == '3':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_3[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == '4':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_4[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == '5':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_5[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]
            elif training == '6':
                # Get task input for trial and figure out which is correct
                task_input_rates = trial_inputs_6[t, :]
                correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]

        else:
            # Get task input for trial and figure out which is correct
            task_input_rates = trial_inputs[t, :]
            correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]



        # Run trial
        subject.net.reinit(states=True)
        subject.run_trial(sim_params, task_input_rates)
        print task_input_rates

        # Record trial
        session_monitor.record_trial(t, task_input_rates, correct_input, subject.wta_network, subject.wta_monitor)


    # Write output
    if output_file is not None:
        session_monitor.write_output(output_file)

    # Plot
    if plot:
        session_monitor.plot()
        plt.show()


def run_nostim_training_subjects(subj_ids, stim_types, coherence_levels, trials_per_condition):
    """
    Run subjects with no stimulation during training - runs baseline without stim, training without stim, and test without stim
    nsubjects = number of subjects to simulation
    coherence_levels = number of coherence levels to use
    """
    conditions = OrderedDict({'baseline': (simulation_params(ntrials=trials_per_condition), True, coherence_levels)})

    # conditions = OrderedDict({'training': (simulation_params(ntrials=trials_per_condition, plasticity=True), True, coherence_levels)})
    conditions['training']= (simulation_params(ntrials=trials_per_condition, plasticity=True), True, coherence_levels)

    # conditions = OrderedDict({'training': (simulation_params(ntrials=trials_per_condition, plasticity=True), True, coherence_levels)})
    conditions['testing'] = (simulation_params(ntrials=trials_per_condition), True, coherence_levels)

    # run baseline with stim
    # for stim in stim_types:
    #      p_dcs, i_dcs = stim_types.get(stim)
    #      conditions['baseline_%s' % stim] = (simulation_params(ntrials=trials_per_condition, plasticity=False, p_dcs=p_dcs, i_dcs=i_dcs, dcs_start_time=0*second, dcs_end_time=4*second), True, coherence_levels)


    print conditions.keys()
    # Run subject
    run_virtual_subjects(subj_ids, conditions, '/home/jeff/projects/pySBI/data/stdp_maxstim_2_plasticity_14_40_fixed/final/6/control/final_test_final',
        '/home/jeff/projects/pySBI/data/rerw/fitted_behavioral_params.h5')


def run_depolarizing_learning_subjects(subj_ids, stim_types, coherence_levels, trials_per_condition):
    (p_dcs, i_dcs) = (stim_intensity_max, -0.5*stim_intensity_max)
    conditions= OrderedDict({'training': (simulation_params(ntrials=trials_per_condition, plasticity=True, p_dcs=p_dcs, i_dcs=i_dcs, dcs_start_time=0*second, dcs_end_time=4*second), True, coherence_levels)})
    conditions['testing'] = (simulation_params(ntrials=trials_per_condition), True, coherence_levels)
    print conditions.keys()
    print p_dcs, i_dcs
    # Run subjects
    run_virtual_subjects(subj_ids, conditions,
        '/home/jeff/projects/pySBI/data/stdp_maxstim_2_plasticity_14_40_fixed/final/1/depolarizing/final_test_final',
        '/home/jeff/projects/pySBI/data/rerw/fitted_behavioral_params.h5')


def run_hyperpolarizing_learning_subjects(subj_ids, stim_types, coherence_levels, trials_per_condition):
    (p_dcs, i_dcs) = (-stim_intensity_max, 0.5*stim_intensity_max)
    conditions= OrderedDict({'training': (simulation_params(ntrials=trials_per_condition, plasticity=True, p_dcs=p_dcs, i_dcs=i_dcs, dcs_start_time=0*second, dcs_end_time=4*second), True, coherence_levels)})
    conditions['testing'] = (simulation_params(ntrials=trials_per_condition), True, coherence_levels)
    print conditions.keys()
    print p_dcs, i_dcs
    # Run subjects
    run_virtual_subjects(subj_ids, conditions,
        '/home/jeff/projects/pySBI/data/stdp_maxstim_2_plasticity_14_40_fixed/final/1/hyperpolarizing/final_test_final',
        '/home/jeff/projects/pySBI/data/rerw/fitted_behavioral_params.h5')


if __name__=='__main__':
    # Max stimulation intensity
    stim_intensity_max=2*pA
    # Stimulation intensities
    stim_types={
         'depolarizing': (stim_intensity_max, -0.5*stim_intensity_max),
         'hyperpolarizing': (-1*stim_intensity_max, 0.5*stim_intensity_max)
    }
    training = 'control' # control, easy, difficult


    # Run on six coherence levels
    coherence_levels=[.016, .032, .064, .128, .256, .512]
    # Trials per condition
    trials_per_condition=120
    total_conditions = 1
    #
    for subs in range(total_conditions):
        if subs == 0:
            print 'control_learning'
            run_nostim_training_subjects(range(0,1), stim_types, coherence_levels, trials_per_condition)
        elif subs == 1:
            print 'depolarizing_learning'
            run_depolarizing_learning_subjects(range(0,1), stim_types, coherence_levels, trials_per_condition)
        elif subs == 2:
            print 'hyperpolarizing_learning'
            run_hyperpolarizing_learning_subjects(range(0,1), stim_types, coherence_levels, trials_per_condition)

