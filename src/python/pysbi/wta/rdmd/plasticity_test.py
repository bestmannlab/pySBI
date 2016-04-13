from brian import nS, pA, second
import h5py
from pysbi.wta.network import pyr_params, simulation_params, default_params, plasticity_params
from pysbi.wta.monitor import SessionMonitor
import numpy as np
import matplotlib.pyplot as plt
from pysbi.wta.virtual_subject import VirtualSubject
import os

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
        beta_hist,beta_bins=np.histogram(beta_vals[np.where(alpha_vals<.99)[0]], density=True)
        bin_width=beta_bins[1]-beta_bins[0]
        beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        beta=beta_bin+np.random.rand()*bin_width

        # Create virtual subject parameters - background freq from beta dist, resp threshold between 20 and 30Hz
        wta_params=default_params(background_freq=(beta-161.08)/-.17, resp_threshold=15+np.random.uniform(10))
        # Set initial input weights
        plasticity_pyr_params=pyr_params(w_ampa_ext_correct=1.1*nS, w_ampa_ext_incorrect=0.6*nS)
        plas_params=plasticity_params()

        # Create a virtual subject
        subject=VirtualSubject(subj_id, wta_params=wta_params, pyr_params=plasticity_pyr_params,
            plasticity_params=plas_params)

        # Run through each condition
        for condition, (sim_params, reinit, coherence_levels) in conditions.iteritems():

            # Reinitialize state variables in subject network
            if reinit:
                subject.net.reinit(states=True)

            # Run session
            run_session(subject, condition, sim_params, coherence_levels,
                output_file=os.path.join(output_dir, 'subject.%d.%s.h5' % (subj_id,condition)), plot=True)


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

    # Shuffle trials
    trial_inputs = np.random.permutation(trial_inputs)

    # Simulate each trial
    for t in range(sim_params.ntrials):
        print('Trial %d' % t)

        # Get task input for trial and figure out which is correct
        task_input_rates = trial_inputs[t, :]
        correct_input = np.where(task_input_rates == np.max(task_input_rates))[0]

        # Run trial
        subject.run_trial(sim_params, task_input_rates)

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
    Run subjects with no stimulation during training - runs baseline without stim, training without stim, and test with
    and without stim
    nsubjects = number of subjects to simulation
    coherence_levels = number of coherence levels to use
    """
    conditions = {
        # Baseline - 120 trials with no learning
        'baseline': (simulation_params(ntrials=trials_per_condition), True, coherence_levels),
        # Training - 120 trials with learning
        'training': (simulation_params(ntrials=trials_per_condition, plasticity=True), False, coherence_levels),
        # Test control - 120 trials with no learning and no stimulation
        'testing_control': (simulation_params(ntrials=trials_per_condition), True, coherence_levels),
    }
    # Test conditions - 120 trials with no learning and depolarizing or hyperpolarizing stimulation
    for stim_type, (p_dcs, i_dcs) in stim_types.iteritems():
        conditions['testing_%s' % stim_type] = (
        simulation_params(ntrials=trials_per_condition, p_dcs=p_dcs, i_dcs=i_dcs, dcs_start_time=0 * second,
            dcs_end_time=4 * second), True, coherence_levels)
    # Run subjects
    run_virtual_subjects(subj_ids, conditions, '/data/pySBI/stdp/control_learning',
        '/home/jbonaiuto/Projects/pySBI/data/rerw/subjects/fitted_behavioral_params.h5')


def run_stim_training_subjects(subj_ids, stim_types, coherence_levels):
    # Run subjects with depolarizing or hyperpolarizing stimulation during training
    for stim_type, (p_dcs, i_dcs) in stim_types.iteritems():
        conditions={
            # Baseline - 120 trials with no learning and no stimulation
            'baseline': (simulation_params(ntrials=120), True, coherence_levels),
            # Training - 120 trials with learning and stimulation
            'training': (simulation_params(ntrials=120, plasticity=True, p_dcs=p_dcs, i_dcs=i_dcs, dcs_start_time=0*second,
                dcs_end_time=4*second), False, coherence_levels),
            # Testing - 120 trials with no learning and no stimulation
            'testing': (simulation_params(ntrials=120), True, coherence_levels),
            }
        # Run subjects
        run_virtual_subjects(subj_ids, conditions,
            '/data/pySBI/stdp/%s_learning' % stim_type,
            '/home/jbonaiuto/Projects/pySBI/data/rerw/subjects/fitted_behavioral_params.h5')

if __name__=='__main__':
    # Max stimulation intensity
    stim_intensity_max=2*pA
    # Stimulation intensities
    stim_types={
        'depolarizing': (stim_intensity_max, -0.5*stim_intensity_max),
        'hyperpolarizing': (-1*stim_intensity_max, 0.5*stim_intensity_max)
    }
    # Run on six coherence levels
    coherence_levels=[.016, .032, .064, .128, .256, .512]
    # Trials per condition
    trials_per_condition=120

    # Run subjects with no stimulation during training
    run_nostim_training_subjects(range(20), stim_types, coherence_levels, trials_per_condition)

    # Run subjects with stimulation during training
    run_stim_training_subjects(range(20), stim_types, coherence_levels)

