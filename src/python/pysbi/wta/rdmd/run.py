from brian import Hz, nS, pA, second
import os
from pysbi.wta.monitor import SessionMonitor
from pysbi.wta.network import default_params, pyr_params, simulation_params
from pysbi.wta.virtual_subject import VirtualSubject
import numpy as np
import matplotlib.pyplot as plt

def run_virtual_subjects(subj_ids, conditions, output_dir):
    for subj_id in subj_ids:
        wta_params=default_params(background_freq=950*Hz)
        pyramidal_params=pyr_params(w_nmda=0.14*nS)
        subject=VirtualSubject(subj_id, wta_params=wta_params, pyr_params=pyramidal_params)

        for condition, sim_params in conditions.iteritems():
            run_session(subject, sim_params, output_file=os.path.join(output_dir, 'subject.%d.%s.h5' % (subj_id,condition)))

def run_session(subject, sim_params, output_file, plot=False):

    session_monitor=SessionMonitor(subject.wta_network, sim_params, {}, record_connections=[], conv_window=10,
        record_firing_rates=True)

    coherence_levels=[0.032, .064, .128, .256, .512]
    trials_per_level=20
    trial_inputs=np.zeros((trials_per_level*len(coherence_levels),2))
    for i in range(len(coherence_levels)):
        coherence=coherence_levels[i]
        # Left
        trial_inputs[i*trials_per_level:i*trials_per_level+trials_per_level/2,0]=subject.wta_params.mu_0+subject.wta_params.p_a*coherence*100.0
        trial_inputs[i*trials_per_level:i*trials_per_level+trials_per_level/2,1]=subject.wta_params.mu_0-subject.wta_params.p_b*coherence*100.0

        #Right
        trial_inputs[i*trials_per_level+trials_per_level/2:i*trials_per_level+trials_per_level,0]=subject.wta_params.mu_0-subject.wta_params.p_b*coherence*100.0
        trial_inputs[i*trials_per_level+trials_per_level/2:i*trials_per_level+trials_per_level,1]=subject.wta_params.mu_0+subject.wta_params.p_a*coherence*100.0

    trial_inputs=np.random.permutation(trial_inputs)

    for t in range(sim_params.ntrials):
        task_input_rates=trial_inputs[t,:]
        correct_input=np.where(task_input_rates==np.max(task_input_rates))[0]
        subject.run_trial(sim_params, task_input_rates)
        session_monitor.record_trial(t, task_input_rates, correct_input, subject.wta_network, subject.wta_monitor)

    if output_file is not None:
        session_monitor.write_output(output_file)

    if plot:
        session_monitor.plot()
        plt.show()

if __name__=='__main__':
    conditions={
        'control': simulation_params(ntrials=100),
        'depolarizing': simulation_params(ntrials=100, p_dcs=2*pA, i_dcs=-1*pA, dcs_start_time=0*second, dcs_end_time=4*second),
        'hyperpolarizing': simulation_params(ntrials=100, p_dcs=-2*pA, i_dcs=1*pA, dcs_start_time=0*second, dcs_end_time=4*second)
    }
    run_virtual_subjects(range(20), conditions, '/home/jbonaiuto/Projects/pySBI/data/rdmd/')