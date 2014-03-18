import argparse
from brian.clock import defaultclock
from brian.stdunits import ms
from brian.units import second
import numpy as np
import scipy.io
import h5py
from pysbi.wta.network import default_params, run_wta
from pysbi.wta.rl.fit import fit_behavior


def run_rl_simulation(mat_file, wta_params, background_freq=5, output_file=None):
    mat = scipy.io.loadmat(mat_file)
    prob_walk=mat['store']['dat'][0][0][0][0][13]
    mags=mat['store']['dat'][0][0][0][0][15]

    num_groups=2
    exp_rew=np.array([0.5, 0.5])
    trial_duration=3*second
    alpha=0.4

    trials=200

    choice=np.zeros(trials)
    rew=np.zeros(trials)
    for trial in range(trials):
        input_freq=np.array([exp_rew[0]-exp_rew[1], exp_rew[1]-exp_rew[0]])
        input_freq=10.0+input_freq*2.5
        reward_probs=prob_walk[:,trial]
        reward_mags=mags[:,trial]/100.0
        trial_monitor=run_wta(wta_params, num_groups, input_freq, trial_duration, background_freq=background_freq,
            record_lfp=False, record_voxel=False, record_neuron_state=False, record_spikes=False,
            record_firing_rate=True, record_inputs=False, plot_output=False)
        endIdx=int((trial_duration-1*second)/defaultclock.dt)
        startIdx=endIdx-500
        e_mean_final=[]
        for i in range(num_groups):
            rate_monitor=trial_monitor.monitors['excitatory_rate_%d' % i]
            e_rate=rate_monitor.smooth_rate(width=5*ms, filter='gaussian')
            e_mean_final.append(np.mean(e_rate[startIdx:endIdx]))
        decision_idx=0
        if e_mean_final[1]>e_mean_final[0]:
            decision_idx=1
        print('Input frequencies=[%.2f, %.2f]' % (input_freq[0],input_freq[1]))
        print('Expected reward=[%.2f, %.2f]' % (exp_rew[0],exp_rew[1]))
        print('Decision=%d' % decision_idx)
        reward=0.0
        if np.random.random()<=reward_probs[decision_idx]:
            reward=reward_mags[decision_idx]
        print('Reward=%.2f' % reward)
        exp_rew[decision_idx]=(1.0-alpha)*exp_rew[decision_idx]+alpha*reward
        choice[trial]=decision_idx
        rew[trial]=reward

    param_ests,prop_correct=fit_behavior(prob_walk, mags, rew, choice)
    f = h5py.File(output_file, 'w')
    f.attrs['alpha']=param_ests[0]
    f.attrs['beta']=param_ests[1]
    f.attrs['prop_correct']=prop_correct
    f.close()


if __name__=='__main__':
    ap = argparse.ArgumentParser(description='Run the WTA model')
    ap.add_argument('--mat_file', type=str, default='../../data/rerw/subjects/value1_s1_t2.mat', help='Subject mat file')
    ap.add_argument('--background', type=float, default=5.0, help='Background firing rate (Hz)')
    ap.add_argument('--p_b_e', type=float, default=0.03, help='Connection prob from background to excitatory neurons')
    ap.add_argument('--output_file', type=str, default=None, help='HDF5 output file')

    argvals = ap.parse_args()

    wta_params=default_params
    wta_params.p_b_e=argvals.p_b_e

    run_rl_simulation(argvals.mat_file, wta_params, background_freq=argvals.background, output_file=argvals.output_file)