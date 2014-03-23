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
    prob_walk=prob_walk.astype(np.float32, copy=False)
    mags=mags.astype(np.float32, copy=False)
    mags /= 100.0

    num_groups=2
    exp_rew=np.array([0.5, 0.5])
    trial_duration=3*second
    alpha=0.4

    trials=200

    choice=np.zeros(trials)
    rew=np.zeros(trials)
    for trial in range(trials):
        ev=exp_rew*mags[:,trial]
        input_freq=np.array([ev[0]-ev[1], ev[1]-ev[0]])
        input_freq=10.0+input_freq*2.5

        trial_monitor=run_wta(wta_params, num_groups, input_freq, trial_duration, background_freq=background_freq,
            record_lfp=False, record_voxel=False, record_neuron_state=False, record_spikes=False,
            record_firing_rate=True, record_inputs=False, plot_output=False)

        rate1_monitor=trial_monitor.monitors['excitatory_rate_0']
        e_rate1=rate1_monitor.smooth_rate(width=5*ms, filter='gaussian')
        rate2_monitor=trial_monitor.monitors['excitatory_rate_1']
        e_rate2=rate2_monitor.smooth_rate(width=5*ms, filter='gaussian')
        times=np.array(range(len(e_rate1)))*.0001
        decision_idx=-1
        for idx,time in enumerate(times):
            if e_rate1[idx]>=30 and e_rate1[idx]>e_rate2[idx]:
                decision_idx=0
            elif e_rate2[idx]>=30 and e_rate2[idx]>e_rate1[idx]:
                decision_idx=1

        print('Input frequencies=[%.2f, %.2f]' % (input_freq[0],input_freq[1]))
        print('Expected reward=[%.2f, %.2f]' % (exp_rew[0],exp_rew[1]))
        print('Decision=%d' % decision_idx)

        reward=0.0
        if np.random.random()<=prob_walk[decision_idx,trial]:
            reward=1.0
        print('Reward=%.2f' % reward)

        exp_rew[decision_idx]=(1.0-alpha)*exp_rew[decision_idx]+alpha*reward
        choice[trial]=decision_idx
        rew[trial]=reward

    param_ests,prop_correct=fit_behavior(prob_walk, mags, rew, choice)
    f = h5py.File(output_file, 'w')
    f.attrs['p_b_e']=wta_params.p_b_e
    f.attrs['background_freq']=background_freq
    f.attrs['alpha']=param_ests[0]
    f.attrs['beta']=param_ests[1]
    f.attrs['prop_correct']=prop_correct
    f.close()


if __name__=='__main__':
    ap = argparse.ArgumentParser(description='Run the WTA model')
    ap.add_argument('--mat_file', type=str, default='../../data/rerw/subjects/value1_s1_t2.mat', help='Subject mat file')
    ap.add_argument('--background', type=float, default=5.0, help='Background firing rate (Hz)')
    ap.add_argument('--p_x_e', type=float, default=0.01, help='Connection prob from task inputs to excitatory neurons')
    ap.add_argument('--p_b_e', type=float, default=0.03, help='Connection prob from background to excitatory neurons')
    ap.add_argument('--output_file', type=str, default=None, help='HDF5 output file')

    argvals = ap.parse_args()

    wta_params=default_params
    wta_params.p_b_e=argvals.p_b_e
    wta_params.p_x_e=argvals.p_x_e

    run_rl_simulation(argvals.mat_file, wta_params, background_freq=argvals.background, output_file=argvals.output_file)