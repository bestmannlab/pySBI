import argparse
from brian.stdunits import ms, pA, Hz
from brian.units import second
import numpy as np
import scipy.io
import h5py
from pysbi.util.utils import get_response_time
from pysbi.wta.network import default_params, run_wta, pyr_params, inh_params, simulation_params
from pysbi.wta.rl.fit import fit_behavior


def run_rl_simulation(mat_file, alpha=0.4, beta=5.0, background_freq=None, p_dcs=0*pA, i_dcs=0*pA, dcs_start_time=0*ms,
                      output_file=None):
    mat = scipy.io.loadmat(mat_file)
    prob_idx=-1
    mags_idx=-1
    for idx,(dtype,o) in enumerate(mat['store']['dat'][0][0].dtype.descr):
        if dtype=='probswalk':
            prob_idx=idx
        elif dtype=='mags':
            mags_idx=idx
    prob_walk=mat['store']['dat'][0][0][0][0][prob_idx]
    mags=mat['store']['dat'][0][0][0][0][mags_idx]
    prob_walk=prob_walk.astype(np.float32, copy=False)
    mags=mags.astype(np.float32, copy=False)
    mags /= 100.0

    wta_params=default_params()
    wta_params.input_var=0*Hz

    sim_params=simulation_params()
    sim_params.p_dcs=p_dcs
    sim_params.i_dcs=i_dcs
    sim_params.dcs_start_time=dcs_start_time

    exp_rew=np.array([0.5, 0.5])
    if background_freq is None:
        background_freq=(beta-161.08)/-.17
    wta_params.background_freq=background_freq


    trials=prob_walk.shape[1]
    sim_params.ntrials=trials

    vals=np.zeros(prob_walk.shape)
    choice=np.zeros(trials)
    rew=np.zeros(trials)
    rts=np.zeros(trials)
    inputs=np.zeros(prob_walk.shape)

    if output_file is not None:
        f = h5py.File(output_file, 'w')

        f.attrs['alpha']=alpha
        f.attrs['beta']=beta
        f.attrs['mat_file']=mat_file

        f_sim_params=f.create_group('sim_params')
        for attr, value in sim_params.iteritems():
            f_sim_params.attrs[attr] = value

        f_network_params=f.create_group('network_params')
        for attr, value in wta_params.iteritems():
            f_network_params.attrs[attr] = value

        f_pyr_params=f.create_group('pyr_params')
        for attr, value in pyr_params.iteritems():
            f_pyr_params.attrs[attr] = value

        f_inh_params=f.create_group('inh_params')
        for attr, value in inh_params.iteritems():
            f_inh_params.attrs[attr] = value

    for trial in range(sim_params.ntrials):
        print('Trial %d' % trial)
        vals[:,trial]=exp_rew
        ev=vals[:,trial]*mags[:,trial]
        inputs[0,trial]=ev[0]
        inputs[1,trial]=ev[1]
        inputs[:,trial]=40.0+40.0*inputs[:,trial]

        trial_monitor=run_wta(wta_params, inputs[:,trial], sim_params, record_lfp=False, record_voxel=False,
            record_neuron_state=False, record_spikes=True, record_firing_rate=True, record_inputs=False,
            plot_output=False)

        e_rates = []
        for i in range(wta_params.num_groups):
            e_rates.append(trial_monitor.monitors['excitatory_rate_%d' % i].smooth_rate(width=5 * ms, filter='gaussian'))
        i_rates = [trial_monitor.monitors['inhibitory_rate'].smooth_rate(width=5 * ms, filter='gaussian')]

        if output_file is not None:
            trial_group=f.create_group('trial %d' % trial)
            trial_group['e_rates'] = np.array(e_rates)

            trial_group['i_rates'] = np.array(i_rates)

        rt,decision_idx=get_response_time(e_rates, sim_params.stim_start_time, sim_params.stim_end_time,
            upper_threshold=wta_params.resp_threshold, lower_threshold=None, dt=sim_params.dt)

        reward=0.0
        if decision_idx>=0 and np.random.random()<=prob_walk[decision_idx,trial]:
            reward=1.0

        exp_rew[decision_idx]=(1.0-alpha)*exp_rew[decision_idx]+alpha*reward
        choice[trial]=decision_idx
        rts[trial]=rt
        rew[trial]=reward

    param_ests,prop_correct=fit_behavior(prob_walk, mags, rew, choice)

    if output_file is not None:
        f.attrs['est_alpha']=param_ests[0]
        f.attrs['est_beta']=param_ests[1]
        f.attrs['prop_correct']=prop_correct

        f['prob_walk']=prob_walk
        f['mags']=mags
        f['rew']=rew
        f['choice']=choice
        f['vals']=vals
        f['inputs']=inputs
        f['rts']=rts
        f.close()


if __name__=='__main__':
    ap = argparse.ArgumentParser(description='Simulate a subject')
    ap.add_argument('--stim_mat_file', type=str, default='../../data/rerw/subjects/value1_s1_t2.mat', help='Subject stim mat file')
    ap.add_argument('--p_dcs', type=float, default=0.0, help='Pyramidal cell DCS')
    ap.add_argument('--i_dcs', type=float, default=0.0, help='Interneuron cell DCS')
    ap.add_argument('--dcs_start_time', type=float, default=0.0, help='Time to start dcs')
    ap.add_argument('--alpha', type=float, default=0.4, help='Learning rate')
    ap.add_argument('--beta', type=float, default=5.0, help='Temperature')
    ap.add_argument('--background', type=float, default=None, help='Background firing rate (Hz)')
    ap.add_argument('--output_file', type=str, default=None, help='HDF5 output file')

    argvals = ap.parse_args()

    run_rl_simulation(argvals.stim_mat_file, alpha=argvals.alpha, beta=argvals.beta, background_freq=argvals.background,
        p_dcs=argvals.p_dcs*pA, i_dcs=argvals.i_dcs*pA, dcs_start_time=argvals.dcs_start_time*second,
        output_file=argvals.output_file)
