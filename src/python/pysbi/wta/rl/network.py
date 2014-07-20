import argparse
from brian.stdunits import ms, pA
from brian.units import second
import numpy as np
import scipy.io
import h5py
from pysbi.util.utils import get_response_time
from pysbi.wta.network import default_params, run_wta, pyr_params, inh_params
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

    resp_thresh=25
    num_groups=2
    exp_rew=np.array([0.5, 0.5])
    trial_duration=4*second
    if background_freq is None:
        #background_freq=(beta-87.46)/-12.5
        background_freq=(beta-148.14)/-17.29

    trials=prob_walk.shape[1]

    vals=np.zeros(prob_walk.shape)
    choice=np.zeros(trials)
    rew=np.zeros(trials)
    rts=np.zeros(trials)
    inputs=np.zeros(prob_walk.shape)

    if output_file is not None:
        f = h5py.File(output_file, 'w')
        f.attrs['trials']=trials
        f.attrs['alpha']=alpha
        f.attrs['beta']=beta
        f.attrs['mat_file']=mat_file
        f.attrs['resp_threshold']=resp_thresh
        f.attrs['num_groups'] = num_groups
        f.attrs['trial_duration'] = trial_duration
        f.attrs['background_freq'] = background_freq
        f.attrs['C'] = default_params.C
        f.attrs['gL'] = default_params.gL
        f.attrs['EL'] = default_params.EL
        f.attrs['VT'] = default_params.VT
        f.attrs['Vr'] = default_params.Vr
        f.attrs['DeltaT'] = default_params.DeltaT
        f.attrs['Mg'] = default_params.Mg
        f.attrs['E_ampa'] = default_params.E_ampa
        f.attrs['E_nmda'] = default_params.E_nmda
        f.attrs['E_gaba_a'] = default_params.E_gaba_a
        f.attrs['tau_ampa'] = default_params.tau_ampa
        f.attrs['tau1_nmda'] = default_params.tau1_nmda
        f.attrs['tau2_nmda'] = default_params.tau2_nmda
        f.attrs['tau_gaba_a'] = default_params.tau_gaba_a
        f.attrs['p_e_e'] = default_params.p_e_e
        f.attrs['p_e_i'] = default_params.p_e_i
        f.attrs['p_i_i'] = default_params.p_i_i
        f.attrs['p_i_e'] = default_params.p_i_e
        f.attrs['p_dcs']=p_dcs
        f.attrs['i_dcs']=i_dcs
        f.attrs['dcs_start_time']=dcs_start_time

        pyr_param_group=f.create_group('pyr_params')
        pyr_param_group.attrs['C']=pyr_params.C
        pyr_param_group.attrs['gL']=pyr_params.gL
        pyr_param_group.attrs['refractory']=pyr_params.refractory
        pyr_param_group.attrs['w_nmda']=pyr_params.w_nmda
        pyr_param_group.attrs['w_ampa_ext']=pyr_params.w_ampa_ext
        pyr_param_group.attrs['w_ampa_rec']=pyr_params.w_ampa_rec
        pyr_param_group.attrs['w_gaba']=pyr_params.w_gaba

        inh_param_group=f.create_group('inh_params')
        inh_param_group.attrs['C']=inh_params.C
        inh_param_group.attrs['gL']=inh_params.gL
        inh_param_group.attrs['refractory']=inh_params.refractory
        inh_param_group.attrs['w_nmda']=inh_params.w_nmda
        inh_param_group.attrs['w_ampa_ext']=inh_params.w_ampa_ext
        inh_param_group.attrs['w_ampa_rec']=inh_params.w_ampa_rec
        inh_param_group.attrs['w_gaba']=inh_params.w_gaba

    for trial in range(trials):
        print('Trial %d' % trial)
        vals[:,trial]=exp_rew
        ev=vals[:,trial]*mags[:,trial]
        inputs[0,trial]=ev[0]-ev[1]
        inputs[1,trial]=ev[1]-ev[0]
        inputs[:,trial]=40.0+40.0*(inputs[:,trial]+1.0)*.5

        trial_monitor=run_wta(default_params, num_groups, inputs[:,trial], trial_duration, background_freq=background_freq,
            p_dcs=p_dcs, i_dcs=i_dcs, dcs_start_time=dcs_start_time, record_lfp=False, record_voxel=False,
            record_neuron_state=False, record_spikes=False, record_firing_rate=True, record_inputs=False,
            plot_output=False)

        e_rates = []
        for i in range(num_groups):
            e_rates.append(trial_monitor.monitors['excitatory_rate_%d' % i].smooth_rate(width=5 * ms, filter='gaussian'))
        i_rates = [trial_monitor.monitors['inhibitory_rate'].smooth_rate(width=5 * ms, filter='gaussian')]

        if output_file is not None:
            trial_group=f.create_group('trial %d' % trial)
            trial_group['e_rates'] = np.array(e_rates)

            trial_group['i_rates'] = np.array(i_rates)

        rt,decision_idx=get_response_time(e_rates, 1*second, trial_duration-1*second, upper_threshold=resp_thresh,
            lower_threshold=None, dt=.5*ms)

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
