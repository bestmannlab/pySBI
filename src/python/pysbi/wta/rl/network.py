import argparse
import subprocess
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

    trials=prob_walk.shape[1]

    vals=np.zeros(prob_walk.shape)
    choice=np.zeros(trials)
    rew=np.zeros(trials)
    inputs=np.zeros(prob_walk.shape)

    if output_file is not None:
        f = h5py.File(output_file, 'w')
        f.attrs['trials']=trials
        f.attrs['alpha']=alpha

        f.attrs['num_groups'] = num_groups
        f.attrs['trial_duration'] = trial_duration
        f.attrs['background_freq'] = background_freq
        f.attrs['C'] = wta_params.C
        f.attrs['gL'] = wta_params.gL
        f.attrs['EL'] = wta_params.EL
        f.attrs['VT'] = wta_params.VT
        f.attrs['Mg'] = wta_params.Mg
        f.attrs['DeltaT'] = wta_params.DeltaT
        f.attrs['E_ampa'] = wta_params.E_ampa
        f.attrs['E_nmda'] = wta_params.E_nmda
        f.attrs['E_gaba_a'] = wta_params.E_gaba_a
        f.attrs['tau_ampa'] = wta_params.tau_ampa
        f.attrs['tau1_nmda'] = wta_params.tau1_nmda
        f.attrs['tau2_nmda'] = wta_params.tau2_nmda
        f.attrs['tau_gaba_a'] = wta_params.tau_gaba_a
        f.attrs['pyr_w_ampa_ext']=wta_params.pyr_w_ampa_ext
        f.attrs['pyr_w_ampa_rec']=wta_params.pyr_w_ampa_rec
        f.attrs['int_w_ampa_ext']=wta_params.int_w_ampa_ext
        f.attrs['int_w_ampa_rec']=wta_params.int_w_ampa_rec
        f.attrs['pyr_w_nmda']=wta_params.pyr_w_nmda
        f.attrs['int_w_nmda']=wta_params.int_w_nmda
        f.attrs['pyr_w_gaba_a']=wta_params.pyr_w_gaba_a
        f.attrs['int_w_gaba_a']=wta_params.int_w_gaba_a
        f.attrs['p_b_e'] = wta_params.p_b_e
        f.attrs['p_x_e'] = wta_params.p_x_e
        f.attrs['p_e_e'] = wta_params.p_e_e
        f.attrs['p_e_i'] = wta_params.p_e_i
        f.attrs['p_i_i'] = wta_params.p_i_i
        f.attrs['p_i_e'] = wta_params.p_i_e
        #f.attrs['p_dcs']=p_dcs
        #f.attrs['i_dcs']=i_dcs

    for trial in range(trials):
        vals[:,trial]=exp_rew
        ev=vals[:,trial]*mags[:,trial]
        inputs[0,trial]=ev[0]-ev[1]
        inputs[1,trial]=ev[1]-ev[0]
        inputs[:,trial]=10.0+inputs[:,trial]*2.5

        trial_monitor=run_wta(wta_params, num_groups, inputs[:,trial], trial_duration, background_freq=background_freq,
            record_lfp=False, record_voxel=False, record_neuron_state=False, record_spikes=False,
            record_firing_rate=True, record_inputs=False, plot_output=False)

        trial_group=f.create_group('trial %d' % trial)
        e_rates = []
        for i in range(num_groups):
            e_rates.append(trial_monitor.monitors['excitatory_rate_%d' % i].smooth_rate(width=5 * ms, filter='gaussian'))
        trial_group['e_rates'] = np.array(e_rates)

        i_rates = [trial_monitor.monitors['inhibitory_rate'].smooth_rate(width=5 * ms, filter='gaussian')]
        trial_group['i_rates'] = np.array(i_rates)

        times=np.array(range(len(e_rates[0])))*.0001
        decision_idx=-1
        for idx,time in enumerate(times):
            if e_rates[0][idx]>=30 and e_rates[0][idx]>e_rates[1][idx]:
                decision_idx=0
            elif e_rates[1][idx]>=30 and e_rates[1][idx]>e_rates[0][idx]:
                decision_idx=1

        reward=0.0
        if np.random.random()<=prob_walk[decision_idx,trial]:
            reward=1.0

        exp_rew[decision_idx]=(1.0-alpha)*exp_rew[decision_idx]+alpha*reward
        choice[trial]=decision_idx
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
        f.close()

def launch_processes(background_freq_range, p_b_e_range,p_x_e):
    for background_freq in background_freq_range:
        for p_b_e in p_b_e_range:
            file_base='noise.background_%.2f.p_b_e_%0.4f.p_x_e_%0.4f' % (background_freq,p_b_e,p_x_e)
            out_file='../../data/rerw/%s.h5' % file_base
            log_filename='%s.txt' % file_base
            log_file=open(log_filename,'wb')
            args=['nohup','python','pysbi/wta/rl/network.py','--background',str(background_freq),'--p_b_e',str(p_b_e),
                  '--p_x_e',str(p_x_e),'--output_file',out_file]
            subprocess.Popen(args,stdout=log_file)

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