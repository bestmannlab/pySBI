import argparse
import os
import subprocess
from brian.clock import defaultclock
from brian.stdunits import ms, pA, Hz
from brian.units import second
import numpy as np
import scipy.io
import h5py
from pysbi.wta.analysis import get_response_time
from pysbi.wta.network import default_params, run_wta
from pysbi.wta.rl.analysis import FileInfo
from pysbi.wta.rl.fit import fit_behavior, stim_order, LAT, NOSTIM1


def run_rl_simulation(mat_file, wta_params, alpha=0.4, background_freq=5.0, p_dcs=0*pA, i_dcs=0*pA, output_file=None):
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

    num_groups=2
    exp_rew=np.array([0.5, 0.5])
    trial_duration=4*second

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
        f.attrs['pyr_w_ampa_bak']=wta_params.pyr_w_ampa_bak
        f.attrs['pyr_w_ampa_rec']=wta_params.pyr_w_ampa_rec
        f.attrs['int_w_ampa_ext']=wta_params.int_w_ampa_ext
        f.attrs['int_w_ampa_bak']=wta_params.int_w_ampa_bak
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
            p_dcs=p_dcs, i_dcs=i_dcs, record_lfp=False, record_voxel=False, record_neuron_state=False,
            record_spikes=False, record_firing_rate=True, record_inputs=False, plot_output=False)

        trial_group=f.create_group('trial %d' % trial)
        e_rates = []
        for i in range(num_groups):
            e_rates.append(trial_monitor.monitors['excitatory_rate_%d' % i].smooth_rate(width=5 * ms, filter='gaussian'))
        trial_group['e_rates'] = np.array(e_rates)

        i_rates = [trial_monitor.monitors['inhibitory_rate'].smooth_rate(width=5 * ms, filter='gaussian')]
        trial_group['i_rates'] = np.array(i_rates)

        rt,decision_idx=get_response_time(e_rates, 1*second, 2*second)

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


def simulate_subject(control_mat_file, stim_mat_file, wta_params, alpha, beta, output_file):
    background_freq=(beta-87.46)/-12.5
    run_rl_simulation(control_mat_file, wta_params, alpha=alpha, background_freq=background_freq,
        output_file=output_file % 'control')
    run_rl_simulation(stim_mat_file, wta_params, alpha=alpha, background_freq=background_freq, p_dcs=4*pA, i_dcs=-2*pA,
        output_file=output_file % 'anode')


def resume_subject_simulation(data_dir, num_virtual_subjects):
    for i in range(num_virtual_subjects):
        subj_filename=os.path.join(data_dir,'virtual_subject_%d.control.h5' % i)
        print(subj_filename)
        if os.path.exists(subj_filename):
            data=FileInfo(subj_filename)
            beta=(data.background_freq/Hz*-12.5)+87.46
            stim_file_name=find_matching_subject_stim_file(os.path.join(data_dir,'subjects'), data.prob_walk, 24)
            if stim_file_name is not None:
                file_base='virtual_subject_'+str(i)+'.%s'
                out_file='../../data/rerw/%s.h5' % file_base
                log_filename='%s.txt' % file_base
                log_file=open(log_filename,'wb')
                args=['nohup','python','pysbi/wta/rl/network.py','--control_mat_file','nothing','--stim_mat_file',
                      stim_file_name,'--p_b_e',str(data.wta_params.p_b_e),'--p_x_e',str(data.wta_params.p_x_e),
                      '--alpha',str(data.alpha),'--beta',str(beta), '--output_file',out_file]
                subprocess.Popen(args,stdout=log_file)
            else:
                print('stim file for subjec %d not found' % i)

def find_matching_subject_stim_file(data_dir, control_prob_walk, num_real_subjects):
    for j in range(num_real_subjects):
        subj_id=j+1
        subj_stim_session_number=stim_order[j,LAT]
        stim_file_name=os.path.join(data_dir,'value%d_s%d_t2.mat' % (subj_id,subj_stim_session_number))
        subj_control_session_number=stim_order[j,NOSTIM1]
        control_file_name=os.path.join(data_dir,'value%d_s%d_t2.mat' % (subj_id,subj_control_session_number))
        if os.path.exists(stim_file_name) and os.path.exists(control_file_name):
            mat = scipy.io.loadmat(control_file_name)
            prob_idx=-1
            for idx,(dtype,o) in enumerate(mat['store']['dat'][0][0].dtype.descr):
                if dtype=='probswalk':
                    prob_idx=idx
            prob_walk=mat['store']['dat'][0][0][0][0][prob_idx]
            prob_walk=prob_walk.astype(np.float32, copy=False)
            match=True
            for k in range(prob_walk.shape[0]):
                for l in range(prob_walk.shape[1]):
                    if not prob_walk[k,l]==control_prob_walk[k,l]:
                        match=False
                        break
            if match:
                return stim_file_name
    return None

def simulate_subjects(data_dir, num_real_subjects, num_virtual_subjects, behavioral_param_file, p_b_e, p_x_e):
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])
    for j in range(num_virtual_subjects):
        stim_file_name=None
        control_file_name=None
        while True:
            i=np.random.choice(range(num_real_subjects))
            subj_id=i+1
            subj_stim_session_number=stim_order[i,LAT]
            stim_file_name=os.path.join(data_dir,'value%d_s%d_t2.mat' % (subj_id,subj_stim_session_number))
            subj_control_session_number=stim_order[i,NOSTIM1]
            control_file_name=os.path.join(data_dir,'value%d_s%d_t2.mat' % (subj_id,subj_control_session_number))
            if os.path.exists(stim_file_name) and os.path.exists(control_file_name):
                break
        alpha_hist,alpha_bins=np.histogram(alpha_vals, density=True)
        bin_width=alpha_bins[1]-alpha_bins[0]
        alpha_bin=np.random.choice(alpha_bins[:-1], p=alpha_hist*bin_width)
        alpha=alpha_bin+np.random.rand()*bin_width
        beta_hist,beta_bins=np.histogram(beta_vals, density=True)
        bin_width=beta_bins[1]-beta_bins[0]
        beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        beta=beta_bin+np.random.rand()*bin_width
        file_base='virtual_subject_'+str(j)+'.%s'
        out_file='../../data/rerw/%s.h5' % file_base
        log_filename='%s.txt' % file_base
        log_file=open(log_filename,'wb')
        args=['nohup','python','pysbi/wta/rl/network.py','--control_mat_file',control_file_name,'--stim_mat_file',
              stim_file_name,'--p_b_e',str(p_b_e),'--p_x_e',str(p_x_e),'--alpha',str(alpha),'--beta',str(beta),
              '--output_file',out_file]
        subprocess.Popen(args,stdout=log_file)

def launch_background_freq_processes(background_freq_range, p_b_e, p_x_e, trials):
    for background_freq in background_freq_range:
        for trial in range(trials):
            file_base='noise.background_%.2f.p_b_e_%0.4f.p_x_e_%0.4f.trial_%d' % (background_freq,p_b_e,p_x_e,trial)
            out_file='../../data/rerw/%s.h5' % file_base
            log_filename='%s.txt' % file_base
            log_file=open(log_filename,'wb')
            args=['nohup','python','pysbi/wta/rl/network.py','--background',str(background_freq),'--p_b_e',str(p_b_e),
                  '--p_x_e',str(p_x_e),'--output_file',out_file]
            subprocess.Popen(args,stdout=log_file)

if __name__=='__main__':
    #simulate_subjects('../../data/rerw/subjects/',24,50,'../../data/rerw/subjects/fitted_behavioral_params.h5',0.03,0.06)
    ap = argparse.ArgumentParser(description='Simulate a subject')
    ap.add_argument('--control_mat_file', type=str, default='../../data/rerw/subjects/value1_s1_t2.mat', help='Subject control mat file')
    ap.add_argument('--stim_mat_file', type=str, default='../../data/rerw/subjects/value1_s1_t2.mat', help='Subject stim mat file')
    ap.add_argument('--p_x_e', type=float, default=0.01, help='Connection prob from task inputs to excitatory neurons')
    ap.add_argument('--p_b_e', type=float, default=0.03, help='Connection prob from background to excitatory neurons')
    ap.add_argument('--alpha', type=float, default=0.4, help='Learning rate')
    ap.add_argument('--beta', type=float, default=5.0, help='Temperature')
    ap.add_argument('--output_file', type=str, default=None, help='HDF5 output file')

    argvals = ap.parse_args()

    wta_params=default_params
    wta_params.p_b_e=argvals.p_b_e
    wta_params.p_x_e=argvals.p_x_e

    simulate_subject(argvals.control_mat_file, argvals.stim_mat_file, wta_params, argvals.alpha, argvals.beta, argvals.output_file)
