import os
import random
import h5py
import numpy as np
from brian.units import second
from ezrcluster.launcher import Launcher
from scikits.learn.linear_model.base import LinearRegression
from pysbi.analysis import FileInfo, run_bayesian_analysis
from pysbi.config import SRC_DIR
from pysbi.random_distributions import make_distribution_curve
from pysbi.reports.utils import get_tested_param_combos

def get_wta_cmds(num_groups, inputs, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, trial, single_inh_pop=False,
                 record_lfp=True, record_voxel=False, record_neuron_state=False, record_spikes=True,
                 record_firing_rate=True):
    cmds = ['python', '/tmp/pySBI/src/python/pysbi/wta.py']
    file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.trial.%d' %\
              (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, trial)
    log_file_template='%s.log' % file_desc
    output_file='/tmp/wta-output/%s.h5' % file_desc
    cmds.append('--num_groups')
    cmds.append('%d' % num_groups)
    cmds.append('--inputs')
    cmds.append(','.join([str(input) for input in inputs]))
    cmds.append('--trial_duration')
    cmds.append('%0.3f' % trial_duration)
    cmds.append('--p_b_e')
    cmds.append('%0.3f' % p_b_e)
    cmds.append('--p_x_e')
    cmds.append('%0.3f' % p_x_e)
    cmds.append('--p_e_e')
    cmds.append('%0.3f' % p_e_e)
    cmds.append('--p_e_i')
    cmds.append('%0.3f' % p_e_i)
    cmds.append('--p_i_i')
    cmds.append('%0.3f' % p_i_i)
    cmds.append('--p_i_e')
    cmds.append('%0.3f' % p_i_e)
    cmds.append('--output_file')
    cmds.append(output_file)
    cmds.append('--single_inh_pop')
    if single_inh_pop:
        cmds.append('1')
    else:
        cmds.append('0')
    cmds.append('--record_lfp')
    if record_lfp:
        cmds.append('1')
    else:
        cmds.append('0')
    cmds.append('--record_voxel')
    if record_voxel:
        cmds.append('1')
    else:
        cmds.append('0')
    cmds.append('--record_neuron_state')
    if record_neuron_state:
        cmds.append('1')
    else:
        cmds.append('0')
    cmds.append('--record_spikes')
    if record_spikes:
        cmds.append('1')
    else:
        cmds.append('0')
    cmds.append('--record_firing_rate')
    if record_firing_rate:
        cmds.append('1')
    else:
        cmds.append('0')

    return cmds, log_file_template, output_file
    
def post_wta_jobs(nodes, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range, num_trials,
                  single_inh_pop=False, start_nodes=True):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()
    for p_b_e in p_b_e_range:
        for p_x_e in p_x_e_range:
            for p_e_e in p_e_e_range:
                for p_e_i in p_e_i_range:
                    for p_i_i in p_i_i_range:
                        for p_i_e in p_i_e_range:
                            for t in range(num_trials):
                                inputs=np.zeros(2)
                                inputs[0]=np.random.rand()*input_sum
                                inputs[1]=input_sum-inputs[0]
                                cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                                    p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, t, single_inh_pop=single_inh_pop, record_lfp=True,
                                    record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                                    record_spikes=True)
                                launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

def post_broken_wta_jobs(nodes, num_trials, data_path, single_inh_pop=False, start_nodes=True):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    contrast_range=[float(x)*(1.0/(num_trials-1)) for x in range(0,num_trials)]
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()
    param_combos=get_tested_param_combos(data_path, num_groups, trial_duration, num_trials)
    for (p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e) in param_combos:
        for t,contrast in enumerate(contrast_range):
            file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.trial.%d.h5' %\
                      (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, t)
            recreate=False
            file_name=os.path.join(data_path,file_desc)
            print('Checking %s' % file_name)
            if not os.path.exists(file_name):
                recreate=True
            else:
                try:
                    data=FileInfo(file_name)
                except Exception:
                    recreate=True
                    os.remove(file_name)
            if recreate:
                print('*** Recreating %s ***' % file_desc)
                inputs=np.zeros(2)
                inputs[0]=(input_sum*(contrast+1.0)/2.0)
                inputs[1]=input_sum-inputs[0]
                np.random.shuffle(inputs)
                cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                    p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, t, single_inh_pop=single_inh_pop,
                    record_lfp=True, record_voxel=True, record_neuron_state=False,
                    record_firing_rate=True, record_spikes=True)
                launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

def post_missing_wta_jobs(nodes, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
                          num_trials, data_path, single_inh_pop=False, start_nodes=True):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()
    for p_b_e in p_b_e_range:
        for p_x_e in p_x_e_range:
            for p_e_e in p_e_e_range:
                for p_e_i in p_e_i_range:
                    for p_i_i in p_i_i_range:
                        for p_i_e in p_i_e_range:
                            for t in range(num_trials):
                                file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.trial.%d.h5' %\
                                          (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, t)
                                recreate=False
                                file_name=os.path.join(data_path,file_desc)
                                print('Checking %s' % file_name)
                                if not os.path.exists(file_name):
                                    recreate=True
                                else:
                                    try:
                                        data=FileInfo(file_name)
                                    except Exception:
                                        recreate=True
                                        os.remove(file_name)
                                if recreate:
                                    print('*** Recreating %s ***' % file_desc)
                                    inputs=np.zeros(2)
                                    inputs[0]=np.random.rand()*input_sum
                                    inputs[1]=input_sum-inputs[0]
                                    cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                                        p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, t, single_inh_pop=single_inh_pop,
                                        record_lfp=True, record_voxel=True, record_neuron_state=False,
                                        record_firing_rate=True, record_spikes=True)
                                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)


def sample(param_marginal_posterior, param_range, alpha=0.1, precision=3):
    param_val = param_range[0]
    idx=0
    if not param_range[-1] == param_range[0]:
        curve = make_distribution_curve(param_range, param_marginal_posterior)
        uniformRandVal = random.random() #random value between 0 and 1
        if np.random.random()<alpha:
            param_val=uniformRandVal*(param_range[-1]-param_range[0])+param_range[0]
        else:
            #The random value that follows the desired distribution.
            param_val = curve(uniformRandVal)
        param_val=max(param_range[0],min(param_range[-1],round(param_val,precision)))
        idx = int((param_val - param_range[0]) / (param_range[-1] - param_range[0]) * (len(param_range)-1))
    return idx, param_val


def remove_probabilistic_sample_files(data_path, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
                                      num_trials, num_groups, trial_duration):
    for file in os.listdir(data_path):
        if file.endswith('.h5'):
            file_split=file.split('.')
            p_b_e=float('%s.%s' % (file_split[7],file_split[8]))
            p_x_e=float('%s.%s' % (file_split[10],file_split[11]))
            p_e_e=float('%s.%s' % (file_split[13],file_split[14]))
            p_e_i=float('%s.%s' % (file_split[16],file_split[17]))
            p_i_i=float('%s.%s' % (file_split[19],file_split[20]))
            p_i_e=float('%s.%s' % (file_split[22],file_split[23]))

            if not (p_b_e in p_b_e_range and p_x_e in p_x_e_range and p_e_e in p_e_e_range and p_e_i in p_e_i_range and
                p_i_i in p_i_i_range and p_i_e in p_i_e_range):
                for i in range(num_trials):
                    file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.trial.%d.h5' %\
                              (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, i)
                    file_prefix=os.path.join(data_path,file_desc)
                    if os.path.exists(file_prefix):
                        os.remove(file_prefix)

def probabilistic_sample(nodes, summary_filename, data_path, single_inh_pop=False, start_nodes=True, num_samples=0,
                         post_jobs=True):
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()
    if not num_samples:
        num_samples=len(nodes)

    f=h5py.File(summary_filename)
    num_groups=int(f.attrs['num_groups'])
    num_trials=int(f.attrs['num_trials'])
    trial_duration=float(f.attrs['trial_duration'])
    p_b_e_range=np.array(f['p_b_e_range'])
    p_x_e_range=np.array(f['p_x_e_range'])
    p_e_e_range=np.array(f['p_e_e_range'])
    p_e_i_range=np.array(f['p_e_i_range'])
    p_i_i_range=np.array(f['p_i_i_range'])
    p_i_e_range=np.array(f['p_i_e_range'])
    input_contrast=np.array(f['input_contrast'])
    max_bold=np.array(f['max_bold'])
    auc=np.array(f['auc'])
    f.close()

    bc_slope=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range), len(p_i_i_range),
                       len(p_i_e_range)])
    bc_intercept=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range), len(p_i_i_range),
                           len(p_i_e_range)])
    bc_rsqr=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range), len(p_i_i_range),
                       len(p_i_e_range)])
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for m,p_i_i in enumerate(p_i_i_range):
                        for n,p_i_e in enumerate(p_i_e_range):
                            if auc[i,j,k,l,m,n]>0:
                                combo_contrast = input_contrast[i, j, k, l, m, n, :]
                                combo_max_bold = max_bold[i, j, k, l, m, n, :]
                                clf = LinearRegression()
                                clf.fit(combo_contrast.reshape([num_trials, 1]), combo_max_bold)
                                bc_slope[i,j,k,l,m,n] = clf.coef_[0]
                                bc_intercept[i,j,k,l,m,n] = clf.intercept_
                                bc_rsqr[i,j,k,l,m,n]=clf.score(combo_contrast.reshape([num_trials, 1]), combo_max_bold)
                            else:
                                bc_slope[i,j,k,l,m,n] = 0
                                bc_intercept[i,j,k,l,m,n] = 0
                                bc_rsqr[i,j,k,l,m,n]=0

    bayes_analysis=run_bayesian_analysis(auc, bc_slope, bc_intercept, bc_rsqr, num_trials, p_b_e_range, p_e_e_range,
        p_e_i_range, p_i_e_range, p_i_i_range, p_x_e_range)

    samples=[]
    posterior=bayes_analysis.l1_pos_posterior
    while len(samples)<num_samples:
        p_b_e_posterior=np.sum(np.sum(np.sum(np.sum(np.sum(posterior,axis=1),axis=1),axis=1),axis=1),axis=1)
        i, p_b_e = sample(p_b_e_posterior, p_b_e_range)

        p_x_e_posterior=np.sum(np.sum(np.sum(np.sum(posterior[i,:,:,:,:,:],axis=1),axis=1),axis=1),axis=1)
        j,p_x_e = sample(p_x_e_posterior, p_x_e_range)

        p_e_e_posterior=np.sum(np.sum(np.sum(posterior[i,j,:,:,:,:],axis=1),axis=1),axis=1)
        k,p_e_e = sample(p_e_e_posterior, p_e_e_range)

        p_e_i_posterior=np.sum(np.sum(posterior[i,j,k,:,:,:],axis=1),axis=1)
        l,p_e_i=sample(p_e_i_posterior, p_e_i_range)

        p_i_i_posterior=np.sum(posterior[i,j,k,l,:,:],axis=1)
        m,p_i_i=sample(p_i_i_posterior, p_i_i_range)

        p_i_e_posterior=posterior[i,j,k,l,m,:]
        n,p_i_e=sample(p_i_e_posterior, p_i_e_range)

        file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.trial.0.h5' %\
                  (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e)
        file_name=os.path.join(data_path,file_desc)
        if not os.path.exists(file_name) and not (p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e) in samples:
            print('p_b_e=%0.3f, p_x_e=%0.3f, p_e_e=%0.3f, p_e_i=%0.3f, p_i_i=%0.3f, p_i_e=%0.3f' % (p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e))
            samples.append((p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e))
            if post_jobs:
                input_sum=40.0
                contrast_range=[float(x)*(1.0/(num_trials-1)) for x in range(0,num_trials)]
                for t,contrast in enumerate(contrast_range):
                    inputs=np.zeros(2)
                    inputs[0]=(input_sum*(contrast+1.0)/2.0)
                    inputs[1]=input_sum-inputs[0]
                    np.random.shuffle(inputs)
                    cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                        p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, t, single_inh_pop=single_inh_pop, record_lfp=True,
                        record_voxel=True, record_neuron_state=False, record_firing_rate=True, record_spikes=True)
                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)
    return np.array(samples)
