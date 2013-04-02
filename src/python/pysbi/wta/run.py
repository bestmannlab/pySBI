import os
import random
from brian.stdunits import nS
import numpy as np
from brian.units import second, siemens
from ezrcluster.launcher import Launcher
from pysbi.wta.analysis import FileInfo, run_bayesian_analysis
from pysbi.config import SRC_DIR
from pysbi.util.random_distributions import make_distribution_curve
from pysbi.reports.summary import SummaryData
from pysbi.reports.utils import get_tested_param_combos

def get_wta_cmds(num_groups, inputs, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, contrast,trial, single_inh_pop=False,
                 muscimol_amount=0*nS, injection_site=0, record_lfp=True, record_voxel=False, record_neuron_state=False, record_spikes=True,
                 record_firing_rate=True, save_summary_only=True):
    cmds = ['python', '/tmp/pySBI/src/python/pysbi/wta/network.py']
    if muscimol_amount>0:
        e_desc='lesioned'
    else:
        e_desc='control'
    file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.%s.contrast.%0.4f.trial.%d' %\
              (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, e_desc, contrast, trial)
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
    if muscimol_amount>0:
        cmds.append('--muscimol_amount')
        cmds.append(str(muscimol_amount/siemens))
        cmds.append('--injection_site')
        cmds.append('%d' % injection_site)
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
    cmds.append('--save_summary_only')
    if save_summary_only:
        cmds.append('1')
    else:
        cmds.append('0')

    return cmds, log_file_template, output_file

def post_missing_one_param_wta_jobs(nodes, p_b_e, p_x_e, p_range, num_trials, data_dir, single_inh_pop=True, start_nodes=True,
                                    muscimol_amount=0*nS, injection_site=0):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
    for p in p_range:
        for i,contrast in enumerate(contrast_range):
            for t in range(num_trials):
                file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.%s.contrast.%0.4f.trial.%d.h5' %\
                          (num_groups, trial_duration, p_b_e, p_x_e, p, p, p, p, 'control', contrast, t)
                file_name=os.path.join(data_dir, file_desc)
                if not os.path.exists(file_name):
                    inputs=np.zeros(2)
                    inputs[0]=(input_sum*(contrast+1.0)/2.0)
                    inputs[1]=input_sum-inputs[0]
                    np.random.shuffle(inputs)
                    cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                        p_x_e, p, p, p, p, contrast, t, single_inh_pop=single_inh_pop, record_lfp=True,
                        record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                        record_spikes=True, save_summary_only=True)
                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

                file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.%s.contrast.%0.4f.trial.%d' %\
                          (num_groups, trial_duration, p_b_e, p_x_e, p, p, p, p, 'control', contrast, t)
                file_name=os.path.join(data_dir, file_desc)
                if not os.path.exists(file_name):
                    inputs=np.zeros(2)
                    inputs[0]=(input_sum*(contrast+1.0)/2.0)
                    inputs[1]=input_sum-inputs[0]
                    np.random.shuffle(inputs)
                    cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                        p_x_e, p, p, p, p, contrast, t, single_inh_pop=single_inh_pop, muscimol_amount=muscimol_amount,
                        injection_site=injection_site, record_lfp=True, record_voxel=True, record_neuron_state=False,
                        record_firing_rate=True, record_spikes=True, save_summary_only=True)
                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

def post_one_param_wta_jobs(nodes, p_b_e, p_x_e, p_range, num_trials, single_inh_pop=True, start_nodes=True,
                            muscimol_amount=0*nS, injection_site=0):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
    for p in p_range:
        for i,contrast in enumerate(contrast_range):
            inputs=np.zeros(2)
            inputs[0]=(input_sum*(contrast+1.0)/2.0)
            inputs[1]=input_sum-inputs[0]
            for t in range(num_trials):
                np.random.shuffle(inputs)
                cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                    p_x_e, p, p, p, p, contrast, t, single_inh_pop=single_inh_pop, record_lfp=True,
                    record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                    record_spikes=True, save_summary_only=True)
                launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)
            for t in range(num_trials):
                np.random.shuffle(inputs)
                cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                    p_x_e, p, p, p, p, contrast, t, single_inh_pop=single_inh_pop, muscimol_amount=muscimol_amount,
                    injection_site=injection_site, record_lfp=True, record_voxel=True, record_neuron_state=False,
                    record_firing_rate=True, record_spikes=True, save_summary_only=True)
                launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

def post_wta_jobs(nodes, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range, num_trials,
                  single_inh_pop=False, muscimol_amount=0*nS, injection_site=0, start_nodes=True):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]

    for p_b_e in p_b_e_range:
        for p_x_e in p_x_e_range:
            for p_e_e in p_e_e_range:
                for p_e_i in p_e_i_range:
                    for p_i_i in p_i_i_range:
                        for p_i_e in p_i_e_range:
                            for i,contrast in enumerate(contrast_range):
                                inputs=np.zeros(2)
                                inputs[0]=(input_sum*(contrast+1.0)/2.0)
                                inputs[1]=input_sum-inputs[0]
                                for t in range(num_trials):
                                    np.random.shuffle(inputs)
                                    cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, trial_duration, p_b_e,
                                        p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, contrast, t, single_inh_pop=single_inh_pop,
                                        muscimol_amount=muscimol_amount, injection_site=injection_site, record_lfp=True,
                                        record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                                        record_spikes=True)
                                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

def post_broken_wta_jobs(nodes, num_trials, data_path, e_desc, single_inh_pop=False, muscimol_amount=0*nS,
                         injection_site=0, start_nodes=True):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]

    param_combos=get_tested_param_combos(data_path, num_groups, trial_duration, contrast_range, num_trials, e_desc)
    for (p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e) in param_combos:
        for t,contrast in enumerate(contrast_range):
            for t in range(num_trials):
                file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.%s.contrast.%0.4f.trial.%d.h5' %\
                          (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, e_desc, contrast, t)
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
                        p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, contrast, t, single_inh_pop=single_inh_pop,
                        muscimol_amount=muscimol_amount, injection_site=injection_site, record_lfp=True,
                        record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                        record_spikes=True)
                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

def post_missing_wta_jobs(summary_file_name, nodes, num_trials, data_path, e_desc, single_inh_pop=False, muscimol_amount=0*nS,
                          injection_site=0, start_nodes=True):

    summary_data=SummaryData()
    summary_data.read_from_file(summary_file_name)

    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()
    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
    for i,p_b_e in enumerate(summary_data.p_b_e_range):
        for j,p_x_e in enumerate(summary_data.p_x_e_range):
            for k,p_e_e in enumerate(summary_data.p_e_e_range):
                for l,p_e_i in enumerate(summary_data.p_e_i_range):
                    for m,p_i_i in enumerate(summary_data.p_i_i_range):
                        for n,p_i_e in enumerate(summary_data.p_i_e_range):
                            if not summary_data.auc[i,j,k,l,m,n]:
                                for contrast in contrast_range:
                                    for t in range(num_trials):
                                        file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.%s.contrast.%0.4f.trial.%d.h5' %\
                                                  (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, e_desc, contrast, t)
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
                                                p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, contrast, t, single_inh_pop=single_inh_pop,
                                                muscimol_amount=muscimol_amount, injection_site=injection_site, record_lfp=True,
                                                record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                                                record_spikes=True)
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

    summary_data=SummaryData()
    summary_data.read_from_file(summary_filename)

    bayes_analysis=run_bayesian_analysis(summary_data.auc, summary_data.bc_slope, summary_data.bc_intercept,
        summary_data.bc_r_sqr, summary_data.num_trials, summary_data.p_b_e_range, summary_data.p_e_e_range,
        summary_data.p_e_i_range, summary_data.p_i_e_range, summary_data.p_i_i_range, summary_data.p_x_e_range)

    samples=[]
    posterior=bayes_analysis.l1_pos_posterior
    while len(samples)<num_samples:
        p_b_e_posterior=np.sum(np.sum(np.sum(np.sum(np.sum(posterior,axis=1),axis=1),axis=1),axis=1),axis=1)
        i, p_b_e = sample(p_b_e_posterior, summary_data.p_b_e_range)

        p_x_e_posterior=np.sum(np.sum(np.sum(np.sum(posterior[i,:,:,:,:,:],axis=1),axis=1),axis=1),axis=1)
        j,p_x_e = sample(p_x_e_posterior, summary_data.p_x_e_range)

        p_e_e_posterior=np.sum(np.sum(np.sum(posterior[i,j,:,:,:,:],axis=1),axis=1),axis=1)
        k,p_e_e = sample(p_e_e_posterior, summary_data.p_e_e_range)

        p_e_i_posterior=np.sum(np.sum(posterior[i,j,k,:,:,:],axis=1),axis=1)
        l,p_e_i=sample(p_e_i_posterior, summary_data.p_e_i_range)

        p_i_i_posterior=np.sum(posterior[i,j,k,l,:,:],axis=1)
        m,p_i_i=sample(p_i_i_posterior, summary_data.p_i_i_range)

        p_i_e_posterior=posterior[i,j,k,l,m,:]
        n,p_i_e=sample(p_i_e_posterior, summary_data.p_i_e_range)

        file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.trial.0.h5' %\
                  (summary_data.num_groups, summary_data.trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e)
        file_name=os.path.join(data_path,file_desc)
        if not os.path.exists(file_name) and not (p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e) in samples:
            print('p_b_e=%0.3f, p_x_e=%0.3f, p_e_e=%0.3f, p_e_i=%0.3f, p_i_i=%0.3f, p_i_e=%0.3f' % (p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e))
            samples.append((p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e))
            if post_jobs:
                input_sum=40.0
                contrast_range=[float(x)*(1.0/(summary_data.num_trials-1)) for x in range(0,summary_data.num_trials)]
                for t,contrast in enumerate(contrast_range):
                    inputs=np.zeros(2)
                    inputs[0]=(input_sum*(contrast+1.0)/2.0)
                    inputs[1]=input_sum-inputs[0]
                    np.random.shuffle(inputs)
                    cmds,log_file_template,out_file=get_wta_cmds(summary_data.num_groups, inputs, summary_data.trial_duration, p_b_e,
                        p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, t, single_inh_pop=single_inh_pop, record_lfp=True,
                        record_voxel=True, record_neuron_state=False, record_firing_rate=True, record_spikes=True)
                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)
    return np.array(samples)
