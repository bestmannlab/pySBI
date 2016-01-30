from brian import pA
import os
import numpy as np
from brian.units import siemens
from ezrcluster.launcher import Launcher
import h5py
from pysbi.config import SRC_DIR
from pysbi.wta.network import default_params, simulation_params
from pysbi.wta.run import get_wta_cmds


def launch_virtual_subject_processes(nodes, mu_0, virtual_subj_ids, behavioral_param_file, trials, stim_conditions,
                                     start_nodes=True):
    """
    nodes = nodes to run simulation on
    data_dir = directory containing subject data
    num_real_subjects = number of real subjects
    num_virtual_subjects = number of virtual subjects to run
    behavioral_param_file = file containing subject fitted behavioral parameters
    start_nodes = whether or not to start nodes
    """

    # Setup launcher
    launcher=Launcher(nodes)

    wta_params=default_params()
    wta_params.mu_0=mu_0
    wta_params.p_a=wta_params.mu_0/100.0
    wta_params.p_b=wta_params.p_a

    # Get subject alpha and beta values
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])

    # For each virtual subject
    for virtual_subj_id in virtual_subj_ids:

        # Sample beta from subject distribution - don't use subjects with high alpha
        beta_hist,beta_bins=np.histogram(beta_vals[np.where(alpha_vals<.99)[0]], density=True)
        bin_width=beta_bins[1]-beta_bins[0]
        beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        beta=beta_bin+np.random.rand()*bin_width
        wta_params.background_freq=(beta-161.08)/-.17

        contrast_range=[0.0, .016, .032, .064, .096, .128, .256, .512]
        for i,contrast in enumerate(contrast_range):
            inputs=np.zeros(2)
            inputs[0]=wta_params.mu_0+wta_params.p_a*contrast*100.0
            inputs[1]=wta_params.mu_0-wta_params.p_b*contrast*100.0
            for t in range(trials):
                np.random.shuffle(inputs)
                for stim_condition,stim_values in stim_conditions.iteritems():
                    sim_params=simulation_params()
                    sim_params.p_dcs=stim_values[0]
                    sim_params.i_dcs=stim_values[1]
                    cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params, contrast, t,
                        record_lfp=True, record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                        record_spikes=True, save_summary_only=False,
                        e_desc='virtual_subject.%d.%s' % (virtual_subj_id,stim_condition))
                    launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)

    launcher.post_jobs()

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

def get_dcs_cmds(wta_params, inputs, sim_params, contrast,trial, record_lfp=True, record_voxel=False,
                 record_neuron_state=False, record_spikes=True, record_firing_rate=True, save_summary_only=True):
    cmds = ['nohup', 'python', 'pysbi/wta/network.py']
    e_desc=''
    if sim_params.muscimol_amount>0:
        e_desc+='lesioned'
    else:
        e_desc+='control'
    file_desc='wta.groups.%d.duration.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.p_dcs.%0.4f.i_dcs.%0.4f.%s.contrast.%0.4f.trial.%d' %\
              (wta_params.num_groups, sim_params.trial_duration, wta_params.p_e_e, wta_params.p_e_i, wta_params.p_i_i,
               wta_params.p_i_e, sim_params.p_dcs/pA, sim_params.i_dcs/pA, e_desc, contrast, trial)
    log_file='%s.log' % file_desc
    output_file='../../data/%s.h5' % file_desc
    cmds.append('--num_groups')
    cmds.append('%d' % wta_params.num_groups)
    cmds.append('--inputs')
    cmds.append(','.join([str(input) for input in inputs]))
    cmds.append('--background')
    cmds.append('%0.3f' % wta_params.background_freq)
    cmds.append('--trial_duration')
    cmds.append('%0.3f' % sim_params.trial_duration)
    cmds.append('--p_e_e')
    cmds.append('%0.3f' % wta_params.p_e_e)
    cmds.append('--p_e_i')
    cmds.append('%0.3f' % wta_params.p_e_i)
    cmds.append('--p_i_i')
    cmds.append('%0.3f' % wta_params.p_i_i)
    cmds.append('--p_i_e')
    cmds.append('%0.3f' % wta_params.p_i_e)
    cmds.append('--output_file')
    cmds.append(output_file)
    if sim_params.muscimol_amount>0:
        cmds.append('--muscimol_amount')
        cmds.append(str(sim_params.muscimol_amount/siemens))
        cmds.append('--injection_site')
        cmds.append('%d' % sim_params.injection_site)
    if sim_params.p_dcs>0 or sim_params.p_dcs<0:
        cmds.append('--p_dcs')
        cmds.append(str(sim_params.p_dcs/pA))
    if sim_params.i_dcs>0 or sim_params.i_dcs<0:
        cmds.append('--i_dcs')
        cmds.append(str(sim_params.i_dcs/pA))
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

    return cmds, log_file, output_file


