import os
from brian import pA
from ezrcluster.launcher import Launcher
import h5py
import numpy as np
from pysbi.config import SRC_DIR
from pysbi.wta.network import default_params, simulation_params
from pysbi.wta.run import get_wta_cmds


def launch_virtual_subject_processes(nodes, mu_0, virtual_subj_ids, behavioral_param_file, trials,
                                     stim_gains=[8,6,4,2,1,0.5,0.25], start_nodes=True):
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

    # Get subject alpha and beta values
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])

    # For each virtual subject
    for virtual_subj_id in virtual_subj_ids:

        wta_params=default_params()
        wta_params.mu_0=mu_0
        wta_params.p_a=mu_0/100.0
        wta_params.p_b=wta_params.p_a

        # Sample beta from subject distribution - don't use subjects with high alpha
        beta_hist,beta_bins=np.histogram(beta_vals[np.where(alpha_vals<.99)[0]], density=True)
        bin_width=beta_bins[1]-beta_bins[0]
        beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        beta=beta_bin+np.random.rand()*bin_width
        wta_params.background_freq=(beta-161.08)/-.17

        contrast_range=[0.0, .032, .064, .128, .256, .512]
        for i,contrast in enumerate(contrast_range):
            inputs=np.zeros(2)
            inputs[0]=wta_params.mu_0+wta_params.p_a*contrast*100.0
            inputs[1]=wta_params.mu_0-wta_params.p_b*contrast*100.0
            for t in range(trials):
                np.random.shuffle(inputs)

                for idx, stim_gain in enumerate(stim_gains):
                    sim_params=simulation_params()
                    sim_params.p_dcs=stim_gain*pA
                    sim_params.i_dcs=-.5*stim_gain*pA
                    cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params, contrast, t,
                        record_lfp=True, record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                        record_spikes=True, save_summary_only=False,
                        e_desc='virtual_subject.%d.anode' % virtual_subj_id)
                    launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)

                    sim_params=simulation_params()
                    sim_params.p_dcs=-stim_gain*pA
                    sim_params.i_dcs=.5*stim_gain*pA
                    cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params, contrast, t,
                        record_lfp=True, record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                        record_spikes=True, save_summary_only=False,
                        e_desc='virtual_subject.%d.cathode' % virtual_subj_id)
                    launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)

                    if idx==0:
                        sim_params=simulation_params()
                        cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params, contrast, t,
                            record_lfp=True, record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                            record_spikes=True, save_summary_only=False,
                            e_desc='virtual_subject.%d.control' % virtual_subj_id)
                        launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)

    launcher.post_jobs()

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()


def launch_control_virtual_subject_processes(nodes, mu_0, virtual_subj_ids, behavioral_param_file, trials,
                                             stim_gains=[8,6,4,2,1,0.5,0.25], start_nodes=True):
    """
    Launch stimulation intensity simulations with DCS applied only to pyramidal population
    nodes = nodes to run simulation on
    data_dir = directory containing subject data
    num_real_subjects = number of real subjects
    num_virtual_subjects = number of virtual subjects to run
    behavioral_param_file = file containing subject fitted behavioral parameters
    start_nodes = whether or not to start nodes
    """

    # Setup launcher
    launcher=Launcher(nodes)

    # Get subject alpha and beta values
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])

    # For each virtual subject
    for virtual_subj_id in virtual_subj_ids:

        wta_params=default_params()
        wta_params.mu_0=mu_0
        wta_params.p_a=mu_0/100.0
        wta_params.p_b=wta_params.p_a

        # Sample beta from subject distribution - don't use subjects with high alpha
        beta_hist,beta_bins=np.histogram(beta_vals[np.where(alpha_vals<.99)[0]], density=True)
        bin_width=beta_bins[1]-beta_bins[0]
        beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        beta=beta_bin+np.random.rand()*bin_width
        wta_params.background_freq=(beta-161.08)/-.17

        contrast_range=[0.0, .032, .064, .128, .256, .512]
        for i,contrast in enumerate(contrast_range):
            inputs=np.zeros(2)
            inputs[0]=wta_params.mu_0+wta_params.p_a*contrast*100.0
            inputs[1]=wta_params.mu_0-wta_params.p_b*contrast*100.0
            for t in range(trials):
                np.random.shuffle(inputs)

                for idx, stim_gain in enumerate(stim_gains):
                    sim_params=simulation_params()
                    sim_params.p_dcs=stim_gain*pA
                    cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params, contrast, t,
                        record_lfp=True, record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                        record_spikes=True, save_summary_only=False,
                        e_desc='virtual_subject.%d.anode' % virtual_subj_id)
                    launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)

                    sim_params=simulation_params()
                    sim_params.p_dcs=-stim_gain*pA
                    cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params, contrast, t,
                        record_lfp=True, record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                        record_spikes=True, save_summary_only=False,
                        e_desc='virtual_subject.%d.cathode' % virtual_subj_id)
                    launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)

                    if idx==0:
                        sim_params=simulation_params()
                        cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params, contrast, t,
                            record_lfp=True, record_voxel=True, record_neuron_state=False, record_firing_rate=True,
                            record_spikes=True, save_summary_only=False,
                            e_desc='virtual_subject.%d.control' % virtual_subj_id)
                        launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)

    launcher.post_jobs()

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()