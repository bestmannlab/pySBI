import h5py
import numpy as np
import os
import scipy.io
from brian import pA, second
from ezrcluster.launcher import Launcher
from pysbi.config import SRC_DIR
from pysbi.wta.rl.analysis import FileInfo
from pysbi.wta.rl.fit import stim_order, LAT, NOSTIM1

def get_rerw_commands(mat_file, p_dcs, i_dcs, dcs_start_time, alpha, beta, background_freq, e_desc=''):
    cmds = ['python', '/tmp/pySBI/src/python/pysbi/wta/rl/network.py']
    file_desc='rl.%s' % e_desc
    log_file_template='%s.log' % file_desc
    output_file='/tmp/wta-output/%s.h5' % file_desc

    cmds.append('--stim_mat_file')
    cmds.append(os.path.join('/tmp/pySBI/data/rerw/subjects',mat_file))
    cmds.append('--output_file')
    cmds.append(output_file)
    if p_dcs>0 or p_dcs<0:
        cmds.append('--p_dcs')
        cmds.append(str(p_dcs/pA))
    if i_dcs>0 or i_dcs<0:
        cmds.append('--i_dcs')
        cmds.append(str(i_dcs/pA))
    cmds.append('--dcs_start_time')
    cmds.append('%0.3f' % dcs_start_time)
    if background_freq is not None:
        cmds.append('--background')
        cmds.append('%0.3f' % background_freq)
    else:
        cmds.append('--beta')
        cmds.append('%0.4f' % beta)
    cmds.append('--alpha')
    cmds.append('%0.4f' % alpha)

    return cmds, log_file_template, output_file

def launch_background_freq_processes(nodes, background_freq_range, trials, start_nodes=True):
    mat_file='/tmp/pySBI/data/rerw/subjects/value1_s1_t2.mat'

    launcher=Launcher(nodes)

    for background_freq in background_freq_range:
        for trial in range(trials):
            cmds, log_file_template, out_file=get_rerw_commands(mat_file, 0*pA, 0*pA, 0*second, 0.4, 5.0,
                background_freq, e_desc='background_freq.%.3f.trial.%d' % (background_freq,trial))
            launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()


def launch_missing_background_freq_processes(nodes, data_dir, background_freq_range, trials, start_nodes=True):
    mat_file='value1_s1_t2.mat'

    launcher=Launcher(nodes)
    for background_freq in background_freq_range:
        for trial in range(trials):
            cmds, log_file_template, out_file=get_rerw_commands(mat_file, 0*pA, 0*pA, 0*second, 0.4, 5.0,
                background_freq, e_desc='background_freq.%.3f.trial.%d' % (background_freq,trial))
            out_path,out_filename=os.path.split(out_file)
            if not os.path.exists(os.path.join(data_dir,out_filename)):
                launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()


def launch_baseline_virtual_subject_processes(nodes, data_dir, num_real_subjects, virtual_subj_ids, start_nodes=True):

    alpha_range=(0.0, 1.0)
    beta_range=(1.69,6.63)

    # Setup launcher
    launcher=Launcher(nodes)

    # For each virtual subject
    for virtual_subj_id in virtual_subj_ids:

        # Choose an actual subject
        stim_file_name=None
        control_file_name=None
        while True:
            i=np.random.choice(range(num_real_subjects))
            subj_id=i+1
            subj_stim_session_number=stim_order[i,LAT]
            stim_file_name='value%d_s%d_t2.mat' % (subj_id,subj_stim_session_number)
            subj_control_session_number=stim_order[i,NOSTIM1]
            control_file_name='value%d_s%d_t2.mat' % (subj_id,subj_control_session_number)
            if os.path.exists(os.path.join(data_dir,stim_file_name)) and\
               os.path.exists(os.path.join(data_dir,control_file_name)):
                break

        # Sample alpha from subject distribution
        alpha=alpha_range[0]+np.random.rand()*(alpha_range[1]-alpha_range[0])

        # Sample beta from subject distribution - don't use subjects with high alpha
        beta=beta_range[0]+np.random.rand()*(beta_range[1]-beta_range[0])

        cmds, log_file_template, out_file=get_rerw_commands(control_file_name, 0*pA, 0*pA, 0*second, alpha, beta, None,
            e_desc='baseline.virtual_subject.%d.control' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 4*pA, -2*pA, 0*second, alpha, beta, None,
            e_desc='baseline.virtual_subject.%d.anode' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -4*pA, 2*pA, 0*second, alpha, beta, None,
            e_desc='baseline.virtual_subject.%d.cathode' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 2*pA, -4*pA, 0*second, alpha, beta, None,
            e_desc='baseline.virtual_subject.%d.anode_control_1' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -2*pA, 4*pA, 0*second, alpha, beta, None,
            e_desc='baseline.virtual_subject.%d.cathode_control_1' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()


def launch_virtual_subject_processes(nodes, data_dir, num_real_subjects, virtual_subj_ids, behavioral_param_file,
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

    # Get subject alpha and beta values
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])

    # For each virtual subject
    for virtual_subj_id in virtual_subj_ids:

        # Choose an actual subject
        stim_file_name=None
        control_file_name=None
        while True:
            i=np.random.choice(range(num_real_subjects))
            subj_id=i+1
            subj_stim_session_number=stim_order[i,LAT]
            stim_file_name='value%d_s%d_t2.mat' % (subj_id,subj_stim_session_number)
            subj_control_session_number=stim_order[i,NOSTIM1]
            control_file_name='value%d_s%d_t2.mat' % (subj_id,subj_control_session_number)
            if os.path.exists(os.path.join(data_dir,stim_file_name)) and \
               os.path.exists(os.path.join(data_dir,control_file_name)):
                break


        # Sample alpha from subject distribution - don't use subjects with high alpha
        alpha_hist,alpha_bins=np.histogram(alpha_vals[np.where(alpha_vals<.99)[0]], density=True)
        bin_width=alpha_bins[1]-alpha_bins[0]
        alpha_bin=np.random.choice(alpha_bins[:-1], p=alpha_hist*bin_width)
        alpha=alpha_bin+np.random.rand()*bin_width

        # Sample beta from subject distribution - don't use subjects with high alpha
        beta_hist,beta_bins=np.histogram(beta_vals[np.where(alpha_vals<.99)[0]], density=True)
        bin_width=beta_bins[1]-beta_bins[0]
        beta_bin=np.random.choice(beta_bins[:-1], p=beta_hist*bin_width)
        beta=beta_bin+np.random.rand()*bin_width

        cmds, log_file_template, out_file=get_rerw_commands(control_file_name, 0*pA, 0*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.control' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 4*pA, -2*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.anode' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -4*pA, 2*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.cathode' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 2*pA, -4*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.anode_control_1' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -2*pA, 4*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.cathode_control_1' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()


def launch_extra_virtual_subject_processes(nodes, virtual_subj_data_dir, virtual_subj_ids, start_nodes=True):
    # Setup launcher
    launcher=Launcher(nodes)

    for virtual_subj_id in virtual_subj_ids:
        virtual_subj_data=FileInfo(os.path.join(virtual_subj_data_dir,'rl.virtual_subject.%d.anode.h5' % virtual_subj_id))
        alpha=virtual_subj_data.alpha
        beta=virtual_subj_data.beta
        stim_file_name=virtual_subj_data.mat_file

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 2*pA, 0*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.anode_control_2' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 2*pA, 4*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.anode_control_3' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 4*pA, 0*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.anode_control_4' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 4*pA, 2*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.anode_control_5' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -2*pA, 0*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.cathode_control_2' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -2*pA, -4*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.cathode_control_3' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -4*pA, 0*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.cathode_control_4' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, -4*pA, -2*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.cathode_control_5' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

def launch_extra_virtual_subject_processes2(nodes, virtual_subj_data_dir, virtual_subj_ids, start_nodes=True):
    # Setup launcher
    launcher=Launcher(nodes)

    for virtual_subj_id in virtual_subj_ids:
        virtual_subj_data=FileInfo(os.path.join(virtual_subj_data_dir,'rl.virtual_subject.%d.anode.h5' % virtual_subj_id))
        alpha=virtual_subj_data.alpha
        beta=virtual_subj_data.beta
        stim_file_name=virtual_subj_data.mat_file

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 0*pA, -2*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.anode_control_6' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, 0*pA, 2*pA, 0*second, alpha, beta, None,
            e_desc='virtual_subject.%d.cathode_control_6' % virtual_subj_id)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

if __name__=='__main__':
    launch_virtual_subject_processes({}, '/home/jbonaiuto/Projects/pySBI/data/rerw/subjects', 24, 25,
        '/home/jbonaiuto/Projects/pySBI/data/rerw/subjects/fitted_behavioral_params.h5', 0.02, 0.02, start_nodes=False)
