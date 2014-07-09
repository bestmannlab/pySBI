import h5py
import numpy as np
import os
from brian import pA, second
from ezrcluster.launcher import Launcher
from pysbi.config import SRC_DIR
from pysbi.wta.rl.fit import stim_order, LAT, NOSTIM1

def get_rerw_commands(mat_file, p_b_e, p_x_e, p_dcs, i_dcs, dcs_start_time, alpha, beta, background_freq, e_desc=''):
    cmds = ['python', '/tmp/pySBI/src/python/pysbi/wta/rl/network.py']
    file_desc='rl.p_b_e.%0.3f.p_x_e.%0.3f.p_dcs.%0.4f.i_dcs.%0.4f.dcs_start_time.%0.3f.alpha.%0.3f.beta.%0.3f.%s' %\
          (p_b_e, p_x_e, p_dcs/pA, i_dcs/pA, dcs_start_time, alpha, beta, e_desc)
    if background_freq is not None:
        file_desc='rl.p_b_e.%0.3f.p_x_e.%0.3f.p_dcs.%0.4f.i_dcs.%0.4f.dcs_start_time.%0.3f.alpha.%0.3f.background.%0.3f.%s' %\
                  (p_b_e, p_x_e, p_dcs/pA, i_dcs/pA, dcs_start_time, alpha, background_freq, e_desc)
    log_file_template='%s.log' % file_desc
    output_file='/tmp/wta-output/%s.h5' % file_desc

    cmds.append('--stim_mat_file')
    cmds.append(os.path.join('/tmp/pySBI/data/rerw/subjects',mat_file))
    cmds.append('--p_b_e')
    cmds.append('%0.3f' % p_b_e)
    cmds.append('--p_x_e')
    cmds.append('%0.3f' % p_x_e)
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

def launch_background_freq_processes(nodes, background_freq_range, p_b_e, p_x_e, trials, start_nodes=True):
    mat_file='/tmp/pySBI/data/rerw/subjects/value1_s1_t2.mat'

    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    for background_freq in background_freq_range:
        for trial in range(trials):
            cmds, log_file_template, out_file=get_rerw_commands(mat_file, p_b_e, p_x_e, 0*pA, 0*pA, 0*second, 0.4, 5.0,
                background_freq, e_desc='trial.%d' % trial)
            launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

def launch_missing_background_freq_processes(nodes, data_dir, background_freq_range, p_b_e, p_x_e, trials, start_nodes=True):
    mat_file='value1_s1_t2.mat'

    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    for background_freq in background_freq_range:
        for trial in range(trials):
            cmds, log_file_template, out_file=get_rerw_commands(mat_file, p_b_e, p_x_e, 0*pA, 0*pA, 0*second, 0.4, 5.0,
                background_freq, e_desc='trial.%d' % trial)
            out_path,out_filename=os.path.split(out_file)
            if not os.path.exists(os.path.join(data_dir,out_filename)):
                launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)


def launch_virtual_subject_processes(nodes, data_dir, num_real_subjects, num_virtual_subjects, behavioral_param_file,
                                     p_b_e, p_x_e, start_nodes=True):
    """
    nodes = nodes to run simulation on
    data_dir = directory containing subject data
    num_real_subjects = number of real subjects
    num_virtual_subjects = number of virtual subjects to run
    behavioral_param_file = file containing subject fitted behavioral parameters
    p_b_e
    p_x_e
    start_nodes = whether or not to start nodes
    """

    # Setup launcher
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    # Get subject alpha and beta values
    f = h5py.File(behavioral_param_file)
    control_group=f['control']
    alpha_vals=np.array(control_group['alpha'])
    beta_vals=np.array(control_group['beta'])

    # For each virtual subject
    for j in range(num_virtual_subjects):

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

        cmds, log_file_template, out_file=get_rerw_commands(control_file_name, p_b_e, p_x_e, 0*pA, 0*pA, 0*second,
            alpha, beta, None, e_desc='virtual_subject.%d.control' % j)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, p_b_e, p_x_e, 4*pA, -2*pA, 0*second,
            alpha, beta, None, e_desc='virtual_subject.%d.anode' % j)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, p_b_e, p_x_e, -4*pA, 2*pA, 0*second,
            alpha, beta, None, e_desc='virtual_subject.%d.cathode' % j)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, p_b_e, p_x_e, 2*pA, -4*pA, 0*second,
            alpha, beta, None, e_desc='virtual_subject.%d.anode_control_1' % j)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

        cmds, log_file_template, out_file=get_rerw_commands(stim_file_name, p_b_e, p_x_e, -2*pA, 4*pA, 0*second,
            alpha, beta, None, e_desc='virtual_subject.%d.cathode_control_1' % j)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)


if __name__=='__main__':
    launch_virtual_subject_processes({}, '/home/jbonaiuto/Projects/pySBI/data/rerw/subjects', 24, 25,
        '/home/jbonaiuto/Projects/pySBI/data/rerw/subjects/fitted_behavioral_params.h5', 0.02, 0.02, start_nodes=False)