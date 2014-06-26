import os
from brian import pA, second
from ezrcluster.launcher import Launcher
from pysbi.config import SRC_DIR

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
    cmds.append(mat_file)
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
