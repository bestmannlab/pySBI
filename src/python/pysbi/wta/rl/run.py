import numpy as np
import os
from ezrcluster.launcher import Launcher
from pysbi.config import SRC_DIR

def get_rerw_commands(mat_file, background_freq, p_b_e):
    cmds = ['python', '/tmp/pySBI/src/python/pysbi/wta/rl/network.py']

    file_desc='rl.p_b_e.%0.3f.background_freq.%d' % (p_b_e,background_freq)
    log_file_template='%s.log' % file_desc
    output_file='/tmp/wta-output/%s.h5' % file_desc
    cmds.append('--mat_file')
    cmds.append(mat_file)
    cmds.append('--background')
    cmds.append('%d' % background_freq)
    cmds.append('--p_b_e')
    cmds.append('%.3f' % p_b_e)
    return cmds, log_file_template, output_file

def post_noise_jobs(nodes, start_nodes=True):
    background_freq=5
    mat_file='../../data/rerw/subjects/value1_s1_t2.mat'

    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    p_b_e_range=0.01+np.array(range(10))*.01
    for p_b_e in p_b_e_range:
        cmds, log_file_template, out_file=get_rerw_commands(mat_file, background_freq, p_b_e)
        launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)


