from brian import pA
import os
import numpy as np
from brian.units import second
from ezrcluster.launcher import Launcher
from pysbi.config import SRC_DIR
from pysbi.wta.run import get_wta_cmds

def post_wta_dcs_jobs(nodes, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, background_freq, trials, start_nodes=True,
                      p_dcs=0*pA, i_dcs=0*pA):
    num_groups=2
    trial_duration=4*second
    input_sum=20.0
    launcher=Launcher(nodes)
    if start_nodes:
        launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
        launcher.start_nodes()

    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
    for i,contrast in enumerate(contrast_range):
        inputs=np.zeros(2)
        inputs[0]=(input_sum*(contrast+1.0)/2.0)
        inputs[1]=input_sum-inputs[0]
        for t in trials:
            np.random.shuffle(inputs)
            cmds,log_file_template,out_file=get_wta_cmds(num_groups, inputs, background_freq, trial_duration, p_b_e,
                p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, contrast, t, p_dcs=p_dcs, i_dcs=i_dcs, record_lfp=True,
                record_voxel=True, record_neuron_state=False, record_firing_rate=True, record_spikes=True,
                save_summary_only=False)
            launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

