import os
from brian.stdunits import nS, pA
import numpy as np
from brian.units import siemens
from ezrcluster.launcher import Launcher
from pysbi.config import SRC_DIR
from pysbi.wta.network import simulation_params, default_params

def get_wta_cmds(wta_params, inputs, sim_params, contrast, trial, record_lfp=True, record_voxel=False,
                 record_neuron_state=False, record_spikes=True, record_firing_rate=True, save_summary_only=True,
                 e_desc=''):
    cmds = ['python', '/tmp/pySBI/src/python/pysbi/wta/network.py']
    file_desc='wta.groups.%d.duration.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.p_dcs.%0.4f.i_dcs.%0.4f.%s.contrast.%0.4f.trial.%d' %\
              (wta_params.num_groups, sim_params.trial_duration, wta_params.p_e_e, wta_params.p_e_i, wta_params.p_i_i,
               wta_params.p_i_e, sim_params.p_dcs/pA, sim_params.i_dcs/pA, e_desc, contrast, trial)
    log_file_template='%s.log' % file_desc
    output_file='/tmp/wta-output/%s.h5' % file_desc
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

    return cmds, log_file_template, output_file

def post_wta_jobs(nodes, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range, num_trials,
                   muscimol_amount=0*nS, injection_site=0, start_nodes=True):
    sim_params=simulation_params()
    sim_params.muscimol_amount=muscimol_amount
    sim_params.injection_site=injection_site

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
                            wta_params=default_params()
                            wta_params.p_b_e=p_b_e
                            wta_params.p_x_e=p_x_e
                            wta_params.p_e_e=p_e_e
                            wta_params.p_e_i=p_e_i
                            wta_params.p_i_i=p_i_i
                            wta_params.p_i_e=p_i_e
                            for i,contrast in enumerate(contrast_range):
                                inputs=np.zeros(2)
                                inputs[0]=(input_sum*(contrast+1.0)/2.0)
                                inputs[1]=input_sum-inputs[0]
                                for t in range(num_trials):
                                    np.random.shuffle(inputs)
                                    cmds,log_file_template,out_file=get_wta_cmds(wta_params, inputs, sim_params,
                                        contrast, t, record_lfp=True, record_voxel=True, record_neuron_state=False,
                                        record_firing_rate=True, record_spikes=True)
                                    launcher.add_job(cmds, log_file_template=log_file_template, output_file=out_file)

