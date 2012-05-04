import os
from brian.units import second
from ezrcluster.launcher import Launcher
from pysbi.config import SRC_DIR

def get_wta_cmds(num_groups, input_pattern, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e,
                 record_voxel=False, record_neuron_state=False, record_spikes=False, record_firing_rate=True):
    cmds = ['python', '/tmp/pySBI/src/python/pysbi/wta.py']
    file_desc='wta.groups.%d.input.%s.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f' %\
              (num_groups, input_pattern, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e)
    log_file_template='%s.log' % file_desc
    output_file='/tmp/wta-output/%s.h5' % file_desc
    cmds.append('--num_groups')
    cmds.append('%d' % num_groups)
    cmds.append('--input_pattern')
    cmds.append(input_pattern)
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
    cmds.append('--output')
    cmds.append(output_file)
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
    
def post_wta_jobs(instances, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range):
    num_groups=3
    trial_duration=2*second
    launcher=Launcher(instances)
    launcher.set_application_script(os.path.join(SRC_DIR, 'sh/ezrcluster-application-script.sh'))
    launcher.start_instances()
    for p_b_e in p_b_e_range:
        for p_x_e in p_x_e_range:
            for p_e_e in p_e_e_range:
                for p_e_i in p_e_i_range:
                    for p_i_i in p_i_i_range:
                        for p_i_e in p_i_e_range:
                            cmds,log_file_template,out_file=get_wta_cmds(num_groups, 'low', trial_duration, p_b_e,
                                p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, record_voxel=True, record_neuron_state=False,
                                record_firing_rate=True, record_spikes=True)
                            launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)
                            
                            cmds,log_file_template,out_file=get_wta_cmds(num_groups, 'high', trial_duration, p_b_e,
                                p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, record_voxel=True, record_neuron_state=False,
                                record_firing_rate=True, record_spikes=True)
                            launcher.add_batch_job(cmds, log_file_template=log_file_template, output_file=out_file)
    launcher.post_jobs()