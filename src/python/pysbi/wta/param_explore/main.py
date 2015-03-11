import logging
import os
from brian import second, Hz, ms, hertz
import numpy as np
from pypet import Environment, cartesian_product

import pysbi.config as config
from pysbi.wta.network import default_params, run_wta


def add_parameters(traj):
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    traj.f_add_parameter('wta_params.p_e_e', 0.01, comment='pyr->pyr connection probability')
    traj.f_add_parameter('wta_params.p_e_i', 0.01, comment='pyr->inh connection probability')
    traj.f_add_parameter('wta_params.p_i_i', 0.01, comment='inh->inh connection probability')
    traj.f_add_parameter('wta_params.p_i_e', 0.01, comment='inh->pyr connection probability')


def add_exploration(traj):
    """Explores different values of `I` and `tau_ref`."""

    print('Adding exploration of p_e_e, p_e_i, p_i_i, and p_i_e')

    param_vals=[0.01]
    param_vals.extend(np.arange(0.1,1.1,0.1).tolist())
    explore_dict = {'wta_params.p_e_e': param_vals,
                    'wta_params.p_e_i': param_vals,
                    'wta_params.p_i_i': param_vals,
                    'wta_params.p_i_e': param_vals}

    explore_dict = cartesian_product(explore_dict, ('wta_params.p_e_e', 'wta_params.p_e_i','wta_params.p_i_i',
                                                    'wta_params.p_i_e'))
    # The second argument, the tuple, specifies the order of the cartesian product,
    # The variable on the right most side changes fastest and defines the
    # 'inner for-loop' of the cartesian product

    traj.f_explore(explore_dict)


def run_sim(traj):
    wta_params=default_params()
    wta_params.p_e_e=traj.par.wta_params.p_e_e
    wta_params.p_e_i=traj.par.wta_params.p_e_i
    wta_params.p_i_i=traj.par.wta_params.p_i_i
    wta_params.p_i_e=traj.par.wta_params.p_i_e
    wta_monitor=run_wta(wta_params, 2, [30 ,50], 4*second, 910*Hz, record_lfp=False, record_voxel=False,
        record_neuron_state=False, record_spikes=False, plot_output=False)
    traj.f_add_result('wta.$', e0_times=wta_monitor.monitors['excitatory_rate_0'].times/ms,
        e0_rate=wta_monitor.monitors['excitatory_rate_0'].smooth_rate(width=5*ms)/hertz,
        e1_times=wta_monitor.monitors['excitatory_rate_1'].times/ms,
        e1_rate=wta_monitor.monitors['excitatory_rate_1'].smooth_rate(width=5*ms)/hertz,
        i_times=wta_monitor.monitors['inhibitory_rate'].times/ms,
        i_rate=wta_monitor.monitors['inhibitory_rate'].smooth_rate(width=5*ms)/hertz,
        comment='Contains time steps and firing rates for pyr1,pyr2 and inh populations')


def main():
    filename=os.path.join(config.DATA_DIR,'wta_param_explore.h5')
    env=Environment(trajectory='wta_param_explore',
        comment='Experiment to find suitable parameters for WTA network',
        add_time=False,
        log_folder='logs',
        log_level=logging.INFO,
        log_stdout=True,
        multiproc=True,
        ncores=10, #My laptop has 2 cores ;-)
        wrap_mode='QUEUE',
        filename=filename,
        overwrite_file=True,
        git_repository='/home/jbonaiuto/Projects/pySBI',
        sumatra_project='/home/jbonaiuto/Projects/pySBI'
    )

    traj = env.v_trajectory

    # Add parameters
    add_parameters(traj)

    # Let's explore
    add_exploration(traj)

    # Ad the postprocessing function
    #env.f_add_postprocessing(wta_post_proc)

    # Run the experiment
    env.f_run(run_sim)

    # Finally disable logging and close all log-files
    env.f_disable_logging()


if __name__ == '__main__':
    main()
