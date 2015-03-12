import os
from brian import ms
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pypet import Trajectory
from pysbi import config

def main():
    traj = Trajectory('wta_param_explore', add_time=False)

    # Let's load the trajectory from the file
    # Only load the parameters, we will load the results on the fly as we need them
    filename = os.path.join(config.DATA_DIR,'wta_param_explore.h5')
    traj.f_load(load_parameters=2, load_derived_parameters=0, load_results=0,
        load_other_data=0, filename=filename)

    # We'll simply use auto loading so all data will be loaded when needed.
    traj.v_auto_load = True

    for run_name in traj.f_get_run_names(sort=True):
        traj.f_as_run(run_name)
        p_e_e=traj.wta_params.p_e_e
        p_e_i=traj.wta_params.p_e_i
        p_i_i=traj.wta_params.p_i_i
        p_i_e=traj.wta_params.p_i_e

        results=traj.f_get_results()
        e0_times=results['wta.%s.e0_times' % run_name]
        e0_rate=results['wta.%s.e0_rate' % run_name]
        e1_times=results['wta.%s.e1_times' % run_name]
        e1_rate=results['wta.%s.e1_rate' % run_name]
        i_times=results['wta.%s.i_times' % run_name]
        i_rate=results['wta.%s.i_rate' % run_name]

        mean_e0_rate=np.mean(e0_rate[int((2500*ms)/(.5*ms)):int((3000*ms)/(.5*ms))])
        mean_e1_rate=np.mean(e1_rate[int((2500*ms)/(.5*ms)):int((3000*ms)/(.5*ms))])
        if mean_e1_rate>25 and mean_e0_rate<10:
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(e0_times, e0_rate, color='b', label='e0')
            ax.plot(e1_times, e1_rate, color='g', label='e1')
            ax.plot(i_times, i_rate, color='r', label='i')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_title('p_e_e=%.5f, p_e_i=%.5f, p_i_i=%.5f, p_i_e=%.5f' % (p_e_e,p_e_i,p_i_i,p_i_e))
            ax.legend(loc=0)
    plt.show()
