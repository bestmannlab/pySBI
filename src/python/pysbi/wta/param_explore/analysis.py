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

    # idx_iterator=traj.f_find_idx(['wta_params.p_e_e','wta_params.p_e_i','wta_params.p_i_i','wta_params.p_i_e'],myfilter_function)
    # for idx in idx_iterator:
    #     traj.v_idx=idx
    #     p_e_e=traj.wta_params.p_e_e
    #     p_e_i=traj.wta_params.p_e_i
    #     p_i_i=traj.wta_params.p_i_i
    #     p_i_e=traj.wta_params.p_i_e
    #     e0_times=traj.results.wta.crun.e0_times
    #     e0_rate=traj.results.wta.crun.e0_rate
    #     e1_times=traj.results.wta.crun.e1_times
    #     e1_rate=traj.results.wta.crun.e1_rate
    #     i_times=traj.results.wta.crun.i_times
    #     i_rate=traj.results.wta.crun.i_rate
    #     fig=Figure()
    #     ax=fig.add_subplot(1,1,1)
    #     ax.plot(e0_times, e0_rate, color='b', label='e0')
    #     ax.plot(e1_times, e1_rate, color='g', label='e1')
    #     ax.plot(i_times, i_rate, color='r', label='i')
    #     ax.set_xlabel('Time (s)')
    #     ax.set_ylabel('Firing Rate (Hz)')
    #     ax.set_title('p_e_e=%.5f, p_e_i=%.5f, p_i_i=%.5f, p_i_e=%.5f' % (p_e_e,p_e_i,p_i_i,p_i_e))
    #     ax.legend(loc=0)

    for idx,run_name in enumerate(traj.f_get_run_names(sort=True)):
        if idx>0:
            traj.v_idx=idx
            p_e_e=traj.wta_params.p_e_e
            p_e_i=traj.wta_params.p_e_i
            p_i_i=traj.wta_params.p_i_i
            p_i_e=traj.wta_params.p_i_e
            print('(%.3f,%.3f,%.3f,%.3f)' % (p_e_e,p_e_i,p_i_i,p_i_e))

            e0_times=traj.results.wta.crun.e0_times
            e0_rate=traj.results.wta.crun.e0_rate
            e1_times=traj.results.wta.crun.e1_times
            e1_rate=traj.results.wta.crun.e1_rate
            i_times=traj.results.wta.crun.i_times
            i_rate=traj.results.wta.crun.i_rate

            pre_mean_e0_rate=np.mean(e0_rate[int((500*ms)/(.5*ms)):int((1000*ms)/(.5*ms))])
            pre_mean_e1_rate=np.mean(e1_rate[int((500*ms)/(.5*ms)):int((1000*ms)/(.5*ms))])

            mean_e0_rate=np.mean(e0_rate[int((2500*ms)/(.5*ms)):int((3000*ms)/(.5*ms))])
            mean_e1_rate=np.mean(e1_rate[int((2500*ms)/(.5*ms)):int((3000*ms)/(.5*ms))])
            mean_i_rate=np.mean(i_rate[int((2500*ms)/(.5*ms)):int((3000*ms)/(.5*ms))])
            if 100>mean_e1_rate>25 and mean_e0_rate<10 and np.abs(pre_mean_e0_rate-pre_mean_e1_rate)<5 and mean_i_rate>mean_e1_rate and mean_e1_rate:
                plt.figure()
                plt.plot(e0_times, e0_rate, color='b', label='e0')
                plt.plot(e1_times, e1_rate, color='g', label='e1')
                plt.plot(i_times, i_rate, color='r', label='i')
                plt.xlabel('Time (s)')
                plt.ylabel('Firing Rate (Hz)')
                plt.title('p_e_e=%.5f, p_e_i=%.5f, p_i_i=%.5f, p_i_e=%.5f' % (p_e_e,p_e_i,p_i_i,p_i_e))
                plt.legend(loc=0)
    plt.show()

myfilter_function = lambda p_e_e,p_e_i,p_i_i,p_i_e: p_e_e==0.1 and p_e_i==0.1 and p_i_i==0.1 and p_i_e==0.2

if __name__=='__main__':
    main()