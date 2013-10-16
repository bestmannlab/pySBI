from brian.clock import defaultclock, reinit_default_clock
from brian.monitor import MultiStateMonitor
from brian.network import network_operation, Network
from brian.stdunits import nS, ms
from brian.units import second
from matplotlib.pyplot import subplot, figure, ylabel, xlabel, plot, legend, title
from pysbi.voxel import Voxel, default_params, zheng_params, ZhengVoxel, TestVoxel
import numpy as np

def test_voxel():

    voxel_params=default_params
#    voxel_params.e_base=.7
#    voxel_params.tau_o=2.7*second
#    voxel_params.tau_s=0.95
    voxel=Voxel(params=voxel_params)
    voxel.G_base=0.01*nS
    voxel_monitor = MultiStateMonitor(voxel, vars=['G_total','s','f_in','v','f_out','q','y'], record=True)

    @network_operation(when='start')
    def get_input():
        if 5*second < defaultclock.t < 5.05*second:
            voxel.G_total=1*nS
        else:
            voxel.G_total=voxel.G_base

    net=Network(voxel, get_input, voxel_monitor)
    reinit_default_clock()
    net.run(15*second)

    voxel_params=zheng_params
    #voxel_params.e_base=.7
    #voxel_params.tau_o=2.7*second
    #voxel_params.tau_s=0.95
    new_voxel=ZhengVoxel(params=voxel_params)
    new_voxel.G_base=0.01*nS
    new_voxel_monitor = MultiStateMonitor(new_voxel, vars=['G_total','s','f_in','v','f_out','q','y','g','cmr_o','o_e','cb'],
        record=True)

    @network_operation(when='start')
    def get_new_input():
        if 5*second < defaultclock.t < 5.05*second:
            new_voxel.G_total=1*nS
        else:
            new_voxel.G_total=new_voxel.G_base

    @network_operation()
    def compute_cmro():
        #print('phi=%.3f, f_in=%.3f, o_e=%.3f, o_elog=%.3f, c_ab=%.3f, cb=%.3f, cmr_o=%.3f, g=%.3f' %
        #      (new_voxel.phi, new_voxel.f_in, new_voxel.o_e, new_voxel.oe_log, new_voxel.c_ab, new_voxel.cb, new_voxel.cmr_o, new_voxel.g))
        #new_voxel.cmr_o=(new_voxel.cb-new_voxel.g*new_voxel.params.c_ab)/(new_voxel.params.cb_0-new_voxel.params.g_0*new_voxel.params.c_ab)
        new_voxel.oe_log=np.log(1.0-new_voxel.o_e/(1.0-new_voxel.g))

    new_net=Network(new_voxel, get_new_input, compute_cmro, new_voxel_monitor)
    reinit_default_clock()
    new_net.run(15*second)

    plot_results(voxel_monitor, new_voxel_monitor)


def plot_voxel_details(voxel_monitor, new_voxel_monitor):
    t=voxel_monitor['y'].times/second
    t_ms=t/ms

    figure()
    ax = subplot(5,2,1)
    ax.plot(t_ms, voxel_monitor['G_total'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['G_total'][0], label='new')
    legend()
    ylabel('G_total')

    ax = subplot(5,2,2)
    ax.plot(t_ms, voxel_monitor['s'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['s'][0], label='new')
    legend()
    ylabel('s')

    ax = subplot(5,2,3)
    ax.plot(t_ms, voxel_monitor['f_in'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['f_in'][0], label='new')
    legend()
    ylabel('f_in')

    ax = subplot(5,2,4)
    ax.plot(t_ms, voxel_monitor['f_out'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['f_out'][0], label='new')
    legend()
    ylabel('f_out')

    ax = subplot(5,2,5)
    ax.plot(t_ms, voxel_monitor['v'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['v'][0], label='new')
    legend()
    ylabel('v')

    ax = subplot(5,2,6)
    ax.plot(t_ms, voxel_monitor['q'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['q'][0], label='new')
    legend()
    ylabel('q')

    ax = subplot(5,2,7)
    ax.plot(t_ms, new_voxel_monitor['cb'][0], label='new')
    ylabel('cb')

    ax = subplot(5,2,8)
    ax.plot(t_ms, new_voxel_monitor['o_e'][0], label='new')
    ylabel('o_e')

    ax = subplot(5,2,9)
    ax.plot(t_ms, new_voxel_monitor['cmr_o'][0], label='new')
    ylabel('cmr_o')

    ax = subplot(5,2,10)
    ax.plot(t_ms, new_voxel_monitor['g'][0], label='new')
    ylabel('g')


def plot_results(voxel_monitor, new_voxel_monitor):#h_hrf):
    balloon_bold=voxel_monitor['y'][0]
    max_balloon_bold=np.max(balloon_bold)
    max_balloon_ind=np.where(balloon_bold==max_balloon_bold)[0]

    new_balloon_bold=new_voxel_monitor['y'][0]
    new_max_balloon_bold=np.max(new_balloon_bold)
    new_max_balloon_ind=np.where(new_balloon_bold==new_max_balloon_bold)[0]

    t=voxel_monitor['y'].times/second
    t_ms=t/ms

    figure()
    #ax=subplot(211)
    ax=subplot(111)
    ax.plot(t_ms, balloon_bold, label='balloon')
    ax.plot(t_ms, new_balloon_bold, label='new balloon')
    ax.plot([t_ms[max_balloon_ind], t_ms[max_balloon_ind]],[0,max_balloon_bold], label='balloon peak')
    ax.plot([t_ms[new_max_balloon_ind], t_ms[new_max_balloon_ind]],[0,new_max_balloon_bold], label='new balloon peak')
    xlabel('Time (ms)')
    ylabel('BOLD')
    legend()

    plot_voxel_details(voxel_monitor, new_voxel_monitor)
#    ax=subplot(212)
#    ax.plot(voxel_monitor['y'].times / ms, voxel_monitor['y'][0], label='balloon')
#    ax.plot(t / ms,h_hrf, label='human gamma')
#    ax.plot([t_ms[max_balloon_ind], t_ms[max_balloon_ind]],[0,max_balloon_bold], label='balloon peak')
#    ax.plot([t_ms[max_h_hrf_ind], t_ms[max_h_hrf_ind]],[0,max_h_hrf], label='human gamma peak')
#    xlabel('Time (ms)')
#    ylabel('BOLD')
#    legend()

if __name__=='__main__':
    test_voxel()