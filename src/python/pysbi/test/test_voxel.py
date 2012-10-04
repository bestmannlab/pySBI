from brian.clock import defaultclock, reinit_default_clock
from brian.equations import Equations
from brian.monitor import MultiStateMonitor
from brian.network import network_operation, Network
from brian.neurongroup import NeuronGroup
from brian.stdunits import nS, ms
from brian.units import second
import math
from matplotlib.pyplot import subplot, figure, ylabel, xlabel, plot, legend, title
from pysbi.voxel import Voxel, default_params
import numpy as np

def test_voxel():

    voxel_params=default_params
    voxel_params.e_base=.7
    voxel_params.tau_o=2.7*second
    voxel_params.tau_s=0.95
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

    t=voxel_monitor['y'].times/second
    max_balloon_bold=np.max(voxel_monitor['y'][0])
    delta = 5.05
    lagsize = int(round(delta/.0001))
    m_tau = 1
    #h_tau = 1.25
    n = 3   # default 3

    m_hrf = ((t/m_tau)**(n-1) * np.exp(-t/m_tau)) / (m_tau) * math.factorial(n-1)
    m_lagged_hrf=np.zeros(len(m_hrf))
    m_lagged_hrf[lagsize:]=m_hrf[:len(m_hrf)-lagsize]
    m_lagged_hrf*=max_balloon_bold

    #h_hrf = ((t/h_tau)**(n-1) * np.exp(-t/h_tau)) / (h_tau) * math.factorial(n-1)
    #h_lagged_hrf=np.zeros(len(h_hrf))
    #h_lagged_hrf[lagsize:]=h_hrf[:len(h_hrf)-lagsize]
    #h_lagged_hrf*=max_balloon_bold

    #voxel_params.tau_s=0.7
    #voxel_params.tau_s=0.9
    #voxel_params.tau_s=1.0

    new_voxel=Voxel(params=voxel_params)
    new_voxel.G_base=0.01*nS
    new_voxel_monitor = MultiStateMonitor(new_voxel, vars=['G_total','s','f_in','v','f_out','q','y'], record=True)

    @network_operation(when='start')
    def get_new_input():
        if 5*second < defaultclock.t < 5.05*second:
            new_voxel.G_total=1*nS
        else:
            new_voxel.G_total=new_voxel.G_base

    new_net=Network(new_voxel, get_new_input, new_voxel_monitor)
    reinit_default_clock()
    new_net.run(15*second)

    plot_results(voxel_monitor, m_lagged_hrf, new_voxel_monitor)


def plot_voxel_details(voxel_monitor, new_voxel_monitor):
    t=voxel_monitor['y'].times/second
    t_ms=t/ms

    figure()
    ax = subplot(611)
    ax.plot(t_ms, voxel_monitor['G_total'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['G_total'][0], label='new')
    legend()
    ylabel('G_total')

    ax = subplot(612)
    ax.plot(t_ms, voxel_monitor['s'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['s'][0], label='new')
    legend()
    ylabel('s')

    ax = subplot(613)
    ax.plot(t_ms, voxel_monitor['f_in'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['f_in'][0], label='new')
    legend()
    ylabel('f_in')

    ax = subplot(614)
    ax.plot(t_ms, voxel_monitor['f_out'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['f_out'][0], label='new')
    legend()
    ylabel('f_out')

    ax = subplot(615)
    ax.plot(t_ms, voxel_monitor['v'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['v'][0], label='new')
    legend()
    ylabel('v')

    ax = subplot(616)
    ax.plot(t_ms, voxel_monitor['q'][0], label='original')
    ax.plot(t_ms, new_voxel_monitor['q'][0], label='new')
    legend()
    ylabel('q')


def plot_results(voxel_monitor, m_hrf, new_voxel_monitor):#h_hrf):
    balloon_bold=voxel_monitor['y'][0]
    max_balloon_bold=np.max(balloon_bold)
    max_balloon_ind=np.where(balloon_bold==max_balloon_bold)[0]

    max_m_hrf=np.max(m_hrf)
    max_m_hrf_ind=np.where(m_hrf==max_m_hrf)[0]

    #max_h_hrf=np.max(h_hrf)
    #max_h_hrf_ind=np.where(h_hrf==max_h_hrf)[0]
    new_balloon_bold=new_voxel_monitor['y'][0]
    new_max_balloon_bold=np.max(new_balloon_bold)
    new_max_balloon_ind=np.where(new_balloon_bold==new_max_balloon_bold)[0]

    t=voxel_monitor['y'].times/second
    t_ms=t/ms

    figure()
    #ax=subplot(211)
    ax=subplot(111)
    ax.plot(t_ms, balloon_bold, label='balloon')
    ax.plot(t_ms,m_hrf, label='monkey gamma')
    ax.plot(t_ms, new_balloon_bold, label='new balloon')
    ax.plot([t_ms[max_balloon_ind], t_ms[max_balloon_ind]],[0,max_balloon_bold], label='balloon peak')
    ax.plot([t_ms[max_m_hrf_ind], t_ms[max_m_hrf_ind]],[0,max_m_hrf], label='monkey gamma peak')
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
    params=default_params
    test_voxel(params)