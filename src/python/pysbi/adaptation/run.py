import os
from brian import defaultclock
from brian.stdunits import ms
from brian.units import second
import matplotlib.pyplot as plt
import numpy as np
from pysbi.adaptation.popcode import default_params, ProbabilisticPopulationCode, run_pop_code, SamplingPopulationCode
from pysbi.util.utils import save_to_png, save_to_eps

def run_full_adaptation_simulation():
    N=150
    network_params=default_params
    network_params.tau_a=5*second
    trial_duration=10*second
    isi=6*second

    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+100*ms

    stim2_start_time=stim1_end_time+isi
    stim2_end_time=stim2_start_time+100*ms

    low_var=5
    high_var=15
    low_mean=50
    high_mean=100

    stim=[(low_mean,low_var),(low_mean,high_var),(high_mean,low_var),(high_mean,high_var)]
    prob_adaptation=np.zeros([4,4])
    samp_adaptation=np.zeros([4,4])

    for i,(stim1_mean,stim1_var) in enumerate(stim):
        for j,(stim2_mean,stim2_var) in enumerate(stim):
            print('stim1: mean=%d, var=%d; stim2: mean=%d, var=%d' % (stim1_mean,stim1_var,stim2_mean,stim2_var))
            pop_monitor,voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [stim1_mean,stim2_mean],
                [stim1_var,stim2_var], [stim1_start_time,stim2_start_time],[stim1_end_time,stim2_end_time],
                trial_duration)
            y_max=np.max(voxel_monitor['y'][0][60000:])

            pop_monitor,voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [stim2_mean],
                [stim2_var], [stim1_start_time],[stim1_end_time], trial_duration)
            baseline_y_max=np.max(voxel_monitor['y'][0])
            prob_adaptation[i,j]=(baseline_y_max-y_max)/baseline_y_max
            print('prob_adaptation=%.4f' % prob_adaptation[i,j])

            pop_monitor,voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [stim1_mean,stim2_mean],
                [stim1_var,stim2_var], [stim1_start_time,stim2_start_time],[stim1_end_time,stim2_end_time],
                trial_duration)
            y_max=np.max(voxel_monitor['y'][0][60000:])

            pop_monitor,voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [stim2_mean],
                [stim2_var], [stim1_start_time],[stim1_end_time], trial_duration)
            baseline_y_max=np.max(voxel_monitor['y'][0])
            samp_adaptation[i,j]=(baseline_y_max-y_max)/baseline_y_max
            print('samp adaptation=%.4f' % samp_adaptation[i,j])

    data_dir='../../data/full_adaptation'

    fig=plt.figure()
    plt.title('Probabilistic Population')
    plt.imshow(prob_adaptation,interpolation='none')
    plt.colorbar()
    fname='prob_pop.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population')
    plt.imshow(samp_adaptation,interpolation='none')
    plt.colorbar()
    fname='samp_pop.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)


def run_mean_adaptation_simulation():
    N=150
    network_params=default_params
    network_params.tau_a=5*second
    trial_duration=10*second
    isi=6*second

    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+100*ms

    stim2_start_time=stim1_end_time+isi
    stim2_end_time=stim2_start_time+100*ms
    
    low_var=5
    high_var=15
    
    x_delta_range=np.array(range(0,int(N/3),5))
    prob_low_var_adaptation=np.zeros(len(x_delta_range))
    prob_high_var_adaptation=np.zeros(len(x_delta_range))
    samp_low_var_adaptation=np.zeros(len(x_delta_range))
    samp_high_var_adaptation=np.zeros(len(x_delta_range))
    for i,x_delta in enumerate(x_delta_range):
        print('x_delta=%d' % x_delta)

        prob_low_var_pop_monitor,prob_low_var_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
            [int(N/3), int(N/3)+x_delta], [low_var,low_var], [stim1_start_time,stim2_start_time],
            [stim1_end_time,stim2_end_time], trial_duration)
        y_max=np.max(prob_low_var_voxel_monitor['y'][0][60000:])
        prob_low_var_baseline_pop_monitor,prob_low_var_baseline_voxel_monitor=run_pop_code(ProbabilisticPopulationCode,
            N, network_params, [int(N/3)+x_delta], [low_var], [stim1_start_time], [stim1_end_time], 2*second)
        baseline=np.max(prob_low_var_baseline_voxel_monitor['y'][0])
        prob_low_var_adaptation[i]=(baseline-y_max)/baseline

        prob_high_var_pop_monitor,prob_high_var_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
            [int(N/3), int(N/3)+x_delta], [high_var,high_var], [stim1_start_time,stim2_start_time],
            [stim1_end_time,stim2_end_time], trial_duration)
        y_max=np.max(prob_high_var_voxel_monitor['y'][0][60000:])
        prob_high_var_baseline_pop_monitor,prob_high_var_baseline_voxel_monitor=run_pop_code(ProbabilisticPopulationCode,
            N, network_params, [int(N/3)+x_delta], [high_var], [stim1_start_time], [stim1_end_time], 2*second)
        baseline=np.max(prob_high_var_baseline_voxel_monitor['y'][0])
        prob_high_var_adaptation[i]=(baseline-y_max)/baseline

        samp_low_var_pop_monitor,samp_low_var_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params,
            [int(N/3), int(N/3)+x_delta], [low_var,low_var], [stim1_start_time,stim2_start_time],
            [stim1_end_time,stim2_end_time], trial_duration)
        y_max=np.max(samp_low_var_voxel_monitor['y'][0][60000:])
        samp_low_var_baseline_pop_monitor,samp_low_var_baseline_voxel_monitor=run_pop_code(SamplingPopulationCode,
            N, network_params, [int(N/3)+x_delta], [low_var], [stim1_start_time], [stim1_end_time], 2*second)
        baseline=np.max(samp_low_var_baseline_voxel_monitor['y'][0])
        samp_low_var_adaptation[i]=(baseline-y_max)/baseline

        samp_high_var_pop_monitor,samp_high_var_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params,
            [int(N/3), int(N/3)+x_delta], [high_var,high_var], [stim1_start_time,stim2_start_time],
            [stim1_end_time,stim2_end_time], trial_duration)
        y_max=np.max(samp_high_var_voxel_monitor['y'][0][60000:])
        samp_high_var_baseline_pop_monitor,samp_high_var_baseline_voxel_monitor=run_pop_code(SamplingPopulationCode,
            N, network_params, [int(N/3)+x_delta], [high_var], [stim1_start_time], [stim1_end_time], 2*second)
        baseline=np.max(samp_high_var_baseline_voxel_monitor['y'][0])
        samp_high_var_adaptation[i]=(baseline-y_max)/baseline

    data_dir='../../data/mean_shift'

    fig=plt.figure()
    plt.title('Probabilistic Population Code')
    plt.plot(x_delta_range,prob_low_var_adaptation,'r',label='low var')
    plt.plot(x_delta_range,prob_high_var_adaptation,'b',label='high var')
    plt.legend(loc='best')
    fname='prob_pop.mean_adaptation.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population Code')
    plt.plot(x_delta_range,samp_low_var_adaptation,'r',label='low var')
    plt.plot(x_delta_range,samp_high_var_adaptation,'b',label='high var')
    plt.legend(loc='best')
    fname='samp_pop.mean_adaptation.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

def run_uncertainty_adaptation_simulation():

    N=150
    network_params=default_params
    network_params.tau_a=5*second
    trial_duration=10*second
    isi=6*second

    low_var=5
    high_var=15
    x=50

    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+100*ms

    stim2_start_time=stim1_end_time+isi
    stim2_end_time=stim2_start_time+100*ms

    print('Prob low->high')
    prob_low_high_pop_monitor,prob_low_high_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
        [x,x], [low_var,high_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time], trial_duration)
    prob_low_high_y_max=np.max(prob_low_high_voxel_monitor['y'][0][60000:])
    
    print('Prob high')
    prob_high_pop_monitor,prob_high_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x],
        [high_var], [stim1_start_time],[stim1_end_time],trial_duration)
    prob_high_y_max=np.max(prob_high_voxel_monitor['y'][0])
    
    print('Prob high->low')
    prob_high_low_pop_monitor,prob_high_low_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
        [x,x], [high_var,low_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time], trial_duration)
    prob_high_low_y_max=np.max(prob_high_low_voxel_monitor['y'][0][60000:])
    
    print('Prob low')
    prob_low_pop_monitor,prob_low_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x], [low_var],
        [stim1_start_time],[stim1_end_time],trial_duration)
    prob_low_y_max=np.max(prob_low_voxel_monitor['y'][0])
    
    print('Samp low->high')
    samp_low_high_pop_monitor,samp_low_high_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params,
        [x,x], [low_var,high_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time], trial_duration)
    samp_low_high_y_max=np.max(samp_low_high_voxel_monitor['y'][0][60000:])
    
    print('Samp high')
    samp_high_pop_monitor,samp_high_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [x], [high_var],
        [stim1_start_time],[stim1_end_time],trial_duration)
    samp_high_y_max=np.max(samp_high_voxel_monitor['y'][0])
    
    print('Samp high->low')
    samp_high_low_pop_monitor,samp_high_low_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params,
        [x,x], [high_var,low_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time], trial_duration)
    samp_high_low_y_max=np.max(samp_high_low_voxel_monitor['y'][0][60000:])
    
    print('Samp low')
    samp_low_pop_monitor,samp_low_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [x], [low_var],
        [stim1_start_time],[stim1_end_time],trial_duration)
    samp_low_y_max=np.max(samp_low_voxel_monitor['y'][0])
    
    data_dir='../../data/var_shift'

    fig=plt.figure()
    plt.plot([0,1],[(prob_high_y_max-prob_low_high_y_max)/prob_high_y_max,
                    (prob_low_y_max-prob_high_low_y_max)/prob_low_y_max],label='prob')
    plt.plot([0,1],[(samp_high_y_max-samp_low_high_y_max)/samp_high_y_max,
                    (samp_low_y_max-samp_high_low_y_max)/samp_low_y_max],label='samp')
    plt.legend(loc='best')
    fname='var.adaptation.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Probabilistic Population')
    plt.plot(prob_low_high_voxel_monitor['y'][0], 'b', label='low->high')
    plt.plot(prob_high_voxel_monitor['y'][0], 'b--', label='high only')
    plt.plot(prob_high_low_voxel_monitor['y'][0], 'r', label='high->low')
    plt.plot(prob_low_voxel_monitor['y'][0], 'r--', label='low only')
    plt.legend(loc='best')
    fname='var.adaptation.prob.bold.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)
    
    fig=plt.figure()
    plt.subplot(411)
    plt.title('Probabilistic Population - low->high')
    plt.imshow(prob_low_high_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    plt.subplot(412)
    plt.title('high')
    plt.imshow(prob_high_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    plt.subplot(413)
    plt.title('high->low')
    plt.imshow(prob_high_low_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    plt.subplot(414)
    plt.title('low')
    plt.imshow(prob_low_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    fname='var.adaptation.prob.e.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)
    
    fig=plt.figure()
    plt.title('Probabilistic Population')
    plt.plot(prob_low_high_pop_monitor['total_e'][0],'b',label='low->high')
    plt.plot(prob_high_pop_monitor['total_e'][0],'b-.',label='high')
    plt.plot(prob_high_low_pop_monitor['total_e'][0],'r',label='high->low')
    plt.plot(prob_low_pop_monitor['total_e'][0],'r-.',label='low')
    plt.legend(loc='best')
    fname='var.adaptation.prob.total_e.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Probabilistic Population')
    plt.plot(prob_low_high_pop_monitor['total_r'][0],'b',label='low->high')
    plt.plot(prob_high_pop_monitor['total_r'][0],'b-.',label='high')
    plt.plot(prob_high_low_pop_monitor['total_r'][0],'r',label='high->low')
    plt.plot(prob_low_pop_monitor['total_r'][0],'r-.',label='low')
    plt.legend(loc='best')
    fname='var.adaptation.prob.total_r.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)
    
    fig=plt.figure()
    plt.title('Probabilistic Population')
    plt.plot(prob_low_high_voxel_monitor['G_total'][0][0:100000],'b',label='low->high')
    plt.plot(prob_high_voxel_monitor['G_total'][0][0:100000],'b-.',label='high')
    plt.plot(prob_high_low_voxel_monitor['G_total'][0][0:100000],'r',label='high->low')
    plt.plot(prob_low_voxel_monitor['G_total'][0][0:100000],'r-.',label='low')
    plt.legend(loc='best')
    fname='var.adaptation.prob.g_total.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population')
    plt.plot(samp_low_high_voxel_monitor['y'][0], 'b', label='low->high')
    plt.plot(samp_high_voxel_monitor['y'][0], 'b-.', label='high only')
    plt.plot(samp_high_low_voxel_monitor['y'][0], 'r', label='high->low')
    plt.plot(samp_low_voxel_monitor['y'][0], 'r-.', label='low only')
    plt.legend(loc='best')
    fname='var.adaptation.samp.bold.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.subplot(411)
    plt.title('Sampling Population - low->high')
    plt.imshow(samp_low_high_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    plt.subplot(412)
    plt.title('high')
    plt.imshow(samp_high_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    plt.subplot(413)
    plt.title('high->low')
    plt.imshow(samp_high_low_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    plt.subplot(414)
    plt.title('low')
    plt.imshow(samp_low_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    fname='var.adaptation.samp.e.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population')
    plt.plot(samp_low_high_pop_monitor['total_e'][0],'b',label='low->high')
    plt.plot(samp_high_pop_monitor['total_e'][0],'b-.',label='high')
    plt.plot(samp_high_low_pop_monitor['total_e'][0],'r',label='high->low')
    plt.plot(samp_low_pop_monitor['total_e'][0],'r-.',label='low')
    plt.legend(loc='best')
    fname='var.adaptation.samp.total_e.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population')
    plt.plot(samp_low_high_pop_monitor['total_r'][0],'b',label='low->high')
    plt.plot(samp_high_pop_monitor['total_r'][0],'b-.',label='high')
    plt.plot(samp_high_low_pop_monitor['total_r'][0],'r',label='high->low')
    plt.plot(samp_low_pop_monitor['total_r'][0],'r-.',label='low')
    plt.legend(loc='best')
    fname='var.adaptation.samp.total_r.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population')
    plt.plot(samp_low_high_voxel_monitor['G_total'][0][0:100000],'b',label='low->high')
    plt.plot(samp_high_voxel_monitor['G_total'][0][0:100000],'b-.',label='high')
    plt.plot(samp_high_low_voxel_monitor['G_total'][0][0:100000],'r',label='high->low')
    plt.plot(samp_low_voxel_monitor['G_total'][0][0:100000],'r-.',label='low')
    plt.legend(loc='best')
    fname='var.adaptation.samp.g_total.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)


def run_isi_simulation():
    N=150
    network_params=default_params
    trial_duration=2*second
    var=10
    x1=50
    x2=100
    isi_times=range(25,750,25)
    adaptation=np.zeros(len(isi_times))
    for i,isi in enumerate(isi_times):
        print('Testing isi=%dms' % isi)
        stim1_start_time=1*second
        stim1_end_time=stim1_start_time+100*ms

        stim2_start_time=stim1_end_time+isi*ms
        stim2_end_time=stim2_start_time+100*ms

        same_pop_monitor,same_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x1,x1], [var,var],
            [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)
        diff_pop_monitor,diff_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x1,x2], [var,var],
            [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

        same_y_max=np.max(same_voxel_monitor['y'][0])
        diff_y_max=np.max(diff_voxel_monitor['y'][0])
        adaptation[i]=(diff_y_max-same_y_max)/diff_y_max*100.0

    data_dir='../../data/isi'
    fig=plt.figure()
    plt.plot(isi_times,adaptation)
    plt.xlabel('ISI (ms)')
    plt.ylabel('Adaptation')
    fname='adaptation.isi.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)


def test_simulation():
    N=150
    network_params=default_params
    network_params.tau_a=5*second
    trial_duration=10*second
    isi=6*second
    stim_duration=100*ms
    var=10
    x1=50
    x2=100
    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+stim_duration
    stim2_start_time=stim1_end_time+isi
    stim2_end_time=stim2_start_time+stim_duration

    same_pop_monitor,same_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x1,x1], [var,var],
        [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)
    diff_pop_monitor,diff_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x1,x2], [var,var],
        [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)
    single_pop_monitor,single_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x1], [var],
        [stim1_start_time], [stim1_end_time], trial_duration)

    data_dir='../../data/adaptation_test/'
    fig=plt.figure()
    plt.subplot(311)
    plt.title('Same')
    plt.imshow(same_pop_monitor['r'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.subplot(312)
    plt.title('Different')
    plt.imshow(diff_pop_monitor['r'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.subplot(313)
    plt.title('Single')
    plt.imshow(single_pop_monitor['r'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    fname='firing_rate.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.subplot(311)
    plt.title('Same')
    plt.imshow(same_pop_monitor['e'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,1)
    plt.subplot(312)
    plt.title('Different')
    plt.imshow(diff_pop_monitor['e'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,1)
    plt.subplot(313)
    plt.title('Single')
    plt.imshow(single_pop_monitor['e'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,1)
    fname='efficacy.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('BOLD')
    plt.plot(same_voxel_monitor['y'][0],label='same')
    plt.plot(diff_voxel_monitor['y'][0],label='different')
    plt.plot(single_voxel_monitor['y'][0],label='single')
    plt.legend(loc='best')
    fname='bold.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.plot(same_pop_monitor['total_e'][0],label='same')
    plt.plot(diff_pop_monitor['total_e'][0],label='different')
    plt.plot(single_pop_monitor['total_e'][0],label='single')
    plt.legend(loc='best')
    fname='total_e.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.plot(same_pop_monitor['total_r'][0],label='same')
    plt.plot(diff_pop_monitor['total_r'][0],label='different')
    plt.plot(single_pop_monitor['total_r'][0],label='single')
    plt.legend(loc='best')
    fname='total_r.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.plot(same_voxel_monitor['G_total'][0][0:100000],label='same')
    plt.plot(diff_voxel_monitor['G_total'][0][0:100000],label='different')
    plt.plot(single_voxel_monitor['G_total'][0][0:100000],label='single')
    plt.legend(loc='best')
    fname='g_total.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)


def demo(N, network_params, trial_duration, x1, x2, low_var, high_var, isi):
    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+100*ms

    stim2_start_time=stim1_end_time+isi
    stim2_end_time=stim2_start_time+100*ms

    low_var_prob_pop_monitor,low_var_prob_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
        [x1,x2],[low_var,low_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

    high_var_prob_pop_monitor,high_var_prob_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
        [x1,x2],[high_var,high_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],
        trial_duration)

    low_var_samp_pop_monitor,low_var_samp_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [x1,x2],
        [low_var,low_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

    high_var_samp_pop_monitor,high_var_samp_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [x1,x2],
        [high_var,high_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

    data_dir='../../data/demo'

    fig=plt.figure()
    plt.subplot(411)
    plt.title('Probabilistic population, low variance - rate')
    plt.imshow(low_var_prob_pop_monitor['r'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,.2)
    plt.subplot(412)
    plt.title('Probabilistic population, high variance - rate')
    plt.imshow(high_var_prob_pop_monitor['r'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,.2)
    plt.subplot(413)
    plt.title('Sampling population, low variance - rate')
    plt.imshow(low_var_samp_pop_monitor['r'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,.2)
    plt.subplot(414)
    plt.title('Sampling population, high variance - rate')
    plt.imshow(high_var_samp_pop_monitor['r'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,.2)
    fname='firing_rate.%s'
    save_to_png(fig, os.path.join(data_dir,fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir,fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.subplot(411)
    plt.title('Probabilistic population, low variance - efficacy')
    plt.imshow(low_var_prob_pop_monitor['e'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,1)
    plt.subplot(412)
    plt.title('Probabilistic population, high variance - efficacy')
    plt.imshow(high_var_prob_pop_monitor['e'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,1)
    plt.subplot(413)
    plt.title('Sampling population, low variance - efficacy')
    plt.imshow(low_var_samp_pop_monitor['e'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,1)
    plt.subplot(414)
    plt.title('Sampling population, high variance - efficacy')
    plt.imshow(high_var_samp_pop_monitor['e'][:],aspect='auto')
    plt.xlabel('time')
    plt.ylabel('neuron')
    plt.colorbar()
    plt.clim(0,1)
    fname='efficacy.%s'
    save_to_png(fig, os.path.join(data_dir, fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir, fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('BOLD')
    plt.plot(low_var_prob_voxel_monitor['y'][0],label='prob,low var')
    plt.plot(high_var_prob_voxel_monitor['y'][0],label='prob,high var')
    plt.plot(low_var_samp_voxel_monitor['y'][0],label='samp,low var')
    plt.plot(high_var_samp_voxel_monitor['y'][0],label='samp,high var')
    plt.legend(loc='best')
    fname='bold.%s'
    save_to_png(fig, os.path.join(data_dir, fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir, fname % 'eps'))
    plt.close(fig)

    stim1_mid_time=stim1_start_time+(stim1_end_time-stim1_start_time)/2
    idx1=int(stim1_mid_time/defaultclock.dt)
    stim2_mid_time=stim2_start_time+(stim2_end_time-stim2_start_time)/2
    idx2=int(stim2_mid_time/defaultclock.dt)

    fig=plt.figure()
    plt.title('Probabilistic Population snapshot')
    plt.plot(low_var_prob_pop_monitor['r'][:,idx1],'r', label='low var, stim 1')
    plt.plot(low_var_prob_pop_monitor['r'][:,idx2],'r--', label='low var stim 2')
    plt.plot(high_var_prob_pop_monitor['r'][:,idx1],'b', label='high var, stim 1')
    plt.plot(high_var_prob_pop_monitor['r'][:,idx2],'b--', label='high var stim 2')
    plt.legend(loc='best')
    fname='prob_pop.firing_rate.snapshot.%s'
    save_to_png(fig, os.path.join(data_dir, fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir, fname % 'eps'))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population snapshot')
    plt.plot(low_var_samp_pop_monitor['r'][:,idx1],'r', label='low var, stim 1')
    plt.plot(low_var_samp_pop_monitor['r'][:,idx2],'r--', label='low var, stim 2')
    plt.plot(high_var_samp_pop_monitor['r'][:,idx1],'b', label='high var, stim 1')
    plt.plot(high_var_samp_pop_monitor['r'][:,idx2],'b--', label='high var, stim 2')
    plt.legend(loc='best')
    fname='samp_pop.firing_rate.snapshot.%s'
    save_to_png(fig, os.path.join(data_dir, fname % 'png'))
    save_to_eps(fig, os.path.join(data_dir, fname % 'eps'))
    plt.close(fig)


if __name__=='__main__':
    #demo(150, default_params, 2.0*second, 50,75,5,15, 60*ms)
    #test_simulation()
    run_mean_adaptation_simulation()
    #run_uncertainty_adaptation_simulation()
    #run_isi_simulation()
    #run_full_adaptation_simulation()

