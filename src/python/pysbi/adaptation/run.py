import h5py
import os
from brian import defaultclock, Parameters
from brian.stdunits import ms
from brian.units import second
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from pysbi.adaptation.popcode import default_params, ProbabilisticPopulationCode, run_pop_code, SamplingPopulationCode, Stimulus, run_restricted_pop_code
from pysbi.util.utils import save_to_png, save_to_eps

rapid_design_params=Parameters(
    trial_duration=2*second,
    isi=100*ms,
    stim_dur=300*ms
)

long_design_params=Parameters(
    trial_duration=10*second,
    isi=6*second,
    stim_dur=100*ms
)

def adaptation_simulation(design, baseline, pop_class, N, network_params, sim_params, stim1_mean, stim2_mean, stim1_var,
                          stim2_var, data_dir, edesc):
    # Compute stimulus start and end times
    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+sim_params.stim_dur

    stim2_start_time=stim1_end_time+sim_params.isi
    stim2_end_time=stim2_start_time+sim_params.stim_dur

    pop_monitor,voxel_monitor,y_max=run_pop_code(pop_class, N, network_params,
        [Stimulus(stim1_mean, stim1_var, stim1_start_time, stim1_end_time),
         Stimulus(stim2_mean, stim2_var, stim2_start_time, stim2_end_time)],
        sim_params.trial_duration)

    if baseline=='repeated':
        baseline_pop_monitor,baseline_voxel_monitor,baseline_y_max=run_pop_code(pop_class, N, network_params,
            [Stimulus(stim1_mean,stim1_var,stim1_start_time,stim1_end_time),
             Stimulus(stim1_mean,stim1_var,stim2_start_time,stim2_end_time)],
            sim_params.trial_duration)
        adaptation=(y_max-baseline_y_max)/baseline_y_max
    elif baseline=='single':
        baseline_pop_monitor,baseline_voxel_monitor,baseline_y_max=run_pop_code(pop_class, N, network_params,
            [Stimulus(stim2_mean, stim2_var, stim1_start_time, stim1_end_time)],
            rapid_design_params.trial_duration)
        adaptation=(y_max-baseline_y_max)/baseline_y_max

    fig=plt.figure()
    plt.plot(voxel_monitor['y'][0], 'b', label='test')
    plt.plot(baseline_voxel_monitor['y'][0], 'r', label='baseline')
    plt.legend(loc='best')
    fname='%s.baseline-%s.%s.%s.bold' % (design,baseline,edesc,pop_class.__name__)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

    fig=plt.figure()
    plt.subplot(211)
    plt.title('test')
    plt.imshow(pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    plt.subplot(212)
    plt.title('baseline')
    plt.imshow(baseline_pop_monitor['e'][:],aspect='auto')
    plt.clim(0,1)
    plt.colorbar()
    fname='%s.baseline-%s.%s.%s.e' % (design,baseline,edesc,pop_class.__name__)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

    fig=plt.figure()
    plt.plot(pop_monitor['total_e'][0],'b',label='test')
    plt.plot(baseline_pop_monitor['total_e'][0],'r',label='baseline')
    plt.legend(loc='best')
    fname='%s.baseline-%s.%s.%s.total_e' % (design,baseline,edesc,pop_class.__name__)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

    fig=plt.figure()
    plt.plot(pop_monitor['total_r'][0],'b',label='test')
    plt.plot(baseline_pop_monitor['total_r'][0],'r',label='baseline')
    plt.legend(loc='best')
    fname='%s.baseline-%s.%s.%s.total_r' % (design,baseline,edesc,pop_class.__name__)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

    fig=plt.figure()
    plt.plot(voxel_monitor['G_total'][0][0:100000],'b',label='test')
    plt.plot(baseline_voxel_monitor['G_total'][0][0:100000],'r',label='baseline')
    plt.legend(loc='best')
    fname='%s.baseline-%s.%s.%s.g_total' % (design,baseline,edesc,pop_class.__name__)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

    return adaptation

def run_repeated_test():
    N=150
    # Set trial duration, inter-stimulus-interval, and stimulus duration depending on design
    network_params=default_params
    sim_params=rapid_design_params

    # Compute stimulus start and end times
    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+sim_params.stim_dur

    stim2_start_time=stim1_end_time+sim_params.isi
    stim2_end_time=stim2_start_time+sim_params.stim_dur

    x_delta_iter=5
    # If baseline is single stimulus - need to test x_delta=0
    x_delta_range=np.array(range(0,int(N/3),x_delta_iter))

    # High and low variance examples
    low_var=5
    x=int(N/3)

    prob_combined_y_max=np.zeros(len(x_delta_range))
    prob_repeated_y_max=np.zeros(len(x_delta_range))
    samp_combined_y_max=np.zeros(len(x_delta_range))
    samp_repeated_y_max=np.zeros(len(x_delta_range))
    for i,x_delta in enumerate(x_delta_range):
        print('x_delta=%d' % x_delta)

        pop_monitor,voxel_monitor,prob_repeated_y_max[i]=run_pop_code(ProbabilisticPopulationCode, N, network_params,
            [Stimulus(x,low_var,stim1_start_time,stim1_end_time),
             Stimulus(x+x_delta,low_var,stim2_start_time,stim2_end_time)],
            sim_params.trial_duration)
        pop_monitor,voxel_monitor,prob_first_y_max=run_pop_code(ProbabilisticPopulationCode, N, network_params,
            [Stimulus(x, low_var, stim1_start_time, stim1_end_time)],
            sim_params.trial_duration)
        pop_monitor,voxel_monitor,prob_second_y_max=run_pop_code(ProbabilisticPopulationCode, N, network_params,
            [Stimulus(x+x_delta, low_var, stim1_start_time, stim1_end_time)],
            sim_params.trial_duration)
        prob_combined_y_max[i]=prob_first_y_max+prob_second_y_max

        pop_monitor,voxel_monitor,samp_repeated_y_max[i]=run_pop_code(SamplingPopulationCode, N, network_params,
            [Stimulus(x,low_var,stim1_start_time,stim1_end_time),
             Stimulus(x+x_delta,low_var,stim2_start_time,stim2_end_time)],
            sim_params.trial_duration)
        pop_monitor,voxel_monitor,samp_first_y_max=run_pop_code(SamplingPopulationCode, N, network_params,
            [Stimulus(x, low_var, stim1_start_time, stim1_end_time)],
            sim_params.trial_duration)
        pop_monitor,voxel_monitor,samp_second_y_max=run_pop_code(SamplingPopulationCode, N, network_params,
            [Stimulus(x+x_delta, low_var, stim1_start_time, stim1_end_time)],
            sim_params.trial_duration)
        samp_combined_y_max[i]=samp_first_y_max+samp_second_y_max

    data_dir='../../data/adaptation/repeated_test/'

    fig=plt.figure()
    plt.plot(x_delta_range,prob_combined_y_max-prob_repeated_y_max,'r',label='prob')
    plt.plot(x_delta_range,samp_combined_y_max-samp_repeated_y_max,'b',label='samp')
    plt.legend(loc='best')
    fname='repeated_test'
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)



def run_full_adaptation_simulation(design, baseline):
    N=150
    # Set trial duration, inter-stimulus-interval, and stimulus duration depending on design
    network_params=default_params
    sim_params=rapid_design_params
    if design=='long':
        network_params.tau_a=5*second
        sim_params=long_design_params

    data_dir='../../data/adaptation/full_adaptation/'

    # High and low mean and variance examples
    low_var=5
    high_var=15
    low_mean=50
    high_mean=100

    stim=[(low_mean,low_var),(low_mean,high_var),(high_mean,low_var),(high_mean,high_var)]
    prob_adaptation=np.zeros([4,4])
    samp_adaptation=np.zeros([4,4])

    for i,(stim1_mean,stim1_var) in enumerate(stim):
        for j,(stim2_mean,stim2_var) in enumerate(stim):
            edesc=''
            if stim1_mean==low_mean:
                edesc+='low_mean.'
            else:
                edesc+='high_mean.'
            if stim1_var==low_var:
                edesc+='low_var-'
            else:
                edesc+='high_var-'
            if stim2_mean==low_mean:
                edesc+='low_mean.'
            else:
                edesc+='high_mean.'
            if stim2_var==low_var:
                edesc+='low_var'
            else:
                edesc+='high_var'
            print('prob stim1: mean=%d, var=%d; stim2: mean=%d, var=%d' % (stim1_mean,stim1_var,stim2_mean,stim2_var))
            prob_adaptation[i,j]=adaptation_simulation(design, baseline, ProbabilisticPopulationCode, N, network_params,
                sim_params, stim1_mean, stim2_mean, stim1_var, stim2_var, os.path.join(data_dir,'prob',edesc),edesc)
            print('prob adaptation=%.4f' % prob_adaptation[i,j])

            print('samp stim1: mean=%d, var=%d; stim2: mean=%d, var=%d' % (stim1_mean,stim1_var,stim2_mean,stim2_var))
            samp_adaptation[i,j]=adaptation_simulation(design, baseline, SamplingPopulationCode, N, network_params,
                sim_params, stim1_mean, stim2_mean, stim1_var, stim2_var, os.path.join(data_dir,'samp',edesc),edesc)
            print('samp adaptation=%.4f' % samp_adaptation[i,j])


    fig=plt.figure()
    plt.title('Probabilistic Population')
    plt.imshow(prob_adaptation,interpolation='none')
    plt.colorbar()
    fname='%s.baseline-%s.prob_pop' % (design,baseline)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population')
    plt.imshow(samp_adaptation,interpolation='none')
    plt.colorbar()
    fname='%s.baseline-%s.samp_pop' % (design,baseline)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

def run_mean_adaptation_simulation(design, baseline):
    N=150
    # Set trial duration, inter-stimulus-interval, and stimulus duration depending on design
    network_params=default_params
    sim_params=rapid_design_params
    if design=='long':
        network_params.tau_a=5*second
        sim_params=long_design_params

    data_dir='../../data/adaptation/mean_shift/'

    x_delta_iter=10
    # If baseline is single stimulus - need to test x_delta=0
    x_delta_range=np.array(range(0,int(N/3),x_delta_iter))

    # High and low variance examples
    low_var=5
    high_var=15
    x=int(N/3)

    prob_low_var_adaptation=np.zeros(len(x_delta_range))
    prob_high_var_adaptation=np.zeros(len(x_delta_range))
    samp_low_var_adaptation=np.zeros(len(x_delta_range))
    samp_high_var_adaptation=np.zeros(len(x_delta_range))
    for i,x_delta in enumerate(x_delta_range):
        print('x_delta=%d' % x_delta)

        prob_low_var_adaptation[i]=adaptation_simulation(design, baseline, ProbabilisticPopulationCode, N, network_params,
            sim_params, x, x+x_delta, low_var, low_var, os.path.join(data_dir,'prob','low_var'),'low_var.xdelta-%d' % x_delta)
        prob_high_var_adaptation[i]=adaptation_simulation(design, baseline, ProbabilisticPopulationCode, N, network_params,
            sim_params, x, x+x_delta, high_var, high_var, os.path.join(data_dir,'prob','high_var'),'high_var.xdelta-%d' % x_delta)

        samp_low_var_adaptation[i]=adaptation_simulation(design, baseline, SamplingPopulationCode, N, network_params,
            sim_params, x, x+x_delta, low_var, low_var, os.path.join(data_dir,'samp','low_var'),'low_var.xdelta-%d' % x_delta)
        samp_high_var_adaptation[i]=adaptation_simulation(design, baseline, SamplingPopulationCode, N, network_params,
            sim_params, x, x+x_delta, high_var, high_var, os.path.join(data_dir,'samp','high_var'),'high_var.xdelta-%d' % x_delta)


    fig=plt.figure()
    plt.title('Probabilistic Population Code')
    plt.plot(x_delta_range,prob_low_var_adaptation,'r',label='low var')
    plt.plot(x_delta_range,prob_high_var_adaptation,'b',label='high var')
    plt.legend(loc='best')
    fname='%s.baseline-%s.prob_pop.mean_adaptation' % (design, baseline)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)

    fig=plt.figure()
    plt.title('Sampling Population Code')
    plt.plot(x_delta_range,samp_low_var_adaptation,'r',label='low var')
    plt.plot(x_delta_range,samp_high_var_adaptation,'b',label='high var')
    plt.legend(loc='best')
    fname='%s.baseline-%s.samp_pop.mean_adaptation' % (design, baseline)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)


def run_uncertainty_adaptation_simulation(design, baseline):

    N=150
    # Set trial duration, inter-stimulus-interval, and stimulus duration depending on design
    network_params=default_params
    sim_params=rapid_design_params
    if design=='long':
        network_params.tau_a=5*second
        sim_params=long_design_params

    # Variance and mean values used
    low_var=5
    high_var=15
    x=int(N/3)

    data_dir='../../data/adaptation/var_shift/'

    print('prob low->high')
    prob_low_high_adaptation=adaptation_simulation(design, baseline, ProbabilisticPopulationCode, N, network_params, sim_params,
        x, x, low_var, high_var, os.path.join(data_dir,'prob','low-high'), 'low_high')
    print('prob high->low')
    prob_high_low_adaptation=adaptation_simulation(design, baseline, ProbabilisticPopulationCode, N, network_params, sim_params,
        x, x, high_var, low_var, os.path.join(data_dir,'prob','high-low'), 'high_low')

    print('samp low->high')
    samp_low_high_adaptation=adaptation_simulation(design, baseline, SamplingPopulationCode, N, network_params, sim_params,
        x, x, low_var, high_var, os.path.join(data_dir,'samp','low-high'), 'low_high')
    print('samp high->low')
    samp_high_low_adaptation=adaptation_simulation(design, baseline, SamplingPopulationCode, N, network_params, sim_params,
        x, x, high_var, low_var, os.path.join(data_dir,'prob','high-low'), 'high_low')


    fig=plt.figure()
    plt.plot([0,1],[prob_low_high_adaptation, prob_high_low_adaptation],label='prob')
    plt.plot([0,1],[samp_low_high_adaptation, samp_high_low_adaptation],label='samp')
    plt.legend(loc='best')
    fname='%s.baseline-%s.var.adaptation' % (design,baseline)
    save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    plt.close(fig)


def run_isi_simulation():
    N=150
    network_params=default_params
    trial_duration=2*second
    var=10
    x1=50
    x2=100
    isi_times=range(25,750,25)
    stim_dur=100*ms
    adaptation=np.zeros(len(isi_times))
    for i,isi in enumerate(isi_times):
        print('Testing isi=%dms' % isi)
        stim1_start_time=1*second
        stim1_end_time=stim1_start_time+stim_dur

        stim2_start_time=stim1_end_time+isi*ms
        stim2_end_time=stim2_start_time+stim_dur

        same_pop_monitor,same_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x1,x1], [var,var],
            [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)
        diff_pop_monitor,diff_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params, [x1,x2], [var,var],
            [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

        same_y_max=np.max(same_voxel_monitor['y'][0])
        diff_y_max=np.max(diff_voxel_monitor['y'][0])
        adaptation[i]=(diff_y_max-same_y_max)/diff_y_max*100.0

    data_dir='../../data/adaptation/isi'
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
    #isi=6*second
    isi=100*ms
    #stim_duration=100*ms
    stim_duration=300*ms
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

    data_dir='../../data/adaptation/adaptation_test/rapid'
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


def demo(N, network_params, trial_duration, x1, x2, low_var, high_var, isi, stim_dur):
    stim1_start_time=1*second
    stim1_end_time=stim1_start_time+stim_dur

    stim2_start_time=stim1_end_time+isi
    stim2_end_time=stim2_start_time+stim_dur

    low_var_prob_pop_monitor,low_var_prob_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
        [x1,x2],[low_var,low_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

    high_var_prob_pop_monitor,high_var_prob_voxel_monitor=run_pop_code(ProbabilisticPopulationCode, N, network_params,
        [x1,x2],[high_var,high_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],
        trial_duration)

    low_var_samp_pop_monitor,low_var_samp_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [x1,x2],
        [low_var,low_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

    high_var_samp_pop_monitor,high_var_samp_voxel_monitor=run_pop_code(SamplingPopulationCode, N, network_params, [x1,x2],
        [high_var,high_var], [stim1_start_time,stim2_start_time], [stim1_end_time,stim2_end_time],trial_duration)

    data_dir='../../data/adaptation/demo'

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


def run_correlation_analysis(stim_mat_file, pop_class, output_file):
    mat=scipy.io.loadmat(stim_mat_file)
    trial_info=mat['trials_for_simulation']
    stim_duration=300*ms
    isi=100*ms
    iti=2.5*second

    network_params=default_params

    N=360
    low_var=5
    high_var=15

    trial_duration=2*trial_info.shape[0]*stim_duration+trial_info.shape[0]*isi+(trial_info.shape[0]-1)*iti

    stimuli=[]
    trial_start_time=0*second

    for i in range(trial_info.shape[0]):
        diff=trial_info[i,2]
        stim1_mean=N/2.0+diff/2.0
        stim2_mean=N/2.0-diff/2.0
        stim1_var=low_var
        if trial_info[i,0]==2:
            stim1_var=high_var
        stim2_var=low_var
        if trial_info[i,1]==2:
            stim2_var=high_var
        stim1_start=trial_start_time
        stim1_end=stim1_start+stim_duration
        stim2_start=stim1_end+isi
        stim2_end=stim2_start+stim_duration
        stim1=Stimulus(stim1_mean, stim1_var, stim1_start, stim1_end)
        stim2=Stimulus(stim2_mean, stim2_var, stim2_start, stim2_end)
        stimuli.extend([stim1,stim2])
        trial_start_time+=2*stim_duration+isi+iti

    voxel_monitor=run_restricted_pop_code(pop_class, N, network_params, stimuli, trial_duration, report='text')

    f = h5py.File(output_file, 'w')
    f['y'] = voxel_monitor['y'].values
    f.close()
#    fig=plt.figure()
#    plt.plot(prob_voxel_monitor['y'][0], 'b', label='probabilistic')
#    plt.plot(samp_voxel_monitor['y'][0], 'r', label='sharpening')
#    plt.legend(loc='best')
#    plt.show()
    #fname='%s.baseline-%s.%s.%s.bold' % (design,baseline,edesc,pop_class.__name__)
    #save_to_png(fig, os.path.join(data_dir,'%s.png' % fname))
    #save_to_eps(fig, os.path.join(data_dir,'%s.eps' % fname))
    #plt.close(fig)

if __name__=='__main__':
    #demo(150, default_params, 2.0*second, 50,75,5,15, 100*ms, 300*ms)
    #test_simulation()
    #run_mean_adaptation_simulation()
    #run_uncertainty_adaptation_simulation()
    #run_isi_simulation()
    #run_full_adaptation_simulation()
    run_correlation_analysis('/home/jbonaiuto/Projects/fmri_adaptation/trials_for_simulation.mat')

