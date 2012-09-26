from math import exp
from time import time
from brian import ms, hertz, raster_plot, Hz, second, PoissonGroup, Network, reinit_default_clock
from brian.globalprefs import set_global_preferences
from matplotlib.pyplot import figure, subplot, legend, xlabel, ylabel, title
import numpy as np
from pysbi.neglect.monitor import BrainMonitor
from pysbi.neglect.network import BrainNetworkGroup
from pysbi.voxel import LFPSource, Voxel, get_bold_signal

set_global_preferences(useweave=True)

def test_neglect(net_params, input_level, trial_duration, output_base, record_lfp=True, record_voxel=True,
                 record_neuron_state=True, record_spikes=True, record_pop_firing_rate=True, record_neuron_firing_rate=False,
                 record_inputs=False, plot_output=False):
    instructed_output=None
    if not output_base is None:
        instructed_output='%s.instructed.h5' % output_base
    instructed_monitor=run_neglect(net_params, [input_level,0], trial_duration, output_file=instructed_output,
        record_lfp=record_lfp, record_voxel=record_voxel, record_neuron_state=record_neuron_state,
        record_spikes=record_spikes, record_pop_firing_rate=record_pop_firing_rate,
        record_neuron_firing_rate=record_neuron_firing_rate, record_inputs=record_inputs)

    free_choice_output=None
    if not output_base is None:
        free_choice_output='%s.free_choice.h5' % output_base
    free_choice_monitor=run_neglect(net_params, [input_level,input_level], trial_duration, output_file=free_choice_output,
        record_lfp=record_lfp, record_voxel=record_voxel, record_neuron_state=record_neuron_state,
        record_spikes=record_spikes, record_pop_firing_rate=record_pop_firing_rate,
        record_neuron_firing_rate=record_neuron_firing_rate, record_inputs=record_inputs,)

    if plot_output:
        if record_pop_firing_rate:
            figure()
            ax=subplot(221)
            ax.plot(instructed_monitor.population_rate_monitors['left_ec'].times/ms,
                instructed_monitor.population_rate_monitors['left_ec'].smooth_rate(width=5*ms)/hertz, label='left LIP EC')
            ax.plot(instructed_monitor.population_rate_monitors['left_ei'].times/ms,
                instructed_monitor.population_rate_monitors['left_ei'].smooth_rate(width=5*ms)/hertz, label='left LIP EI')
            ax.plot(instructed_monitor.population_rate_monitors['left_ic'].times/ms,
                instructed_monitor.population_rate_monitors['left_ic'].smooth_rate(width=5*ms)/hertz, label='left LIP IC')
            ax.plot(instructed_monitor.population_rate_monitors['left_ii'].times/ms,
                instructed_monitor.population_rate_monitors['left_ii'].smooth_rate(width=5*ms)/hertz, label='left LIP II')
            legend()
            xlabel('Time (ms)')
            ylabel('Population Firing Rate (Hz)')
            title('Instructed')

            ax=subplot(222)
            ax.plot(instructed_monitor.population_rate_monitors['right_ec'].times/ms,
                instructed_monitor.population_rate_monitors['right_ec'].smooth_rate(width=5*ms)/hertz, label='right LIP EC')
            ax.plot(instructed_monitor.population_rate_monitors['right_ei'].times/ms,
                instructed_monitor.population_rate_monitors['right_ei'].smooth_rate(width=5*ms)/hertz, label='right LIP EI')
            ax.plot(instructed_monitor.population_rate_monitors['right_ic'].times/ms,
                instructed_monitor.population_rate_monitors['right_ic'].smooth_rate(width=5*ms)/hertz, label='right LIP IC')
            ax.plot(instructed_monitor.population_rate_monitors['right_ii'].times/ms,
                instructed_monitor.population_rate_monitors['right_ii'].smooth_rate(width=5*ms)/hertz, label='right LIP II')
            legend()
            xlabel('Time (ms)')
            ylabel('Population Firing Rate (Hz)')
            title('Instructed')

            ax=subplot(223)
            ax.plot(free_choice_monitor.population_rate_monitors['left_ec'].times/ms,
                free_choice_monitor.population_rate_monitors['left_ec'].smooth_rate(width=5*ms)/hertz, label='left LIP EC')
            ax.plot(free_choice_monitor.population_rate_monitors['left_ei'].times/ms,
                free_choice_monitor.population_rate_monitors['left_ei'].smooth_rate(width=5*ms)/hertz, label='left LIP EI')
            ax.plot(free_choice_monitor.population_rate_monitors['left_ic'].times/ms,
                free_choice_monitor.population_rate_monitors['left_ic'].smooth_rate(width=5*ms)/hertz, label='left LIP IC')
            ax.plot(free_choice_monitor.population_rate_monitors['left_ii'].times/ms,
                free_choice_monitor.population_rate_monitors['left_ii'].smooth_rate(width=5*ms)/hertz, label='left LIP II')
            legend()
            xlabel('Time (ms)')
            ylabel('Population Firing Rate (Hz)')
            title('Free choice')

            ax=subplot(224)
            ax.plot(free_choice_monitor.population_rate_monitors['right_ec'].times/ms,
                free_choice_monitor.population_rate_monitors['right_ec'].smooth_rate(width=5*ms)/hertz, label='right LIP EC')
            ax.plot(free_choice_monitor.population_rate_monitors['right_ei'].times/ms,
                free_choice_monitor.population_rate_monitors['right_ei'].smooth_rate(width=5*ms)/hertz, label='right LIP EI')
            ax.plot(free_choice_monitor.population_rate_monitors['right_ic'].times/ms,
                free_choice_monitor.population_rate_monitors['right_ic'].smooth_rate(width=5*ms)/hertz, label='right LIP IC')
            ax.plot(free_choice_monitor.population_rate_monitors['right_ii'].times/ms,
                free_choice_monitor.population_rate_monitors['right_ii'].smooth_rate(width=5*ms)/hertz, label='right LIP II')
            legend()
            xlabel('Time (ms)')
            ylabel('Population Firing Rate (Hz)')
            title('Free Choice')

        if record_voxel:
            figure()
            ax=subplot(111)
            ax.plot(instructed_monitor.left_voxel_monitor['y'].times / ms, instructed_monitor.left_voxel_monitor['y'][0],
                label='instructed left LIP')
            ax.plot(instructed_monitor.right_voxel_monitor['G_total'].times / ms, instructed_monitor.right_voxel_monitor['y'][0],
                label='instructed right LIP')
            ax.plot(free_choice_monitor.left_voxel_monitor['G_total'].times / ms, free_choice_monitor.left_voxel_monitor['y'][0],
                label='free choice left LIP')
            ax.plot(free_choice_monitor.right_voxel_monitor['G_total'].times / ms, free_choice_monitor.right_voxel_monitor['y'][0],
                label='free choice right LIP')
            legend()
            xlabel('Time (ms)')
            ylabel('BOLD')

        # Spike raster plots
        if record_spikes:
            figure()
            title('Free Choice')
            subplot(811)
            raster_plot(*free_choice_monitor.spike_monitors['left_ec'],newfigure=False)
            ylabel('Left EC')
            subplot(812)
            raster_plot(*free_choice_monitor.spike_monitors['left_ei'],newfigure=False)
            ylabel('Left EI')
            subplot(813)
            raster_plot(*free_choice_monitor.spike_monitors['left_ic'],newfigure=False)
            ylabel('Left IC')
            subplot(814)
            raster_plot(*free_choice_monitor.spike_monitors['left_ii'],newfigure=False)
            ylabel('Left II')
            subplot(815)
            raster_plot(*free_choice_monitor.spike_monitors['right_ec'],newfigure=False)
            ylabel('Right EC')
            subplot(816)
            raster_plot(*free_choice_monitor.spike_monitors['right_ei'],newfigure=False)
            ylabel('Right EI')
            subplot(817)
            raster_plot(*free_choice_monitor.spike_monitors['right_ic'],newfigure=False)
            ylabel('Right IC')
            subplot(818)
            raster_plot(*free_choice_monitor.spike_monitors['right_ii'],newfigure=False)
            ylabel('Right II')

            figure()
            title('Instructed')
            subplot(811)
            raster_plot(*instructed_monitor.spike_monitors['left_ec'],newfigure=False)
            ylabel('Left EC')
            subplot(812)
            raster_plot(*instructed_monitor.spike_monitors['left_ei'],newfigure=False)
            ylabel('Left EI')
            subplot(813)
            raster_plot(*instructed_monitor.spike_monitors['left_ic'],newfigure=False)
            ylabel('Left IC')
            subplot(814)
            raster_plot(*instructed_monitor.spike_monitors['left_ii'],newfigure=False)
            ylabel('Left II')
            subplot(815)
            raster_plot(*instructed_monitor.spike_monitors['right_ec'],newfigure=False)
            ylabel('Right EC')
            subplot(816)
            raster_plot(*instructed_monitor.spike_monitors['right_ei'],newfigure=False)
            ylabel('Right EI')
            subplot(817)
            raster_plot(*instructed_monitor.spike_monitors['right_ic'],newfigure=False)
            ylabel('Right IC')
            subplot(818)
            raster_plot(*instructed_monitor.spike_monitors['right_ii'],newfigure=False)
            ylabel('Right II')


def run_neglect(net_params, input_freq, trial_duration, output_file=None, record_lfp=True, record_voxel=True,
                record_neuron_state=False, record_spikes=True, record_pop_firing_rate=True, record_neuron_firing_rate=False,
                record_inputs=False, plot_output=False, mem_trial=False):

    # Init simulation parameters
    background_input_size=2000
    background_rate=10*Hz

    visual_input_size=1000
    visual_background_rate=5*Hz
    visual_stim_min_rate=10*Hz
    visual_stim_tau=0.5

    go_input_size=500
    go_rate=0*Hz
    #go_rate=0*Hz
    #go_background_rate=1*Hz
    go_background_rate=0*Hz

    lip_size=8000

    stim_start_time=1.8*second
    stim_end_time=2*second

    go_start_time=3*second
    go_end_time=3.2*second

    # Create network inputs
    background_inputs=[PoissonGroup(background_input_size, rates=background_rate), PoissonGroup(background_input_size, rates=background_rate)]

    def make_mem_rate_function(rate):
        return lambda t: ((stim_start_time<t<stim_end_time and np.max([visual_background_rate,rate*exp(-(t-stim_start_time)/visual_stim_tau)])) or visual_background_rate)

    def make_delay_rate_function(rate):
        return lambda t: ((stim_start_time<t and np.max([visual_stim_min_rate,rate*exp(-(t-stim_start_time)/visual_stim_tau)])) or visual_background_rate)

    def make_go_rate_function():
        return lambda t: ((go_start_time<t<go_end_time and go_rate) or go_background_rate)

    lrate=input_freq[0]*Hz
    rrate=input_freq[1]*Hz

    if mem_trial:
        visual_cortex_inputs=[PoissonGroup(visual_input_size, rates=make_mem_rate_function(lrate)),
                              PoissonGroup(visual_input_size, rates=make_mem_rate_function(rrate))]
    else:
        visual_cortex_inputs=[PoissonGroup(visual_input_size, rates=make_delay_rate_function(lrate)),
                              PoissonGroup(visual_input_size, rates=make_delay_rate_function(rrate))]

    go_input=PoissonGroup(go_input_size, rates=make_go_rate_function())

    # Create WTA network
    brain_network=BrainNetworkGroup(lip_size, params=net_params, background_inputs=background_inputs,
        visual_cortex_input=visual_cortex_inputs, go_input=go_input)

    # LFP source
    left_lip_lfp_source=LFPSource(brain_network.left_lip.e_group)
    right_lip_lfp_source=LFPSource(brain_network.right_lip.e_group)

    # Create voxel
    left_lip_voxel=Voxel(network=brain_network.left_lip.neuron_group)
    right_lip_voxel=Voxel(network=brain_network.right_lip.neuron_group)

    # Create network monitor
    brain_monitor=BrainMonitor(background_inputs, visual_cortex_inputs, go_input, brain_network, left_lip_lfp_source,
        right_lip_lfp_source, left_lip_voxel, right_lip_voxel, record_lfp=record_lfp, record_voxel=record_voxel,
        record_neuron_state=record_neuron_state, record_spikes=record_spikes, record_pop_firing_rate=record_pop_firing_rate,
        record_neuron_firing_rates=record_neuron_firing_rate, record_inputs=record_inputs)

    # Create Brian network and reset clock
    net=Network(background_inputs, visual_cortex_inputs, go_input, brain_network, left_lip_lfp_source, right_lip_lfp_source,
        left_lip_voxel, right_lip_voxel, brain_network.connections, brain_monitor.monitors)
    reinit_default_clock()

    # Run simulation
    start_time = time()
    net.run(trial_duration, report='text')
    print "Simulation time:", time() - start_time

    # Compute BOLD signal
    if record_voxel:
        brain_monitor.left_voxel_exc_monitor=get_bold_signal(brain_monitor.left_voxel_monitor['G_total_exc'].values[0],
            left_lip_voxel.params, [500, 1500])
        brain_monitor.left_voxel_monitor=get_bold_signal(brain_monitor.left_voxel_monitor['G_total'].values[0],
            left_lip_voxel.params, [500, 1500])

        brain_monitor.right_voxel_exc_monitor=get_bold_signal(brain_monitor.right_voxel_monitor['G_total_exc'].values[0],
            right_lip_voxel.params, [500, 1500])
        brain_monitor.right_voxel_monitor=get_bold_signal(brain_monitor.right_voxel_monitor['G_total'].values[0],
            right_lip_voxel.params, [500, 1500])

    # Plot outputs
    if plot_output:
        brain_monitor.plot(trial_duration)

    return brain_monitor