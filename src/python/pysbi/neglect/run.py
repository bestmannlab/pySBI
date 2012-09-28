from math import exp
from time import time
from brian import ms, hertz, raster_plot, Hz, second, PoissonGroup, Network, reinit_default_clock, Parameters
from brian.globalprefs import set_global_preferences
from brian.stdunits import pF, nS, mV
import h5py
from matplotlib.pyplot import figure, subplot, legend, xlabel, ylabel, title
import numpy as np
from pysbi.neglect.monitor import BrainMonitor
from pysbi.neglect.network import BrainNetworkGroup
from pysbi.voxel import LFPSource, Voxel, get_bold_signal

set_global_preferences(useweave=True)

default_params=Parameters(
    # Neuron parameters
    C = 200 * pF,
    gL = 20 * nS,
    EL = -70 * mV,
    VT = -55 * mV,
    DeltaT = 3 * mV,
    # Magnesium concentration
    Mg = 1,
    # Synapse parameters
    E_ampa = 0 * mV,
    E_nmda = 0 * mV,
    E_gaba_a = -70 * mV,
    E_gaba_b = -95 * mV,
    tau_ampa = 2.5*ms,
    tau1_nmda = 10*ms,
    tau2_nmda = 100*ms,
    tau_gaba_a = 2.5*ms,
    tau1_gaba_b = 10*ms,
    tau2_gaba_b =100*ms,
    w_ampa_min=0.35*nS,
    w_ampa_max=1.0*nS,
    w_nmda_min=0.01*nS,
    w_nmda_max=0.6*nS,
    w_gaba_a_min=0.25*nS,
    w_gaba_a_max=1.2*nS,
    w_gaba_b_min=0.1*nS,
    w_gaba_b_max=1.0*nS,

    # Connection probabilities
    p_g_e=0.05,
    p_b_e=0.1,
    p_v_ec_vis=0.05,
    #p_ec_mem_ec_mem=0.007,
    #p_ec_mem_ec_mem=0.008,
    p_ec_mem_ec_mem=0.0075,
    p_ec_vis_ec_mem=0.005,
    p_ii_ec=0.015,
    #p_ec_vis_ei_vis=0.05,
    #p_ec_vis_ei_vis=0.06,
    p_ec_vis_ei_vis=0.055,
    #p_ec_mem_ei_mem=0.03,
    p_ec_mem_ei_mem=0.04,
    #p_ei_mem_ei_mem=0.007,
    #p_ei_mem_ei_mem=0.008,
    p_ei_mem_ei_mem=0.0075,
    p_ei_vis_ei_mem=0.005,
    p_ic_ei=0.015,
    p_ei_ii=0.01,
    p_ec_ii=0.0075,
    p_ec_ic=0.01)

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


def run_neglect(input_freq, trial_duration, net_params=default_params, output_file=None, record_lfp=True, record_voxel=True,
                record_neuron_state=False, record_spikes=True, record_pop_firing_rate=True, record_neuron_firing_rate=False,
                record_inputs=False, plot_output=False, mem_trial=False):

    start_time=time()

    # Init simulation parameters
    background_input_size=1000
    #background_rate=20*Hz
    #background_rate=30*Hz
    background_rate=25*Hz

    visual_input_size=1000
    visual_background_rate=10*Hz
    visual_stim_min_rate=20*Hz
    visual_stim_tau=0.5

    go_input_size=500
    go_rate=20*Hz
    #go_background_rate=1*Hz
    go_background_rate=0*Hz

    lip_size=6250

    stim_start_time=1.8*second
    stim_end_time=2*second
    #stim_start_time=.3*second
    #stim_end_time=.5*second

    go_start_time=3*second
    go_end_time=3.1*second
    #go_start_time=1.5*second
    #go_end_time=1.6*second

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

    print "Initialization time:", time() - start_time

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

    if output_file is not None:
        write_output(brain_network, output_file, record_firing_rate, record_neuron_state, record_spikes,
            record_voxel, record_lfp, record_inputs, stim_end_time, stim_start_time, trial_duration,
            left_voxel, right_voxel, brain_monitor)

    return brain_monitor

## Write monitor data to HDF5 file
#       background_input_size = number of background inputs
#       bacground rate = background firing rate
#       input_freq = input firing rates
#       network_group_size = number of neurons per input group
#       num_groups = number of input groups
#       single_inh_pop = single inhibitory population
#       output_file = filename to write to
#       record_firing_rate = write network firing rate data when true
#       record_neuron_stae = write neuron state data when true
#       record_spikes = write spike data when true
#       record_voxel = write voxel data when true
#       record_lfp = write LFP data when true
#       record_inputs = write input firing rates when true
#       stim_end_time = stimulation end time
#       stim_start_time = stimulation start time
#       task_input_size = number of neurons in each task input group
#       trial_duration = duration of the trial
#       voxel = voxel for network
#       wta_monitor = network monitor
#       wta_params = network parameters
def write_output(brain_network, background_input_size, background_rate, vc_size, vc_rates, trial_duration, stim_start_time,
                 stim_end_time, go_start_time, go_end_time, record_firing_rate, record_neuron_state, record_spikes,
                 record_voxel, record_lfp, record_inputs, output_file, left_voxel, right_voxel, brain_monitor):

    f = h5py.File(output_file, 'w')

    # Write basic parameters
    param_group=f.create_group('parameters')
    param_group.attrs['background_input_size'] = background_input_size
    param_group.attrs['background_rate'] = background_rate
    param_group.attrs['vc_size'] = vc_size
    param_group.attrs['vc_rates'] = vc_rates
    param_group.attrs['trial_duration'] = trial_duration
    param_group.attrs['stim_start_time'] = stim_start_time
    param_group.attrs['stim_end_time'] = stim_end_time
    param_group.attrs['go_start_time'] = go_start_time
    param_group.attrs['go_end_time'] = go_end_time
    param_group.attrs['lip_size'] = brain_network.lip_size
    param_group.attrs['C'] = brain_network.params.C
    param_group.attrs['gL'] = brain_network.params.gL
    param_group.attrs['EL'] = brain_network.params.EL
    param_group.attrs['VT'] = brain_network.params.VT
    param_group.attrs['Mg'] = brain_network.params.Mg
    param_group.attrs['DeltaT'] = brain_network.params.DeltaT
    param_group.attrs['E_ampa'] = brain_network.params.E_ampa
    param_group.attrs['E_nmda'] = brain_network.params.E_nmda
    param_group.attrs['E_gaba_a'] = brain_network.params.E_gaba_a
    param_group.attrs['tau_ampa'] = brain_network.params.tau_ampa
    param_group.attrs['tau1_nmda'] = brain_network.params.tau1_nmda
    param_group.attrs['tau2_nmda'] = brain_network.params.tau2_nmda
    param_group.attrs['tau_gaba_a'] = brain_network.params.tau_gaba_a
    param_group.attrs['tau1_gaba_b'] = brain_network.params.tau1_gaba_b
    param_group.attrs['tau2_gaba_b'] = brain_network.params.tau2_gaba_b
    param_group.attrs['w_ampa_min'] = brain_network.params.w_ampa_b
    param_group.attrs['w_ampa_max'] = brain_network.params.w_ampa_x
    param_group.attrs['w_nmda_min'] = brain_network.params.w_nmda
    param_group.attrs['w_nmda_max'] = brain_network.params.w_nmda
    param_group.attrs['w_gaba_a_min'] = brain_network.params.w_gaba_a
    param_group.attrs['w_gaba_a_max'] = brain_network.params.w_gaba_a
    param_group.attrs['w_gaba_b_min'] = brain_network.params.w_gaba_a
    param_group.attrs['w_gaba_b_max'] = brain_network.params.w_gaba_a
    param_group.attrs['p_g_e'] = brain_network.params.p_g_e
    param_group.attrs['p_b_e'] = brain_network.params.p_b_e
    param_group.attrs['p_v_ec_vis'] = brain_network.params.p_v_ec_vis
    param_group.attrs['p_ec_mem_ec_mem'] = brain_network.params.p_ec_mem_ec_mem
    param_group.attrs['p_ec_vis_ec_mem'] = brain_network.params.p_ec_vis_ec_mem
    param_group.attrs['p_ii_ec'] = brain_network.params.p_ii_ec
    param_group.attrs['p_ec_ei'] = brain_network.params.p_ec_ei
    param_group.attrs['p_ei_mem_ei_mem'] = brain_network.params.p_ei_mem_ei_mem
    param_group.attrs['p_ei_vis_ei_mem'] = brain_network.params.p_ei_vis_ei_mem
    param_group.attrs['p_ic_ei'] = brain_network.params.p_ic_ei
    param_group.attrs['p_ei_ii'] = brain_network.params.p_ei_ii
    param_group.attrs['p_ec_ii'] = brain_network.params.p_ec_ii
    param_group.attrs['p_ec_ic'] = brain_network.params.p_ec_ic


    # Write LFP data
    if record_lfp:
        lfp_group = f.create_group('lfp')
        lfp_group['left_lfp']=brain_monitor.left_lfp_monitor.values
        lfp_group['right_lfp']=brain_monitor.right_lfp_monitor.values

    # Write voxel data
    if record_voxel:
        voxel_group = f.create_group('voxel')
        f_vox.attrs['eta'] = voxel.eta
        f_vox.attrs['G_base'] = voxel.G_base
        f_vox.attrs['tau_f'] = voxel.tau_f
        f_vox.attrs['tau_s'] = voxel.tau_s
        f_vox.attrs['tau_o'] = voxel.tau_o
        f_vox.attrs['e_base'] = voxel.e_base
        f_vox.attrs['v_base'] = voxel.v_base
        f_vox.attrs['alpha'] = voxel.alpha
        f_vox.attrs['T_2E'] = voxel.params.T_2E
        f_vox.attrs['T_2I'] = voxel.params.T_2I
        f_vox.attrs['s_e_0'] = voxel.params.s_e_0
        f_vox.attrs['s_i_0'] = voxel.params.s_i_0
        f_vox.attrs['B0'] = voxel.params.B0
        f_vox.attrs['TE'] = voxel.params.TE
        f_vox.attrs['s_e'] = voxel.params.s_e
        f_vox.attrs['s_i'] = voxel.params.s_i
        f_vox.attrs['beta'] = voxel.params.beta
        f_vox.attrs['k2'] = voxel.k2
        f_vox.attrs['k3'] = voxel.k3

        f_vox_total=f_vox.create_group('total_syn')
        f_vox_total['G_total'] = wta_monitor.voxel_monitor['G_total'].values
        f_vox_total['s'] = wta_monitor.voxel_monitor['s'].values
        f_vox_total['f_in'] = wta_monitor.voxel_monitor['f_in'].values
        f_vox_total['v'] = wta_monitor.voxel_monitor['v'].values
        f_vox_total['q'] = wta_monitor.voxel_monitor['q'].values
        f_vox_total['y'] = wta_monitor.voxel_monitor['y'].values

        f_vox_exc=f_vox.create_group('exc_syn')
        f_vox_exc['G_total'] = wta_monitor.voxel_exc_monitor['G_total'].values
        f_vox_exc['s'] = wta_monitor.voxel_exc_monitor['s'].values
        f_vox_exc['f_in'] = wta_monitor.voxel_exc_monitor['f_in'].values
        f_vox_exc['v'] = wta_monitor.voxel_exc_monitor['v'].values
        f_vox_exc['q'] = wta_monitor.voxel_exc_monitor['q'].values
        f_vox_exc['y'] = wta_monitor.voxel_exc_monitor['y'].values

    # Write neuron state data
    if record_neuron_state:
        f_state = f.create_group('neuron_state')
        f_state['g_ampa_r'] = wta_monitor.network_monitor['g_ampa_r'].values
        f_state['g_ampa_x'] = wta_monitor.network_monitor['g_ampa_x'].values
        f_state['g_ampa_b'] = wta_monitor.network_monitor['g_ampa_b'].values
        f_state['g_nmda'] = wta_monitor.network_monitor['g_nmda'].values
        f_state['g_gaba_a'] = wta_monitor.network_monitor['g_gaba_a'].values
        f_state['g_gaba_b'] = wta_monitor.network_monitor['g_gaba_b'].values
        f_state['I_ampa_r'] = wta_monitor.network_monitor['I_ampa_r'].values
        f_state['I_ampa_x'] = wta_monitor.network_monitor['I_ampa_x'].values
        f_state['I_ampa_b'] = wta_monitor.network_monitor['I_ampa_b'].values
        f_state['I_nmda'] = wta_monitor.network_monitor['I_nmda'].values
        f_state['I_gaba_a'] = wta_monitor.network_monitor['I_gaba_a'].values
        f_state['I_gaba_b'] = wta_monitor.network_monitor['I_gaba_b'].values
        f_state['vm'] = wta_monitor.network_monitor['vm'].values
        f_state['record_idx'] = np.array(wta_monitor.record_idx)

    # Write network firing rate data
    if record_firing_rate:
        f_rates = f.create_group('firing_rates')
        e_rates = []
        for rate_monitor in wta_monitor.population_rate_monitors['excitatory']:
            e_rates.append(rate_monitor.smooth_rate(width=5 * ms, filter='gaussian'))
        f_rates['e_rates'] = np.array(e_rates)

        i_rates = []
        for rate_monitor in wta_monitor.population_rate_monitors['inhibitory']:
            i_rates.append(rate_monitor.smooth_rate(width=5 * ms, filter='gaussian'))
        f_rates['i_rates'] = np.array(i_rates)

    # Write input firing rate data
    if record_inputs:
        back_rate=f.create_group('background_rate')
        back_rate['firing_rate']=wta_monitor.background_rate_monitor.smooth_rate(width=5*ms,filter='gaussian')
        task_rates=f.create_group('task_rates')
        t_rates=[]
        for task_monitor in wta_monitor.task_rate_monitors:
            t_rates.append(task_monitor.smooth_rate(width=5*ms,filter='gaussian'))
        task_rates['firing_rates']=np.array(t_rates)

    # Write spike data
    if record_spikes:
        f_spikes = f.create_group('spikes')
        for idx, spike_monitor in enumerate(wta_monitor.spike_monitors['excitatory']):
            if len(spike_monitor.spikes):
                f_spikes['e.%d.spike_neurons' % idx] = np.array([s[0] for s in spike_monitor.spikes])
                f_spikes['e.%d.spike_times' % idx] = np.array([s[1] for s in spike_monitor.spikes])

        for idx, spike_monitor in enumerate(wta_monitor.spike_monitors['inhibitory']):
            if len(spike_monitor.spikes):
                f_spikes['i.%d.spike_neurons' % idx] = np.array([s[0] for s in spike_monitor.spikes])
                f_spikes['i.%d.spike_times' % idx] = np.array([s[1] for s in spike_monitor.spikes])
    f.close()