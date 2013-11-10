import h5py
import numpy as np
from brian import StateMonitor, MultiStateMonitor, PopulationRateMonitor, SpikeMonitor, raster_plot, ms, hertz, nS, nA, mA
from matplotlib.pyplot import figure, subplot, ylim, legend, ylabel, xlabel, show, title

# Collection of monitors for WTA network
class WTAMonitor():

    ## Constructor
    #       network = network to monitor
    #       lfp_source = LFP source to monitor
    #       voxel = voxel to monitor
    #       record_lfp = record LFP signals if true
    #       record_voxel = record voxel signals if true
    #       record_neuron_state = record neuron state signals if true
    #       record_spikes = record spikes if true
    #       record_firing_rate = record firing rate if true
    #       record_inputs = record inputs if true
    def __init__(self, network, lfp_source, voxel, record_lfp=True, record_voxel=True, record_neuron_state=False,
                 record_spikes=True, record_firing_rate=True, record_inputs=False, save_summary_only=False):
        self.num_groups=network.num_groups
        self.N=network.N
        self.monitors=[]
        self.save_summary_only=save_summary_only

        # LFP monitor
        if record_lfp:
            self.lfp_monitor = StateMonitor(lfp_source, 'LFP', record=0)
            self.monitors.append(self.lfp_monitor)
        else:
            self.lfp_monitor=None

        # Voxel monitor
        if record_voxel:
            self.voxel_monitor = MultiStateMonitor(voxel, vars=['G_total','G_total_exc','y'],
                record=True)
            self.monitors.append(self.voxel_monitor)
        else:
            self.voxel_monitor=None
        self.voxel_exc_monitor=None

        # Network monitor
        if record_neuron_state:
            self.record_idx=[]
            for i in range(self.num_groups):
                e_idx=i*int(.8*self.N/self.num_groups)
                self.record_idx.append(e_idx)
            i_idx=int(.8*self.N)
            self.record_idx.append(i_idx)
            self.network_monitor = MultiStateMonitor(network, vars=['vm','g_ampa_r','g_ampa_x','g_ampa_b','g_gaba_a',
                                                                    #'g_gaba_b','g_nmda','I_ampa_r','I_ampa_x',
                                                                    'g_nmda','I_ampa_r','I_ampa_x',
                                                                    #'I_ampa_b','I_gaba_a','I_gaba_b','I_nmda'],
                                                                    'I_ampa_b','I_gaba_a','I_nmda'],
                record=self.record_idx)
            self.monitors.append(self.network_monitor)
        else:
            self.network_monitor=None

        # Population rate monitors
        if record_firing_rate:
            self.population_rate_monitors={'excitatory':[], 'inhibitory':[]}
            for group_e in network.groups_e:
                e_rate_monitor=PopulationRateMonitor(group_e)
                self.population_rate_monitors['excitatory'].append(e_rate_monitor)
                self.monitors.append(e_rate_monitor)

            i_rate_monitor=PopulationRateMonitor(network.group_i)
            self.population_rate_monitors['inhibitory'].append(i_rate_monitor)
            self.monitors.append(i_rate_monitor)
        else:
            self.population_rate_monitors=None

        # Input rate monitors
        if record_inputs:
            self.background_rate_monitor=PopulationRateMonitor(network.background_input)
            self.monitors.append(self.background_rate_monitor)
            self.task_rate_monitors=[]
            for task_input in network.task_inputs:
                task_monitor=PopulationRateMonitor(task_input)
                self.task_rate_monitors.append(task_monitor)
                self.monitors.append(task_monitor)
        else:
            self.background_rate_monitor=None
            self.task_rate_monitors=None

        # Spike monitors
        if record_spikes:
            self.spike_monitors={'excitatory':[], 'inhibitory':[]}
            for group_e in network.groups_e:
                e_spike_monitor=SpikeMonitor(group_e)
                self.spike_monitors['excitatory'].append(e_spike_monitor)
                self.monitors.append(e_spike_monitor)

            i_spike_monitor=SpikeMonitor(network.group_i)
            self.spike_monitors['inhibitory'].append(i_spike_monitor)
            self.monitors.append(i_spike_monitor)
        else:
            self.spike_monitors=None

    # Plot monitor data
    def plot(self):

        # Spike raster plots
        if self.spike_monitors is not None:
            figure()
            subplot(211)
            raster_plot(*self.spike_monitors['excitatory'],newfigure=False)
            subplot(212)
            raster_plot(*self.spike_monitors['inhibitory'],newfigure=False)

        # Network firing rate plots
        if self.population_rate_monitors is not None:
            figure()
            ax=subplot(211)
            max_rates=[np.max(pop_rate_monitor.smooth_rate(width=5*ms)/hertz) for pop_rate_monitor in
                       self.population_rate_monitors['excitatory']]
            max_rates.extend([np.max(pop_rate_monitor.smooth_rate(width=5*ms)/hertz) for pop_rate_monitor in
                              self.population_rate_monitors['inhibitory']])
            max_rate=np.max(max_rates)

            for idx,pop_rate_monitor in enumerate(self.population_rate_monitors['excitatory']):
                ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms)/hertz, label='e %d' % idx)
                ylim(0,max_rate)
            legend()
            ylabel('Firing rate (Hz)')

            ax=subplot(212)
            for idx,pop_rate_monitor in enumerate(self.population_rate_monitors['inhibitory']):
                ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms)/hertz, label='i %d' % idx)
                ylim(0,max_rate)
            legend()
            ylabel('Firing rate (Hz)')
            xlabel('Time (ms)')

        # Input firing rate plots
        if self.background_rate_monitor is not None and self.task_rate_monitors is not None:
            figure()
            max_rate=np.max(self.background_rate_monitor.smooth_rate(width=5*ms)/hertz)
            for task_monitor in self.task_rate_monitors:
                max_rate=np.max(max_rate, np.max(task_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz))
            ax=subplot(111)
            ax.plot(self.background_rate_monitor.times/ms, self.background_rate_monitor.smooth_rate(width=5*ms)/hertz)
            ylim(0,max_rate)
            for task_monitor in self.task_rate_monitors:
                ax.plot(task_monitor.times/ms, task_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz)
                ylim(0,max_rate)

        # Network state plots
        if self.network_monitor is not None:
            max_conductances=[]
            for neuron_idx in self.record_idx:
                max_conductances.append(np.max(self.network_monitor['g_ampa_r'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_ampa_x'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_ampa_b'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_nmda'][neuron_idx]/nS))
                max_conductances.append(np.max(self.network_monitor['g_gaba_a'][neuron_idx]/nS))
                #max_conductances.append(np.max(self.network_monitor['g_gaba_b'][neuron_idx]/nS))
            max_conductance=np.max(max_conductances)

            fig=figure()
            for i in range(self.num_groups):
                neuron_idx=self.record_idx[i]
                ax=subplot(int('%d1%d' % (self.num_groups+1,i+1)))
                title('e%d' % i)
                ax.plot(self.network_monitor['g_ampa_r'].times/ms, self.network_monitor['g_ampa_r'][neuron_idx]/nS,
                    label='AMPA-recurrent')
                ax.plot(self.network_monitor['g_ampa_x'].times/ms, self.network_monitor['g_ampa_x'][neuron_idx]/nS,
                    label='AMPA-task')
                ax.plot(self.network_monitor['g_ampa_b'].times/ms, self.network_monitor['g_ampa_b'][neuron_idx]/nS,
                    label='AMPA-backgrnd')
                ax.plot(self.network_monitor['g_nmda'].times/ms, self.network_monitor['g_nmda'][neuron_idx]/nS,
                    label='NMDA')
                ax.plot(self.network_monitor['g_gaba_a'].times/ms, self.network_monitor['g_gaba_a'][neuron_idx]/nS,
                    label='GABA_A')
                #ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][neuron_idx]/nS,
                #    label='GABA_B')
                ylim(0,max_conductance)
                xlabel('Time (ms)')
                ylabel('Conductance (nS)')
                legend()

            neuron_idx=self.record_idx[self.num_groups]
            ax=subplot('%d1%d' % (self.num_groups+1,self.num_groups+1))
            title('i')
            ax.plot(self.network_monitor['g_ampa_r'].times/ms, self.network_monitor['g_ampa_r'][neuron_idx]/nS,
                label='AMPA-recurrent')
            ax.plot(self.network_monitor['g_ampa_x'].times/ms, self.network_monitor['g_ampa_x'][neuron_idx]/nS,
                label='AMPA-task')
            ax.plot(self.network_monitor['g_ampa_b'].times/ms, self.network_monitor['g_ampa_b'][neuron_idx]/nS,
                label='AMPA-backgrnd')
            ax.plot(self.network_monitor['g_nmda'].times/ms, self.network_monitor['g_nmda'][neuron_idx]/nS,
                label='NMDA')
            ax.plot(self.network_monitor['g_gaba_a'].times/ms, self.network_monitor['g_gaba_a'][neuron_idx]/nS,
                label='GABA_A')
            #ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][neuron_idx]/nS,
            #    label='GABA_B')
            ylim(0,max_conductance)
            xlabel('Time (ms)')
            ylabel('Conductance (nS)')
            legend()

            min_currents=[]
            max_currents=[]
            for neuron_idx in self.record_idx:
                max_currents.append(np.max(self.network_monitor['I_ampa_r'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_ampa_x'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_ampa_b'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_nmda'][neuron_idx]/nS))
                max_currents.append(np.max(self.network_monitor['I_gaba_a'][neuron_idx]/nS))
                #max_currents.append(np.max(self.network_monitor['I_gaba_b'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_r'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_x'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_ampa_b'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_nmda'][neuron_idx]/nS))
                min_currents.append(np.min(self.network_monitor['I_gaba_a'][neuron_idx]/nS))
                #min_currents.append(np.min(self.network_monitor['I_gaba_b'][neuron_idx]/nS))
            max_current=np.max(max_currents)
            min_current=np.min(min_currents)

            fig=figure()
            for i in range(self.num_groups):
                ax=subplot(int('%d1%d' % (self.num_groups+1,i+1)))
                neuron_idx=self.record_idx[i]
                title('e%d' % i)
                ax.plot(self.network_monitor['I_ampa_r'].times/ms, self.network_monitor['I_ampa_r'][neuron_idx]/nA,
                    label='AMPA-recurrent')
                ax.plot(self.network_monitor['I_ampa_x'].times/ms, self.network_monitor['I_ampa_x'][neuron_idx]/nA,
                    label='AMPA-task')
                ax.plot(self.network_monitor['I_ampa_b'].times/ms, self.network_monitor['I_ampa_b'][neuron_idx]/nA,
                    label='AMPA-backgrnd')
                ax.plot(self.network_monitor['I_nmda'].times/ms, self.network_monitor['I_nmda'][neuron_idx]/nA,
                    label='NMDA')
                ax.plot(self.network_monitor['I_gaba_a'].times/ms, self.network_monitor['I_gaba_a'][neuron_idx]/nA,
                    label='GABA_A')
                #ax.plot(self.network_monitor['I_gaba_b'].times/ms, self.network_monitor['I_gaba_b'][neuron_idx]/nA,
                #    label='GABA_B')
                ylim(min_current,max_current)
                xlabel('Time (ms)')
                ylabel('Current (nA)')
                legend()

            ax=subplot(int('%d1%d' % (self.num_groups+1,self.num_groups+1)))
            neuron_idx=self.record_idx[self.num_groups]
            title('i')
            ax.plot(self.network_monitor['I_ampa_r'].times/ms, self.network_monitor['I_ampa_r'][neuron_idx]/nA,
                label='AMPA-recurrent')
            ax.plot(self.network_monitor['I_ampa_x'].times/ms, self.network_monitor['I_ampa_x'][neuron_idx]/nA,
                label='AMPA-task')
            ax.plot(self.network_monitor['I_ampa_b'].times/ms, self.network_monitor['I_ampa_b'][neuron_idx]/nA,
                label='AMPA-backgrnd')
            ax.plot(self.network_monitor['I_nmda'].times/ms, self.network_monitor['I_nmda'][neuron_idx]/nA,
                label='NMDA')
            ax.plot(self.network_monitor['I_gaba_a'].times/ms, self.network_monitor['I_gaba_a'][neuron_idx]/nA,
                label='GABA_A')
            #ax.plot(self.network_monitor['I_gaba_b'].times/ms, self.network_monitor['I_gaba_b'][neuron_idx]/nA,
            #    label='GABA_B')
            ylim(min_current,max_current)
            xlabel('Time (ms)')
            ylabel('Current (nA)')
            legend()

        # LFP plot
        if self.lfp_monitor is not None:
            figure()
            ax=subplot(111)
            ax.plot(self.lfp_monitor.times / ms, self.lfp_monitor[0] / mA)
            xlabel('Time (ms)')
            ylabel('LFP (mA)')

        # Voxel activity plots
        if self.voxel_monitor is not None:
            syn_max=np.max(self.voxel_monitor['G_total'][0] / nS)
            y_max=np.max(self.voxel_monitor['y'][0])
            y_min=np.min(self.voxel_monitor['y'][0])
            figure()
            if self.voxel_exc_monitor is None:
                ax=subplot(211)
            else:
                ax=subplot(221)
                syn_max=np.max([syn_max, np.max(self.voxel_exc_monitor['G_total'][0])])
                y_max=np.max([y_max, np.max(self.voxel_exc_monitor['y'][0])])
                y_min=np.min([y_min, np.min(self.voxel_exc_monitor['y'][0])])
            ax.plot(self.voxel_monitor['G_total'].times / ms, self.voxel_monitor['G_total'][0] / nS)
            xlabel('Time (ms)')
            ylabel('Total Synaptic Activity (nS)')
            ylim(0, syn_max)
            if self.voxel_exc_monitor is None:
                ax=subplot(212)
            else:
                ax=subplot(222)
            ax.plot(self.voxel_monitor['y'].times / ms, self.voxel_monitor['y'][0])
            xlabel('Time (ms)')
            ylabel('BOLD')
            ylim(y_min, y_max)
            if self.voxel_exc_monitor is not None:
                ax=subplot(223)
                ax.plot(self.voxel_exc_monitor['G_total'].times / ms, self.voxel_exc_monitor['G_total'][0] / nS)
                xlabel('Time (ms)')
                ylabel('Total Synaptic Activity (nS)')
                ylim(0, syn_max)
                ax=subplot(224)
                ax.plot(self.voxel_exc_monitor['y'].times / ms, self.voxel_exc_monitor['y'][0])
                xlabel('Time (ms)')
                ylabel('BOLD')
                ylim(y_min, y_max)
        show()


## Write monitor data to HDF5 file
#       background_input_size = number of background inputs
#       background_freq rate = background firing rate
#       input_freq = input firing rates
#       network_group_size = number of neurons per input group
#       num_groups = number of input groups
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
def write_output(background_input_size, background_freq, input_freq, network_group_size, num_groups, output_file,
                 record_firing_rate, record_neuron_state, record_spikes, record_voxel, record_lfp, record_inputs,
                 stim_end_time, stim_start_time, task_input_size, trial_duration, voxel, wta_monitor, wta_params,
                 muscimol_amount, injection_site, p_dcs, i_dcs):

    f = h5py.File(output_file, 'w')

    # Write basic parameters
    f.attrs['num_groups'] = num_groups
    f.attrs['input_freq'] = input_freq
    f.attrs['trial_duration'] = trial_duration
    f.attrs['background_freq'] = background_freq
    f.attrs['stim_start_time'] = stim_start_time
    f.attrs['stim_end_time'] = stim_end_time
    f.attrs['network_group_size'] = network_group_size
    f.attrs['background_input_size'] = background_input_size
    f.attrs['task_input_size'] = task_input_size
    f.attrs['C'] = wta_params.C
    f.attrs['gL'] = wta_params.gL
    f.attrs['EL'] = wta_params.EL
    f.attrs['VT'] = wta_params.VT
    f.attrs['Mg'] = wta_params.Mg
    f.attrs['DeltaT'] = wta_params.DeltaT
    f.attrs['E_ampa'] = wta_params.E_ampa
    f.attrs['E_nmda'] = wta_params.E_nmda
    f.attrs['E_gaba_a'] = wta_params.E_gaba_a
    #f.attrs['E_gaba_b'] = wta_params.E_gaba_b
    f.attrs['tau_ampa'] = wta_params.tau_ampa
    f.attrs['tau1_nmda'] = wta_params.tau1_nmda
    f.attrs['tau2_nmda'] = wta_params.tau2_nmda
    f.attrs['tau_gaba_a'] = wta_params.tau_gaba_a
    #f.attrs['tau1_gaba_b'] = wta_params.tau1_gaba_b
    #f.attrs['tau2_gaba_b'] = wta_params.tau2_gaba_b
#    f.attrs['w_ampa_min'] = wta_params.w_ampa_min
#    f.attrs['w_ampa_max'] = wta_params.w_ampa_max
#    f.attrs['w_nmda_min'] = wta_params.w_nmda_min
#    f.attrs['w_nmda_max'] = wta_params.w_nmda_max
#    f.attrs['w_gaba_a_min'] = wta_params.w_gaba_a_min
#    f.attrs['w_gaba_a_max'] = wta_params.w_gaba_a_max
#    f.attrs['w_gaba_b_min'] = wta_params.w_gaba_b_min
#    f.attrs['w_gaba_b_max'] = wta_params.w_gaba_b_max
    f.attrs['pyr_w_ampa_ext']=wta_params.pyr_w_ampa_ext
    f.attrs['pyr_w_ampa_rec']=wta_params.pyr_w_ampa_rec
    f.attrs['int_w_ampa_ext']=wta_params.int_w_ampa_ext
    f.attrs['int_w_ampa_rec']=wta_params.int_w_ampa_rec
    f.attrs['pyr_w_nmda']=wta_params.pyr_w_nmda
    f.attrs['int_w_nmda']=wta_params.int_w_nmda
    f.attrs['pyr_w_gaba_a']=wta_params.pyr_w_gaba_a
    f.attrs['int_w_gaba_a']=wta_params.int_w_gaba_a
    f.attrs['p_b_e'] = wta_params.p_b_e
    f.attrs['p_x_e'] = wta_params.p_x_e
    f.attrs['p_e_e'] = wta_params.p_e_e
    f.attrs['p_e_i'] = wta_params.p_e_i
    f.attrs['p_i_i'] = wta_params.p_i_i
    f.attrs['p_i_e'] = wta_params.p_i_e
    f.attrs['muscimol_amount'] = muscimol_amount
    f.attrs['injection_site'] = injection_site
    f.attrs['p_dcs']=p_dcs
    f.attrs['i_dcs']=i_dcs

    if not wta_monitor.save_summary_only:
        # Write LFP data
        if record_lfp:
            f_lfp = f.create_group('lfp')
            f_lfp['lfp']=wta_monitor.lfp_monitor.values

        # Write voxel data
        if record_voxel:
            f_vox = f.create_group('voxel')
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
            #f_state['g_gaba_b'] = wta_monitor.network_monitor['g_gaba_b'].values
            f_state['I_ampa_r'] = wta_monitor.network_monitor['I_ampa_r'].values
            f_state['I_ampa_x'] = wta_monitor.network_monitor['I_ampa_x'].values
            f_state['I_ampa_b'] = wta_monitor.network_monitor['I_ampa_b'].values
            f_state['I_nmda'] = wta_monitor.network_monitor['I_nmda'].values
            f_state['I_gaba_a'] = wta_monitor.network_monitor['I_gaba_a'].values
            #f_state['I_gaba_b'] = wta_monitor.network_monitor['I_gaba_b'].values
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

    else:
        f_summary=f.create_group('summary')
        e_mean_final=[]
        e_max=[]
        for rate_monitor in wta_monitor.population_rate_monitors['excitatory']:
            e_rate=rate_monitor.smooth_rate(width=5*ms, filter='gaussian')
            e_mean_final.append(np.mean(e_rate[6500:7500]))
            e_max.append(np.max(e_rate))
        i_mean_final=[]
        i_max=[]
        for rate_monitor in wta_monitor.population_rate_monitors['inhibitory']:
            i_rate=rate_monitor.smooth_rate(width=5*ms, filter='gaussian')
            i_mean_final.append(np.mean(i_rate[6500:7500]))
            i_max.append(np.max(i_rate))
        f_summary['e_mean']=np.array(e_mean_final)
        f_summary['e_max']=np.array(e_max)
        f_summary['i_mean']=np.array(i_mean_final)
        f_summary['i_max']=np.array(i_max)
        f_summary['bold_max']=np.max(wta_monitor.voxel_monitor['y'].values)
        f_summary['bold_exc_max']=np.max(wta_monitor.voxel_exc_monitor['y'].values)

    f.close()