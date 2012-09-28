from brian import StateMonitor, MultiStateMonitor, PopulationRateMonitor, SpikeMonitor, raster_plot, ms, hertz, nS, nA, mA
from matplotlib.pyplot import figure, subplot, xlim, ylim, ylabel, legend, xlabel, show, title
import numpy as np

class BrainMonitor():

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
    def __init__(self, background_inputs, visual_cortex_inputs, go_input, brain_network, left_lfp_source, right_lfp_source,
                 left_voxel, right_voxel, record_lfp=True, record_voxel=True, record_neuron_state=False,
                 record_spikes=True, record_pop_firing_rate=True, record_neuron_firing_rates=False,
                 record_inputs=False):

        self.monitors=[]

        self.brain_network=brain_network

        # LFP monitor
        if record_lfp:
            self.left_lfp_monitor = StateMonitor(left_lfp_source, 'LFP', record=0)
            self.right_lfp_monitor = StateMonitor(right_lfp_source, 'LFP', record=0)
            self.monitors.append(self.left_lfp_monitor)
            self.monitors.append(self.right_lfp_monitor)
        else:
            self.left_lfp_monitor=None
            self.right_lfp_monitor=None

        # Voxel monitor
        if record_voxel:
            self.left_voxel_monitor = MultiStateMonitor(left_voxel, vars=['G_total','G_total_exc','s','f_in','v',
                                                                          'f_out','q','y'], record=True)
            self.right_voxel_monitor = MultiStateMonitor(right_voxel, vars=['G_total','G_total_exc','s','f_in','v',
                                                                            'f_out','q','y'], record=True)
            self.monitors.append(self.left_voxel_monitor)
            self.monitors.append(self.right_voxel_monitor)
        else:
            self.left_voxel_monitor=None
            self.right_voxel_monitor=None
        self.left_voxel_exc_monitor=None
        self.right_voxel_exc_monitor=None

        # Network monitor
        if record_neuron_state:
            self.left_record_idx=[]
            # left
            # e_contra_vis
            self.left_record_idx.append(1)
            # e_contra_mem
            self.left_record_idx.append(brain_network.left_lip.e_contra_vis_size+1)
            # e_ipsi vis
            self.left_record_idx.append(brain_network.left_lip.e_contra_size+1)
            # e_ipsi mem
            self.left_record_idx.append(brain_network.left_lip.e_contra_size+brain_network.left_lip.e_ipsi_vis_size+1)
            # i_contra
            self.left_record_idx.append(brain_network.left_lip.e_contra_size+brain_network.left_lip.e_ipsi_size+1)
            # i ipsi
            self.left_record_idx.append(brain_network.left_lip.e_contra_size+brain_network.left_lip.e_ipsi_size+brain_network.left_lip.i_contra_size+1)

            self.right_record_idx=[]
            # right
            # e_contra_vis
            self.right_record_idx.append(1)
            # e_contra_mem
            self.right_record_idx.append(brain_network.right_lip.e_contra_vis_size+1)
            # e_ipsi vis
            self.right_record_idx.append(brain_network.right_lip.e_contra_size+1)
            # e_ipsi mem
            self.right_record_idx.append(brain_network.right_lip.e_contra_size+brain_network.right_lip.e_ipsi_vis_size+1)
            # i_contra
            self.right_record_idx.append(brain_network.right_lip.e_contra_size+brain_network.right_lip.e_ipsi_size+1)
            # i ipsi
            self.right_record_idx.append(brain_network.right_lip.e_contra_size+brain_network.right_lip.e_ipsi_size+brain_network.right_lip.i_contra_size+1)

            self.left_network_monitor=MultiStateMonitor(brain_network.left_lip.neuron_group,
                vars=['vm','g_ampa_r','g_ampa_x','g_ampa_g','g_ampa_b','g_gaba_a','g_gaba_b','g_nmda','I_ampa_r',
                      'I_ampa_x','I_ampa_b','I_ampa_g','I_gaba_a','I_gaba_b','I_nmda'],
                record=self.left_record_idx)
            self.right_network_monitor=MultiStateMonitor(brain_network.right_lip.neuron_group,
                vars=['vm','g_ampa_r','g_ampa_x','g_ampa_g','g_ampa_b','g_gaba_a','g_gaba_b','g_nmda','I_ampa_r',
                      'I_ampa_x','I_ampa_b','I_ampa_g','I_gaba_a','I_gaba_b','I_nmda'],
                record=self.right_record_idx)

            self.monitors.append(self.left_network_monitor)
            self.monitors.append(self.right_network_monitor)

        else:
            self.left_network_monitor=None
            self.right_network_monitor=None

        # Population rate monitors
        if record_pop_firing_rate:
            self.population_rate_monitors={'left_ec_vis':PopulationRateMonitor(brain_network.left_lip.e_contra_vis),
                                           'left_ec_mem':PopulationRateMonitor(brain_network.left_lip.e_contra_mem),
                                           'left_ei_vis':PopulationRateMonitor(brain_network.left_lip.e_ipsi_vis),
                                           'left_ei_mem':PopulationRateMonitor(brain_network.left_lip.e_ipsi_mem),
                                           'left_ic':PopulationRateMonitor(brain_network.left_lip.i_contra),
                                           'left_ii':PopulationRateMonitor(brain_network.left_lip.i_ipsi),
                                           'right_ec_vis':PopulationRateMonitor(brain_network.right_lip.e_contra_vis),
                                           'right_ec_mem':PopulationRateMonitor(brain_network.right_lip.e_contra_mem),
                                           'right_ei_vis':PopulationRateMonitor(brain_network.right_lip.e_ipsi_vis),
                                           'right_ei_mem':PopulationRateMonitor(brain_network.right_lip.e_ipsi_mem),
                                           'right_ic':PopulationRateMonitor(brain_network.right_lip.i_contra),
                                           'right_ii':PopulationRateMonitor(brain_network.right_lip.i_ipsi)}
            for mon_name, mon in self.population_rate_monitors.iteritems():
                self.monitors.append(mon)
        else:
            self.population_rate_monitors=None

        if record_neuron_firing_rates:
            self.neuron_rate_monitors={'left_ec_vis':[],
                                       'left_ec_mem':[],
                                       'left_ei_vis':[],
                                       'left_ei_mem':[],
                                       'left_ic':[],
                                       'left_ii':[],
                                       'right_ec_vis': [],
                                       'right_ec_mem': [],
                                       'right_ei_vis': [],
                                       'right_ei_mem': [],
                                       'right_ic': [],
                                       'right_ii': []}


        # Input rate monitors
        if record_inputs:
            self.left_background_rate_monitor=PopulationRateMonitor(background_inputs[0])
            self.monitors.append(self.left_background_rate_monitor)
            self.right_background_rate_monitor=PopulationRateMonitor(background_inputs[1])
            self.monitors.append(self.right_background_rate_monitor)
            self.left_visual_cortex_monitor=PopulationRateMonitor(visual_cortex_inputs[0])
            self.monitors.append(self.left_visual_cortex_monitor)
            self.right_visual_cortex_monitor=PopulationRateMonitor(visual_cortex_inputs[1])
            self.monitors.append(self.right_visual_cortex_monitor)
            self.go_input_monitor=PopulationRateMonitor(go_input)
            self.monitors.append(self.go_input_monitor)
        else:
            self.left_background_rate_monitor=None
            self.right_background_rate_monitor=None
            self.left_visual_cortex_monitor=None
            self.right_visual_cortex_monitor=None
            self.go_input_monitor=None

        # Spike monitors
        if record_spikes:
            self.spike_monitors={'left_ec':SpikeMonitor(brain_network.left_lip.e_contra),
                                 'left_ei':SpikeMonitor(brain_network.left_lip.e_ipsi),
                                 'left_ic':SpikeMonitor(brain_network.left_lip.i_contra),
                                 'left_ii':SpikeMonitor(brain_network.left_lip.i_ipsi),
                                 'right_ec':SpikeMonitor(brain_network.right_lip.e_contra),
                                 'right_ei':SpikeMonitor(brain_network.right_lip.e_ipsi),
                                 'right_ic':SpikeMonitor(brain_network.right_lip.i_contra),
                                 'right_ii':SpikeMonitor(brain_network.right_lip.i_ipsi)}
            for mon_name, mon in self.spike_monitors.iteritems():
                self.monitors.append(mon)

        else:
            self.spike_monitors=None

    # Plot monitor data
    def plot(self, trial_duration):

        # Spike raster plots
        if self.spike_monitors is not None:
            figure()
            subplot(811)
            raster_plot(self.spike_monitors['left_ec'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.left_lip.e_contra_size)
            ylabel('left EC')
            subplot(812)
            raster_plot(self.spike_monitors['left_ei'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.left_lip.e_ipsi_size)
            ylabel('left EI')
            subplot(813)
            raster_plot(self.spike_monitors['left_ic'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.left_lip.i_contra_size)
            ylabel('left IC')
            subplot(814)
            raster_plot(self.spike_monitors['left_ii'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.left_lip.i_ipsi_size)
            ylabel('left II')
            subplot(815)
            raster_plot(self.spike_monitors['right_ec'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.right_lip.e_contra_size)
            ylabel('right EC')
            subplot(816)
            raster_plot(self.spike_monitors['right_ei'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.right_lip.e_ipsi_size)
            ylabel('right EI')
            subplot(817)
            raster_plot(self.spike_monitors['right_ic'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.right_lip.i_contra_size)
            ylabel('right IC')
            subplot(818)
            raster_plot(self.spike_monitors['right_ii'],newfigure=False)
            xlim(0,trial_duration/ms)
            ylim(0,self.brain_network.right_lip.i_ipsi_size)
            ylabel('right II')

        # Network firing rate plots
        if self.population_rate_monitors is not None:
            max_rate=np.max([np.max(self.population_rate_monitors['left_ec_vis'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['left_ec_mem'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['left_ei_vis'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['left_ei_mem'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['left_ic'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['left_ii'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['right_ec_vis'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['right_ec_mem'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['right_ei_vis'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['right_ei_mem'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['right_ic'].smooth_rate(width=5*ms)/hertz),
                             np.max(self.population_rate_monitors['right_ii'].smooth_rate(width=5*ms)/hertz)])
            figure()
            ax=subplot(211)
            ax.plot(self.population_rate_monitors['left_ec_vis'].times/ms,
                self.population_rate_monitors['left_ec_vis'].smooth_rate(width=5*ms)/hertz, label='left LIP EC vis')
            ax.plot(self.population_rate_monitors['left_ec_mem'].times/ms,
                self.population_rate_monitors['left_ec_mem'].smooth_rate(width=5*ms)/hertz, label='left LIP EC mem')
            ax.plot(self.population_rate_monitors['left_ei_vis'].times/ms,
                self.population_rate_monitors['left_ei_vis'].smooth_rate(width=5*ms)/hertz, label='left LIP EI vis')
            ax.plot(self.population_rate_monitors['left_ei_mem'].times/ms,
                self.population_rate_monitors['left_ei_mem'].smooth_rate(width=5*ms)/hertz, label='left LIP EI mem')
            ax.plot(self.population_rate_monitors['left_ic'].times/ms,
                self.population_rate_monitors['left_ic'].smooth_rate(width=5*ms)/hertz, label='left LIP IC')
            ax.plot(self.population_rate_monitors['left_ii'].times/ms,
                self.population_rate_monitors['left_ii'].smooth_rate(width=5*ms)/hertz, label='left LIP II')
            ylim(0,max_rate)
            legend()
            xlabel('Time (ms)')
            ylabel('Population Firing Rate (Hz)')

            ax=subplot(212)
            ax.plot(self.population_rate_monitors['right_ec_vis'].times/ms,
                self.population_rate_monitors['right_ec_vis'].smooth_rate(width=5*ms)/hertz, label='right LIP EC vis')
            ax.plot(self.population_rate_monitors['right_ec_mem'].times/ms,
                self.population_rate_monitors['right_ec_mem'].smooth_rate(width=5*ms)/hertz, label='right LIP EC mem')
            ax.plot(self.population_rate_monitors['right_ei_vis'].times/ms,
                self.population_rate_monitors['right_ei_vis'].smooth_rate(width=5*ms)/hertz, label='right LIP EI vis')
            ax.plot(self.population_rate_monitors['right_ei_mem'].times/ms,
                self.population_rate_monitors['right_ei_mem'].smooth_rate(width=5*ms)/hertz, label='right LIP EI mem')
            ax.plot(self.population_rate_monitors['right_ic'].times/ms,
                self.population_rate_monitors['right_ic'].smooth_rate(width=5*ms)/hertz, label='right LIP IC')
            ax.plot(self.population_rate_monitors['right_ii'].times/ms,
                self.population_rate_monitors['right_ii'].smooth_rate(width=5*ms)/hertz, label='right LIP II')
            ylim(0,max_rate)
            legend()
            xlabel('Time (ms)')
            ylabel('Population Firing Rate (Hz)')

        # Input firing rate plots
        if self.left_background_rate_monitor is not None and self.right_background_rate_monitor is not None and\
           self.left_visual_cortex_monitor is not None and self.right_visual_cortex_monitor is not None and \
           self.go_input_monitor is not None:
            figure()
            max_rate=np.max([np.max(self.left_background_rate_monitor.smooth_rate(width=5*ms)/hertz),
                             np.max(self.right_background_rate_monitor.smooth_rate(width=5*ms)/hertz),
                             np.max(self.left_visual_cortex_monitor.smooth_rate(width=5*ms)/hertz),
                             np.max(self.right_visual_cortex_monitor.smooth_rate(width=5*ms)/hertz),
                             np.max(self.go_input_monitor.smooth_rate(width=5*ms)/hertz)])
            ax=subplot(111)
            ax.plot(self.left_background_rate_monitor.times/ms, self.left_background_rate_monitor.smooth_rate(width=5*ms)/hertz, label='left background')
            ax.plot(self.right_background_rate_monitor.times/ms, self.right_background_rate_monitor.smooth_rate(width=5*ms)/hertz, label='right background')
            ax.plot(self.left_visual_cortex_monitor.times/ms, self.left_visual_cortex_monitor.smooth_rate(width=5*ms)/hertz, label='left VC')
            ax.plot(self.right_visual_cortex_monitor.times/ms, self.right_visual_cortex_monitor.smooth_rate(width=5*ms)/hertz, label='right VC')
            ax.plot(self.go_input_monitor.times/ms, self.go_input_monitor.smooth_rate(width=5*ms)/hertz, label='Go')
            legend()
            ylim(0,max_rate)

        # Network state plots
        if self.left_network_monitor is not None and self.right_network_monitor is not None:
            max_conductances=[]
            for idx in self.left_record_idx:
                max_conductances.append(np.max(self.left_network_monitor['g_ampa_r'][idx]/nS))
                max_conductances.append(np.max(self.left_network_monitor['g_ampa_x'][idx]/nS))
                max_conductances.append(np.max(self.left_network_monitor['g_ampa_b'][idx]/nS))
                max_conductances.append(np.max(self.left_network_monitor['g_ampa_g'][idx]/nS))
                max_conductances.append(np.max(self.left_network_monitor['g_nmda'][idx]/nS))
                max_conductances.append(np.max(self.left_network_monitor['g_gaba_a'][idx]/nS))
                max_conductances.append(np.max(self.left_network_monitor['g_gaba_b'][idx]/nS))
            for idx in self.right_record_idx:
                max_conductances.append(np.max(self.right_network_monitor['g_ampa_r'][idx]/nS))
                max_conductances.append(np.max(self.right_network_monitor['g_ampa_x'][idx]/nS))
                max_conductances.append(np.max(self.right_network_monitor['g_ampa_b'][idx]/nS))
                max_conductances.append(np.max(self.right_network_monitor['g_ampa_g'][idx]/nS))
                max_conductances.append(np.max(self.right_network_monitor['g_nmda'][idx]/nS))
                max_conductances.append(np.max(self.right_network_monitor['g_gaba_a'][idx]/nS))
                max_conductances.append(np.max(self.right_network_monitor['g_gaba_b'][idx]/nS))
            max_conductance=np.max(max_conductances)

            figure()
            labels=['e_contra_vis','e_contra_mem','e_ipsi_vis','e_ipsi_mem','i_contra','i_ipsi']
            for i,idx in enumerate(self.left_record_idx):
                ax=subplot(len(self.left_record_idx),1,i+1)
                ax.plot(self.left_network_monitor['g_ampa_r'].times/ms, self.left_network_monitor['g_ampa_r'][idx]/nS,
                    label='AMPA recurrent')
                ax.plot(self.left_network_monitor['g_ampa_x'].times/ms, self.left_network_monitor['g_ampa_x'][idx]/nS,
                    label='AMPA task')
                ax.plot(self.left_network_monitor['g_ampa_b'].times/ms, self.left_network_monitor['g_ampa_b'][idx]/nS,
                    label='AMPA backgrnd')
                ax.plot(self.left_network_monitor['g_ampa_g'].times/ms, self.left_network_monitor['g_ampa_g'][idx]/nS,
                    label='AMPA go')
                ax.plot(self.left_network_monitor['g_nmda'].times/ms, self.left_network_monitor['g_nmda'][idx]/nS,
                    label='NMDA')
                ax.plot(self.left_network_monitor['g_gaba_a'].times/ms, self.left_network_monitor['g_gaba_a'][idx]/nS,
                    label='GABA_A')
                ax.plot(self.left_network_monitor['g_gaba_b'].times/ms, self.left_network_monitor['g_gaba_b'][idx]/nS,
                    label='GABA_B')
                ylim(0,max_conductance)
                ylabel(labels[i])
                if not i:
                    title('Left LIP - Conductance (nS)')
                    legend()
            xlabel('Time (ms)')

            figure()
            for i,idx in enumerate(self.right_record_idx):
                ax=subplot(len(self.right_record_idx),1,i+1)
                ax.plot(self.right_network_monitor['g_ampa_r'].times/ms, self.right_network_monitor['g_ampa_r'][idx]/nS,
                    label='AMPA recurrent')
                ax.plot(self.right_network_monitor['g_ampa_x'].times/ms, self.right_network_monitor['g_ampa_x'][idx]/nS,
                    label='AMPA task')
                ax.plot(self.right_network_monitor['g_ampa_b'].times/ms, self.right_network_monitor['g_ampa_b'][idx]/nS,
                    label='AMPA backgrnd')
                ax.plot(self.right_network_monitor['g_ampa_g'].times/ms, self.right_network_monitor['g_ampa_g'][idx]/nS,
                    label='AMPA go')
                ax.plot(self.right_network_monitor['g_nmda'].times/ms, self.right_network_monitor['g_nmda'][idx]/nS,
                    label='NMDA')
                ax.plot(self.right_network_monitor['g_gaba_a'].times/ms, self.right_network_monitor['g_gaba_a'][idx]/nS,
                    label='GABA_A')
                ax.plot(self.right_network_monitor['g_gaba_b'].times/ms, self.right_network_monitor['g_gaba_b'][idx]/nS,
                    label='GABA_B')
                ylim(0,max_conductance)
                ylabel(labels[i])
                if not i:
                    title('Right LIP - Conductance (nS)')
                    legend()
            xlabel('Time (ms)')

            min_currents=[]
            max_currents=[]
            for idx in self.left_record_idx:
                max_currents.append(np.max(self.left_network_monitor['I_ampa_r'][idx]/nA))
                max_currents.append(np.max(self.left_network_monitor['I_ampa_x'][idx]/nA))
                max_currents.append(np.max(self.left_network_monitor['I_ampa_b'][idx]/nA))
                max_currents.append(np.max(self.left_network_monitor['I_ampa_g'][idx]/nA))
                max_currents.append(np.max(self.left_network_monitor['I_nmda'][idx]/nA))
                max_currents.append(np.max(self.left_network_monitor['I_gaba_a'][idx]/nA))
                max_currents.append(np.max(self.left_network_monitor['I_gaba_b'][idx]/nA))
                min_currents.append(np.min(self.left_network_monitor['I_ampa_r'][idx]/nA))
                min_currents.append(np.min(self.left_network_monitor['I_ampa_x'][idx]/nA))
                min_currents.append(np.min(self.left_network_monitor['I_ampa_b'][idx]/nA))
                min_currents.append(np.min(self.left_network_monitor['I_ampa_g'][idx]/nA))
                min_currents.append(np.min(self.left_network_monitor['I_nmda'][idx]/nA))
                min_currents.append(np.min(self.left_network_monitor['I_gaba_a'][idx]/nA))
                min_currents.append(np.min(self.left_network_monitor['I_gaba_b'][idx]/nA))
            for idx in self.right_record_idx:
                max_currents.append(np.max(self.right_network_monitor['I_ampa_r'][idx]/nA))
                max_currents.append(np.max(self.right_network_monitor['I_ampa_x'][idx]/nA))
                max_currents.append(np.max(self.right_network_monitor['I_ampa_b'][idx]/nA))
                max_currents.append(np.max(self.right_network_monitor['I_ampa_g'][idx]/nA))
                max_currents.append(np.max(self.right_network_monitor['I_nmda'][idx]/nA))
                max_currents.append(np.max(self.right_network_monitor['I_gaba_a'][idx]/nA))
                max_currents.append(np.max(self.right_network_monitor['I_gaba_b'][idx]/nA))
                min_currents.append(np.min(self.right_network_monitor['I_ampa_r'][idx]/nA))
                min_currents.append(np.min(self.right_network_monitor['I_ampa_x'][idx]/nA))
                min_currents.append(np.min(self.right_network_monitor['I_ampa_b'][idx]/nA))
                min_currents.append(np.min(self.right_network_monitor['I_ampa_g'][idx]/nA))
                min_currents.append(np.min(self.right_network_monitor['I_nmda'][idx]/nA))
                min_currents.append(np.min(self.right_network_monitor['I_gaba_a'][idx]/nA))
                min_currents.append(np.min(self.right_network_monitor['I_gaba_b'][idx]/nA))
            max_current=np.max(max_currents)
            min_current=np.min(min_currents)

            figure()
            for i,neuron_idx in enumerate(self.left_record_idx):
                ax=subplot(len(self.left_record_idx),1,i+1)
                ax.plot(self.left_network_monitor['I_ampa_r'].times/ms, self.left_network_monitor['I_ampa_r'][neuron_idx]/nA,
                    label='AMPA-recurrent')
                ax.plot(self.left_network_monitor['I_ampa_x'].times/ms, self.left_network_monitor['I_ampa_x'][neuron_idx]/nA,
                    label='AMPA-task')
                ax.plot(self.left_network_monitor['I_ampa_b'].times/ms, self.left_network_monitor['I_ampa_b'][neuron_idx]/nA,
                    label='AMPA-backgrnd')
                ax.plot(self.left_network_monitor['I_ampa_b'].times/ms, self.left_network_monitor['I_ampa_g'][neuron_idx]/nA,
                    label='AMPA-go')
                ax.plot(self.left_network_monitor['I_nmda'].times/ms, self.left_network_monitor['I_nmda'][neuron_idx]/nA,
                    label='NMDA')
                ax.plot(self.left_network_monitor['I_gaba_a'].times/ms, self.left_network_monitor['I_gaba_a'][neuron_idx]/nA,
                    label='GABA_A')
                ax.plot(self.left_network_monitor['I_gaba_b'].times/ms, self.left_network_monitor['I_gaba_b'][neuron_idx]/nA,
                    label='GABA_B')
                ylim(min_current,max_current)
                ylabel(labels[i])
                if not i:
                    title('Left LIP - Current (nA)')
                    legend()
            xlabel('Time (ms)')

            figure()
            for i,neuron_idx in enumerate(self.right_record_idx):
                ax=subplot(len(self.right_record_idx),1,i+1)
                ax.plot(self.right_network_monitor['I_ampa_r'].times/ms, self.right_network_monitor['I_ampa_r'][neuron_idx]/nA,
                    label='AMPA-recurrent')
                ax.plot(self.right_network_monitor['I_ampa_x'].times/ms, self.right_network_monitor['I_ampa_x'][neuron_idx]/nA,
                    label='AMPA-task')
                ax.plot(self.right_network_monitor['I_ampa_b'].times/ms, self.right_network_monitor['I_ampa_b'][neuron_idx]/nA,
                    label='AMPA-backgrnd')
                ax.plot(self.right_network_monitor['I_ampa_b'].times/ms, self.right_network_monitor['I_ampa_g'][neuron_idx]/nA,
                    label='AMPA-go')
                ax.plot(self.right_network_monitor['I_nmda'].times/ms, self.right_network_monitor['I_nmda'][neuron_idx]/nA,
                    label='NMDA')
                ax.plot(self.right_network_monitor['I_gaba_a'].times/ms, self.right_network_monitor['I_gaba_a'][neuron_idx]/nA,
                    label='GABA_A')
                ax.plot(self.right_network_monitor['I_gaba_b'].times/ms, self.right_network_monitor['I_gaba_b'][neuron_idx]/nA,
                    label='GABA_B')
                ylim(min_current,max_current)
                ylabel(labels[i])
                if not i:
                    title('Right LIP - Current (nA)')
                    legend()
            xlabel('Time (ms)')

        # LFP plot
        if self.left_lfp_monitor is not None and self.right_lfp_monitor is not None:
            figure()
            ax=subplot(111)
            ax.plot(self.left_lfp_monitor.times / ms, self.left_lfp_monitor[0] / mA, label='left LIP')
            ax.plot(self.right_lfp_monitor.times / ms, self.right_lfp_monitor[0] / mA, label='right LIP')
            legend()
            xlabel('Time (ms)')
            ylabel('LFP (mA)')

        # Voxel activity plots
        if self.left_voxel_monitor is not None and self.right_voxel_monitor is not None:
            syn_max=np.max([np.max(self.left_voxel_monitor['G_total'][0] / nS),
                            np.max(self.right_voxel_monitor['G_total'][0] / nS)])
            y_max=np.max([np.max(self.left_voxel_monitor['y'][0]), np.max(self.right_voxel_monitor['y'][0])])
            y_min=np.min([np.min(self.left_voxel_monitor['y'][0]), np.min(self.right_voxel_monitor['y'][0])])
            figure()
            if self.left_voxel_exc_monitor is None and self.right_voxel_exc_monitor is None:
                ax=subplot(211)
            else:
                ax=subplot(221)
                syn_max=np.max([syn_max, np.max(self.left_voxel_exc_monitor['G_total'][0]),
                                np.max(self.right_voxel_exc_monitor['G_total'][0])])
                y_max=np.max([y_max, np.max(self.left_voxel_exc_monitor['y'][0]),
                              np.max(self.right_voxel_exc_monitor['y'][0])])
                y_min=np.min([y_min, np.min(self.left_voxel_exc_monitor['y'][0]),
                              np.min(self.right_voxel_exc_monitor['y'][0])])
            ax.plot(self.left_voxel_monitor['G_total'].times / ms, self.left_voxel_monitor['G_total'][0] / nS,
                label='left LIP')
            ax.plot(self.right_voxel_monitor['G_total'].times / ms, self.right_voxel_monitor['G_total'][0] / nS,
                label='right LIP')
            legend()
            xlabel('Time (ms)')
            ylabel('Total Synaptic Activity (nS)')
            ylim(0, syn_max)
            if self.left_voxel_exc_monitor is None and self.right_voxel_exc_monitor is None:
                ax=subplot(212)
            else:
                ax=subplot(222)
            ax.plot(self.left_voxel_monitor['y'].times / ms, self.left_voxel_monitor['y'][0], label='left LIP')
            ax.plot(self.right_voxel_monitor['y'].times / ms, self.right_voxel_monitor['y'][0], label='right LIP')
            legend()
            xlabel('Time (ms)')
            ylabel('BOLD')
            ylim(y_min, y_max)
            if self.left_voxel_exc_monitor is not None and self.right_voxel_exc_monitor is not None:
                ax=subplot(223)
                ax.plot(self.left_voxel_exc_monitor['G_total'].times / ms,
                    self.left_voxel_exc_monitor['G_total'][0] / nS, label='left LIP')
                ax.plot(self.right_voxel_exc_monitor['G_total'].times / ms,
                    self.right_voxel_exc_monitor['G_total'][0] / nS, label='right LIP')
                legend()
                xlabel('Time (ms)')
                ylabel('Total Synaptic Activity (nS)')
                ylim(0, syn_max)
                ax=subplot(224)
                ax.plot(self.left_voxel_exc_monitor['y'].times / ms, self.left_voxel_exc_monitor['y'][0], label='left LIP')
                ax.plot(self.right_voxel_exc_monitor['y'].times / ms, self.right_voxel_exc_monitor['y'][0], label='right LIP')
                legend()
                xlabel('Time (ms)')
                ylabel('BOLD')
                ylim(y_min, y_max)
        show()