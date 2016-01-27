import h5py
from brian.experimental.connectionmonitor import ConnectionMonitor
import numpy as np
from brian import StateMonitor, MultiStateMonitor, PopulationRateMonitor, SpikeMonitor, raster_plot, ms, hertz, nS, nA, mA, defaultclock, second, Clock, Hz
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, ylim, legend, ylabel, xlabel, show, title
# Collection of monitors for WTA network
from pysbi.util.utils import get_response_time

class SessionMonitor():
    def __init__(self, network, ntrials, input_start_time, input_end_time, num_groups, response_threshold=25,
                 trial_dt=0.5*ms, record_connections=[], conv_window=10, recording_firing_rates=False):
        self.ntrials=ntrials
        self.response_threshold=response_threshold
        self.trial_dt=trial_dt
        self.input_start_time=input_start_time
        self.input_end_time=input_end_time
        # Accuracy convolution window size
        self.conv_window=conv_window
        self.trial_inputs=np.zeros((num_groups,ntrials))
        self.trial_rt=np.zeros((1,ntrials))
        self.trial_resp=np.zeros((1,ntrials))
        self.trial_correct=np.zeros((1,ntrials))
        self.record_connections=record_connections
        self.record_firing_rates=recording_firing_rates
        self.trial_weights={}
        for conn in self.record_connections:
            self.trial_weights[conn]=[]
        self.correct_avg = np.zeros((1, ntrials))
        self.pop_rates={}
        if self.record_firing_rates:
            for i,group_e in enumerate(network.groups_e):
                self.pop_rates['excitatory_rate_%d' % i]=[]

                self.pop_rates['inhibitory_rate']=[]
        self.num_no_response=0

    def record_trial(self, trial_idx, inputs, correct_input, wta_net, wta_monitor):
        self.trial_inputs[:,trial_idx]=inputs

        e_rate_0 = wta_monitor.monitors['excitatory_rate_0'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        e_rate_1 = wta_monitor.monitors['excitatory_rate_1'].smooth_rate(width= 5 * ms, filter = 'gaussian')
        i_rate = wta_monitor.monitors['inhibitory_rate'].smooth_rate(width= 5 * ms, filter = 'gaussian')

        if self.record_firing_rates:
            self.pop_rates['excitatory_rate_0'].append(e_rate_0)
            self.pop_rates['excitatory_rate_1'].append(e_rate_1)
            self.pop_rates['inhibitory_rate'].append(i_rate)

        rt, choice = get_response_time(np.array([e_rate_0, e_rate_1]), self.input_start_time, self.input_end_time,
            upper_threshold = self.response_threshold, dt = self.trial_dt)
        correct = choice == correct_input
        if choice>-1:
            print 'response time = %.3f correct = %d' % (rt, int(correct))
        else:
            print 'no response!'
            self.num_no_response+=1
        self.trial_rt[0,trial_idx]=rt
        self.trial_resp[0,trial_idx]=choice
        self.trial_correct[0,trial_idx]=correct
        self.correct_avg[0,trial_idx] = (np.sum(self.trial_correct))/(trial_idx+1)
        for conn in self.record_connections:
            self.trial_weights[conn].append(wta_net.connections[conn].W.todense())

    def plot(self):
        # Convolve accuracy
        correct_ma = (np.convolve(self.trial_correct[0,:], np.ones((self.conv_window,))/self.conv_window, mode='valid'))

        resp_trials=np.where(self.trial_resp[0,:]>-1)[0]
        perc_correct=float(np.sum(self.trial_correct[0,resp_trials]))/float(len(resp_trials))
        perc_correct_overall=float(np.sum(self.trial_correct[0,:]))/float(self.ntrials)
        print('perc correct (overall)=%.2f' % perc_correct_overall)
        print('perc correct (responded)=%.2f' % perc_correct)
        print('no response=%.2f' % (float(self.num_no_response)/float(self.ntrials)))

        trial_diag_weights=np.zeros((len(self.record_connections),self.ntrials))
        for i, conn in enumerate(self.record_connections):
            for trial_idx in range(self.ntrials):
                trial_diag_weights[i,trial_idx]=np.mean(np.diagonal(self.trial_weights[conn][trial_idx]))

        if len(self.record_connections)>0:
            plt.figure()
            for i in range(len(self.record_connections)):
                plt.plot(trial_diag_weights[i,:]/nS, label = self.record_connections[i])
            plt.legend(loc = 'best')
            #plt.ylim(0, gmax/nS)
            plt.xlabel('trial')
            plt.ylabel('average weight')

        plt.figure()
        plt.plot(self.trial_correct[0,:])
        plt.xlabel('trial')
        plt.ylabel('correct choice = 1')

        plt.figure()
        plt.plot(self.correct_avg[0,:], label = 'average')
        plt.plot(correct_ma, label = 'moving avg')
        plt.legend(loc = 'best')
        plt.ylim(0,1)
        plt.xlabel('trial')
        plt.ylabel('accuracy')

        plt.figure()
        plt.plot(self.trial_rt[0,:])
        plt.ylim(0, 2000)
        plt.xlabel('trial')
        plt.ylabel('response time')

        plt.figure()
        plt.plot(self.trial_resp[0,:])
        plt.xlabel('trial')
        plt.ylabel('choice e0=0 e1=1')
        plt.show()

    def write_output(self, background_input_size, background_freq, network_group_size, num_groups, output_file,
                     task_input_size, trial_duration, wta_params, pyr_params, inh_params, muscimol_amount,
                     injection_site, p_dcs, i_dcs):

        f = h5py.File(output_file, 'w')

        # Write basic parameters

        f.attrs['ntrials']=self.ntrials
        f.attrs['response_threshold']=self.response_threshold
        f.attrs['conv_window']=self.conv_window

        f.attrs['num_groups'] = num_groups
        f.attrs['trial_duration'] = trial_duration
        f.attrs['dt']=self.trial_dt
        f.attrs['background_freq'] = background_freq
        f.attrs['stim_start_time'] = self.input_start_time
        f.attrs['stim_end_time'] = self.input_end_time
        f.attrs['network_group_size'] = network_group_size
        f.attrs['background_input_size'] = background_input_size
        f.attrs['task_input_size'] = task_input_size

        f_network_params=f.create_group('network_params')
        for attr, value in wta_params.iteritems():
            f_network_params.attrs[attr] = value

        f_pyr_params=f.create_group('pyr_params')
        for attr, value in pyr_params.iteritems():
            f_pyr_params.attrs[attr] = value

        f_inh_params=f.create_group('inh_params')
        for attr, value in inh_params.iteritems():
            f_inh_params.attrs[attr] = value

        f.attrs['muscimol_amount'] = muscimol_amount
        f.attrs['injection_site'] = injection_site
        f.attrs['p_dcs']=p_dcs
        f.attrs['i_dcs']=i_dcs

        f_behav=f.create_group('behavior')
        f_behav['num_no_response']=self.num_no_response
        f_behav['trial_rt']=self.trial_rt
        f_behav['trial_resp']=self.trial_resp
        f_behav['trial_correct']=self.trial_correct

        f_neur=f.create_group('neural')
        f_neur['trial_inputs']=self.trial_inputs
        f_conns=f_neur.create_group('connections')
        for conn in self.record_connections:
            f_conn=f_conns.create_group('conn')
            for trial_idx in range(self.ntrials):
                f_conn['trial_%d' % trial_idx]=self.trial_weights[conn][trial_idx]

        f_rates=f_neur.create_group('firing_rates')
        if self.record_firing_rates:
            for trial_idx in range(self.ntrials):
                f_trial=f_rates.create_group('trial_%d' % trial_idx)
                f_trial['inhibitory_rate']=self.pop_rates['inhibitory_rate'][trial_idx]
                for i in range(num_groups):
                    f_trial['excitatory_rate_%d' % i]=self.pop_rates['excitatory_rate_%d' % i][trial_idx]
        f.close()


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
                 record_spikes=True, record_firing_rate=True, record_inputs=False, record_connections=None,
                 save_summary_only=False, clock=defaultclock):
        self.num_groups=network.num_groups
        self.N=network.N
        self.monitors={}
        self.save_summary_only=save_summary_only
        self.clock=clock
        self.record_lfp=record_lfp
        self.record_voxel=record_voxel
        self.record_neuron_state=record_neuron_state
        self.record_spikes=record_spikes
        self.record_firing_rate=record_firing_rate
        self.record_inputs=record_inputs
        self.record_connections=record_connections
        self.save_summary_only=save_summary_only

        # LFP monitor
        if self.record_lfp:
            self.monitors['lfp'] = StateMonitor(lfp_source, 'LFP', record=0, clock=clock)

        # Voxel monitor
        if self.record_voxel:
            self.monitors['voxel'] = MultiStateMonitor(voxel, vars=['G_total','G_total_exc','y'],
                record=True, clock=clock)

        # Network monitor
        if self.record_neuron_state:
            self.record_idx=[]
            for i in range(self.num_groups):
                e_idx=i*int(.8*self.N/self.num_groups)
                self.record_idx.append(e_idx)
            i_idx=int(.8*self.N)
            self.record_idx.append(i_idx)
            self.monitors['network'] = MultiStateMonitor(network, vars=['vm','g_ampa_r','g_ampa_x','g_ampa_b',
                                                                        'g_gaba_a', 'g_nmda','I_ampa_r','I_ampa_x',
                                                                        'I_ampa_b','I_gaba_a','I_nmda'], 
                record=self.record_idx, clock=clock)

        # Population rate monitors
        if self.record_firing_rate:
            for i,group_e in enumerate(network.groups_e):
                self.monitors['excitatory_rate_%d' % i]=PopulationRateMonitor(group_e)

            self.monitors['inhibitory_rate']=PopulationRateMonitor(network.group_i)

        # Input rate monitors
        if record_inputs:
            self.monitors['background_rate']=PopulationRateMonitor(network.background_input)
            for i,task_input in enumerate(network.task_inputs):
                self.monitors['task_rate_%d' % i]=PopulationRateMonitor(task_input)

        # Spike monitors
        if self.record_spikes:
            for i,group_e in enumerate(network.groups_e):
                self.monitors['excitatory_spike_%d' % i]=SpikeMonitor(group_e)

            self.monitors['inhibitory_spike']=SpikeMonitor(network.group_i)

        # Connection monitors
        if self.record_connections is not None:
            for connection in record_connections:
                self.monitors['connection_%s' % connection]=ConnectionMonitor(network.connections[connection], store=True,
                    clock=Clock(dt=.5*second))

    # Plot monitor data
    def plot(self):

        # Spike raster plots
        if self.record_spikes:
            num_plots=self.num_groups+1
            figure()
            for i in range(self.num_groups):
                subplot(num_plots,1,i+1)
                raster_plot(self.monitors['excitatory_spike_%d' % i],newfigure=False)
            subplot(num_plots,1,num_plots)
            raster_plot(self.monitors['inhibitory_spike'],newfigure=False)

        # Network firing rate plots
        if self.record_firing_rate:
            figure()
            ax=subplot(211)
            max_rates=[np.max(self.monitors['inhibitory_rate'].smooth_rate(width=5*ms)/hertz)]
            for i in range(self.num_groups):
                max_rates.append(np.max(self.monitors['excitatory_rate_%d' % i].smooth_rate(width=5*ms)/hertz))
            max_rate=np.max(max_rates)

            for idx in range(self.num_groups):
                pop_rate_monitor=self.monitors['excitatory_rate_%d' % idx]
                ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms)/hertz, label='e %d' % idx)
                ylim(0,max_rate)
            legend()
            ylabel('Firing rate (Hz)')

            ax=subplot(212)
            pop_rate_monitor=self.monitors['inhibitory_rate']
            ax.plot(pop_rate_monitor.times/ms, pop_rate_monitor.smooth_rate(width=5*ms)/hertz, label='i')
            ylim(0,max_rate)
            legend()
            ylabel('Firing rate (Hz)')
            xlabel('Time (ms)')

        # Input firing rate plots
        if self.record_inputs:
            figure()
            #max_rates=[np.max(self.monitors['background_rate'].smooth_rate(width=5*ms)/hertz)]
            max_rates=[]
            for i in range(self.num_groups):
                max_rates.append(np.max(self.monitors['task_rate_%d' % i].smooth_rate(width=5*ms,
                    filter='gaussian')/hertz))
            max_rate=np.max(max_rates)
            ax=subplot(111)
#            ax.plot(self.monitors['background_rate'].times/ms,
#                self.monitors['background_rate'].smooth_rate(width=5*ms)/hertz)
            ylim(0,max_rate)
            for i in range(self.num_groups):
                task_monitor=self.monitors['task_rate_%d' % i]
                ax.plot(task_monitor.times/ms, task_monitor.smooth_rate(width=5*ms,filter='gaussian')/hertz)
                ylim(0,max_rate)

        # Network state plots
        if self.record_neuron_state:
            network_monitor=self.monitors['network']
            max_conductances=[]
            for neuron_idx in self.record_idx:
                max_conductances.append(np.max(network_monitor['g_ampa_r'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_ampa_x'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_ampa_b'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_nmda'][neuron_idx]/nS))
                max_conductances.append(np.max(network_monitor['g_gaba_a'][neuron_idx]/nS))
                #max_conductances.append(np.max(self.network_monitor['g_gaba_b'][neuron_idx]/nS))
            max_conductance=np.max(max_conductances)

            fig=figure()
            for i in range(self.num_groups):
                neuron_idx=self.record_idx[i]
                ax=subplot(int('%d1%d' % (self.num_groups+1,i+1)))
                title('e%d' % i)
                ax.plot(network_monitor['g_ampa_r'].times/ms, network_monitor['g_ampa_r'][neuron_idx]/nS, 
                    label='AMPA-recurrent')
                ax.plot(network_monitor['g_ampa_x'].times/ms, network_monitor['g_ampa_x'][neuron_idx]/nS,
                    label='AMPA-task')
                ax.plot(network_monitor['g_ampa_b'].times/ms, network_monitor['g_ampa_b'][neuron_idx]/nS,
                    label='AMPA-backgrnd')
                ax.plot(network_monitor['g_nmda'].times/ms, network_monitor['g_nmda'][neuron_idx]/nS,
                    label='NMDA')
                ax.plot(network_monitor['g_gaba_a'].times/ms, network_monitor['g_gaba_a'][neuron_idx]/nS,
                    label='GABA_A')
                #ax.plot(network_monitor['g_gaba_b'].times/ms, network_monitor['g_gaba_b'][neuron_idx]/nS,
                #    label='GABA_B')
                ylim(0,max_conductance)
                xlabel('Time (ms)')
                ylabel('Conductance (nS)')
                legend()

            neuron_idx=self.record_idx[self.num_groups]
            ax=subplot('%d1%d' % (self.num_groups+1,self.num_groups+1))
            title('i')
            ax.plot(network_monitor['g_ampa_r'].times/ms, network_monitor['g_ampa_r'][neuron_idx]/nS,
                label='AMPA-recurrent')
            ax.plot(network_monitor['g_ampa_x'].times/ms, network_monitor['g_ampa_x'][neuron_idx]/nS,
                label='AMPA-task')
            ax.plot(network_monitor['g_ampa_b'].times/ms, network_monitor['g_ampa_b'][neuron_idx]/nS,
                label='AMPA-backgrnd')
            ax.plot(network_monitor['g_nmda'].times/ms, network_monitor['g_nmda'][neuron_idx]/nS,
                label='NMDA')
            ax.plot(network_monitor['g_gaba_a'].times/ms, network_monitor['g_gaba_a'][neuron_idx]/nS,
                label='GABA_A')
            #ax.plot(network_monitor['g_gaba_b'].times/ms, network_monitor['g_gaba_b'][neuron_idx]/nS,
            #    label='GABA_B')
            ylim(0,max_conductance)
            xlabel('Time (ms)')
            ylabel('Conductance (nS)')
            legend()

            min_currents=[]
            max_currents=[]
            for neuron_idx in self.record_idx:
                max_currents.append(np.max(network_monitor['I_ampa_r'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_ampa_x'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_ampa_b'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_nmda'][neuron_idx]/nS))
                max_currents.append(np.max(network_monitor['I_gaba_a'][neuron_idx]/nS))
                #max_currents.append(np.max(network_monitor['I_gaba_b'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_ampa_r'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_ampa_x'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_ampa_b'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_nmda'][neuron_idx]/nS))
                min_currents.append(np.min(network_monitor['I_gaba_a'][neuron_idx]/nS))
                #min_currents.append(np.min(network_monitor['I_gaba_b'][neuron_idx]/nS))
            max_current=np.max(max_currents)
            min_current=np.min(min_currents)

            fig=figure()
            for i in range(self.num_groups):
                ax=subplot(int('%d1%d' % (self.num_groups+1,i+1)))
                neuron_idx=self.record_idx[i]
                title('e%d' % i)
                ax.plot(network_monitor['I_ampa_r'].times/ms, network_monitor['I_ampa_r'][neuron_idx]/nA,
                    label='AMPA-recurrent')
                ax.plot(network_monitor['I_ampa_x'].times/ms, network_monitor['I_ampa_x'][neuron_idx]/nA,
                    label='AMPA-task')
                ax.plot(network_monitor['I_ampa_b'].times/ms, network_monitor['I_ampa_b'][neuron_idx]/nA,
                    label='AMPA-backgrnd')
                ax.plot(network_monitor['I_nmda'].times/ms, network_monitor['I_nmda'][neuron_idx]/nA,
                    label='NMDA')
                ax.plot(network_monitor['I_gaba_a'].times/ms, network_monitor['I_gaba_a'][neuron_idx]/nA,
                    label='GABA_A')
                #ax.plot(network_monitor['I_gaba_b'].times/ms, network_monitor['I_gaba_b'][neuron_idx]/nA,
                #    label='GABA_B')
                ylim(min_current,max_current)
                xlabel('Time (ms)')
                ylabel('Current (nA)')
                legend()

            ax=subplot(int('%d1%d' % (self.num_groups+1,self.num_groups+1)))
            neuron_idx=self.record_idx[self.num_groups]
            title('i')
            ax.plot(network_monitor['I_ampa_r'].times/ms, network_monitor['I_ampa_r'][neuron_idx]/nA,
                label='AMPA-recurrent')
            ax.plot(network_monitor['I_ampa_x'].times/ms, network_monitor['I_ampa_x'][neuron_idx]/nA,
                label='AMPA-task')
            ax.plot(network_monitor['I_ampa_b'].times/ms, network_monitor['I_ampa_b'][neuron_idx]/nA,
                label='AMPA-backgrnd')
            ax.plot(network_monitor['I_nmda'].times/ms, network_monitor['I_nmda'][neuron_idx]/nA,
                label='NMDA')
            ax.plot(network_monitor['I_gaba_a'].times/ms, network_monitor['I_gaba_a'][neuron_idx]/nA,
                label='GABA_A')
            #ax.plot(network_monitor['I_gaba_b'].times/ms, network_monitor['I_gaba_b'][neuron_idx]/nA,
            #    label='GABA_B')
            ylim(min_current,max_current)
            xlabel('Time (ms)')
            ylabel('Current (nA)')
            legend()

        # LFP plot
        if self.record_lfp:
            figure()
            ax=subplot(111)
            ax.plot(self.monitors['lfp'].times / ms, self.monitors['lfp'][0]/mA)
            xlabel('Time (ms)')
            ylabel('LFP (mA)')

        # Voxel activity plots
        if self.record_voxel:
            voxel_monitor=self.monitors['voxel']
            voxel_exc_monitor=None
            if 'voxel_exc' in self.monitors:
                voxel_exc_monitor=self.monitors['voxel_exc']
            syn_max=np.max(voxel_monitor['G_total'][0] / nS)
            y_max=np.max(voxel_monitor['y'][0])
            y_min=np.min(voxel_monitor['y'][0])
            figure()
            if voxel_exc_monitor is None:
                ax=subplot(211)
            else:
                ax=subplot(221)
                syn_max=np.max([syn_max, np.max(voxel_exc_monitor['G_total'][0])])
                y_max=np.max([y_max, np.max(voxel_exc_monitor['y'][0])])
                y_min=np.min([y_min, np.min(voxel_exc_monitor['y'][0])])
            ax.plot(voxel_monitor['G_total'].times / ms, voxel_monitor['G_total'][0] / nS)
            xlabel('Time (ms)')
            ylabel('Total Synaptic Activity (nS)')
            ylim(0, syn_max)
            if voxel_exc_monitor is None:
                ax=subplot(212)
            else:
                ax=subplot(222)
            ax.plot(voxel_monitor['y'].times / ms, voxel_monitor['y'][0])
            xlabel('Time (ms)')
            ylabel('BOLD')
            ylim(y_min, y_max)
            if voxel_exc_monitor is not None:
                ax=subplot(223)
                ax.plot(voxel_exc_monitor['G_total'].times / ms, voxel_exc_monitor['G_total'][0] / nS)
                xlabel('Time (ms)')
                ylabel('Total Synaptic Activity (nS)')
                ylim(0, syn_max)
                ax=subplot(224)
                ax.plot(voxel_exc_monitor['y'].times / ms, voxel_exc_monitor['y'][0])
                xlabel('Time (ms)')
                ylabel('BOLD')
                ylim(y_min, y_max)

        if self.record_connections:
            figure()
            ax=subplot(111)
            for mon in self.monitors:
                if mon.startswith('connection_'):
                    conn_name=mon[11:]
                    conns=np.zeros((len(self.monitors[mon].values),1))
                    conn_times=[]
                    for idx, (time, conn_matrix) in enumerate(self.monitors[mon].values):
                        conn_diag=np.diagonal(conn_matrix.todense())
                        mean_w=np.mean(conn_diag)
                        conns[idx,0]=mean_w
                        conn_times.append(time)
                    ax.plot(np.array(conn_times) / ms, conns[:,0]/nS, label=conn_name)
            legend(loc='best')
            xlabel('Time (ms)')
            ylabel('Connection Weight (nS)')

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
    def write_output(self, background_input_size, background_freq, input_freq, network_group_size, num_groups, output_file,
                     stim_end_time, stim_start_time, task_input_size, trial_duration, voxel, wta_params,
                     pyr_params, inh_params, muscimol_amount, injection_site, p_dcs, i_dcs):

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

        f_network_params=f.create_group('network_params')
        for attr, value in wta_params.iteritems():
            f_network_params.attrs[attr] = value

        f_pyr_params=f.create_group('pyr_params')
        for attr, value in pyr_params.iteritems():
            f_pyr_params.attrs[attr] = value

        f_inh_params=f.create_group('inh_params')
        for attr, value in inh_params.iteritems():
            f_inh_params.attrs[attr] = value

        f.attrs['muscimol_amount'] = muscimol_amount
        f.attrs['injection_site'] = injection_site
        f.attrs['p_dcs']=p_dcs
        f.attrs['i_dcs']=i_dcs

        if not self.save_summary_only:
            # Write LFP data
            if self.record_lfp:
                f_lfp = f.create_group('lfp')
                f_lfp['lfp']=self.monitors['lfp'].values

            # Write voxel data
            if self.record_voxel:
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
                f_vox_total['G_total'] = self.monitors['voxel']['G_total'].values
                f_vox_total['s'] = self.monitors['voxel']['s'].values
                f_vox_total['f_in'] = self.monitors['voxel']['f_in'].values
                f_vox_total['v'] = self.monitors['voxel']['v'].values
                f_vox_total['q'] = self.monitors['voxel']['q'].values
                f_vox_total['y'] = self.monitors['voxel']['y'].values

                f_vox_exc=f_vox.create_group('exc_syn')
                f_vox_exc['G_total'] = self.monitors['voxel_exc']['G_total'].values
                f_vox_exc['s'] = self.monitors['voxel_exc']['s'].values
                f_vox_exc['f_in'] = self.monitors['voxel_exc']['f_in'].values
                f_vox_exc['v'] = self.monitors['voxel_exc']['v'].values
                f_vox_exc['q'] = self.monitors['voxel_exc']['q'].values
                f_vox_exc['y'] = self.monitors['voxel_exc']['y'].values

            # Write neuron state data
            if self.record_neuron_state:
                f_state = f.create_group('neuron_state')
                f_state['g_ampa_r'] = self.monitors['network']['g_ampa_r'].values
                f_state['g_ampa_x'] = self.monitors['network']['g_ampa_x'].values
                f_state['g_ampa_b'] = self.monitors['network']['g_ampa_b'].values
                f_state['g_nmda'] = self.monitors['network']['g_nmda'].values
                f_state['g_gaba_a'] = self.monitors['network']['g_gaba_a'].values
                #f_state['g_gaba_b'] = self.monitors['network']['g_gaba_b'].values
                f_state['I_ampa_r'] = self.monitors['network']['I_ampa_r'].values
                f_state['I_ampa_x'] = self.monitors['network']['I_ampa_x'].values
                f_state['I_ampa_b'] = self.monitors['network']['I_ampa_b'].values
                f_state['I_nmda'] = self.monitors['network']['I_nmda'].values
                f_state['I_gaba_a'] = self.monitors['network']['I_gaba_a'].values
                #f_state['I_gaba_b'] = self.monitors['network']['I_gaba_b'].values
                f_state['vm'] = self.monitors['network']['vm'].values
                f_state['record_idx'] = np.array(self.record_idx)

            # Write network firing rate data
            if self.record_firing_rate:
                f_rates = f.create_group('firing_rates')
                e_rates = []
                for i in range(num_groups):
                    e_rates.append(self.monitors['excitatory_rate_%d' % i].smooth_rate(width=5 * ms, filter='gaussian'))
                f_rates['e_rates'] = np.array(e_rates)

                i_rates = [self.monitors['inhibitory_rate'].smooth_rate(width=5 * ms, filter='gaussian')]
                f_rates['i_rates'] = np.array(i_rates)

            # Write input firing rate data
            if self.record_inputs:
                back_rate=f.create_group('background_rate')
                back_rate['firing_rate']=self.monitors['background_rate'].smooth_rate(width=5*ms,filter='gaussian')
                task_rates=f.create_group('task_rates')
                t_rates=[]
                for i in range(num_groups):
                    t_rates.append(self.monitors['task_rate_%d' % i].smooth_rate(width=5*ms,filter='gaussian'))
                task_rates['firing_rates']=np.array(t_rates)

            # Write spike data
            if self.record_spikes:
                f_spikes = f.create_group('spikes')
                for idx in range(num_groups):
                    spike_monitor=self.monitors['excitatory_spike_%d' % idx]
                    if len(spike_monitor.spikes):
                        f_spikes['e.%d.spike_neurons' % idx] = np.array([s[0] for s in spike_monitor.spikes])
                        f_spikes['e.%d.spike_times' % idx] = np.array([s[1] for s in spike_monitor.spikes])

                spike_monitor=self.monitors['inhibitory_spike']
                if len(spike_monitor.spikes):
                    f_spikes['i.spike_neurons'] = np.array([s[0] for s in spike_monitor.spikes])
                    f_spikes['i.spike_times'] = np.array([s[1] for s in spike_monitor.spikes])

            # Write connection data
            if self.record_connections:
                f_conns=f.create_group('connections')
                for mon in self.monitors:
                    if mon.startswith('connection_'):
                        conn_name=mon[11:]
                        f_conn=f_conns.create_group(conn_name)
                        for idx, (time, conn_matrix) in enumerate(self.monitors[mon].values):
                            f_conn['time_%.4f' % time]=conn_matrix.todense()

        else:
            f_summary=f.create_group('summary')
            endIdx=int(stim_end_time/self.clock.dt)
            startIdx=endIdx-500
            e_mean_final=[]
            e_max=[]
            for idx in range(num_groups):
                rate_monitor=self.monitors['excitatory_rate_%d' % idx]
                e_rate=rate_monitor.smooth_rate(width=5*ms, filter='gaussian')
                e_mean_final.append(np.mean(e_rate[startIdx:endIdx]))
                e_max.append(np.max(e_rate))
            rate_monitor=self.monitors['inhibitory_rate']
            i_rate=rate_monitor.smooth_rate(width=5*ms, filter='gaussian')
            i_mean_final=[np.mean(i_rate[startIdx:endIdx])]
            i_max=[np.max(i_rate)]
            f_summary['e_mean']=np.array(e_mean_final)
            f_summary['e_max']=np.array(e_max)
            f_summary['i_mean']=np.array(i_mean_final)
            f_summary['i_max']=np.array(i_max)
            f_summary['bold_max']=np.max(self.voxel_monitor['y'].values)
            f_summary['bold_exc_max']=np.max(self.voxel_exc_monitor['y'].values)

        f.close()