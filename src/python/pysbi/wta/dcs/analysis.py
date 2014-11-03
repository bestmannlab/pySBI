from scipy.optimize import curve_fit
import matplotlib
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg')
import pylab
from brian import second
import os
import subprocess
from brian.stdunits import Hz, ms, nA, mA
from jinja2 import Environment, FileSystemLoader
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.util.utils import save_to_png, save_to_eps, FitRT, FitWeibull, exp_decay, FitSigmoid
from pysbi.wta.analysis import TrialSeries, get_lfp_signal

condition_colors={
    'control': 'b',
    'anode':'r',
    'cathode':'g',
    }


class TrialReport:
    def __init__(self, trial_idx, trial_summary, report_dir, edesc, dt=.1*ms, version=None):
        self.trial_idx=trial_idx
        self.trial_summary=trial_summary
        self.report_dir=report_dir
        self.edesc=edesc
        self.dt=dt
        self.version=version
        if self.version is None:
            self.version=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])

        self.wta_params=self.trial_summary.data.wta_params
        self.pyr_params=self.trial_summary.data.pyr_params
        self.inh_params=self.trial_summary.data.inh_params
        self.voxel_params=self.trial_summary.data.voxel_params
        self.num_groups=self.trial_summary.data.num_groups
        self.trial_duration=self.trial_summary.data.trial_duration
        self.background_freq=self.trial_summary.data.background_freq
        self.stim_start_time=self.trial_summary.data.stim_start_time
        self.stim_end_time=self.trial_summary.data.stim_end_time
        self.network_group_size=self.trial_summary.data.network_group_size
        self.background_input_size=self.trial_summary.data.background_input_size
        self.task_input_size=self.trial_summary.data.task_input_size
        self.muscimol_amount=self.trial_summary.data.muscimol_amount
        self.injection_site=self.trial_summary.data.injection_site
        self.p_dcs=self.trial_summary.data.p_dcs
        self.i_dcs=self.trial_summary.data.i_dcs
            
    def create_report(self, regenerate_plots=True):
    
        self.firing_rate_url = None
        if self.trial_summary.data.e_firing_rates is not None and self.trial_summary.data.i_firing_rates is not None:
            furl = 'img/firing_rate.contrast.%0.4f.trial.%d' % (self.trial_summary.contrast,
                                                                self.trial_summary.trial_idx)
            fname = os.path.join(self.report_dir, furl)
            self.firing_rate_url = '%s.png' % furl

            if regenerate_plots:
                # figure out max firing rate of all neurons (pyramidal and interneuron)
                max_pop_rate=0
                for i, pop_rate in enumerate(self.trial_summary.data.e_firing_rates):
                    max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])
                for i, pop_rate in enumerate(self.trial_summary.data.i_firing_rates):
                    max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])

                fig=Figure()

                # Plot pyramidal neuron firing rate
                ax=fig.add_subplot(2,1,1)
                for i, pop_rate in enumerate(self.trial_summary.data.e_firing_rates):
                    ax.plot(np.array(range(len(pop_rate)))*self.dt, pop_rate / Hz, label='group %d' % i)
                    # Plot line showing RT
                if self.trial_summary.data.rt:
                    rt_idx=(1*second+self.trial_summary.data.rt*ms)/second
                    ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
                ax.set_ylim([0,10+max_pop_rate])
                ax.legend(loc=0)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Firing Rate (Hz)')

                # Plot interneuron firing rate
                ax = fig.add_subplot(2,1,2)
                for i, pop_rate in enumerate(self.trial_summary.data.i_firing_rates):
                    ax.plot(np.array(range(len(pop_rate)))*self.dt, pop_rate / Hz, label='group %d' % i)
                    # Plot line showing RT
                if self.trial_summary.data.rt:
                    rt_idx=(1*second+self.trial_summary.data.rt*ms)/second
                    ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
                ax.set_ylim([0,10+max_pop_rate])
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Firing Rate (Hz)')
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)
    
            #del self.trial_summary.data.e_firing_rates
            #del self.trial_summary.data.i_firing_rates

        self.neural_state_url=None
        if self.trial_summary.data.neural_state_rec is not None:
            furl = 'img/neural_state.contrast.%0.4f.trial.%d' % (self.trial_summary.contrast,
                                                                 self.trial_summary.trial_idx)
            fname = os.path.join(self.report_dir, furl)
            self.neural_state_url = '%s.png' % furl

            if regenerate_plots:
                fig = plt.figure()
                for i in range(self.trial_summary.data.num_groups):
                    times=np.array(range(len(self.trial_summary.data.neural_state_rec['g_ampa_r'][i*2])))*.1
                    ax = plt.subplot(self.trial_summary.data.num_groups * 100 + 20 + (i * 2 + 1))
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_ampa_r'][i * 2] / nA, label='AMPA-recurrent')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_ampa_x'][i * 2] / nA, label='AMPA-task')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_ampa_b'][i * 2] / nA, label='AMPA-backgrnd')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_nmda'][i * 2] / nA, label='NMDA')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_gaba_a'][i * 2] / nA, label='GABA_A')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Conductance (nA)')
                    ax = plt.subplot(self.trial_summary.data.num_groups * 100 + 20 + (i * 2 + 2))
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_ampa_r'][i * 2 + 1] / nA,
                        label='AMPA-recurrent')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_ampa_x'][i * 2 + 1] / nA, label='AMPA-task')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_ampa_b'][i * 2 + 1] / nA,
                        label='AMPA-backgrnd')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_nmda'][i * 2 + 1] / nA, label='NMDA')
                    ax.plot(times, self.trial_summary.data.neural_state_rec['g_gaba_a'][i * 2 + 1] / nA, label='GABA_A')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Conductance (nA)')
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)
            del self.trial_summary.data.neural_state_rec
    
        self.lfp_url = None
        if self.trial_summary.data.lfp_rec is not None:
            furl = 'img/lfp.contrast.%0.4f.trial.%d' % (self.trial_summary.contrast, self.trial_summary.trial_idx)
            fname = os.path.join(self.report_dir, furl)
            self.lfp_url = '%s.png' % furl
            if regenerate_plots:
                fig = plt.figure()
                ax = plt.subplot(111)
                lfp=get_lfp_signal(self.trial_summary.data)
                ax.plot(np.array(range(len(lfp))), lfp / mA)
                plt.xlabel('Time (ms)')
                plt.ylabel('LFP (mA)')
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)
            del self.trial_summary.data.lfp_rec


class SessionReport:
    def __init__(self, subj_id, stim_condition, data_dir, file_prefix, num_trials, contrast_range, report_dir, edesc, 
                 version=None, dt=.5*ms):
        self.subj_id=subj_id
        self.stim_condition=stim_condition
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.num_trials=num_trials
        self.contrast_range=contrast_range
        self.dt=dt
        self.report_dir=report_dir
        self.edesc=edesc
        self.version=version
        if self.version is None:
            self.version=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.trial_reports=[]

    def create_report(self, regenerate_plots=True, regenerate_trial_plots=True):
        
        make_report_dirs(self.report_dir)
    
        self.series=TrialSeries(self.data_dir, self.file_prefix, self.num_trials, contrast_range=self.contrast_range,
            upper_resp_threshold=25, lower_resp_threshold=None, dt=self.dt)
        self.series.sort_by_correct()

        for idx,trial_summary in enumerate(self.series.trial_summaries):
            trial_report=TrialReport(idx+1,trial_summary, self.report_dir, self.edesc, dt=self.dt, version=self.version)
            trial_report.create_report(regenerate_plots=regenerate_trial_plots)
            self.trial_reports.append(trial_report)

        self.wta_params=self.trial_reports[0].wta_params
        self.pyr_params=self.trial_reports[0].pyr_params
        self.inh_params=self.trial_reports[0].inh_params
        self.voxel_params=self.trial_reports[0].voxel_params
        self.num_groups=self.trial_reports[0].num_groups
        self.trial_duration=self.trial_reports[0].trial_duration
        self.background_freq=self.trial_reports[0].background_freq
        self.stim_start_time=self.trial_reports[0].stim_start_time
        self.stim_end_time=self.trial_reports[0].stim_end_time
        self.network_group_size=self.trial_reports[0].network_group_size
        self.background_input_size=self.trial_reports[0].background_input_size
        self.task_input_size=self.trial_reports[0].task_input_size
        self.muscimol_amount=self.trial_reports[0].muscimol_amount
        self.injection_site=self.trial_reports[0].injection_site
        self.p_dcs=self.trial_reports[0].p_dcs
        self.i_dcs=self.trial_reports[0].i_dcs

        furl='img/roc'
        fname=os.path.join(self.report_dir, furl)
        self.roc_url='%s.png' % furl
        if regenerate_plots:
            self.series.plot_multiclass_roc(filename=fname)
    
        furl='img/rt'
        fname=os.path.join(self.report_dir, furl)
        self.rt_url='%s.png' % furl
        if regenerate_plots:
            self.series.plot_rt(filename=fname)
    
        furl='img/perc_correct'
        fname=os.path.join(self.report_dir, furl)
        self.perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.series.plot_perc_correct(filename=fname)

        self.mean_rate_urls={}
        for contrast in self.contrast_range:
            furl='img/mean_firing_rate_%.4f' % contrast
            self.mean_rate_urls[contrast]='%s.png' % furl
            if regenerate_plots:
                self.plot_mean_firing_rate(furl, contrast, self.dt)

        #create report
        template_file='dcs_session.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)
    
        output_file='dcs_session.%s.html' % self.stim_condition
        fname=os.path.join(self.report_dir,output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

    def get_trial_firing_rates(self, coherence):
        chosen_rates=[]
        unchosen_rates=[]
        inh_rates=[]

        for idx,trial_summary in enumerate(self.series.trial_summaries):
            if trial_summary.contrast==coherence and trial_summary.decision_idx>-1:
                chosen_rates.append(trial_summary.data.e_firing_rates[trial_summary.decision_idx])
                unchosen_rates.append(trial_summary.data.e_firing_rates[1-trial_summary.decision_idx])
                inh_rates.append(trial_summary.data.i_firing_rates[0])

        chosen_rates=np.array(chosen_rates)
        unchosen_rates=np.array(unchosen_rates)
        inh_rates=np.array(inh_rates)

        return chosen_rates, unchosen_rates, inh_rates

    def plot_mean_firing_rate(self, furl, coherence, dt):
        fname=os.path.join(self.report_dir, furl)

        chosen_rates,unchosen_rates,inh_rates=self.get_trial_firing_rates(coherence)

        mean_chosen_rate=np.mean(chosen_rates,axis=0)
        std_chosen_rate=np.std(chosen_rates,axis=0)/np.sqrt(chosen_rates.shape[0])
        mean_unchosen_rate=np.mean(unchosen_rates,axis=0)
        std_unchosen_rate=np.std(unchosen_rates,axis=0)/np.sqrt(unchosen_rates.shape[0])
        mean_inh_rate=np.mean(inh_rates,axis=0)
        std_inh_rate=np.std(inh_rates,axis=0)/np.sqrt(inh_rates.shape[0])

        max_pop_rate=np.max([np.max(mean_chosen_rate), np.max(mean_unchosen_rate), np.max(mean_inh_rate)])

        # Plot pyramidal neuron firing rate
        fig=Figure()
        ax=fig.add_subplot(2,1,1)
        time_ticks=np.array(range(len(mean_chosen_rate)))*dt

        baseline,=ax.plot(time_ticks, mean_chosen_rate, color='b', linestyle='-', label='chosen')
        ax.fill_between(time_ticks, mean_chosen_rate-std_chosen_rate, mean_chosen_rate+std_chosen_rate, alpha=0.5,
            facecolor=baseline.get_color())

        baseline,=ax.plot(time_ticks, mean_unchosen_rate, color='r', linestyle='-', label='unchosen')
        ax.fill_between(time_ticks, mean_unchosen_rate-std_unchosen_rate, mean_unchosen_rate+std_unchosen_rate, alpha=0.5,
            facecolor=baseline.get_color())

        ax.set_ylim([0,10+max_pop_rate])
        ax.legend(loc=0)
        ax.set_title('Coherence=%.4f' % coherence)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')

        # Plot interneuron firing rate
        ax = fig.add_subplot(2,1,2)
        baseline,=ax.plot(time_ticks, mean_inh_rate, color='b', linestyle='-')
        ax.fill_between(time_ticks, mean_inh_rate-std_inh_rate, mean_inh_rate+std_inh_rate, alpha=0.5,
            facecolor=baseline.get_color())
        ax.set_ylim([0,10+max_pop_rate])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)


class SubjectReport:
    def __init__(self, data_dir, file_prefix, subj_id, stim_levels, num_trials, contrast_range, report_dir, edesc, 
                 version=None, dt=.5*ms):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.subj_id=subj_id
        self.stim_levels=stim_levels
        self.num_trials=num_trials
        self.contrast_range=contrast_range
        self.dt=dt
        self.report_dir=report_dir
        self.edesc=edesc
        self.version=version
        if self.version is None:
            self.version=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.sessions={}

    def create_report(self, regenerate_plots=True, regenerate_session_plots=True, regenerate_trial_plots=True):
        make_report_dirs(self.report_dir)

        for stim_level in self.stim_levels:
            print('creating %s report' % stim_level)
            p_dcs=self.stim_levels[stim_level][0]
            i_dcs=self.stim_levels[stim_level][1]
            stim_report_dir=os.path.join(self.report_dir,stim_level)
            prefix='%s.p_dcs.%.4f.i_dcs.%.4f.virtual_subject.%d.%s' % (self.file_prefix,p_dcs,i_dcs,self.subj_id,
                                                                       stim_level)
            self.sessions[stim_level]=SessionReport(self.subj_id, stim_level, self.data_dir,prefix, self.num_trials,
                self.contrast_range, stim_report_dir, self.edesc, version=self.version, dt=self.dt)
            self.sessions[stim_level].create_report(regenerate_plots=regenerate_session_plots,
                regenerate_trial_plots=regenerate_trial_plots)

        self.wta_params=self.sessions['control'].wta_params
        self.pyr_params=self.sessions['control'].pyr_params
        self.inh_params=self.sessions['control'].inh_params
        self.voxel_params=self.sessions['control'].voxel_params
        self.num_groups=self.sessions['control'].num_groups
        self.trial_duration=self.sessions['control'].trial_duration
        self.background_freq=self.sessions['control'].background_freq
        self.stim_start_time=self.sessions['control'].stim_start_time
        self.stim_end_time=self.sessions['control'].stim_end_time
        self.network_group_size=self.sessions['control'].network_group_size
        self.background_input_size=self.sessions['control'].background_input_size
        self.task_input_size=self.sessions['control'].task_input_size
        
        furl='img/rt'
        self.rt_url='%s.png' % furl
        if regenerate_plots:
            self.plot_rt(furl, condition_colors)

        furl='img/perc_correct'
        self.perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.plot_perc_correct(furl, condition_colors)

        self.mean_rate_urls={}
        for contrast in self.contrast_range:
            furl='img/mean_firing_rate_%.4f' % contrast
            self.mean_rate_urls[contrast]='%s.png' % furl
            if regenerate_plots:
                self.plot_mean_firing_rate(furl, contrast, self.dt, condition_colors)

        furl='img/coherence_prestim_bias'
        self.coherence_prestim_bias_url='%s.png' % furl
        if regenerate_plots:
            self.plot_coherence_prestim_bias(furl, self.dt, condition_colors)

        furl='img/bias_input_ratio_rt'
        self.bias_input_ratio_rt_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_input_diff_rt(furl, self.dt, condition_colors)

        furl='img/bias_input_ratio_perc_correct'
        self.bias_input_ratio_perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_input_diff_perc_correct(furl, self.dt, condition_colors)

        furl='img/bias_rt'
        self.bias_rt_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_rt(furl, self.dt, condition_colors)

        furl='img/bias_perc_correct'
        self.bias_perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_perc_correct(furl, self.dt, condition_colors)

        furl='img/bias_perc_left'
        self.bias_perc_left_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_perc_left(furl, self.dt, condition_colors)

        #create report
        template_file='dcs_subject.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        output_file='dcs_subject.%s.html' % self.subj_id
        fname=os.path.join(self.report_dir,output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

    def plot_rt(self, furl, colors):
        fname=os.path.join(self.report_dir, furl)

        fig=plt.figure()
        for stim_level, session_report in self.sessions.iteritems():
            contrast, mean_rt, std_rt = session_report.series.get_contrast_rt_stats()
            rt_fit = FitRT(np.array(contrast), mean_rt, guess=[1,1,1])
            smoothInt = pylab.arange(0.01, max(contrast), 0.001)
            smoothResp = rt_fit.eval(smoothInt)
            plt.errorbar(contrast, mean_rt,yerr=std_rt,fmt='o%s' % colors[stim_level])
            plt.plot(smoothInt, smoothResp, colors[stim_level], label=stim_level)

        plt.xlabel('Contrast')
        plt.ylabel('Decision time (ms)')
        plt.xscale('log')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)


    def plot_perc_correct(self, furl, colors):
        fname=os.path.join(self.report_dir, furl)

        fig=plt.figure()
        for stim_level, session_report in self.sessions.iteritems():
            contrast, perc_correct = session_report.series.get_contrast_perc_correct_stats()
            acc_fit=FitWeibull(contrast, perc_correct, guess=[0.2, 0.5])
            thresh = np.max([0,acc_fit.inverse(0.8)])
            smoothInt = pylab.arange(0.0, max(contrast), 0.001)
            smoothResp = acc_fit.eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '%s' % colors[stim_level], label=stim_level)
            plt.plot(contrast, perc_correct, 'o%s' % colors[stim_level])
            plt.plot([thresh,thresh],[0.4,1.0],'%s' % colors[stim_level])

        plt.xlabel('Contrast')
        plt.ylabel('% correct')
        plt.legend(loc='best')
        plt.xscale('log')
        #plt.ylim([0.4,1])
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)


    def get_mean_firing_rates(self, stim_level, coherence):
        chosen_rates,unchosen_rates,inh_rates=self.sessions[stim_level].get_trial_firing_rates(coherence)
        mean_chosen_rate=np.mean(chosen_rates,axis=0)
        mean_unchosen_rate=np.mean(unchosen_rates,axis=0)
        mean_inh_rate=np.mean(inh_rates,axis=0)
        return mean_chosen_rate,mean_unchosen_rate,mean_inh_rate

    def get_mean_prestim_bias(self, stim_level, coherence, dt):
        chosen_rates,unchosen_rates,inh_rates=self.sessions[stim_level].get_trial_firing_rates(coherence)
        biases=[]
        for trial in range(chosen_rates.shape[0]):
            biases.append(np.mean(chosen_rates[trial,int(500*ms/dt):int(950*ms/dt)])-np.mean(unchosen_rates[trial,int(500*ms/dt):int(950*ms/dt)]))
        return np.mean(biases)

    def plot_coherence_prestim_bias(self, furl, dt, colors):
        fname=os.path.join(self.report_dir, furl)
        mean_prestim_bias={}
        std_prestim_bias={}
        for stim_level, session_report in self.sessions.iteritems():
            mean_prestim_bias[stim_level]=[]
            std_prestim_bias[stim_level]=[]
            for coherence in self.contrast_range:
                chosen_rates,unchosen_rates,inh_rates=self.sessions[stim_level].get_trial_firing_rates(coherence)
                biases=[]
                for trial in range(chosen_rates.shape[0]):
                    biases.append(np.mean(chosen_rates[trial,int(500*ms/dt):int(950*ms/dt)])-np.mean(unchosen_rates[trial,int(500*ms/dt):int(950*ms/dt)]))
                mean_prestim_bias[stim_level].append(np.mean(biases))
                std_prestim_bias[stim_level].append(np.std(biases)/np.sqrt(len(biases)))
        fig=plt.figure()
        for stim_level in self.sessions:
            plt.errorbar(self.contrast_range, mean_prestim_bias[stim_level], yerr=std_prestim_bias[stim_level], fmt='o%s' % colors[stim_level])
            popt,pcov=curve_fit(exp_decay, np.array(self.contrast_range), np.array(mean_prestim_bias[stim_level]))
            y_hat=exp_decay(np.array(self.contrast_range),*popt)
            ybar=np.sum(np.array(mean_prestim_bias[stim_level]))/len(np.array(mean_prestim_bias[stim_level]))
            ssres=np.sum((np.array(mean_prestim_bias[stim_level])-y_hat)**2.0)
            sstot=np.sum((np.array(mean_prestim_bias[stim_level])-ybar)**2.0)
            r_sqr=1.0-ssres/sstot
            min_x=np.min(self.contrast_range)-.01
            max_x=np.max(self.contrast_range)+.01
            x_range=min_x+np.array(range(1000))*(max_x-min_x)/1000.0
            plt.plot(x_range,exp_decay(x_range,*popt),colors[stim_level],label='%s, r^2=%.3f' % (stim_level,r_sqr))
        plt.legend(loc='best')
        plt.xlabel('Coherence')
        plt.ylabel('Pyr Rate Diff (Hz)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_rt(self, furl, dt, colors):
        fname=os.path.join(self.report_dir, furl)
        all_biases=[]
        biases={}
        rts={}
        for stim_level, session_report in self.sessions.iteritems():
            biases[stim_level]=[]
            rts[stim_level]=[]
            for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                if trial_summary.data.rt is not None:
                    prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                        np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                    biases[stim_level].append(prestim_bias)
                    rts[stim_level].append(trial_summary.data.rt)
                    all_biases.append(prestim_bias)

        hist,bins=np.histogram(all_biases, bins=10)

        fig=plt.figure()
        for stim_level in self.sessions:
            mean_biases=[]
            mean_rts=[]
            for i in range(10):
                bin_rts=[]
                bin_biases=[]
                for bias,rt in zip(biases[stim_level],rts[stim_level]):
                    if bias>=bins[i] and bias<bins[i+1]:
                        bin_rts.append(rt)
                        bin_biases.append(bias)
                if len(bin_biases):
                    mean_biases.append(np.mean(bin_biases))
                    mean_rts.append(np.mean(bin_rts))
            plt.plot(mean_biases,mean_rts,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias')
        plt.ylabel('RT')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_perc_correct(self, furl, dt, colors):
        fname=os.path.join(self.report_dir, furl)
        all_biases=[]
        biases={}
        responses={}
        for stim_level, session_report in self.sessions.iteritems():
            biases[stim_level]=[]
            responses[stim_level]=[]
            for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                if trial_summary.data.rt is not None:
                    prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                        np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                    biases[stim_level].append(prestim_bias)
                    if trial_summary.correct:
                        responses[stim_level].append(1.0)
                    else:
                        responses[stim_level].append(0.0)
                    all_biases.append(prestim_bias)

        hist,bins=np.histogram(all_biases, bins=10)

        fig=plt.figure()
        for stim_level in self.sessions:
            mean_bias=[]
            mean_perc_correct=[]
            for i in range(10):
                bin_responses=[]
                bin_biases=[]
                for bias,response in zip(biases[stim_level],responses[stim_level]):
                    if bias>=bins[i] and bias<bins[i+1]:
                        bin_responses.append(response)
                        bin_biases.append(bias)
                if len(bin_biases):
                    mean_bias.append(np.mean(bin_biases))
                    mean_perc_correct.append(np.mean(bin_responses))
            plt.plot(mean_bias,mean_perc_correct,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias')
        plt.ylabel('% correct')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_perc_left(self, furl, dt, colors):
        fname=os.path.join(self.report_dir, furl)
        all_biases=[]
        biases={}
        responses={}
        for stim_level, session_report in self.sessions.iteritems():
            biases[stim_level]=[]
            responses[stim_level]=[]
            for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                if trial_summary.data.rt is not None:
                    prestim_bias=np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-\
                                 np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)])
                    biases[stim_level].append(prestim_bias)
                    if trial_summary.decision_idx==0:
                        responses[stim_level].append(1.0)
                    else:
                        responses[stim_level].append(0.0)
                    all_biases.append(prestim_bias)

        hist,bins=np.histogram(all_biases, bins=10)

        fig=plt.figure()
        for stim_level in biases:
            mean_bias=[]
            mean_perc_left=[]
            for i in range(10):
                bin_responses=[]
                bin_biases=[]
                for bias,response in zip(biases[stim_level],responses[stim_level]):
                    if bias>=bins[i] and bias<bins[i+1]:
                        bin_responses.append(response)
                        bin_biases.append(bias)
                if len(bin_biases):
                    mean_bias.append(np.mean(bin_biases))
                    mean_perc_left.append(np.mean(bin_responses))
            plt.plot(mean_bias,mean_perc_left,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias')
        plt.ylabel('% left')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_input_diff_rt(self, furl, dt, colors):
        fname=os.path.join(self.report_dir, furl)
        all_ratios=[]
        ratios={}
        rts={}
        for stim_level, session_report in self.sessions.iteritems():
            ratios[stim_level]=[]
            rts[stim_level]=[]
            for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                if trial_summary.data.rt is not None:
                    prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                        np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                    input_diff=np.abs(trial_summary.data.input_freq[1] - trial_summary.data.input_freq[0])
                    if input_diff>0:
                        ratios[stim_level].append(prestim_bias/input_diff)
                        rts[stim_level].append(trial_summary.data.rt)
                        all_ratios.append(prestim_bias/input_diff)

        hist,bins=np.histogram(all_ratios, bins=10)

        fig=plt.figure()
        for stim_level in self.sessions:
            mean_ratios=[]
            mean_rts=[]
            for i in range(10):
                bin_rts=[]
                bin_ratios=[]
                for ratio,rt in zip(ratios[stim_level],rts[stim_level]):
                    if ratio>=bins[i] and ratio<bins[i+1]:
                        bin_rts.append(rt)
                        bin_ratios.append(ratio)
                if len(bin_ratios):
                    mean_ratios.append(np.mean(bin_ratios))
                    mean_rts.append(np.mean(bin_rts))
            plt.plot(mean_ratios,mean_rts,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias/Input Diff')
        plt.ylabel('RT')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_input_diff_perc_correct(self, furl, dt, colors):
        fname=os.path.join(self.report_dir, furl)
        all_ratios=[]
        ratios={}
        responses={}
        for stim_level, session_report in self.sessions.iteritems():
            ratios[stim_level]=[]
            responses[stim_level]=[]
            for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                if trial_summary.data.rt is not None:
                    prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                        np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                    input_diff=np.abs(trial_summary.data.input_freq[1] - trial_summary.data.input_freq[0])
                    if input_diff>0:
                        ratios[stim_level].append(prestim_bias/input_diff)
                        if trial_summary.correct:
                            responses[stim_level].append(1.0)
                        else:
                            responses[stim_level].append(0.0)
                        all_ratios.append(prestim_bias/input_diff)

        hist,bins=np.histogram(all_ratios, bins=10)

        fig=plt.figure()
        for stim_level in self.sessions:
            mean_ratios=[]
            mean_perc_correct=[]
            for i in range(10):
                bin_responses=[]
                bin_ratios=[]
                for ratio,response in zip(ratios[stim_level],responses[stim_level]):
                    if ratio>=bins[i] and ratio<bins[i+1]:
                        bin_responses.append(response)
                        bin_ratios.append(ratio)
                if len(bin_ratios):
                    mean_ratios.append(np.mean(bin_ratios))
                    mean_perc_correct.append(np.mean(bin_responses))
            plt.plot(mean_ratios,mean_perc_correct,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias/Input Diff')
        plt.ylabel('% correct')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_mean_firing_rate(self, furl, coherence, dt,  colors):
        fname=os.path.join(self.report_dir, furl)
        chosen_rates={}
        unchosen_rates={}
        inh_rates={}
        max_pop_rate=0
        for stim_level, session_report in self.sessions.iteritems():
            chosen_rates[stim_level],unchosen_rates[stim_level],inh_rates[stim_level]=session_report.get_trial_firing_rates(coherence)
            max_pop_rate=np.max([np.max(np.mean(chosen_rates[stim_level],axis=0)),
                                 np.max(np.mean(unchosen_rates[stim_level],axis=0)),
                                 np.max(np.mean(inh_rates[stim_level],axis=0)),max_pop_rate])


        fig=Figure()
        ax=fig.add_subplot(2,1,1)
        for stim_level in self.sessions:
            mean_chosen_rate=np.mean(chosen_rates[stim_level],axis=0)
            std_chosen_rate=np.std(chosen_rates[stim_level],axis=0)/np.sqrt(chosen_rates[stim_level].shape[0])
            mean_unchosen_rate=np.mean(unchosen_rates[stim_level],axis=0)
            std_unchosen_rate=np.std(unchosen_rates[stim_level],axis=0)/np.sqrt(unchosen_rates[stim_level].shape[0])

            time_ticks=np.array(range(len(mean_chosen_rate)))*dt

            baseline,=ax.plot(time_ticks, mean_chosen_rate, color=colors[stim_level], linestyle='-', label='%s - chosen' % stim_level)
            ax.fill_between(time_ticks, mean_chosen_rate-std_chosen_rate, mean_chosen_rate+std_chosen_rate, alpha=0.5,
                facecolor=baseline.get_color())

            baseline,=ax.plot(time_ticks, mean_unchosen_rate, color=colors[stim_level], linestyle='--', label='%s - unchosen' % stim_level)
            ax.fill_between(time_ticks, mean_unchosen_rate-std_unchosen_rate, mean_unchosen_rate+std_unchosen_rate, alpha=0.5,
                facecolor=baseline.get_color())
        ax.set_ylim([0,10+max_pop_rate])
        ax.legend(loc=0)
        ax.set_title('Coherence=%.4f' % coherence)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')

        ax = fig.add_subplot(2,1,2)
        for stim_level in self.sessions:
            mean_inh_rate=np.mean(inh_rates[stim_level],axis=0)
            std_inh_rate=np.std(inh_rates[stim_level],axis=0)/np.sqrt(inh_rates[stim_level].shape[0])

            time_ticks=np.array(range(len(mean_inh_rate)))*dt

            baseline,=ax.plot(time_ticks, mean_inh_rate, color=colors[stim_level], linestyle='-', label=stim_level)
            ax.fill_between(time_ticks, mean_inh_rate-std_inh_rate, mean_inh_rate+std_inh_rate, alpha=0.5,
                facecolor=baseline.get_color())
        ax.set_ylim([0,10+max_pop_rate])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)


class DCSComparisonReport:
    def __init__(self, data_dir, file_prefix, virtual_subj_ids, stim_levels, num_trials, reports_dir, edesc, dt=.5*ms):
        """
        Create report for DCS simulations
        data_dir=directory where datafiles are stored
        file_prefix=prefix of data files (before p_dcs param)
        stim_levels= dict of conditions - key is name, value is tuple of stimulation levels in pA of pyramidal and interneurons
            i.e. {'control':(0,0),'anode':(4,-2),'cathode':(-4,2)}
        num_trials=number of trials in each condition
        reports_dir=directory to put reports in
        edesc=extra description
        """
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.virtual_subj_ids=virtual_subj_ids
        self.stim_levels=stim_levels
        self.num_trials=num_trials
        self.contrast_range=(0.0, .016, .032, .064, .096, .128, .256, .512)
        self.dt=dt
        self.reports_dir=reports_dir
        self.edesc=edesc
        self.params={}

        self.subjects={}

    def create_report(self, regenerate_plots=True, regenerate_subject_plots=True, regenerate_session_plots=True,
                      regenerate_trial_plots=True):
        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])

        for virtual_subj_id in self.virtual_subj_ids:
            print('Creating report for subject %d' % virtual_subj_id)
            subj_report_dir=os.path.join(self.reports_dir,'virtual_subject.%d' % virtual_subj_id)
            self.subjects[virtual_subj_id]=SubjectReport(self.data_dir, self.file_prefix, virtual_subj_id,
                self.stim_levels, self.num_trials, self.contrast_range, subj_report_dir, self.edesc, 
                version=self.version, dt=self.dt)
            self.subjects[virtual_subj_id].create_report(regenerate_plots=regenerate_subject_plots,
                regenerate_session_plots=regenerate_session_plots, regenerate_trial_plots=regenerate_trial_plots)

        furl='img/rt'
        self.rt_url='%s.png' % furl
        if regenerate_plots:
            self.plot_rt(furl, condition_colors)

        furl='img/rt_diff_bar'
        self.rt_diff_bar_url='%s.png' % furl
        if regenerate_plots:
            self.plot_rt_diff_bar(furl, condition_colors)

        furl='img/rt_diff'
        self.rt_diff_url='%s.png' % furl
        if regenerate_plots:
            self.plot_rt_diff(furl, condition_colors)

        furl='img/perc_correct'
        self.perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.plot_perc_correct(furl, condition_colors)

        self.mean_rate_urls={}
        for contrast in self.contrast_range:
            furl='img/mean_firing_rate_%.4f' % contrast
            self.mean_rate_urls[contrast]='%s.png' % furl
            if regenerate_plots:
                self.plot_mean_firing_rate(furl, contrast, self.dt, condition_colors)

        furl='img/coherence_prestim_bias'
        self.coherence_prestim_bias_url='%s.png' % furl
        if regenerate_plots:
            self.plot_coherence_prestim_bias(furl, self.dt, condition_colors)

        furl='img/bias_input_ratio_rt'
        self.bias_input_ratio_rt_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_input_diff_rt(furl, self.dt, condition_colors)

        furl='img/bias_input_ratio_perc_correct'
        self.bias_input_ratio_perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_input_diff_perc_correct(furl, self.dt, condition_colors)

        furl='img/bias_rt'
        self.bias_rt_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_rt(furl, self.dt, condition_colors)

        furl='img/bias_perc_correct'
        self.bias_perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_perc_correct(furl, self.dt, condition_colors)

        furl='img/bias_perc_left'
        self.bias_perc_left_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_perc_left(furl, self.dt, condition_colors)

        furl='img/bias_bar'
        self.bias_bar_url='%s.png' % furl
        if regenerate_plots:
            self.plot_bias_bar(furl, self.dt, condition_colors)

        self.wta_params=self.subjects[self.subjects.keys()[0]].wta_params
        self.pyr_params=self.subjects[self.subjects.keys()[0]].pyr_params
        self.inh_params=self.subjects[self.subjects.keys()[0]].inh_params
        self.voxel_params=self.subjects[self.subjects.keys()[0]].voxel_params
        self.num_groups=self.subjects[self.subjects.keys()[0]].num_groups
        self.trial_duration=self.subjects[self.subjects.keys()[0]].trial_duration
        self.stim_start_time=self.subjects[self.subjects.keys()[0]].stim_start_time
        self.stim_end_time=self.subjects[self.subjects.keys()[0]].stim_end_time
        self.network_group_size=self.subjects[self.subjects.keys()[0]].network_group_size
        self.background_input_size=self.subjects[self.subjects.keys()[0]].background_input_size
        self.task_input_size=self.subjects[self.subjects.keys()[0]].task_input_size
        
        #create report
        template_file='dcs.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        output_file='dcs.html'
        fname=os.path.join(self.reports_dir,output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

    def plot_rt_diff_bar(self, furl, colors):
        fname=os.path.join(self.reports_dir, furl)
        fig=plt.figure()
        mean_anode_rt_diffs=[]
        mean_cathode_rt_diffs=[]
        for subj_report in self.subjects.itervalues():
            control_contrast,control_mean_rt,control_std_rt=subj_report.sessions['control'].series.get_contrast_rt_stats()
            anode_contrast,anode_mean_rt,anode_std_rt=subj_report.sessions['anode'].series.get_contrast_rt_stats()
            cathode_contrast,cathode_mean_rt,cathode_std_rt=subj_report.sessions['cathode'].series.get_contrast_rt_stats()
            anode_rt_diffs=[]
            cathode_rt_diffs=[]
            for idx in range(len(control_contrast)):
                anode_rt_diffs.append(anode_mean_rt[idx]-control_mean_rt[idx])
                cathode_rt_diffs.append(cathode_mean_rt[idx]-control_mean_rt[idx])
            mean_anode_rt_diffs.append(np.mean(anode_rt_diffs))
            mean_cathode_rt_diffs.append(np.mean(cathode_rt_diffs))
        anode_rt_hist,anode_rt_bins=np.histogram(np.array(mean_anode_rt_diffs), bins=5)
        bin_width=anode_rt_bins[1]-anode_rt_bins[0]
        bars=plt.bar(anode_rt_bins[:-1],anode_rt_hist/float(len(mean_anode_rt_diffs)),width=bin_width, label='anode')
        for bar in bars:
            bar.set_color('r')
        cathode_rt_hist,cathode_rt_bins=np.histogram(np.array(mean_cathode_rt_diffs), bins=5)
        bin_width=cathode_rt_bins[1]-cathode_rt_bins[0]
        bars=plt.bar(cathode_rt_bins[:-1],cathode_rt_hist/float(len(mean_cathode_rt_diffs)),width=bin_width, label='cathode')
        for bar in bars:
            bar.set_color('g')
        plt.legend(loc='best')
        plt.xlim([-175,175])
        plt.xlabel('Mean RT Diff')
        plt.ylabel('Proportion of subjects')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_rt_diff(self, furl, colors):
        fname=os.path.join(self.reports_dir, furl)
        fig=plt.figure()
        anode_coherence_rt_diff={}
        cathode_coherence_rt_diff={}
        for subj_report in self.subjects.itervalues():
            control_contrast,control_mean_rt,control_std_rt=subj_report.sessions['control'].series.get_contrast_rt_stats()
            anode_contrast,anode_mean_rt,anode_std_rt=subj_report.sessions['anode'].series.get_contrast_rt_stats()
            cathode_contrast,cathode_mean_rt,cathode_std_rt=subj_report.sessions['cathode'].series.get_contrast_rt_stats()
            for idx in range(len(control_contrast)):
                if not control_contrast[idx] in anode_coherence_rt_diff:
                    anode_coherence_rt_diff[control_contrast[idx]]=[]
                    cathode_coherence_rt_diff[control_contrast[idx]]=[]
                anode_coherence_rt_diff[control_contrast[idx]].append(anode_mean_rt[idx]-control_mean_rt[idx])
                cathode_coherence_rt_diff[control_contrast[idx]].append(cathode_mean_rt[idx]-control_mean_rt[idx])
        anode_rt_diff_mean=[]
        cathode_rt_diff_mean=[]
        anode_rt_diff_std=[]
        cathode_rt_diff_std=[]
        for idx in range(len(control_contrast)):
            anode_rt_diff_mean.append(np.mean(anode_coherence_rt_diff[control_contrast[idx]]))
            anode_rt_diff_std.append(np.std(anode_coherence_rt_diff[control_contrast[idx]])/np.sqrt(len(anode_coherence_rt_diff[control_contrast[idx]])))
            cathode_rt_diff_mean.append(np.mean(cathode_coherence_rt_diff[control_contrast[idx]]))
            cathode_rt_diff_std.append(np.std(cathode_coherence_rt_diff[control_contrast[idx]])/np.sqrt(len(cathode_coherence_rt_diff[control_contrast[idx]])))

        min_x=control_contrast[1]
        max_x=control_contrast[-1]

        clf = LinearRegression()
        clf.fit(np.reshape(np.array(control_contrast[1:]), (len(control_contrast[1:]),1)),
            np.reshape(np.array(anode_rt_diff_mean[1:]), (len(anode_rt_diff_mean[1:]),1)))
        anode_a = clf.coef_[0][0]
        anode_b = clf.intercept_[0]
        anode_r_sqr=clf.score(np.reshape(np.array(control_contrast[1:]), (len(control_contrast[1:]),1)),
            np.reshape(np.array(anode_rt_diff_mean[1:]), (len(anode_rt_diff_mean[1:]),1)))
        plt.plot([min_x, max_x], [anode_a * min_x + anode_b, anode_a * max_x + anode_b], '--r',
            label='r^2=%.3f' % anode_r_sqr)
        clf = LinearRegression()
        clf.fit(np.reshape(np.array(control_contrast[1:]),(len(control_contrast[1:]),1)),
            np.reshape(np.array(cathode_rt_diff_mean[1:]),(len(cathode_rt_diff_mean[1:]),1)))
        cathode_a = clf.coef_[0][0]
        cathode_b = clf.intercept_[0]
        cathode_r_sqr=clf.score(np.reshape(np.array(control_contrast[1:]), (len(control_contrast[1:]),1)),
            np.reshape(np.array(cathode_rt_diff_mean[1:]), (len(cathode_rt_diff_mean[1:]),1)))
        plt.plot([min_x, max_x], [cathode_a * min_x + cathode_b, cathode_a * max_x + cathode_b], '--g',
            label='r^2=%.3f' % cathode_r_sqr)

        plt.errorbar(control_contrast,anode_rt_diff_mean,yerr=anode_rt_diff_std,fmt='or')
        plt.errorbar(control_contrast,cathode_rt_diff_mean,yerr=cathode_rt_diff_std,fmt='og')
        plt.legend(loc='best')
        plt.xscale('log')
        plt.xlabel('Coherence')
        plt.ylabel('RT Diff')
        plt.ylim([-75,75])
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

            
    def plot_rt(self, furl, colors):
        fname=os.path.join(self.reports_dir, furl)

        fig=plt.figure()
        condition_contrast={}
        condition_rt={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                contrast, mean_rt, std_rt = session_report.series.get_contrast_rt_stats()
                if not stim_level in condition_contrast:
                    condition_contrast[stim_level]=contrast
                if not stim_level in condition_rt:
                    condition_rt[stim_level]=[]
                condition_rt[stim_level].append(mean_rt)

        for condition, contrast in condition_contrast.iteritems():
            mean_rt=np.mean(np.array(condition_rt[condition]),axis=0)
            std_rt=np.std(np.array(condition_rt[condition]),axis=0)/np.sqrt(len(self.subjects))
            rt_fit = FitRT(np.array(contrast), mean_rt, guess=[-550,3,600])
            if not condition in self.params:
                self.params[condition]={}
            self.params[condition]['a']=rt_fit.params[0]
            self.params[condition]['k']=rt_fit.params[1]
            self.params[condition]['tr']=rt_fit.params[2]
            smoothInt = pylab.arange(0.01, max(contrast), 0.001)
            smoothResp = rt_fit.eval(smoothInt)
            plt.errorbar(contrast, mean_rt,yerr=std_rt,fmt='o%s' % colors[condition])
            plt.plot(smoothInt, smoothResp, colors[condition], label=condition)

        plt.ylim([155,710])
        plt.xlabel('Contrast')
        plt.ylabel('Decision time (ms)')
        plt.xscale('log')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)


    def plot_perc_correct(self, furl, colors):
        fname=os.path.join(self.reports_dir, furl)

        fig=plt.figure()
        condition_contrast={}
        condition_perc_correct={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                contrast, perc_correct = session_report.series.get_contrast_perc_correct_stats()
                if not stim_level in condition_contrast:
                    condition_contrast[stim_level]=contrast
                if not stim_level in condition_perc_correct:
                    condition_perc_correct[stim_level]=[]
                condition_perc_correct[stim_level].append(perc_correct)

        for condition, contrast in condition_contrast.iteritems():
            mean_perc_correct=np.mean(np.array(condition_perc_correct[condition]),axis=0)
            std_perc_correct=np.std(np.array(condition_perc_correct[condition]),axis=0)/np.sqrt(len(self.subjects))
            acc_fit=FitWeibull(contrast, mean_perc_correct, guess=[0.08, 1.3])
            if not condition in self.params:
                self.params[condition]={}
            self.params[condition]['alpha']=acc_fit.params[0]
            self.params[condition]['beta']=acc_fit.params[1]
            thresh = np.max([0,acc_fit.inverse(0.8)])
            smoothInt = pylab.arange(0.01, max(contrast), 0.001)
            smoothResp = acc_fit.eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '%s' % colors[condition], label=condition)
            plt.errorbar(contrast, mean_perc_correct,yerr=std_perc_correct,fmt='o%s' % colors[condition])
            plt.plot([thresh,thresh],[0.4,1.0],'%s' % colors[condition])

        plt.xlabel('Contrast')
        plt.ylabel('% correct')
        plt.legend(loc='best')
        plt.xscale('log')
        #plt.ylim([0.4,1])
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_mean_firing_rate(self, furl, coherence, dt,  colors):
        fname=os.path.join(self.reports_dir, furl)
        chosen_rates={}
        unchosen_rates={}
        inh_rates={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                if not stim_level in chosen_rates:
                    chosen_rates[stim_level]=[]
                    unchosen_rates[stim_level]=[]
                    inh_rates[stim_level]=[]
                chosen_mean_rate,unchosen_mean_rate,inh_mean_rate=subj_report.get_mean_firing_rates(stim_level,coherence)
                chosen_rates[stim_level].append(chosen_mean_rate)
                unchosen_rates[stim_level].append(unchosen_mean_rate)
                inh_rates[stim_level].append(inh_mean_rate)

        max_pop_rate=0
        for stim_level in chosen_rates:
            chosen_rates[stim_level]=np.array(chosen_rates[stim_level])
            unchosen_rates[stim_level]=np.array(unchosen_rates[stim_level])
            inh_rates[stim_level]=np.array(inh_rates[stim_level])
            max_pop_rate=np.max([np.max(np.mean(chosen_rates[stim_level],axis=0)),
                                 np.max(np.mean(unchosen_rates[stim_level],axis=0)),
                                 np.max(np.mean(inh_rates[stim_level],axis=0)),max_pop_rate])

        fig=Figure()
        ax=fig.add_subplot(2,1,1)
        for stim_level in chosen_rates:
            mean_chosen_rate=np.mean(chosen_rates[stim_level],axis=0)
            std_chosen_rate=np.std(chosen_rates[stim_level],axis=0)/np.sqrt(chosen_rates[stim_level].shape[0])
            mean_unchosen_rate=np.mean(unchosen_rates[stim_level],axis=0)
            std_unchosen_rate=np.std(unchosen_rates[stim_level],axis=0)/np.sqrt(unchosen_rates[stim_level].shape[0])

            time_ticks=np.array(range(len(mean_chosen_rate)))*dt

            baseline,=ax.plot(time_ticks, mean_chosen_rate, color=colors[stim_level], linestyle='-', label='%s - chosen' % stim_level)
            ax.fill_between(time_ticks, mean_chosen_rate-std_chosen_rate, mean_chosen_rate+std_chosen_rate, alpha=0.5,
                facecolor=baseline.get_color())

            baseline,=ax.plot(time_ticks, mean_unchosen_rate, color=colors[stim_level], linestyle='--', label='%s - unchosen' % stim_level)
            ax.fill_between(time_ticks, mean_unchosen_rate-std_unchosen_rate, mean_unchosen_rate+std_unchosen_rate, alpha=0.5,
                facecolor=baseline.get_color())
        ax.set_ylim([0,10+max_pop_rate])
        ax.legend(loc=0)
        ax.set_title('Coherence=%.4f' % coherence)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')

        ax = fig.add_subplot(2,1,2)
        for stim_level in inh_rates:
            mean_inh_rate=np.mean(inh_rates[stim_level],axis=0)
            std_inh_rate=np.std(inh_rates[stim_level],axis=0)/np.sqrt(inh_rates[stim_level].shape[0])

            time_ticks=np.array(range(len(mean_inh_rate)))*dt

            baseline,=ax.plot(time_ticks, mean_inh_rate, color=colors[stim_level], linestyle='-', label=stim_level)
            ax.fill_between(time_ticks, mean_inh_rate-std_inh_rate, mean_inh_rate+std_inh_rate, alpha=0.5,
                facecolor=baseline.get_color())
        ax.set_ylim([0,10+max_pop_rate])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_coherence_prestim_bias(self, furl, dt, colors):
        fname=os.path.join(self.reports_dir, furl)
        mean_prestim_bias={}
        std_prestim_bias={}
        for contrast in self.contrast_range:
            contrast_biases={}
            for subj_report in self.subjects.itervalues():
                for stim_level, session_report in subj_report.sessions.iteritems():
                    if not stim_level in contrast_biases:
                        contrast_biases[stim_level]=[]
                    contrast_biases[stim_level].append(subj_report.get_mean_prestim_bias(stim_level, contrast, dt))
            for stim_level in contrast_biases:
                if not stim_level in mean_prestim_bias:
                    mean_prestim_bias[stim_level]=[]
                    std_prestim_bias[stim_level]=[]
                mean_prestim_bias[stim_level].append(np.mean(contrast_biases[stim_level]))
                std_prestim_bias[stim_level].append(np.std(contrast_biases[stim_level])/np.sqrt(len(contrast_biases[stim_level])))

        fig=plt.figure()
        for stim_level in mean_prestim_bias:
            plt.errorbar(self.contrast_range, mean_prestim_bias[stim_level], yerr=std_prestim_bias[stim_level], fmt='o%s' % colors[stim_level])
            popt,pcov=curve_fit(exp_decay, np.array(self.contrast_range), np.array(mean_prestim_bias[stim_level]))
            y_hat=exp_decay(np.array(self.contrast_range),*popt)
            ybar=np.sum(np.array(mean_prestim_bias[stim_level]))/len(np.array(mean_prestim_bias[stim_level]))
            ssres=np.sum((np.array(mean_prestim_bias[stim_level])-y_hat)**2.0)
            sstot=np.sum((np.array(mean_prestim_bias[stim_level])-ybar)**2.0)
            r_sqr=1.0-ssres/sstot
            min_x=np.min(self.contrast_range)-.01
            max_x=np.max(self.contrast_range)+.01
            x_range=min_x+np.array(range(1000))*(max_x-min_x)/1000.0
            plt.plot(x_range,exp_decay(x_range,*popt),colors[stim_level],label='%s, r^2=%.3f' % (stim_level,r_sqr))

        plt.legend(loc='best')
        plt.xlabel('Coherence')
        plt.ylabel('Pyr Rate Diff (Hz)')
        plt.xlim([-.01,0.6])
        plt.ylim([-.25,1.75])
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_input_diff_rt(self, furl, dt, colors):
        fname=os.path.join(self.reports_dir, furl)
        ratios={}
        rts={}
        all_ratios=[]
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                ratios[stim_level]=[]
                rts[stim_level]=[]
                for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                    if trial_summary.data.rt is not None:
                        prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                            np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                        input_diff=np.abs(trial_summary.data.input_freq[1] - trial_summary.data.input_freq[0])
                        if input_diff>0:
                            ratios[stim_level].append(prestim_bias/input_diff)
                            rts[stim_level].append(trial_summary.data.rt)
                            all_ratios.append(prestim_bias/input_diff)
        fig=plt.figure()
        hist,bins=np.histogram(all_ratios, bins=10)

        fig=plt.figure()
        for stim_level in ratios:
            mean_ratios=[]
            mean_rts=[]
            for i in range(10):
                bin_rts=[]
                bin_ratios=[]
                for ratio,rt in zip(ratios[stim_level],rts[stim_level]):
                    if ratio>=bins[i] and ratio<bins[i+1]:
                        bin_rts.append(rt)
                        bin_ratios.append(ratio)
                if len(bin_ratios):
                    mean_ratios.append(np.mean(bin_ratios))
                    mean_rts.append(np.mean(bin_rts))
            plt.plot(mean_ratios,mean_rts,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias/Input Diff')
        plt.ylabel('RT')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_input_diff_perc_correct(self, furl, dt, colors):
        fname=os.path.join(self.reports_dir, furl)
        all_ratios=[]
        ratios={}
        responses={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                ratios[stim_level]=[]
                responses[stim_level]=[]
                for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                    if trial_summary.data.rt is not None:
                        prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                            np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                        input_diff=np.abs(trial_summary.data.input_freq[1] - trial_summary.data.input_freq[0])
                        if input_diff>0:
                            ratios[stim_level].append(prestim_bias/input_diff)
                            if trial_summary.correct:
                                responses[stim_level].append(1.0)
                            else:
                                responses[stim_level].append(0.0)
                            all_ratios.append(prestim_bias/input_diff)

        hist,bins=np.histogram(all_ratios, bins=10)

        fig=plt.figure()
        for stim_level in ratios:
            mean_ratios=[]
            mean_perc_correct=[]
            for i in range(10):
                bin_responses=[]
                bin_ratios=[]
                for ratio,response in zip(ratios[stim_level],responses[stim_level]):
                    if ratio>=bins[i] and ratio<bins[i+1]:
                        bin_responses.append(response)
                        bin_ratios.append(ratio)
                if len(bin_ratios):
                    mean_ratios.append(np.mean(bin_ratios))
                    mean_perc_correct.append(np.mean(bin_responses))
            plt.plot(mean_ratios,mean_perc_correct,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias/Input Diff')
        plt.ylabel('% correct')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_rt(self, furl, dt, colors):
        fname=os.path.join(self.reports_dir, furl)
        condition_biases={}
        condition_rts={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                if not stim_level in condition_biases:
                    condition_biases[stim_level]=[]
                    condition_rts[stim_level]=[]
                for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                    if trial_summary.data.rt is not None:
                        prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                            np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                        condition_biases[stim_level].append(prestim_bias)
                        condition_rts[stim_level].append(trial_summary.data.rt)

        fig=plt.figure()
        mean_condition_biases={}
        mean_condition_rts={}
        std_condition_rts={}
        for condition in condition_biases:
            if not condition in mean_condition_biases:
                mean_condition_biases[condition]=[]
                mean_condition_rts[condition]=[]
                std_condition_rts[condition]=[]
            hist,bins=np.histogram(condition_biases[condition], bins=10)
            for i in range(10):
                bin_rts=[]
                bin_biases=[]
                for bias,rt in zip(condition_biases[condition],condition_rts[condition]):
                    if bias>=bins[i] and bias<bins[i+1]:
                        bin_biases.append(bias)
                        bin_rts.append(rt)
                if len(bin_biases)>=10:
                    mean_condition_biases[condition].append(np.mean(bin_biases))
                    mean_condition_rts[condition].append(np.mean(bin_rts))
                    std_condition_rts[condition].append(np.std(bin_rts)/np.sqrt(len(bin_rts)))

        for condition in mean_condition_biases:
            plt.errorbar(mean_condition_biases[condition],mean_condition_rts[condition],
                yerr=std_condition_rts[condition], fmt='o%s' % colors[condition])

            clf = LinearRegression()
            clf.fit(np.reshape(np.array(mean_condition_biases[condition]), (len(mean_condition_biases[condition]),1)),
                np.reshape(np.array(mean_condition_rts[condition]), (len(mean_condition_rts[condition]),1)))
            a = clf.coef_[0][0]
            b = clf.intercept_[0]
            r_sqr=clf.score(np.reshape(np.array(mean_condition_biases[condition]), (len(mean_condition_biases[condition]),1)),
                np.reshape(np.array(mean_condition_rts[condition]), (len(mean_condition_rts[condition]),1)))
            min_x=mean_condition_biases[condition][0]-0.1
            max_x=mean_condition_biases[condition][-1]+0.1
            plt.plot([min_x, max_x], [a * min_x + b, a * max_x + b], '--%s' % colors[condition], label='%s - r^2=%.3f' % (condition,r_sqr))

        plt.legend(loc='best')
        plt.xlabel('Bias')
        plt.ylabel('RT')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_perc_correct(self, furl, dt, colors):
        fname=os.path.join(self.reports_dir, furl)
        all_biases=[]
        biases={}
        responses={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                biases[stim_level]=[]
                responses[stim_level]=[]
                for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                    if trial_summary.data.rt is not None:
                        prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                            np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                        biases[stim_level].append(prestim_bias)
                        if trial_summary.correct:
                            responses[stim_level].append(1.0)
                        else:
                            responses[stim_level].append(0.0)
                        all_biases.append(prestim_bias)

        hist,bins=np.histogram(all_biases, bins=10)

        fig=plt.figure()
        for stim_level in biases:
            mean_bias=[]
            mean_perc_correct=[]
            for i in range(10):
                bin_responses=[]
                bin_biases=[]
                for bias,response in zip(biases[stim_level],responses[stim_level]):
                    if bias>=bins[i] and bias<bins[i+1]:
                        bin_responses.append(response)
                        bin_biases.append(bias)
                if len(bin_biases):
                    mean_bias.append(np.mean(bin_biases))
                    mean_perc_correct.append(np.mean(bin_responses))
            plt.plot(mean_bias,mean_perc_correct,'o%s' % colors[stim_level], label=stim_level)
        plt.legend(loc='best')
        plt.xlabel('Bias')
        plt.ylabel('% correct')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_perc_left(self, furl, dt, colors):
        fname=os.path.join(self.reports_dir, furl)
        condition_biases={}
        condition_responses={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                if not stim_level in condition_biases:
                    condition_biases[stim_level]=[]
                    condition_responses[stim_level]=[]
                for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                    if trial_summary.data.rt is not None:
                        prestim_bias=np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-\
                                     np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)])
                        condition_biases[stim_level].append(prestim_bias)
                        if trial_summary.decision_idx==0:
                            condition_responses[stim_level].append(1.0)
                        else:
                            condition_responses[stim_level].append(0.0)

        fig=plt.figure()
        mean_condition_biases={}
        mean_condition_perc_left={}
        std_condition_perc_left={}
        for condition in condition_biases:
            hist,bins=np.histogram(condition_biases[condition], bins=10)
            if not condition in mean_condition_biases:
                mean_condition_biases[condition]=[]
                mean_condition_perc_left[condition]=[]
                std_condition_perc_left[condition]=[]
            for i in range(10):
                bin_responses=[]
                bin_biases=[]
                for bias,response in zip(condition_biases[condition],condition_responses[condition]):
                    if bias>=bins[i] and bias<bins[i+1]:
                        bin_responses.append(response)
                        bin_biases.append(bias)
                if len(bin_biases)>=10:
                    mean_condition_biases[condition].append(np.mean(bin_biases))
                    mean_condition_perc_left[condition].append(np.mean(bin_responses))
                    std_condition_perc_left[condition].append(np.std(bin_responses)/np.sqrt(len(bin_responses)))

        for condition in mean_condition_biases:
            plt.errorbar(mean_condition_biases[condition],mean_condition_perc_left[condition],
                yerr=std_condition_perc_left[condition], fmt='o%s' % colors[condition])
            fit=FitSigmoid(mean_condition_biases[condition], mean_condition_perc_left[condition], guess=[1.0,0.0])
            smoothInt = pylab.arange(mean_condition_biases[condition][0]-0.1, mean_condition_biases[condition][-1]+0.1, 0.001)
            smoothResp = fit.eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '--%s' % colors[condition], label=condition)

        plt.legend(loc='best')
        plt.xlabel('Bias')
        plt.ylabel('% left')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

    def plot_bias_bar(self, furl, dt, colors):
        fname=os.path.join(self.reports_dir, furl)
        biases={}
        for subj_report in self.subjects.itervalues():
            for stim_level, session_report in subj_report.sessions.iteritems():
                subj_stim_biases=[]
                for idx,trial_summary in enumerate(session_report.series.trial_summaries):
                    if trial_summary.data.rt is not None:
                        prestim_bias=np.abs(np.mean(trial_summary.data.e_firing_rates[0][int(500*ms/dt):int(950*ms/dt)])-
                                            np.mean(trial_summary.data.e_firing_rates[1][int(500*ms/dt):int(950*ms/dt)]))
                        subj_stim_biases.append(prestim_bias)
                if not stim_level in biases:
                    biases[stim_level]=[]
                biases[stim_level].append(np.mean(subj_stim_biases))
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        conditions=['control','anode','cathode']
        pos = np.arange(len(conditions))+0.5    # Center bars on the Y-axis ticks
        for idx in range(len(conditions)):
            bar=ax.bar(pos[idx],np.mean(biases[conditions[idx]]), width=.5,
                yerr=np.std(biases[conditions[idx]])/np.sqrt(len(biases[conditions[idx]])), align='center',ecolor='k')
            bar[0].set_color(colors[conditions[idx]])
        ax.set_xticks(pos)
        ax.set_xticklabels(conditions)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Prestimulus Bias (Hz)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

if __name__=='__main__':
    dcs_report=DCSComparisonReport('/data/pySBI/rdmd/virtual_subjects_half_dcs',
        'wta.groups.2.duration.4.000.p_e_e.0.080.p_e_i.0.100.p_i_i.0.100.p_i_e.0.200',range(20),
        {'control':(0,0),'anode':(1.0,-0.5),'cathode':(-1.0,0.5)},25,
        '/data/pySBI/reports/rdmd/postexp_sim_virtual_subjects_half_dcs','')
    dcs_report.create_report(regenerate_subject_plots=False,regenerate_session_plots=False,regenerate_trial_plots=False)