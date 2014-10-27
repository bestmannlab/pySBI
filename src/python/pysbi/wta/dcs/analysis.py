import matplotlib
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
from pysbi.util.utils import save_to_png, save_to_eps, FitRT, FitWeibull
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
                    rt_idx=(1*second+self.trial_summary.data.rt)/second
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
                    rt_idx=(1*second+self.trial_summary.data.rt)/second
                    ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
                ax.set_ylim([0,10+max_pop_rate])
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Firing Rate (Hz)')
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)
    
            del self.trial_summary.data.e_firing_rates
            del self.trial_summary.data.i_firing_rates
    
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
    def __init__(self, subj_id, stim_condition, data_dir, file_prefix, num_trials, report_dir, edesc, version=None):
        self.subj_id=subj_id
        self.stim_condition=stim_condition
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.num_trials=num_trials
        self.report_dir=report_dir
        self.edesc=edesc
        self.version=version
        if self.version is None:
            self.version=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.trial_reports=[]

    def create_report(self, regenerate_plots=True, regenerate_trial_plots=True):
        
        make_report_dirs(self.report_dir)
    
        self.series=TrialSeries(self.data_dir, self.file_prefix, self.num_trials,
            contrast_range=(0.0, .016, .032, .064, .096, .128, .256, .512), upper_resp_threshold=25,
            lower_resp_threshold=None, dt=.5*ms)
        self.series.sort_by_correct()

        for idx,trial_summary in enumerate(self.series.trial_summaries):
            trial_report=TrialReport(idx+1,trial_summary, self.report_dir, self.edesc, dt=.5*ms, version=self.version)
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
    
        #create report
        template_file='dcs_session.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)
    
        output_file='dcs_session.%s.html' % self.stim_condition
        fname=os.path.join(self.report_dir,output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)
    


class SubjectReport:
    def __init__(self, data_dir, file_prefix, subj_id, stim_levels, num_trials, report_dir, edesc, version=None):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.subj_id=subj_id
        self.stim_levels=stim_levels
        self.num_trials=num_trials
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
                stim_report_dir, self.edesc, version=self.version)
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


class DCSComparisonReport:
    def __init__(self, data_dir, file_prefix, virtual_subj_ids, stim_levels, num_trials, reports_dir, edesc):
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
        self.reports_dir=reports_dir
        self.edesc=edesc

        self.subjects={}

    def create_report(self, regenerate_plots=True, regenerate_subject_plots=True, regenerate_session_plots=True,
                      regenerate_trial_plots=True):
        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])

        for virtual_subj_id in self.virtual_subj_ids:
            print('Creating report for subject %d' % virtual_subj_id)
            subj_report_dir=os.path.join(self.reports_dir,'virtual_subject.%d' % virtual_subj_id)
            self.subjects[virtual_subj_id]=SubjectReport(self.data_dir, self.file_prefix, virtual_subj_id,
                self.stim_levels, self.num_trials, subj_report_dir, self.edesc, version=self.version)
            self.subjects[virtual_subj_id].create_report(regenerate_plots=regenerate_subject_plots,
                regenerate_session_plots=regenerate_session_plots, regenerate_trial_plots=regenerate_trial_plots)

        furl='img/rt'
        self.rt_url='%s.png' % furl
        if regenerate_plots:
            self.plot_rt(furl, condition_colors)

        furl='img/perc_correct'
        self.perc_correct_url='%s.png' % furl
        if regenerate_plots:
            self.plot_perc_correct(furl, condition_colors)

        self.wta_params=self.subjects[self.subjects.keys()[0]].wta_params
        self.pyr_params=self.subjects[self.subjects.keys()[0]].pyr_params
        self.inh_params=self.subjects[self.subjects.keys()[0]].inh_params
        self.voxel_params=self.subjects[self.subjects.keys()[0]].voxel_params
        self.num_groups=self.subjects[self.subjects.keys()[0]].num_groups
        self.trial_duration=self.subjects[self.subjects.keys()[0]].trial_duration
        self.background_freq=self.subjects[self.subjects.keys()[0]].background_freq
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
            std_rt=np.std(np.array(condition_rt[condition]),axis=0)/len(self.subjects)
            rt_fit = FitRT(np.array(contrast), mean_rt, guess=[1,1,1])
            smoothInt = pylab.arange(0.01, max(contrast), 0.001)
            smoothResp = rt_fit.eval(smoothInt)
            plt.errorbar(contrast, mean_rt,yerr=std_rt,fmt='o%s' % colors[condition])
            plt.plot(smoothInt, smoothResp, colors[condition], label=condition)

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
            std_perc_correct=np.std(np.array(condition_perc_correct[condition]),axis=0)/len(self.subjects)
            acc_fit=FitWeibull(contrast, mean_perc_correct, guess=[0.2, 0.5])
            thresh = np.max([0,acc_fit.inverse(0.8)])
            smoothInt = pylab.arange(0.0, max(contrast), 0.001)
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


if __name__=='__main__':
    dcs_report=DCSComparisonReport('/data/pySBI/rdmd/virtual_subjects/mu_40',
        'wta.groups.2.duration.4.000.p_e_e.0.080.p_e_i.0.100.p_i_i.0.100.p_i_e.0.200',range(15),
        {'control':(0,0),'anode':(2,-1),'cathode':(-2,1)},25,
        '/data/pySBI/reports/rdmd/postexp_sim_virtual_subjects_mu_40','')
    dcs_report.create_report()