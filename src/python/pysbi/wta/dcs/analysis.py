import matplotlib
matplotlib.use('Agg')
import pylab
from brian import second
import os
from scipy.optimize import curve_fit
import subprocess
from brian.stdunits import Hz, ms, nA, mA
from jinja2 import Environment, FileSystemLoader
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.util.utils import Struct, save_to_png, save_to_eps, rt_function, weibull, FitRT, FitWeibull
from pysbi.wta.analysis import TrialSeries, get_lfp_signal

def create_trial_report(trial_summary, reports_dir, dt=.1*ms):
    trial_report=Struct()
    trial_report.trial_idx=trial_summary.trial_idx
    trial_report.contrast=trial_summary.contrast
    trial_report.input_freq=trial_summary.data.input_freq
    trial_report.correct=trial_summary.correct
    trial_report.rt=trial_summary.data.rt
    trial_report.max_rate=trial_summary.max_rate
    trial_report.max_bold=trial_summary.data.summary_data.bold_max

    trial_report.firing_rate_url = None
    if trial_summary.data.e_firing_rates is not None and trial_summary.data.i_firing_rates is not None:
        furl = 'img/firing_rate.contrast.%0.4f.trial.%d' % (trial_summary.contrast, trial_summary.trial_idx)
        fname = os.path.join(reports_dir, furl)
        trial_report.firing_rate_url = '%s.png' % furl

        # figure out max firing rate of all neurons (pyramidal and interneuron)
        max_pop_rate=0
        for i, pop_rate in enumerate(trial_summary.data.e_firing_rates):
            max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])
        for i, pop_rate in enumerate(trial_summary.data.i_firing_rates):
            max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])

        fig=Figure()

        # Plot pyramidal neuron firing rate
        ax=fig.add_subplot(2,1,1)
        for i, pop_rate in enumerate(trial_summary.data.e_firing_rates):
            ax.plot(np.array(range(len(pop_rate))) *dt, pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
        if trial_report.rt:
            rt_idx=(1*second+trial_report.rt)/second
            ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
        ax.set_ylim([0,10+max_pop_rate])
        ax.legend(loc=0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')

        # Plot interneuron firing rate
        ax = fig.add_subplot(2,1,2)
        for i, pop_rate in enumerate(trial_summary.data.i_firing_rates):
            ax.plot(np.array(range(len(pop_rate))) *dt, pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
        if trial_report.rt:
            rt_idx=(1*second+trial_report.rt)/second
            ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
        ax.set_ylim([0,10+max_pop_rate])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        del trial_summary.data.e_firing_rates
        del trial_summary.data.i_firing_rates

    trial_report.neural_state_url=None
    if trial_summary.data.neural_state_rec is not None:
        furl = 'img/neural_state.contrast.%0.4f.trial.%d' % (trial_summary.contrast, trial_summary.trial_idx)
        fname = os.path.join(reports_dir, furl)
        trial_report.neural_state_url = '%s.png' % furl
        fig = plt.figure()
        for i in range(trial_summary.data.num_groups):
            times=np.array(range(len(trial_summary.data.neural_state_rec['g_ampa_r'][i*2])))*.1
            ax = plt.subplot(trial_summary.data.num_groups * 100 + 20 + (i * 2 + 1))
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_r'][i * 2] / nA, label='AMPA-recurrent')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_x'][i * 2] / nA, label='AMPA-task')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_b'][i * 2] / nA, label='AMPA-backgrnd')
            ax.plot(times, trial_summary.data.neural_state_rec['g_nmda'][i * 2] / nA, label='NMDA')
            ax.plot(times, trial_summary.data.neural_state_rec['g_gaba_a'][i * 2] / nA, label='GABA_A')
            plt.xlabel('Time (ms)')
            plt.ylabel('Conductance (nA)')
            ax = plt.subplot(trial_summary.data.num_groups * 100 + 20 + (i * 2 + 2))
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_r'][i * 2 + 1] / nA, label='AMPA-recurrent')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_x'][i * 2 + 1] / nA, label='AMPA-task')
            ax.plot(times, trial_summary.data.neural_state_rec['g_ampa_b'][i * 2 + 1] / nA, label='AMPA-backgrnd')
            ax.plot(times, trial_summary.data.neural_state_rec['g_nmda'][i * 2 + 1] / nA, label='NMDA')
            ax.plot(times, trial_summary.data.neural_state_rec['g_gaba_a'][i * 2 + 1] / nA, label='GABA_A')
            plt.xlabel('Time (ms)')
            plt.ylabel('Conductance (nA)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        del trial_summary.data.neural_state_rec

    trial_report.lfp_url = None
    if trial_summary.data.lfp_rec is not None:
        furl = 'img/lfp.contrast.%0.4f.trial.%d' % (trial_summary.contrast, trial_summary.trial_idx)
        fname = os.path.join(reports_dir, furl)
        trial_report.lfp_url = '%s.png' % furl
        fig = plt.figure()
        ax = plt.subplot(111)
        lfp=get_lfp_signal(trial_summary.data)
        ax.plot(np.array(range(len(lfp))), lfp / mA)
        plt.xlabel('Time (ms)')
        plt.ylabel('LFP (mA)')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        del trial_summary.data.lfp_rec

    trial_report.voxel_url = None

    return trial_report

def create_network_report(data_dir, file_prefix, num_trials, reports_dir, edesc, version=None):
    make_report_dirs(reports_dir)

    report_info=Struct()
    if version is None:
        version=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    report_info.version = version
    report_info.edesc=edesc

    report_info.series=TrialSeries(data_dir, file_prefix, num_trials,
        contrast_range=(0.0, .016, .032, .064, .096, .128, .256, .512), upper_resp_threshold=25,
        lower_resp_threshold=None, dt=.5*ms)
    report_info.series.sort_by_correct()

    report_info.wta_params=report_info.series.trial_summaries[0].data.wta_params
    report_info.pyr_params=report_info.series.trial_summaries[0].data.pyr_params
    report_info.inh_params=report_info.series.trial_summaries[0].data.inh_params
    report_info.voxel_params=report_info.series.trial_summaries[0].data.voxel_params
    report_info.num_groups=report_info.series.trial_summaries[0].data.num_groups
    report_info.trial_duration=report_info.series.trial_summaries[0].data.trial_duration
    report_info.background_freq=report_info.series.trial_summaries[0].data.background_freq
    report_info.stim_start_time=report_info.series.trial_summaries[0].data.stim_start_time
    report_info.stim_end_time=report_info.series.trial_summaries[0].data.stim_end_time
    report_info.network_group_size=report_info.series.trial_summaries[0].data.network_group_size
    report_info.background_input_size=report_info.series.trial_summaries[0].data.background_input_size
    report_info.task_input_size=report_info.series.trial_summaries[0].data.task_input_size
    report_info.muscimol_amount=report_info.series.trial_summaries[0].data.muscimol_amount
    report_info.injection_site=report_info.series.trial_summaries[0].data.injection_site
    report_info.p_dcs=report_info.series.trial_summaries[0].data.p_dcs
    report_info.i_dcs=report_info.series.trial_summaries[0].data.i_dcs

    furl='img/roc'
    fname=os.path.join(reports_dir, furl)
    report_info.roc_url='%s.png' % furl
    report_info.series.plot_multiclass_roc(filename=fname)

    furl='img/rt'
    fname=os.path.join(reports_dir, furl)
    report_info.rt_url='%s.png' % furl
    report_info.series.plot_rt(filename=fname)

    furl='img/perc_correct'
    fname=os.path.join(reports_dir, furl)
    report_info.perc_correct_url='%s.png' % furl
    report_info.series.plot_perc_correct(filename=fname)

#    furl='img/bold_contrast_regression'
#    fname=os.path.join(reports_dir,furl)
#    report_info.bold_contrast_regression_url='%s.png' % furl
#    x_min=np.min(report_info.series.contrast_range)
#    x_max=np.max(report_info.series.contrast_range)
#    fig=plt.figure()
#    report_info.series.max_bold_regression.plot(x_max, x_min,'ok','k','Fit')
#    plt.xlabel('Input Contrast')
#    plt.ylabel('Max BOLD')
#    plt.legend(loc='best')
#    plt.xscale('log')
#    save_to_png(fig, '%s.png' % fname)
#    save_to_eps(fig, '%s.eps' % fname)
#    plt.close(fig)

    report_info.trial_reports=[]
    for trial_summary in report_info.series.trial_summaries:
        report_info.trial_reports.append(create_trial_report(trial_summary, reports_dir, dt=.5*ms))

    #create report
    template_file='wta_network_instance_new.html'
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template=env.get_template(template_file)

    output_file='wta_network.%s.html' % file_prefix
    fname=os.path.join(reports_dir,output_file)
    stream=template.stream(rinfo=report_info)
    stream.dump(fname)

    return report_info.series


class DCSComparisonReport:
    def __init__(self, data_dir, file_prefix, stim_levels, num_trials, reports_dir, edesc):
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
        self.stim_levels=stim_levels
        self.num_trials=num_trials
        self.reports_dir=reports_dir
        self.edesc=edesc

        self.series={}

    def create_report(self):
        make_report_dirs(self.reports_dir)

        report_info=Struct()
        report_info.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        report_info.edesc=self.edesc

        report_info.urls={}
        for stim_level in self.stim_levels:
            print('creating %s report' % stim_level)
            p_dcs=self.stim_levels[stim_level][0]
            i_dcs=self.stim_levels[stim_level][1]
            stim_report_dir=os.path.join(self.reports_dir,stim_level)
            prefix='%s.p_dcs.%.4f.i_dcs.%.4f.control' % (self.file_prefix,p_dcs,i_dcs)
            self.series[stim_level]=create_network_report(self.data_dir,prefix,self.num_trials,stim_report_dir,'', version=report_info.version)
            report_info.urls[stim_level]=os.path.join(stim_level,'wta_network.%s.html' % prefix)
        colors={'anode':'g',
                'anode control 1':'r',
                'anode control 2':'m',
                'cathode':'b',
                'cathode control 1':'c',
                'cathode control 2':'y'}
        report_info.rt_url=self.plot_rt(colors)
        report_info.perc_correct_url=self.plot_perc_correct(colors)
        report_info.bold_contrast_regression_url=None
        #report_info.bold_contrast_regression_url=self.plot_bold_contrast_regression(colors)

        report_info.wta_params=self.series['control'].trial_summaries[0].data.wta_params
        report_info.pyr_params=self.series['control'].trial_summaries[0].data.pyr_params
        report_info.inh_params=self.series['control'].trial_summaries[0].data.inh_params
        report_info.voxel_params=self.series['control'].trial_summaries[0].data.voxel_params
        report_info.num_groups=self.series['control'].trial_summaries[0].data.num_groups
        report_info.trial_duration=self.series['control'].trial_summaries[0].data.trial_duration
        report_info.background_freq=self.series['control'].trial_summaries[0].data.background_freq
        report_info.stim_start_time=self.series['control'].trial_summaries[0].data.stim_start_time
        report_info.stim_end_time=self.series['control'].trial_summaries[0].data.stim_end_time
        report_info.network_group_size=self.series['control'].trial_summaries[0].data.network_group_size
        report_info.background_input_size=self.series['control'].trial_summaries[0].data.background_input_size
        report_info.task_input_size=self.series['control'].trial_summaries[0].data.task_input_size
        report_info.muscimol_amount=self.series['control'].trial_summaries[0].data.muscimol_amount
        report_info.injection_site=self.series['control'].trial_summaries[0].data.injection_site

        #create report
        template_file='wta_dcs_comparison.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        output_file='dcs_comparison.%s.html' % self.file_prefix
        fname=os.path.join(self.reports_dir,output_file)
        stream=template.stream(rinfo=report_info)
        stream.dump(fname)

    def plot_rt(self, colors):
        furl='img/rt'
        fname=os.path.join(self.reports_dir, furl)
        fig=plt.figure()
        contrast, mean_rt, std_rt = self.series['control'].get_contrast_rt_stats()
        rt_fit = FitRT(np.array(contrast), mean_rt, guess=[1,1,1])
        smoothInt = pylab.arange(0.01, max(contrast), 0.001)
        smoothResp = rt_fit.eval(smoothInt)
        plt.errorbar(contrast, mean_rt,yerr=std_rt,fmt='ok')
        plt.plot(smoothInt, smoothResp, 'k', label='control')

        for stim_level, stim_series in self.series.iteritems():
            if not stim_level=='control':
                contrast, mean_rt, std_rt = stim_series.get_contrast_rt_stats()
                rt_fit = FitRT(np.array(contrast), mean_rt, guess=[1,1,1])
                smoothInt = pylab.arange(0.01, max(contrast), 0.001)
                smoothResp = rt_fit.eval(smoothInt)
                plt.errorbar(contrast, mean_rt,yerr=std_rt,fmt='o%s' % colors[stim_level])
                plt.plot(smoothInt, smoothResp, colors[stim_level], label=stim_level)

        plt.xlabel('Contrast')
        plt.ylabel('Decision time (s)')
        plt.xscale('log')
        plt.xlim([.001,1])
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        return '%s.png' % furl

    def plot_perc_correct(self, colors):
        furl='img/perc_correct'
        fname=os.path.join(self.reports_dir, furl)
        fig=plt.figure()
        contrast, perc_correct = self.series['control'].get_contrast_perc_correct_stats()
        acc_fit=FitWeibull(contrast, perc_correct, guess=[0.2, 0.5])
        thresh = np.max([0,acc_fit.inverse(0.8)])
        smoothInt = pylab.arange(0.0, max(contrast), 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        plt.plot(smoothInt, smoothResp, 'k', label='control')
        plt.plot(contrast, perc_correct, 'ok')
        plt.plot([thresh,thresh],[0.4,1.0],'k')
        for stim_level, stim_series in self.series.iteritems():
            if not stim_level=='control':
                contrast, perc_correct = stim_series.get_contrast_perc_correct_stats()
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
        plt.xlim([.001,1])
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        return '%s.png' % furl

    def plot_bold_contrast_regression(self,colors):
        furl='img/bold_contrast_regression'
        fname=os.path.join(self.reports_dir,furl)
        x_min=np.min(self.series['control'].contrast_range)
        x_max=np.max(self.series['control'].contrast_range)
        fig=plt.figure()
        self.series['control'].max_bold_regression.plot(x_max, x_min,'ok','k','control')
        for stim_level, stim_series in self.series.iteritems():
            if not stim_level=='control':
                stim_series.max_bold_regression.plot(x_max, x_min,'o'+colors[stim_level],colors[stim_level],stim_level)
        plt.xlabel('Input Contrast')
        plt.ylabel('Max BOLD')
        plt.legend(loc='best')
        plt.xscale('log')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        return '%s.png' % furl

if __name__=='__main__':
    dcs_report=DCSComparisonReport('/data/projects/pySBI/rdmd/data',
        'wta.groups.2.duration.4.000.p_b_e.0.030.p_x_e.0.010.p_e_e.0.030.p_e_i.0.080.p_i_i.0.200.p_i_e.0.080',
        {
            'control':(0,0),
            'anode':(4,-2),
            'anode control 1':(4,0),
            'anode control 2':(4,2),
            'cathode':(-4,2),
            'cathode control 1':(-4,0),
            'cathode control 2':(-4,-2)
        },50,'/data/projects/pySBI/rdmd/report','')
    dcs_report.create_report()