from scipy.stats import norm, ttest_1samp, f_oneway, ttest_rel
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from scipy.optimize import curve_fit
import copy
import shutil
import subprocess
from brian import second, farad, siemens, volt, Hz, ms, amp
from jinja2 import Environment, FileSystemLoader
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from scikits.learn.linear_model import LinearRegression, LogisticRegression
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.util.utils import save_to_png, save_to_eps, get_response_time, reject_outliers, exp_decay
from pysbi.wta.network import default_params
from pysbi.wta.rl.fit import rescorla_td_prediction

class FileInfo:
    def __init__(self, file_name):
        self.file_name=file_name
        f = h5py.File(file_name)
        self.num_trials=int(f.attrs['trials'])
        self.alpha=float(f.attrs['alpha'])
        if 'beta' in f.attrs:
            self.beta=float(f.attrs['beta'])
        if 'mat_file' in f.attrs:
            self.mat_file=str(f.attrs['mat_file'])
        if 'resp_threshold' in f.attrs:
            self.resp_threshold=float(f.attrs['resp_threshold'])
        self.est_alpha=float(f.attrs['est_alpha'])
        self.est_beta=float(f.attrs['est_beta'])
        self.prop_correct=float(f.attrs['prop_correct'])

        self.num_groups=int(f.attrs['num_groups'])
        self.trial_duration=float(f.attrs['trial_duration'])*second
        self.background_freq=float(f.attrs['background_freq'])*Hz

        self.wta_params=default_params
        self.wta_params.C=float(f.attrs['C'])*farad
        self.wta_params.gL=float(f.attrs['gL'])*siemens
        self.wta_params.EL=float(f.attrs['EL'])*volt
        self.wta_params.VT=float(f.attrs['VT'])*volt
        self.wta_params.DeltaT=float(f.attrs['DeltaT'])*volt
        if 'Mg' in f.attrs:
            self.wta_params.Mg=float(f.attrs['Mg'])
        self.wta_params.E_ampa=float(f.attrs['E_ampa'])*volt
        self.wta_params.E_nmda=float(f.attrs['E_nmda'])*volt
        self.wta_params.E_gaba_a=float(f.attrs['E_gaba_a'])*volt
        if 'E_gaba_b' in f.attrs:
            self.wta_params.E_gaba_b=float(f.attrs['E_gaba_b'])*volt
        self.wta_params.tau_ampa=float(f.attrs['tau_ampa'])*second
        self.wta_params.tau1_nmda=float(f.attrs['tau1_nmda'])*second
        self.wta_params.tau2_nmda=float(f.attrs['tau2_nmda'])*second
        self.wta_params.tau_gaba_a=float(f.attrs['tau_gaba_a'])*second
        if 'pyr_w_ampa_ext' in f.attrs:
            self.wta_params.pyr_w_ampa_ext=float(f.attrs['pyr_w_ampa_ext'])*siemens
            self.wta_params.pyr_w_ampa_bak=float(f.attrs['pyr_w_ampa_bak'])*siemens
            self.wta_params.pyr_w_ampa_rec=float(f.attrs['pyr_w_ampa_rec'])*siemens
            self.wta_params.int_w_ampa_ext=float(f.attrs['int_w_ampa_ext'])*siemens
            self.wta_params.int_w_ampa_bak=float(f.attrs['int_w_ampa_bak'])*siemens
            self.wta_params.int_w_ampa_rec=float(f.attrs['int_w_ampa_rec'])*siemens
            self.wta_params.pyr_w_nmda=float(f.attrs['pyr_w_nmda'])*siemens
            self.wta_params.int_w_nmda=float(f.attrs['int_w_nmda'])*siemens
            self.wta_params.pyr_w_gaba_a=float(f.attrs['pyr_w_gaba_a'])*siemens
            self.wta_params.int_w_gaba_a=float(f.attrs['int_w_gaba_a'])*siemens
        else:
            pyr_param_group=f['pyr_params']
            self.wta_params.pyr_w_ampa_ext=float(pyr_param_group.attrs['w_ampa_ext'])*siemens
            self.wta_params.pyr_w_ampa_bak=float(pyr_param_group.attrs['w_ampa_ext'])*siemens
            self.wta_params.pyr_w_ampa_rec=float(pyr_param_group.attrs['w_ampa_rec'])*siemens
            self.wta_params.pyr_w_nmda=float(pyr_param_group.attrs['w_nmda'])*siemens
            self.wta_params.pyr_w_gaba_a=float(pyr_param_group.attrs['w_gaba'])*siemens

            inh_param_group=f['inh_params']
            self.wta_params.int_w_ampa_ext=float(inh_param_group.attrs['w_ampa_ext'])*siemens
            self.wta_params.int_w_ampa_bak=float(inh_param_group.attrs['w_ampa_ext'])*siemens
            self.wta_params.int_w_ampa_rec=float(inh_param_group.attrs['w_ampa_rec'])*siemens
            self.wta_params.int_w_nmda=float(inh_param_group.attrs['w_nmda'])*siemens
            self.wta_params.int_w_gaba_a=float(inh_param_group.attrs['w_gaba'])*siemens

        if 'p_b_e' in f.attrs:
            self.wta_params.p_b_e=float(f.attrs['p_b_e'])
            self.wta_params.p_x_e=float(f.attrs['p_x_e'])
        self.wta_params.p_e_e=float(f.attrs['p_e_e'])
        self.wta_params.p_e_i=float(f.attrs['p_e_i'])
        self.wta_params.p_i_i=float(f.attrs['p_i_i'])
        self.wta_params.p_i_e=float(f.attrs['p_i_e'])
        if 'p_dcs' in f.attrs:
            self.wta_params.p_dcs=float(f.attrs['p_dcs'])*amp
        if 'i_dcs' in f.attrs:
            self.wta_params.i_dcs=float(f.attrs['i_dcs'])*amp

        self.choice=np.array(f['choice'])
        self.inputs=np.array(f['inputs'])
        self.mags=np.array(f['mags'])
        self.prob_walk=np.array(f['prob_walk'])
        self.rew=np.array(f['rew'])
        self.vals=np.array(f['vals'])
        self.rts=None
        if 'rts' in f:
            self.rts=np.array(f['rts'])

        self.trial_e_rates=[]
        self.trial_i_rates=[]
        self.trial_correct=[]
        for i in range(self.num_trials):
            f_trial=f['trial %d' % i]
            self.trial_e_rates.append(np.array(f_trial['e_rates']))
            self.trial_i_rates.append(np.array(f_trial['i_rates']))
        f.close()

def plot_mean_rate(ax, rate_mean, rate_std_err, color, style, label, dt):
    time_ticks=np.array(range(len(rate_mean)))*dt
    if color is not None and style is not None:
        baseline,=ax.plot(time_ticks, rate_mean, color=color, linestyle=style, label=label)
    elif color is not None:
        baseline,=ax.plot(time_ticks, rate_mean, color=color, label=label)
    elif style is not None:
        baseline,=ax.plot(time_ticks, rate_mean, linestyle=style, label=label)
    else:
        baseline,=ax.plot(time_ticks, rate_mean, label=label)
    ax.fill_between(time_ticks, rate_mean-rate_std_err, rate_mean+rate_std_err, alpha=0.5, 
        facecolor=baseline.get_color())
    return baseline
    
class TrialData:
    def __init__(self, trial, trial_duration, val, ev, inputs, choice, rew, file_prefix, reports_dir, e_firing_rates,
                 i_firing_rates, rt=None, upper_resp_threshold=30, lower_resp_threshold=None, dt=.1*ms, regenerate_plots=False):
        self.trial=trial
        self.val=val
        self.ev=ev
        self.inputs=inputs
        self.choice=choice
        self.rew=rew
        self.correct=0
        if (choice==0 and inputs[0]>inputs[1]) or (choice==1 and inputs[1]>inputs[0]):
            self.correct=1

        if rt is None:
            self.rt,winner=get_response_time(e_firing_rates, 1*second, trial_duration-1*second,
                upper_threshold=upper_resp_threshold, lower_threshold=lower_resp_threshold, dt=dt)
        else:
            self.rt=rt

        furl = 'img/firing_rate.%s' % file_prefix
        fname = os.path.join(reports_dir, furl)
        self.firing_rate_url = '%s.png' % furl

        if regenerate_plots:
            # figure out max firing rate of all neurons (pyramidal and interneuron)
            max_pop_rate=0
            for i, pop_rate in enumerate(e_firing_rates):
                max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])
            for i, pop_rate in enumerate(i_firing_rates):
                max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])

            fig=Figure()

            # Plot pyramidal neuron firing rate
            ax=fig.add_subplot(2,1,1)
            for i, pop_rate in enumerate(e_firing_rates):
                ax.plot(np.array(range(len(pop_rate))) *dt, pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
            if self.rt:
                rt_idx=(1*second+self.rt)/second
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
            ax.set_ylim([0,10+max_pop_rate])
            ax.legend(loc=0)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing Rate (Hz)')

            # Plot interneuron firing rate
            ax = fig.add_subplot(2,1,2)
            for i, pop_rate in enumerate(i_firing_rates):
                ax.plot(np.array(range(len(pop_rate))) *dt, pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
            if self.rt:
                rt_idx=(1*second+self.rt)/second
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
            ax.set_ylim([0,10+max_pop_rate])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing Rate (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

class SessionReport:
    def __init__(self, session_id, data_dir, file_prefix, reports_dir, edesc):
        self.data_dir=data_dir
        self.reports_dir=reports_dir
        self.file_prefix=file_prefix
        self.session_id=session_id
        self.edesc=edesc

    def compute_trial_rate_pyr_stats(self, data, min_ev_diff, max_ev_diff):
        ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
        trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]

        chosen_rate_sum=np.zeros(data.trial_e_rates[0][0,:].shape)
        unchosen_rate_sum=np.zeros(data.trial_e_rates[0][0,:].shape)
        trial_count=0.0
        for trial in trials:
            if data.choice[trial]>-1:
                chosen_rate_sum+=data.trial_e_rates[trial][data.choice[trial],:]
                unchosen_rate_sum+=data.trial_e_rates[trial][1-data.choice[trial],:]
                trial_count+=1.0
        if trial_count>0:
            chosen_rate_mean=chosen_rate_sum/trial_count
            unchosen_rate_mean=unchosen_rate_sum/trial_count
        else:
            chosen_rate_mean=chosen_rate_sum
            unchosen_rate_mean=unchosen_rate_sum
        chosen_rate_std_sum=np.zeros(chosen_rate_mean.shape)
        unchosen_rate_std_sum=np.zeros(unchosen_rate_mean.shape)
        for trial in trials:
            if data.choice[trial]>-1:
                chosen_rate_std_sum+=(data.trial_e_rates[trial][data.choice[trial],:]-chosen_rate_mean)**2.0
                unchosen_rate_std_sum+=(data.trial_e_rates[trial][1-data.choice[trial],:]-unchosen_rate_mean)**2.0
        if trial_count>1:
            chosen_rate_std_err=np.sqrt(chosen_rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
            #chosen_rate_std_err=np.sqrt(chosen_rate_std_sum/(trial_count-1))
            unchosen_rate_std_err=np.sqrt(unchosen_rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
            #unchosen_rate_std_err=np.sqrt(unchosen_rate_std_sum/(trial_count-1))
        else:
            chosen_rate_std_err=chosen_rate_std_sum
            unchosen_rate_std_err=unchosen_rate_std_sum
        return chosen_rate_mean,chosen_rate_std_err,unchosen_rate_mean,unchosen_rate_std_err

    def compute_trial_rate_stats_inh(self, data, min_ev_diff, max_ev_diff):
        ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
        trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]

        rate_sum=np.zeros(data.trial_i_rates[0][0,:].shape)
        trial_count=0.0
        for trial in trials:
            if data.choice[trial]>-1:
                rate_sum+=data.trial_i_rates[trial][0,:]
                trial_count+=1.0
        if trial_count>0:
            rate_mean=rate_sum/trial_count
        else:
            rate_mean=rate_sum
        rate_std_sum=np.zeros(rate_mean.shape)
        for trial in trials:
            if data.choice[trial]>-1:
                rate_std_sum+=(data.trial_i_rates[trial][0,:]-rate_mean)**2.0
        if trial_count>1:
            rate_std_err=np.sqrt(rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
            #rate_std_err=np.sqrt(rate_std_sum/(trial_count-1))
        else:
            rate_std_err=rate_std_sum
        return rate_mean,rate_std_err

    def create_report(self, version, data, regenerate_plots=False, regenerate_trial_plots=False):
        make_report_dirs(self.reports_dir)

        self.version = version
        self.edesc=self.edesc

        self.num_trials=data.num_trials
        self.alpha=data.alpha
        #self.beta=(data.background_freq/Hz*-12.5)+87.46
        #self.beta=(data.background_freq/Hz*-17.29)+148.14
        self.beta=(data.background_freq/Hz*-.17)+161.08
        self.est_alpha=data.est_alpha
        self.est_beta=data.est_beta
        self.prop_correctly_predicted=data.prop_correct*100.0

        self.num_groups=data.num_groups
        self.trial_duration=data.trial_duration
        self.background_freq=data.background_freq
        self.wta_params=copy.deepcopy(data.wta_params)

        fit_vals=rescorla_td_prediction(data.rew, data.choice, data.est_alpha)
        fit_probs=np.zeros(fit_vals.shape)
        ev=fit_vals*data.mags
        fit_probs[0,:]=1.0/(1.0+np.exp(-data.est_beta*(ev[0,:]-ev[1,:])))
        fit_probs[1,:]=1.0/(1.0+np.exp(-data.est_beta*(ev[1,:]-ev[0,:])))

        # Create vals plot
        furl = 'img/vals.%s' % self.file_prefix
        fname = os.path.join(self.reports_dir, furl)
        self.vals_url = '%s.png' % furl

        if regenerate_plots:
            fig=Figure()
            ax=fig.add_subplot(3,1,1)
            ax.plot(data.prob_walk[0,:], label='prob walk - o1')
            ax.plot(data.prob_walk[1,:], label='prob walk - o2')
            ax.legend(loc=0)
            ax=fig.add_subplot(3,1,2)
            ax.plot(data.vals[0,:], label='model vals - o1')
            ax.plot(data.vals[1,:], label='model vals - o2')
            ax.legend(loc=0)
            ax=fig.add_subplot(3,1,3)
            ax.plot(fit_vals[0,:], label='fit vals - o1')
            ax.plot(fit_vals[1,:], label='fit vals - o2')
            ax.legend(loc=0)
            ax.set_xlabel('Trial')

            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create probs plot
        furl='img/probs.%s' % self.file_prefix
        fname = os.path.join(self.reports_dir, furl)
        self.probs_url = '%s.png' % furl

        if regenerate_plots:
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(fit_probs[0,:], label='fit probs - o1')
            ax.plot(fit_probs[1,:], label='fit probs - o2')
            ax.legend(loc=0)
            ax.set_xlabel('Trial')

            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff bar plot
        furl='img/ev_diff.%s' % self.file_prefix
        fname = os.path.join(self.reports_dir, furl)
        self.ev_diff_url = '%s.png' % furl
        ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
        hist,bins=np.histogram(np.array(ev_diff), bins=10)
        bin_width=bins[1]-bins[0]
        if regenerate_plots:
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.bar(bins[:-1], hist/float(len(ev_diff)), width=bin_width)
            ax.set_xlabel('EV Diff')
            ax.set_ylabel('% of Trials')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_pyr_firing_rate.%s' % self.file_prefix
        fname = os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_chosen_mean,small_chosen_std_err,small_unchosen_mean,small_unchosen_std_err=self.compute_trial_rate_pyr_stats(data,
                bins[0], bins[3])
            med_chosen_mean,med_chosen_std_err,med_unchosen_mean,med_unchosen_std_err=self.compute_trial_rate_pyr_stats(data,
                bins[3], bins[6])
            large_chosen_mean,large_chosen_std_err,large_unchosen_mean,large_unchosen_std_err=self.compute_trial_rate_pyr_stats(data,
                bins[6], bins[-1])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_chosen_mean, small_chosen_std_err, 'b', None, 'chosen, small', .5*ms)
            plot_mean_rate(ax, small_unchosen_mean, small_unchosen_std_err, 'b', 'dashed', 'unchosen, small', .5*ms)
            plot_mean_rate(ax, med_chosen_mean, med_chosen_std_err, 'g', None, 'chosen, med', .5*ms)
            plot_mean_rate(ax, med_unchosen_mean, med_unchosen_std_err, 'g', 'dashed', 'unchosen, med', .5*ms)
            plot_mean_rate(ax, large_chosen_mean, large_chosen_std_err, 'r', None, 'chosen, large', .5*ms)
            plot_mean_rate(ax, large_unchosen_mean, large_unchosen_std_err, 'r', 'dashed', 'unchosen, large', .5*ms)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_inh_firing_rate.%s' % self.file_prefix
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_mean,small_std_err=self.compute_trial_rate_stats_inh(data, bins[0], bins[3])
            med_mean,med_std_err=self.compute_trial_rate_stats_inh(data, bins[3], bins[6])
            large_mean,large_std_err=self.compute_trial_rate_stats_inh(data, bins[6], bins[-1])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_mean, small_std_err, 'b', None, 'small', .5*ms)
            plot_mean_rate(ax, med_mean, med_std_err, 'g', None, 'med', .5*ms)
            plot_mean_rate(ax, large_mean, large_std_err, 'r', None, 'large', .5*ms)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        self.perc_no_response=0.0
        self.perc_correct_response=0.0
        self.trials=[]
        for trial in range(self.num_trials):
            trial_ev=data.vals[:,trial]*data.mags[:,trial]
            trial_prefix='%s.trial.%d' % (self.file_prefix,trial)
            rt=None
            if data.rts is not None:
                rt=data.rts[trial]*second
            trial_data=TrialData(trial+1, data.trial_duration, data.vals[:,trial], trial_ev, data.inputs[:,trial],
                data.choice[trial], data.rew[trial], trial_prefix, self.reports_dir, data.trial_e_rates[trial],
                data.trial_i_rates[trial], rt=rt, upper_resp_threshold=30, lower_resp_threshold=None, dt=.5*ms,
                regenerate_plots=regenerate_trial_plots)
            if trial_data.choice<0:
                self.perc_no_response+=1.0
            elif trial_data.correct:
                self.perc_correct_response+=1.0
            self.trials.append(trial_data)
        self.perc_correct_response=self.perc_correct_response/(self.num_trials-self.perc_no_response)*100.0
        self.perc_no_response=self.perc_no_response/self.num_trials*100.0

        #create report
        template_file='rl_session.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        output_file='rl_session.%s.html' % self.file_prefix
        fname=os.path.join(self.reports_dir,output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

        self.furl=os.path.join(self.file_prefix,output_file)


class BackgroundBetaReport:
    def __init__(self, data_dir, file_prefix, background_range, reports_dir, trials, edesc):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.background_range=background_range
        self.reports_dir=reports_dir
        self.edesc=edesc
        self.trials=trials

        self.sessions=[]

    def create_report(self):
        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.edesc=self.edesc

#        beta_vals=np.zeros((len(self.background_range)*self.trials,1))
#        alpha_vals=np.zeros((len(self.background_range)*self.trials,1))
#        background_vals=np.zeros((len(self.background_range)*self.trials,1))
        beta_vals=[]
        alpha_vals=[]
        background_vals=[]

        for idx,background_freq in enumerate(self.background_range):
            for trial in range(self.trials):
                print('background=%0.2f Hz, trial %d' % (background_freq, trial))
                session_prefix=self.file_prefix % (background_freq,trial)
                session_report_dir=os.path.join(self.reports_dir,session_prefix)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                if os.path.exists(session_report_file):
                    session_report=SessionReport(trial, self.data_dir, session_prefix, session_report_dir, self.edesc)
                    session_report.subject=trial
                    data=FileInfo(session_report_file)
                    session_report.create_report(self.version, data)
                    self.sessions.append(session_report)
                    #background_vals[idx*self.trials+trial]=background_freq
                    background_vals.append([background_freq])
                    #alpha_vals[idx*self.trials+trial]=session_report.est_alpha
                    alpha_vals.append([session_report.est_alpha])
                    #beta_vals[idx*self.trials+trial]=session_report.est_beta
                    beta_vals.append([session_report.est_beta])

        background_vals=np.array(background_vals)
        alpha_vals=np.array(alpha_vals)
        beta_vals=np.array(beta_vals)

        self.num_trials=self.sessions[0].num_trials
        self.alpha=self.sessions[0].alpha

        self.num_groups=self.sessions[0].num_groups
        self.trial_duration=self.sessions[0].trial_duration
        self.wta_params=self.sessions[0].wta_params

        clf = LinearRegression()
        clf.fit(background_vals, alpha_vals)
        self.alpha_a = clf.coef_[0][0]
        self.alpha_b = clf.intercept_[0]
        self.alpha_r_sqr=clf.score(background_vals, alpha_vals)

        clf = LinearRegression()
        clf.fit(background_vals, beta_vals)
        self.beta_a = clf.coef_[0][0]
        self.beta_b = clf.intercept_[0]
        self.beta_r_sqr=clf.score(background_vals, beta_vals)

        # Create alpha plot
        furl='img/alpha_fit'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_url = '%s.png' % furl

        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(background_vals,alpha_vals,'o')
        ax.plot([self.background_range[0], self.background_range[-1]], [self.alpha_a * self.background_range[0] + self.alpha_b,
                                                                        self.alpha_a * self.background_range[-1] + self.alpha_b],
            label='r^2=%.3f' % self.alpha_r_sqr)
        ax.set_xlabel('background frequency')
        ax.set_ylabel('alpha')
        ax.set_ylim([0,1])
        ax.legend(loc=0)

        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        # Create beta plot
        furl='img/beta_fit'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_url = '%s.png' % furl

        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(background_vals,beta_vals,'o')
        ax.plot([self.background_range[0], self.background_range[-1]], [self.beta_a * self.background_range[0] + self.beta_b,
                                                                        self.beta_a * self.background_range[-1] + self.beta_b],
            label='r^2=%.3f' % self.beta_r_sqr)
        ax.set_xlabel('background frequency')
        ax.set_ylabel('beta')
        ax.legend(loc=0)

        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        #create report
        template_file='rl_background_beta.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='rl_background_beta.html'
        fname=os.path.join(self.reports_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

class StimConditionReport:
    def __init__(self, data_dir, file_prefix, stim_condition, reports_dir, num_subjects, edesc):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.stim_condition=stim_condition
        self.reports_dir=reports_dir
        self.edesc=edesc
        self.num_subjects=num_subjects

        self.sessions=[]
        self.excluded_sessions=[]

    def compute_baseline_diff_rates(self, min_ev_diff, max_ev_diff):
        pyr_rate_diffs=[]
        subjects=0
        for virtual_subj_id in range(self.num_subjects):
            subj_pyr_rate_diffs=[]
            if virtual_subj_id not in self.excluded_sessions:
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                for trial in range(len(data.trial_e_rates)):
                    ev_diff=np.abs(data.vals[0,trial]*data.mags[0,trial]-data.vals[1,trial]*data.mags[1,trial])
                    if ev_diff>=min_ev_diff and ev_diff<max_ev_diff:
                        subjects+=1.0
                        rate1=np.mean(data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        rate2=np.mean(data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        subj_pyr_rate_diffs.append(np.abs(rate1-rate2))
            pyr_rate_diffs.append(np.mean(subj_pyr_rate_diffs))
        #W,p=stats.shapiro(pyr_rate_diffs)
        #print('p=%.4f' % p)
        #return np.mean(pyr_rate_diffs),np.std(pyr_rate_diffs)/np.sqrt(trials)
        return pyr_rate_diffs,subjects

    def compute_baseline_rates(self, min_ev_diff, max_ev_diff):
        pyr_rates=[]
        inh_rates=[]
        subjects=0
        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                subj_pyr_rates=[]
                subj_inh_rates=[]
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                for trial in range(len(data.trial_e_rates)):
                    ev_diff=np.abs(data.vals[0,trial]*data.mags[0,trial]-data.vals[1,trial]*data.mags[1,trial])
                    if ev_diff>=min_ev_diff and ev_diff<max_ev_diff:
                        subjects+=1.0
                        subj_pyr_rates.append(np.mean((data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))]+
                                          data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])/2.0))
                        subj_inh_rates.append(np.mean(data.trial_i_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))]))
                pyr_rates.append(np.mean(subj_pyr_rates))
                inh_rates.append(np.mean(subj_inh_rates))
        #return np.mean(pyr_rates),np.std(pyr_rates)/np.sqrt(trials),np.mean(inh_rates),np.std(inh_rates)/np.sqrt(trials)
        return pyr_rates,inh_rates,subjects
        #return np.mean(pyr_rates),np.std(pyr_rates),np.mean(inh_rates),np.std(inh_rates)

    def compute_ev_diff_rates(self, min_ev_diff, max_ev_diff):
        diff_rates=[]
        subjects=0
        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                subj_diff_rates=[]
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                for trial in range(len(data.trial_e_rates)):
                    ev_diff=np.abs(data.vals[0,trial]*data.mags[0,trial]-data.vals[1,trial]*data.mags[1,trial])
                    if ev_diff>=min_ev_diff and ev_diff<max_ev_diff:
                        if data.choice[trial]>-1:
                            subjects+=1.0
                            chosen_mean=np.mean(data.trial_e_rates[trial][data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                            unchosen_mean=np.mean(data.trial_e_rates[trial][1-data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                            subj_diff_rates.append(chosen_mean-unchosen_mean)
                diff_rates.append(np.mean(subj_diff_rates))
        #return np.mean(diff_rates),np.std(diff_rates)/np.sqrt(trials)
        return diff_rates,subjects

    def compute_trial_rate_pyr_stats(self, min_beta, max_beta, min_ev_diff, max_ev_diff):
        data=FileInfo(os.path.join(self.data_dir,'%s.h5' % self.file_prefix % (0,self.stim_condition)))
        chosen_rate_sum=np.zeros(data.trial_e_rates[0][0,:].shape)
        unchosen_rate_sum=np.zeros(data.trial_e_rates[0][0,:].shape)
        trial_count=0.0

        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                if min_beta <= data.est_beta < max_beta:                
                    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
                    trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]            
                    for trial in trials:
                        if data.choice[trial]>-1:
                            chosen_rate_sum+=data.trial_e_rates[trial][data.choice[trial],:]
                            unchosen_rate_sum+=data.trial_e_rates[trial][1-data.choice[trial],:]
                            trial_count+=1
        chosen_rate_mean=chosen_rate_sum/trial_count
        unchosen_rate_mean=unchosen_rate_sum/trial_count
        chosen_rate_std_sum=np.zeros(chosen_rate_mean.shape)
        unchosen_rate_std_sum=np.zeros(unchosen_rate_mean.shape)
        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                if min_beta <= data.est_beta < max_beta:
                    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
                    trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]
                    for trial in trials:
                        if data.choice[trial]>-1:
                            chosen_rate_std_sum+=(data.trial_e_rates[trial][data.choice[trial],:]-chosen_rate_mean)**2.0
                            unchosen_rate_std_sum+=(data.trial_e_rates[trial][1-data.choice[trial],:]-unchosen_rate_mean)**2.0
        chosen_rate_std_err=np.sqrt(chosen_rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
        #chosen_rate_std_err=np.sqrt(chosen_rate_std_sum/(trial_count-1))
        unchosen_rate_std_err=np.sqrt(unchosen_rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
        #unchosen_rate_std_err=np.sqrt(unchosen_rate_std_sum/(trial_count-1))
        return chosen_rate_mean,chosen_rate_std_err,unchosen_rate_mean,unchosen_rate_std_err

    def compute_trial_rate_inh_stats(self, min_beta, max_beta, min_ev_diff, max_ev_diff):
        data=FileInfo(os.path.join(self.data_dir,'%s.h5' % self.file_prefix % (0,self.stim_condition)))
        rate_sum=np.zeros(data.trial_i_rates[0][0,:].shape)
        trial_count=0.0

        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                if min_beta <= data.est_beta < max_beta:
                    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
                    trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]
                    for trial in trials:
                        if data.choice[trial]>-1:
                            rate_sum+=data.trial_i_rates[trial][0,:]
                            trial_count+=1
        rate_mean=rate_sum/trial_count
        rate_std_sum=np.zeros(rate_mean.shape)
        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                if min_beta <= data.est_beta < max_beta:
                    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
                    trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]
                    for trial in trials:
                        if data.choice[trial]>-1:
                            rate_std_sum+=(data.trial_i_rates[trial][0,:]-rate_mean)**2.0
        rate_std_err=np.sqrt(rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
        #rate_std_err=np.sqrt(rate_std_sum/(trial_count-1))
        return rate_mean,rate_std_err
    
    def create_report(self, version, excluded=None, regenerate_plots=False, regenerate_session_plots=False, regenerate_trial_plots=False):
        make_report_dirs(self.reports_dir)

        self.version=version
        #self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.edesc=self.edesc

        self.condition_alphas=[]
        self.condition_betas=[]
        self.condition_perc_correct=[]
        self.ev_diff=[]

        #beta_hist,beta_bins=np.histogram(reject_outliers(self.condition_betas), bins=10)
        beta_hist,beta_bins=np.histogram(self.condition_betas, bins=10)
        ev_diff_hist,ev_diff_bins=np.histogram(np.array(self.ev_diff), bins=10)

        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                print('subject %d' % virtual_subj_id)
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_dir=os.path.join(self.reports_dir,session_prefix)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                session_report=SessionReport(virtual_subj_id, self.data_dir, session_prefix, session_report_dir, self.edesc)
                data=FileInfo(session_report_file)
                self.ev_diff.extend(np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:]))
                self.condition_alphas.append([data.est_alpha])
                self.condition_betas.append([data.est_beta])
                session_report.create_report(self.version, data, regenerate_plots=regenerate_session_plots,
                    regenerate_trial_plots=regenerate_trial_plots)
                self.sessions.append(session_report)                
                self.condition_perc_correct.append([session_report.perc_correct_response])

        self.condition_alphas=np.array(self.condition_alphas)
        self.condition_betas=np.array(self.condition_betas)
        self.condition_perc_correct=np.array(self.condition_perc_correct)

        # Create beta bar plot
        furl='img/beta_dist'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_url = '%s.png' % furl
        if regenerate_plots:
            #hist,bins=np.histogram(reject_outliers(self.condition_betas), bins=10)
            hist,bins=np.histogram(self.condition_betas, bins=10)
            bin_width=bins[1]-bins[0]
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            #ax.bar(bins[:-1], hist/float(len(reject_outliers(self.condition_betas))), width=bin_width)
            ax.bar(bins[:-1], hist/float(len(self.condition_betas)), width=bin_width)
            ax.set_xlabel('Beta')
            ax.set_ylabel('% of Subjects')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create alpha bar plot
        furl='img/alpha_dist'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_url = '%s.png' % furl
        if regenerate_plots:
            hist,bins=np.histogram(self.condition_alphas, bins=10)
            bin_width=bins[1]-bins[0]
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.bar(bins[:-1], hist/float(len(self.condition_alphas)), width=bin_width)
            ax.set_xlabel('Alpha')
            ax.set_ylabel('% of Subjects')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create alpha - perc correct plot
        furl='img/alpha_perc_correct'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_perc_correct_url='%s.png' % furl
        clf = LinearRegression()
        clf.fit(self.condition_alphas, self.condition_perc_correct/100.0)
        self.alpha_perc_correct_a = clf.coef_[0][0]
        self.alpha_perc_correct_b = clf.intercept_[0]
        self.alpha_perc_correct_r_sqr=clf.score(self.condition_alphas, self.condition_perc_correct/100.0)
        if regenerate_plots:
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(self.condition_alphas,self.condition_perc_correct/100.0,'o')
            min_x=np.min(self.condition_alphas)-.1
            max_x=np.max(self.condition_alphas)+.1
            ax.plot([min_x, max_x], [self.alpha_perc_correct_a * min_x + self.alpha_perc_correct_b,
                                     self.alpha_perc_correct_a * max_x + self.alpha_perc_correct_b],
                label='r^2=%.3f' % self.alpha_perc_correct_r_sqr)
            ax.legend(loc=0)
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create beta - perc correct plot
        furl='img/beta_perc_correct'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_perc_correct_url='%s.png' % furl
        clf = LinearRegression()
        clf.fit(self.condition_betas, self.condition_perc_correct/100.0)
        self.beta_perc_correct_a = clf.coef_[0][0]
        self.beta_perc_correct_b = clf.intercept_[0]
        self.beta_perc_correct_r_sqr=clf.score(self.condition_betas, self.condition_perc_correct/100.0)
        if regenerate_plots:
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(self.condition_betas,self.condition_perc_correct/100.0,'o')
            min_x=np.min(self.condition_betas)-1.0
            max_x=np.max(self.condition_betas)+1.0
            ax.plot([min_x, max_x], [self.beta_perc_correct_a * min_x + self.beta_perc_correct_b,
                                     self.beta_perc_correct_a * max_x + self.beta_perc_correct_b],
                label='r^2=%.3f' % self.beta_perc_correct_r_sqr)
            ax.legend(loc=0)
            ax.set_xlabel('Beta')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/small_ev_diff_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_beta_small_ev_diff_chosen_mean,\
            small_beta_small_ev_diff_chosen_std_err,\
            small_beta_small_ev_diff_unchosen_mean,\
            small_beta_small_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[0], beta_bins[3],
                ev_diff_bins[0], ev_diff_bins[3])
            med_beta_small_ev_diff_chosen_mean,\
            med_beta_small_ev_diff_chosen_std_err,\
            med_beta_small_ev_diff_unchosen_mean,\
            med_beta_small_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[3], beta_bins[6],
                ev_diff_bins[0], ev_diff_bins[3])
            large_beta_small_ev_diff_chosen_mean,\
            large_beta_small_ev_diff_chosen_std_err,\
            large_beta_small_ev_diff_unchosen_mean,\
            large_beta_small_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[6], beta_bins[-1],
                ev_diff_bins[0], ev_diff_bins[3])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_beta_small_ev_diff_chosen_mean, small_beta_small_ev_diff_chosen_std_err, 'b', None,
                'small beta, chosen', .5*ms)
            plot_mean_rate(ax, small_beta_small_ev_diff_unchosen_mean, small_beta_small_ev_diff_unchosen_std_err, 'b', 'dashed',
                'small beta, unchosen', .5*ms)
            plot_mean_rate(ax, med_beta_small_ev_diff_chosen_mean, med_beta_small_ev_diff_chosen_std_err, 'g', None,
                'med beta, chosen', .5*ms)
            plot_mean_rate(ax, med_beta_small_ev_diff_unchosen_mean, med_beta_small_ev_diff_unchosen_std_err, 'g', 'dashed',
                'med beta, unchosen', .5*ms)
            plot_mean_rate(ax, large_beta_small_ev_diff_chosen_mean, large_beta_small_ev_diff_chosen_std_err, 'r', None,
                'large beta, chosen', .5*ms)
            plot_mean_rate(ax, large_beta_small_ev_diff_unchosen_mean, large_beta_small_ev_diff_unchosen_std_err, 'r', 'dashed',
                'large beta, unchosen', .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/med_ev_diff_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_beta_med_ev_diff_chosen_mean,\
            small_beta_med_ev_diff_chosen_std_err,\
            small_beta_med_ev_diff_unchosen_mean,\
            small_beta_med_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[0], beta_bins[3],
                ev_diff_bins[3], ev_diff_bins[6])
            med_beta_med_ev_diff_chosen_mean,\
            med_beta_med_ev_diff_chosen_std_err,\
            med_beta_med_ev_diff_unchosen_mean,\
            med_beta_med_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[3], beta_bins[6],
                ev_diff_bins[3], ev_diff_bins[6])
            large_beta_med_ev_diff_chosen_mean,\
            large_beta_med_ev_diff_chosen_std_err,\
            large_beta_med_ev_diff_unchosen_mean,\
            large_beta_med_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[6], beta_bins[-1],
                ev_diff_bins[3], ev_diff_bins[6])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_beta_med_ev_diff_chosen_mean, small_beta_med_ev_diff_chosen_std_err, 'b', None,
                'small beta, chosen', .5*ms)
            plot_mean_rate(ax, small_beta_med_ev_diff_unchosen_mean, small_beta_med_ev_diff_unchosen_std_err, 'b', 'dashed',
                'small beta, unchosen', .5*ms)
            plot_mean_rate(ax, med_beta_med_ev_diff_chosen_mean, med_beta_med_ev_diff_chosen_std_err, 'g', None,
                'med beta, chosen', .5*ms)
            plot_mean_rate(ax, med_beta_med_ev_diff_unchosen_mean, med_beta_med_ev_diff_unchosen_std_err, 'g', 'dashed',
                'med beta, unchosen', .5*ms)
            plot_mean_rate(ax, large_beta_med_ev_diff_chosen_mean, large_beta_med_ev_diff_chosen_std_err, 'r', None,
                'large beta, chosen', .5*ms)
            plot_mean_rate(ax, large_beta_med_ev_diff_unchosen_mean, large_beta_med_ev_diff_unchosen_std_err, 'r', 'dashed',
                'large beta, unchosen', .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/large_ev_diff_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_beta_large_ev_diff_chosen_mean,\
            small_beta_large_ev_diff_chosen_std_err,\
            small_beta_large_ev_diff_unchosen_mean,\
            small_beta_large_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[0], beta_bins[3],
                ev_diff_bins[6], ev_diff_bins[-1])
            med_beta_large_ev_diff_chosen_mean,\
            med_beta_large_ev_diff_chosen_std_err,\
            med_beta_large_ev_diff_unchosen_mean,\
            med_beta_large_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[3], beta_bins[6],
                ev_diff_bins[6], ev_diff_bins[-1])
            large_beta_large_ev_diff_chosen_mean,\
            large_beta_large_ev_diff_chosen_std_err,\
            large_beta_large_ev_diff_unchosen_mean,\
            large_beta_large_ev_diff_unchosen_std_err=self.compute_trial_rate_pyr_stats(beta_bins[6], beta_bins[-1],
                ev_diff_bins[6], ev_diff_bins[-1])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_beta_large_ev_diff_chosen_mean, small_beta_large_ev_diff_chosen_std_err, 'b', None,
                'small beta, chosen', .5*ms)
            plot_mean_rate(ax, small_beta_large_ev_diff_unchosen_mean, small_beta_large_ev_diff_unchosen_std_err, 'b', 'dashed',
                'small beta, unchosen', .5*ms)
            plot_mean_rate(ax, med_beta_large_ev_diff_chosen_mean, med_beta_large_ev_diff_chosen_std_err, 'g', None,
                'med beta, chosen', .5*ms)
            plot_mean_rate(ax, med_beta_large_ev_diff_unchosen_mean, med_beta_large_ev_diff_unchosen_std_err, 'g', 'dashed',
                'med beta, unchosen', .5*ms)
            plot_mean_rate(ax, large_beta_large_ev_diff_chosen_mean, large_beta_large_ev_diff_chosen_std_err, 'r', None,
                'large beta, chosen', .5*ms)
            plot_mean_rate(ax, large_beta_large_ev_diff_unchosen_mean, large_beta_large_ev_diff_unchosen_std_err, 'r', 'dashed',
                'large beta, unchosen', .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/small_ev_diff_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_beta_small_ev_diff_mean,\
            small_beta_small_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[0], beta_bins[3],
                ev_diff_bins[0], ev_diff_bins[3])
            med_beta_small_ev_diff_mean,\
            med_beta_small_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[3], beta_bins[6],
                ev_diff_bins[0], ev_diff_bins[3])
            large_beta_small_ev_diff_mean,\
            large_beta_small_ev_diff_std_err,=self.compute_trial_rate_inh_stats(beta_bins[6], beta_bins[-1],
                ev_diff_bins[0], ev_diff_bins[3])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_beta_small_ev_diff_mean, small_beta_small_ev_diff_std_err, 'b', None,
                'small beta, chosen', .5*ms)
            plot_mean_rate(ax, med_beta_small_ev_diff_mean, med_beta_small_ev_diff_std_err, 'g', None,
                'med beta, chosen', .5*ms)
            plot_mean_rate(ax, large_beta_small_ev_diff_mean, large_beta_small_ev_diff_std_err, 'r', None,
                'large beta, chosen', .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/med_ev_diff_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_beta_med_ev_diff_mean,\
            small_beta_med_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[0], beta_bins[3],
                ev_diff_bins[3], ev_diff_bins[6])
            med_beta_med_ev_diff_mean,\
            med_beta_med_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[3], beta_bins[6],
                ev_diff_bins[3], ev_diff_bins[6])
            large_beta_med_ev_diff_mean,\
            large_beta_med_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[6], beta_bins[-1],
                ev_diff_bins[3], ev_diff_bins[6])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_beta_med_ev_diff_mean, small_beta_med_ev_diff_std_err, 'b', None,
                'small beta, chosen', .5*ms)
            plot_mean_rate(ax, med_beta_med_ev_diff_mean, med_beta_med_ev_diff_std_err, 'g', None,
                'med beta, chosen', .5*ms)
            plot_mean_rate(ax, large_beta_med_ev_diff_mean, large_beta_med_ev_diff_std_err, 'r', None,
                'large beta, chosen', .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/large_ev_diff_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            small_beta_large_ev_diff_mean,\
            small_beta_large_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[0], beta_bins[3],
                ev_diff_bins[6], ev_diff_bins[-1])
            med_beta_large_ev_diff_mean,\
            med_beta_large_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[3], beta_bins[6],
                ev_diff_bins[6], ev_diff_bins[-1])
            large_beta_large_ev_diff_mean,\
            large_beta_large_ev_diff_std_err=self.compute_trial_rate_inh_stats(beta_bins[6], beta_bins[-1],
                ev_diff_bins[6], ev_diff_bins[-1])
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            plot_mean_rate(ax, small_beta_large_ev_diff_mean, small_beta_large_ev_diff_std_err, 'b', None,
                'small beta, chosen', .5*ms)
            plot_mean_rate(ax, med_beta_large_ev_diff_mean, med_beta_large_ev_diff_std_err, 'g', None,
                'med beta, chosen', .5*ms)
            plot_mean_rate(ax, large_beta_large_ev_diff_mean, large_beta_large_ev_diff_std_err, 'r', None,
                'large beta, chosen', .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        self.num_trials=self.sessions[0].num_trials
        self.alpha=self.sessions[0].alpha

        self.num_groups=self.sessions[0].num_groups
        self.trial_duration=self.sessions[0].trial_duration
        self.wta_params=self.sessions[0].wta_params

        #create report
        template_file='rl_stim_condition.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='rl_%s.html' % self.stim_condition
        fname=os.path.join(self.reports_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

class RLReport:
    def __init__(self, data_dir, file_prefix, stim_conditions, reports_dir, num_subjects, edesc):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.stim_conditions=stim_conditions
        self.reports_dir=reports_dir
        self.edesc=edesc
        self.num_subjects=num_subjects

        self.stim_condition_reports={}

    def create_report(self, regenerate_plots=False, regenerate_condition_plots=False, regenerate_session_plots=False,
                      regenerate_trial_plots=False):
        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.edesc=self.edesc

        condition_colors={'control':'b','anode':'r','cathode':'g'}

        excluded=None
        ev_diffs=[]
        for stim_condition in self.stim_conditions:
            print(stim_condition)
            stim_condition_report_dir=os.path.join(self.reports_dir,stim_condition)
            self.stim_condition_reports[stim_condition] = StimConditionReport(self.data_dir, self.file_prefix,
                stim_condition, stim_condition_report_dir, self.num_subjects, self.edesc)
            self.stim_condition_reports[stim_condition].create_report(self.version, excluded=excluded,
                regenerate_plots=regenerate_condition_plots, regenerate_session_plots=regenerate_session_plots,
                regenerate_trial_plots=regenerate_trial_plots)
            excluded=self.stim_condition_reports[stim_condition].excluded_sessions
            ev_diffs.extend(self.stim_condition_reports[stim_condition].ev_diff)

        ev_diff_hist,ev_diff_bins=np.histogram(np.array(ev_diffs), bins=10)

        if regenerate_plots:

            self.anode_stim_condition_chosen_rate_means={}
            self.anode_stim_condition_chosen_rate_std_err={}
            self.anode_stim_condition_unchosen_rate_means={}
            self.anode_stim_condition_unchosen_rate_std_err={}
            self.anode_stim_condition_inh_rate_means={}
            self.anode_stim_condition_inh_rate_std_err={}

            self.cathode_stim_condition_chosen_rate_means={}
            self.cathode_stim_condition_chosen_rate_std_err={}
            self.cathode_stim_condition_unchosen_rate_means={}
            self.cathode_stim_condition_unchosen_rate_std_err={}
            self.cathode_stim_condition_inh_rate_means={}
            self.cathode_stim_condition_inh_rate_std_err={}

            self.anode_stim_condition_small_ev_chosen_rate_means={}
            self.anode_stim_condition_small_ev_chosen_rate_std_err={}
            self.anode_stim_condition_small_ev_unchosen_rate_means={}
            self.anode_stim_condition_small_ev_unchosen_rate_std_err={}
            self.anode_stim_condition_small_ev_inh_rate_means={}
            self.anode_stim_condition_small_ev_inh_rate_std_err={}

            self.cathode_stim_condition_small_ev_chosen_rate_means={}
            self.cathode_stim_condition_small_ev_chosen_rate_std_err={}
            self.cathode_stim_condition_small_ev_unchosen_rate_means={}
            self.cathode_stim_condition_small_ev_unchosen_rate_std_err={}
            self.cathode_stim_condition_small_ev_inh_rate_means={}
            self.cathode_stim_condition_small_ev_inh_rate_std_err={}

            self.anode_stim_condition_med_ev_chosen_rate_means={}
            self.anode_stim_condition_med_ev_chosen_rate_std_err={}
            self.anode_stim_condition_med_ev_unchosen_rate_means={}
            self.anode_stim_condition_med_ev_unchosen_rate_std_err={}
            self.anode_stim_condition_med_ev_inh_rate_means={}
            self.anode_stim_condition_med_ev_inh_rate_std_err={}

            self.cathode_stim_condition_med_ev_chosen_rate_means={}
            self.cathode_stim_condition_med_ev_chosen_rate_std_err={}
            self.cathode_stim_condition_med_ev_unchosen_rate_means={}
            self.cathode_stim_condition_med_ev_unchosen_rate_std_err={}
            self.cathode_stim_condition_med_ev_inh_rate_means={}
            self.cathode_stim_condition_med_ev_inh_rate_std_err={}

            self.anode_stim_condition_large_ev_chosen_rate_means={}
            self.anode_stim_condition_large_ev_chosen_rate_std_err={}
            self.anode_stim_condition_large_ev_unchosen_rate_means={}
            self.anode_stim_condition_large_ev_unchosen_rate_std_err={}
            self.anode_stim_condition_large_ev_inh_rate_means={}
            self.anode_stim_condition_large_ev_inh_rate_std_err={}

            self.cathode_stim_condition_large_ev_chosen_rate_means={}
            self.cathode_stim_condition_large_ev_chosen_rate_std_err={}
            self.cathode_stim_condition_large_ev_unchosen_rate_means={}
            self.cathode_stim_condition_large_ev_unchosen_rate_std_err={}
            self.cathode_stim_condition_large_ev_inh_rate_means={}
            self.cathode_stim_condition_large_ev_inh_rate_std_err={}

        self.stim_condition_perc_correct={}
        self.stim_condition_no_response={}
        for stim_condition in self.stim_conditions:
            if regenerate_plots:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    self.anode_stim_condition_chosen_rate_means[stim_condition],\
                    self.anode_stim_condition_chosen_rate_std_err[stim_condition],\
                    self.anode_stim_condition_unchosen_rate_means[stim_condition],\
                    self.anode_stim_condition_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,0,100)
                    self.anode_stim_condition_inh_rate_means[stim_condition],\
                    self.anode_stim_condition_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,0,100)

                    self.anode_stim_condition_small_ev_chosen_rate_means[stim_condition],\
                    self.anode_stim_condition_small_ev_chosen_rate_std_err[stim_condition],\
                    self.anode_stim_condition_small_ev_unchosen_rate_means[stim_condition],\
                    self.anode_stim_condition_small_ev_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,ev_diff_bins[0],ev_diff_bins[3])
                    self.anode_stim_condition_small_ev_inh_rate_means[stim_condition],\
                    self.anode_stim_condition_small_ev_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,ev_diff_bins[0],ev_diff_bins[3])

                    self.anode_stim_condition_med_ev_chosen_rate_means[stim_condition],\
                    self.anode_stim_condition_med_ev_chosen_rate_std_err[stim_condition],\
                    self.anode_stim_condition_med_ev_unchosen_rate_means[stim_condition],\
                    self.anode_stim_condition_med_ev_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,ev_diff_bins[3],ev_diff_bins[6])
                    self.anode_stim_condition_med_ev_inh_rate_means[stim_condition],\
                    self.anode_stim_condition_med_ev_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,ev_diff_bins[3],ev_diff_bins[6])

                    self.anode_stim_condition_large_ev_chosen_rate_means[stim_condition],\
                    self.anode_stim_condition_large_ev_chosen_rate_std_err[stim_condition],\
                    self.anode_stim_condition_large_ev_unchosen_rate_means[stim_condition],\
                    self.anode_stim_condition_large_ev_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,ev_diff_bins[6],ev_diff_bins[-1])
                    self.anode_stim_condition_large_ev_inh_rate_means[stim_condition],\
                    self.anode_stim_condition_large_ev_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,ev_diff_bins[6],ev_diff_bins[-1])

                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    self.cathode_stim_condition_chosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_chosen_rate_std_err[stim_condition],\
                    self.cathode_stim_condition_unchosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,0,100)
                    self.cathode_stim_condition_inh_rate_means[stim_condition],\
                    self.cathode_stim_condition_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,0,100)

                    self.cathode_stim_condition_small_ev_chosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_small_ev_chosen_rate_std_err[stim_condition],\
                    self.cathode_stim_condition_small_ev_unchosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_small_ev_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,ev_diff_bins[0],ev_diff_bins[3])
                    self.cathode_stim_condition_small_ev_inh_rate_means[stim_condition],\
                    self.cathode_stim_condition_small_ev_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,ev_diff_bins[0],ev_diff_bins[3])

                    self.cathode_stim_condition_med_ev_chosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_med_ev_chosen_rate_std_err[stim_condition],\
                    self.cathode_stim_condition_med_ev_unchosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_med_ev_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,ev_diff_bins[3],ev_diff_bins[6])
                    self.cathode_stim_condition_med_ev_inh_rate_means[stim_condition],\
                    self.cathode_stim_condition_med_ev_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,ev_diff_bins[3],ev_diff_bins[6])

                    self.cathode_stim_condition_large_ev_chosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_large_ev_chosen_rate_std_err[stim_condition],\
                    self.cathode_stim_condition_large_ev_unchosen_rate_means[stim_condition],\
                    self.cathode_stim_condition_large_ev_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_pyr_stats(0,10000,ev_diff_bins[6],ev_diff_bins[-1])
                    self.cathode_stim_condition_large_ev_inh_rate_means[stim_condition],\
                    self.cathode_stim_condition_large_ev_inh_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_inh_stats(0,10000,ev_diff_bins[6],ev_diff_bins[-1])

            self.stim_condition_perc_correct[stim_condition]=[]
            self.stim_condition_no_response[stim_condition]=[]
            for session in self.stim_condition_reports[stim_condition].sessions:
                self.stim_condition_perc_correct[stim_condition].append(session.perc_correct_response)
                self.stim_condition_no_response[stim_condition].append(session.perc_no_response)
        for stim_condition in self.stim_conditions:
            self.stim_condition_perc_correct[stim_condition]=np.array(self.stim_condition_perc_correct[stim_condition])
            self.stim_condition_no_response[stim_condition]=np.array(self.stim_condition_no_response[stim_condition])

        # Create % correct plot
        furl='img/perc_correct'
        fname=os.path.join(self.reports_dir,furl)
        self.perc_correct_url='%s.png' % furl

        self.perc_correct_freidman=stats.friedmanchisquare(*self.stim_condition_perc_correct.values())
        #self.perc_correct_control_ttest={}
        #self.perc_correct_anode_ttest={}
        #self.perc_correct_cathode_ttest={}
        self.perc_correct_control_wilcoxon={}
        self.perc_correct_anode_wilcoxon={}
        self.perc_correct_cathode_wilcoxon={}
        num_comparisons=0.0
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                num_comparisons+=1.0
            if stim_condition.startswith('anode_control'):
                num_comparisons+=1.0
            elif stim_condition.startswith('cathode_control'):
                num_comparisons+=1.0
        self.perc_correct_mean={}
        self.perc_correct_std={}
        perc_correct_mean=[]
        perc_correct_std_err=[]
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                #self.perc_correct_control_ttest[stim_condition]=stats.ttest_rel(self.stim_condition_perc_correct['control'],
                #    self.stim_condition_perc_correct[stim_condition])
                T,p=stats.wilcoxon(self.stim_condition_perc_correct['control'],
                    self.stim_condition_perc_correct[stim_condition])
                self.perc_correct_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                #self.perc_correct_anode_ttest[stim_condition]=stats.ttest_rel(self.stim_condition_perc_correct['anode'],
                #    self.stim_condition_perc_correct[stim_condition])
                T,p=stats.wilcoxon(self.stim_condition_perc_correct['anode'],
                    self.stim_condition_perc_correct[stim_condition])
                self.perc_correct_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                #self.perc_correct_cathode_ttest[stim_condition]=stats.ttest_rel(self.stim_condition_perc_correct['cathode'],
                #    self.stim_condition_perc_correct[stim_condition])
                T,p=stats.wilcoxon(self.stim_condition_perc_correct['cathode'],
                    self.stim_condition_perc_correct[stim_condition])
                self.perc_correct_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            self.perc_correct_mean[stim_condition]=np.mean(self.stim_condition_perc_correct[stim_condition])
            self.perc_correct_std[stim_condition]=np.std(self.stim_condition_perc_correct[stim_condition])/np.sqrt(len(self.stim_condition_perc_correct[stim_condition]))
            perc_correct_mean.append(self.perc_correct_mean[stim_condition])
            perc_correct_std_err.append(self.perc_correct_std[stim_condition])

        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax=fig.add_subplot(2,1,1)
            ax.bar(pos,perc_correct_mean, width=.5,yerr=perc_correct_std_err,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('% Correct')
            ax.set_ylim([50,90])
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

            fname=os.path.join(self.reports_dir,'img/perc_correct_anode_only')
            fig=Figure()
            pos = np.arange(2)+0.5
            ax=fig.add_subplot(2,1,1)
            ax.bar(pos,[self.perc_correct_mean['control'],self.perc_correct_mean['anode']],width=.5,
                yerr=[self.perc_correct_std['control'],self.perc_correct_std['anode']],align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(['Control','Anode'])
            ax.set_xlabel('Condition')
            ax.set_ylabel('% Correct')
            ax.set_ylim([50,90])
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)
            
        # Create perc no response plot
        furl='img/perc_no_response'
        fname=os.path.join(self.reports_dir,furl)
        self.perc_no_response_url='%s.png' % furl
        self.no_response_freidman=stats.friedmanchisquare(*self.stim_condition_no_response.values())
        self.no_response_control_wilcoxon={}
        self.no_response_anode_wilcoxon={}
        self.no_response_cathode_wilcoxon={}
        perc_no_response_mean=[]
        perc_no_response_std_err=[]
        self.perc_no_response_mean={}
        self.perc_no_response_std_err={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(self.stim_condition_no_response['control'],
                    self.stim_condition_no_response[stim_condition])
                self.no_response_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(self.stim_condition_no_response['anode'],
                    self.stim_condition_no_response[stim_condition])
                self.no_response_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(self.stim_condition_no_response['cathode'],
                    self.stim_condition_no_response[stim_condition])
                self.no_response_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            self.perc_no_response_mean[stim_condition]=np.mean(self.stim_condition_no_response[stim_condition])
            self.perc_no_response_std_err[stim_condition]=np.std(self.stim_condition_no_response[stim_condition])/np.sqrt(len(self.stim_condition_no_response[stim_condition]))
            perc_no_response_mean.append(self.perc_no_response_mean[stim_condition])
            perc_no_response_std_err.append(self.perc_no_response_std_err[stim_condition])
            #perc_no_response_std_err.append(np.std(self.stim_condition_no_response[stim_condition]))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))            
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax=fig.add_subplot(2,1,1)
            ax.bar(pos,perc_no_response_mean, width=.5,yerr=perc_no_response_std_err,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('% No Response')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create baseline rate plot
        furl='img/baseline_rate'
        fname=os.path.join(self.reports_dir,furl)
        self.baseline_rate_url='%s.png' % furl
        pyr_means=[]
        pyr_std_errs=[]
        inh_means=[]
        inh_sd_errs=[]
        self.baseline_pyr_means={}
        self.baseline_pyr_std_errs={}
        self.baseline_inh_means={}
        self.baseline_inh_std_errs={}
        stim_pyr_rates={}
        stim_inh_rates={}
        for stim_condition in self.stim_conditions:
            stim_pyr_rates[stim_condition],stim_inh_rates[stim_condition],trials=self.stim_condition_reports[stim_condition].compute_baseline_rates(0,100000)
            self.baseline_pyr_means[stim_condition]=np.mean(stim_pyr_rates[stim_condition])
            self.baseline_pyr_std_errs[stim_condition]=np.std(stim_pyr_rates[stim_condition])/np.sqrt(trials)
            pyr_means.append(self.baseline_pyr_means[stim_condition])
            pyr_std_errs.append(self.baseline_pyr_std_errs[stim_condition])
            self.baseline_inh_means[stim_condition]=np.mean(stim_inh_rates[stim_condition])
            self.baseline_inh_std_errs[stim_condition]=np.std(stim_inh_rates[stim_condition])/np.sqrt(trials)
            inh_means.append(self.baseline_inh_means[stim_condition])
            inh_sd_errs.append(self.baseline_inh_std_errs[stim_condition])
        for stim_condition in self.stim_conditions:
            stim_pyr_rates[stim_condition]=np.array(stim_pyr_rates[stim_condition])
            stim_inh_rates[stim_condition]=np.array(stim_inh_rates[stim_condition])
        self.baseline_pyr_friedman=stats.friedmanchisquare(*stim_pyr_rates.values())
        self.baseline_pyr_control_wilcoxon={}
        self.baseline_pyr_anode_wilcoxon={}
        self.baseline_pyr_cathode_wilcoxon={}
        self.baseline_inh_friedman=stats.friedmanchisquare(*stim_inh_rates.values())
        self.baseline_inh_control_wilcoxon={}
        self.baseline_inh_anode_wilcoxon={}
        self.baseline_inh_cathode_wilcoxon={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(stim_pyr_rates['control'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['control'], stim_inh_rates[stim_condition])
                self.baseline_inh_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(stim_pyr_rates['anode'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['anode'], stim_inh_rates[stim_condition])
                self.baseline_inh_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(stim_pyr_rates['cathode'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['cathode'], stim_inh_rates[stim_condition])
                self.baseline_inh_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax=fig.add_subplot(2,1,1)
            ax.bar(pos,pyr_means, width=.5,yerr=pyr_std_errs,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('Pyramidal Rate (Hz)')
            ax=fig.add_subplot(2,1,2)
            ax.bar(pos,inh_means, width=.5,yerr=inh_sd_errs,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('Interneuron Rate (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)


        # Create baseline rate plot
        furl='img/baseline_rate_small_ev_diff'
        fname=os.path.join(self.reports_dir,furl)
        self.baseline_rate_small_ev_diff_url='%s.png' % furl
        pyr_means=[]
        pyr_std_errs=[]
        inh_means=[]
        inh_sd_errs=[]
        self.baseline_pyr_small_ev_diff_means={}
        self.baseline_pyr_small_ev_diff_std_errs={}
        self.baseline_inh_small_ev_diff_means={}
        self.baseline_inh_small_ev_diff_std_errs={}
        stim_pyr_rates={}
        stim_inh_rates={}
        for stim_condition in self.stim_conditions:
            stim_pyr_rates[stim_condition],stim_inh_rates[stim_condition],trials=self.stim_condition_reports[stim_condition].compute_baseline_rates(ev_diff_bins[0],ev_diff_bins[5])
            self.baseline_pyr_small_ev_diff_means[stim_condition]=np.mean(stim_pyr_rates[stim_condition])
            self.baseline_pyr_small_ev_diff_std_errs[stim_condition]=np.std(stim_pyr_rates[stim_condition])/np.sqrt(trials)
            pyr_means.append(self.baseline_pyr_small_ev_diff_means[stim_condition])
            pyr_std_errs.append(self.baseline_pyr_small_ev_diff_std_errs[stim_condition])
            self.baseline_inh_small_ev_diff_means[stim_condition]=np.mean(stim_inh_rates[stim_condition])
            self.baseline_inh_small_ev_diff_std_errs[stim_condition]=np.std(stim_inh_rates[stim_condition])/np.sqrt(trials)
            inh_means.append(self.baseline_inh_small_ev_diff_means[stim_condition])
            inh_sd_errs.append(self.baseline_inh_small_ev_diff_std_errs[stim_condition])
        for stim_condition in self.stim_conditions:
            stim_pyr_rates[stim_condition]=np.array(stim_pyr_rates[stim_condition])
            stim_inh_rates[stim_condition]=np.array(stim_inh_rates[stim_condition])
        self.baseline_pyr_small_ev_diff_friedman=stats.friedmanchisquare(*stim_pyr_rates.values())
        self.baseline_pyr_small_ev_diff_control_wilcoxon={}
        self.baseline_pyr_small_ev_diff_anode_wilcoxon={}
        self.baseline_pyr_small_ev_diff_cathode_wilcoxon={}
        self.baseline_inh_small_ev_diff_friedman=stats.friedmanchisquare(*stim_inh_rates.values())
        self.baseline_inh_small_ev_diff_control_wilcoxon={}
        self.baseline_inh_small_ev_diff_anode_wilcoxon={}
        self.baseline_inh_small_ev_diff_cathode_wilcoxon={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(stim_pyr_rates['control'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_small_ev_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['control'], stim_inh_rates[stim_condition])
                self.baseline_inh_small_ev_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(stim_pyr_rates['anode'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_small_ev_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['anode'], stim_inh_rates[stim_condition])
                self.baseline_inh_small_ev_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(stim_pyr_rates['cathode'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_small_ev_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['cathode'], stim_inh_rates[stim_condition])
                self.baseline_inh_small_ev_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax=fig.add_subplot(2,1,1)
            ax.bar(pos,pyr_means, width=.5,yerr=pyr_std_errs,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('Pyramidal Rate (Hz)')
            ax=fig.add_subplot(2,1,2)
            ax.bar(pos,inh_means, width=.5,yerr=inh_sd_errs,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('Interneuron Rate (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create baseline rate plot
        furl='img/baseline_rate_large_ev_diff'
        fname=os.path.join(self.reports_dir,furl)
        self.baseline_rate_large_ev_diff_url='%s.png' % furl
        pyr_means=[]
        pyr_std_errs=[]
        inh_means=[]
        inh_sd_errs=[]
        self.baseline_pyr_large_ev_diff_means={}
        self.baseline_pyr_large_ev_diff_std_errs={}
        self.baseline_inh_large_ev_diff_means={}
        self.baseline_inh_large_ev_diff_std_errs={}
        stim_pyr_rates={}
        stim_inh_rates={}
        for stim_condition in self.stim_conditions:
            stim_pyr_rates[stim_condition],stim_inh_rates[stim_condition],trials=self.stim_condition_reports[stim_condition].compute_baseline_rates(ev_diff_bins[5],ev_diff_bins[-1])
            self.baseline_pyr_large_ev_diff_means[stim_condition]=np.mean(stim_pyr_rates[stim_condition])
            self.baseline_pyr_large_ev_diff_std_errs[stim_condition]=np.std(stim_pyr_rates[stim_condition])/np.sqrt(trials)
            pyr_means.append(self.baseline_pyr_large_ev_diff_means[stim_condition])
            pyr_std_errs.append(self.baseline_pyr_large_ev_diff_std_errs[stim_condition])
            self.baseline_inh_large_ev_diff_means[stim_condition]=np.mean(stim_inh_rates[stim_condition])
            self.baseline_inh_large_ev_diff_std_errs[stim_condition]=np.std(stim_inh_rates[stim_condition])/np.sqrt(trials)
            inh_means.append(self.baseline_inh_large_ev_diff_means[stim_condition])
            inh_sd_errs.append(self.baseline_inh_large_ev_diff_std_errs[stim_condition])
        for stim_condition in self.stim_conditions:
            stim_pyr_rates[stim_condition]=np.array(stim_pyr_rates[stim_condition])
            stim_inh_rates[stim_condition]=np.array(stim_inh_rates[stim_condition])
        self.baseline_pyr_large_ev_diff_friedman=stats.friedmanchisquare(*stim_pyr_rates.values())
        self.baseline_pyr_large_ev_diff_control_wilcoxon={}
        self.baseline_pyr_large_ev_diff_anode_wilcoxon={}
        self.baseline_pyr_large_ev_diff_cathode_wilcoxon={}
        self.baseline_inh_large_ev_diff_friedman=stats.friedmanchisquare(*stim_inh_rates.values())
        self.baseline_inh_large_ev_diff_control_wilcoxon={}
        self.baseline_inh_large_ev_diff_anode_wilcoxon={}
        self.baseline_inh_large_ev_diff_cathode_wilcoxon={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(stim_pyr_rates['control'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_large_ev_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['control'], stim_inh_rates[stim_condition])
                self.baseline_inh_large_ev_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(stim_pyr_rates['anode'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_large_ev_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['anode'], stim_inh_rates[stim_condition])
                self.baseline_inh_large_ev_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(stim_pyr_rates['cathode'], stim_pyr_rates[stim_condition])
                self.baseline_pyr_large_ev_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(stim_inh_rates['cathode'], stim_inh_rates[stim_condition])
                self.baseline_inh_large_ev_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax=fig.add_subplot(2,1,1)
            ax.bar(pos,pyr_means, width=.5,yerr=pyr_std_errs,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('Pyramidal Rate (Hz)')
            ax=fig.add_subplot(2,1,2)
            ax.bar(pos,inh_means, width=.5,yerr=inh_sd_errs,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('Interneuron Rate (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create baseline diff plot
        furl='img/baseline_diff_rate'
        fname=os.path.join(self.reports_dir,furl)
        self.baseline_diff_rate_url='%s.png' % furl
        self.baseline_diff_means={}
        self.baseline_diff_std_errs={}
        baseline_diff_means=[]
        baseline_diff_std_errs=[]
        stim_baseline_diffs={}
        for stim_condition in self.stim_conditions:
            stim_baseline_diffs[stim_condition],trials=self.stim_condition_reports[stim_condition].compute_baseline_diff_rates(ev_diff_bins[0],ev_diff_bins[3])
            self.baseline_diff_means[stim_condition]=np.mean(stim_baseline_diffs[stim_condition])
            self.baseline_diff_std_errs[stim_condition]=np.std(stim_baseline_diffs[stim_condition])/np.sqrt(trials)
            baseline_diff_means.append(self.baseline_diff_means[stim_condition])
            baseline_diff_std_errs.append(self.baseline_diff_std_errs[stim_condition])
        for stim_condition in self.stim_conditions:
            stim_baseline_diffs[stim_condition]=np.array(stim_baseline_diffs[stim_condition])

        self.baseline_diff_freidman=stats.friedmanchisquare(*stim_baseline_diffs.values())
        self.baseline_diff_control_wilcoxon={}
        self.baseline_diff_anode_wilcoxon={}
        self.baseline_diff_cathode_wilcoxon={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(stim_baseline_diffs['control'], stim_baseline_diffs[stim_condition])
                # get two sided p
                self.baseline_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(stim_baseline_diffs['anode'], stim_baseline_diffs[stim_condition])
                # get two sided p
                self.baseline_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(stim_baseline_diffs['cathode'], stim_baseline_diffs[stim_condition])
                # get two sided p
                self.baseline_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax=fig.add_subplot(1,1,1)
            ax.bar(pos,baseline_diff_means, width=.5,yerr=baseline_diff_std_errs,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_xlabel('Condition')
            ax.set_ylabel('Pyramidal Rate Diff (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create rate diff firing rate plot
        furl='img/firing_rate_diff'
        fname = os.path.join(self.reports_dir, furl)
        self.firing_rate_diff_url = '%s.png' % furl
        self.mean_diff_rates={}
        self.std_err_diff_rates={}
        mean_diff_rates=[]
        std_err_diff_rates=[]
        stim_diffs={}
        for stim_condition in self.stim_conditions:
            stim_diffs[stim_condition],trials=self.stim_condition_reports[stim_condition].compute_ev_diff_rates(0,100000)
            self.mean_diff_rates[stim_condition]=np.mean(stim_diffs[stim_condition])
            self.std_err_diff_rates[stim_condition]=np.std(stim_diffs[stim_condition])/np.sqrt(trials)
            mean_diff_rates.append(self.mean_diff_rates[stim_condition])
            std_err_diff_rates.append(self.std_err_diff_rates[stim_condition])
        for stim_condition in self.stim_conditions:
            stim_diffs[stim_condition]=np.array(stim_diffs[stim_condition])
        self.stim_diff_freidman=stats.friedmanchisquare(*stim_diffs.values())
        self.stim_diff_control_wilcoxon={}
        self.stim_diff_anode_wilcoxon={}
        self.stim_diff_cathode_wilcoxon={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(stim_diffs['control'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(stim_diffs['anode'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(stim_diffs['cathode'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            ax=fig.add_subplot(1,1,1)
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax.bar(pos,mean_diff_rates,width=.5,yerr=std_err_diff_rates,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_ylim([0,2])
            ax.set_xlabel('Condition')
            ax.set_ylabel('Firing Rate Diff (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create rate diff firing rate plot
        furl='img/firing_rate_diff_small_ev'
        fname = os.path.join(self.reports_dir, furl)
        self.firing_rate_diff_small_ev_diff_url = '%s.png' % furl
        self.mean_diff_rates_small_ev_diff={}
        self.std_err_diff_rates_small_ev_diff={}
        mean_diff_rates=[]
        std_err_diff_rates=[]
        stim_diffs={}
        for stim_condition in self.stim_conditions:
            stim_diffs[stim_condition],trials=self.stim_condition_reports[stim_condition].compute_ev_diff_rates(ev_diff_bins[0],ev_diff_bins[5])
            self.mean_diff_rates_small_ev_diff[stim_condition]=np.mean(stim_diffs[stim_condition])
            self.std_err_diff_rates_small_ev_diff[stim_condition]=np.std(stim_diffs[stim_condition])/np.sqrt(trials)
            mean_diff_rates.append(self.mean_diff_rates_small_ev_diff[stim_condition])
            std_err_diff_rates.append(self.std_err_diff_rates_small_ev_diff[stim_condition])
        for stim_condition in self.stim_conditions:
            stim_diffs[stim_condition]=np.array(stim_diffs[stim_condition])
        self.stim_diff_small_ev_diff_freidman=stats.friedmanchisquare(*stim_diffs.values())
        self.stim_diff_small_ev_diff_control_wilcoxon={}
        self.stim_diff_small_ev_diff_anode_wilcoxon={}
        self.stim_diff_small_ev_diff_cathode_wilcoxon={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(stim_diffs['control'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_small_ev_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(stim_diffs['anode'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_small_ev_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(stim_diffs['cathode'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_small_ev_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            ax=fig.add_subplot(1,1,1)
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax.bar(pos,mean_diff_rates,width=.5,yerr=std_err_diff_rates,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_ylim([0,2])
            ax.set_xlabel('Condition')
            ax.set_ylabel('Firing Rate Diff (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create rate diff firing rate plot
        furl='img/firing_rate_diff_large_ev'
        fname = os.path.join(self.reports_dir, furl)
        self.firing_rate_diff_large_ev_diff_url = '%s.png' % furl
        self.mean_diff_rates_large_ev_diff={}
        self.std_err_diff_rates_large_ev_diff={}
        mean_diff_rates=[]
        std_err_diff_rates=[]
        stim_diffs={}
        for stim_condition in self.stim_conditions:
            stim_diffs[stim_condition],trials=self.stim_condition_reports[stim_condition].compute_ev_diff_rates(ev_diff_bins[5],ev_diff_bins[-1])
            self.mean_diff_rates_large_ev_diff[stim_condition]=np.mean(stim_diffs[stim_condition])
            self.std_err_diff_rates_large_ev_diff[stim_condition]=np.std(stim_diffs[stim_condition])/np.sqrt(trials)
            mean_diff_rates.append(self.mean_diff_rates_large_ev_diff[stim_condition])
            std_err_diff_rates.append(self.std_err_diff_rates_large_ev_diff[stim_condition])
        for stim_condition in self.stim_conditions:
            stim_diffs[stim_condition]=np.array(stim_diffs[stim_condition])
        self.stim_diff_large_ev_diff_freidman=stats.friedmanchisquare(*stim_diffs.values())
        self.stim_diff_large_ev_diff_control_wilcoxon={}
        self.stim_diff_large_ev_diff_anode_wilcoxon={}
        self.stim_diff_large_ev_diff_cathode_wilcoxon={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                T,p=stats.wilcoxon(stim_diffs['control'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_large_ev_diff_control_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(stim_diffs['anode'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_large_ev_diff_anode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(stim_diffs['cathode'], stim_diffs[stim_condition])
                # get two sided p
                self.stim_diff_large_ev_diff_cathode_wilcoxon[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
        if regenerate_plots:
            fig=Figure(figsize=(20,6))
            ax=fig.add_subplot(1,1,1)
            pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
            ax.bar(pos,mean_diff_rates,width=.5,yerr=std_err_diff_rates,align='center',ecolor='k')
            ax.set_xticks(pos)
            ax.set_xticklabels(self.stim_conditions)
            ax.set_ylim([0,2])
            ax.set_xlabel('Condition')
            ax.set_ylabel('Firing Rate Diff (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/ev_dff_pyr_firing_rate'
        fname=os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_chosen_rate_std_err[stim_condition], condition_colors[stim_condition],
                        None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_chosen_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_inh_rate_means[stim_condition],
                        self.anode_stim_condition_inh_rate_std_err[stim_condition], condition_colors[stim_condition],
                        None, '%s' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_inh_rate_std_err[stim_condition], condition_colors[stim_condition],
                        None, '%s' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_anode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_pyr_firing_rate_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_chosen_rate_std_err[stim_condition], None, None,
                        '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_cathode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_pyr_firing_rate_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_chosen_rate_std_err[stim_condition], None, None, '%s, chosen' % stim_condition,
                        .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_anode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_inh_firing_rate_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_inh_rate_means[stim_condition],
                        self.anode_stim_condition_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_cathode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_inh_firing_rate_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/small_ev_diff_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_small_ev_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_small_ev_chosen_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_small_ev_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_small_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 
                        'dashed', '%s, unchosen' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_small_ev_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_small_ev_chosen_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_small_ev_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_small_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 
                        'dashed', '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_ylim([-5,40])
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/med_ev_diff_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_med_ev_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_med_ev_chosen_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_med_ev_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_med_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(),
                        'dashed', '%s, unchosen' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_med_ev_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_med_ev_chosen_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_med_ev_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_med_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(),
                        'dashed', '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_ylim([-5,40])
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/large_ev_diff_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_pyr_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_large_ev_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_large_ev_chosen_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_large_ev_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_large_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(),
                        'dashed', '%s, unchosen' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_large_ev_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_large_ev_chosen_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_large_ev_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_large_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(),
                        'dashed', '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_ylim([-5,40])
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/small_ev_diff_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_small_ev_inh_rate_means[stim_condition],
                        self.anode_stim_condition_small_ev_inh_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_small_ev_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_small_ev_inh_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s, chosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_ylim([-1,8])
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/med_ev_diff_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_med_ev_inh_rate_means[stim_condition],
                        self.anode_stim_condition_med_ev_inh_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_med_ev_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_med_ev_inh_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_ylim([-1,8])
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/large_ev_diff_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_inh_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_large_ev_inh_rate_means[stim_condition],
                        self.anode_stim_condition_large_ev_inh_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s' % stim_condition, .5*ms)
                elif stim_condition=='cathode':
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_large_ev_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_large_ev_inh_rate_std_err[stim_condition],
                        condition_colors[stim_condition], None, '%s' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_ylim([-1,8])
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)
            
        # Create ev diff firing rate plot
        furl='img/small_ev_diff_anode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_pyr_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_small_ev_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_small_ev_chosen_rate_std_err[stim_condition], None, None, '%s, chosen' % stim_condition,
                        .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_small_ev_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_small_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/small_ev_diff_cathode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_pyr_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_small_ev_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_small_ev_chosen_rate_std_err[stim_condition], None, None, '%s, chosen' % stim_condition,
                        .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_small_ev_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_small_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/small_ev_diff_anode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_inh_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_small_ev_inh_rate_means[stim_condition],
                        self.anode_stim_condition_small_ev_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/small_ev_diff_cathode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_inh_firing_rate_small_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_small_ev_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_small_ev_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/med_ev_diff_anode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_pyr_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_med_ev_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_med_ev_chosen_rate_std_err[stim_condition], None, None, '%s, chosen' % stim_condition,
                        .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_med_ev_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_med_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/med_ev_diff_cathode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_pyr_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_med_ev_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_med_ev_chosen_rate_std_err[stim_condition], None, None, '%s, chosen' % stim_condition,
                        .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_med_ev_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_med_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/med_ev_diff_anode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_inh_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_med_ev_inh_rate_means[stim_condition],
                        self.anode_stim_condition_med_ev_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/med_ev_diff_cathode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_inh_firing_rate_med_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_med_ev_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_med_ev_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/large_ev_diff_anode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_pyr_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_large_ev_chosen_rate_means[stim_condition],
                        self.anode_stim_condition_large_ev_chosen_rate_std_err[stim_condition], None, None, '%s, chosen' % stim_condition,
                        .5*ms)
                    plot_mean_rate(ax, self.anode_stim_condition_large_ev_unchosen_rate_means[stim_condition],
                        self.anode_stim_condition_large_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/large_ev_diff_cathode_pyr_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_pyr_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_large_ev_chosen_rate_means[stim_condition],
                        self.cathode_stim_condition_large_ev_chosen_rate_std_err[stim_condition], None, None, '%s, chosen' % stim_condition,
                        .5*ms)
                    plot_mean_rate(ax, self.cathode_stim_condition_large_ev_unchosen_rate_means[stim_condition],
                        self.cathode_stim_condition_large_ev_unchosen_rate_std_err[stim_condition], baseline.get_color(), 'dashed',
                        '%s, unchosen' % stim_condition, .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/large_ev_diff_anode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.anode_mean_inh_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('anode'):
                    baseline=plot_mean_rate(ax, self.anode_stim_condition_large_ev_inh_rate_means[stim_condition],
                        self.anode_stim_condition_large_ev_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/large_ev_diff_cathode_inh_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.cathode_mean_inh_firing_rate_large_ev_diff_url = '%s.png' % furl
        if regenerate_plots:
            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition.startswith('cathode'):
                    baseline=plot_mean_rate(ax, self.cathode_stim_condition_large_ev_inh_rate_means[stim_condition],
                        self.cathode_stim_condition_large_ev_inh_rate_std_err[stim_condition], None, None, '%s' % stim_condition,
                        .5*ms)
            ax.set_xlabel('Time')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.legend(loc=0)
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create alpha - % correct plot
        furl='img/alpha_perc_correct'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_perc_correct_url = '%s.png' % furl
        if regenerate_plots:
            all_condition_alphas=[]
            all_condition_perc_correct=[]
            fig=Figure(figsize=(16,12))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                baseline,=ax.plot(self.stim_condition_reports[stim_condition].condition_alphas,
                    self.stim_condition_reports[stim_condition].condition_perc_correct/100.0,'o')
                min_x=np.min(self.stim_condition_reports[stim_condition].condition_alphas)-0.1
                max_x=np.max(self.stim_condition_reports[stim_condition].condition_alphas)+0.1
                ax.plot([min_x, max_x], [self.stim_condition_reports[stim_condition].alpha_perc_correct_a * min_x +
                                         self.stim_condition_reports[stim_condition].alpha_perc_correct_b,
                                         self.stim_condition_reports[stim_condition].alpha_perc_correct_a * max_x +
                                         self.stim_condition_reports[stim_condition].alpha_perc_correct_b],
                    label='%s r^2=%.3f' % (stim_condition,self.stim_condition_reports[stim_condition].alpha_perc_correct_r_sqr),
                    color=baseline.get_color())

                all_condition_alphas.extend(self.stim_condition_reports[stim_condition].condition_alphas)
                all_condition_perc_correct.extend(self.stim_condition_reports[stim_condition].condition_perc_correct/100.0)
            ax.legend(loc=0)
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create alpha - % correct plot
        furl='img/alpha_main_perc_correct'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_main_perc_correct_url = '%s.png' % furl
        if regenerate_plots:
            #fig=Figure(figsize=(16,12))
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode' or stim_condition=='cathode':
                    baseline,=ax.plot(self.stim_condition_reports[stim_condition].condition_alphas,
                        self.stim_condition_reports[stim_condition].condition_perc_correct/100.0,
                        'o%s' % condition_colors[stim_condition])
                    min_x=np.min(self.stim_condition_reports[stim_condition].condition_alphas)-0.1
                    max_x=np.max(self.stim_condition_reports[stim_condition].condition_alphas)+0.1
                    ax.plot([min_x, max_x], [self.stim_condition_reports[stim_condition].alpha_perc_correct_a * min_x +
                                             self.stim_condition_reports[stim_condition].alpha_perc_correct_b,
                                             self.stim_condition_reports[stim_condition].alpha_perc_correct_a * max_x +
                                             self.stim_condition_reports[stim_condition].alpha_perc_correct_b],
                        label='%s r^2=%.3f' % (stim_condition,self.stim_condition_reports[stim_condition].alpha_perc_correct_r_sqr),
                        color=baseline.get_color())

            ax.legend(loc=0)
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create alpha - % correct plot
        furl='img/alpha_main_perc_correct_anode_only'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_main_perc_correct_anode_only_url = '%s.png' % furl
        if regenerate_plots:
            #fig=Figure(figsize=(16,12))
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    baseline,=ax.plot(self.stim_condition_reports[stim_condition].condition_alphas,
                        self.stim_condition_reports[stim_condition].condition_perc_correct/100.0,
                        'o%s' % condition_colors[stim_condition])
                    min_x=np.min(self.stim_condition_reports[stim_condition].condition_alphas)-0.1
                    max_x=np.max(self.stim_condition_reports[stim_condition].condition_alphas)+0.1
                    ax.plot([min_x, max_x], [self.stim_condition_reports[stim_condition].alpha_perc_correct_a * min_x +
                                             self.stim_condition_reports[stim_condition].alpha_perc_correct_b,
                                             self.stim_condition_reports[stim_condition].alpha_perc_correct_a * max_x +
                                             self.stim_condition_reports[stim_condition].alpha_perc_correct_b],
                        label='%s r^2=%.3f' % (stim_condition,self.stim_condition_reports[stim_condition].alpha_perc_correct_r_sqr),
                        color=baseline.get_color())

            ax.legend(loc=0)
            ax.set_xlim([0.0,1.0])
            ax.set_ylim([0.5,1.0])
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)
        
        furl='img/all_alpha_perc_corect'
        fname = os.path.join(self.reports_dir, furl)
        self.all_alpha_perc_correct_url = '%s.png' % furl
        if regenerate_plots:
            all_condition_alphas_vec=np.zeros([len(all_condition_alphas),1])
            all_condition_alphas_vec[:]=all_condition_alphas
            all_condition_perc_correct_vec=np.zeros([len(all_condition_perc_correct),1])
            all_condition_perc_correct_vec[:]=all_condition_perc_correct
            clf = LinearRegression()
            clf.fit(all_condition_alphas_vec, all_condition_perc_correct_vec)
            self.alpha_perc_correct_a = clf.coef_[0][0]
            self.alpha_perc_correct_b = clf.intercept_[0]
            self.alpha_perc_correct_r_sqr=clf.score(all_condition_alphas_vec, all_condition_perc_correct_vec)
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(all_condition_alphas_vec, all_condition_perc_correct,'o')
            min_x=np.min(all_condition_alphas_vec)-.1
            max_x=np.max(all_condition_alphas_vec)+.1
            ax.plot([min_x, max_x], [self.alpha_perc_correct_a * min_x + self.alpha_perc_correct_b,
                                     self.alpha_perc_correct_a * max_x + self.alpha_perc_correct_b],
                label='r^2=%.3f' % self.alpha_perc_correct_r_sqr)
            ax.legend(loc=0)
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)
        
        # Create beta - % correct plot
        furl='img/beta_perc_correct'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_perc_correct_url = '%s.png' % furl
        if regenerate_plots:
            all_condition_betas=[]
            fig=Figure(figsize=(16,12))
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                d = np.abs(self.stim_condition_reports[stim_condition].condition_betas - np.median(self.stim_condition_reports[stim_condition].condition_betas))
                mdev = np.median(d)
                s = d/mdev if mdev else 0
                filtered_betas=self.stim_condition_reports[stim_condition].condition_betas[s<2]
                filtered_perc_correct=self.stim_condition_reports[stim_condition].condition_perc_correct[s<2]/100.0
                baseline,=ax.plot(filtered_betas, filtered_perc_correct,'o')
                min_x=np.min(filtered_betas)-1.0
                max_x=np.max(filtered_betas)+1.0
                ax.plot([min_x, max_x], [self.stim_condition_reports[stim_condition].beta_perc_correct_a * min_x +
                                         self.stim_condition_reports[stim_condition].beta_perc_correct_b,
                                         self.stim_condition_reports[stim_condition].beta_perc_correct_a * max_x +
                                         self.stim_condition_reports[stim_condition].beta_perc_correct_b],
                    label='%s r^2=%.3f' % (stim_condition,self.stim_condition_reports[stim_condition].beta_perc_correct_r_sqr),
                    color=baseline.get_color())
                all_condition_betas.extend(self.stim_condition_reports[stim_condition].condition_betas)
            ax.legend(loc=0)
            ax.set_xlabel('Beta')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create beta - % correct plot
        furl='img/beta_main_perc_correct'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_main_perc_correct_url = '%s.png' % furl
        if regenerate_plots:
            #fig=Figure(figsize=(16,12))
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode' or stim_condition=='cathode':
                    d = np.abs(self.stim_condition_reports[stim_condition].condition_betas - np.median(self.stim_condition_reports[stim_condition].condition_betas))
                    mdev = np.median(d)
                    s = d/mdev if mdev else 0
                    filtered_betas=self.stim_condition_reports[stim_condition].condition_betas[s<2]
                    filtered_perc_correct=self.stim_condition_reports[stim_condition].condition_perc_correct[s<2]/100.0
                    baseline,=ax.plot(filtered_betas, filtered_perc_correct,'o%s' % condition_colors[stim_condition])
                    min_x=np.min(filtered_betas)-1.0
                    max_x=np.max(filtered_betas)+1.0
                    ax.plot([min_x, max_x], [self.stim_condition_reports[stim_condition].beta_perc_correct_a * min_x +
                                             self.stim_condition_reports[stim_condition].beta_perc_correct_b,
                                             self.stim_condition_reports[stim_condition].beta_perc_correct_a * max_x +
                                             self.stim_condition_reports[stim_condition].beta_perc_correct_b],
                        label='%s r^2=%.3f' % (stim_condition,self.stim_condition_reports[stim_condition].beta_perc_correct_r_sqr),
                        color=baseline.get_color())
            ax.legend(loc=0)
            ax.set_xlabel('Beta')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create beta - % correct plot
        furl='img/beta_main_perc_correct_anode_only'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_main_perc_correct_anode_only_url = '%s.png' % furl
        if regenerate_plots:
            #fig=Figure(figsize=(16,12))
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode':
                    d = np.abs(self.stim_condition_reports[stim_condition].condition_betas - np.median(self.stim_condition_reports[stim_condition].condition_betas))
                    mdev = np.median(d)
                    s = d/mdev if mdev else 0
                    filtered_betas=self.stim_condition_reports[stim_condition].condition_betas[s<2]
                    filtered_perc_correct=self.stim_condition_reports[stim_condition].condition_perc_correct[s<2]/100.0
                    baseline,=ax.plot(filtered_betas, filtered_perc_correct,'o%s' % condition_colors[stim_condition])
                    min_x=np.min(filtered_betas)-1.0
                    max_x=np.max(filtered_betas)+1.0
                    ax.plot([min_x, max_x], [self.stim_condition_reports[stim_condition].beta_perc_correct_a * min_x +
                                             self.stim_condition_reports[stim_condition].beta_perc_correct_b,
                                             self.stim_condition_reports[stim_condition].beta_perc_correct_a * max_x +
                                             self.stim_condition_reports[stim_condition].beta_perc_correct_b],
                        label='%s r^2=%.3f' % (stim_condition,self.stim_condition_reports[stim_condition].beta_perc_correct_r_sqr),
                        color=baseline.get_color())
            ax.legend(loc=0)
            ax.set_xlim([0,15])
            ax.set_ylim([0.5,1.0])
            ax.set_xlabel('Beta')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/all_beta_perc_corect'
        fname = os.path.join(self.reports_dir, furl)
        self.all_beta_perc_correct_url = '%s.png' % furl
        if regenerate_plots:
            all_condition_betas_vec=np.zeros([len(all_condition_betas),1])
            all_condition_betas_vec[:]=all_condition_betas
            all_condition_perc_correct_vec=np.zeros([len(all_condition_perc_correct),1])
            all_condition_perc_correct_vec[:]=all_condition_perc_correct
            clf = LinearRegression()
            clf.fit(all_condition_betas_vec, all_condition_perc_correct_vec)
            self.beta_perc_correct_a = clf.coef_[0][0]
            self.beta_perc_correct_b = clf.intercept_[0]
            self.beta_perc_correct_r_sqr=clf.score(all_condition_betas_vec, all_condition_perc_correct_vec)
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(all_condition_betas, all_condition_perc_correct_vec,'o')
            min_x=np.min(all_condition_betas)-1
            max_x=np.max(all_condition_betas)+1
            ax.plot([min_x, max_x], [self.beta_perc_correct_a * min_x + self.beta_perc_correct_b,
                                     self.beta_perc_correct_a * max_x + self.beta_perc_correct_b],
                label='r^2=%.3f' % self.beta_perc_correct_r_sqr)
            ax.legend(loc=0)
            ax.set_xlabel('Beta')
            ax.set_ylabel('Prop Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)
        
        self.num_trials=self.stim_condition_reports['control'].sessions[0].num_trials
        self.alpha=self.stim_condition_reports['control'].sessions[0].alpha

        self.num_groups=self.stim_condition_reports['control'].sessions[0].num_groups
        self.trial_duration=self.stim_condition_reports['control'].sessions[0].trial_duration
        self.wta_params=self.stim_condition_reports['control'].sessions[0].wta_params

        condition_alphas={}
        condition_betas={}
        for stim_condition in self.stim_conditions:
            condition_alphas[stim_condition]=np.squeeze(self.stim_condition_reports[stim_condition].condition_alphas)
            condition_betas[stim_condition]=np.squeeze(self.stim_condition_reports[stim_condition].condition_betas)

        self.stim_alpha_change_urls={}
        self.stim_beta_change_urls={}
        self.stim_alpha_mean_change={}
        self.stim_alpha_std_change={}
        self.stim_beta_mean_change={}
        self.stim_beta_std_change={}
        self.alpha_friedman=stats.friedmanchisquare(*condition_alphas.values())
        self.beta_friedman=stats.friedmanchisquare(*condition_betas.values())
        self.alpha_wilcoxon_test={}
        self.beta_wilcoxon_test={}
        self.anode_alpha_wilcoxon_test={}
        self.anode_beta_wilcoxon_test={}
        self.cathode_alpha_wilcoxon_test={}
        self.cathode_beta_wilcoxon_test={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                # Create alpha plot
                furl='img/%s_alpha' % stim_condition
                fname = os.path.join(self.reports_dir, furl)
                self.stim_alpha_change_urls[stim_condition] = '%s.png' % furl
                if regenerate_plots:
                    fig=plot_param_diff(stim_condition,'alpha',
                        self.stim_condition_reports['control'].condition_alphas,
                        self.stim_condition_reports[stim_condition].condition_alphas,(0.0,1.0))
                        #(-1.0,1.0))
                    save_to_png(fig, '%s.png' % fname)
                    save_to_eps(fig, '%s.eps' % fname)
                    plt.close(fig)

                    fname = os.path.join(self.reports_dir, 'img/%s_alpha_bar' % stim_condition)
                    fig=plot_params(stim_condition,'alpha',self.stim_condition_reports['control'].condition_alphas,
                        self.stim_condition_reports[stim_condition].condition_alphas)
                    save_to_png(fig, '%s.png' % fname)
                    save_to_eps(fig, '%s.eps' % fname)
                    plt.close(fig)

                alpha_diff=self.stim_condition_reports[stim_condition].condition_alphas-\
                           self.stim_condition_reports['control'].condition_alphas
                filtered_alpha_diffs=reject_outliers(alpha_diff)
                self.stim_alpha_mean_change[stim_condition]=np.mean(filtered_alpha_diffs)
                self.stim_alpha_std_change[stim_condition]=np.std(filtered_alpha_diffs)
                T,p=stats.wilcoxon(condition_alphas['control'], condition_alphas[stim_condition])
                self.alpha_wilcoxon_test[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))

                # Create beta plot
                furl='img/%s_beta' % stim_condition
                fname = os.path.join(self.reports_dir, furl)
                self.stim_beta_change_urls[stim_condition] = '%s.png' % furl
                if regenerate_plots:
                    fig=plot_param_diff(stim_condition,'beta',self.stim_condition_reports['control'].condition_betas,
                        self.stim_condition_reports[stim_condition].condition_betas,(0,15.0))
                        #(-10.0, 10.0))
                    save_to_png(fig, '%s.png' % fname)
                    save_to_eps(fig, '%s.eps' % fname)
                    plt.close(fig)

                    fname = os.path.join(self.reports_dir, 'img/%s_beta_bar' % stim_condition)
                    fig=plot_params(stim_condition,'alpha',self.stim_condition_reports['control'].condition_betas,
                        self.stim_condition_reports[stim_condition].condition_betas)
                    save_to_png(fig, '%s.png' % fname)
                    save_to_eps(fig, '%s.eps' % fname)
                    plt.close(fig)

                beta_diff=self.stim_condition_reports[stim_condition].condition_betas-\
                           self.stim_condition_reports['control'].condition_betas
                filtered_beta_diffs=reject_outliers(beta_diff)
                self.stim_beta_mean_change[stim_condition]=np.mean(filtered_beta_diffs)
                self.stim_beta_std_change[stim_condition]=np.std(filtered_beta_diffs)
                T,p=stats.wilcoxon(condition_betas['control'], condition_betas[stim_condition])
                self.beta_wilcoxon_test[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))

        for stim_condition in self.stim_conditions:
            if stim_condition.startswith('anode_control'):
                T,p=stats.wilcoxon(condition_alphas['anode'], condition_alphas[stim_condition])
                self.anode_alpha_wilcoxon_test[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(condition_betas['anode'], condition_betas[stim_condition])
                self.anode_beta_wilcoxon_test[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
            elif stim_condition.startswith('cathode_control'):
                T,p=stats.wilcoxon(condition_alphas['cathode'], condition_alphas[stim_condition])
                self.cathode_alpha_wilcoxon_test[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))
                T,p=stats.wilcoxon(condition_betas['cathode'], condition_betas[stim_condition])
                self.cathode_beta_wilcoxon_test[stim_condition]=(T,p*num_comparisons,norm.isf(np.min([1.0,p*num_comparisons/2.0])))

        ratio_furl='img/rate_diff_ratio_perc_correct'
        ratio_fname = os.path.join(self.reports_dir,ratio_furl)
        self.rate_diff_ratio_perc_correct_url='%s.png' % ratio_furl

        logistic_furl='img/rate_diff_perc_correct_logistic'
        logistic_fname = os.path.join(self.reports_dir,logistic_furl)
        self.rate_diff_perc_correct_logistic_url='%s.png' % logistic_furl

        logistic_noise_only_furl='img/rate_diff_perc_correct_logistic_noise_only'
        logistic_noise_only_fname = os.path.join(self.reports_dir,logistic_noise_only_furl)
        self.rate_diff_perc_correct_logistic_noise_only_url='%s.png' % logistic_noise_only_furl

        if regenerate_plots:
            rate_diff_ratios=[]
            correct=[]
            ev_diffs=[]

            all_biases={}
            all_ev_diffs={}
            all_correct={}
            
            for stim_condition in self.stim_conditions:
                if stim_condition=='control' or stim_condition=='anode' or stim_condition=='cathode':
                    stim_report=self.stim_condition_reports[stim_condition]
                    for virtual_subj_id in range(stim_report.num_subjects):
                        if not virtual_subj_id in stim_report.excluded_sessions:
                            session_prefix=self.file_prefix % (virtual_subj_id,stim_condition)
                            session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                            data=FileInfo(session_report_file)
                            for trial in range(len(data.trial_e_rates)):
                                if data.choice[trial]>-1 and np.abs(data.inputs[data.choice[trial],trial]-data.inputs[1-data.choice[trial],trial])>0:
                                    chosen_mean=np.mean(data.trial_e_rates[trial][data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                                    unchosen_mean=np.mean(data.trial_e_rates[trial][1-data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                                    bias=np.abs(chosen_mean-unchosen_mean)
                                    ev_diff=np.abs(data.inputs[data.choice[trial],trial]-data.inputs[1-data.choice[trial],trial])
                                    ratio=bias/ev_diff
                                    choice_correct=0.0
                                    if (data.choice[trial]==0 and data.inputs[0,trial]>data.inputs[1,trial]) or (data.choice[trial]==1 and data.inputs[1,trial]>data.inputs[0,trial]):
                                        choice_correct=1.0
                                    if ratio<1.9:
                                        rate_diff_ratios.append(ratio)
                                        correct.append(choice_correct)
                                        ev_diffs.append(ev_diff)
                                    if not stim_condition in all_biases:
                                        all_biases[stim_condition]=[]
                                        all_ev_diffs[stim_condition]=[]
                                        all_correct[stim_condition]=[]
                                    all_biases[stim_condition].append(bias)
                                    all_ev_diffs[stim_condition].append(ev_diff)
                                    all_correct[stim_condition].append(choice_correct)

            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ev_diffs=np.array(ev_diffs)
            rate_diff_ratios=np.array(rate_diff_ratios)
            correct=np.array(correct)
            hist,bins=np.histogram(ev_diffs, bins=10)
            ev_plot_diffs=[]
            ev_perc_correct=[]
            for i in range(10):
                trials=np.where((ev_diffs>=bins[i]) & (ev_diffs<bins[i+1]))[0]
                ev_plot_diffs.append(np.mean(rate_diff_ratios[trials]))
                ev_perc_correct.append(np.mean(correct[trials]))
            ev_plot_diffs=np.reshape(np.array(ev_plot_diffs),(len(ev_plot_diffs),1))
            ev_perc_correct=np.reshape(np.array(ev_perc_correct),(len(ev_perc_correct),1))*100.0
            clf = LinearRegression()
            clf.fit(ev_plot_diffs, ev_perc_correct)
            self.rate_diff_ratio_perc_correct_a = clf.coef_[0][0]
            self.rate_diff_ratio_perc_correct_b = clf.intercept_[0]
            self.rate_diff_ratio_perc_correct_r_sqr=clf.score(ev_plot_diffs, ev_perc_correct)
            ax.plot(ev_plot_diffs, ev_perc_correct,'o')
            min_x=np.min(ev_plot_diffs)
            max_x=np.max(ev_plot_diffs)
            ax.plot([min_x, max_x], [self.rate_diff_ratio_perc_correct_a * min_x + self.rate_diff_ratio_perc_correct_b,
                                     self.rate_diff_ratio_perc_correct_a * max_x + self.rate_diff_ratio_perc_correct_b],
                label='r^2=%.3f' % self.rate_diff_ratio_perc_correct_r_sqr)
            ax.legend(loc='best')
            ax.set_xlabel('prestim bias/input diff')
            ax.set_ylabel('% correct')
            save_to_png(fig, '%s.png' % ratio_fname)
            save_to_eps(fig, '%s.eps' % ratio_fname)
            plt.close(fig)

            width=0.3

            fig=Figure(figsize=(16,6))
            ax=fig.add_subplot(1,1,1)
            stim_conditions=['control','anode','cathode']
            ind=np.array(range(1,7))
            rects=[]
            for idx,stim_condition in enumerate(stim_conditions):
                num_trials=len(all_biases[stim_condition])
                x=np.zeros((num_trials,2))
                x[:,0]=np.array(all_biases[stim_condition])
                x[:,1]=np.array(all_ev_diffs[stim_condition])
                print(np.cov(np.transpose(x))[0,1])
                y=np.array(all_correct[stim_condition])
                coeffs=np.zeros((2,100))
                for i in range(100):
                    permute_trials=np.random.permutation(range(num_trials))
                    logit = LogisticRegression()
                    logit = logit.fit(x[permute_trials[0:int(num_trials/2.0)],:], y[permute_trials[0:int(num_trials/2)]])
                    coeffs[:,i]=logit.coef_[0]
                rect=ax.bar(np.array([1,2])+width*.5+(idx-1)*width, np.mean(coeffs,axis=1), width,
                    yerr=np.std(coeffs,axis=1), color=condition_colors[stim_condition])
                rects.append(rect)
            for idx,stim_condition in enumerate(stim_conditions):
                small_ev_diff_biases=[]
                small_ev_diff_ev_diffs=[]
                small_ev_diff_correct=[]
                for i in range(len(all_ev_diffs[stim_condition])):
                    if all_ev_diffs[stim_condition][i]<=bins[5]:
                        small_ev_diff_biases.append(all_biases[stim_condition][i])
                        small_ev_diff_ev_diffs.append(all_ev_diffs[stim_condition][i])
                        small_ev_diff_correct.append(all_correct[stim_condition][i])
                num_trials=len(small_ev_diff_biases)
                x=np.zeros((num_trials,2))
                x[:,0]=np.array(small_ev_diff_biases)
                x[:,1]=np.array(small_ev_diff_ev_diffs)
                y=np.array(small_ev_diff_correct)
                coeffs=np.zeros((2,100))
                for i in range(100):
                    permute_trials=np.random.permutation(range(num_trials))
                    logit = LogisticRegression()
                    logit = logit.fit(x[permute_trials[0:int(num_trials/2.0)],:], y[permute_trials[0:int(num_trials/2)]])
                    coeffs[:,i]=logit.coef_[0]
                rect=ax.bar(np.array([3,4])+width*.5+(idx-1)*width, np.mean(coeffs,axis=1), width,
                    yerr=np.std(coeffs,axis=1),color=condition_colors[stim_condition])
            for idx,stim_condition in enumerate(stim_conditions):
                large_ev_diff_biases=[]
                large_ev_diff_ev_diffs=[]
                large_ev_diff_correct=[]
                for i in range(len(all_ev_diffs[stim_condition])):
                    if all_ev_diffs[stim_condition][i]>bins[5]:
                        large_ev_diff_biases.append(all_biases[stim_condition][i])
                        large_ev_diff_ev_diffs.append(all_ev_diffs[stim_condition][i])
                        large_ev_diff_correct.append(all_correct[stim_condition][i])
                num_trials=len(large_ev_diff_biases)
                x=np.zeros((num_trials,2))
                x[:,0]=np.array(large_ev_diff_biases)
                x[:,1]=np.array(large_ev_diff_ev_diffs)
                y=np.array(large_ev_diff_correct)
                coeffs=np.zeros((2,100))
                for i in range(100):
                    permute_trials=np.random.permutation(range(num_trials))
                    logit=LogisticRegression()
                    logit=logit.fit(x[permute_trials[0:int(num_trials/2.0)],:],y[permute_trials[0:int(num_trials/2)]])
                    coeffs[:,i]=logit.coef_[0]
                rect=ax.bar(np.array([5,6])+width*.5+(idx-1)*width, np.mean(coeffs,axis=1), width,
                    yerr=np.std(coeffs,axis=1),color=condition_colors[stim_condition])
            ax.set_ylabel('Coefficient')
            ax.set_xticks(ind+width)
            ax.set_xticklabels(['Bias (overall)','EV Diff (overall)', 'Bias (small EV diff)', 'EV Diff (small EV diff)', 'Bias (large EV diff)', 'EV Diff (large EV diff)'])
            ax.legend([rect[0] for rect in rects],stim_conditions,loc='best')
            save_to_png(fig, '%s.png' % logistic_fname)
            save_to_eps(fig, '%s.eps' % logistic_fname)
            plt.close(fig)

            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ind=np.array(range(1,4))
            stim_conditions=['control','anode','cathode']
            rects=[]
            for idx,stim_condition in enumerate(stim_conditions):
                num_trials=len(all_biases[stim_condition])
                x=np.zeros((num_trials,1))
                x[:,0]=np.array(all_biases[stim_condition])
                y=np.array(all_correct[stim_condition])
                coeffs=np.zeros((1,100))
                for i in range(100):
                    permute_trials=np.random.permutation(range(num_trials))
                    logit = LogisticRegression()
                    logit = logit.fit(x[permute_trials[0:int(num_trials/2.0)],:], y[permute_trials[0:int(num_trials/2)]])
                    coeffs[:,i]=logit.coef_[0]
                rect=ax.bar(1+width*.5+(idx-1)*width, np.mean(coeffs,axis=1), width,
                    yerr=np.std(coeffs,axis=1), color=condition_colors[stim_condition])
                rects.append(rect)
            for idx,stim_condition in enumerate(stim_conditions):
                small_ev_diff_biases=[]
                small_ev_diff_ev_diffs=[]
                small_ev_diff_correct=[]
                for i in range(len(all_ev_diffs[stim_condition])):
                    if all_ev_diffs[stim_condition][i]<=bins[5]:
                        small_ev_diff_biases.append(all_biases[stim_condition][i])
                        small_ev_diff_ev_diffs.append(all_ev_diffs[stim_condition][i])
                        small_ev_diff_correct.append(all_correct[stim_condition][i])
                num_trials=len(small_ev_diff_biases)
                x=np.zeros((num_trials,1))
                x[:,0]=np.array(small_ev_diff_biases)
                y=np.array(small_ev_diff_correct)
                coeffs=np.zeros((1,100))
                for i in range(100):
                    permute_trials=np.random.permutation(range(num_trials))
                    logit=LogisticRegression()
                    logit=logit.fit(x[permute_trials[0:int(num_trials/2.0)],:], y[permute_trials[0:int(num_trials/2)]])
                    coeffs[:,i]=logit.coef_[0]
                rect=ax.bar(2+width*.5+(idx-1)*width, np.mean(coeffs,axis=1), width,
                    yerr=np.std(coeffs,axis=1), color=condition_colors[stim_condition])
            for idx,stim_condition in enumerate(stim_conditions):
                large_ev_diff_biases=[]
                large_ev_diff_ev_diffs=[]
                large_ev_diff_correct=[]
                for i in range(len(all_ev_diffs[stim_condition])):
                    if all_ev_diffs[stim_condition][i]>bins[5]:
                        large_ev_diff_biases.append(all_biases[stim_condition][i])
                        large_ev_diff_ev_diffs.append(all_ev_diffs[stim_condition][i])
                        large_ev_diff_correct.append(all_correct[stim_condition][i])
                num_trials=len(large_ev_diff_biases)
                x=np.zeros((num_trials,1))
                x[:,0]=np.array(large_ev_diff_biases)
                y=np.array(large_ev_diff_correct)
                coeffs=np.zeros((1,100))
                for i in range(100):
                    permute_trials=np.random.permutation(range(num_trials))
                    logit=LogisticRegression()
                    logit=logit.fit(x[permute_trials[0:int(num_trials/2.0)],:], y[permute_trials[0:int(num_trials/2)]])
                    coeffs[:,i]=logit.coef_[0]
                rect=ax.bar(3+width*.5+(idx-1)*width, np.mean(coeffs,axis=1), width,
                    yerr=np.std(coeffs,axis=1), color=condition_colors[stim_condition])
            ax.set_ylabel('Coefficient')
            ax.set_xticks(ind+width)
            ax.set_xticklabels(['Bias (overall)', 'Bias (small EV diff)', 'Bias (large EV diff)'])
            ax.legend([rect[0] for rect in rects],stim_conditions,loc='best')
            save_to_png(fig, '%s.png' % logistic_noise_only_fname)
            save_to_eps(fig, '%s.eps' % logistic_noise_only_fname)
            plt.close(fig)

        furl='img/beta_pyr_rate_diff'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_pyr_rate_diff_url = '%s.png' % furl
        if regenerate_plots:
            subject_betas=[]
            subject_pyr_rate_diff=[]
            for stim_condition in self.stim_conditions:
                stim_report=self.stim_condition_reports[stim_condition]
                for virtual_subj_id in range(stim_report.num_subjects):
                    if not virtual_subj_id in stim_report.excluded_sessions:
                        subj_diff_rates=[]
                        session_prefix=self.file_prefix % (virtual_subj_id,stim_condition)
                        session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                        data=FileInfo(session_report_file)
                        if data.est_beta<30.0:
                            for trial in range(len(data.trial_e_rates)):
                                if data.choice[trial]>-1:
                                    chosen_mean=np.mean(data.trial_e_rates[trial][data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                                    unchosen_mean=np.mean(data.trial_e_rates[trial][1-data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                                    subj_diff_rates.append(chosen_mean-unchosen_mean)
                            subject_pyr_rate_diff.append(np.mean(subj_diff_rates))
                            subject_betas.append(data.est_beta)
            
            subject_betas_vec=np.reshape(np.array(subject_betas),(len(subject_betas),1))
            subject_pyr_rate_diff_vec=np.reshape(np.array(subject_pyr_rate_diff),(len(subject_pyr_rate_diff),1))
            clf = LinearRegression()
            clf.fit(subject_betas_vec, subject_pyr_rate_diff_vec)
            self.beta_pyr_rate_diff_a = clf.coef_[0][0]
            self.beta_pyr_rate_diff_b = clf.intercept_[0]
            self.beta_pyr_rate_diff_lin_r_sqr=clf.score(subject_betas_vec, subject_pyr_rate_diff_vec)

            popt,pcov=curve_fit(exp_decay, np.array(subject_betas), np.array(subject_pyr_rate_diff))
            self.beta_pyr_rate_diff_n=popt[0]
            self.beta_pyr_rate_diff_lam=popt[1]
            y_hat=exp_decay(np.array(subject_betas),*popt)
            ybar=np.sum(np.array(subject_pyr_rate_diff))/len(np.array(subject_pyr_rate_diff))
            ssres=np.sum((np.array(subject_pyr_rate_diff)-y_hat)**2.0)
            sstot=np.sum((np.array(subject_pyr_rate_diff)-ybar)**2.0)
            self.beta_pyr_rate_diff_exp_r_sqr=1.0-ssres/sstot

            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(subject_betas_vec, subject_pyr_rate_diff_vec,'o')
            min_x=np.min(subject_betas_vec)-1
            max_x=np.max(subject_betas_vec)+1
            x_range=min_x+np.array(range(1000))*(max_x-min_x)/1000.0
            ax.plot(x_range,exp_decay(x_range,*popt),label='r^2=%.3f' % self.beta_pyr_rate_diff_exp_r_sqr)
            ax.plot([min_x, max_x], [self.beta_pyr_rate_diff_a * min_x + self.beta_pyr_rate_diff_b,
                                     self.beta_pyr_rate_diff_a * max_x + self.beta_pyr_rate_diff_b],
                label='r^2=%.3f' % self.beta_pyr_rate_diff_lin_r_sqr)
            ax.legend(loc=0)
            ax.set_xlabel('Beta')
            ax.set_ylabel('Pyr Rate Diff')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/alpha_pyr_rate_diff'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_pyr_rate_diff_url = '%s.png' % furl
        if regenerate_plots:
            subject_alphas=[]
            subject_pyr_rate_diff=[]
            for stim_condition in self.stim_conditions:
                stim_report=self.stim_condition_reports[stim_condition]
                for virtual_subj_id in range(stim_report.num_subjects):
                    if not virtual_subj_id in stim_report.excluded_sessions:
                        subj_diff_rates=[]
                        session_prefix=self.file_prefix % (virtual_subj_id,stim_condition)
                        session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                        data=FileInfo(session_report_file)
                        for trial in range(len(data.trial_e_rates)):
                            if data.choice[trial]>-1:
                                chosen_mean=np.mean(data.trial_e_rates[trial][data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                                unchosen_mean=np.mean(data.trial_e_rates[trial][1-data.choice[trial],int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                                subj_diff_rates.append(chosen_mean-unchosen_mean)
                        subject_pyr_rate_diff.append(np.mean(subj_diff_rates))
                        subject_alphas.append(data.est_alpha)

            subject_alphas_vec=np.reshape(np.array(subject_alphas),(len(subject_alphas),1))
            subject_pyr_rate_diff_vec=np.reshape(np.array(subject_pyr_rate_diff),(len(subject_pyr_rate_diff),1))
            clf = LinearRegression()
            clf.fit(subject_alphas_vec, subject_pyr_rate_diff_vec)
            self.alpha_pyr_rate_diff_a = clf.coef_[0][0]
            self.alpha_pyr_rate_diff_b = clf.intercept_[0]
            self.alpha_pyr_rate_diff_lin_r_sqr=clf.score(subject_alphas_vec, subject_pyr_rate_diff_vec)

            popt,pcov=curve_fit(exp_decay, np.array(subject_alphas), np.array(subject_pyr_rate_diff))
            self.alpha_pyr_rate_diff_n=popt[0]
            self.alpha_pyr_rate_diff_lam=popt[1]
            y_hat=exp_decay(np.array(subject_alphas),*popt)
            ybar=np.sum(np.array(subject_pyr_rate_diff))/len(np.array(subject_pyr_rate_diff))
            ssres=np.sum((np.array(subject_pyr_rate_diff)-y_hat)**2.0)
            sstot=np.sum((np.array(subject_pyr_rate_diff)-ybar)**2.0)
            self.alpha_pyr_rate_diff_exp_r_sqr=1.0-ssres/sstot

            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.plot(subject_alphas_vec, subject_pyr_rate_diff_vec,'o')
            min_x=np.min(subject_alphas_vec)-.1
            max_x=np.max(subject_alphas_vec)+.1
            x_range=min_x+np.array(range(1000))*(max_x-min_x)/1000.0
            ax.plot(x_range,exp_decay(x_range,*popt),label='r^2=%.3f' % self.alpha_pyr_rate_diff_exp_r_sqr)
            ax.plot([min_x, max_x], [self.alpha_pyr_rate_diff_a * min_x + self.alpha_pyr_rate_diff_b,
                                     self.alpha_pyr_rate_diff_a * max_x + self.alpha_pyr_rate_diff_b],
                label='r^2=%.3f' % self.alpha_pyr_rate_diff_lin_r_sqr)
            ax.legend(loc=0)
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Pyr Rate Diff')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        #create report
        template_file='rl.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='rl.html'
        fname=os.path.join(self.reports_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

#def plot_param_diff(cond_name, param_name, orig_vals, new_vals, diff_range=None):
#    diff_vals=new_vals-orig_vals
#    fig=plt.figure()
#    filtered_diffs=reject_outliers(np.array(diff_vals))
#    hist,bins=np.histogram(filtered_diffs, bins=10, range=diff_range)
#    bin_width=bins[1]-bins[0]
#    plt.bar(bins[:-1], hist/float(len(filtered_diffs)), width=bin_width)
#    if diff_range is not None:
#        plt.xlim(diff_range)
#    plt.xlabel('Change in %s' % param_name)
#    plt.ylabel('Proportion of Subjects')
#    plt.title(cond_name)
#    return fig

def plot_param_diff(cond_name, param_name, orig_vals, new_vals, val_range=None):
    fig=plt.figure()
    filtered_orig=reject_outliers(np.array(orig_vals))
    orig_hist,orig_bins=np.histogram(filtered_orig, bins=10, range=val_range)
    orig_bin_width=orig_bins[1]-orig_bins[0]
    bar=plt.bar(orig_bins[:-1], orig_hist/float(len(filtered_orig)), width=orig_bin_width)
    for b in bar:
        b.set_color('b')
    filtered_new=reject_outliers(np.array(new_vals))
    new_hist,new_bins=np.histogram(filtered_new, bins=10, range=val_range)
    new_bin_width=new_bins[1]-new_bins[0]
    bar=plt.bar(new_bins[:-1], new_hist/float(len(filtered_new)), width=new_bin_width)
    for b in bar:
        b.set_color('r')
    if val_range is not None:
        plt.xlim(val_range)
    plt.xlabel(param_name)
    plt.ylabel('Proportion of Subjects')
    plt.title(cond_name)
    return fig

def plot_params(cond_name, param_name, orig_vals, new_vals, val_range=None):
    fig=plt.figure()
    pos = np.arange(2)+0.5    # Center bars on the Y-axis ticks
    ax=fig.add_subplot(2,1,1)
    filtered_orig=reject_outliers(np.array(orig_vals))
    filtered_new=reject_outliers(np.array(new_vals))
    bar=ax.bar(pos[0],np.mean(filtered_orig), width=.5, yerr=np.std(filtered_orig), align='center',ecolor='k')
    bar[0].set_color('b')
    bar=ax.bar(pos[1],np.mean(filtered_new), width=.5, yerr=np.std(filtered_new),align='center',ecolor='k')
    bar[0].set_color('r')
    ax.set_xticks(pos)
    ax.set_xticklabels(['Control',cond_name])
    ax.set_xlabel('Condition')
    ax.set_ylabel(param_name)
    return fig

def plot_trials_ev_diff(data_dir,file_name):
    data=FileInfo(os.path.join(data_dir,file_name))
    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
    hist,bins=np.histogram(np.array(ev_diff), bins=10)
    bin_width=bins[1]-bins[0]
    plt.bar(bins[:-1], hist/float(len(ev_diff)), width=bin_width)
    plt.show()

#def plot_mean_firing_rate(data_dir, file_name):
#    data=FileInfo(os.path.join(data_dir,file_name))
#    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
#
#    min_ev_diff=0.5
#    max_ev_diff=0.6
#    trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]
#    chosen_firing_rates=[]
#    unchosen_firing_rates=[]
#    for trial in trials:
#        if data.choice[trial]>-1:
#            chosen_firing_rates.append(data.trial_e_rates[trial][data.choice[trial],:])
#            unchosen_firing_rates.append(data.trial_e_rates[trial][1-data.choice[trial],:])
#
#    chosen_firing_rates=np.array(chosen_firing_rates)
#    unchosen_firing_rates=np.array(unchosen_firing_rates)
#
#    fig=plt.figure()
#    plt.plot(np.mean(chosen_firing_rates,axis=0))
#    plt.plot(np.mean(unchosen_firing_rates,axis=0))
#    plt.show()

def debug_trial_plot(file_name):
    f = h5py.File(file_name)
    num_trials=int(f.attrs['trials'])
    trial_e_rates=[]
    trial_i_rates=[]
    for i in range(num_trials):
        f_trial=f['trial %d' % i]
        trial_e_rates.append(np.array(f_trial['e_rates']))
        trial_i_rates.append(np.array(f_trial['i_rates']))
    f.close()

    for i in range(num_trials):
        e_firing_rates=trial_e_rates[i]
        i_firing_rates=trial_i_rates[i]

        # figure out max firing rate of all neurons (pyramidal and interneuron)
        max_pop_rate=0
        for i, pop_rate in enumerate(e_firing_rates):
            max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])
        for i, pop_rate in enumerate(i_firing_rates):
            max_pop_rate=np.max([max_pop_rate,np.max(pop_rate)])

        fig=plt.figure()

        # Plot pyramidal neuron firing rate
        ax=plt.subplot(2,1,1)
        for i, pop_rate in enumerate(e_firing_rates):
            #ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
            ax.plot(pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
        #plt.ylim([0,10+max_pop_rate])
        ax.legend(loc=0)
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')

        # Plot interneuron firing rate
        ax = fig.add_subplot(2,1,2)
        for i, pop_rate in enumerate(i_firing_rates):
            #ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
            ax.plot(pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
        #plt.ylim([0,10+max_pop_rate])
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        plt.show()

def run_accuracy_logistic(reports_dir, data_dir, file_prefix, num_subjects, use_z=False):
    stim_conditions=['control','anode','cathode']
    stim_condition_reports={}
    excluded=None
    for stim_condition in stim_conditions:
        print(stim_condition)
        stim_condition_report_dir=os.path.join(reports_dir,stim_condition)
        stim_condition_reports[stim_condition] = StimConditionReport(data_dir, file_prefix,
            stim_condition, stim_condition_report_dir, num_subjects, '')
        stim_condition_reports[stim_condition].create_report(None, excluded=excluded,
            regenerate_plots=False, regenerate_session_plots=False, regenerate_trial_plots=False)

    fig=Figure(figsize=(16,6))
    ax=fig.add_subplot(1,1,1)
    ind=np.array([1,2,3,4,5,6])
    width=0.3
    rects=[]
    condition_colors={'control':'b','anode':'r','cathode':'g'}

    condition_coeffs={}
    condition_small_ev_diff_coeffs={}
    condition_large_ev_diff_coeffs={}

    for idx,stim_condition in enumerate(stim_conditions):
        stim_report=stim_condition_reports[stim_condition]
        coeffs=[]
        small_ev_diff_coeffs=[]
        large_ev_diff_coeffs=[]
        intercepts=[]
        small_ev_diff_intercepts=[]
        large_ev_diff_intercepts=[]
        overall_accuracy=[]
        small_ev_diff_accuracy=[]
        large_ev_diff_accuracy=[]
        for virtual_subj_id in range(stim_report.num_subjects):
            if not virtual_subj_id in stim_report.excluded_sessions:
                session_prefix=file_prefix % (virtual_subj_id,stim_condition)
                session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)

                biases=[]
                ev_diffs=[]
                correct=[]

                small_ev_diff_biases=[]
                small_ev_diff_ev_diffs=[]
                small_ev_diff_correct=[]

                large_ev_diff_biases=[]
                large_ev_diff_ev_diffs=[]
                large_ev_diff_correct=[]

                for trial in range(len(data.trial_e_rates)):
                    if data.choice[trial]>-1:
                        left_mean=np.mean(data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        right_mean=np.mean(data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        bias=np.abs(left_mean-right_mean)
                        ev_diff=np.abs(data.inputs[0,trial]-data.inputs[1,trial])
                        biases.append(bias)
                        ev_diffs.append(ev_diff)
                        choice_correct=0.0
                        if (data.choice[trial]==0 and data.inputs[0,trial]>data.inputs[1,trial]) or (data.choice[trial]==1 and data.inputs[1,trial]>data.inputs[0,trial]):
                            choice_correct=1.0
                        correct.append(choice_correct)
                biases=np.array(biases)
                ev_diffs=np.array(ev_diffs)
                if use_z:
                    biases=(biases-np.mean(biases))/np.std(biases)
                    ev_diffs=(ev_diffs-np.mean(ev_diffs))/np.std(ev_diffs)
                ev_lower_lim=np.percentile(np.abs(np.array(ev_diffs)), 25)
                ev_upper_lim=np.percentile(np.abs(np.array(ev_diffs)), 75)
                for i in range(len(biases)):
                    if np.abs(ev_diffs[i])<ev_lower_lim:
                        small_ev_diff_biases.append(biases[i])
                        small_ev_diff_ev_diffs.append(ev_diffs[i])
                        small_ev_diff_correct.append(correct[i])
                    elif np.abs(ev_diffs[i])>ev_upper_lim:
                        large_ev_diff_biases.append(biases[i])
                        large_ev_diff_ev_diffs.append(ev_diffs[i])
                        large_ev_diff_correct.append(correct[i])

                small_ev_diff_biases=np.array(small_ev_diff_biases)
                small_ev_diff_ev_diffs=np.array(small_ev_diff_ev_diffs)
                large_ev_diff_biases=np.array(large_ev_diff_biases)
                large_ev_diff_ev_diffs=np.array(large_ev_diff_ev_diffs)
                
                x=np.zeros((len(biases),2))
                x[:,0]=biases
                x[:,1]=ev_diffs
                y=np.array(correct)
                logit = LogisticRegression(C=1000.0)
                logit = logit.fit(x, y)
                coeffs.append(logit.coef_[0])
                intercepts.append(logit.intercept_)
                y_mod=logit.predict(x)
                overall_accuracy.append(float(len(np.where(y-y_mod==0)[0]))/float(len(y)))

                x=np.zeros((len(small_ev_diff_biases),2))
                x[:,0]=small_ev_diff_biases
                x[:,1]=small_ev_diff_ev_diffs
                y=np.array(small_ev_diff_correct)
                logit = LogisticRegression(C=1000.0)
                logit = logit.fit(x, y)
                y_mod=logit.predict(x)
                small_ev_diff_accuracy.append(float(len(np.where(y-y_mod==0)[0]))/float(len(y)))
                small_ev_diff_coeffs.append(logit.coef_[0])
                small_ev_diff_intercepts.append(logit.intercept_)

                x=np.zeros((len(large_ev_diff_biases),2))
                x[:,0]=large_ev_diff_biases
                x[:,1]=large_ev_diff_ev_diffs
                y=np.array(large_ev_diff_correct)
                logit = LogisticRegression(C=1000.0)
                logit = logit.fit(x, y)
                y_mod=logit.predict(x)
                large_ev_diff_accuracy.append(float(len(np.where(y-y_mod==0)[0]))/float(len(y)))
                large_ev_diff_coeffs.append(logit.coef_[0])
                large_ev_diff_intercepts.append(logit.intercept_)

        print('overall, mean accuracy=%.4f' % np.mean(overall_accuracy))
        print('small EV Diff, mean accuracy=%.4f' % np.mean(small_ev_diff_accuracy))
        print('large EV Diff, mean accuracy=%.4f' % np.mean(large_ev_diff_accuracy))

        coeffs=np.array(coeffs)
        (t,p)=ttest_1samp(coeffs[:,0],0.0)
        print('overall, %s, bias, t=%.3f, p=%.5f' % (stim_condition,t,p))
        (t,p)=ttest_1samp(coeffs[:,1],0.0)
        print('overall, %s, ev diff, t=%.3f, p=%.5f' % (stim_condition,t,p))
        small_ev_diff_coeffs=np.array(small_ev_diff_coeffs)
        (t,p)=ttest_1samp(small_ev_diff_coeffs[:,0],0.0)
        print('small EV Diff, %s, bias, t=%.3f, p=%.5f' % (stim_condition,t,p))
        (t,p)=ttest_1samp(small_ev_diff_coeffs[:,1],0.0)
        print('small EV Diff, %s, ev diff, t=%.3f, p=%.5f' % (stim_condition,t,p))
        large_ev_diff_coeffs=np.array(large_ev_diff_coeffs)
        (t,p)=ttest_1samp(large_ev_diff_coeffs[:,0],0.0)
        print('large EV Diff, %s, bias, t=%.3f, p=%.5f' % (stim_condition,t,p))
        (t,p)=ttest_1samp(large_ev_diff_coeffs[:,1],0.0)
        print('large EV Diff, %s, ev diff, t=%.3f, p=%.5f' % (stim_condition,t,p))

        condition_coeffs[stim_condition]=coeffs
        condition_small_ev_diff_coeffs[stim_condition]=small_ev_diff_coeffs
        condition_large_ev_diff_coeffs[stim_condition]=large_ev_diff_coeffs

        coeff_array=[]
        coeff_array.extend(np.mean(coeffs,axis=0))
        coeff_array.extend(np.mean(small_ev_diff_coeffs,axis=0))
        coeff_array.extend(np.mean(large_ev_diff_coeffs,axis=0))
        
        coeff_std_err_array=[]
        coeff_std_err_array.extend(np.std(coeffs,axis=0)/np.sqrt(len(coeffs)))
        coeff_std_err_array.extend(np.std(small_ev_diff_coeffs,axis=0)/np.sqrt(len(small_ev_diff_coeffs)))
        coeff_std_err_array.extend(np.std(large_ev_diff_coeffs,axis=0)/np.sqrt(len(large_ev_diff_coeffs)))

        rect=ax.bar(np.array([1,2,3,4,5,6])+width*.5+(idx-1)*width, coeff_array, width,
            yerr=coeff_std_err_array, ecolor='k', color=condition_colors[stim_condition])
        rects.append(rect)

        mean_coeffs=np.mean(coeffs,axis=0)
        mean_intercept=np.mean(intercepts)

        mean_small_ev_diff_coeffs=np.mean(small_ev_diff_coeffs,axis=0)
        mean_small_ev_diff_intercept=np.mean(small_ev_diff_intercepts)

        mean_large_ev_diff_coeffs=np.mean(large_ev_diff_coeffs,axis=0)
        mean_large_ev_diff_intercept=np.mean(large_ev_diff_intercepts)

#        num_trials=0.0
#        num_correct=0.0
#        num_small_ev_diff_trials=0.0
#        num_small_ev_diff_correct=0.0
#        num_large_ev_diff_trials=0.0
#        num_large_ev_diff_correct=0.0
#        for virtual_subj_id in range(stim_report.num_subjects):
#            if not virtual_subj_id in stim_report.excluded_sessions:
#                session_prefix=file_prefix % (virtual_subj_id,stim_condition)
#                session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
#                data=FileInfo(session_report_file)
#                biases=[]
#                ev_diffs=[]
#                choice=[]
#
#                small_ev_diff_biases=[]
#                small_ev_diff_ev_diffs=[]
#                small_ev_diff_choice=[]
#
#                large_ev_diff_biases=[]
#                large_ev_diff_ev_diffs=[]
#                large_ev_diff_choice=[]
#
#                for trial in range(len(data.trial_e_rates)):
#                    if data.choice[trial]>-1:
#                        left_mean=np.mean(data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
#                        right_mean=np.mean(data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
#                        bias=left_mean-right_mean
#                        ev_diff=data.inputs[0,trial]-data.inputs[1,trial]
#                        biases.append(bias)
#                        ev_diffs.append(ev_diff)
#                        choice.append(data.choice[trial])
#                biases=np.array(biases)
#                ev_diffs=np.array(ev_diffs)
#                if use_z:
#                    biases=(biases-np.mean(biases))/np.std(biases)
#                    ev_diffs=(ev_diffs-np.mean(ev_diffs))/np.std(ev_diffs)
#                ev_lower_lim=np.percentile(ev_diffs, 25)
#                ev_upper_lim=np.percentile(ev_diffs, 75)
#                for i in range(len(biases)):
#                    if ev_diffs[i]<ev_lower_lim:
#                        small_ev_diff_biases.append(biases[i])
#                        small_ev_diff_ev_diffs.append(ev_diffs[i])
#                        small_ev_diff_choice.append(choice[i])
#                    elif ev_diffs[i]>ev_upper_lim:
#                        large_ev_diff_biases.append(biases[i])
#                        large_ev_diff_ev_diffs.append(ev_diffs[i])
#                        large_ev_diff_choice.append(choice[i])
#
#                small_ev_diff_biases=np.array(small_ev_diff_biases)
#                small_ev_diff_ev_diffs=np.array(small_ev_diff_ev_diffs)
#                large_ev_diff_biases=np.array(large_ev_diff_biases)
#                large_ev_diff_ev_diffs=np.array(large_ev_diff_ev_diffs)
#
#                x=np.zeros((len(biases),2))
#                x[:,0]=biases
#                x[:,1]=ev_diffs
#                y=np.array(choice)
#                y_pred=np.zeros(y.shape)
#                logit=LogisticRegression()
#                logit.raw_coef_=np.array([[-mean_coeffs[0],-mean_coeffs[1],-mean_intercept]])
#                logit.coef_[0]=mean_coeffs
#                logit.intercept_=mean_intercept
#                y_mod=logit.predict(x)
#                num_trials+=len(y)
#                num_correct+=len(np.where(y-y_mod==0)[0])
#
#                x=np.zeros((len(small_ev_diff_biases),2))
#                x[:,0]=small_ev_diff_biases
#                x[:,1]=small_ev_diff_ev_diffs
#                y=np.array(small_ev_diff_choice)
#                y_pred=np.zeros(y.shape)
#                logit=LogisticRegression()
#                logit.raw_coef_=np.array([[-mean_small_ev_diff_coeffs[0],-mean_small_ev_diff_coeffs[1],-mean_small_ev_diff_intercept]])
#                logit.coef_[0]=mean_small_ev_diff_coeffs
#                logit.intercept_=mean_small_ev_diff_intercept
#                y_mod=logit.predict(x)
#                num_small_ev_diff_trials+=len(y)
#                num_small_ev_diff_correct+=len(np.where(y-y_mod==0)[0])
#
#                x=np.zeros((len(large_ev_diff_biases),2))
#                x[:,0]=large_ev_diff_biases
#                x[:,1]=large_ev_diff_ev_diffs
#                y=np.array(large_ev_diff_choice)
#                y_pred=np.zeros(y.shape)
#                logit=LogisticRegression()
#                logit.raw_coef_=np.array([[-mean_large_ev_diff_coeffs[0],-mean_large_ev_diff_coeffs[1],-mean_large_ev_diff_intercept]])
#                logit.coef_[0]=mean_large_ev_diff_coeffs
#                logit.intercept_=mean_large_ev_diff_intercept
#                y_mod=logit.predict(x)
#                num_large_ev_diff_trials+=len(y)
#                num_large_ev_diff_correct+=len(np.where(y-y_mod==0)[0])
#
#        print('%s, overall accuracy=%.3f' % (stim_condition,float(num_correct)/float(num_trials)))
#        print('%s, small EV diff accuracy=%.3f' % (stim_condition,float(num_small_ev_diff_correct)/float(num_small_ev_diff_trials)))
#        print('%s, large EV diff accuracy=%.3f' % (stim_condition,float(num_large_ev_diff_correct)/float(num_large_ev_diff_trials)))

    ax.set_ylabel('Coefficient')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(['Bias','EV Diff','Bias (small EV Diff)','EV Diff (small EV Diff)','Bias (large EV Diff)', 'EV Diff (large EV Diff)'])
    ax.legend([rect[0] for rect in rects],stim_conditions,loc='best')
    logistic_furl='img/rate_diff_perc_correct_logistic_reversed'
    if use_z:
        logistic_furl='%s_z' % logistic_furl
    logistic_fname = os.path.join(reports_dir,logistic_furl)
    save_to_png(fig, '%s.png' % logistic_fname)
    save_to_eps(fig, '%s.eps' % logistic_fname)
    plt.close(fig)

    (F,p)=f_oneway(condition_coeffs['control'][:,0],condition_coeffs['anode'][:,0],condition_coeffs['cathode'][:,0])
    print('ANOVA: bias, F=%.3f, p=%.5f' % (F,p))
    (F,p)=f_oneway(condition_coeffs['control'][:,1],condition_coeffs['anode'][:,1],condition_coeffs['cathode'][:,1])
    print('ANOVA: EV Diff, F=%.3f, p=%.5f' % (F,p))

    (F,p)=f_oneway(condition_small_ev_diff_coeffs['control'][:,0],condition_small_ev_diff_coeffs['anode'][:,0],condition_small_ev_diff_coeffs['cathode'][:,0])
    print('ANOVA: small EV Diff, bias, F=%.3f, p=%.5f' % (F,p))
    (F,p)=f_oneway(condition_small_ev_diff_coeffs['control'][:,1],condition_small_ev_diff_coeffs['anode'][:,1],condition_small_ev_diff_coeffs['cathode'][:,1])
    print('ANOVA: small EV Diff, EV Diff, F=%.3f, p=%.5f' % (F,p))

    (F,p)=f_oneway(condition_large_ev_diff_coeffs['control'][:,0],condition_large_ev_diff_coeffs['anode'][:,0],condition_large_ev_diff_coeffs['cathode'][:,0])
    print('ANOVA: large EV Diff, bias, F=%.3f, p=%.5f' % (F,p))
    (F,p)=f_oneway(condition_large_ev_diff_coeffs['control'][:,1],condition_large_ev_diff_coeffs['anode'][:,1],condition_large_ev_diff_coeffs['cathode'][:,1])
    print('ANOVA: large EV Diff, EV Diff, F=%.3f, p=%.5f' % (F,p))

    num_comparisons=3.0
    (t,p)=ttest_rel(condition_coeffs['control'][:,0],condition_coeffs['anode'][:,0])
    print('Posthoc, bias, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['control'][:,0],condition_coeffs['cathode'][:,0])
    print('Posthoc, bias, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['anode'][:,0],condition_coeffs['cathode'][:,0])
    print('Posthoc, bias, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_coeffs['control'][:,1],condition_coeffs['anode'][:,1])
    print('Posthoc, EV Diff, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['control'][:,1],condition_coeffs['cathode'][:,1])
    print('Posthoc, EV Diff, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['anode'][:,1],condition_coeffs['cathode'][:,1])
    print('Posthoc, EV Diff, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,0],condition_small_ev_diff_coeffs['anode'][:,0])
    print('Posthoc, small EV Diff, bias, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,0],condition_small_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, small EV Diff, bias, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['anode'][:,0],condition_small_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, small EV Diff, bias, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,1],condition_small_ev_diff_coeffs['anode'][:,1])
    print('Posthoc, small EV Diff, EV Diff, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,1],condition_small_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, small EV Diff, EV Diff, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['anode'][:,1],condition_small_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, small EV Diff, EV Diff, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,0],condition_large_ev_diff_coeffs['anode'][:,0])
    print('Posthoc, large EV Diff, bias, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,0],condition_large_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, large EV Diff, bias, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['anode'][:,0],condition_large_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, large EV Diff, bias, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,1],condition_large_ev_diff_coeffs['anode'][:,1])
    print('Posthoc, large EV Diff, EV Diff, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,1],condition_large_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, large EV Diff, EV Diff, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['anode'][:,1],condition_large_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, large EV Diff, EV Diff, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

def run_choice_logistic(reports_dir, data_dir, file_prefix, num_subjects, use_z=False):
    stim_conditions=['control','anode','cathode']
    stim_condition_reports={}
    excluded=None
    for stim_condition in stim_conditions:
        print(stim_condition)
        stim_condition_report_dir=os.path.join(reports_dir,stim_condition)
        stim_condition_reports[stim_condition] = StimConditionReport(data_dir, file_prefix,
            stim_condition, stim_condition_report_dir, num_subjects, '')
        stim_condition_reports[stim_condition].create_report(None, excluded=excluded,
            regenerate_plots=False, regenerate_session_plots=False, regenerate_trial_plots=False)

    fig=Figure(figsize=(16,6))
    ax=fig.add_subplot(1,1,1)
    ind=np.array([1,2,3,4,5,6])
    width=0.3
    rects=[]
    condition_colors={'control':'b','anode':'r','cathode':'g'}

    condition_coeffs={}
    condition_small_ev_diff_coeffs={}
    condition_large_ev_diff_coeffs={}

    for idx,stim_condition in enumerate(stim_conditions):
        stim_report=stim_condition_reports[stim_condition]
        coeffs=[]
        small_ev_diff_coeffs=[]
        large_ev_diff_coeffs=[]
        intercepts=[]
        small_ev_diff_intercepts=[]
        large_ev_diff_intercepts=[]
        for virtual_subj_id in range(stim_report.num_subjects):
            if not virtual_subj_id in stim_report.excluded_sessions:
                session_prefix=file_prefix % (virtual_subj_id,stim_condition)
                session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)

                biases=[]
                ev_diffs=[]
                choice=[]

                small_ev_diff_biases=[]
                small_ev_diff_ev_diffs=[]
                small_ev_diff_choice=[]

                large_ev_diff_biases=[]
                large_ev_diff_ev_diffs=[]
                large_ev_diff_choice=[]

                for trial in range(len(data.trial_e_rates)):
                    if data.choice[trial]>-1:
                        left_mean=np.mean(data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        right_mean=np.mean(data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        bias=left_mean-right_mean
                        ev_diff=data.inputs[0,trial]-data.inputs[1,trial]
                        biases.append(bias)
                        ev_diffs.append(ev_diff)
                        choice.append(data.choice[trial])
                biases=np.array(biases)
                ev_diffs=np.array(ev_diffs)
                if use_z:
                    biases=(biases-np.mean(biases))/np.std(biases)
                    ev_diffs=(ev_diffs-np.mean(ev_diffs))/np.std(ev_diffs)
                ev_lower_lim=np.percentile(ev_diffs, 25)
                ev_upper_lim=np.percentile(ev_diffs, 75)
                for i in range(len(biases)):
                    if ev_diffs[i]<ev_lower_lim:
                        small_ev_diff_biases.append(biases[i])
                        small_ev_diff_ev_diffs.append(ev_diffs[i])
                        small_ev_diff_choice.append(choice[i])
                    elif ev_diffs[i]>ev_upper_lim:
                        large_ev_diff_biases.append(biases[i])
                        large_ev_diff_ev_diffs.append(ev_diffs[i])
                        large_ev_diff_choice.append(choice[i])

                small_ev_diff_biases=np.array(small_ev_diff_biases)
                small_ev_diff_ev_diffs=np.array(small_ev_diff_ev_diffs)
                large_ev_diff_biases=np.array(large_ev_diff_biases)
                large_ev_diff_ev_diffs=np.array(large_ev_diff_ev_diffs)

                x=np.zeros((len(biases),2))
                x[:,0]=biases
                x[:,1]=ev_diffs
                y=np.array(choice)
                logit = LogisticRegression()
                logit = logit.fit(x, y)
                coeffs.append(logit.coef_[0])
                intercepts.append(logit.intercept_)

                x=np.zeros((len(small_ev_diff_biases),2))
                x[:,0]=small_ev_diff_biases
                x[:,1]=small_ev_diff_ev_diffs
                y=np.array(small_ev_diff_choice)
                logit = LogisticRegression()
                logit = logit.fit(x, y)
                small_ev_diff_coeffs.append(logit.coef_[0])
                small_ev_diff_intercepts.append(logit.intercept_)

                x=np.zeros((len(large_ev_diff_biases),2))
                x[:,0]=large_ev_diff_biases
                x[:,1]=large_ev_diff_ev_diffs
                y=np.array(large_ev_diff_choice)
                logit = LogisticRegression()
                logit = logit.fit(x, y)
                large_ev_diff_coeffs.append(logit.coef_[0])
                large_ev_diff_intercepts.append(logit.intercept_)

        coeffs=np.array(coeffs)
        (t,p)=ttest_1samp(coeffs[:,0],0.0)
        print('overall, %s, bias, t=%.3f, p=%.5f' % (stim_condition,t,p))
        (t,p)=ttest_1samp(coeffs[:,1],0.0)
        print('overall, %s, ev diff, t=%.3f, p=%.5f' % (stim_condition,t,p))
        small_ev_diff_coeffs=np.array(small_ev_diff_coeffs)
        (t,p)=ttest_1samp(small_ev_diff_coeffs[:,0],0.0)
        print('small EV Diff, %s, bias, t=%.3f, p=%.5f' % (stim_condition,t,p))
        (t,p)=ttest_1samp(small_ev_diff_coeffs[:,1],0.0)
        print('small EV Diff, %s, ev diff, t=%.3f, p=%.5f' % (stim_condition,t,p))
        large_ev_diff_coeffs=np.array(large_ev_diff_coeffs)
        (t,p)=ttest_1samp(large_ev_diff_coeffs[:,0],0.0)
        print('large EV Diff, %s, bias, t=%.3f, p=%.5f' % (stim_condition,t,p))
        (t,p)=ttest_1samp(large_ev_diff_coeffs[:,1],0.0)
        print('large EV Diff, %s, ev diff, t=%.3f, p=%.5f' % (stim_condition,t,p))

        condition_coeffs[stim_condition]=coeffs
        condition_small_ev_diff_coeffs[stim_condition]=small_ev_diff_coeffs
        condition_large_ev_diff_coeffs[stim_condition]=large_ev_diff_coeffs

        coeff_array=[]
        coeff_array.extend(np.mean(coeffs,axis=0))
        coeff_array.extend(np.mean(small_ev_diff_coeffs,axis=0))
        coeff_array.extend(np.mean(large_ev_diff_coeffs,axis=0))

        coeff_std_err_array=[]
        coeff_std_err_array.extend(np.std(coeffs,axis=0)/np.sqrt(len(coeffs)))
        coeff_std_err_array.extend(np.std(small_ev_diff_coeffs,axis=0)/np.sqrt(len(small_ev_diff_coeffs)))
        coeff_std_err_array.extend(np.std(large_ev_diff_coeffs,axis=0)/np.sqrt(len(large_ev_diff_coeffs)))

        rect=ax.bar(np.array([1,2,3,4,5,6])+width*.5+(idx-1)*width, coeff_array, width,
            yerr=coeff_std_err_array, ecolor='k', color=condition_colors[stim_condition])
        rects.append(rect)

        mean_coeffs=np.mean(coeffs,axis=0)
        mean_intercept=np.mean(intercepts)
        logit=LogisticRegression()
        logit.coef_[0]=mean_coeffs
        logit.intercept_=mean_intercept
        
        mean_small_ev_diff_coeffs=np.mean(small_ev_diff_coeffs,axis=0)
        mean_small_ev_diff_intercept=np.mean(small_ev_diff_intercepts)
        small_ev_diff_logit=LogisticRegression()
        small_ev_diff_logit.coef_[0]=mean_small_ev_diff_coeffs
        small_ev_diff_logit.intercept_=mean_small_ev_diff_intercept

        mean_large_ev_diff_coeffs=np.mean(large_ev_diff_coeffs,axis=0)
        mean_large_ev_diff_intercept=np.mean(large_ev_diff_intercepts)
        large_ev_diff_logit=LogisticRegression()
        large_ev_diff_logit.coef_[0]=mean_large_ev_diff_coeffs
        large_ev_diff_logit.intercept_=mean_large_ev_diff_intercept
        
        num_trials=0.0
        num_correct=0.0
        num_small_ev_diff_trials=0.0
        num_small_ev_diff_correct=0.0
        num_large_ev_diff_trials=0.0
        num_large_ev_diff_correct=0.0
        for virtual_subj_id in range(stim_report.num_subjects):
            if not virtual_subj_id in stim_report.excluded_sessions:
                session_prefix=file_prefix % (virtual_subj_id,stim_condition)
                session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                biases=[]
                ev_diffs=[]
                choice=[]

                small_ev_diff_biases=[]
                small_ev_diff_ev_diffs=[]
                small_ev_diff_choice=[]

                large_ev_diff_biases=[]
                large_ev_diff_ev_diffs=[]
                large_ev_diff_choice=[]

                for trial in range(len(data.trial_e_rates)):
                    if data.choice[trial]>-1:
                        left_mean=np.mean(data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        right_mean=np.mean(data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        bias=left_mean-right_mean
                        ev_diff=data.inputs[0,trial]-data.inputs[1,trial]
                        biases.append(bias)
                        ev_diffs.append(ev_diff)
                        choice.append(data.choice[trial])
                biases=np.array(biases)
                ev_diffs=np.array(ev_diffs)
                if use_z:
                    biases=(biases-np.mean(biases))/np.std(biases)
                    ev_diffs=(ev_diffs-np.mean(ev_diffs))/np.std(ev_diffs)
                ev_lower_lim=np.percentile(ev_diffs, 25)
                ev_upper_lim=np.percentile(ev_diffs, 75)
                for i in range(len(biases)):
                    if ev_diffs[i]<ev_lower_lim:
                        small_ev_diff_biases.append(biases[i])
                        small_ev_diff_ev_diffs.append(ev_diffs[i])
                        small_ev_diff_choice.append(choice[i])
                    elif ev_diffs[i]>ev_upper_lim:
                        large_ev_diff_biases.append(biases[i])
                        large_ev_diff_ev_diffs.append(ev_diffs[i])
                        large_ev_diff_choice.append(choice[i])

                small_ev_diff_biases=np.array(small_ev_diff_biases)
                small_ev_diff_ev_diffs=np.array(small_ev_diff_ev_diffs)
                large_ev_diff_biases=np.array(large_ev_diff_biases)
                large_ev_diff_ev_diffs=np.array(large_ev_diff_ev_diffs)

                x=np.zeros((len(biases),2))
                x[:,0]=biases
                x[:,1]=ev_diffs
                y=np.array(choice)
                y_pred=logit.predict(x)
                num_trials+=len(y)
                num_correct+=len(np.where(y-y_pred==0)[0])

                x=np.zeros((len(small_ev_diff_biases),2))
                x[:,0]=small_ev_diff_biases
                x[:,1]=small_ev_diff_ev_diffs
                y=np.array(small_ev_diff_choice)
                y_pred=small_ev_diff_logit.predict(x)
                num_small_ev_diff_trials+=len(y)
                num_small_ev_diff_correct+=len(np.where(y-y_pred==0)[0])

                x=np.zeros((len(large_ev_diff_biases),2))
                x[:,0]=large_ev_diff_biases
                x[:,1]=large_ev_diff_ev_diffs
                y=np.array(large_ev_diff_choice)
                y_pred=large_ev_diff_logit.predict(x)
                num_large_ev_diff_trials+=len(y)
                num_large_ev_diff_correct+=len(np.where(y-y_pred==0)[0])
        
        print('%s, overall accuracy=%.3f' % (stim_condition,float(num_correct)/float(num_trials)))
        print('%s, small EV diff accuracy=%.3f' % (stim_condition,float(num_small_ev_diff_correct)/float(num_small_ev_diff_trials)))
        print('%s, large EV diff accuracy=%.3f' % (stim_condition,float(num_large_ev_diff_correct)/float(num_large_ev_diff_trials)))

    ax.set_ylabel('Coefficient')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(['Bias','EV Diff','Bias (small EV Diff)','EV Diff (small EV Diff)','Bias (large EV Diff)', 'EV Diff (large EV Diff)'])
    ax.legend([rect[0] for rect in rects],stim_conditions,loc='best')
    logistic_furl='img/rate_diff_choice_logistic_reversed'
    if use_z:
        logistic_furl='%s_z' % logistic_furl
    logistic_fname = os.path.join(reports_dir,logistic_furl)
    save_to_png(fig, '%s.png' % logistic_fname)
    save_to_eps(fig, '%s.eps' % logistic_fname)
    plt.close(fig)

    (F,p)=f_oneway(condition_coeffs['control'][:,0],condition_coeffs['anode'][:,0],condition_coeffs['cathode'][:,0])
    print('ANOVA: bias, F=%.3f, p=%.5f' % (F,p))
    (F,p)=f_oneway(condition_coeffs['control'][:,1],condition_coeffs['anode'][:,1],condition_coeffs['cathode'][:,1])
    print('ANOVA: EV Diff, F=%.3f, p=%.5f' % (F,p))

    (F,p)=f_oneway(condition_small_ev_diff_coeffs['control'][:,0],condition_small_ev_diff_coeffs['anode'][:,0],condition_small_ev_diff_coeffs['cathode'][:,0])
    print('ANOVA: small EV Diff, bias, F=%.3f, p=%.5f' % (F,p))
    (F,p)=f_oneway(condition_small_ev_diff_coeffs['control'][:,1],condition_small_ev_diff_coeffs['anode'][:,1],condition_small_ev_diff_coeffs['cathode'][:,1])
    print('ANOVA: small EV Diff, EV Diff, F=%.3f, p=%.5f' % (F,p))

    (F,p)=f_oneway(condition_large_ev_diff_coeffs['control'][:,0],condition_large_ev_diff_coeffs['anode'][:,0],condition_large_ev_diff_coeffs['cathode'][:,0])
    print('ANOVA: large EV Diff, bias, F=%.3f, p=%.5f' % (F,p))
    (F,p)=f_oneway(condition_large_ev_diff_coeffs['control'][:,1],condition_large_ev_diff_coeffs['anode'][:,1],condition_large_ev_diff_coeffs['cathode'][:,1])
    print('ANOVA: large EV Diff, EV Diff, F=%.3f, p=%.5f' % (F,p))

    num_comparisons=3.0
    (t,p)=ttest_rel(condition_coeffs['control'][:,0],condition_coeffs['anode'][:,0])
    print('Posthoc, bias, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['control'][:,0],condition_coeffs['cathode'][:,0])
    print('Posthoc, bias, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['anode'][:,0],condition_coeffs['cathode'][:,0])
    print('Posthoc, bias, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_coeffs['control'][:,1],condition_coeffs['anode'][:,1])
    print('Posthoc, EV Diff, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['control'][:,1],condition_coeffs['cathode'][:,1])
    print('Posthoc, EV Diff, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_coeffs['anode'][:,1],condition_coeffs['cathode'][:,1])
    print('Posthoc, EV Diff, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,0],condition_small_ev_diff_coeffs['anode'][:,0])
    print('Posthoc, small EV Diff, bias, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,0],condition_small_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, small EV Diff, bias, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['anode'][:,0],condition_small_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, small EV Diff, bias, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,1],condition_small_ev_diff_coeffs['anode'][:,1])
    print('Posthoc, small EV Diff, EV Diff, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['control'][:,1],condition_small_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, small EV Diff, EV Diff, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_small_ev_diff_coeffs['anode'][:,1],condition_small_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, small EV Diff, EV Diff, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,0],condition_large_ev_diff_coeffs['anode'][:,0])
    print('Posthoc, large EV Diff, bias, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,0],condition_large_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, large EV Diff, bias, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['anode'][:,0],condition_large_ev_diff_coeffs['cathode'][:,0])
    print('Posthoc, large EV Diff, bias, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,1],condition_large_ev_diff_coeffs['anode'][:,1])
    print('Posthoc, large EV Diff, EV Diff, control-anode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['control'][:,1],condition_large_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, large EV Diff, EV Diff, control-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))
    (t,p)=ttest_rel(condition_large_ev_diff_coeffs['anode'][:,1],condition_large_ev_diff_coeffs['cathode'][:,1])
    print('Posthoc, large EV Diff, EV Diff, anode-cathode, t=%.3f, p=%.5f' % (t,p*num_comparisons/2.0))

def generate_accuracy_logistic_files(reports_dir, data_dir, file_prefix, num_subjects):
    stim_conditions=['control','anode','cathode']
    stim_condition_reports={}
    excluded=None
    for stim_condition in stim_conditions:
        print(stim_condition)
        stim_condition_report_dir=os.path.join(reports_dir,stim_condition)
        stim_condition_reports[stim_condition] = StimConditionReport(data_dir, file_prefix,
            stim_condition, stim_condition_report_dir, num_subjects, '')
        stim_condition_reports[stim_condition].create_report(None, excluded=excluded,
            regenerate_plots=False, regenerate_session_plots=False, regenerate_trial_plots=False)

    for idx,stim_condition in enumerate(stim_conditions):
        stim_report=stim_condition_reports[stim_condition]
        for virtual_subj_id in range(stim_report.num_subjects):
            if not virtual_subj_id in stim_report.excluded_sessions:
                f=open('%s_subj_%d.csv' % (os.path.join(reports_dir,stim_condition),virtual_subj_id),'w')
                session_prefix=file_prefix % (virtual_subj_id,stim_condition)
                session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                for trial in range(len(data.trial_e_rates)):
                    if data.choice[trial]>-1:
                        left_mean=np.mean(data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        right_mean=np.mean(data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        bias=np.abs(left_mean-right_mean)
                        ev_diff=np.abs(data.inputs[0,trial]-data.inputs[1,trial])
                        choice_correct=0.0
                        if (data.choice[trial]==0 and data.inputs[0,trial]>data.inputs[1,trial]) or (data.choice[trial]==1 and data.inputs[1,trial]>data.inputs[0,trial]):
                            choice_correct=1.0
                        f.write('%0.4f,%0.4f,%d\n' % (bias,ev_diff,choice_correct))
                f.close()


def generate_choice_logistic_files(reports_dir, data_dir, file_prefix, num_subjects):
    stim_conditions=['control','anode','cathode']
    stim_condition_reports={}
    excluded=None
    for stim_condition in stim_conditions:
        print(stim_condition)
        stim_condition_report_dir=os.path.join(reports_dir,stim_condition)
        stim_condition_reports[stim_condition] = StimConditionReport(data_dir, file_prefix,
            stim_condition, stim_condition_report_dir, num_subjects, '')
        stim_condition_reports[stim_condition].create_report(None, excluded=excluded,
            regenerate_plots=False, regenerate_session_plots=False, regenerate_trial_plots=False)

    for idx,stim_condition in enumerate(stim_conditions):
        stim_report=stim_condition_reports[stim_condition]
        for virtual_subj_id in range(stim_report.num_subjects):
            if not virtual_subj_id in stim_report.excluded_sessions:
                f=open('%s_subj_%d_choice.csv' % (os.path.join(reports_dir,stim_condition),virtual_subj_id),'w')
                session_prefix=file_prefix % (virtual_subj_id,stim_condition)
                session_report_file=os.path.join(stim_report.data_dir,'%s.h5' % session_prefix)
                data=FileInfo(session_report_file)
                for trial in range(len(data.trial_e_rates)):
                    if data.choice[trial]>-1:
                        left_mean=np.mean(data.trial_e_rates[trial][0,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        right_mean=np.mean(data.trial_e_rates[trial][1,int((500*ms)/(.5*ms)):int((950*ms)/(.5*ms))])
                        bias=left_mean-right_mean
                        ev_diff=data.inputs[0,trial]-data.inputs[1,trial]
                        f.write('%0.4f,%0.4f,%d\n' % (bias,ev_diff,data.choice[trial]))
                f.close()


def rename_data_files(data_dir):
    for file_name in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, file_name)):
            if 'dcs_start_time' in file_name:
                filename_parts=file_name.split('.')
                new_filename='rl.virtual_subject.%s.%s.h5' % (filename_parts[23],filename_parts[24])
                shutil.copyfile(os.path.join(data_dir,file_name),os.path.join(data_dir,new_filename))

if __name__=='__main__':
    report=RLReport('/data/pySBI/rl/virtual_subjects.v2/','rl.virtual_subject.%d.%s',
        ['control','anode','anode_control_4','anode_control_5','anode_control_6','cathode','cathode_control_4','cathode_control_5','cathode_control_6'],'/data/pySBI/reports/rl/virtual_subjects.v2',
        25,'')
    #report.create_report()
    #plot_trials_ev_diff('../../data/rerw','virtual_subject_0.control.h5')
    #plot_mean_firing_rate('../../data/rerw','virtual_subject_0.control.h5')
    #debug_trial_plot('../../data/rerw/virtual_subject_0.control.h5')
    #back_range=850+np.array(range(5))*20
    #report=BackgroundBetaReport('/data/pySBI/rl/background_beta.v2','rl.background_freq.%0.3f.trial.%d',back_range,'/data/pySBI/reports/rl/background.v2',10,'')
    report.create_report(regenerate_plots=True)
