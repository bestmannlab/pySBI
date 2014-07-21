import matplotlib
matplotlib.use('Agg')
from scipy import stats
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
from scikits.learn.linear_model import LinearRegression
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.util.utils import save_to_png, save_to_eps, get_response_time, reject_outliers
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


class TrialData:
    def __init__(self, trial, trial_duration, val, ev, inputs, choice, rew, file_prefix, reports_dir, e_firing_rates,
                 i_firing_rates, rt=None, upper_resp_threshold=30, lower_resp_threshold=None, dt=.1*ms):
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

        if not os.path.exists('%s.png' % fname):
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
                ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
            if self.rt:
                rt_idx=(1*second+self.rt)/dt
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
            plt.ylim([0,10+max_pop_rate])
            ax.legend(loc=0)
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing Rate (Hz)')

            # Plot interneuron firing rate
            ax = fig.add_subplot(2,1,2)
            for i, pop_rate in enumerate(i_firing_rates):
                ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
            if self.rt:
                rt_idx=(1*second+self.rt)/dt
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
            plt.ylim([0,10+max_pop_rate])
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing Rate (Hz)')
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

    def compute_trial_rate_stats(self, data, min_ev_diff, max_ev_diff):
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
        chosen_rate_mean=chosen_rate_sum/trial_count
        unchosen_rate_mean=unchosen_rate_sum/trial_count
        chosen_rate_std_sum=np.zeros(chosen_rate_mean.shape)
        unchosen_rate_std_sum=np.zeros(unchosen_rate_mean.shape)
        for trial in trials:
            if data.choice[trial]>-1:
                chosen_rate_std_sum+=(data.trial_e_rates[trial][data.choice[trial],:]-chosen_rate_mean)**2.0
                unchosen_rate_std_sum+=(data.trial_e_rates[trial][1-data.choice[trial],:]-unchosen_rate_mean)**2.0
        chosen_rate_std_err=np.sqrt(chosen_rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
        unchosen_rate_std_err=np.sqrt(unchosen_rate_std_sum/(trial_count-1))/np.sqrt(trial_count)
        return chosen_rate_mean,chosen_rate_std_err,unchosen_rate_mean,unchosen_rate_std_err

    def sort_trials(self, data, min_ev_diff, max_ev_diff):
        ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
        trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]
        chosen_firing_rates=[]
        unchosen_firing_rates=[]
        for trial in trials:
            if data.choice[trial]>-1:
                chosen_firing_rates.append(data.trial_e_rates[trial][data.choice[trial],:])
                unchosen_firing_rates.append(data.trial_e_rates[trial][1-data.choice[trial],:])

        chosen_firing_rates=np.array(chosen_firing_rates)
        unchosen_firing_rates=np.array(unchosen_firing_rates)
        return chosen_firing_rates, unchosen_firing_rates

    def create_report(self, version, data):
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

        if not os.path.exists('%s.png' % fname):
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

        if not os.path.exists('%s.png' % fname):
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
        if not os.path.exists('%s.png' % fname):
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            ax.bar(bins[:-1], hist/float(len(ev_diff)), width=bin_width)
            ax.set_xlabel('EV Diff')
            ax.set_ylabel('% of Trials')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_firing_rate.%s' % self.file_prefix
        fname = os.path.join(self.reports_dir, furl)
        self.mean_firing_rate_ev_diff_url = '%s.png' % furl

        small_chosen_mean,small_chosen_std_err,small_unchosen_mean,small_unchosen_std_err=self.compute_trial_rate_stats(data,
            bins[0], bins[3])
        med_chosen_mean,med_chosen_std_err,med_unchosen_mean,med_unchosen_std_err=self.compute_trial_rate_stats(data,
            bins[3], bins[6])
        large_chosen_mean,large_chosen_std_err,large_unchosen_mean,large_unchosen_std_err=self.compute_trial_rate_stats(data,
            bins[6], bins[-1])
        
        if not os.path.exists('%s.png' % fname):
            fig=Figure()
            ax=fig.add_subplot(1,1,1)
            baseline,=ax.plot(small_chosen_mean,'b',label='chosen, small')
            ax.fill_between(range(len(small_chosen_mean)),
                small_chosen_mean-small_chosen_std_err,small_chosen_mean+small_chosen_std_err,alpha=0.5,
                facecolor=baseline.get_color())
            baseline,=ax.plot(small_unchosen_mean,'b--',label='unchosen, small')
            ax.fill_between(range(len(small_unchosen_mean)),
                small_unchosen_mean-small_unchosen_std_err,small_unchosen_mean+small_unchosen_std_err,alpha=0.5,
                facecolor=baseline.get_color())
            baseline,=ax.plot(med_chosen_mean,'g',label='chosen, med')
            ax.fill_between(range(len(med_chosen_mean)),
                med_chosen_mean-med_chosen_std_err,med_chosen_mean+med_chosen_std_err,alpha=0.5,
                facecolor=baseline.get_color())
            baseline,=ax.plot(med_unchosen_mean,'g--',label='unchosen, med')
            ax.fill_between(range(len(med_unchosen_mean)),
                med_unchosen_mean-med_unchosen_std_err,med_unchosen_mean+med_unchosen_std_err,alpha=0.5,
                facecolor=baseline.get_color())
            baseline,=ax.plot(large_chosen_mean,'r',label='chosen, large')
            ax.fill_between(range(len(large_chosen_mean)),
                large_chosen_mean-large_chosen_std_err,large_chosen_mean+large_chosen_std_err,alpha=0.5,
                facecolor=baseline.get_color())
            baseline,=ax.plot(large_unchosen_mean,'r--',label='unchosen, large')
            ax.fill_between(range(len(large_unchosen_mean)),
                large_unchosen_mean-large_unchosen_std_err,large_unchosen_mean+large_unchosen_std_err,alpha=0.5,
                facecolor=baseline.get_color())
            ax.set_xlabel('Time')
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
                data.trial_i_rates[trial], rt=rt, upper_resp_threshold=30, lower_resp_threshold=None, dt=.5*ms)
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


    def compute_trial_rate_stats(self, min_beta, max_beta, min_ev_diff, max_ev_diff):
        data=FileInfo(os.path.join(self.data_dir,'%s.h5' % self.file_prefix % (0,self.stim_condition)))
        chosen_rate_sum=np.zeros(data.trial_e_rates[0][0,:].shape)
        unchosen_rate_sum=np.zeros(data.trial_e_rates[0][0,:].shape)
        trial_count=0.0

        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                print('subject %d' % virtual_subj_id)
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
                print('subject %d' % virtual_subj_id)
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
        unchosen_rate_std_err=np.sqrt(unchosen_rate_std_sum/trial_count-1)/np.sqrt(trial_count)
        return chosen_rate_mean,chosen_rate_std_err,unchosen_rate_mean,unchosen_rate_std_err
    
    def create_report(self, version, excluded=None):
        make_report_dirs(self.reports_dir)

        self.version=version
        #self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.edesc=self.edesc

        self.condition_alphas=[]
        self.condition_betas=[]
        self.condition_perc_correct=[]
        ev_diff=[]
        for virtual_subj_id in range(self.num_subjects):
            data=FileInfo(os.path.join(self.data_dir,'%s.h5' % self.file_prefix % (virtual_subj_id,self.stim_condition)))
            ev_diff.extend(np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:]))
            #if (excluded is None and data.est_alpha<.98) or (excluded is not None and virtual_subj_id not in excluded):
            self.condition_alphas.append([data.est_alpha])
            self.condition_betas.append([data.est_beta])
            #else:
            #   self.excluded_sessions.append(virtual_subj_id)

        self.condition_alphas=np.array(self.condition_alphas)
        self.condition_betas=np.array(self.condition_betas)

        beta_hist,beta_bins=np.histogram(reject_outliers(self.condition_betas), bins=10)
        ev_diff_hist,ev_diff_bins=np.histogram(np.array(ev_diff), bins=10)

        for virtual_subj_id in range(self.num_subjects):
            if virtual_subj_id not in self.excluded_sessions:
                print('subject %d' % virtual_subj_id)
                session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
                session_report_dir=os.path.join(self.reports_dir,session_prefix)
                session_report_file=os.path.join(self.data_dir,'%s.h5' % session_prefix)
                session_report=SessionReport(virtual_subj_id, self.data_dir, session_prefix, session_report_dir, self.edesc)
                data=FileInfo(session_report_file)
                session_report.create_report(self.version, data)
                self.sessions.append(session_report)                
                self.condition_perc_correct.append([session_report.perc_correct_response])

        self.condition_perc_correct=np.array(self.condition_perc_correct)

        small_beta_small_ev_diff_chosen_mean,\
        small_beta_small_ev_diff_chosen_std_err,\
        small_beta_small_ev_diff_unchosen_mean,\
        small_beta_small_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[0], beta_bins[3],
            ev_diff_bins[0], ev_diff_bins[3])
        small_beta_med_ev_diff_chosen_mean,\
        small_beta_med_ev_diff_chosen_std_err,\
        small_beta_med_ev_diff_unchosen_mean,\
        small_beta_med_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[0], beta_bins[3],
            ev_diff_bins[3], ev_diff_bins[6])
        small_beta_large_ev_diff_chosen_mean,\
        small_beta_large_ev_diff_chosen_std_err,\
        small_beta_large_ev_diff_unchosen_mean,\
        small_beta_large_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[0], beta_bins[3],
            ev_diff_bins[6], ev_diff_bins[-1])

        med_beta_small_ev_diff_chosen_mean,\
        med_beta_small_ev_diff_chosen_std_err,\
        med_beta_small_ev_diff_unchosen_mean,\
        med_beta_small_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[3], beta_bins[6],
            ev_diff_bins[0], ev_diff_bins[3])
        med_beta_med_ev_diff_chosen_mean,\
        med_beta_med_ev_diff_chosen_std_err,\
        med_beta_med_ev_diff_unchosen_mean,\
        med_beta_med_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[3], beta_bins[6],
            ev_diff_bins[3], ev_diff_bins[6])
        med_beta_large_ev_diff_chosen_mean,\
        med_beta_large_ev_diff_chosen_std_err,\
        med_beta_large_ev_diff_unchosen_mean,\
        med_beta_large_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[3], beta_bins[6],
            ev_diff_bins[6], ev_diff_bins[-1])

        large_beta_small_ev_diff_chosen_mean,\
        large_beta_small_ev_diff_chosen_std_err,\
        large_beta_small_ev_diff_unchosen_mean,\
        large_beta_small_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[6], beta_bins[-1],
            ev_diff_bins[0], ev_diff_bins[3])
        large_beta_med_ev_diff_chosen_mean,\
        large_beta_med_ev_diff_chosen_std_err,\
        large_beta_med_ev_diff_unchosen_mean,\
        large_beta_med_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[6], beta_bins[-1],
            ev_diff_bins[3], ev_diff_bins[6])
        large_beta_large_ev_diff_chosen_mean,\
        large_beta_large_ev_diff_chosen_std_err,\
        large_beta_large_ev_diff_unchosen_mean,\
        large_beta_large_ev_diff_unchosen_std_err=self.compute_trial_rate_stats(beta_bins[6], beta_bins[-1],
            ev_diff_bins[6], ev_diff_bins[-1])
        
        # Create beta bar plot
        furl='img/beta_dist'
        fname = os.path.join(self.reports_dir, furl)
        self.beta_url = '%s.png' % furl
        hist,bins=np.histogram(reject_outliers(self.condition_betas), bins=10)
        bin_width=bins[1]-bins[0]
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        ax.bar(bins[:-1], hist/float(len(reject_outliers(self.condition_betas))), width=bin_width)
        ax.set_xlabel('Beta')
        ax.set_ylabel('% of Subjects')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        # Create alpha bar plot
        furl='img/alpha_dist'
        fname = os.path.join(self.reports_dir, furl)
        self.alpha_url = '%s.png' % furl
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

        d = np.abs(self.condition_betas - np.median(self.condition_betas))
        mdev = np.median(d)
        s = d/mdev if mdev else 0
        filtered_betas=self.condition_betas[s<2]
        filtered_betas=np.reshape(filtered_betas,(len(filtered_betas),1))
        filtered_perc_correct=self.condition_perc_correct[s<2]/100.0
        filtered_perc_correct=np.reshape(filtered_perc_correct,(len(filtered_perc_correct),1))
        clf.fit(filtered_betas, filtered_perc_correct)
        self.beta_perc_correct_a = clf.coef_[0][0]
        self.beta_perc_correct_b = clf.intercept_[0]
        self.beta_perc_correct_r_sqr=clf.score(filtered_betas, filtered_perc_correct)
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(filtered_betas,filtered_perc_correct,'o')
        min_x=np.min(filtered_betas)-1.0
        max_x=np.max(filtered_betas)+1.0
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
        furl='img/small_ev_diff_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_firing_rate_small_ev_diff_url = '%s.png' % furl
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        baseline,=ax.plot(small_beta_small_ev_diff_chosen_mean,'b',label='small beta, chosen')
        ax.fill_between(range(len(small_beta_small_ev_diff_chosen_mean)),
            small_beta_small_ev_diff_chosen_mean-small_beta_small_ev_diff_chosen_std_err,
            small_beta_small_ev_diff_chosen_mean+small_beta_small_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(small_beta_small_ev_diff_unchosen_mean,'b--',label='small beta, unchosen')
        ax.fill_between(range(len(small_beta_small_ev_diff_unchosen_mean)),
            small_beta_small_ev_diff_unchosen_mean-small_beta_small_ev_diff_unchosen_std_err,
            small_beta_small_ev_diff_unchosen_mean+small_beta_small_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(med_beta_small_ev_diff_chosen_mean,'g',label='med beta, chosen')
        ax.fill_between(range(len(med_beta_small_ev_diff_chosen_mean)),
            med_beta_small_ev_diff_chosen_mean-med_beta_small_ev_diff_chosen_std_err,
            med_beta_small_ev_diff_chosen_mean+med_beta_small_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(med_beta_small_ev_diff_unchosen_mean,'g--',label='med beta, unchosen')
        ax.fill_between(range(len(med_beta_small_ev_diff_unchosen_mean)),
            med_beta_small_ev_diff_unchosen_mean-med_beta_small_ev_diff_unchosen_std_err,
            med_beta_small_ev_diff_unchosen_mean+med_beta_small_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(large_beta_small_ev_diff_chosen_mean,'r',label='large beta, chosen')
        ax.fill_between(range(len(large_beta_small_ev_diff_chosen_mean)),
            large_beta_small_ev_diff_chosen_mean-large_beta_small_ev_diff_chosen_std_err,
            large_beta_small_ev_diff_chosen_mean+large_beta_small_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(large_beta_small_ev_diff_unchosen_mean,'r--',label='large beta, unchosen')
        ax.fill_between(range(len(large_beta_small_ev_diff_unchosen_mean)),
            large_beta_small_ev_diff_unchosen_mean-large_beta_small_ev_diff_unchosen_std_err,
            large_beta_small_ev_diff_unchosen_mean+large_beta_small_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        ax.set_xlabel('Time')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.legend(loc=0)
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/med_ev_diff_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_firing_rate_med_ev_diff_url = '%s.png' % furl
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        baseline,=ax.plot(small_beta_med_ev_diff_chosen_mean,'b',label='small beta, chosen')
        ax.fill_between(range(len(small_beta_med_ev_diff_chosen_mean)),
            small_beta_med_ev_diff_chosen_mean-small_beta_med_ev_diff_chosen_std_err,
            small_beta_med_ev_diff_chosen_mean+small_beta_med_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(small_beta_med_ev_diff_unchosen_mean,'b--',label='small beta, unchosen')
        ax.fill_between(range(len(small_beta_med_ev_diff_unchosen_mean)),
            small_beta_med_ev_diff_unchosen_mean-small_beta_med_ev_diff_unchosen_std_err,
            small_beta_med_ev_diff_unchosen_mean+small_beta_med_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(med_beta_med_ev_diff_chosen_mean,'g',label='med beta, chosen')
        ax.fill_between(range(len(med_beta_med_ev_diff_chosen_mean)),
            med_beta_med_ev_diff_chosen_mean-med_beta_med_ev_diff_chosen_std_err,
            med_beta_med_ev_diff_chosen_mean+med_beta_med_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(med_beta_med_ev_diff_unchosen_mean,'g--',label='med beta, unchosen')
        ax.fill_between(range(len(med_beta_med_ev_diff_unchosen_mean)),
            med_beta_med_ev_diff_unchosen_mean-med_beta_med_ev_diff_unchosen_std_err,
            med_beta_med_ev_diff_unchosen_mean+med_beta_med_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(large_beta_med_ev_diff_chosen_mean,'r',label='large beta, chosen')
        ax.fill_between(range(len(large_beta_med_ev_diff_chosen_mean)),
            large_beta_med_ev_diff_chosen_mean-large_beta_med_ev_diff_chosen_std_err,
            large_beta_med_ev_diff_chosen_mean+large_beta_med_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(large_beta_med_ev_diff_unchosen_mean,'r--',label='large beta, unchosen')
        ax.fill_between(range(len(large_beta_med_ev_diff_unchosen_mean)),
            large_beta_med_ev_diff_unchosen_mean-large_beta_med_ev_diff_unchosen_std_err,
            large_beta_med_ev_diff_unchosen_mean+large_beta_med_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        ax.set_xlabel('Time')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.legend(loc=0)
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/large_ev_diff_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_firing_rate_large_ev_diff_url = '%s.png' % furl
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        baseline,=ax.plot(small_beta_large_ev_diff_chosen_mean,'b',label='small beta, chosen')
        ax.fill_between(range(len(small_beta_large_ev_diff_chosen_mean)),
            small_beta_large_ev_diff_chosen_mean-small_beta_large_ev_diff_chosen_std_err,
            small_beta_large_ev_diff_chosen_mean+small_beta_large_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(small_beta_large_ev_diff_unchosen_mean,'b--',label='small beta, unchosen')
        ax.fill_between(range(len(small_beta_large_ev_diff_unchosen_mean)),
            small_beta_large_ev_diff_unchosen_mean-small_beta_large_ev_diff_unchosen_std_err,
            small_beta_large_ev_diff_unchosen_mean+small_beta_large_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(med_beta_large_ev_diff_chosen_mean,'g',label='med beta, chosen')
        ax.fill_between(range(len(med_beta_large_ev_diff_chosen_mean)),
            med_beta_large_ev_diff_chosen_mean-med_beta_large_ev_diff_chosen_std_err,
            med_beta_large_ev_diff_chosen_mean+med_beta_large_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(med_beta_large_ev_diff_unchosen_mean,'g--',label='med beta, unchosen')
        ax.fill_between(range(len(med_beta_large_ev_diff_unchosen_mean)),
            med_beta_large_ev_diff_unchosen_mean-med_beta_large_ev_diff_unchosen_std_err,
            med_beta_large_ev_diff_unchosen_mean+med_beta_large_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(large_beta_large_ev_diff_chosen_mean,'r',label='large beta, chosen')
        ax.fill_between(range(len(large_beta_large_ev_diff_chosen_mean)),
            large_beta_large_ev_diff_chosen_mean-large_beta_large_ev_diff_chosen_std_err,
            large_beta_large_ev_diff_chosen_mean+large_beta_large_ev_diff_chosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
        baseline,=ax.plot(large_beta_large_ev_diff_unchosen_mean,'r--',label='large beta, unchosen')
        ax.fill_between(range(len(large_beta_large_ev_diff_unchosen_mean)),
            large_beta_large_ev_diff_unchosen_mean-large_beta_large_ev_diff_unchosen_std_err,
            large_beta_large_ev_diff_unchosen_mean+large_beta_large_ev_diff_unchosen_std_err,alpha=0.5,
            facecolor=baseline.get_color())
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

    def create_report(self):
        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.edesc=self.edesc

        self.stim_condition_chosen_rate_means={}
        self.stim_condition_chosen_rate_std_err={}
        self.stim_condition_unchosen_rate_means={}
        self.stim_condition_unchosen_rate_std_err={}
        self.stim_condition_rate_diffs={}
        excluded=None
        for stim_condition in self.stim_conditions:
            print(stim_condition)
            stim_condition_report_dir=os.path.join(self.reports_dir,stim_condition)
            self.stim_condition_reports[stim_condition] = StimConditionReport(self.data_dir, self.file_prefix,
                stim_condition, stim_condition_report_dir, self.num_subjects, self.edesc)
            self.stim_condition_reports[stim_condition].create_report(self.version, excluded=excluded)
            excluded=self.stim_condition_reports[stim_condition].excluded_sessions

            self.stim_condition_chosen_rate_means[stim_condition],\
            self.stim_condition_chosen_rate_std_err[stim_condition],\
            self.stim_condition_unchosen_rate_means[stim_condition],\
            self.stim_condition_unchosen_rate_std_err[stim_condition]=self.stim_condition_reports[stim_condition].compute_trial_rate_stats(0,10000,0,100)

#            for chosen_rate,unchosen_rate in zip(self.stim_condition_chosen_rate_means[stim_condition],
#                self.stim_condition_unchosen_rate_means[stim_condition]):
#                self.stim_condition_rate_diffs[stim_condition]=np.mean(chosen_rate[int(900*ms/.5*ms):int(990*ms/.5*ms)])-np.mean(unchosen_rate[int(900*ms/.5*ms):int(900*ms/.5*ms)])

        # Create rate diff firing rate plot
#        furl='img/firing_rate_diff'
#        fname = os.path.join(self.reports_dir, furl)
#        self.firing_rate_diff_url = '%s.png' % furl
#        fig=Figure()
#        ax=fig.add_subplot(1,1,1)
#        mean_diffs=[]
#        std_diffs=[]
#        for stim_condition in self.stim_conditions:
#            mean_diffs.append(np.mean(self.stim_condition_rate_diffs[stim_condition]))
#            std_diffs.append(np.std(self.stim_condition_rate_diffs[stim_condition])/np.sqrt(len(self.stim_condition_rate_diffs[stim_condition])))
#        pos = np.arange(len(self.stim_conditions))+0.5    # Center bars on the Y-axis ticks
#        ax.bar(pos,mean_diffs,width=.5,yerr=std_diffs,align='center',ecolor='k')
#        ax.set_xticks(pos)
#        ax.set_xticklabels(self.stim_conditions)
#        ax.set_xlabel('Condition')
#        ax.set_ylabel('Firing Rate Diff')
#        save_to_png(fig, '%s.png' % fname)
#        save_to_eps(fig, '%s.eps' % fname)
#        plt.close(fig)

        # Create ev diff firing rate plot
        furl='img/ev_diff_firing_rate'
        fname = os.path.join(self.reports_dir, furl)
        self.mean_firing_rate_ev_diff_url = '%s.png' % furl
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        for stim_condition in self.stim_conditions:
            base_line,=ax.plot(self.stim_condition_chosen_rate_means[stim_condition],label='%s, chosen' % stim_condition)
            ax.fill_between(range(len(self.stim_condition_chosen_rate_means[stim_condition])),
                self.stim_condition_chosen_rate_means[stim_condition]-self.stim_condition_chosen_rate_std_err[stim_condition],
                self.stim_condition_chosen_rate_means[stim_condition]+self.stim_condition_chosen_rate_std_err[stim_condition],
                alpha=0.5,facecolor=base_line.get_color())
            base_line,=ax.plot(self.stim_condition_unchosen_rate_means[stim_condition],color=base_line.get_color(),
                linestyle='dashed',label='%s, unchosen' % stim_condition)
            ax.fill_between(range(len(self.stim_condition_unchosen_rate_means[stim_condition])),
                self.stim_condition_unchosen_rate_means[stim_condition]-self.stim_condition_unchosen_rate_std_err[stim_condition],
                self.stim_condition_unchosen_rate_means[stim_condition]+self.stim_condition_unchosen_rate_std_err[stim_condition],
                alpha=0.5,facecolor=base_line.get_color())
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
        all_condition_alphas=[]
        all_condition_perc_correct=[]
        fig=Figure()
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
        
        furl='img/all_alpha_perc_corect'
        fname = os.path.join(self.reports_dir, furl)
        self.all_alpha_perc_correct_url = '%s.png' % furl
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
        all_condition_betas=[]
        fig=Figure()
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

        furl='img/all_beta_perc_corect'
        fname = os.path.join(self.reports_dir, furl)
        self.all_beta_perc_correct_url = '%s.png' % furl
        all_condition_betas_vec=np.zeros([len(all_condition_betas),1])
        all_condition_betas_vec[:]=all_condition_betas
        all_condition_perc_correct_vec=np.zeros([len(all_condition_perc_correct),1])
        all_condition_perc_correct_vec[:]=all_condition_perc_correct
        d = np.abs(all_condition_betas_vec - np.median(all_condition_betas_vec))
        mdev = np.median(d)
        s = d/mdev if mdev else 0
        filtered_betas=all_condition_betas_vec[s<2]
        filtered_betas=np.reshape(filtered_betas,(len(filtered_betas),1))
        filtered_perc_correct=all_condition_perc_correct_vec[s<2]
        filtered_perc_correct=np.reshape(filtered_perc_correct,(len(filtered_perc_correct),1))
        clf = LinearRegression()
        clf.fit(filtered_betas, filtered_perc_correct)
        self.beta_perc_correct_a = clf.coef_[0][0]
        self.beta_perc_correct_b = clf.intercept_[0]
        self.beta_perc_correct_r_sqr=clf.score(filtered_betas, filtered_perc_correct)
        fig=Figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(filtered_betas, filtered_perc_correct,'o')
        min_x=np.min(filtered_betas)-1
        max_x=np.max(filtered_betas)+1
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

        self.stim_alpha_change_urls={}
        self.stim_beta_change_urls={}
        self.stim_alpha_mean_change={}
        self.stim_alpha_std_change={}
        self.stim_beta_mean_change={}
        self.stim_beta_std_change={}
        self.alpha_wilcoxon_test={}
        self.beta_wilcoxon_test={}
        for stim_condition in self.stim_conditions:
            if not stim_condition=='control':
                # Create alpha plot
                furl='img/%s_alpha' % stim_condition
                fname = os.path.join(self.reports_dir, furl)
                self.stim_alpha_change_urls[stim_condition] = '%s.png' % furl
                fig=plot_param_diff(stim_condition,'alpha',
                    self.stim_condition_reports['control'].condition_alphas,
                    self.stim_condition_reports[stim_condition].condition_alphas,
                    (-1.0,1.0))
                save_to_png(fig, '%s.png' % fname)
                plt.close(fig)
                alpha_diff=self.stim_condition_reports[stim_condition].condition_alphas-\
                           self.stim_condition_reports['control'].condition_alphas
                self.stim_alpha_mean_change[stim_condition]=np.mean(alpha_diff)
                self.stim_alpha_std_change[stim_condition]=np.std(alpha_diff)
                self.alpha_wilcoxon_test[stim_condition]=stats.wilcoxon(np.squeeze(self.stim_condition_reports['control'].condition_alphas),
                    np.squeeze(self.stim_condition_reports[stim_condition].condition_alphas))

                # Create beta plot
                furl='img/%s_beta' % stim_condition
                fname = os.path.join(self.reports_dir, furl)
                self.stim_beta_change_urls[stim_condition] = '%s.png' % furl
                fig=plot_param_diff(stim_condition,'beta',self.stim_condition_reports['control'].condition_betas,
                    self.stim_condition_reports[stim_condition].condition_betas,
                    (-10.0,10.0))
                save_to_png(fig, '%s.png' % fname)
                plt.close(fig)
                beta_diff=self.stim_condition_reports[stim_condition].condition_betas-\
                           self.stim_condition_reports['control'].condition_betas
                self.stim_beta_mean_change[stim_condition]=np.mean(beta_diff)
                self.stim_beta_std_change[stim_condition]=np.std(beta_diff)
                self.beta_wilcoxon_test[stim_condition]=stats.wilcoxon(np.squeeze(self.stim_condition_reports['control'].condition_betas),
                    np.squeeze(self.stim_condition_reports[stim_condition].condition_betas))


        #create report
        template_file='rl.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='rl.html'
        fname=os.path.join(self.reports_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

def plot_param_diff(cond_name, param_name, orig_vals, new_vals, diff_range):
    diff_vals=new_vals-orig_vals
    fig=plt.figure()
    hist,bins=np.histogram(np.array(diff_vals), bins=10, range=diff_range)
    bin_width=bins[1]-bins[0]
    plt.bar(bins[:-1], hist/float(len(diff_vals)), width=bin_width)
    plt.xlim(diff_range)
    plt.xlabel('Change in %s' % param_name)
    plt.ylabel('Proportion of Subjects')
    plt.title(cond_name)
    return fig

def plot_trials_ev_diff(data_dir,file_name):
    data=FileInfo(os.path.join(data_dir,file_name))
    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])
    hist,bins=np.histogram(np.array(ev_diff), bins=10)
    bin_width=bins[1]-bins[0]
    plt.bar(bins[:-1], hist/float(len(ev_diff)), width=bin_width)
    plt.show()

def plot_mean_firing_rate(data_dir, file_name):
    data=FileInfo(os.path.join(data_dir,file_name))
    ev_diff=np.abs(data.vals[0,:]*data.mags[0,:]-data.vals[1,:]*data.mags[1,:])

    min_ev_diff=0.5
    max_ev_diff=0.6
    trials=np.where((ev_diff>=min_ev_diff) & (ev_diff<max_ev_diff))[0]
    chosen_firing_rates=[]
    unchosen_firing_rates=[]
    for trial in trials:
        if data.choice[trial]>-1:
            chosen_firing_rates.append(data.trial_e_rates[trial][data.choice[trial],:])
            unchosen_firing_rates.append(data.trial_e_rates[trial][1-data.choice[trial],:])

    chosen_firing_rates=np.array(chosen_firing_rates)
    unchosen_firing_rates=np.array(unchosen_firing_rates)

    fig=plt.figure()
    plt.plot(np.mean(chosen_firing_rates,axis=0))
    plt.plot(np.mean(unchosen_firing_rates,axis=0))
    plt.show()

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

def rename_data_files(data_dir):
    for file_name in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, file_name)):
            if 'dcs_start_time' in file_name:
                filename_parts=file_name.split('.')
                new_filename='rl.virtual_subject.%s.%s.h5' % (filename_parts[23],filename_parts[24])
                shutil.copyfile(os.path.join(data_dir,file_name),os.path.join(data_dir,new_filename))

if __name__=='__main__':
    #report=RLReport('/data/ezrcluster/data/output/','rl.virtual_subject.%d.%s',
    #    ['control','anode','anode_control_1','cathode','cathode_control_1'],'/data/pySBI/reports/rl/virtual_subjects',
    #    25,'')
    #report.create_report()
    #plot_trials_ev_diff('../../data/rerw','virtual_subject_0.control.h5')
    #plot_mean_firing_rate('../../data/rerw','virtual_subject_0.control.h5')
    #debug_trial_plot('../../data/rerw/virtual_subject_0.control.h5')
    back_range=850+np.array(range(5))*20
    report=BackgroundBetaReport('/data/pySBI/rl/background_beta.v2','rl.background_freq.%0.3f.trial.%d',back_range,'/data/pySBI/reports/rl/background.v2',10,'')
    report.create_report()
