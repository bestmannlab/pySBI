from scipy import stats
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
from pysbi.util.utils import Struct, save_to_png, save_to_eps
from pysbi.wta.analysis import get_response_time
from pysbi.wta.network import default_params
from pysbi.wta.rl.fit import rescorla_td_prediction

class FileInfo:
    def __init__(self, file_name):
        self.file_name=file_name
        f = h5py.File(file_name)
        self.num_trials=int(f.attrs['trials'])
        self.alpha=float(f.attrs['alpha'])
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
        for i in range(self.num_trials):
            f_trial=f['trial %d' % i]
            self.trial_e_rates.append(np.array(f_trial['e_rates']))
            self.trial_i_rates.append(np.array(f_trial['i_rates']))
        f.close()


class TrialData:
    def __init__(self, trial, trial_duration, val, ev, inputs, choice, rew, file_prefix, reports_dir, e_firing_rates,
                 i_firing_rates, rt=None):
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
            self.rt,winner=get_response_time(e_firing_rates, 1*second, trial_duration-1*second)
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
                rt_idx=(1*second+self.rt)/ms
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
            plt.ylim([0,10+max_pop_rate])
            ax.legend(loc='best')
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing Rate (Hz)')

            # Plot interneuron firing rate
            ax = fig.add_subplot(2,1,2)
            for i, pop_rate in enumerate(i_firing_rates):
                ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
            # Plot line showing RT
            if self.rt:
                rt_idx=(1*second+self.rt)/ms
                ax.plot([rt_idx,rt_idx],[0,max_pop_rate],'r')
            plt.ylim([0,10+max_pop_rate])
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing Rate (Hz)')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

class SessionReport:
    def __init__(self, data_dir, file_prefix, reports_dir, edesc):
        self.data_dir=data_dir
        self.reports_dir=reports_dir
        self.file_prefix=file_prefix
        self.edesc=edesc

    def create_report(self):
        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.edesc=self.edesc

        data=FileInfo(os.path.join(self.data_dir,'%s.h5' % self.file_prefix))

        self.num_trials=data.num_trials
        self.alpha=data.alpha
        self.beta=(data.background_freq/Hz*-12.5)+87.46
        self.est_alpha=data.est_alpha
        self.est_beta=data.est_beta
        self.prop_correct=data.prop_correct*100.0

        self.num_groups=data.num_groups
        self.trial_duration=data.trial_duration
        self.background_freq=data.background_freq
        self.wta_params=data.wta_params

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
            ax.legend(loc='best')
            ax=fig.add_subplot(3,1,2)
            ax.plot(data.vals[0,:], label='model vals - o1')
            ax.plot(data.vals[1,:], label='model vals - o2')
            ax.legend(loc='best')
            ax=fig.add_subplot(3,1,3)
            ax.plot(fit_vals[0,:], label='fit vals - o1')
            ax.plot(fit_vals[1,:], label='fit vals - o2')
            ax.legend(loc='best')
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
            ax.legend(loc='best')
            ax.set_xlabel('Trial')

            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        self.perc_no_response=0.0
        self.trials=[]
        for trial in range(self.num_trials):
            trial_ev=data.vals[:,trial]*data.mags[:,trial]
            trial_prefix='%s.trial.%d' % (self.file_prefix,trial)
            rt=None
            if data.rts is not None:
                rt=data.rts[trial]*second
            trial_data=TrialData(trial+1, data.trial_duration, data.vals[:,trial], trial_ev, data.inputs[:,trial],
                data.choice[trial], data.rew[trial], trial_prefix, self.reports_dir, data.trial_e_rates[trial],
                data.trial_i_rates[trial], rt=rt)
            if trial_data.choice<0:
                self.perc_no_response+=1.0
            self.trials.append(trial_data)
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

        beta_vals=np.zeros((len(self.background_range)*self.trials,1))
        alpha_vals=np.zeros((len(self.background_range)*self.trials,1))
        background_vals=np.zeros((len(self.background_range)*self.trials,1))

        for idx,background_freq in enumerate(self.background_range):
            for trial in range(self.trials):
                print('background=%0.2f Hz, trial %d' % (background_freq, trial))
                session_prefix=self.file_prefix % (background_freq,trial)
                session_report_dir=os.path.join(self.reports_dir,session_prefix)
                session_report=SessionReport(self.data_dir, session_prefix, session_report_dir, self.edesc)
                session_report.create_report()
                self.sessions.append(session_report)
                background_vals[idx*self.trials+trial]=background_freq
                alpha_vals[idx*self.trials+trial]=session_report.est_alpha
                beta_vals[idx*self.trials+trial]=session_report.est_beta

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
        ax.legend(loc='best')

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
        ax.legend(loc='best')

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

    def create_report(self):
        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        self.edesc=self.edesc

        self.condition_alphas=np.zeros(self.num_subjects)
        self.condition_betas=np.zeros(self.num_subjects)

        for virtual_subj_id in range(self.num_subjects):
            print('subject %d' % virtual_subj_id)
            session_prefix=self.file_prefix % (virtual_subj_id,self.stim_condition)
            session_report_dir=os.path.join(self.reports_dir,session_prefix)
            session_report=SessionReport(self.data_dir, session_prefix, session_report_dir, self.edesc)
            session_report.create_report()
            self.sessions.append(session_report)

            self.condition_alphas[virtual_subj_id]=session_report.est_alpha
            self.condition_betas[virtual_subj_id]=session_report.est_beta

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

        for stim_condition in self.stim_conditions:
            print(stim_condition)
            stim_condition_report_dir=os.path.join(self.reports_dir,stim_condition)
            self.stim_condition_reports[stim_condition] = StimConditionReport(self.data_dir, self.file_prefix,
                stim_condition, stim_condition_report_dir, self.num_subjects, self.edesc)
            self.stim_condition_reports[stim_condition].create_report()

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
                self.alpha_wilcoxon_test[stim_condition]=stats.wilcoxon(self.stim_condition_reports['control'].condition_alphas,
                    self.stim_condition_reports[stim_condition].condition_alphas)

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
                self.beta_wilcoxon_test[stim_condition]=stats.wilcoxon(self.stim_condition_reports['control'].condition_betas,
                    self.stim_condition_reports[stim_condition].condition_betas)

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



if __name__=='__main__':
    report=RLReport('/data/projects/pySBI/rl','virtual_subject_%d.%s',
        ['anode','anode_control_1','anode_control_2','cathode','cathode_control_1','cathode_control_2','control'],
        '/data/projects/pySBI/rl/report',50,'')
    report.create_report()