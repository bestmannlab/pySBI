import copy
from brian import pA, second, Parameters, farad, siemens, volt, Hz, amp, ms, hertz
import h5py
import os
import math
from matplotlib.mlab import normpdf
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LinearRegression
from pysbi.util.plot import plot_condition_choice_probability, plot_network_firing_rates
from pysbi.util.utils import mdm_outliers, FitSigmoid, get_twod_confidence_interval, FitWeibull, FitRT, \
    get_response_time
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pysbi.wta.network import simulation_params
from pysbi.wta.rdmd.run import run_virtual_subjects

colors={
    'control': 'b',
    'depolarizing': 'r',
    'hyperpolarizing': 'g'
}

coherences=[0.0320, 0.0640, 0.1280, 0.2560, 0.5120]

conditions=['control','depolarizing','hyperpolarizing']

def read_subject_data(data_dir, subj_id, conditions, filter=True, response_only=True, neural_data=False):
    subj_data={}
    for condition in conditions:
        f = h5py.File(os.path.join(data_dir,'subject.%d.%s.h5' % (subj_id,condition)),'r')

        f_network_params = f['network_params']
        network_params=Parameters(
            # Neuron parameters
            C = float(f_network_params.attrs['C']) * farad,
            gL = float(f_network_params.attrs['gL']) * siemens,
            EL = float(f_network_params.attrs['EL']) * volt,
            VT = float(f_network_params.attrs['VT']) * volt,
            DeltaT = float(f_network_params.attrs['DeltaT']) * volt,
            Vr = float(f_network_params.attrs['Vr']) * volt,
            Mg = float(f_network_params.attrs['Mg']),
            E_ampa = float(f_network_params.attrs['E_ampa'])*volt,
            E_nmda = float(f_network_params.attrs['E_nmda'])*volt,
            E_gaba_a = float(f_network_params.attrs['E_gaba_a'])*volt,
            tau_ampa = float(f_network_params.attrs['tau_ampa'])*second,
            tau1_nmda = float(f_network_params.attrs['tau1_nmda'])*second,
            tau2_nmda = float(f_network_params.attrs['tau2_nmda'])*second,
            tau_gaba_a = float(f_network_params.attrs['tau_gaba_a'])*second,
            p_e_e=float(f_network_params.attrs['p_e_e']),
            p_e_i=float(f_network_params.attrs['p_e_i']),
            p_i_i=float(f_network_params.attrs['p_i_i']),
            p_i_e=float(f_network_params.attrs['p_i_e']),
            background_freq=float(f_network_params.attrs['background_freq'])*Hz,
            input_var=float(f_network_params.attrs['input_var'])*Hz,
            refresh_rate=float(f_network_params.attrs['refresh_rate'])*Hz,
            num_groups=int(f_network_params.attrs['num_groups']),
            network_group_size=int(f_network_params.attrs['network_group_size']),
            background_input_size=int(f_network_params.attrs['background_input_size']),
            mu_0=float(f_network_params.attrs['mu_0']),
            f=float(f_network_params.attrs['f']),
            task_input_resting_rate=float(f_network_params.attrs['task_input_resting_rate'])*Hz,
            resp_threshold=float(f_network_params.attrs['resp_threshold']),
            p_a=float(f_network_params.attrs['p_a']),
            p_b=float(f_network_params.attrs['p_b']),
            task_input_size=int(f_network_params.attrs['task_input_size'])
        )

        f_sim_params = f['sim_params']
        sim_params=Parameters(
            trial_duration=float(f_sim_params.attrs['trial_duration'])*second,
            stim_start_time=float(f_sim_params.attrs['stim_start_time'])*second,
            stim_end_time=float(f_sim_params.attrs['stim_end_time'])*second,
            dt=float(f_sim_params.attrs['dt'])*second,
            ntrials=int(f_sim_params.attrs['ntrials']),
            muscimol_amount=float(f_sim_params.attrs['muscimol_amount'])*siemens,
            injection_site=int(f_sim_params.attrs['injection_site']),
            p_dcs=float(f_sim_params.attrs['p_dcs'])*amp,
            i_dcs=float(f_sim_params.attrs['i_dcs'])*amp,
            dcs_start_time=float(f_sim_params.attrs['dcs_start_time'])*second,
            dcs_end_time=float(f_sim_params.attrs['dcs_end_time'])*second,
            plasticity=int(f_sim_params.attrs['plasticity'])
        )

        f_behav = f['behavior']
        trial_rt = np.array(f_behav['trial_rt'])
        trial_resp = np.array(f_behav['trial_resp'])
        trial_correct = np.array(f_behav['trial_correct'])

        f_neur = f['neural']
        trial_inputs = np.array(f_neur['trial_inputs'])
        trial_data = []
        trial_rates={
            'inhibitory_rate':[],
            'excitatory_rate_0':[],
            'excitatory_rate_1':[],
        }

        if neural_data:
            f_rates=f_neur['firing_rates']
            for trial_idx in range(trial_rt.shape[1]):
                f_trial=f_rates['trial_%d' % trial_idx]
                trial_rates['inhibitory_rate'].append(np.array(f_trial['inhibitory_rate']))
                trial_rates['excitatory_rate_0'].append(np.array(f_trial['excitatory_rate_0']))
                trial_rates['excitatory_rate_1'].append(np.array(f_trial['excitatory_rate_1']))

        last_resp = float('NaN')
        for trial_idx in range(trial_rt.shape[1]):
            direction = np.where(trial_inputs[:, trial_idx] == np.max(trial_inputs[:, trial_idx]))[0][0]
            if direction == 0:
                direction = -1
            coherence = np.abs((trial_inputs[0, trial_idx] - network_params.mu_0) / (network_params.p_a * 100.0))
            coherence_diffs=np.abs(np.array(coherences)-coherence)
            coherence=coherences[np.where(coherence_diffs==np.min(coherence_diffs))[0][0]]
            correct = int(trial_correct[0, trial_idx])
            resp = int(trial_resp[0, trial_idx])
            if resp == -1:
                resp=float('NaN')
            elif resp == 0:
                resp = -1
            rt = trial_rt[0, trial_idx]

            #if rt > 100:
            #if not response_only or not math.isnan(rt):
            trial_data.append([trial_idx, direction, coherence, correct, resp, last_resp, rt])
            last_resp = resp
        trial_data = np.array(trial_data)
        if filter:
            outliers=mdm_outliers(trial_data[:,6])
            trial_data[outliers,4:5]=float('NaN')
            #trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]
        subj_data[condition]={
            'behavior': trial_data,
            'neural': trial_rates,
            'sim_params': sim_params,
            'network_params': network_params
        }

    return subj_data


def analyze_subject_firing_rates(subject, plot=False):
    choose_bias_rates={}
    choose_unbias_rates={}
    for condition,subj_data in subject.iteritems():
        choose_bias_rates[condition]={
            'chosen_rate': [],
            'unchosen_rate': [],
        }
        choose_unbias_rates[condition]={
            'chosen_rate': [],
            'unchosen_rate': [],
        }
        trial_data=subj_data['behavior']
        trial_rates=subj_data['neural']
        # For each trial
        for trial_idx in range(1,trial_data.shape[0]):
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]

            if not math.isnan(resp) and not math.isnan(last_resp):
                if resp==last_resp:
                    if resp==-1:
                        choose_bias_rates[condition]['chosen_rate'].append(trial_rates['excitatory_rate_0'][trial_idx])
                        choose_bias_rates[condition]['unchosen_rate'].append(trial_rates['excitatory_rate_1'][trial_idx])
                    else:
                        choose_bias_rates[condition]['chosen_rate'].append(trial_rates['excitatory_rate_1'][trial_idx])
                        choose_bias_rates[condition]['unchosen_rate'].append(trial_rates['excitatory_rate_0'][trial_idx])
                else:
                    if resp==-1:
                        choose_unbias_rates[condition]['chosen_rate'].append(trial_rates['excitatory_rate_0'][trial_idx])
                        choose_unbias_rates[condition]['unchosen_rate'].append(trial_rates['excitatory_rate_1'][trial_idx])
                    else:
                        choose_unbias_rates[condition]['chosen_rate'].append(trial_rates['excitatory_rate_1'][trial_idx])
                        choose_unbias_rates[condition]['unchosen_rate'].append(trial_rates['excitatory_rate_0'][trial_idx])

        if plot:
            plt.figure()
            ax= plt.subplot(211)
            chosen_pop_rates=np.array(choose_bias_rates[condition]['chosen_rate'])
            unchosen_pop_rates=np.array(choose_bias_rates[condition]['unchosen_rate'])
            mean_e_pop_rates=np.array([np.mean(chosen_pop_rates,axis=0), np.mean(unchosen_pop_rates,axis=0)])
            std_e_pop_rates=np.array([np.std(chosen_pop_rates,axis=0)/np.sqrt(len(chosen_pop_rates)),
                                      np.std(unchosen_pop_rates,axis=0)/np.sqrt(len(unchosen_pop_rates))])
            plot_network_firing_rates(np.array(mean_e_pop_rates), subj_data['sim_params'], subj_data['network_params'],
                std_e_rates=std_e_pop_rates, plt_title=condition, labels=['chosen','unchosen'], ax=ax)

            ax= plt.subplot(212)
            chosen_pop_rates=np.array(choose_unbias_rates[condition]['chosen_rate'])
            unchosen_pop_rates=np.array(choose_unbias_rates[condition]['unchosen_rate'])
            mean_e_pop_rates=np.array([np.mean(chosen_pop_rates,axis=0), np.mean(unchosen_pop_rates,axis=0)])
            std_e_pop_rates=np.array([np.std(chosen_pop_rates,axis=0)/np.sqrt(len(chosen_pop_rates)),
                                      np.std(unchosen_pop_rates,axis=0)/np.sqrt(len(unchosen_pop_rates))])
            plot_network_firing_rates(np.array(mean_e_pop_rates), subj_data['sim_params'], subj_data['network_params'],
                std_e_rates=std_e_pop_rates, plt_title='', labels=['chosen','unchosen'], ax=ax)

    return choose_bias_rates, choose_unbias_rates



def analyze_subject_accuracy_rt(subject, plot=False):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    condition_overall_accuracy_rt={}
    for condition,subj_data in subject.iteritems():
        trial_data=subj_data['behavior']
        condition_coherence_accuracy[condition]={}
        condition_coherence_rt[condition]={}
        condition_overall_accuracy_rt[condition]=[[],[]]
        # For each trial
        for trial_idx in range(trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]
            correct=trial_data[trial_idx,3]
            rt=trial_data[trial_idx,6]

            if not math.isnan(rt):
                if not coherence in condition_coherence_accuracy[condition]:
                    condition_coherence_accuracy[condition][coherence]=[]
                condition_coherence_accuracy[condition][np.abs(coherence)].append(float(correct))

                if not coherence in condition_coherence_rt[condition]:
                    condition_coherence_rt[condition][coherence]=[]
                condition_coherence_rt[condition][np.abs(coherence)].append(rt)

                condition_overall_accuracy_rt[condition][0].append(correct)
                condition_overall_accuracy_rt[condition][1].append(rt)

        coherences = sorted(condition_coherence_accuracy[condition].keys())
        accuracy=[]
        for coherence in coherences:
            accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
        acc_fit = FitWeibull(coherences, accuracy, guess=[0.0, 0.2], display=0)
        condition_accuracy_thresh[condition]=acc_fit.inverse(0.8)

        condition_overall_accuracy_rt[condition][0]=np.mean(condition_overall_accuracy_rt[condition][0])
        condition_overall_accuracy_rt[condition][1]=np.mean(condition_overall_accuracy_rt[condition][1])

    for stim_condition in ['depolarizing', 'hyperpolarizing']:
        condition_coherence_rt_diff[stim_condition]={}
        coherences=sorted(condition_coherence_rt[stim_condition].keys())
        for coherence in coherences:
            condition_coherence_rt_diff[stim_condition][coherence]=np.mean(condition_coherence_rt[stim_condition][coherence])-np.mean(condition_coherence_rt['control'][coherence])

    if plot:
        plot_choice_accuracy(colors, condition_coherence_accuracy)

        plot_choice_rt(colors, condition_coherence_rt)

        plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=False)

    return condition_coherence_accuracy, condition_coherence_rt, condition_coherence_rt_diff, condition_accuracy_thresh, condition_overall_accuracy_rt


def analyze_subject_choice_hysteresis(subject, plot=False):
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    condition_logistic_params={
        'a1': {},
        'a2': {}
    }
    for condition,subj_data in subject.iteritems():
        trial_data=subj_data['behavior']
        # Dict of coherence levels
        condition_coherence_choices['L*'][condition]={}
        condition_coherence_choices['R*'][condition]={}

        # For each trial
        for trial_idx in range(1,trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]*trial_data[trial_idx,1]
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]

            if not math.isnan(resp) and not math.isnan(last_resp):

                if last_resp<0:
                    if not coherence in condition_coherence_choices['L*'][condition]:
                        condition_coherence_choices['L*'][condition][coherence]=[]
                        # Append 0 to list if left (-1) or 1 if right
                    condition_coherence_choices['L*'][condition][coherence].append(np.max([0,resp]))
                elif last_resp>0:
                    # List of rightward choices (0=left, 1=right)
                    if not coherence in condition_coherence_choices['R*'][condition]:
                        condition_coherence_choices['R*'][condition][coherence]=[]
                        # Append 0 to list if left (-1) or 1 if right
                    condition_coherence_choices['R*'][condition][coherence].append(np.max([0,resp]))

        choice_probs=[]
        full_coherences=[]
        for coherence in condition_coherence_choices['L*'][condition]:
            choice_probs.append(np.mean(condition_coherence_choices['L*'][condition][coherence]))
            full_coherences.append(coherence)
        acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
        condition_sigmoid_offsets['L*'][condition]=acc_fit.inverse(0.5)

        choice_probs=[]
        full_coherences=[]
        for coherence in condition_coherence_choices['R*'][condition]:
            choice_probs.append(np.mean(condition_coherence_choices['R*'][condition][coherence]))
            full_coherences.append(coherence)
        acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
        condition_sigmoid_offsets['R*'][condition]=acc_fit.inverse(0.5)

        data=pd.DataFrame({
            'resp': np.clip(trial_data[1:,4],0,1),
            'coh': trial_data[1:,2]*trial_data[1:,1],
            'last_resp': trial_data[1:,5]
        })
        data['intercept']=1.0

        data=data.dropna(axis=0)
        logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
        result = logit.fit(method='bfgs',disp=False)
        condition_logistic_params['a1'][condition]=result.params['coh']
        condition_logistic_params['a2'][condition]=result.params['last_resp']

    if plot:
        plot_choice_probability(colors, condition_coherence_choices)

    return condition_coherence_choices, condition_sigmoid_offsets, condition_logistic_params


def analyze_firing_rates(subjects):
    condition_choose_bias_rates={}
    condition_choose_unbias_rates={}
    subject_net_params={}
    subject_sim_params={}
    for subject in subjects:
        subj_choose_bias_rates, subj_choose_unbias_rates=analyze_subject_firing_rates(subject, plot=False)
        for condition in conditions:
            if not condition in condition_choose_bias_rates:
                condition_choose_bias_rates[condition]={
                    'chosen_rate':[],
                    'unchosen_rate': [],
                }
                condition_choose_unbias_rates[condition]={
                    'chosen_rate':[],
                    'unchosen_rate':[]
                }
                subject_net_params[condition]=[]
                subject_sim_params[condition]=[]
            condition_choose_bias_rates[condition]['chosen_rate'].append(np.mean(np.array(subj_choose_bias_rates[condition]['chosen_rate']),axis=0))
            condition_choose_bias_rates[condition]['unchosen_rate'].append(np.mean(np.array(subj_choose_bias_rates[condition]['unchosen_rate']),axis=0))
            condition_choose_unbias_rates[condition]['chosen_rate'].append(np.mean(np.array(subj_choose_unbias_rates[condition]['chosen_rate']),axis=0))
            condition_choose_unbias_rates[condition]['unchosen_rate'].append(np.mean(np.array(subj_choose_unbias_rates[condition]['unchosen_rate']),axis=0))
            subject_net_params[condition].append(subject[condition]['network_params'])
            subject_sim_params[condition].append(subject[condition]['sim_params'])

    subj_thresholds=[]
    sim_params=subjects[0]['control']['sim_params']
    for subject in subjects:
        subj_thresholds.append(subject['control']['network_params'].resp_threshold/Hz)
    max_rates=[np.mean(subj_thresholds)]
    condition_mean_choose_bias_rates={}
    condition_stderr_choose_bias_rates={}
    condition_mean_choose_unbias_rates={}
    condition_stderr_choose_unbias_rates={}
    for condition in conditions:
        condition_mean_choose_bias_rates[condition]={}
        condition_stderr_choose_bias_rates[condition]={}
        condition_mean_choose_unbias_rates[condition]={}
        condition_stderr_choose_unbias_rates[condition]={}
        for group in ['chosen_rate','unchosen_rate']:
            condition_mean_choose_bias_rates[condition][group]=np.mean(np.array(condition_choose_bias_rates[condition][group]),axis=0)
            condition_stderr_choose_bias_rates[condition][group]=np.std(np.array(condition_choose_bias_rates[condition][group]),axis=0)/np.sqrt(len(subjects))
            condition_mean_choose_unbias_rates[condition][group]=np.mean(np.array(condition_choose_unbias_rates[condition][group]),axis=0)
            condition_stderr_choose_unbias_rates[condition][group]=np.std(np.array(condition_choose_unbias_rates[condition][group]),axis=0)/np.sqrt(len(subjects))
            max_rates.append(np.max(condition_mean_choose_bias_rates[condition][group][500:]))
            max_rates.append(np.max(condition_mean_choose_unbias_rates[condition][group][500:]))
    max_rate=np.max(max_rates)

    fig1=plt.figure()
    ax1= fig1.add_subplot(211)
    fig2=plt.figure()
    ax2=fig2.add_subplot(211)

    rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
        alpha=0.25, facecolor='yellow', edgecolor='none')
    ax1.add_patch(rect)

    for condition in conditions:
        chosen_e_rate=condition_mean_choose_bias_rates[condition]['chosen_rate']
        std_chosen_e_rate=condition_stderr_choose_bias_rates[condition]['chosen_rate']
        time_ticks=(np.array(range(chosen_e_rate.shape[0]))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        ax1.plot(time_ticks, chosen_e_rate, colors[condition], label='chosen')
        ax1.fill_between(time_ticks, chosen_e_rate-std_chosen_e_rate, chosen_e_rate+std_chosen_e_rate, alpha=0.5, facecolor=colors[condition])
        ax2.plot(time_ticks, chosen_e_rate, colors[condition], label='chosen')
        ax2.fill_between(time_ticks, chosen_e_rate-std_chosen_e_rate, chosen_e_rate+std_chosen_e_rate, alpha=0.5, facecolor=colors[condition])

        unchosen_e_rate=condition_mean_choose_bias_rates[condition]['unchosen_rate']
        unchosen_std_e_rate=condition_stderr_choose_bias_rates[condition]['unchosen_rate']
        time_ticks=(np.array(range(unchosen_e_rate.shape[0]))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        ax1.plot(time_ticks, unchosen_e_rate, '--%s' % colors[condition], label='unchosen')
        ax1.fill_between(time_ticks, unchosen_e_rate-unchosen_std_e_rate, unchosen_e_rate+unchosen_std_e_rate, alpha=0.5, facecolor=colors[condition])
        ax2.plot(time_ticks, unchosen_e_rate, '--%s' % colors[condition], label='unchosen')
        ax2.fill_between(time_ticks, unchosen_e_rate-unchosen_std_e_rate, unchosen_e_rate+unchosen_std_e_rate, alpha=0.5, facecolor=colors[condition])

        rts=[]
        for resp_thresh in subj_thresholds:
            rt, choice = get_response_time(np.array([chosen_e_rate,unchosen_e_rate]), sim_params.stim_start_time,
                                           sim_params.stim_end_time, upper_threshold = resp_thresh*Hz, dt = sim_params.dt)
            if rt is not None:
                rts.append(rt)
        mean_rt=np.mean(rts)
        stderr_rt=np.std(rts)/np.sqrt(len(rts))
        ax1.plot([mean_rt,mean_rt],[0, max_rate+5],'%s--' % colors[condition])
        #ax.fill_between([mean_rt-stderr_rt,mean_rt+stderr_rt],[0,max_rate+5],facecolor=colors[condition],alpha=0.5)
        rect=Rectangle((mean_rt-stderr_rt,0),2*stderr_rt, max_rate+5, alpha=0.25, facecolor=colors[condition], edgecolor='none')
        ax1.add_patch(rect)


    ax1.set_xlim([-950,1950])
    ax1.set_ylim(0,max_rate+5)
    ax2.set_xlim([-500,0])
    ax2.set_ylim([4,8])
    mean_resp_threshold=np.mean(subj_thresholds)
    stderr_resp_threshold=np.std(subj_thresholds)/np.sqrt(len(subj_thresholds))
    ax1.plot([0-sim_params.stim_start_time/ms, (sim_params.trial_duration-sim_params.stim_start_time)/ms],
        [mean_resp_threshold/hertz, mean_resp_threshold/hertz], 'k--')
    #ax.fill_between([0-sim_params.stim_start_time/ms, (sim_params.trial_duration-sim_params.stim_start_time)/ms],
    #                [(mean_resp_threshold-stderr_resp_threshold), (mean_resp_threshold+stderr_resp_threshold)], 'k')
    rect=Rectangle((0-sim_params.stim_start_time/ms,mean_resp_threshold-stderr_resp_threshold),
                   sim_params.trial_duration/ms,2*stderr_resp_threshold, alpha=0.25, facecolor='black', edgecolor='none')
    ax1.add_patch(rect)

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1= fig1.add_subplot(212)
    ax2= fig2.add_subplot(212)

    rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
        alpha=0.25, facecolor='yellow', edgecolor='none')
    ax1.add_patch(rect)

    for condition in conditions:
        chosen_e_rate=condition_mean_choose_unbias_rates[condition]['chosen_rate']
        std_chosen_e_rate=condition_stderr_choose_unbias_rates[condition]['chosen_rate']
        time_ticks=(np.array(range(chosen_e_rate.shape[0]))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        ax1.plot(time_ticks, chosen_e_rate, colors[condition], label='chosen')
        ax1.fill_between(time_ticks, chosen_e_rate-std_chosen_e_rate, chosen_e_rate+std_chosen_e_rate, alpha=0.5, facecolor=colors[condition])
        ax2.plot(time_ticks, chosen_e_rate, colors[condition], label='chosen')
        ax2.fill_between(time_ticks, chosen_e_rate-std_chosen_e_rate, chosen_e_rate+std_chosen_e_rate, alpha=0.5, facecolor=colors[condition])

        unchosen_e_rate=condition_mean_choose_unbias_rates[condition]['unchosen_rate']
        unchosen_std_e_rate=condition_stderr_choose_unbias_rates[condition]['unchosen_rate']
        time_ticks=(np.array(range(unchosen_e_rate.shape[0]))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        ax1.plot(time_ticks, unchosen_e_rate, '--%s' % colors[condition], label='unchosen')
        ax1.fill_between(time_ticks, unchosen_e_rate-unchosen_std_e_rate, unchosen_e_rate+unchosen_std_e_rate, alpha=0.5, facecolor=colors[condition])
        ax2.plot(time_ticks, unchosen_e_rate, '--%s' % colors[condition], label='unchosen')
        ax2.fill_between(time_ticks, unchosen_e_rate-unchosen_std_e_rate, unchosen_e_rate+unchosen_std_e_rate, alpha=0.5, facecolor=colors[condition])

        rts=[]
        for resp_thresh in subj_thresholds:
            rt, choice = get_response_time(np.array([chosen_e_rate,unchosen_e_rate]), sim_params.stim_start_time,
                                           sim_params.stim_end_time, upper_threshold = resp_thresh*Hz, dt = sim_params.dt)
            if rt is not None:
                rts.append(rt)
        mean_rt=np.mean(rts)
        stderr_rt=np.std(rts)/np.sqrt(len(rts))
        ax1.plot([mean_rt,mean_rt],[0, max_rate+5],'%s--' % colors[condition])
        rect=Rectangle((mean_rt-stderr_rt,0),2*stderr_rt, max_rate+5, alpha=0.25, facecolor=colors[condition], edgecolor='none')
        ax1.add_patch(rect)

    ax1.set_xlim([-950,1950])
    ax1.set_ylim(0,max_rate+5)
    ax2.set_xlim([-500,0])
    ax2.set_ylim([4,8])
    mean_resp_threshold=np.mean(subj_thresholds)
    stderr_resp_threshold=np.std(subj_thresholds)/np.sqrt(len(subj_thresholds))
    ax1.plot([0-sim_params.stim_start_time/ms, (sim_params.trial_duration-sim_params.stim_start_time)/ms],
        [mean_resp_threshold/hertz, mean_resp_threshold/hertz], 'k--')
    # ax.fill_between([0-sim_params.stim_start_time/ms, (sim_params.trial_duration-sim_params.stim_start_time)/ms],
    #                  [(mean_resp_threshold-stderr_resp_threshold)/hertz, (mean_resp_threshold+stderr_resp_threshold)/hertz], 'k')
    rect=Rectangle((0-sim_params.stim_start_time/ms,mean_resp_threshold-stderr_resp_threshold),
                   sim_params.trial_duration/ms,2*stderr_resp_threshold, alpha=0.25, facecolor='black', edgecolor='none')
    ax1.add_patch(rect)

    ax1.legend(loc='best')
    ax2.legend(loc='best')



def analyze_accuracy_rt(subjects, plot=True, print_stats=True):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    condition_overall_accuracy_rt={}
    # For each subject
    for subject in subjects:

        subj_condition_coherence_accuracy, subj_condition_coherence_rt, subj_condition_coherence_rt_diff,\
            subj_condition_accuracy_thresh,subj_condition_overall_accuracy_rt=analyze_subject_accuracy_rt(subject,
                                                                                                          plot=False)

        for condition in conditions:
            if not condition in condition_coherence_accuracy:
                condition_coherence_accuracy[condition]={}
                condition_coherence_rt[condition]={}
                condition_accuracy_thresh[condition]=[]
                condition_overall_accuracy_rt[condition]=[[],[]]
            condition_accuracy_thresh[condition].append(subj_condition_accuracy_thresh[condition])
            condition_overall_accuracy_rt[condition][0].append(subj_condition_overall_accuracy_rt[condition][0])
            condition_overall_accuracy_rt[condition][1].append(subj_condition_overall_accuracy_rt[condition][1])

            for coherence in subj_condition_coherence_accuracy[condition]:
                if not coherence in condition_coherence_accuracy[condition]:
                    condition_coherence_accuracy[condition][coherence]=[]
                condition_coherence_accuracy[condition][coherence].append(np.mean(subj_condition_coherence_accuracy[condition][coherence]))

            for coherence in subj_condition_coherence_rt[condition]:
                if not coherence in condition_coherence_rt[condition]:
                    condition_coherence_rt[condition][coherence]=[]
                condition_coherence_rt[condition][coherence].append(np.mean(subj_condition_coherence_rt[condition][coherence]))

        for condition in subj_condition_coherence_rt_diff:
            if not condition in condition_coherence_rt_diff:
                condition_coherence_rt_diff[condition]={}
            for coherence in subj_condition_coherence_rt_diff[condition]:
                if not coherence in condition_coherence_rt_diff[condition]:
                    condition_coherence_rt_diff[condition][coherence]=[]
                condition_coherence_rt_diff[condition][coherence].append(subj_condition_coherence_rt_diff[condition][coherence])

    if plot:
        plot_sat(colors, condition_overall_accuracy_rt)

        plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=True)

        plot_choice_rt(colors, condition_coherence_rt)

        plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=True)

        plot_accuracy_thresh(colors, condition_accuracy_thresh)

    thresh_results={
        'depolarizing': {},
        'hyperpolarizing': {},
    }
    (thresh_results['hyperpolarizing']['x'],thresh_results['hyperpolarizing']['p'])=wilcoxon(condition_accuracy_thresh['control'],condition_accuracy_thresh['hyperpolarizing'])
    (thresh_results['depolarizing']['x'],thresh_results['depolarizing']['p'])=wilcoxon(condition_accuracy_thresh['control'],condition_accuracy_thresh['depolarizing'])

    rtdiff_results={
        'depolarizing': {'coh': {}, 'intercept':{}},
        'hyperpolarizing': {'coh': {}, 'intercept':{}}
    }
    for condition in ['depolarizing','hyperpolarizing']:
        coh=[]
        rt_diff=[]
        for coherence in condition_coherence_rt_diff[condition]:
            for diff in condition_coherence_rt_diff[condition][coherence]:
                coh.append(coherence)
                rt_diff.append(diff)
        data=pd.DataFrame({
            'coh': coh,
            'rt_diff': rt_diff
        })
        data['intercept']=1.0
        lr = sm.GLM(data['rt_diff'], data[['coh','intercept']])
        result = lr.fit()
        for param in ['coh','intercept']:
            rtdiff_results[condition.lower()][param]['x']=result.params[param]
            rtdiff_results[condition.lower()][param]['t']=result.tvalues[param]
            rtdiff_results[condition.lower()][param]['p']=result.pvalues[param]

    sat_results={
        'hyperpolarizing': {},
        'depolarizing': {}
    }
    control_overall_accuracy_rt=condition_overall_accuracy_rt['control']
    control_sat_ratio=(1000.0/np.array(control_overall_accuracy_rt[1]))/np.array(control_overall_accuracy_rt[0])
    for condition in ['depolarizing','hyperpolarizing']:
        cond_overall_accuracy_rt=condition_overall_accuracy_rt[condition]
        cond_sat_ratio=(1000.0/np.array(cond_overall_accuracy_rt[1]))/np.array(cond_overall_accuracy_rt[0])
        (sat_results[condition]['x'],sat_results[condition]['p'])=wilcoxon(control_sat_ratio,cond_sat_ratio)

    if print_stats:
        print('Accuracy Threshold')
        for condition, results in thresh_results.iteritems():
            print('%s: x=%.4f, p=%.4f' % (condition, results['x'],results['p']))

        print('')
        print('RT Diff')
        for condition, results in rtdiff_results.iteritems():
            print('%s, coherence: x=%.4f, t=%.4f, p=%.4f, intercept: x=%.4f, t=%.4f, p=%.4f' % (condition,
                                                                                                results['coh']['x'],
                                                                                                results['coh']['t'],
                                                                                                results['coh']['p'],
                                                                                                results['intercept']['x'],
                                                                                                results['intercept']['t'],
                                                                                                results['intercept']['p']))

        print('')
        print('SAT')
        for condition, results in sat_results.iteritems():
            print('%s: x=%.4f, p=%.4f' % (condition, results['x'],results['p']))

    return thresh_results, rtdiff_results, sat_results


def analyze_choice_hysteresis(subjects, plot=True, print_stats=True):
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    condition_logistic_params={
        'a1': {},
        'a2': {}
    }

    # For each subject
    for idx,subject in enumerate(subjects):
        subj_condition_coherence_choices, subj_condition_sigmoid_offsets, subj_condition_logistic_params=analyze_subject_choice_hysteresis(subject, plot=False)

        for condition in conditions:
            if not condition in condition_coherence_choices['L*']:
                condition_coherence_choices['L*'][condition]={}
                condition_coherence_choices['R*'][condition]={}
                condition_sigmoid_offsets['L*'][condition]=[]
                condition_sigmoid_offsets['R*'][condition]=[]
                condition_logistic_params['a1'][condition]=[]
                condition_logistic_params['a2'][condition]=[]

            condition_sigmoid_offsets['L*'][condition].append(subj_condition_sigmoid_offsets['L*'][condition])
            condition_sigmoid_offsets['R*'][condition].append(subj_condition_sigmoid_offsets['R*'][condition])

            condition_logistic_params['a1'][condition].append(subj_condition_logistic_params['a1'][condition])
            condition_logistic_params['a2'][condition].append(subj_condition_logistic_params['a2'][condition])

            for coherence in subj_condition_coherence_choices['L*'][condition]:
                if not coherence in condition_coherence_choices['L*'][condition]:
                    condition_coherence_choices['L*'][condition][coherence]=[]
                condition_coherence_choices['L*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['L*'][condition][coherence]))

            for coherence in subj_condition_coherence_choices['R*'][condition]:
                if not coherence in condition_coherence_choices['R*'][condition]:
                    condition_coherence_choices['R*'][condition][coherence]=[]
                condition_coherence_choices['R*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['R*'][condition][coherence]))

    if plot:
        plot_indifference(colors, condition_sigmoid_offsets)

        plot_indifference_hist(colors, condition_sigmoid_offsets)

        plot_choice_probability(colors, condition_coherence_choices)

        plot_logistic_parameter_ratio(colors, condition_logistic_params)

    indec_results={
        'depolarizing': {},
        'hyperpolarizing': {},
    }
    (indec_results['hyperpolarizing']['x'],indec_results['hyperpolarizing']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['control'])-np.array(condition_sigmoid_offsets['R*']['control']),
        np.array(condition_sigmoid_offsets['L*']['hyperpolarizing'])-np.array(condition_sigmoid_offsets['R*']['hyperpolarizing']))
    (indec_results['depolarizing']['x'],indec_results['depolarizing']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['control'])-np.array(condition_sigmoid_offsets['R*']['control']),
        np.array(condition_sigmoid_offsets['L*']['depolarizing'])-np.array(condition_sigmoid_offsets['R*']['depolarizing']))

    log_results={
        'depolarizing': {},
        'hyperpolarizing': {},
    }
    control_ratio=np.array(condition_logistic_params['a2']['control'])/np.array(condition_logistic_params['a1']['control'])
    anode_ratio=np.array(condition_logistic_params['a2']['depolarizing'])/np.array(condition_logistic_params['a1']['depolarizing'])
    (log_results['depolarizing']['x'],log_results['depolarizing']['p'])=wilcoxon(control_ratio, anode_ratio)

    cathode_ratio=np.array(condition_logistic_params['a2']['hyperpolarizing'])/np.array(condition_logistic_params['a1']['hyperpolarizing'])
    (log_results['hyperpolarizing']['x'],log_results['hyperpolarizing']['p'])=wilcoxon(control_ratio, cathode_ratio)

    if print_stats:
        print('')
        print('Indecision Points')
        for condition, results in indec_results.iteritems():
            print('%s, x=%.4f, p=%.6f' % (condition,results['x'],results['p']))

        print('')
        print('Logistic Regression')
        print('Control, median=%.4f' % np.median(control_ratio))
        print('Anode, median=%.4f' % np.median(anode_ratio))
        print('Cathode, median=%.4f' % np.median(cathode_ratio))
        for condition, results in log_results.iteritems():
            print('%s, x=%.4f, p=%.4f' % (condition, results['x'], results['p']))

    return indec_results, log_results


def plot_indifference(colors, condition_sigmoid_offsets):
    fig = plt.figure()
    limits = [-.6, .6]
    ax = fig.add_subplot(1, 1, 1, aspect='equal',adjustable='box-forced')
    for condition in conditions:
        ellipse_x, ellipse_y=get_twod_confidence_interval(condition_sigmoid_offsets['L*'][condition],condition_sigmoid_offsets['R*'][condition])
        ax.plot(ellipse_x,ellipse_y,'%s-' % colors[condition])
        ax.plot(condition_sigmoid_offsets['L*'][condition],condition_sigmoid_offsets['R*'][condition],'o%s' % colors[condition], label=condition)
    ax.plot(limits, limits, '--k')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel('Indifference Point for L* Trials')
    ax.set_ylabel('Indifference Point for R* Trials')
    ax.legend(loc='best')


def plot_indifference_hist(colors, condition_sigmoid_offsets):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    binwidth=0.03
    lims=[-.2,.4]
    xx=np.arange(lims[0],lims[1],0.001)
    for condition in conditions:
        diff=np.array(condition_sigmoid_offsets['L*'][condition])-np.array(condition_sigmoid_offsets['R*'][condition])
        bins=np.arange(min(diff), max(diff)+binwidth, binwidth)
        hist,edges=np.histogram(diff, bins=bins)
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist/float(len(diff))*100.0, color=colors[condition], alpha=0.75, label=condition, width=binwidth)
        (mu, sigma) = norm.fit(diff)
        y = normpdf(xx, mu, sigma)*binwidth*100.0
        ax.plot(xx, y,'%s--' % colors[condition], linewidth=2)
    ax.set_xlim(lims)
    ax.set_ylim([0,30])
    ax.set_xlabel('L*-R* Indifference')
    ax.set_ylabel('% subjects')
    ax.legend(loc='best')


def plot_sat(colors, condition_overall_accuracy_rt):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for condition,overall_accuracy_rt in condition_overall_accuracy_rt.iteritems():
        ellipse_x, ellipse_y=get_twod_confidence_interval(overall_accuracy_rt[1],1-np.array(overall_accuracy_rt[0]))
        ax.plot(ellipse_x,ellipse_y,'%s-' % colors[condition])
        ax.plot(overall_accuracy_rt[1],1-np.array(overall_accuracy_rt[0]),'o%s' % colors[condition], label=condition)
    ax.legend(loc='best')
    ax.set_xlabel('Mean RT')
    ax.set_ylabel('Error Rate')

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    binwidth=.075
    lims=[-1,3]
    xx=np.arange(lims[0],lims[1],0.001)
    control_overall_accuracy_rt=condition_overall_accuracy_rt['control']
    control_sat_ratio=(1000.0/np.array(control_overall_accuracy_rt[1]))/np.array(control_overall_accuracy_rt[0])

    for stim_condition in ['depolarizing', 'hyperpolarizing']:
        cond_overall_accuracy_rt=condition_overall_accuracy_rt[stim_condition]
        cond_sat_ratio=(1000.0/np.array(cond_overall_accuracy_rt[1]))/np.array(cond_overall_accuracy_rt[0])

        sat_diff=cond_sat_ratio-control_sat_ratio
        bins=np.arange(min(sat_diff), max(sat_diff) + binwidth, binwidth)
        hist,edges=np.histogram(sat_diff, bins=bins)
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist/float(len(control_sat_ratio))*100.0, color=colors[stim_condition], alpha=0.75, label=stim_condition, width=binwidth)
        (mu, sigma) = norm.fit(sat_diff)
        y = normpdf(xx, mu, sigma)*binwidth*100.0
        ax.plot(xx, y, '%s--' % colors[stim_condition], linewidth=2)
    ax.legend(loc='best')
    ax.set_xlim(lims)
    ax.set_xlabel('Speed/Accuracy Change')
    ax.set_ylabel('% Subjects')

def plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for condition in conditions:
        coherences = sorted(condition_coherence_accuracy[condition].keys())
        mean_accuracy=[]
        stderr_accuracy=[]
        for coherence in coherences:
            mean_accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
            if plot_err:
                stderr_accuracy.append(np.std(condition_coherence_accuracy[condition][coherence])/np.sqrt(len(condition_coherence_accuracy[condition][coherence])))
        acc_fit = FitWeibull(coherences, mean_accuracy, guess=[0.0, 0.2], display=0)
        smoothInt = np.arange(.01, 1.0, 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothResp, colors[condition], label=condition)
        if plot_err:
            ax.errorbar(coherences, mean_accuracy, yerr=stderr_accuracy, fmt='o%s' % colors[condition])
        else:
            ax.plot(coherences, mean_accuracy, 'o%s' % colors[condition])
        thresh=acc_fit.inverse(0.8)
        ax.plot([thresh,thresh],[0.5,1],'--%s' % colors[condition])
    ax.legend(loc='best')
    ax.set_xlim([0.01,1.0])
    ax.set_ylim([0.5,1.0])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% Correct')


def plot_choice_rt(colors, condition_coherence_rt):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for condition in conditions:
        coherences = sorted(condition_coherence_rt[condition].keys())
        mean_rt=[]
        stderr_rt=[]
        for coherence in coherences:
            mean_rt.append(np.mean(condition_coherence_rt[condition][coherence]))
            stderr_rt.append(np.std(condition_coherence_rt[condition][coherence])/np.sqrt(len(condition_coherence_rt[condition][coherence])))
        rt_fit = FitRT(coherences, mean_rt, guess=[1,1,1], display=0)
        smoothInt = np.arange(min(coherences), max(coherences), 0.001)
        smoothRT = rt_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothRT, colors[condition], label=condition)
        ax.errorbar(coherences, mean_rt, yerr=stderr_rt,fmt='o%s' % colors[condition])
    ax.set_xlim([0.02,1])
    ax.set_ylim([250, 600])
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT')


def plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=False):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for stim_condition in ['depolarizing', 'hyperpolarizing']:
        coherences = np.array(sorted(condition_coherence_rt_diff[stim_condition].keys()))
        mean_diff=[]
        stderr_diff=[]
        for coherence in coherences:
            mean_diff.append(np.mean(condition_coherence_rt_diff[stim_condition][coherence]))
            if plot_err:
                stderr_diff.append(np.std(condition_coherence_rt_diff[stim_condition][coherence])/np.sqrt(len(condition_coherence_rt_diff[stim_condition][coherence])))
        mean_diff=np.array(mean_diff)

        clf = LinearRegression()
        clf.fit(np.expand_dims(coherences,axis=1),np.expand_dims(mean_diff,axis=1))
        a = clf.coef_[0][0]
        b = clf.intercept_[0]
        r_sqr=clf.score(np.expand_dims(coherences,axis=1), np.expand_dims(mean_diff,axis=1))
        ax.plot([np.min(coherences), np.max(coherences)], [a * np.min(coherences) + b, a * np.max(coherences) + b], '--%s' % colors[stim_condition],
            label='r^2=%.3f' % r_sqr)

        if plot_err:
            ax.errorbar(coherences, mean_diff, yerr=stderr_diff, fmt='o%s' % colors[stim_condition], label=stim_condition)
        else:
            ax.plot(coherences, mean_diff, 'o%s' % colors[stim_condition], label=stim_condition)
    ax.legend(loc='best')
    ax.set_xlim([0,0.55])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT Difference')


def plot_choice_probability(colors, condition_coherence_choices):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for condition in conditions:
        left_coherences=sorted(condition_coherence_choices['L*'][condition].keys())
        right_coherences=sorted(condition_coherence_choices['R*'][condition].keys())
        left_choice_probs = []
        right_choice_probs = []
        for coherence in left_coherences:
            left_choice_probs.append(np.mean(condition_coherence_choices['L*'][condition][coherence]))
        for coherence in right_coherences:
            right_choice_probs.append(np.mean(condition_coherence_choices['R*'][condition][coherence]))
        plot_condition_choice_probability(ax, colors[condition],  left_coherences, left_choice_probs, right_coherences, right_choice_probs, extra_label=condition)


def plot_logistic_parameter_ratio(colors, condition_logistic_params):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lims=[-.1,.25]
    xx=np.arange(lims[0],lims[1],0.001)
    binwidth=.02
    for condition in conditions:
        ratio=np.array(condition_logistic_params['a2'][condition]) / np.array(condition_logistic_params['a1'][condition])
        bins=np.arange(min(ratio), max(ratio) + binwidth, binwidth)
        hist,edges=np.histogram(ratio, bins=bins)
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist/float(len(ratio))*100.0, color=colors[condition], alpha=0.75, label=condition, width=binwidth)
        (mu, sigma) = norm.fit(ratio)
        y = normpdf(xx, mu, sigma)*binwidth*100.0
        ax.plot(xx, y, '%s--' % colors[condition], linewidth=2)
    ax.legend(loc='best')
    ax.set_xlim(lims)
    ax.set_ylim([0, 40])
    ax.set_xlabel('a2/a1')
    ax.set_ylabel('% subjects')


def plot_accuracy_thresh(colors, condition_accuracy_thresh):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for condition in conditions:
        ax.hist(condition_accuracy_thresh[condition], label=condition, color=colors[condition], bins=10, alpha=0.75)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('# Subjects')
    ax.legend(loc='best')


def export_behavioral_data_long(subjects, output_filename):
    out_file=open(output_filename,'w')
    out_file.write('Subject,Condition,Trial,Direction,Coherence,Resp,LastResp,Correct,RT\n')
    for subj_id,subj_data in enumerate(subjects):
        for condition,trial_data in subj_data.iteritems():
            for i in range(trial_data.shape[0]):
                out_file.write('%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (subj_id, condition, int(trial_data[i,0]),
                                                                             trial_data[i,1], trial_data[i,2],
                                                                             trial_data[i,4], trial_data[i,5],
                                                                             trial_data[i,3], trial_data[i,6]))



def compute_score(virtual_subjects):
    (indec_results, log_results)=analyze_choice_hysteresis(virtual_subjects, plot=False, print_stats=False)
    (thresh_results, rtdiff_results, sat_results)=analyze_accuracy_rt(virtual_subjects, plot=False, print_stats=False)

    indec_score=1.0
    log_score=1.0
    thresh_score=1.0
    rt_diff_score=1.0
    for condition in ['depolarizing','hyperpolarizing']:
        if indec_results[condition]['p']>=0.03:
            indec_score=indec_score*(1.0-indec_results[condition]['p'])
        if log_results[condition]['p']>=0.03:
            log_score=log_score*(1.0-log_results[condition]['p'])
        if thresh_results[condition]['p']<=0.1:
            thresh_score=thresh_score*thresh_results[condition]['p']
        if rtdiff_results[condition]['coh']['p']>=0.03:
            rt_diff_score=rt_diff_score*(1.0-rtdiff_results[condition]['coh']['p'])
    return indec_score*log_score*thresh_score*rt_diff_score



def optimize(subj_ids):
    virtual_subjects={}
    for subj_id in subj_ids:
        virtual_subjects[subj_id]=read_subject_data('/home/jbonaiuto/Projects/pySBI/data/rdmd', subj_id,
                                                  ['control','depolarizing','hyperpolarizing'], filter=True,
                                                  response_only=True)
    base_score=compute_score(virtual_subjects.values())
    print('Base score=%.4f' % base_score)
    best_subjects=copy.deepcopy(virtual_subjects)

    max_subj_id=np.max(subj_ids)

    improvement=True
    best_score=base_score
    # Iterate while we can improve
    while improvement:
        # Find the worst subject - the subject who improves the score by being removed
        worst_subj=-1
        worst_subj_score=0
        # Try removing each subject
        for test_subj_id in best_subjects.keys():
            # Get the remaining subjects
            remaining_subjects={}
            for subj_id, subject in best_subjects.iteritems():
                if not subj_id==test_subj_id:
                    remaining_subjects[subj_id]=subject
            # Get the score without this subject
            score=compute_score(remaining_subjects.values())
            # If it improves the score - update the worst subjet and score
            if score>worst_subj_score:
                worst_subj=test_subj_id
                worst_subj_score=score

        # If we found a subject who by removing, improves the score, remove that subject and try to improve that subject
        if worst_subj>-1:
            del best_subjects[worst_subj]
            tries=0
            improved_score=worst_subj_score
            while True:
                new_subj_idx=max_subj_id+1
                max_subj_id=new_subj_idx
                # Trials per condition
                trials_per_condition=100
                # Max stimulation intensity
                stim_intensity_max=0.5*pA
                # Stimulation conditions
                conditions={
                    'control': simulation_params(ntrials=trials_per_condition, trial_duration=3*second, stim_start_time=1*second,
                                                 stim_end_time=2*second),
                    'depolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3*second, stim_start_time=1*second,
                                                 stim_end_time=2*second, p_dcs=stim_intensity_max, i_dcs=-0.5*stim_intensity_max,
                                                 dcs_start_time=0*second, dcs_end_time=3*second),
                    'hyperpolarizing': simulation_params(ntrials=trials_per_condition, trial_duration=3*second, stim_start_time=1*second,
                                                 stim_end_time=2*second, p_dcs=-1*stim_intensity_max, i_dcs=0.5*stim_intensity_max,
                                                 dcs_start_time=0*second, dcs_end_time=3*second)
                }
                run_virtual_subjects([new_subj_idx], conditions, '/home/jbonaiuto/Projects/pySBI/data/rdmd/',
                                     '/home/jbonaiuto/Projects/pySBI/data/rerw/subjects/fitted_behavioral_params.h5')

                # Test score with new subject
                test_subjects=copy.deepcopy(best_subjects)
                test_subjects[new_subj_idx]=read_subject_data('/home/jbonaiuto/Projects/pySBI/data/rdmd', new_subj_idx,
                                                  ['control','depolarizing','hyperpolarizing'], filter=True,
                                                  response_only=True)
                new_score=compute_score(test_subjects.values())
                if new_score>best_score:
                    improved_score=new_score
                    best_subjects[new_subj_idx]=test_subjects[new_subj_idx]
                    break
                tries=tries+1
                if tries>5:
                    break
            if improved_score<=best_score:
                improvement=False
                best_subjects[worst_subj]=read_subject_data('/home/jbonaiuto/Projects/pySBI/data/rdmd', worst_subj,
                                                  ['control','depolarizing','hyperpolarizing'], filter=True,
                                                  response_only=True)
            else:
                best_score=improved_score
                print(best_subjects.keys())
                print('Score=%.4f' % best_score)
        else:
            improvement=False

    print(best_subjects.keys())
    analyze_choice_hysteresis(best_subjects.values())
    print('')
    analyze_accuracy_rt(best_subjects.values())
    plt.show()







if __name__=='__main__':
    #subj_ids=[0, 32, 2, 35, 4, 7, 8, 9, 10, 11, 12, 14, 16, 19, 20, 21, 22, 23, 36]
    subj_ids=[0, 32, 2, 35, 4, 7, 8, 9, 10, 43, 12, 14, 16, 19, 20, 21, 22, 23, 36, 43]
    # virtual_subjects=[]
    # for subj_idx in subj_ids:
    #      virtual_subjects.append(read_subject_data('/home/jbonaiuto/Projects/pySBI/data/rdmd', subj_idx,
    #                                             ['control','depolarizing','hyperpolarizing'], filter=True,
    #                                             response_only=True, neural_data=False))
    # analyze_choice_hysteresis(virtual_subjects)
    # print('')
    # analyze_accuracy_rt(virtual_subjects)
    #analyze_subject_firing_rates(virtual_subjects[0], plot=True)
    # analyze_firing_rates(virtual_subjects)
    # plt.show()
    #for subject in virtual_subjects:
    #    analyze_subject_accuracy_rt(subject, plot=True)
    #plt.show()
    optimize(subj_ids)