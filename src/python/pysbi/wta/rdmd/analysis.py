import h5py
import os
import math
from matplotlib.mlab import normpdf
import numpy as np
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LinearRegression
from pysbi.util.utils import mdm_outliers, FitSigmoid, get_twod_confidence_interval, FitWeibull, FitRT
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

colors={
    'control': 'b',
    'depolarizing': 'r',
    'hyperpolarizing': 'g'
}

coherences=[0.0320, 0.0640, 0.1280, 0.2560, 0.5120]

conditions=['control','depolarizing','hyperpolarizing']

def read_subject_data(data_dir, subj_id, conditions, filter=True, response_only=True):
    subj_data={}
    for condition in conditions:
        f = h5py.File(os.path.join(data_dir,'subject.%d.%s.h5' % (subj_id,condition)),'r')

        f_network_params = f['network_params']
        mu_0 = float(f_network_params.attrs['mu_0'])
        p_a = float(f_network_params.attrs['p_a'])
        f_behav = f['behavior']
        trial_rt = np.array(f_behav['trial_rt'])
        trial_resp = np.array(f_behav['trial_resp'])
        trial_correct = np.array(f_behav['trial_correct'])
        f_neur = f['neural']
        trial_inputs = np.array(f_neur['trial_inputs'])
        trial_data = []
        last_resp = float('NaN')
        for trial_idx in range(trial_rt.shape[1]):
            direction = np.where(trial_inputs[:, trial_idx] == np.max(trial_inputs[:, trial_idx]))[0][0]
            if direction == 0:
                direction = -1
            coherence = np.abs((trial_inputs[0, trial_idx] - mu_0) / (p_a * 100.0))
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
            if not response_only or not math.isnan(rt):
                trial_data.append([trial_idx, direction, coherence, correct, resp, last_resp, rt])
            last_resp = resp
        trial_data = np.array(trial_data)
        if filter:
            outliers=mdm_outliers(trial_data[:,6])
            trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]
        subj_data[condition]=trial_data

    return subj_data


def analyze_subject_accuracy_rt(subject, plot=False):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    for condition,trial_data in subject.iteritems():
        condition_coherence_accuracy[condition]={}
        condition_coherence_rt[condition]={}
        # For each trial
        for trial_idx in range(trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]
            correct=trial_data[trial_idx,3]
            rt=trial_data[trial_idx,6]

            if not coherence in condition_coherence_accuracy[condition]:
                condition_coherence_accuracy[condition][coherence]=[]
            condition_coherence_accuracy[condition][np.abs(coherence)].append(float(correct))

            if not coherence in condition_coherence_rt[condition]:
                condition_coherence_rt[condition][coherence]=[]
            condition_coherence_rt[condition][np.abs(coherence)].append(rt)

        coherences = sorted(condition_coherence_accuracy[condition].keys())
        accuracy=[]
        for coherence in coherences:
            accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
        acc_fit = FitWeibull(coherences, accuracy, guess=[0.0, 0.2], display=0)
        condition_accuracy_thresh[condition]=acc_fit.inverse(0.8)

    for stim_condition in ['depolarizing', 'hyperpolarizing']:
        condition_coherence_rt_diff[stim_condition]={}
        coherences=sorted(condition_coherence_rt[stim_condition].keys())
        for coherence in coherences:
            condition_coherence_rt_diff[stim_condition][coherence]=np.mean(condition_coherence_rt[stim_condition][coherence])-np.mean(condition_coherence_rt['control'][coherence])

    if plot:
        plot_choice_accuracy(colors, condition_coherence_accuracy)

        plot_choice_rt(colors, condition_coherence_rt)

        plot_choice_rt_diff(colors, condition_coherence_rt, plot_err=False)

    return condition_coherence_accuracy, condition_coherence_rt, condition_coherence_rt_diff, condition_accuracy_thresh


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
    for condition,trial_data in subject.iteritems():
        # Dict of coherence levels
        condition_coherence_choices['L*'][condition]={}
        condition_coherence_choices['R*'][condition]={}

        # For each trial
        for trial_idx in range(trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]*trial_data[trial_idx,1]
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]

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


def analyze_accuracy_rt(subjects, plot=True, print_stats=True):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    # For each subject
    for subject in subjects:

        subj_condition_coherence_accuracy, subj_condition_coherence_rt, subj_condition_coherence_rt_diff,subj_condition_accuracy_thresh=analyze_subject_accuracy_rt(subject, plot=False)

        for condition in conditions:
            if not condition in condition_coherence_accuracy:
                condition_coherence_accuracy[condition]={}
                condition_coherence_rt[condition]={}
                condition_accuracy_thresh[condition]=[]
            condition_accuracy_thresh[condition].append(subj_condition_accuracy_thresh[condition])

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

    return thresh_results, rtdiff_results


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
    for subject in subjects:
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
    limits = [-.5, .5]
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
    #ax.set_xlim([0.02,1])
    #ax.set_ylim([490, 680])
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
        plot_condition_choice_probability(ax, colors[condition], condition, left_coherences, left_choice_probs, right_coherences, right_choice_probs)


def plot_condition_choice_probability(ax, color, condition, left_coherences, left_choice_probs, right_coherences, right_choice_probs):
    acc_fit = FitSigmoid(left_coherences, left_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(left_coherences), max(left_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, '--%s' % color, label='L* %s' % condition)
    ax.plot(left_coherences, left_choice_probs, 'o%s' % color)
    acc_fit = FitSigmoid(right_coherences, right_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(right_coherences), max(right_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, color, label='R* %s' % condition)
    ax.plot(right_coherences, right_choice_probs, 'o%s' % color)
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% of Right Choices')


def plot_logistic_parameter_ratio(colors, condition_logistic_params):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    binwidth=.02
    for condition in conditions:
        ratio=np.array(condition_logistic_params['a2'][condition]) / np.array(condition_logistic_params['a1'][condition])
        bins=np.arange(min(ratio), max(ratio) + binwidth, binwidth)
        ax.hist(ratio, normed=1, label=condition, color=colors[condition], bins=bins, alpha=.75)
        (mu, sigma) = norm.fit(ratio)
        y = normpdf( np.arange(-.1,.2,0.001), mu, sigma)
        ax.plot(np.arange(-.1,.2,0.001), y, '%s--' % colors[condition], linewidth=2)
    ax.legend(loc='best')
    #ax.set_xlim([-.1, .2])
    #ax.set_ylim([0, 18])
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



if __name__=='__main__':
    virtual_subjects=[]
    for subj_idx in range(20):
        virtual_subjects.append(read_subject_data('/home/jbonaiuto/Projects/pySBI/data/rdmd', subj_idx,
                                               ['control','depolarizing','hyperpolarizing'], filter=True,
                                               response_only=True))
   # export_behavioral_data_long(range(20),['control','depolarizing','hyperpolarizing'],
   #     '/home/jbonaiuto/Projects/pySBI/data/rdmd','/home/jbonaiuto/Projects/pySBI/data/rdmd/behav_data_long.csv')
   #  for subject_idx in range(20):
   #      subject=virtual_subjects[subject_idx]
   #      print(subject_idx)
   #      analyze_subject_choice_hysteresis(subject, plot=True)
   #      analyze_subject_accuracy_rt(subject, plot=True)
   #      plt.show()

    analyze_choice_hysteresis(virtual_subjects)
    print('')
    analyze_accuracy_rt(virtual_subjects)
    plt.show()
    #analyze_single_subject_choice_prob(1,['control','depolarizing','hyperpolarizing'],'/home/jbonaiuto/Projects/pySBI/data/rdmd', plot=True)