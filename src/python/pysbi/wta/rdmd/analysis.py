import h5py
import os
import numpy as np
from scipy.stats import wilcoxon
from pysbi.util.utils import mdm_outliers, FitSigmoid
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

colors={
    'Control': 'b',
    'Anode': 'r',
    'Cathode': 'g'
}


def read_subject_condition_data(data_dir, subj_id, condition, filter=True):
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
        correct = int(trial_correct[0, trial_idx])
        resp = int(trial_resp[0, trial_idx])
        if resp == 0:
            resp = -1
        rt = trial_rt[0, trial_idx]

        #if rt > 100:
        trial_data.append([trial_idx, direction, coherence, correct, resp, last_resp, rt])
        last_resp = resp
    trial_data = np.array(trial_data)
    if filter:
        outliers=mdm_outliers(trial_data[:,6])
        trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]

    return trial_data


def analyze_single_subject_choice_prob(subj_id, conditions, data_dir, plot=False):
# Dict of conditions
    subj_last_left_coherence_choices={}
    subj_last_right_coherence_choices={}
    subj_last_left_sigmoid_offsets={}
    subj_last_right_sigmoid_offsets={}
    for condition in conditions:
        # Read condition data for this subject
        trial_data = read_subject_condition_data(data_dir, subj_id, condition, filter=False)
        # Dict of coherence levels
        subj_last_left_coherence_choices[condition]={}
        subj_last_right_coherence_choices[condition]={}
        # For each trial
        for trial_idx in range(trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]*trial_data[trial_idx,1]
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]
            if last_resp<0:
                # List of rigtward choices (0=left, 1=right)
                if not coherence in subj_last_left_coherence_choices[condition]:
                    subj_last_left_coherence_choices[condition][coherence]=[]
                    # Append 0 to list if left (-1) or 1 if right
                subj_last_left_coherence_choices[condition][coherence].append(np.max([0,resp]))
            elif last_resp>0:
                # List of rigtward choices (0=left, 1=right)
                if not coherence in subj_last_right_coherence_choices[condition]:
                    subj_last_right_coherence_choices[condition][coherence]=[]
                    # Append 0 to list if left (-1) or 1 if right
                subj_last_right_coherence_choices[condition][coherence].append(np.max([0,resp]))

        coherences=sorted(subj_last_left_coherence_choices[condition].keys())
        choice_probs=[]
        for coherence in coherences:
            choice_probs.append(np.mean(subj_last_left_coherence_choices[condition][coherence]))
        acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
        subj_last_left_sigmoid_offsets[condition]=acc_fit.inverse(0.5)

        coherences=sorted(subj_last_right_coherence_choices[condition].keys())
        choice_probs=[]
        for coherence in coherences:
            choice_probs.append(np.mean(subj_last_right_coherence_choices[condition][coherence]))
        acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
        subj_last_right_sigmoid_offsets[condition]=acc_fit.inverse(0.5)

    if plot:
        fig=plt.figure()
        for condition in conditions:
            coherences=sorted(subj_last_left_coherence_choices[condition].keys())
            choice_probs=[]
            for coherence in coherences:
                choice_probs.append(np.mean(subj_last_left_coherence_choices[condition][coherence]))
            acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
            smoothInt = np.arange(min(coherences), max(coherences), 0.001)
            smoothResp = acc_fit.eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '--%s' % colors[condition], label='left* - %s' % condition)
            plt.plot(coherences,choice_probs,'o%s' % colors[condition])

            coherences=sorted(subj_last_right_coherence_choices[condition].keys())
            choice_probs=[]
            for coherence in coherences:
                choice_probs.append(np.mean(subj_last_right_coherence_choices[condition][coherence]))
            acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
            smoothInt = np.arange(min(coherences), max(coherences), 0.001)
            smoothResp = acc_fit.eval(smoothInt)
            plt.plot(smoothInt, smoothResp, colors[condition], label='right* - %s' % condition)
            plt.plot(coherences,choice_probs,'o%s' % colors[condition])
        plt.legend(loc='best')
        plt.xlabel('Coherence')
        plt.ylabel('% of Right Choices')

        plt.show()
    return subj_last_left_coherence_choices, subj_last_right_coherence_choices, subj_last_left_sigmoid_offsets, subj_last_right_sigmoid_offsets


def analyze_choice_prob(subj_ids, conditions, data_dir):

    last_left_coherence_choices={}
    last_right_coherence_choices={}

    last_left_sigmoid_offsets={}
    last_right_sigmoid_offsets={}

    # For each subject
    for subj_id in subj_ids:
        # Dict of conditions
        subj_last_left_coherence_choices, subj_last_right_coherence_choices, subj_left_sigmoid_offsets, subj_right_sigmoid_offsets=analyze_single_subject_choice_prob(subj_id,
            conditions, data_dir)
        for condition in conditions:
            if not condition in last_left_coherence_choices:
                last_left_coherence_choices[condition]={}
                last_right_coherence_choices[condition]={}
                last_left_sigmoid_offsets[condition]=[]
                last_right_sigmoid_offsets[condition]=[]

            for coherence in subj_last_left_coherence_choices[condition]:
                if not coherence in last_left_coherence_choices[condition]:
                    last_left_coherence_choices[condition][coherence]=[]
                last_left_coherence_choices[condition][coherence].append(np.mean(subj_last_left_coherence_choices[condition][coherence]))
            for coherence in subj_last_right_coherence_choices[condition]:
                if not coherence in last_right_coherence_choices[condition]:
                    last_right_coherence_choices[condition][coherence]=[]
                last_right_coherence_choices[condition][coherence].append(np.mean(subj_last_right_coherence_choices[condition][coherence]))


    fig=plt.figure()
    left_offsets=[]
    right_offsets=[]
    for condition in conditions:
        left_offsets.extend(last_left_sigmoid_offsets[condition])
        right_offsets.extend(last_right_sigmoid_offsets[condition])
    ax=fig.add_subplot(1,2,1)
    ax.plot(left_offsets,right_offsets,'o')
    #ax.plot([-.25,.25],[-.25,.25],'--k')
    #ax.set_xlim([-.25, .25])
    #ax.set_ylim([-.25, .25])
    ax.set_xlabel('Indifference Point for L* Trials')
    ax.set_ylabel('Indifference Point for R* Trials')

    ax=fig.add_subplot(1,2,2)
    for condition in conditions:
        ax.plot(last_left_sigmoid_offsets[condition],last_right_sigmoid_offsets[condition],'o%s' % colors[condition], label=condition)
        #ax.plot([-.25,.25],[-.25,.25],'--k')
        #ax.set_xlim([-.25, .25])
        #ax.set_ylim([-.25, .25])
        ax.set_xlabel('Indifference Point for L* Trials')
        ax.set_ylabel('Indifference Point for R* Trials')
        ax.legend(loc='best')

    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    coherences=sorted(last_left_coherence_choices["Control"].keys())
    choice_probs=[]
    for coherence in coherences:
        last_left_choices=[]
        for condition in colors:
            last_left_choices.extend(last_left_coherence_choices[condition][coherence])
        choice_probs.append(np.mean(last_left_choices))
    acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
    smoothInt = np.arange(min(coherences), max(coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, '--b', label='left*')
    ax.plot(coherences,choice_probs,'ob')

    choice_probs=[]
    for coherence in coherences:
        last_right_choices=[]
        for condition in colors:
            last_right_choices.extend(last_right_coherence_choices[condition][coherence])
        choice_probs.append(np.mean(last_right_choices))
    acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
    smoothInt = np.arange(min(coherences), max(coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, 'b', label='right*')
    ax.plot(coherences,choice_probs,'ob')

    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% of Right Choices')

    ax=fig.add_subplot(1,2,2)
    for condition in conditions:
        coherences=sorted(last_left_coherence_choices[condition].keys())
        choice_probs=[]
        for coherence in coherences:
            choice_probs.append(np.mean(last_left_coherence_choices[condition][coherence]))
        acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
        smoothInt = np.arange(min(coherences), max(coherences), 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        ax.plot(smoothInt, smoothResp, '--%s' % colors[condition], label='left* - %s' % condition)
        ax.plot(coherences,choice_probs,'o%s' % colors[condition])

        coherences=sorted(last_right_coherence_choices[condition].keys())
        choice_probs=[]
        for coherence in coherences:
            choice_probs.append(np.mean(last_right_coherence_choices[condition][coherence]))
        acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
        smoothInt = np.arange(min(coherences), max(coherences), 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        ax.plot(smoothInt, smoothResp, colors[condition], label='right* - %s' % condition)
        ax.plot(coherences,choice_probs,'o%s' % colors[condition])
        ax.legend(loc='best')
        ax.set_xlabel('Coherence')
        ax.set_ylabel('% of Right Choices')

    (x,p)=wilcoxon(np.array(last_left_sigmoid_offsets['Control'])-np.array(last_right_sigmoid_offsets['Control']),
        np.array(last_left_sigmoid_offsets['Cathode'])-np.array(last_right_sigmoid_offsets['Cathode']))
    print('Cathode, x=%.4f, p=%.6f' % (x,p))
    (x,p)=wilcoxon(np.array(last_left_sigmoid_offsets['Control'])-np.array(last_right_sigmoid_offsets['Control']),
        np.array(last_left_sigmoid_offsets['Anode'])-np.array(last_right_sigmoid_offsets['Anode']))
    print('Anode, x=%.4f, p=%.6f' % (x,p))

    plt.show()


def analyze_logistic(subj_ids, conditions, data_dir):
    a1={
        'ShamPreAnode': [],
        'Anode': [],
        'ShamPreCathode': [],
        'Cathode': [],
        }
    a2={
        'ShamPreAnode': [],
        'Anode': [],
        'ShamPreCathode': [],
        'Cathode': [],
        }
    colors={
        'ShamPreAnode': 'b',
        'Anode': 'r',
        'ShamPreCathode': 'b',
        'Cathode': 'g'
    }
    for subj_id in subj_ids:
        for condition in conditions:
            # Read condition data for this subject
            trial_data = read_subject_condition_data(data_dir, subj_id, condition, filter=False)

            data=pd.DataFrame({
                'resp': np.clip(trial_data[1:,4],0,1),
                'coh': trial_data[1:,2]*trial_data[1:,1],
                'last_resp': trial_data[1:,5]
            })
            data['intercept']=1.0

            logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
            result = logit.fit()
            a1[condition].append(result.params['coh'])
            a2[condition].append(result.params['last_resp'])

    fig=plt.figure()
    for cond_idx,stim_condition in enumerate(['Anode','Cathode']):
        (x,p)=wilcoxon(np.array(a2['ShamPre%s' % stim_condition])/np.array(a1['ShamPre%s' % stim_condition]),
            np.array(a2[stim_condition])/np.array(a1[stim_condition]))
        print('%s, x=%.4f, p=%.4f' % (stim_condition,x,p))
        ax=fig.add_subplot(1,2,cond_idx)
        for condition in ['ShamPre%s' % stim_condition,stim_condition]:
            ax.hist(np.array(a2[condition])/np.array(a1[condition]), label=condition, color=colors[condition], bins=10)
        ax.set_xlim([-.2, .2])
        ax.set_ylim([0, 9])
        ax.legend(loc='best')
        ax.set_xlabel('a2/a1')
        ax.set_ylabel('# subjects')
    plt.show()

def export_behavioral_data_long(subj_ids, conditions, data_dir, output_filename):
    out_file=open(output_filename,'w')
    out_file.write('Subject,Condition,Trial,Direction,Coherence,Resp,LastResp,Correct,RT\n')
    for subj_id in subj_ids:
        for condition in conditions:

            trial_data = read_subject_condition_data(data_dir, subj_id, condition)
            for i in range(trial_data.shape[0]):
                out_file.write('%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (subj_id, condition, int(trial_data[i,0]),
                                                                             trial_data[i,1], trial_data[i,2],
                                                                             trial_data[i,4], trial_data[i,5],
                                                                             trial_data[i,3], trial_data[i,6]))



if __name__=='__main__':
   # export_behavioral_data_long(range(20),['control','depolarizing','hyperpolarizing'],
   #     '/home/jbonaiuto/Projects/pySBI/data/rdmd','/home/jbonaiuto/Projects/pySBI/data/rdmd/behav_data_long.csv')
    analyze_choice_prob(range(20),['control','depolarizing','hyperpolarizing'],
        '/home/jbonaiuto/Projects/pySBI/data/rdmd')
    analyze_logistic(range(20),['control','depolarizing','hyperpolarizing'],
        '/home/jbonaiuto/Projects/pySBI/data/rdmd')
    #analyze_single_subject_choice_prob(1,['control','depolarizing','hyperpolarizing'],'/home/jbonaiuto/Projects/pySBI/data/rdmd', plot=True)