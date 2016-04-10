import h5py
import os
import numpy as np
from pysbi.util.utils import mdm_outliers, FitWeibull, FitSigmoid
import matplotlib.pyplot as plt


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

        if rt > 100:
            trial_data.append([trial_idx, direction, coherence, correct, resp, last_resp, rt])
        last_resp = resp
    trial_data = np.array(trial_data)
    if filter:
        outliers=mdm_outliers(trial_data[:,6])
        trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]

    return trial_data


def analyze_choice_prob(subj_ids, conditions, data_dir):
    colors={
        'control': 'b',
        'depolarizing': 'r',
        'hyperpolarizing': 'g'
    }

    last_left_coherence_choices={}
    last_right_coherence_choices={}

    # For each subject
    for subj_id in subj_ids:
        # Dict of conditions
        subj_last_left_coherence_choices={}
        subj_last_right_coherence_choices={}
        for condition in conditions:
            # Read condition data for this subject
            trial_data = read_subject_condition_data(data_dir, subj_id, condition)
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

            if not condition in last_left_coherence_choices:
                last_left_coherence_choices[condition]={}
                last_right_coherence_choices[condition]={}

            for coherence in subj_last_left_coherence_choices[condition]:
                if not coherence in last_left_coherence_choices[condition]:
                    last_left_coherence_choices[condition][coherence]=[]
                last_left_coherence_choices[condition][coherence].append(np.mean(subj_last_left_coherence_choices[condition][coherence]))
            for coherence in subj_last_right_coherence_choices[condition]:
                if not coherence in last_right_coherence_choices[condition]:
                    last_right_coherence_choices[condition][coherence]=[]
                last_right_coherence_choices[condition][coherence].append(np.mean(subj_last_right_coherence_choices[condition][coherence]))


    fig=plt.figure()
    for condition in conditions:
        coherences=sorted(last_left_coherence_choices[condition].keys())
        choice_probs=[]
        for coherence in coherences:
            choice_probs.append(np.mean(last_left_coherence_choices[condition][coherence]))
        acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
        smoothInt = np.arange(min(coherences), max(coherences), 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        plt.plot(smoothInt, smoothResp, '--%s' % colors[condition], label='left* - %s' % condition)
        plt.plot(coherences,choice_probs,'o%s' % colors[condition])

        coherences=sorted(last_right_coherence_choices[condition].keys())
        choice_probs=[]
        for coherence in coherences:
            choice_probs.append(np.mean(last_right_coherence_choices[condition][coherence]))
        acc_fit=FitSigmoid(coherences, choice_probs, guess=[0.0, 0.2])
        smoothInt = np.arange(min(coherences), max(coherences), 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        plt.plot(smoothInt, smoothResp, colors[condition], label='right* - %s' % condition)
        plt.plot(coherences,choice_probs,'o%s' % colors[condition])
    plt.legend(loc='best')
    plt.xlabel('Coherence')
    plt.ylabel('% of Right Choices')

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