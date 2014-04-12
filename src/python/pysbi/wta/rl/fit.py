import numpy as np
import os
import scipy.io
from scipy.optimize import leastsq, fmin_ncg, fmin
import h5py
import matplotlib.pyplot as plt

"""
    stim_cond: 1=lat, 2=med, 3=nostim1, 4=nostim2
    """
LAT=0
MED=1
NOSTIM1=2
NOSTIM2=3

stim_order=np.array([
    [6,1,2,4],
    [6,1,2,4],
    [6,1,2,4],
    [4,1,2,5],
    [4,1,2,5],
    [4,1,2,5],
    [6,3,1,4],
    [6,3,1,4],
    [6,3,1,4],
    [4,3,1,5],
    [4,3,1,5],
    [4,3,1,5],
    [3,4,1,5],
    [3,4,1,5],
    [3,4,1,5],
    [1,6,2,4],
    [1,6,2,4],
    [1,6,2,4],
    [3,6,1,4],
    [3,6,1,4],
    [3,6,1,4],
    [1,4,2,5],
    [1,4,2,5],
    [1,4,2,5]
])

def rescorla_td_prediction(walkset, choices, learning_rate):
    x_pre=np.zeros((2,walkset.shape[0]))
    x_pre[:,0]=[0.5,0.5]

    for t in range(len(choices)-1):
        x_pre[:,t+1]=x_pre[:,t]
        x_pre[choices[t],t+1]=(1.0-learning_rate)*x_pre[choices[t],t]+learning_rate*walkset[t]

    return x_pre

def energy_learn_rewards(x, mags, rewards, choice):
    alpha=x[0]
    beta=x[1]
    # don't allow negative estimates
    if alpha<=0 or beta<=0 or alpha>1:
        return 10000000

    # get modelled probs using Rescorla-Wagner model, then vals, then energy
    model_probs = rescorla_td_prediction(rewards,choice,alpha);
    model_vals = model_probs*mags;
    ch_valdiffs=model_vals[0,:]-model_vals[1,:]
    ch_valdiffs[np.where(choice==1)]=-ch_valdiffs[np.where(choice==1)] # swap if RH chosen
    energy=-np.sum(np.log(1/(1+np.exp(-beta*ch_valdiffs)))) # maximise log likelihood (softmax function)
    return energy

def fit_subject_behavior(mat_file):
    mat = scipy.io.loadmat(mat_file)
    prob_idx=-1
    mags_idx=-1
    rew_idx=-1
    resp_idx=-1
    for idx,(dtype,o) in enumerate(mat['store']['dat'][0][0].dtype.descr):
        if dtype=='RESP':
            resp_idx=idx
        elif dtype=='probswalk':
            prob_idx=idx
        elif dtype=='mags':
            mags_idx=idx
        elif dtype=='outcrec':
            rew_idx=idx
    prob_walk=mat['store']['dat'][0][0][0][0][prob_idx]
    mags=mat['store']['dat'][0][0][0][0][mags_idx]
    rew=np.squeeze(mat['store']['dat'][0][0][0][0][rew_idx])
    resp=np.squeeze(mat['store']['dat'][0][0][0][0][resp_idx])

    prob_walk=prob_walk.astype(np.float32, copy=False)
    mags=mags.astype(np.float32, copy=False)
    rew=rew.astype(np.float32, copy=False)
    resp=resp.astype(np.float32, copy=False)

    rh_button=7
    lh_button=22
    choice=np.zeros(resp.shape)
    choice[np.where(resp==rh_button)]=0
    choice[np.where(resp==lh_button)]=1

    missed_idx=np.where(np.isnan(resp)==True)[0]
    choice=np.delete(choice, missed_idx)
    rew=np.delete(rew, missed_idx)
    mags=np.delete(mags, missed_idx, axis=1)
    prob_walk=np.delete(prob_walk, missed_idx, axis=1)

    mags /= 100.0
    return fit_behavior(prob_walk, mags, rew, choice)

def test_fit_behavior(mat_file):
    alpha=0.2
    beta=10.0

    mat = scipy.io.loadmat(mat_file)
    prob_walk=mat['store']['dat'][0][0][0][0][13]
    mags=mat['store']['dat'][0][0][0][0][15]

    prob_walk=prob_walk.astype(np.float32, copy=False)
    mags=mags.astype(np.float32, copy=False)
    mags /= 100.0

    model_choices=np.zeros(prob_walk.shape[1])
    model_rew=np.zeros(prob_walk.shape[1])
    model_vals=np.zeros(prob_walk.shape)
    model_probs=np.zeros(prob_walk.shape)

    exp_rew=np.array([0.5,0.5])
    for i in range(prob_walk.shape[1]):
        model_vals[:,i]=exp_rew
        ev=model_vals[:,i]*mags[:,i]
        model_probs[0,i]=1.0/(1.0+np.exp(-beta*(ev[0]-ev[1])))
        model_probs[1,i]=1.0/(1.0+np.exp(-beta*(ev[1]-ev[0])))
        model_choices[i]=np.random.choice([0,1],p=model_probs[:,i])
        if np.random.rand()<=prob_walk[model_choices[i],i]:
            model_rew[i]=1.0
        exp_rew[model_choices[i]]=(1.0-alpha)*exp_rew[model_choices[i]]+alpha*model_rew[i]
    return fit_behavior(prob_walk,mags,model_rew,model_choices)


def fit_behavior(prob_walk, mags, rew, choice, plot=False):
    n_fits=100
    all_param_estimates=np.zeros((2,n_fits))
    all_energy=np.zeros(n_fits)
    for i in range(n_fits):
        params=np.random.rand(2)
        all_param_estimates[:,i]=fmin(energy_learn_rewards, params, args=(mags, rew, choice), disp=False)
        all_energy[i]=energy_learn_rewards(all_param_estimates[:,i],mags,rew,choice)

    min_idx=np.where(all_energy==np.min(all_energy))[0][0]
    param_ests=all_param_estimates[:,min_idx]

    fit_vals=rescorla_td_prediction(rew, choice, param_ests[0])
    fit_probs=np.zeros(fit_vals.shape)
    ev=fit_vals*mags
    fit_probs[0,:]=1.0/(1.0+np.exp(-param_ests[1]*(ev[0,:]-ev[1,:])))
    fit_probs[1,:]=1.0/(1.0+np.exp(-param_ests[1]*(ev[1,:]-ev[0,:])))
    prop_correct_vec=np.zeros(100)
    for i in range(100):
        fit_choices=np.zeros(choice.shape)
        for t in range(len(choice)):
            fit_choices[t]=np.random.choice([0,1], p=fit_probs[:,t])
        prop_correct_vec[i]=float(len(np.where(fit_choices==choice)[0]))/float(len(choice))
    prop_correct=np.mean(prop_correct_vec)

    if plot:
        plt.figure()
        ax=plt.subplot(3,1,1)
        plt.title('Real probs')
        plt.plot(np.transpose(prob_walk))

        ax=plt.subplot(3,1,2)
        plt.title('Modelled probs + rewards')
        plt.plot(np.transpose(fit_probs))
        plt.plot(rew,'o')

        ax=plt.subplot(3,1,3)
        plt.title('Modelled vals - chosen vs unchosen')
        ch_val=np.zeros(prob_walk.shape[1])
        unch_val=np.zeros(prob_walk.shape[1])
        for t in range(prob_walk.shape[1]):
            ch_val[t] = fit_vals[choice[t],t]
            unch_val[t] = fit_vals[1-choice[t],t];
        plt.plot(ch_val)
        plt.plot(unch_val,'r')
        plt.show()

    print('Learning rate: %.4f' % param_ests[0])
    print('Beta: %.4f' % param_ests[1])
    print('Proportion of correctly predicted choices: %.3f' % prop_correct)

    return param_ests,prop_correct

def fit_subjects(data_dir, num_subjects, output_file):
    beta_stim_vals=[]
    alpha_stim_vals=[]
    beta_control_vals=[]
    alpha_control_vals=[]
    for i in range(num_subjects):
        subj_id=i+1
        subj_stim_session_number=stim_order[i,LAT]
        stim_file_name=os.path.join(data_dir,'value%d_s%d_t2.mat' % (subj_id,subj_stim_session_number))
        subj_control_session_number=stim_order[i,NOSTIM1]
        control_file_name=os.path.join(data_dir,'value%d_s%d_t2.mat' % (subj_id,subj_control_session_number))
        if os.path.exists(stim_file_name) and os.path.exists(control_file_name):
            print('processing subject %d' % subj_id)
            control_param_ests,control_prop_correct=fit_subject_behavior(control_file_name)
            alpha_control_vals.append(control_param_ests[0])
            beta_control_vals.append(control_param_ests[1])
            stim_param_ests,stim_prop_correct=fit_subject_behavior(stim_file_name)
            alpha_stim_vals.append(stim_param_ests[0])
            beta_stim_vals.append(stim_param_ests[1])
            #alpha_diff_vals.append(stim_param_ests[0]-control_param_ests[0])
            #beta_diff_vals.append(stim_param_ests[1]-control_param_ests[1])

    alpha_control_vals=np.array(alpha_control_vals)
    beta_control_vals=np.array(beta_control_vals)
    alpha_stim_vals=np.array(alpha_stim_vals)
    beta_stim_vals=np.array(beta_stim_vals)

    alpha_diff_vals=alpha_stim_vals-alpha_control_vals
    beta_diff_vals=beta_stim_vals-beta_control_vals

    f = h5py.File(output_file, 'w')
    control_group=f.create_group('control')
    control_group['alpha']=alpha_control_vals
    control_group['beta']=beta_control_vals
    stim_group=f.create_group('stim')
    stim_group['alpha']=alpha_stim_vals
    stim_group['beta']=beta_stim_vals
    f.close()

    fig=plt.figure()
    alpha_hist,alpha_bins=np.histogram(np.array(alpha_diff_vals), bins=10, range=(-1.0,1.0), density=True)
    bin_width=alpha_bins[1]-alpha_bins[0]
    plt.bar(alpha_bins[:-1], alpha_hist, width=bin_width)
    plt.xlim(-1.0,1.0)
    plt.xlabel('Change in alpha')
    plt.ylabel('Proportion of Subjects')

    fig=plt.figure()
    beta_hist,beta_bins=np.histogram(np.array(beta_diff_vals), bins=10, range=(-10.0,10.0), density=True)
    bin_width=beta_bins[1]-beta_bins[0]
    plt.bar(beta_bins[:-1], beta_hist, width=bin_width)
    plt.xlim(-10.0,10.0)
    plt.xlabel('Change in beta')
    plt.ylabel('Proportion of Subjects')

    plt.show()

if __name__=='__main__':
    #fit_subject_behavior('../../data/rerw/subjects/value1_s1_t2.mat')
    #test_fit_behavior('../../data/rerw/subjects/value1_s1_t2.mat')
    fit_subjects('../../data/rerw/subjects/',24,'../../data/rerw/subjects/fitted_behavioral_params.h5')