import numpy as np
import scipy.io
from scipy.optimize import leastsq, fmin_ncg, fmin
import matplotlib.pyplot as plt

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
    prob_walk=mat['store']['dat'][0][0][0][0][13]
    mags=mat['store']['dat'][0][0][0][0][15]
    rew=np.squeeze(mat['store']['dat'][0][0][0][0][16])
    resp=np.squeeze(mat['store']['dat'][0][0][0][0][10])

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

    mags /= 100.0
    return fit_behavior(prob_walk, mags, rew, choice)

def fit_behavior(prob_walk, mags, rew, choice):
    n_fits=100
    all_param_estimates=np.zeros((2,n_fits))
    all_energy=np.zeros(n_fits)
    for i in range(n_fits):
        params=np.random.rand(2)
        all_param_estimates[:,i]=fmin(energy_learn_rewards, params, args=(mags, rew, choice))
        all_energy[i]=energy_learn_rewards(all_param_estimates[:,i],mags,rew,choice)

    min_idx=np.where(all_energy==np.min(all_energy))[0][0]
    param_ests=all_param_estimates[:,min_idx]

    fit_vals=rescorla_td_prediction(rew, choice, param_ests[0])
    fit_probs=np.zeros(fit_vals.shape)
    ev=fit_vals*mags
    fit_probs[0,:]=1/(1+np.exp(-param_ests[1]*(ev[0,:]-ev[1,:])))
    fit_probs[1,:]=1/(1+np.exp(-param_ests[1]*(ev[1,:]-ev[0,:])))
    prop_correct_vec=np.zeros(100)
    for i in range(100):
        fit_choices=np.zeros(choice.shape)
        for t in range(len(choice)):
            fit_choices[t]=np.random.choice([0,1], p=fit_probs[:,t])
        prop_correct_vec[i]=float(len(np.where(fit_choices==choice)))/float(len(choice))
    prop_correct=np.mean(prop_correct_vec)

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

if __name__=='__main__':
    fit_subject_behavior('../../data/rerw/subjects/value1_s1_t2.mat')