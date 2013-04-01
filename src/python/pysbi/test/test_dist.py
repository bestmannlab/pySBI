import os
from brian.units import second
import numpy as np
from pysbi.wta.analysis import  get_auc
from pysbi.wta.network import default_params, run_wta

def test_estimate_distribution(output_dir, random_seed=None):
    # number of parameters
    num_params=4
    # number of iterations
    num_iterations=1000
    # Q variance
    covar=0.05*np.identity(num_params)

    all_param_prob={}

    markov_chain=np.zeros([num_iterations,num_params])
    markov_chain_prob=np.zeros(num_iterations)
    x=np.clip(0.1*np.random.multivariate_normal(np.zeros(num_params),covar),0.0,0.1)
    x_prob=get_prob(x, output_dir)
    markov_chain[0,:]=x
    markov_chain_prob[0]=x_prob
    print('iteration 0: x=(%0.3f,%0.3f,%0.3f,%0.3f), prob=%0.3f' % (x[0],x[1],x[2],x[3],x_prob))

    for i in range(1,num_iterations):
        innovation=0.1*np.random.multivariate_normal(x,covar)
        candidate=np.clip(x+innovation,0.0,0.1)
        candidate_key='%0.3f:%0.3f:%0.3f:%0.3f' % (candidate[0],candidate[1],candidate[2],candidate[3])
        if candidate_key in all_param_prob:
            candidate_prob=all_param_prob[candidate_key]
        else:
            candidate_prob=get_prob(candidate, output_dir)
            all_param_prob[candidate_key]=candidate_prob

        transition_prob=np.min([1,candidate_prob/x_prob])
        if np.random.random()<transition_prob:
            x=candidate
            x_prob=candidate_prob
        print('iteration %d: x=(%0.3f,%0.3f,%0.3f,%0.3f), prob=%0.3f' % (i,x[0],x[1],x[2],x[3],x_prob))
        markov_chain[i,:]=x
        markov_chain_prob[i]=x_prob
    return markov_chain,markov_chain_prob

def get_prob(x, output_dir):
    num_groups=2
    trial_duration=1*second
    input_sum=40.0
    num_trials=5
    num_extra_trials=10

    wta_params=default_params()
    wta_params.p_b_e=0.1
    wta_params.p_x_e=0.05
    wta_params.p_e_e=x[0]
    wta_params.p_e_i=x[1]
    wta_params.p_i_i=x[2]
    wta_params.p_i_e=x[3]

    file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f' % \
              (num_groups, trial_duration, wta_params.p_b_e, wta_params.p_x_e, wta_params.p_e_e, wta_params.p_e_i,
               wta_params.p_i_i, wta_params.p_i_e)
    file_prefix=os.path.join(output_dir,file_desc)

    num_example_trials=[0,0]
    for trial in range(num_trials):
        inputs=np.zeros(2)
        inputs[0]=np.random.random()*input_sum
        inputs[1]=input_sum-inputs[0]

        if inputs[0]>inputs[1]:
            num_example_trials[0]+=1
        else:
            num_example_trials[1]+=1

        if trial==num_trials-1:
            if num_example_trials[0]==0:
                inputs[0]=input_sum*0.5+np.random.random()*input_sum*0.5
                inputs[1]=input_sum-inputs[0]
                num_example_trials[0]+=1
            elif num_example_trials[1]==0:
                inputs[1]=input_sum*0.5+np.random.random()*input_sum*0.5
                inputs[0]=input_sum-inputs[1]
                num_example_trials[1]+=1

        output_file='%s.trial.%d.h5' % (file_prefix,trial)

        run_wta(wta_params, num_groups, inputs, trial_duration, output_file=output_file, record_lfp=False,
            record_voxel=False, record_neuron_state=False, record_spikes=False, record_firing_rate=True,
            record_inputs=True, single_inh_pop=False)

    auc=get_auc(file_prefix, num_trials, num_extra_trials, num_groups)
    return auc

