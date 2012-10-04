import os
from brian.stdunits import Hz, nS
from brian.units import second
from scikits.learn.linear_model.base import LinearRegression
from pysbi.wta import default_params, run_wta
import numpy as np
import matplotlib.pylab as plt

def test_wta(p_e_e, p_e_i, p_i_i, p_i_e, inputs, single_inh_pop=False, muscimol_amount=0*nS, injection_site=0):
    wta_params=default_params()
    wta_params.p_b_e=0.1
    wta_params.p_x_e=0.05
    wta_params.p_e_e=p_e_e
    wta_params.p_e_i=p_e_i
    wta_params.p_i_i=p_i_i
    wta_params.p_i_e=p_i_e

    input_freq=np.zeros(2)
    for i in range(2):
        input_freq[i]=float(inputs[i])*Hz

    run_wta(wta_params, 2, input_freq, 1.0*second, record_lfp=False, record_neuron_state=True, plot_output=True,
        single_inh_pop=single_inh_pop, muscimol_amount=muscimol_amount, injection_site=injection_site)

def test_contrast(p_e_e, p_e_i, p_i_i, p_i_e, num_trials, data_path, muscimol_amount=0*nS, injection_site=0):
    num_groups=2
    trial_duration=1.0*second

    wta_params=default_params()
    wta_params.p_b_e=0.1
    wta_params.p_x_e=0.05
    wta_params.p_e_e=p_e_e
    wta_params.p_e_i=p_e_i
    wta_params.p_i_i=p_i_i
    wta_params.p_i_e=p_i_e
    input_sum=40.0

    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
    trial_contrast=np.zeros([len(contrast_range)*num_trials,1])
    trial_max_bold=np.zeros(len(contrast_range)*num_trials)
    trial_max_exc_bold=np.zeros(len(contrast_range)*num_trials)
    for i,contrast in enumerate(contrast_range):
        print('Testing contrast %0.4f' % contrast)
        inputs=np.zeros(2)
        inputs[0]=(input_sum*(contrast+1.0)/2.0)
        inputs[1]=input_sum-inputs[0]

        for j in range(num_trials):
            print('Trial %d' % j)
            trial_contrast[i*num_trials+j]=contrast
            np.random.shuffle(inputs)

            input_freq=np.zeros(num_groups)
            for k in range(num_groups):
                input_freq[k]=float(inputs[k])*Hz

            file='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.contrast.%0.4f.trial.%d.h5' %\
                 (num_groups, trial_duration, wta_params.p_b_e, wta_params.p_x_e, wta_params.p_e_e, wta_params.p_e_i,
                  wta_params.p_i_i, wta_params.p_i_e, contrast, j)

            wta_monitor=run_wta(wta_params, num_groups, input_freq, trial_duration, record_neuron_state=True,
                output_file=os.path.join(data_path,file), muscimol_amount=muscimol_amount, injection_site=injection_site)

            trial_max_bold[i*num_trials+j]=np.max(wta_monitor.voxel_monitor['y'].values)
            trial_max_exc_bold[i*num_trials+j]=np.max(wta_monitor.voxel_exc_monitor['y'].values)

    x_min=np.min(contrast_range)
    x_max=np.max(contrast_range)

    fig=plt.figure()
    clf=LinearRegression()
    clf.fit(trial_contrast,trial_max_bold)
    a=clf.coef_[0]
    b=clf.intercept_

    plt.plot(trial_contrast, trial_max_bold, 'x')
    plt.plot([x_min,x_max],[a*x_min+b,a*x_max+b],'--')
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD')
    plt.show()

    fig=plt.figure()
    clf=LinearRegression()
    clf.fit(trial_contrast,trial_max_exc_bold)
    a=clf.coef_[0]
    b=clf.intercept_

    plt.plot(trial_contrast, trial_max_exc_bold, 'o')
    plt.plot([x_min,x_max],[a*x_min+b,a*x_max+b],'--')
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD (exc only)')
    plt.show()


def test_contrast_lesion(p_e_e, p_e_i, p_i_i, p_i_e, num_trials, data_path, muscimol_amount=0*nS, injection_site=0):
    num_groups=2
    trial_duration=1.0*second

    wta_params=default_params()
    wta_params.p_b_e=0.1
    wta_params.p_x_e=0.05
    wta_params.p_e_e=p_e_e
    wta_params.p_e_i=p_e_i
    wta_params.p_i_i=p_i_i
    wta_params.p_i_e=p_i_e
    input_sum=40.0

    contrast_range=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
    trial_contrast=np.zeros([len(contrast_range)*num_trials,1])
    trial_max_bold=np.zeros(len(contrast_range)*num_trials)
    trial_max_exc_bold=np.zeros(len(contrast_range)*num_trials)
    for i,contrast in enumerate(contrast_range):
        print('Testing contrast %0.4f' % contrast)
        inputs=np.zeros(2)
        inputs[0]=(input_sum*(contrast+1.0)/2.0)
        inputs[1]=input_sum-inputs[0]

        for j in range(num_trials):
            print('Trial %d' % j)
            trial_contrast[i*num_trials+j]=contrast
            np.random.shuffle(inputs)

            input_freq=np.zeros(num_groups)
            for k in range(num_groups):
                input_freq[k]=float(inputs[k])*Hz

            file='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.contrast.%0.4f.trial.%d.h5' %\
                 (num_groups, trial_duration, wta_params.p_b_e, wta_params.p_x_e, wta_params.p_e_e, wta_params.p_e_i,
                  wta_params.p_i_i, wta_params.p_i_e, contrast, j)

            out_file=None
            if not data_path is None:
                out_file=os.path.join(data_path,file)
            wta_monitor=run_wta(wta_params, num_groups, input_freq, trial_duration, record_neuron_state=True,
                output_file=out_file)

            trial_max_bold[i*num_trials+j]=np.max(wta_monitor.voxel_monitor['y'].values)
            trial_max_exc_bold[i*num_trials+j]=np.max(wta_monitor.voxel_exc_monitor['y'].values)

    lesioned_trial_max_bold=np.zeros(len(contrast_range)*num_trials)
    lesioned_trial_max_exc_bold=np.zeros(len(contrast_range)*num_trials)
    for i,contrast in enumerate(contrast_range):
        print('Testing contrast %0.4f' % contrast)
        inputs=np.zeros(2)
        inputs[0]=(input_sum*(contrast+1.0)/2.0)
        inputs[1]=input_sum-inputs[0]

        for j in range(num_trials):
            print('Trial %d' % j)
            trial_contrast[i*num_trials+j]=contrast
            np.random.shuffle(inputs)

            input_freq=np.zeros(num_groups)
            for k in range(num_groups):
                input_freq[k]=float(inputs[k])*Hz

            file='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.contrast.%0.4f.trial.%d.h5' %\
                 (num_groups, trial_duration, wta_params.p_b_e, wta_params.p_x_e, wta_params.p_e_e, wta_params.p_e_i,
                  wta_params.p_i_i, wta_params.p_i_e, contrast, j)

            out_file=None
            if not data_path is None:
                out_file=os.path.join(data_path,file)
            wta_monitor=run_wta(wta_params, num_groups, input_freq, trial_duration, record_neuron_state=True,
                output_file=out_file, muscimol_amount=muscimol_amount, injection_site=injection_site)

            lesioned_trial_max_bold[i*num_trials+j]=np.max(wta_monitor.voxel_monitor['y'].values)
            lesioned_trial_max_exc_bold[i*num_trials+j]=np.max(wta_monitor.voxel_exc_monitor['y'].values)

    x_min=np.min(contrast_range)
    x_max=np.max(contrast_range)

    fig=plt.figure()
    clf=LinearRegression()
    clf.fit(trial_contrast,trial_max_bold)
    a=clf.coef_[0]
    b=clf.intercept_

    clf_l=LinearRegression()
    clf_l.fit(trial_contrast,lesioned_trial_max_bold)
    a_l=clf_l.coef_[0]
    b_l=clf_l.intercept_

    plt.plot(trial_contrast, trial_max_bold, 'xb')
    plt.plot(trial_contrast, lesioned_trial_max_bold, 'xr')
    plt.plot([x_min,x_max],[a*x_min+b,a*x_max+b],'--b',label='Control')
    plt.plot([x_min,x_max],[a_l*x_min+b_l,a_l*x_max+b_l],'--r',label='Lesioned')
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD')
    plt.legend()
    plt.show()

    fig=plt.figure()
    clf=LinearRegression()
    clf.fit(trial_contrast,trial_max_exc_bold)
    a=clf.coef_[0]
    b=clf.intercept_

    clf_l=LinearRegression()
    clf_l.fit(trial_contrast,lesioned_trial_max_exc_bold)
    a_l=clf_l.coef_[0]
    b_l=clf_l.intercept_

    plt.plot(trial_contrast, trial_max_exc_bold, 'ob')
    plt.plot(trial_contrast, lesioned_trial_max_exc_bold, 'or')
    plt.plot([x_min,x_max],[a*x_min+b,a*x_max+b],'--b',label='Control')
    plt.plot([x_min,x_max],[a_l*x_min+b_l,a_l*x_max+b_l],'--r',label='Lesioned')
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD (exc only)')
    plt.legend()
    plt.show()