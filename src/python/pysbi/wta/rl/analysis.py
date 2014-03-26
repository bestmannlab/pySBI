import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from scikits.learn.linear_model import LinearRegression

def analyze_background_beta(data_dir, file_prefix, background_range):
    beta_vals=np.zeros((len(background_range),1))
    alpha_vals=np.zeros((len(background_range),1))
    background_vals=np.zeros((len(background_range),1))
    for idx,background in enumerate(background_range):
        file_name=os.path.join(data_dir,file_prefix % background)
        f = h5py.File(file_name)
        alpha=float(f.attrs['est_alpha'])
        beta=float(f.attrs['est_beta'])
        beta_vals[idx]=beta
        alpha_vals[idx]=alpha
        background_vals[idx]=background

    clf = LinearRegression()
    clf.fit(background_vals, alpha_vals)
    alpha_a = clf.coef_[0]
    alpha_b = clf.intercept_
    alpha_r_sqr=clf.score(background_vals, alpha_vals)

    clf = LinearRegression()
    clf.fit(background_vals, beta_vals)
    beta_a = clf.coef_[0]
    beta_b = clf.intercept_
    beta_r_sqr=clf.score(background_vals, beta_vals)

    plt.figure()
    plt.plot(background_range,alpha_vals,'o')
    plt.plot([background_range[0], background_range[-1]], [alpha_a * background_range[0] + alpha_b, alpha_a * background_range[-1] + alpha_b], label='r^2=%.3f' % alpha_r_sqr)
    plt.xlabel('background')
    plt.ylabel('alpha')
    plt.legend()

    plt.figure()
    plt.plot(background_range,beta_vals,'o')
    plt.plot([background_range[0], background_range[-1]], [beta_a * background_range[0] + beta_b, beta_a * background_range[-1] + beta_b], label='r^2=%.3f' % beta_r_sqr)
    plt.xlabel('background')
    plt.ylabel('beta')
    plt.legend()

    plt.show()

def analyze_p_b_e_beta(data_dir, file_prefix, p_b_e_range):
    beta_vals=np.zeros((len(p_b_e_range),1))
    alpha_vals=np.zeros((len(p_b_e_range),1))
    p_b_e_vals=np.zeros((len(p_b_e_range),1))
    for idx,p_b_e in enumerate(p_b_e_range):
        file_name=os.path.join(data_dir,file_prefix % p_b_e)
        f = h5py.File(file_name)
        alpha=float(f.attrs['alpha'])
        beta=float(f.attrs['beta'])
        beta_vals[idx]=beta
        alpha_vals[idx]=alpha
        p_b_e_vals[idx]=p_b_e

    clf = LinearRegression()
    clf.fit(p_b_e_vals, alpha_vals)
    alpha_a = clf.coef_[0]
    alpha_b = clf.intercept_
    alpha_r_sqr=clf.score(p_b_e_vals, alpha_vals)

    clf = LinearRegression()
    clf.fit(p_b_e_vals, beta_vals)
    beta_a = clf.coef_[0]
    beta_b = clf.intercept_
    beta_r_sqr=clf.score(p_b_e_vals, beta_vals)

    plt.figure()
    plt.plot(p_b_e_range,alpha_vals,'o')
    plt.plot([p_b_e_range[0], p_b_e_range[-1]], [alpha_a * p_b_e_range[0] + alpha_b, alpha_a * p_b_e_range[-1] + alpha_b], label='r^2=%.3f' % alpha_r_sqr)
    plt.xlabel('p_b_e')
    plt.ylabel('alpha')
    plt.legend()

    plt.figure()
    plt.plot(p_b_e_range,beta_vals,'o')
    plt.plot([p_b_e_range[0], p_b_e_range[-1]], [beta_a * p_b_e_range[0] + beta_b, beta_a * p_b_e_range[-1] + beta_b], label='r^2=%.3f' % beta_r_sqr)
    plt.xlabel('p_b_e')
    plt.ylabel('beta')
    plt.legend()

    plt.show()
