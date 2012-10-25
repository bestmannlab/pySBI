from jinja2 import Environment, FileSystemLoader
import os
import h5py
import numpy as np
from pysbi.analysis import run_bayesian_analysis
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.bayesian import create_bayesian_report, render_joint_marginal_report
from pysbi.reports.utils import get_local_average, make_report_dirs
from pysbi.util.utils import Struct
import matplotlib.pylab as plt

class SummaryData:
    def __init__(self, num_groups=0, num_trials=0, trial_duration=0, p_b_e_range=np.zeros([1]),
                 p_x_e_range=np.zeros([1]), p_e_e_range=np.zeros([1]), p_e_i_range=np.zeros([1]),
                 p_i_i_range=np.zeros([1]), p_i_e_range=np.zeros([1])):
        self.num_groups=num_groups
        self.num_trials=num_trials
        self.trial_duration=trial_duration
        self.p_b_e_range=p_b_e_range
        self.p_x_e_range=p_x_e_range
        self.p_e_e_range=p_e_e_range
        self.p_e_i_range=p_e_i_range
        self.p_i_i_range=p_i_i_range
        self.p_i_e_range=p_i_e_range
        self.bc_slope=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range), len(p_i_i_range),
                                len(p_i_e_range)])
        self.bc_intercept=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range),
                                    len(p_i_i_range), len(p_i_e_range)])
        self.bc_r_sqr=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range), len(p_i_i_range),
                                len(p_i_e_range)])
        self.auc=np.zeros([len(p_b_e_range), len(p_x_e_range), len(p_e_e_range), len(p_e_i_range), len(p_i_i_range),
                           len(p_i_e_range)])

    def fill(self, auc_dict, bc_slope_dict, bc_intercept_dict, bc_r_sqr_dict, smooth_missing_params=False):
        for i,p_b_e in enumerate(self.p_b_e_range):
            for j,p_x_e in enumerate(self.p_x_e_range):
                for k,p_e_e in enumerate(self.p_e_e_range):
                    for l,p_e_i in enumerate(self.p_e_i_range):
                        for m,p_i_i in enumerate(self.p_i_i_range):
                            for n,p_i_e in enumerate(self.p_i_e_range):
                                if (i,j,k,l,m,n) in auc_dict:
                                    self.auc[i,j,k,l,m,n]=np.mean(auc_dict[(i,j,k,l,m,n)])
                                    self.bc_slope[i,j,k,l,m,n]=np.mean(bc_slope_dict[(i,j,k,l,m,n)])
                                    self.bc_intercept[i,j,k,l,m,n]=np.mean(bc_intercept_dict[(i,j,k,l,m,n)])
                                    self.bc_r_sqr[i,j,k,l,m,n]=np.mean(bc_r_sqr_dict[(i,j,k,l,m,n)])
                                elif smooth_missing_params:
                                    self.auc[i,j,k,l,m,n]=get_local_average(auc_dict,[i,j,k,l,m,n])
                                    self.bc_slope[i,j,k,l,m,n]=get_local_average(bc_slope_dict,[i,j,k,l,m,n])
                                    self.bc_intercept[i,j,k,l,m,n]=get_local_average(bc_intercept_dict,[i,j,k,l,m,n])
                                    self.bc_r_sqr[i,j,k,l,m,n]=get_local_average(bc_r_sqr_dict,[i,j,k,l,m,n])

    def read_from_file(self, filename):
        f=h5py.File(filename)
        self.num_groups=int(f.attrs['num_groups'])
        self.num_trials=int(f.attrs['num_trials'])
        self.trial_duration=float(f.attrs['trial_duration'])
        self.p_b_e_range=np.array(f['p_b_e_range'])
        self.p_x_e_range=np.array(f['p_x_e_range'])
        self.p_e_e_range=np.array(f['p_e_e_range'])
        self.p_e_i_range=np.array(f['p_e_i_range'])
        self.p_i_i_range=np.array(f['p_i_i_range'])
        self.p_i_e_range=np.array(f['p_i_e_range'])
        self.bc_slope=np.array(f['bold_contrast_slope'])
        self.bc_intercept=np.array(f['bold_contrast_intercept'])
        self.bc_r_sqr=np.array(f['bold_contrast_r_sqr'])
        self.auc=np.array(f['auc'])
        f.close()

    def write_to_file(self, filename):
        f = h5py.File(filename, 'w')
        f.attrs['num_groups'] = self.num_groups
        f.attrs['num_trials'] = self.num_trials
        f.attrs['trial_duration'] = self.trial_duration
        f['p_b_e_range']=self.p_b_e_range
        f['p_x_e_range']=self.p_x_e_range
        f['p_e_e_range']=self.p_e_e_range
        f['p_e_i_range']=self.p_e_i_range
        f['p_i_i_range']=self.p_i_i_range
        f['p_i_e_range']=self.p_i_e_range
        f['bold_contrast_slope']=self.bc_slope
        f['bold_contrast_intercept']=self.bc_intercept
        f['bold_contrast_r_sqr']=self.bc_r_sqr
        f['auc']=self.auc
        f.close()


def create_summary_report(summary_file_name, base_report_dir, e_desc):
    make_report_dirs(base_report_dir)

    summary_data=SummaryData()
    summary_data.read_from_file(summary_file_name)

    report_info=Struct()
    report_info.edesc=e_desc
    report_info.roc_auc={}
    report_info.io_slope={}
    report_info.io_intercept={}
    report_info.io_r_sqr={}
    report_info.bc_slope={}
    report_info.bc_intercept={}
    report_info.bc_r_sqr={}
    for i,p_b_e in enumerate(summary_data.p_b_e_range):
        for j,p_x_e in enumerate(summary_data.p_x_e_range):
            for k,p_e_e in enumerate(summary_data.p_e_e_range):
                for l,p_e_i in enumerate(summary_data.p_e_i_range):
                    for m,p_i_i in enumerate(summary_data.p_i_i_range):
                        for n,p_i_e in enumerate(summary_data.p_i_e_range):
                            if summary_data.auc[i,j,k,l,m,n]>0:
                                report_info.roc_auc[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=summary_data.auc[i,j,k,l,m,n]
                                report_info.bc_slope[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=summary_data.bc_slope[i,j,k,l,m,n]
                                report_info.bc_intercept[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=summary_data.bc_intercept[i,j,k,l,m,n]
                                report_info.bc_r_sqr[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=summary_data.bc_r_sqr[i,j,k,l,m,n]
                            else:
                                report_info.roc_auc[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=0
                                report_info.bc_slope[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=0
                                report_info.bc_intercept[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=0
                                report_info.bc_r_sqr[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=0

    report_info.num_groups=summary_data.num_groups
    report_info.trial_duration=summary_data.trial_duration
    report_info.num_trials=summary_data.num_trials
    report_info.p_b_e_range=summary_data.p_b_e_range
    report_info.p_x_e_range=summary_data.p_x_e_range
    report_info.p_e_e_range=summary_data.p_e_e_range
    report_info.p_e_i_range=summary_data.p_e_i_range
    report_info.p_i_i_range=summary_data.p_i_i_range
    report_info.p_i_e_range=summary_data.p_i_e_range

    bayes_analysis=run_bayesian_analysis(summary_data.auc, summary_data.bc_slope, summary_data.bc_intercept,
        summary_data.bc_r_sqr, summary_data.num_trials, summary_data.p_b_e_range, summary_data.p_e_e_range,
        summary_data.p_e_i_range, summary_data.p_i_e_range, summary_data.p_i_i_range, summary_data.p_x_e_range)

    render_summary_report(base_report_dir, bayes_analysis, summary_data.p_b_e_range, summary_data.p_e_e_range,
        summary_data.p_e_i_range, summary_data.p_i_e_range, summary_data.p_i_i_range, summary_data.p_x_e_range,
        report_info)


def plot_l1_pos_bayes_marginals(summary_file_name, y_max=0.3):
    summary_data=SummaryData()
    summary_data.read_from_file(summary_file_name)
    bayes_analysis=run_bayesian_analysis(summary_data.auc, summary_data.bc_slope, summary_data.bc_intercept,
        summary_data.bc_r_sqr, summary_data.num_trials, summary_data.p_b_e_range, summary_data.p_e_e_range,
        summary_data.p_e_i_range, summary_data.p_i_e_range, summary_data.p_i_i_range, summary_data.p_x_e_range)

    fig = plt.figure()
    param_step=summary_data.p_e_e_range[1]-summary_data.p_e_e_range[0]
    bayes_analysis.l1_pos_marginals.posterior_p_e_e[bayes_analysis.l1_pos_marginals.posterior_p_e_e==0]=1e-7
    plt.bar(np.array(summary_data.p_e_e_range) - .5*param_step, bayes_analysis.l1_pos_marginals.posterior_p_e_e, param_step)
    plt.xlabel('p_e_e')
    plt.ylabel('p(p_e_e|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_e_i_range[1]-summary_data.p_e_i_range[0]
    bayes_analysis.l1_pos_marginals.posterior_p_e_i[bayes_analysis.l1_pos_marginals.posterior_p_e_i==0]=1e-7
    plt.bar(np.array(summary_data.p_e_i_range) - .5*param_step, bayes_analysis.l1_pos_marginals.posterior_p_e_i, param_step)
    plt.xlabel('p_e_i')
    plt.ylabel('p(p_e_i|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_i_i_range[1]-summary_data.p_i_i_range[0]
    bayes_analysis.l1_pos_marginals.posterior_p_i_i[bayes_analysis.l1_pos_marginals.posterior_p_i_i==0]=1e-7
    plt.bar(np.array(summary_data.p_i_i_range) - .5*param_step, bayes_analysis.l1_pos_marginals.posterior_p_i_i, param_step)
    plt.xlabel('p_i_i')
    plt.ylabel('p(p_i_i|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_i_e_range[1]-summary_data.p_i_e_range[0]
    bayes_analysis.l1_pos_marginals.posterior_p_i_e[bayes_analysis.l1_pos_marginals.posterior_p_i_e==0]=1e-7
    plt.bar(np.array(summary_data.p_i_e_range) - .5*param_step, bayes_analysis.l1_pos_marginals.posterior_p_i_e, param_step)
    plt.xlabel('p_i_e')
    plt.ylabel('p(p_i_e|A,M)')
    plt.ylim(0.0,y_max)

    plt.show()

def plot_l1_pos_l2_pos_bayes_marginals(summary_file_name, y_max=0.3):
    summary_data=SummaryData()
    summary_data.read_from_file(summary_file_name)
    bayes_analysis=run_bayesian_analysis(summary_data.auc, summary_data.bc_slope, summary_data.bc_intercept,
        summary_data.bc_r_sqr, summary_data.num_trials, summary_data.p_b_e_range, summary_data.p_e_e_range,
        summary_data.p_e_i_range, summary_data.p_i_e_range, summary_data.p_i_i_range, summary_data.p_x_e_range)

    fig = plt.figure()
    param_step=summary_data.p_e_e_range[1]-summary_data.p_e_e_range[0]
    bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_e_e[bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_e_e==0]=1e-7
    plt.bar(np.array(summary_data.p_e_e_range) - .5*param_step, bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_e_e, param_step)
    plt.xlabel('p_e_e')
    plt.ylabel('p(p_e_e|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_e_i_range[1]-summary_data.p_e_i_range[0]
    bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_e_i[bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_e_i==0]=1e-7
    plt.bar(np.array(summary_data.p_e_i_range) - .5*param_step, bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_e_i, param_step)
    plt.xlabel('p_e_i')
    plt.ylabel('p(p_e_i|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_i_i_range[1]-summary_data.p_i_i_range[0]
    bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_i_i[bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_i_i==0]=1e-7
    plt.bar(np.array(summary_data.p_i_i_range) - .5*param_step, bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_i_i, param_step)
    plt.xlabel('p_i_i')
    plt.ylabel('p(p_i_i|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_i_e_range[1]-summary_data.p_i_e_range[0]
    bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_i_e[bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_i_e==0]=1e-7
    plt.bar(np.array(summary_data.p_i_e_range) - .5*param_step, bayes_analysis.l1_pos_l2_pos_marginals.posterior_p_i_e, param_step)
    plt.xlabel('p_i_e')
    plt.ylabel('p(p_i_e|A,M)')
    plt.ylim(0.0,y_max)

    plt.show()

def plot_l1_pos_l2_zero_bayes_marginals(summary_file_name, y_max=0.3):
    summary_data=SummaryData()
    summary_data.read_from_file(summary_file_name)
    bayes_analysis=run_bayesian_analysis(summary_data.auc, summary_data.bc_slope, summary_data.bc_intercept,
        summary_data.bc_r_sqr, summary_data.num_trials, summary_data.p_b_e_range, summary_data.p_e_e_range,
        summary_data.p_e_i_range, summary_data.p_i_e_range, summary_data.p_i_i_range, summary_data.p_x_e_range)

    fig = plt.figure()
    param_step=summary_data.p_e_e_range[1]-summary_data.p_e_e_range[0]
    bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_e_e[bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_e_e==0]=1e-7
    plt.bar(np.array(summary_data.p_e_e_range) - .5*param_step, bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_e_e, param_step)
    plt.xlabel('p_e_e')
    plt.ylabel('p(p_e_e|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_e_i_range[1]-summary_data.p_e_i_range[0]
    bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_e_i[bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_e_i==0]=1e-7
    plt.bar(np.array(summary_data.p_e_i_range) - .5*param_step, bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_e_i, param_step)
    plt.xlabel('p_e_i')
    plt.ylabel('p(p_e_i|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_i_i_range[1]-summary_data.p_i_i_range[0]
    bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_i_i[bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_i_i==0]=1e-7
    plt.bar(np.array(summary_data.p_i_i_range) - .5*param_step, bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_i_i, param_step)
    plt.xlabel('p_i_i')
    plt.ylabel('p(p_i_i|A,M)')
    plt.ylim(0.0,y_max)

    fig = plt.figure()
    param_step=summary_data.p_i_e_range[1]-summary_data.p_i_e_range[0]
    bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_i_e[bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_i_e==0]=1e-7
    plt.bar(np.array(summary_data.p_i_e_range) - .5*param_step, bayes_analysis.l1_pos_l2_zero_marginals.posterior_p_i_e, param_step)
    plt.xlabel('p_i_e')
    plt.ylabel('p(p_i_e|A,M)')
    plt.ylim(0.0,y_max)

    plt.show()

def render_summary_report(base_report_dir, bayes_analysis, p_b_e_range, p_e_e_range, p_e_i_range, p_i_e_range,
                          p_i_i_range, p_x_e_range, report_info):
    template_file = 'bayes_analysis.html'
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(template_file)


    report_info.l1_pos_report_info = create_bayesian_report('Level 1 - Positive', report_info.num_groups, report_info.trial_duration,
        report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept, report_info.bc_r_sqr,
        bayes_analysis.l1_pos_evidence, bayes_analysis.l1_pos_posterior, bayes_analysis.l1_pos_marginals, p_b_e_range, p_x_e_range,
        p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range, 'l1_pos', base_report_dir, report_info.edesc)
    output_file = 'l1_pos_bayes_analysis.html'
    report_info.l1_pos_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_pos_report_info)
    stream.dump(fname)

    report_info.l1_neg_report_info = create_bayesian_report('Level 1 - Negative', report_info.num_groups, report_info.trial_duration,
        report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept, report_info.bc_r_sqr,
        bayes_analysis.l1_neg_evidence, bayes_analysis.l1_neg_posterior, bayes_analysis.l1_neg_marginals, p_b_e_range, p_x_e_range,
        p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range, 'l1_neg', base_report_dir, report_info.edesc)
    output_file = 'l1_neg_bayes_analysis.html'
    report_info.l1_neg_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_neg_report_info)
    stream.dump(fname)


    report_info.l1_pos_l2_neg_report_info = create_bayesian_report('Level 1 - Positive, Level 2 - Negative Bold-Contrast Slope',
        report_info.num_groups, report_info.trial_duration, report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept,
        report_info.bc_r_sqr, bayes_analysis.l1_pos_l2_neg_evidence, bayes_analysis.l1_pos_l2_neg_posterior,
        bayes_analysis.l1_pos_l2_neg_marginals, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
        'l1_pos_l2_neg', base_report_dir, report_info.edesc)
    output_file = 'l1_pos_l2_neg_bayes_analysis.html'
    report_info.l1_pos_l2_neg_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_pos_l2_neg_report_info)
    stream.dump(fname)

    report_info.l1_neg_l2_neg_report_info = create_bayesian_report('Level 1 - Negative, Level 2 - Negative Bold-Contrast Slope',
        report_info.num_groups, report_info.trial_duration, report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept,
        report_info.bc_r_sqr, bayes_analysis.l1_neg_l2_neg_evidence, bayes_analysis.l1_neg_l2_neg_posterior,
        bayes_analysis.l1_neg_l2_neg_marginals, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
        'l1_neg_l2_neg', base_report_dir, report_info.edesc)
    output_file = 'l1_neg_l2_neg_bayes_analysis.html'
    report_info.l1_neg_l2_neg_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_neg_l2_neg_report_info)
    stream.dump(fname)

    report_info.l1_pos_l2_pos_report_info = create_bayesian_report('Level 1- Positive, Level 2 - Positive Bold-Contrast Slope',
        report_info.num_groups, report_info.trial_duration, report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept,
        report_info.bc_r_sqr, bayes_analysis.l1_pos_l2_pos_evidence, bayes_analysis.l1_pos_l2_pos_posterior,
        bayes_analysis.l1_pos_l2_pos_marginals, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
        'l1_pos_l2_pos', base_report_dir, report_info.edesc)
    output_file = 'l1_pos_l2_pos_bayes_analysis.html'
    report_info.l1_pos_l2_pos_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_pos_l2_pos_report_info)
    stream.dump(fname)

    report_info.l1_neg_l2_pos_report_info = create_bayesian_report('Level 1- Negative, Level 2 - Positive Bold-Contrast Slope',
        report_info.num_groups, report_info.trial_duration, report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept,
        report_info.bc_r_sqr, bayes_analysis.l1_neg_l2_pos_evidence, bayes_analysis.l1_neg_l2_pos_posterior,
        bayes_analysis.l1_neg_l2_pos_marginals, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
        'l1_neg_l2_pos', base_report_dir, report_info.edesc)
    output_file = 'l1_neg_l2_pos_bayes_analysis.html'
    report_info.l1_neg_l2_pos_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_neg_l2_pos_report_info)
    stream.dump(fname)

    report_info.l1_pos_l2_zero_report_info = create_bayesian_report('Level 1 - Positive, Level 2 - Zero Bold-Contrast Slope',
        report_info.num_groups, report_info.trial_duration, report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept,
        report_info.bc_r_sqr, bayes_analysis.l1_pos_l2_zero_evidence, bayes_analysis.l1_pos_l2_zero_posterior,
        bayes_analysis.l1_pos_l2_zero_marginals, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
        'l1_pos_l2_zero', base_report_dir, report_info.edesc)
    output_file = 'l1_pos_l2_zero_bayes_analysis.html'
    report_info.l1_pos_l2_zero_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_pos_l2_zero_report_info)
    stream.dump(fname)

    report_info.l1_neg_l2_zero_report_info = create_bayesian_report('Level 1 - Negative, Level 2 - Zero Bold-Contrast Slope',
        report_info.num_groups, report_info.trial_duration, report_info.roc_auc, report_info.bc_slope, report_info.bc_intercept,
        report_info.bc_r_sqr, bayes_analysis.l1_neg_l2_zero_evidence, bayes_analysis.l1_neg_l2_zero_posterior,
        bayes_analysis.l1_neg_l2_zero_marginals, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range, p_i_e_range,
        'l1_neg_l2_zero', base_report_dir, report_info.edesc)
    output_file = 'l1_neg_l2_zero_bayes_analysis.html'
    report_info.l1_neg_l2_zero_url = output_file
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info.l1_neg_l2_zero_report_info)
    stream.dump(fname)


    template_file = 'wta_network.html'
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(template_file)
    output_file = 'wta_network.html'
    fname = os.path.join(base_report_dir, output_file)
    stream = template.stream(rinfo=report_info)
    stream.dump(fname)


def regenerate_bayesian_figures(summary_filename, reports_dir):
    summary_data=SummaryData()
    summary_data.read_from_file(summary_filename)
    bayes_analysis=run_bayesian_analysis(summary_data.auc, summary_data.bc_slope, summary_data.bc_intercept,
        summary_data.bc_r_sqr, summary_data.num_trials, summary_data.p_b_e_range, summary_data.p_e_e_range,
        summary_data.p_e_i_range, summary_data.p_i_e_range, summary_data.p_i_i_range, summary_data.p_x_e_range)

    p_b_e_range=summary_data.p_b_e_range
    p_x_e_range=summary_data.p_x_e_range
    p_e_e_range=summary_data.p_e_e_range
    p_e_i_range=summary_data.p_e_i_range
    p_i_i_range=summary_data.p_i_i_range
    p_i_e_range=summary_data.p_i_e_range

    marginal_list=[bayes_analysis.l1_pos_marginals, bayes_analysis.l1_neg_marginals,
                   bayes_analysis.l1_pos_l2_neg_marginals, bayes_analysis.l1_neg_l2_neg_marginals,
                   bayes_analysis.l1_pos_l2_pos_marginals, bayes_analysis.l1_neg_l2_pos_marginals,
                   bayes_analysis.l1_pos_l2_zero_marginals, bayes_analysis.l1_neg_l2_zero_marginals]
    file_prefix_list=['l1_pos','l1_neg','l1_pos_l2_neg','l1_neg_l2_neg','l1_pos_l2_pos','l1_neg_l2_pos','l1_pos_l2_zero',
                      'l1_neg_l2_zero']

    for(marginals,file_prefix) in zip(marginal_list,file_prefix_list):
        render_joint_marginal_report('p_b_e', 'p_x_e', p_b_e_range, p_x_e_range, marginals.posterior_p_b_e_p_x_e, file_prefix,
            reports_dir)

        render_joint_marginal_report('p_e_e', 'p_e_i', p_e_e_range, p_e_i_range, marginals.posterior_p_e_e_p_e_i, file_prefix,
            reports_dir)

        render_joint_marginal_report('p_e_e', 'p_i_i', p_e_e_range, p_i_i_range, marginals.posterior_p_e_e_p_i_i, file_prefix,
            reports_dir)

        render_joint_marginal_report('p_e_e', 'p_i_e', p_e_e_range, p_i_e_range, marginals.posterior_p_e_e_p_i_e, file_prefix,
            reports_dir)

        render_joint_marginal_report('p_e_i', 'p_i_i', p_e_i_range, p_i_i_range, marginals.posterior_p_e_i_p_i_i, file_prefix,
            reports_dir)

        render_joint_marginal_report('p_e_i', 'p_i_e', p_e_i_range, p_i_e_range, marginals.posterior_p_e_i_p_i_e, file_prefix,
            reports_dir)

        render_joint_marginal_report('p_i_i', 'p_i_e', p_i_i_range, p_i_e_range, marginals.posterior_p_i_i_p_i_e, file_prefix,
            reports_dir)