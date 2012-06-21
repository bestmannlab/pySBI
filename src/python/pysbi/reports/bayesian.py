import os
import matplotlib.pylab as plt
import numpy as np
from pysbi.analysis import run_bayesian_analysis
from pysbi.reports.summary import SummaryData
from pysbi.utils import Struct, save_to_png

def create_bayesian_report(title, num_groups, trial_duration, roc_auc, bc_slope, bc_intercept, bc_r_sqr, evidence,
                           posterior, marginals, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range,
                           p_i_e_range, file_prefix, reports_dir):
    report_info=Struct()
    report_info.title=title
    report_info.evidence=evidence
    report_info.roc_auc=roc_auc
    report_info.bc_slope=bc_slope
    report_info.bc_intercept=bc_intercept
    report_info.bc_r_sqr=bc_r_sqr
    report_info.num_groups=num_groups
    report_info.trial_duration=trial_duration
    report_info.p_b_e_range=p_b_e_range
    report_info.p_x_e_range=p_x_e_range
    report_info.p_e_e_range=p_e_e_range
    report_info.p_e_i_range=p_e_i_range
    report_info.p_i_i_range=p_i_i_range
    report_info.p_i_e_range=p_i_e_range

    report_info.posterior={}
    for i,p_b_e in enumerate(p_b_e_range):
        for j,p_x_e in enumerate(p_x_e_range):
            for k,p_e_e in enumerate(p_e_e_range):
                for l,p_e_i in enumerate(p_e_i_range):
                    for m,p_i_i in enumerate(p_i_i_range):
                        for n,p_i_e in enumerate(p_i_e_range):
                            if posterior[i,j,k,l,m,n]>0:
                                report_info.posterior[(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)]=posterior[i,j,k,l,m,n]
    report_info.marginal_prior_p_b_e_url,\
    report_info.marginal_likelihood_p_b_e_url,\
    report_info.marginal_posterior_p_b_e_url=render_marginal_report('p_b_e', p_b_e_range,
        marginals.prior_p_b_e, marginals.likelihood_p_b_e, marginals.posterior_p_b_e, file_prefix, reports_dir)

    report_info.marginal_prior_p_x_e_url,\
    report_info.marginal_likelihood_p_x_e_url,\
    report_info.marginal_posterior_p_x_e_url=render_marginal_report('p_x_e', p_x_e_range,
        marginals.prior_p_x_e, marginals.likelihood_p_x_e, marginals.posterior_p_x_e, file_prefix, reports_dir)

    report_info.marginal_prior_p_e_e_url,\
    report_info.marginal_likelihood_p_e_e_url,\
    report_info.marginal_posterior_p_e_e_url=render_marginal_report('p_e_e', p_e_e_range,
        marginals.prior_p_e_e, marginals.likelihood_p_e_e, marginals.posterior_p_e_e, file_prefix, reports_dir)

    report_info.marginal_prior_p_e_i_url,\
    report_info.marginal_likelihood_p_e_i_url,\
    report_info.marginal_posterior_p_e_i_url=render_marginal_report('p_e_i', p_e_i_range,
        marginals.prior_p_e_i, marginals.likelihood_p_e_i, marginals.posterior_p_e_i, file_prefix, reports_dir)

    report_info.marginal_prior_p_i_i_url,\
    report_info.marginal_likelihood_p_i_i_url,\
    report_info.marginal_posterior_p_i_i_url=render_marginal_report('p_i_i', p_i_i_range,
        marginals.prior_p_i_i, marginals.likelihood_p_i_i, marginals.posterior_p_i_i, file_prefix, reports_dir)

    report_info.marginal_prior_p_i_e_url,\
    report_info.marginal_likelihood_p_i_e_url,\
    report_info.marginal_posterior_p_i_e_url=render_marginal_report('p_i_e', p_i_e_range,
        marginals.prior_p_i_e, marginals.likelihood_p_i_e, marginals.posterior_p_i_e, file_prefix, reports_dir)


    report_info.joint_marginal_p_b_e_p_x_e_url = render_joint_marginal_report('p_b_e', 'p_x_e', p_b_e_range, p_x_e_range,
        marginals.posterior_p_b_e_p_x_e, file_prefix, reports_dir)

    report_info.joint_marginal_p_e_e_p_e_i_url = render_joint_marginal_report('p_e_e', 'p_e_i', p_e_e_range, p_e_i_range,
        marginals.posterior_p_e_e_p_e_i, file_prefix, reports_dir)

    report_info.joint_marginal_p_e_e_p_i_i_url = render_joint_marginal_report('p_e_e', 'p_i_i', p_e_e_range, p_i_i_range,
        marginals.posterior_p_e_e_p_i_i, file_prefix, reports_dir)

    report_info.joint_marginal_p_e_e_p_i_e_url = render_joint_marginal_report('p_e_e', 'p_i_e', p_e_e_range, p_i_e_range,
        marginals.posterior_p_e_e_p_i_e, file_prefix, reports_dir)

    report_info.joint_marginal_p_e_i_p_i_i_url = render_joint_marginal_report('p_e_i', 'p_i_i', p_e_i_range, p_i_i_range,
        marginals.posterior_p_e_i_p_i_i, file_prefix, reports_dir)

    report_info.joint_marginal_p_e_i_p_i_e_url = render_joint_marginal_report('p_e_i', 'p_i_e', p_e_i_range, p_i_e_range,
        marginals.posterior_p_e_i_p_i_e, file_prefix, reports_dir)

    report_info.joint_marginal_p_i_i_p_i_e_url = render_joint_marginal_report('p_i_i', 'p_i_e', p_i_i_range, p_i_e_range,
        marginals.posterior_p_i_i_p_i_e, file_prefix, reports_dir)

    return report_info


def render_marginal_report(param_name, param_range, param_prior, param_likelihood, param_posterior, file_prefix, reports_dir):
    if len(param_range) > 1:
        param_step=param_range[1]-param_range[0]

        fig = plt.figure()
        param_prior[param_prior==0]=1e-7
        plt.bar(np.array(param_range) - .5*param_step, param_prior, param_step)
        plt.xlabel(param_name)
        plt.ylabel('p(%s|M)' % param_name)
        prior_furl = 'img/bayes_%s_marginal_prior_%s.png' % (file_prefix,param_name)
        fname = os.path.join(reports_dir, prior_furl)
        save_to_png(fig, fname)
        plt.close()

        fig = plt.figure()
        param_likelihood[param_likelihood==0]=1e-7
        plt.bar(np.array(param_range) - .5*param_step, param_likelihood, param_step)
        plt.xlabel(param_name)
        plt.ylabel('p(WTA|%s,M)' % param_name)
        likelihood_furl = 'img/bayes_%s_marginal_likelihood_%s.png' % (file_prefix,param_name)
        fname = os.path.join(reports_dir, likelihood_furl)
        save_to_png(fig, fname)
        plt.close()

        fig = plt.figure()
        param_posterior[param_posterior==0]=1e-7
        plt.bar(np.array(param_range) - .5*param_step, param_posterior, param_step)
        plt.xlabel(param_name)
        plt.ylabel('p(%s|WTA,M)' % param_name)
        posterior_furl = 'img/bayes_%s_marginal_posterior_%s.png' % (file_prefix, param_name)
        fname = os.path.join(reports_dir, posterior_furl)
        save_to_png(fig, fname)
        plt.close()

        return prior_furl, likelihood_furl, posterior_furl
    return None, None, None


def render_joint_marginal_report(param1_name, param2_name, param1_range, param2_range, joint_posterior, file_prefix,
                                 reports_dir):
    if len(param1_range) > 1 < len(param2_range):
        fig = plt.figure()
        im = plt.imshow(joint_posterior, extent=[min(param2_range), max(param2_range), min(param1_range),
                                                 max(param1_range)], interpolation='nearest', origin='lower')
        fig.colorbar(im)
        plt.xlabel(param2_name)
        plt.ylabel(param1_name)
        furl = 'img/bayes_%s_joint_marginal_%s_%s.png' % (file_prefix, param1_name, param2_name)
        fname = os.path.join(reports_dir, furl)
        save_to_png(fig, fname)
        plt.close()
        return furl
    return None


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