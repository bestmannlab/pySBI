import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
from jinja2 import Environment, FileSystemLoader
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.util.utils import save_to_png, save_to_eps, twoway_interaction_r, pairwise_comparisons
from pysbi.wta.dcs.analysis import DCSComparisonReport, condition_colors


class ParamExploreReport():
    def __init__(self, data_dir, file_prefix, virtual_subj_ids, num_trials, reports_dir, control=False,
                 stim_gains=[8,6,4,2,1,0.5,0.25]):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.virtual_subj_ids=virtual_subj_ids
        self.num_trials=num_trials
        self.reports_dir=reports_dir
        self.stim_gains=stim_gains
        #self.stim_gains=[8,7,6,5,4,3,2,1,0.5,0.25]
        #self.stim_gains=[8,6,4,2,1,0.5,0.25]
        #self.stim_gains=[4,2,1,0.5,0.25]
        self.control=control
        self.stim_level_reports={}    

    def create_report(self, regenerate_stim_level_plots=False, regenerate_subject_plots=False,
                      regenerate_session_plots=False, regenerate_trial_plots=False):

        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])

        thresh={'control':{},'anode':{},'cathode':{}}
        self.thresh_difference={'anode':{},'cathode':{}}
        self.rt_diff_slope={'anode':{},'cathode':{}}
        self.prestim_bias_diff={'anode':{},'cathode':{}}
        self.prestim_bias={'anode':{},'cathode':{}}
        self.coherence_prestim_bias={'n':{'anode':{},'cathode':{}},'lambda':{'anode':{},'cathode':{}}}
        self.prestim_bias_rt={'offset':{'anode':{}, 'cathode':{}},'slope':{'anode':{}, 'cathode':{}}}
        self.logistic_coeff={'bias':{'anode':{}, 'cathode':{}},'ev diff':{'anode':{}, 'cathode':{}}}
        self.small_input_diff_logistic_coeff={'bias':{'anode':{}, 'cathode':{}},'ev diff':{'anode':{}, 'cathode':{}}}
        self.large_input_diff_logistic_coeff={'bias':{'anode':{}, 'cathode':{}},'ev diff':{'anode':{}, 'cathode':{}}}
        self.perc_no_response={'anode':{},'cathode':{}}
        self.bias_perc_left_params={'k':{'anode':{}, 'cathode':{}}}

        thresh_diff_groups=[]
        rt_diff_slope_groups=[]
        prestim_bias_diff_groups=[]
        prestim_bias_groups=[]
        coherence_prestim_bias_groups={
            'n':[],
            'lambda':[]
        }
        prestim_bias_rt_groups={
            'offset':[],
            'slope':[]
        }
        logistic_coeff_groups={
            'bias':[],
            'ev diff':[]
        }
        small_input_diff_logistic_coeff_groups={
            'bias':[],
            'ev diff':[]
        }
        large_input_diff_logistic_coeff_groups={
            'bias':[],
            'ev diff':[]
        }
        perc_no_response_groups=[]
        bias_perc_left_groups={
            'k':[]
        }

        out_file=file(os.path.join(self.reports_dir,'stim_data.csv'),'w')
        out_file.write('%s\n' % ','.join(['subject','intensity','condition','thresh_diff','rt_diff_slope',
                                          'prestim_bias_diff','prestim_bias','coherence_prestim_bias_n',
                                          'coherence_prestim_bias_lambda','prestim_bias_rt_offset',
                                          'prestim_bias_rt_slope','logistic_coeff_bias','logistic_coeff_ev_diff',
                                          'small_input_diff_logistic_coeff_bias', 'small_input_diff_logistic_coeff_ev_diff',
                                          'large_input_diff_logistic_coeff_bias', 'large_input_diff_logistic_coeff_ev_diff',
                                          'perc_no_resp','bias_perc_left_n']))
        for stim_gain in self.stim_gains:
            stim_levels={'control':(0,0),'anode':(1.0*stim_gain,-0.5*stim_gain), 'cathode':(-1.0*stim_gain,0.5*stim_gain)}
            if self.control:
                stim_levels={'control':(0,0),'anode':(1.0*stim_gain,0), 'cathode':(-1.0*stim_gain,0)}
            report_dir=os.path.join(self.reports_dir,'level_%.2f' % stim_gain)
            self.stim_level_reports[stim_gain]=DCSComparisonReport(self.data_dir,
                self.file_prefix,self.virtual_subj_ids, stim_levels, self.num_trials, report_dir, '',
                contrast_range=(0.0, .032, .064, .128, .256, .512), xlog=False)
            self.stim_level_reports[stim_gain].create_report(regenerate_plots=regenerate_stim_level_plots,
                regenerate_subject_plots=regenerate_subject_plots, regenerate_session_plots=regenerate_session_plots,
                regenerate_trial_plots=regenerate_trial_plots)

            thresh['control'][stim_gain]=[]
            thresh['anode'][stim_gain]=[]
            thresh['cathode'][stim_gain]=[]

            self.thresh_difference['anode'][stim_gain]=[]
            self.thresh_difference['cathode'][stim_gain]=[]

            self.rt_diff_slope['anode'][stim_gain]=[]
            self.rt_diff_slope['cathode'][stim_gain]=[]

            self.prestim_bias_diff['anode'][stim_gain]=[]
            self.prestim_bias_diff['cathode'][stim_gain]=[]

            self.prestim_bias['anode'][stim_gain]=[]
            self.prestim_bias['cathode'][stim_gain]=[]

            self.coherence_prestim_bias['n']['anode'][stim_gain]=[]
            self.coherence_prestim_bias['n']['cathode'][stim_gain]=[]
            self.coherence_prestim_bias['lambda']['anode'][stim_gain]=[]
            self.coherence_prestim_bias['lambda']['cathode'][stim_gain]=[]

            self.prestim_bias_rt['offset']['anode'][stim_gain]=[]
            self.prestim_bias_rt['offset']['cathode'][stim_gain]=[]
            self.prestim_bias_rt['slope']['anode'][stim_gain]=[]
            self.prestim_bias_rt['slope']['cathode'][stim_gain]=[]

            self.logistic_coeff['bias']['anode'][stim_gain]=self.stim_level_reports[stim_gain].logistic_coeffs['bias']['anode']
            self.logistic_coeff['bias']['cathode'][stim_gain]=self.stim_level_reports[stim_gain].logistic_coeffs['bias']['cathode']
            self.logistic_coeff['ev diff']['anode'][stim_gain]=self.stim_level_reports[stim_gain].logistic_coeffs['ev diff']['anode']
            self.logistic_coeff['ev diff']['cathode'][stim_gain]=self.stim_level_reports[stim_gain].logistic_coeffs['ev diff']['cathode']

            self.small_input_diff_logistic_coeff['bias']['anode'][stim_gain]=self.stim_level_reports[stim_gain].small_input_diff_logistic_coeffs['bias']['anode']
            self.small_input_diff_logistic_coeff['bias']['cathode'][stim_gain]=self.stim_level_reports[stim_gain].small_input_diff_logistic_coeffs['bias']['cathode']
            self.small_input_diff_logistic_coeff['ev diff']['anode'][stim_gain]=self.stim_level_reports[stim_gain].small_input_diff_logistic_coeffs['ev diff']['anode']
            self.small_input_diff_logistic_coeff['ev diff']['cathode'][stim_gain]=self.stim_level_reports[stim_gain].small_input_diff_logistic_coeffs['ev diff']['cathode']

            self.large_input_diff_logistic_coeff['bias']['anode'][stim_gain]=self.stim_level_reports[stim_gain].large_input_diff_logistic_coeffs['bias']['anode']
            self.large_input_diff_logistic_coeff['bias']['cathode'][stim_gain]=self.stim_level_reports[stim_gain].large_input_diff_logistic_coeffs['bias']['cathode']
            self.large_input_diff_logistic_coeff['ev diff']['anode'][stim_gain]=self.stim_level_reports[stim_gain].large_input_diff_logistic_coeffs['ev diff']['anode']
            self.large_input_diff_logistic_coeff['ev diff']['cathode'][stim_gain]=self.stim_level_reports[stim_gain].large_input_diff_logistic_coeffs['ev diff']['cathode']

            self.perc_no_response['anode'][stim_gain]=np.array(self.stim_level_reports[stim_gain].perc_no_response['anode'])
            self.perc_no_response['cathode'][stim_gain]=np.array(self.stim_level_reports[stim_gain].perc_no_response['cathode'])

            self.bias_perc_left_params['k']['anode'][stim_gain]=[]
            self.bias_perc_left_params['k']['cathode'][stim_gain]=[]

            for idx,subj in enumerate(self.stim_level_reports[stim_gain].virtual_subj_ids):
                thresh['control'][stim_gain].append(self.stim_level_reports[stim_gain].subjects[subj].thresh['control'])
                for condition in ['anode','cathode']:
                    thresh[condition][stim_gain].append(self.stim_level_reports[stim_gain].subjects[subj].thresh[condition])
                    thresh_diff=self.stim_level_reports[stim_gain].subjects[subj].thresh[condition]-self.stim_level_reports[stim_gain].subjects[subj].thresh['control']
                    rt_diff_slope=self.stim_level_reports[stim_gain].subjects[subj].rt_diff_slope[condition]
                    prestim_bias_diff=self.stim_level_reports[stim_gain].subjects[subj].mean_biases[condition]-self.stim_level_reports[stim_gain].subjects[subj].mean_biases['control']
                    prestim_bias=self.stim_level_reports[stim_gain].subjects[subj].mean_biases[condition]
                    data_vals=['%d' % subj,'%.3f' % stim_gain,condition,'%.4f' % thresh_diff, '%.4f' % rt_diff_slope,
                               '%.4f' % prestim_bias_diff,'%.4f' % prestim_bias]
                    self.thresh_difference[condition][stim_gain].append(thresh_diff)
                    self.rt_diff_slope[condition][stim_gain].append(rt_diff_slope)
                    self.prestim_bias_diff[condition][stim_gain].append(prestim_bias_diff)
                    self.prestim_bias[condition][stim_gain].append(prestim_bias)
                    for param in ['n','lambda']:
                        if condition in self.stim_level_reports[stim_gain].subjects[subj].coherence_prestim_bias_params[param]:
                            coherence_bias_param=self.stim_level_reports[stim_gain].subjects[subj].coherence_prestim_bias_params[param][condition]
                            self.coherence_prestim_bias[param][condition][stim_gain].append(coherence_bias_param)
                            data_vals.append('%.4f' % coherence_bias_param)
                            coherence_prestim_bias_groups[param].append((coherence_bias_param,stim_gain,condition))
                        else:
                            data_vals.append('NA')
                            coherence_prestim_bias_groups[param].append((float('NaN'),stim_gain,condition))
                    for param in ['offset','slope']:
                        if condition in self.stim_level_reports[stim_gain].subjects[subj].bias_rt_params[param]:
                            bias_rt_param=self.stim_level_reports[stim_gain].subjects[subj].bias_rt_params[param][condition]
                            self.prestim_bias_rt[param][condition][stim_gain].append(bias_rt_param)
                            data_vals.append('%.4f' % bias_rt_param)
                            prestim_bias_rt_groups[param].append((bias_rt_param,stim_gain,condition))
                        else:
                            data_vals.append('NA')
                            prestim_bias_rt_groups[param].append((float('NaN'),stim_gain,condition))
                    data_vals.append('%.4f' % self.logistic_coeff['bias'][condition][stim_gain][idx])
                    data_vals.append('%.4f' % self.logistic_coeff['ev diff'][condition][stim_gain][idx])
                    data_vals.append('%.4f' % self.small_input_diff_logistic_coeff['bias'][condition][stim_gain][idx])
                    data_vals.append('%.4f' % self.small_input_diff_logistic_coeff['ev diff'][condition][stim_gain][idx])
                    data_vals.append('%.4f' % self.large_input_diff_logistic_coeff['bias'][condition][stim_gain][idx])
                    data_vals.append('%.4f' % self.large_input_diff_logistic_coeff['ev diff'][condition][stim_gain][idx])
                    data_vals.append('%.4f' % self.perc_no_response[condition][stim_gain][idx])
                    bias_per_left_param=self.stim_level_reports[stim_gain].subjects[subj].bias_perc_left_params['k'][condition]
                    self.bias_perc_left_params['k'][condition][stim_gain].append(bias_per_left_param)
                    data_vals.append('%.4f' % bias_per_left_param)
                    out_file.write('%s\n' % ','.join(data_vals))

                    thresh_diff_groups.append((thresh_diff,stim_gain,condition))
                    rt_diff_slope_groups.append((rt_diff_slope,stim_gain,condition))
                    prestim_bias_diff_groups.append((prestim_bias_diff,stim_gain,condition))
                    prestim_bias_groups.append((prestim_bias,stim_gain,condition))
                    logistic_coeff_groups['bias'].append((self.logistic_coeff['bias'][condition][stim_gain][idx],stim_gain,condition))
                    logistic_coeff_groups['ev diff'].append((self.logistic_coeff['ev diff'][condition][stim_gain][idx],stim_gain,condition))
                    small_input_diff_logistic_coeff_groups['bias'].append((self.small_input_diff_logistic_coeff['bias'][condition][stim_gain][idx],stim_gain,condition))
                    small_input_diff_logistic_coeff_groups['ev diff'].append((self.small_input_diff_logistic_coeff['ev diff'][condition][stim_gain][idx],stim_gain,condition))
                    large_input_diff_logistic_coeff_groups['bias'].append((self.large_input_diff_logistic_coeff['bias'][condition][stim_gain][idx],stim_gain,condition))
                    large_input_diff_logistic_coeff_groups['ev diff'].append((self.large_input_diff_logistic_coeff['ev diff'][condition][stim_gain][idx],stim_gain,condition))
                    perc_no_response_groups.append((self.perc_no_response[condition][stim_gain][idx],stim_gain,condition))
                    bias_perc_left_groups['k'].append((bias_per_left_param,stim_gain,condition))


                self.stim_level_reports[stim_gain].subjects[subj].sessions={}
            #thresh_diff_groups.append([self.thresh_difference['anode'][stim_gain],self.thresh_difference['cathode'][stim_gain]])
            #rt_diff_slope_groups.append([self.rt_diff_slope['anode'][stim_gain],self.rt_diff_slope['cathode'][stim_gain]])
            #prestim_bias_diff_groups.append([self.prestim_bias_diff['anode'][stim_gain],self.prestim_bias_diff['cathode'][stim_gain]])
            #prestim_bias_groups.append([self.prestim_bias['anode'][stim_gain],self.prestim_bias['cathode'][stim_gain]])
            #coherence_prestim_bias_groups['n'].append([self.coherence_prestim_bias['n']['anode'][stim_gain],self.coherence_prestim_bias['n']['cathode'][stim_gain]])
            #coherence_prestim_bias_groups['lambda'].append([self.coherence_prestim_bias['lambda']['anode'][stim_gain],self.coherence_prestim_bias['lambda']['cathode'][stim_gain]])
            #prestim_bias_rt_groups['offset'].append([self.prestim_bias_rt['offset']['anode'][stim_gain],self.prestim_bias_rt['offset']['cathode'][stim_gain]])
            #prestim_bias_rt_groups['slope'].append([self.prestim_bias_rt['slope']['anode'][stim_gain],self.prestim_bias_rt['slope']['cathode'][stim_gain]])
            #logistic_coeff_groups['bias'].append([self.logistic_coeff['bias']['anode'][stim_gain],self.logistic_coeff['bias']['cathode'][stim_gain]])
            #logistic_coeff_groups['ev diff'].append([self.logistic_coeff['ev diff']['anode'][stim_gain],self.logistic_coeff['ev diff']['cathode'][stim_gain]])
            #perc_no_response_groups.append([self.perc_no_response['anode'][stim_gain],self.perc_no_response['cathode'][stim_gain]])
            #bias_perc_left_groups['k'].append([self.bias_perc_left_params['k']['anode'][stim_gain],self.bias_perc_left_params['k']['cathode'][stim_gain]])

        self.mean_thresh={'control':{},'anode':{},'cathode':{}}
        self.std_thresh={'control':{},'anode':{},'cathode':{}}
        for condition in self.mean_thresh:
            for stim_gain in self.stim_gains:
                self.mean_thresh[condition][stim_gain]=np.mean(thresh[condition][stim_gain])
                self.std_thresh[condition][stim_gain]=np.std(thresh[condition][stim_gain])/np.sqrt(len(thresh[condition][stim_gain]))

        out_file.close()

        furl='img/thresh'
        fname=os.path.join(self.reports_dir, furl)
        self.thresh_url='%s.png' % furl
        fig=plt.figure()
        mean_thresh_diffs={'anode':[],'cathode':[]}
        std_thresh_diffs={'anode':[],'cathode':[]}
        for stim_gain in self.stim_gains:
            mean_thresh_diffs['anode'].append(np.mean(self.thresh_difference['anode'][stim_gain]))
            std_thresh_diffs['anode'].append(np.std(self.thresh_difference['anode'][stim_gain])/np.sqrt(len(self.thresh_difference['anode'][stim_gain])))
            mean_thresh_diffs['cathode'].append(np.mean(self.thresh_difference['cathode'][stim_gain]))
            std_thresh_diffs['cathode'].append(np.std(self.thresh_difference['cathode'][stim_gain])/np.sqrt(len(self.thresh_difference['cathode'][stim_gain])))
        plt.errorbar(self.stim_gains, mean_thresh_diffs['anode'],yerr=std_thresh_diffs['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_thresh_diffs['cathode'],yerr=std_thresh_diffs['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Threshold difference')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.thresh_anova=twoway_interaction(thresh_diff_groups,'stim gain','condition','html')
        self.thresh_diff_stats=twoway_interaction_r('thresh_diff',['stim_intensity','condition'],thresh_diff_groups)
        self.thresh_diff_pairwise=pairwise_comparisons(self.thresh_difference,self.stim_gains,['anode','cathode'])

        furl='img/rt_diff_slope'
        fname=os.path.join(self.reports_dir, furl)
        self.rt_diff_slope_url='%s.png' % furl
        fig=plt.figure()
        mean_rt_diff_slopes={'anode':[],'cathode':[]}
        std_rt_diff_slopes={'anode':[],'cathode':[]}
        for stim_gain in self.stim_gains:
            mean_rt_diff_slopes['anode'].append(np.mean(self.rt_diff_slope['anode'][stim_gain]))
            std_rt_diff_slopes['anode'].append(np.std(self.rt_diff_slope['anode'][stim_gain]))
            mean_rt_diff_slopes['cathode'].append(np.mean(self.rt_diff_slope['cathode'][stim_gain]))
            std_rt_diff_slopes['cathode'].append(np.std(self.rt_diff_slope['cathode'][stim_gain]))
        plt.errorbar(self.stim_gains, mean_rt_diff_slopes['anode'],yerr=std_rt_diff_slopes['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_rt_diff_slopes['cathode'],yerr=std_rt_diff_slopes['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('RT Difference slope')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.rt_diff_slope_anova=twoway_interaction(rt_diff_slope_groups,'stim gain','condition','html')
        self.rt_diff_slope_stats=twoway_interaction_r('rt_diff_slope',['stim_intensity','condition'],rt_diff_slope_groups)
        self.rt_diff_slope_pairwise=pairwise_comparisons(self.rt_diff_slope,self.stim_gains,['anode','cathode'])

        furl='img/prestim_bias_diff'
        fname=os.path.join(self.reports_dir, furl)
        self.prestim_bias_diff_url='%s.png' % furl
        fig=plt.figure()
        mean_prestim_bias_diffs={'anode':[],'cathode':[]}
        std_prestim_bias_diffs={'anode':[],'cathode':[]}
        for stim_gain in self.stim_gains:
            mean_prestim_bias_diffs['anode'].append(np.mean(self.prestim_bias_diff['anode'][stim_gain]))
            std_prestim_bias_diffs['anode'].append(np.std(self.prestim_bias_diff['anode'][stim_gain])/np.sqrt(len(self.prestim_bias_diff['anode'][stim_gain])))
            mean_prestim_bias_diffs['cathode'].append(np.mean(self.prestim_bias_diff['cathode'][stim_gain]))
            std_prestim_bias_diffs['cathode'].append(np.std(self.prestim_bias_diff['cathode'][stim_gain])/np.sqrt(len(self.prestim_bias_diff['cathode'][stim_gain])))
        plt.errorbar(self.stim_gains, mean_prestim_bias_diffs['anode'],yerr=std_prestim_bias_diffs['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_prestim_bias_diffs['cathode'],yerr=std_prestim_bias_diffs['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([-3,3])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Prestimulus Bias Difference')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.prestim_bias_diff_anova=twoway_interaction(prestim_bias_diff_groups,'stim gain','condition','html')
        self.prestim_bias_diff_stats=twoway_interaction_r('prestim_bias_diff',['stim_intensity','condition'],prestim_bias_diff_groups)
        self.prestim_bias_diff_pairwise=pairwise_comparisons(self.prestim_bias_diff,self.stim_gains,['anode','cathode'])

        furl='img/prestim_bias'
        fname=os.path.join(self.reports_dir, furl)
        self.prestim_bias_url='%s.png' % furl
        fig=plt.figure()
        mean_prestim_bias={'anode':[],'cathode':[]}
        std_prestim_bias={'anode':[],'cathode':[]}
        for stim_gain in self.stim_gains:
            mean_prestim_bias['anode'].append(np.mean(self.prestim_bias['anode'][stim_gain]))
            std_prestim_bias['anode'].append(np.std(self.prestim_bias['anode'][stim_gain])/np.sqrt(len(self.prestim_bias['anode'][stim_gain])))
            mean_prestim_bias['cathode'].append(np.mean(self.prestim_bias['cathode'][stim_gain]))
            std_prestim_bias['cathode'].append(np.std(self.prestim_bias['cathode'][stim_gain])/np.sqrt(len(self.prestim_bias['cathode'][stim_gain])))
        plt.errorbar(self.stim_gains, mean_prestim_bias['anode'],yerr=std_prestim_bias['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_prestim_bias['cathode'],yerr=std_prestim_bias['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        #plt.ylim([-3,3])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Prestimulus Bias')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.prestim_bias_anova=twoway_interaction(prestim_bias_groups,'stim gain','condition','html')
        self.prestim_bias_stats=twoway_interaction_r('prestim_bias',['stim_intensity','condition'],prestim_bias_groups)
        self.prestim_bias_pairwise=pairwise_comparisons(self.prestim_bias,self.stim_gains,['anode','cathode'])

        mean_coherence_prestim_bias={'n':{'anode':[],'cathode':[]},'lambda':{'anode':[],'cathode':[]}}
        std_coherence_prestim_bias={'n':{'anode':[],'cathode':[]},'lambda':{'anode':[],'cathode':[]}}
        for stim_gain in self.stim_gains:
            for param in mean_coherence_prestim_bias:
                for condition in mean_coherence_prestim_bias[param]:
                    mean_coherence_prestim_bias[param][condition].append(np.mean(self.coherence_prestim_bias[param][condition][stim_gain]))
                    std_coherence_prestim_bias[param][condition].append(np.std(self.coherence_prestim_bias[param][condition][stim_gain])/np.sqrt(len(self.coherence_prestim_bias[param][condition][stim_gain])))

        furl='img/coherence_prestim_bias_n'
        fname=os.path.join(self.reports_dir, furl)
        self.coherence_prestim_bias_n_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_coherence_prestim_bias['n']['anode'],yerr=std_coherence_prestim_bias['n']['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_coherence_prestim_bias['n']['cathode'],yerr=std_coherence_prestim_bias['n']['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Coherence - Prestim Bias: n')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.coherence_prestim_bias_n_anova=''#twoway_interaction(coherence_prestim_bias_groups['n'],'stim gain','condition','html')
        self.coherence_prestim_bias_n_stats=twoway_interaction_r('coherence_prestim_bias_n',['stim_intensity','condition'],coherence_prestim_bias_groups['n'])
        self.coherence_prestim_bias_n_pairwise=pairwise_comparisons(self.coherence_prestim_bias['n'],self.stim_gains,['anode','cathode'])

        furl='img/coherence_prestim_bias_lambda'
        fname=os.path.join(self.reports_dir, furl)
        self.coherence_prestim_bias_lambda_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_coherence_prestim_bias['lambda']['anode'],yerr=std_coherence_prestim_bias['lambda']['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_coherence_prestim_bias['lambda']['cathode'],yerr=std_coherence_prestim_bias['lambda']['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Coherence - Prestim Bias: lambda')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.coherence_prestim_bias_lambda_anova=''#twoway_interaction(coherence_prestim_bias_groups['lambda'],'stim gain','condition','html')
        self.coherence_prestim_bias_lambda_stats=twoway_interaction_r('coherence_prestim_bias_lambda',['stim_intensity','condition'],coherence_prestim_bias_groups['lambda'])
        self.coherence_prestim_bias_lambda_pairwise=pairwise_comparisons(self.coherence_prestim_bias['lambda'],self.stim_gains,['anode','cathode'])

        mean_prestim_bias_rt={'offset':{'anode':[],'cathode':[]},'slope':{'anode':[],'cathode':[]}}
        std_prestim_bias_rt={'offset':{'anode':[],'cathode':[]},'slope':{'anode':[],'cathode':[]}}
        for stim_gain in self.stim_gains:
            for param in mean_prestim_bias_rt:
                for condition in mean_prestim_bias_rt[param]:
                    mean_prestim_bias_rt[param][condition].append(np.mean(self.prestim_bias_rt[param][condition][stim_gain]))
                    std_prestim_bias_rt[param][condition].append(np.std(self.prestim_bias_rt[param][condition][stim_gain])/np.sqrt(len(self.prestim_bias_rt[param][condition][stim_gain])))

        furl='img/prestim_bias_rt_offset'
        fname=os.path.join(self.reports_dir, furl)
        self.prestim_bias_rt_offset_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_prestim_bias_rt['offset']['anode'],yerr=std_prestim_bias_rt['offset']['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_prestim_bias_rt['offset']['cathode'],yerr=std_prestim_bias_rt['offset']['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([0, 1000])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Prestim Bias - RT: offset')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.prestim_bias_rt_offset_anova=''#twoway_interaction(prestim_bias_rt_groups['offset'],'stim gain','condition','html')
        self.prestim_bias_rt_offset_stats=twoway_interaction_r('prestim_bias_rt_offset',['stim_intensity','condition'],prestim_bias_rt_groups['offset'])
        self.prestim_bias_rt_offset_pairwise=pairwise_comparisons(self.prestim_bias_rt['offset'],self.stim_gains,['anode','cathode'])

        furl='img/prestim_bias_rt_slope'
        fname=os.path.join(self.reports_dir, furl)
        self.prestim_bias_rt_slope_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_prestim_bias_rt['slope']['anode'],yerr=std_prestim_bias_rt['slope']['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_prestim_bias_rt['slope']['cathode'],yerr=std_prestim_bias_rt['slope']['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([-300, 300])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Prestim Bias - RT: slope')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.prestim_bias_rt_slope_anova=''#twoway_interaction(prestim_bias_rt_groups['slope'],'stim gain','condition','html')
        self.prestim_bias_rt_slope_stats=twoway_interaction_r('prestim_bias_rt_slope',['stim_intensity','condition'],prestim_bias_rt_groups['slope'])
        self.prestim_bias_rt_slope_pairwise=pairwise_comparisons(self.prestim_bias_rt['slope'],self.stim_gains,['anode','cathode'])
        
        mean_logistic_coeff={'bias':{'anode':[],'cathode':[]},'ev diff':{'anode':[],'cathode':[]}}
        std_logistic_coeff={'bias':{'anode':[],'cathode':[]},'ev diff':{'anode':[],'cathode':[]}}
        mean_small_input_diff_logistic_coeff={'bias':{'anode':[],'cathode':[]},'ev diff':{'anode':[],'cathode':[]}}
        std_small_input_diff_logistic_coeff={'bias':{'anode':[],'cathode':[]},'ev diff':{'anode':[],'cathode':[]}}
        mean_large_input_diff_logistic_coeff={'bias':{'anode':[],'cathode':[]},'ev diff':{'anode':[],'cathode':[]}}
        std_large_input_diff_logistic_coeff={'bias':{'anode':[],'cathode':[]},'ev diff':{'anode':[],'cathode':[]}}
        for stim_gain in self.stim_gains:
            for param in mean_logistic_coeff:
                for condition in mean_logistic_coeff[param]:
                    mean_logistic_coeff[param][condition].append(np.mean(self.logistic_coeff[param][condition][stim_gain]))
                    std_logistic_coeff[param][condition].append(np.std(self.logistic_coeff[param][condition][stim_gain])/np.sqrt(len(self.logistic_coeff[param][condition][stim_gain])))

                    mean_small_input_diff_logistic_coeff[param][condition].append(np.mean(self.small_input_diff_logistic_coeff[param][condition][stim_gain]))
                    std_small_input_diff_logistic_coeff[param][condition].append(np.std(self.small_input_diff_logistic_coeff[param][condition][stim_gain])/np.sqrt(len(self.small_input_diff_logistic_coeff[param][condition][stim_gain])))

                    mean_large_input_diff_logistic_coeff[param][condition].append(np.mean(self.large_input_diff_logistic_coeff[param][condition][stim_gain]))
                    std_large_input_diff_logistic_coeff[param][condition].append(np.std(self.large_input_diff_logistic_coeff[param][condition][stim_gain])/np.sqrt(len(self.large_input_diff_logistic_coeff[param][condition][stim_gain])))

        furl='img/logistic_coeff_bias'
        fname=os.path.join(self.reports_dir, furl)
        self.logistic_coeff_bias_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_logistic_coeff['bias']['anode'],yerr=std_logistic_coeff['bias']['anode'],
            fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_logistic_coeff['bias']['cathode'],yerr=std_logistic_coeff['bias']['cathode'],
            fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([0,12])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Logistic Coefficient: bias')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.logistic_coeff_bias_anova=twoway_interaction(logistic_coeff_groups['bias'],'stim gain','condition','html')
        self.logistic_coeff_bias_stats=twoway_interaction_r('logistic_coeff_bias',['stim_intensity','condition'],logistic_coeff_groups['bias'])
        self.logistic_coeff_bias_pairwise=pairwise_comparisons(self.logistic_coeff['bias'],self.stim_gains,['anode','cathode'])

        furl='img/logistic_coeff_ev_diff'
        fname=os.path.join(self.reports_dir, furl)
        self.logistic_coeff_ev_diff_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_logistic_coeff['ev diff']['anode'],yerr=std_logistic_coeff['ev diff']['anode'],
            fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_logistic_coeff['ev diff']['cathode'],yerr=std_logistic_coeff['ev diff']['cathode'],
            fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([0,12])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Logistic Coefficient: Input diff')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.logistic_coeff_ev_diff_anova=twoway_interaction(logistic_coeff_groups['ev diff'],'stim gain','condition','html')
        self.logistic_coeff_ev_diff_stats=twoway_interaction_r('logistic_coeff_ev_diff',['stim_intensity','condition'],logistic_coeff_groups['ev diff'])
        self.logistic_coeff_ev_diff_pairwise=pairwise_comparisons(self.logistic_coeff['ev diff'],self.stim_gains,['anode','cathode'])

        furl='img/small_input_diff_logistic_coeff_bias'
        fname=os.path.join(self.reports_dir, furl)
        self.small_input_diff_logistic_coeff_bias_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_small_input_diff_logistic_coeff['bias']['anode'],yerr=std_small_input_diff_logistic_coeff['bias']['anode'],
            fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_small_input_diff_logistic_coeff['bias']['cathode'],yerr=std_small_input_diff_logistic_coeff['bias']['cathode'],
            fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([0,12])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Logistic Coefficient: bias')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.small_input_diff_logistic_coeff_bias_anova=twoway_interaction(small_input_diff_logistic_coeff_groups['bias'],'stim gain','condition','html')
        self.small_input_diff_logistic_coeff_bias_stats=twoway_interaction_r('small_input_diff_logistic_coeff_bias',['stim_intensity','condition'],small_input_diff_logistic_coeff_groups['bias'])
        self.small_input_diff_logistic_coeff_bias_pairwise=pairwise_comparisons(self.small_input_diff_logistic_coeff['bias'],self.stim_gains,['anode','cathode'])

        furl='img/small_input_diff_logistic_coeff_ev_diff'
        fname=os.path.join(self.reports_dir, furl)
        self.small_input_diff_logistic_coeff_ev_diff_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_small_input_diff_logistic_coeff['ev diff']['anode'],yerr=std_small_input_diff_logistic_coeff['ev diff']['anode'],
            fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_small_input_diff_logistic_coeff['ev diff']['cathode'],yerr=std_small_input_diff_logistic_coeff['ev diff']['cathode'],
            fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([0,12])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Logistic Coefficient: Input diff')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.small_input_diff_logistic_coeff_ev_diff_anova=twoway_interaction(small_input_diff_logistic_coeff_groups['ev diff'],'stim gain','condition','html')
        self.small_input_diff_logistic_coeff_ev_diff_stats=twoway_interaction_r('small_input_diff_logistic_coeff_ev_diff',['stim_intensity','condition'],small_input_diff_logistic_coeff_groups['ev diff'])
        self.small_input_diff_logistic_coeff_ev_diff_pairwise=pairwise_comparisons(self.small_input_diff_logistic_coeff['ev diff'],self.stim_gains,['anode','cathode'])

        furl='img/large_input_diff_logistic_coeff_bias'
        fname=os.path.join(self.reports_dir, furl)
        self.large_input_diff_logistic_coeff_bias_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_large_input_diff_logistic_coeff['bias']['anode'],yerr=std_large_input_diff_logistic_coeff['bias']['anode'],
            fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_large_input_diff_logistic_coeff['bias']['cathode'],yerr=std_large_input_diff_logistic_coeff['bias']['cathode'],
            fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([0,12])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Logistic Coefficient: bias')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.large_input_diff_logistic_coeff_bias_anova=twoway_interaction(large_input_diff_logistic_coeff_groups['bias'],'stim gain','condition','html')
        self.large_input_diff_logistic_coeff_bias_stats=twoway_interaction_r('large_input_diff_logistic_coeff_bias',['stim_intensity','condition'],large_input_diff_logistic_coeff_groups['bias'])
        self.large_input_diff_logistic_coeff_bias_pairwise=pairwise_comparisons(self.large_input_diff_logistic_coeff['bias'],self.stim_gains,['anode','cathode'])

        furl='img/large_input_diff_logistic_coeff_ev_diff'
        fname=os.path.join(self.reports_dir, furl)
        self.large_input_diff_logistic_coeff_ev_diff_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_large_input_diff_logistic_coeff['ev diff']['anode'],yerr=std_large_input_diff_logistic_coeff['ev diff']['anode'],
            fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_large_input_diff_logistic_coeff['ev diff']['cathode'],yerr=std_large_input_diff_logistic_coeff['ev diff']['cathode'],
            fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([0,12])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Logistic Coefficient: Input diff')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.large_input_diff_logistic_coeff_ev_diff_anova=twoway_interaction(large_input_diff_logistic_coeff_groups['ev diff'],'stim gain','condition','html')
        self.large_input_diff_logistic_coeff_ev_diff_stats=twoway_interaction_r('large_input_diff_logistic_coeff_ev_diff',['stim_intensity','condition'],large_input_diff_logistic_coeff_groups['ev diff'])
        self.large_input_diff_logistic_coeff_ev_diff_pairwise=pairwise_comparisons(self.large_input_diff_logistic_coeff['ev diff'],self.stim_gains,['anode','cathode'])
        
        furl='img/perc_no_response'
        fname=os.path.join(self.reports_dir, furl)
        self.perc_no_response_url='%s.png' % furl
        fig=plt.figure()
        mean_perc_no_response={'anode':[],'cathode':[]}
        std_perc_no_response={'anode':[],'cathode':[]}
        for stim_gain in sorted(self.stim_gains):
                for condition in mean_perc_no_response:
                    mean_perc_no_response[condition].append(np.mean(self.perc_no_response[condition][stim_gain]))
                    std_perc_no_response[condition].append(np.std(self.perc_no_response[condition][stim_gain])/np.sqrt(len(self.perc_no_response[condition][stim_gain])))
        ax=fig.add_subplot(1,1,1)
        ind=np.array(range(len(self.stim_gains)))+1.0
        width=0.4
        rects=[]
        for idx,stim_condition in enumerate(['anode','cathode']):
            rect=ax.bar(ind-idx*width+.5, mean_perc_no_response[stim_condition], width,
                        yerr=std_perc_no_response[stim_condition], ecolor='k', color=condition_colors[stim_condition])
            rects.append(rect)
        ax.set_xticks(ind+width)
        ax.set_xticklabels([str(x) for x in sorted(self.stim_gains)])
        ax.legend([rect[0] for rect in rects],['anode','cathode'],loc='best')
        ax.set_xlabel('Stimulation Gain')
        ax.set_ylabel('% No Response')
        ax.set_ylim([0,.2])
        ax.set_xlim([0,len(self.stim_gains)+1])
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.perc_no_response_anova=twoway_interaction(perc_no_response_groups,'stim gain','condition','html')
        self.perc_no_response_stats=twoway_interaction_r('perc_no_response',['stim_intensity','condition'],perc_no_response_groups)
        self.perc_no_response_pairwise=pairwise_comparisons(self.perc_no_response,self.stim_gains,['anode','cathode'])

        mean_bias_perc_left_params={'k':{'anode':[],'cathode':[]}}
        std_bias_perc_left_params={'k':{'anode':[],'cathode':[]}}
        for stim_gain in self.stim_gains:
            mean_bias_perc_left_params['k']['anode'].append(np.mean(self.bias_perc_left_params['k']['anode'][stim_gain]))
            std_bias_perc_left_params['k']['anode'].append(np.std(self.bias_perc_left_params['k']['anode'][stim_gain])/np.sqrt(len(self.bias_perc_left_params['k']['anode'][stim_gain])))
            mean_bias_perc_left_params['k']['cathode'].append(np.mean(self.bias_perc_left_params['k']['cathode'][stim_gain]))
            std_bias_perc_left_params['k']['cathode'].append(np.std(self.bias_perc_left_params['k']['cathode'][stim_gain])/np.sqrt(len(self.bias_perc_left_params['k']['cathode'][stim_gain])))

        furl='img/bias_perc_left_k'
        fname=os.path.join(self.reports_dir, furl)
        self.bias_perc_left_k_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.stim_gains, mean_bias_perc_left_params['k']['anode'],yerr=std_bias_perc_left_params['k']['anode'],
                     fmt='o%s' % condition_colors['anode'],label='anode')
        plt.errorbar(self.stim_gains, mean_bias_perc_left_params['k']['cathode'],yerr=std_bias_perc_left_params['k']['cathode'],
                     fmt='o%s' % condition_colors['cathode'],label='cathode')
        plt.xlim([0,np.max(self.stim_gains)+.5])
        plt.ylim([-4,4])
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Bias - % Left: n')
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)
        #self.bias_perc_left_k_anova=twoway_interaction(bias_perc_left_groups['k'],'stim gain','condition','html')
        self.bias_perc_left_k_stats=twoway_interaction_r('bias_perc_left_k',['stim_intensity','condition'],bias_perc_left_groups['k'])
        self.bias_perc_left_k_pairwise=pairwise_comparisons(self.bias_perc_left_params,self.stim_gains,['anode','cathode'])

        self.wta_params=self.stim_level_reports[self.stim_level_reports.keys()[0]].wta_params
        self.pyr_params=self.stim_level_reports[self.stim_level_reports.keys()[0]].pyr_params
        self.inh_params=self.stim_level_reports[self.stim_level_reports.keys()[0]].inh_params
        self.voxel_params=self.stim_level_reports[self.stim_level_reports.keys()[0]].voxel_params
        self.num_groups=self.stim_level_reports[self.stim_level_reports.keys()[0]].num_groups
        self.trial_duration=self.stim_level_reports[self.stim_level_reports.keys()[0]].trial_duration
        self.stim_start_time=self.stim_level_reports[self.stim_level_reports.keys()[0]].stim_start_time
        self.stim_end_time=self.stim_level_reports[self.stim_level_reports.keys()[0]].stim_end_time
        self.network_group_size=self.stim_level_reports[self.stim_level_reports.keys()[0]].network_group_size
        self.background_input_size=self.stim_level_reports[self.stim_level_reports.keys()[0]].background_input_size
        self.task_input_size=self.stim_level_reports[self.stim_level_reports.keys()[0]].task_input_size
        
        #create report
        template_file='dcs_param_explore.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        output_file='dcs_param_explore.html'
        fname=os.path.join(self.reports_dir,output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)


if __name__=='__main__':
    report=ParamExploreReport('/data/pySBI/rdmd/virtual_subjects_param_explore',
        'wta.groups.2.duration.4.000.p_e_e.0.080.p_e_i.0.100.p_i_i.0.100.p_i_e.0.200', range(20), 20,
        '/data/pySBI/reports/rdmd/virtual_subjects_param_explore', control=False)
    # report=ParamExploreReport('/data/pySBI/rdmd/virtual_subjects_param_explore_control',
    #     'wta.groups.2.duration.4.000.p_e_e.0.080.p_e_i.0.100.p_i_i.0.100.p_i_e.0.200', range(20), 20,
    #     '/data/pySBI/reports/rdmd/virtual_subjects_param_explore_control', control=True, stim_gains=[10,8,6,4,2,1,0.5,0.25])
    report.create_report(regenerate_stim_level_plots=True, regenerate_subject_plots=True, regenerate_session_plots=False,
                         regenerate_trial_plots=False)
