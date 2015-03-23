import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import subprocess
from jinja2 import Environment, FileSystemLoader
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.util.utils import save_to_png, save_to_eps
from pysbi.wta.dcs.analysis import DCSComparisonReport

class ParamExploreReport():
    def __init__(self, data_dir, file_prefix, virtual_subj_ids, num_trials, reports_dir):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.virtual_subj_ids=virtual_subj_ids
        self.num_trials=num_trials
        self.reports_dir=reports_dir
        self.stim_gains=[8,7,6,5,4,3,2,1,0.5,0.25]
        #self.stim_gains=[4,2,1,0.5,0.25]
        self.stim_level_reports={}

    def create_report(self, regenerate_stim_level_plots=False, regenerate_subject_plots=False,
                      regenerate_session_plots=False, regenerate_trial_plots=False):

        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])

        self.thresh_difference={'anode':[],'cathode':[]}
        self.rt_diff_slope={'anode':[],'cathode':[]}
        self.prestim_bias_diff={'anode':[],'cathode':[]}
        self.coherence_prestim_bias_diff={'n':{'anode':[],'cathode':[]},'lambda':{'anode':[],'cathode':[]}}
        self.prestim_bias_rt_diff={'offset':{'anode':[], 'cathode':[]},'slope':{'anode':[], 'cathode':[]}}
        
        for stim_gain in self.stim_gains:
            report_dir=os.path.join(self.reports_dir,'level_%.2f' % stim_gain)
            self.stim_level_reports[stim_gain]=DCSComparisonReport(self.data_dir,
                self.file_prefix,self.virtual_subj_ids,
                {'control':(0,0),'anode':(1.0*stim_gain,-0.5*stim_gain), 'cathode':(-1.0*stim_gain,0.5*stim_gain)},
                self.num_trials, report_dir, '', contrast_range=(0.0, .032, .064, .128, .256, .512))
            self.stim_level_reports[stim_gain].create_report(regenerate_plots=regenerate_stim_level_plots,
                regenerate_subject_plots=regenerate_subject_plots, regenerate_session_plots=regenerate_session_plots,
                regenerate_trial_plots=regenerate_trial_plots)
            for subj in self.stim_level_reports[stim_gain].subjects:
                self.stim_level_reports[stim_gain].subjects[subj].sessions={}
            self.thresh_difference['anode'].append(self.stim_level_reports[stim_gain].thresh['anode']-self.stim_level_reports[stim_gain].thresh['control'])
            self.thresh_difference['cathode'].append(self.stim_level_reports[stim_gain].thresh['cathode']-self.stim_level_reports[stim_gain].thresh['control'])
            self.rt_diff_slope['anode'].append(self.stim_level_reports[stim_gain].rt_diff_slope['anode'])
            self.rt_diff_slope['cathode'].append(self.stim_level_reports[stim_gain].rt_diff_slope['cathode'])
            self.prestim_bias_diff['anode'].append(self.stim_level_reports[stim_gain].mean_biases['anode']-self.stim_level_reports[stim_gain].mean_biases['control'])
            self.prestim_bias_diff['cathode'].append(self.stim_level_reports[stim_gain].mean_biases['cathode']-self.stim_level_reports[stim_gain].mean_biases['control'])
            self.coherence_prestim_bias_diff['n']['anode'].append(self.stim_level_reports[stim_gain].coherence_prestim_bias_params['n']['anode']-self.stim_level_reports[stim_gain].coherence_prestim_bias_params['n']['control'])
            self.coherence_prestim_bias_diff['lambda']['anode'].append(self.stim_level_reports[stim_gain].coherence_prestim_bias_params['lambda']['anode']-self.stim_level_reports[stim_gain].coherence_prestim_bias_params['lambda']['control'])
            self.coherence_prestim_bias_diff['n']['cathode'].append(self.stim_level_reports[stim_gain].coherence_prestim_bias_params['n']['cathode']-self.stim_level_reports[stim_gain].coherence_prestim_bias_params['n']['control'])
            self.coherence_prestim_bias_diff['lambda']['cathode'].append(self.stim_level_reports[stim_gain].coherence_prestim_bias_params['lambda']['cathode']-self.stim_level_reports[stim_gain].coherence_prestim_bias_params['lambda']['control'])
            self.prestim_bias_rt_diff['offset']['anode'].append(self.stim_level_reports[stim_gain].bias_rt_params['offset']['anode']-self.stim_level_reports[stim_gain].bias_rt_params['offset']['control'])
            self.prestim_bias_rt_diff['offset']['cathode'].append(self.stim_level_reports[stim_gain].bias_rt_params['offset']['cathode']-self.stim_level_reports[stim_gain].bias_rt_params['offset']['control'])
            self.prestim_bias_rt_diff['slope']['anode'].append(self.stim_level_reports[stim_gain].bias_rt_params['slope']['anode']-self.stim_level_reports[stim_gain].bias_rt_params['slope']['control'])
            self.prestim_bias_rt_diff['slope']['cathode'].append(self.stim_level_reports[stim_gain].bias_rt_params['slope']['cathode']-self.stim_level_reports[stim_gain].bias_rt_params['slope']['control'])

        furl='img/thresh'
        fname=os.path.join(self.reports_dir, furl)
        self.thresh_url='%s.png' % furl
        fig=plt.figure()
        plt.plot(self.stim_gains,self.thresh_difference['anode'],'ro',label='anode',)
        plt.plot(self.stim_gains,self.thresh_difference['cathode'],'go',label='cathode')
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Threshold difference')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/rt_diff_slope'
        fname=os.path.join(self.reports_dir, furl)
        self.rt_diff_slope_url='%s.png' % furl
        fig=plt.figure()
        plt.plot(self.stim_gains,self.rt_diff_slope['anode'],'ro',label='anode',)
        plt.plot(self.stim_gains,self.rt_diff_slope['cathode'],'go',label='cathode')
        plt.xlabel('Stimulation Gain')
        plt.ylabel('RT Difference slope')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/prestim_bias_diff'
        fname=os.path.join(self.reports_dir, furl)
        self.prestim_bias_diff_url='%s.png' % furl
        fig=plt.figure()
        plt.plot(self.stim_gains,self.prestim_bias_diff['anode'],'ro',label='anode',)
        plt.plot(self.stim_gains,self.prestim_bias_diff['cathode'],'go',label='cathode')
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Prestimulus Bias Difference')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/coherence_prestim_bias_n'
        fname=os.path.join(self.reports_dir, furl)
        self.coherence_prestim_bias_n_url='%s.png' % furl
        fig=plt.figure()
        plt.plot(self.stim_gains,self.coherence_prestim_bias_diff['n']['anode'],'ro',label='anode')
        plt.plot(self.stim_gains,self.coherence_prestim_bias_diff['n']['cathode'],'go',label='cathode')
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Coherence - Prestim Bias: n')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/coherence_prestim_bias_lambda'
        fname=os.path.join(self.reports_dir, furl)
        self.coherence_prestim_bias_lambda_url='%s.png' % furl
        fig=plt.figure()
        plt.plot(self.stim_gains,self.coherence_prestim_bias_diff['lambda']['anode'],'ro',label='anode')
        plt.plot(self.stim_gains,self.coherence_prestim_bias_diff['lambda']['cathode'],'go',label='cathode')
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Coherence - Prestim Bias: lambda')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/prestim_bias_rt_diff_offset'
        fname=os.path.join(self.reports_dir, furl)
        self.prestim_bias_rt_diff_offset_url='%s.png' % furl
        fig=plt.figure()
        plt.plot(self.stim_gains,self.prestim_bias_rt_diff['offset']['anode'],'ro',label='anode')
        plt.plot(self.stim_gains,self.prestim_bias_rt_diff['offset']['cathode'],'go',label='cathode')
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Prestim Bias - RT: offset')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

        furl='img/prestim_bias_rt_diff_slope'
        fname=os.path.join(self.reports_dir, furl)
        self.prestim_bias_rt_diff_slope_url='%s.png' % furl
        fig=plt.figure()
        plt.plot(self.stim_gains,self.prestim_bias_rt_diff['slope']['anode'],'ro',label='anode')
        plt.plot(self.stim_gains,self.prestim_bias_rt_diff['slope']['cathode'],'go',label='cathode')
        plt.xlabel('Stimulation Gain')
        plt.ylabel('Prestim Bias - RT: slope')
        save_to_png(fig, '%s.png' % fname)
        save_to_eps(fig, '%s.eps' % fname)
        plt.close(fig)

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
        '/data/pySBI/reports/rdmd/virtual_subjects_param_explore')
    report.create_report(regenerate_stim_level_plots=True, regenerate_subject_plots=False, regenerate_session_plots=False,
                         regenerate_trial_plots=False)
