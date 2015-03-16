import os
import subprocess
from jinja2 import Environment, FileSystemLoader
from pysbi.config import TEMPLATE_DIR
from pysbi.reports.utils import make_report_dirs
from pysbi.wta.dcs.analysis import DCSComparisonReport

class ParamExploreReport():
    def __init__(self, data_dir, file_prefix, virtual_subj_ids, num_trials, reports_dir):
        self.data_dir=data_dir
        self.file_prefix=file_prefix
        self.virtual_subj_ids=virtual_subj_ids
        self.num_trials=num_trials
        self.reports_dir=reports_dir
        self.stim_gains=[8,4,2,1,0.5,0.25]
        self.stim_level_reports={}

    def create_report(self, regenerate_stim_level_plots=False, regenerate_subject_plots=False,
                      regenerate_session_plots=False, regenerate_trial_plots=False):

        make_report_dirs(self.reports_dir)

        self.version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        for stim_gain in self.stim_gains:
            report_dir=os.path.join(self.reports_dir,'level_%d' % stim_gain)
            self.stim_level_reports[stim_gain]=DCSComparisonReport(self.data_dir,
                self.file_prefix,self.virtual_subj_ids,
                {'control':(0,0),'anode':(1.0*stim_gain,-0.5*stim_gain), 'cathode':(-1.0*stim_gain,0.5*stim_gain)},
                self.num_trials, report_dir, '', contrast_range=(0.0, .032, .064, .128, .256, .512))
            self.stim_level_reports[stim_gain].create_report(regenerate_plots=regenerate_stim_level_plots,
                regenerate_subject_plots=regenerate_subject_plots, regenerate_session_plots=regenerate_session_plots,
                regenerate_trial_plots=regenerate_trial_plots)

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
        'wta.groups.2.duration.4.000.p_e_e.0.080.p_e_i.0.100.p_i_i.0.100.p_i_e.0.200', range(10), 20,
        '/data/pySBI/reports/rdmd/virtual_subjects_param_explore')
    report.create_report(regenerate_stim_level_plots=True, regenerate_subject_plots=True,
        regenerate_session_plots=True, regenerate_trial_plots=True)