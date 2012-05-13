from glob import glob
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
import os
from brian.stdunits import nA, mA, Hz, ms
import matplotlib.pylab as plt
import numpy as np
from shutil import copytree, copyfile
from pysbi.analysis import FileInfo, get_roc_single_option, get_auc, get_bold_signal, get_auc_single_option, get_lfp_signal
from pysbi.config import TEMPLATE_DIR
from pysbi.utils import save_to_png, Struct

def create_all_reports(data_dir, num_groups, trial_duration, p_b_e_range, p_x_e_range, p_e_e_range, p_e_i_range, p_i_i_range,
                       p_i_e_range, num_trials, base_report_dir):
    template_file='wta_network.html'
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template=env.get_template(template_file)

    for p_b_e in p_b_e_range:
        for p_x_e in p_x_e_range:
            for p_e_e in p_e_e_range:
                for p_e_i in p_e_i_range:
                    for p_i_i in p_i_i_range:
                        for p_i_e in p_i_e_range:
                            file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f' %\
                                      (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e)
                            file_prefix=os.path.join(data_dir,file_desc)
                            reports_dir=os.path.join(base_report_dir,file_desc)
                            if os.path.exists('%s.trial.0.h5' % file_prefix):
                                print('Creating report for %s' % file_desc)
                                create_wta_network_report(file_prefix, num_trials, reports_dir, template)


def create_wta_network_report(file_prefix, num_trials, reports_dir, template):
    make_report_dirs(reports_dir)

    report_info=Struct()
    report_info.trials=[]

    file_name='%s.trial.%d.h5' % (file_prefix, 0)
    data=FileInfo(file_name)

    report_info.wta_params=data.wta_params
    report_info.voxel_params=data.voxel_params
    report_info.num_groups=data.num_groups
    report_info.trial_duration=data.trial_duration
    report_info.background_rate=data.background_rate
    report_info.stim_start_time=data.stim_start_time
    report_info.stim_end_time=data.stim_end_time
    report_info.network_group_size=data.network_group_size
    report_info.background_input_size=data.background_input_size
    report_info.task_input_size=data.task_input_size

    (data_dir, data_file_prefix) = os.path.split(file_prefix)

    trial_bold=[]
    trial_max_input=np.zeros(num_trials)
    trial_max_rate=np.zeros(num_trials)
    for i in range(num_trials):
        file_name='%s.trial.%d.h5' % (file_prefix, i)
        data=FileInfo(file_name)
        bold_signal=get_bold_signal(data, [500, 2500])
        trial = create_trial_report(data, reports_dir, bold_signal, i)
        trial_bold.append(bold_signal)
        trial_max_input[i]=trial.max_input
        trial_max_rate[i]=trial.max_rate
        report_info.trials.append(trial)

    fig=plt.figure()
    plt.plot(trial_max_input, trial_max_rate, 'x')
    plt.xlabel('Max Input Rate')
    plt.ylabel('Max Population Rate')
    furl='img/input_output_rate.png'
    fname=os.path.join(reports_dir, furl)
    save_to_png(fig, fname)
    report_info.input_output_rate_url=furl
    plt.clf()

    report_info.contrast_bold_url=create_bold_report(reports_dir, trial_bold, file_prefix, num_trials)

    report_info.roc=create_roc_report(file_prefix, data.num_groups, num_trials, reports_dir)

    #create report
    output_file='wta_network.%s.html' % data_file_prefix
    fname=os.path.join(reports_dir,output_file)
    stream=template.stream(rinfo=report_info)
    stream.dump(fname)


def create_bold_report(reports_dir, trial_bold, file_prefix, num_trials):
    diff=[]
    max_bold=[]
    for i in range(num_trials):
        file_name='%s.trial.%d.h5' % (file_prefix, i)
        data=FileInfo(file_name)
        diff.append(abs(data.input_freq[0]-data.input_freq[1]))
        max_bold.append(max(trial_bold[i][0]))
    fig=plt.figure()
    plt.plot(np.array(diff), np.array(max_bold), 'x')
    plt.xlabel('Input Contrast')
    plt.ylabel('Max BOLD')
    furl='img/contrast_bold.png'
    fname=os.path.join(reports_dir, furl)
    save_to_png(fig, fname)
    plt.clf()
    return furl

def create_trial_report(data, reports_dir, bold_signal, trial_idx):
    trial = Struct()
    trial.input_freq=data.input_freq

    max_input_idx=np.where(trial.input_freq==np.max(trial.input_freq))[0][0]
    trial.max_input=max_input_idx
    trial.max_rate=np.max(data.e_firing_rates[max_input_idx])

    trial.e_raster_url = None
    trial.i_raster_url = None
    if data.e_spike_neurons is not None and data.i_spike_neurons is not None:
        e_group_sizes=[int(4*data.network_group_size/5) for i in range(data.num_groups)]
        fig=plot_raster(data.e_spike_neurons, data.e_spike_times, e_group_sizes)
        furl='img/e_raster.trial.%d.png' % trial_idx
        trial.e_raster_url = furl
        fname=os.path.join(reports_dir, furl)
        save_to_png(fig, fname)
        plt.clf()

        i_group_sizes=[int(data.network_group_size/5) for i in range(data.num_groups)]
        fig=plot_raster(data.i_spike_neurons, data.i_spike_times, i_group_sizes)
        furl='img/i_raster.trial.%d.png' % trial_idx
        trial.i_raster_url = furl
        fname=os.path.join(reports_dir, furl)
        save_to_png(fig, fname)
        plt.clf()

    trial.firing_rate_url = None
    if data.e_firing_rates is not None and data.i_firing_rates is not None:
        fig = plt.figure()
        ax = plt.subplot(211)
        for i, pop_rate in enumerate(data.e_firing_rates):
            ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        ax = plt.subplot(212)
        for i, pop_rate in enumerate(data.i_firing_rates):
            ax.plot(np.array(range(len(pop_rate))) *.1, pop_rate / Hz, label='group %d' % i)
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        furl = 'img/firing_rate.trial.%d.png' % trial_idx
        trial.firing_rate_url = furl
        fname = os.path.join(reports_dir, furl)
        save_to_png(fig, fname)
        plt.clf()

    trial.neural_state_url=None
    if data.neural_state_rec is not None:
        fig = plt.figure()
        for i in range(data.num_groups):
            times=np.array(range(len(data.neural_state_rec['g_ampa_r'][i*2])))*.1
            ax = plt.subplot(data.num_groups * 100 + 20 + (i * 2 + 1))
            ax.plot(times, data.neural_state_rec['g_ampa_r'][i * 2] / nA, label='AMPA-recurrent')
            ax.plot(times, data.neural_state_rec['g_ampa_x'][i * 2] / nA, label='AMPA-task')
            ax.plot(times, data.neural_state_rec['g_ampa_b'][i * 2] / nA, label='AMPA-backgrnd')
            ax.plot(times, data.neural_state_rec['g_nmda'][i * 2] / nA, label='NMDA')
            ax.plot(times, data.neural_state_rec['g_gaba_a'][i * 2] / nA, label='GABA_A')
            #ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][0]/nA, label='GABA_B')
            plt.xlabel('Time (ms)')
            plt.ylabel('Conductance (nA)')
            ax = plt.subplot(data.num_groups * 100 + 20 + (i * 2 + 2))
            ax.plot(times, data.neural_state_rec['g_ampa_r'][i * 2 + 1] / nA, label='AMPA-recurrent')
            ax.plot(times, data.neural_state_rec['g_ampa_x'][i * 2 + 1] / nA, label='AMPA-task')
            ax.plot(times, data.neural_state_rec['g_ampa_b'][i * 2 + 1] / nA, label='AMPA-backgrnd')
            ax.plot(times, data.neural_state_rec['g_nmda'][i * 2 + 1] / nA, label='NMDA')
            ax.plot(times, data.neural_state_rec['g_gaba_a'][i * 2 + 1] / nA, label='GABA_A')
            #ax.plot(self.network_monitor['g_gaba_b'].times/ms, self.network_monitor['g_gaba_b'][0]/nA, label='GABA_B')
            plt.xlabel('Time (ms)')
            plt.ylabel('Conductance (nA)')
        furl = 'img/neural_state.trial.%d.png' % trial_idx
        trial.neural_state_url = furl
        fname = os.path.join(reports_dir, furl)
        save_to_png(fig, fname)
        plt.clf()

    trial.lfp_url = None
    if data.lfp_rec is not None:
        fig = plt.figure()
        ax = plt.subplot(111)
        lfp=get_lfp_signal(data)
        ax.plot(np.array(range(len(lfp))), lfp / mA)
        plt.xlabel('Time (ms)')
        plt.ylabel('LFP (mA)')
        furl = 'img/lfp.trial.%d.png' % trial_idx
        trial.lfp_url = furl
        fname = os.path.join(reports_dir, furl)
        save_to_png(fig, fname)
        plt.clf()

    trial.voxel_url = None
    if data.voxel_rec is not None:
        fig = plt.figure()
        ax = plt.subplot(211)
        ax.plot(np.array(range(len(data.voxel_rec['G_total'][0])))*.1, data.voxel_rec['G_total'][0] / nA)
        plt.xlabel('Time (ms)')
        plt.ylabel('Total Synaptic Activity (nA)')
        ax = plt.subplot(212)
        ax.plot(np.array(range(len(bold_signal[0])))*.1*ms, bold_signal[0])
        plt.xlabel('Time (s)')
        plt.ylabel('BOLD')
        furl = 'img/voxel.trial.%d.png' % trial_idx
        trial.voxel_url = furl
        fname = os.path.join(reports_dir, furl)
        save_to_png(fig, fname)
        plt.clf()
    return trial


def create_roc_report(file_prefix, num_groups, num_trials, reports_dir):
    roc_report=Struct()
    roc_report.auc=get_auc(file_prefix, num_trials, num_groups)
    roc_report.auc_single_option=[]
    fig=plt.figure()
    for i in range(num_groups):
        roc=get_roc_single_option(file_prefix, num_trials, i)
        plt.plot(roc[:,0],roc[:,1],'x-',label='option %d' % i)
        roc_report.auc_single_option.append(get_auc_single_option(file_prefix, num_trials, i))
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_url = 'img/roc.png'
    fname = os.path.join(reports_dir, roc_url)
    save_to_png(fig, fname)
    plt.clf()
    roc_report.roc_url=roc_url
    return roc_report

def make_report_dirs(output_dir):

    rdirs = ['img']
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except Exception:
            print 'Could not make directory %s' % output_dir

    for d in rdirs:
        dname = os.path.join(output_dir, d)
        if not os.path.exists(dname):
            try:
                os.mkdir(dname)
            except Exception:
                print 'Could not make directory %s' % dname

    dirs_to_copy = ['js', 'css']
    for d in dirs_to_copy:
        srcdir = os.path.join(TEMPLATE_DIR, d)
        destdir = os.path.join(output_dir, d)
        if not os.path.exists(destdir):
            try:
                copytree(srcdir, destdir)
            except Exception:
                print 'Problem copying %s to %s' % (srcdir, destdir)

    imgfiles = glob(os.path.join(TEMPLATE_DIR, '*.gif'))
    for ipath in imgfiles:
        [rootdir, ifile] = os.path.split(ipath)
        destfile = os.path.join(output_dir, ifile)
        if not os.path.exists(destfile):
            copyfile(ipath, destfile)

def plot_raster(group_spike_neurons, group_spike_times, group_sizes):
    if len(group_spike_times) and len(group_spike_neurons)==len(group_spike_times):
        spacebetween = .1
        allsn = []
        allst = []
        for i, spike_times in enumerate(group_spike_times):
            mspikes=zip(group_spike_neurons[i],group_spike_times[i])

            if len(mspikes):
                sn, st = np.array(mspikes).T
            else:
                sn, st = np.array([]), np.array([])
            st /= ms
            allsn.append(i + ((1. - spacebetween) / float(group_sizes[i])) * sn)
            allst.append(st)
        sn = np.hstack(allsn)
        st = np.hstack(allst)
        fig=plt.figure()
        plt.plot(st, sn, '.')
        plt.ylabel('Group number')
        plt.xlabel('Time (ms)')
        return fig
