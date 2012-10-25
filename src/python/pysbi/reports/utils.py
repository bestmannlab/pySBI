import copy
from exceptions import Exception
from glob import glob
import os
from shutil import copytree, copyfile
import numpy as np
from pysbi.config import TEMPLATE_DIR

def get_local_average(dict, param_array):
    values=[]
    for i,idx in enumerate(param_array):
        new_array=copy.copy(param_array)
        new_array[i]=idx-1
        if tuple(new_array) in dict:
            values.extend(dict[tuple(new_array)])
        new_array=copy.copy(param_array)
        new_array[i]=idx+1
        if tuple(new_array) in dict:
            values.extend(dict[tuple(new_array)])
    return np.mean(values)


def all_trials_exist(file_prefix, contrast_range, num_trials):
    for contrast in contrast_range:
        for i in range(num_trials):
            fname='%s.contrast.%0.4f.trial.%d.h5' % (file_prefix, contrast, i)
            if not os.path.exists(fname):
                return False
    return True


def get_tested_param_combos(data_dir, num_groups, trial_duration, contrast_range, num_trials, edesc):
    param_combos=[]
    for file in os.listdir(data_dir):
        if file.endswith('.h5'):
            file_split=file.split('.')
            p_b_e=float('%s.%s' % (file_split[7],file_split[8]))
            p_x_e=float('%s.%s' % (file_split[10],file_split[11]))
            p_e_e=float('%s.%s' % (file_split[13],file_split[14]))
            p_e_i=float('%s.%s' % (file_split[16],file_split[17]))
            p_i_i=float('%s.%s' % (file_split[19],file_split[20]))
            p_i_e=float('%s.%s' % (file_split[22],file_split[23]))

            file_desc='wta.groups.%d.duration.%0.3f.p_b_e.%0.3f.p_x_e.%0.3f.p_e_e.%0.3f.p_e_i.%0.3f.p_i_i.%0.3f.p_i_e.%0.3f.%s' %\
                      (num_groups, trial_duration, p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e, edesc)
            file_prefix=os.path.join(data_dir,file_desc)
            param_tuple=(p_b_e,p_x_e,p_e_e,p_e_i,p_i_i,p_i_e)
            if not param_tuple in param_combos and all_trials_exist(file_prefix, contrast_range, num_trials):
                param_combos.append(param_tuple)
    param_combos.sort()
    return param_combos


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