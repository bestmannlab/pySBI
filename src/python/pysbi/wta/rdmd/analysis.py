import h5py
import os
import numpy as np
from pysbi.util.utils import mdm_outliers

def export_behavioral_data_long(subj_ids, conditions, data_dir, output_filename):
    out_file=open(output_filename,'w')
    out_file.write('Subject,Condition,Trial,Direction,Coherence,Resp,LastResp,Correct,RT\n')
    for subj_id in subj_ids:
        for condition in conditions:
            f = h5py.File(os.path.join(data_dir,'subject.%d.%s.h5' % (subj_id,condition)),'r')

            f_network_params=f['network_params']
            mu_0=float(f_network_params.attrs['mu_0'])
            p_a=float(f_network_params.attrs['p_a'])

            f_behav=f['behavior']
            trial_rt=np.array(f_behav['trial_rt'])
            trial_resp=np.array(f_behav['trial_resp'])
            trial_correct=np.array(f_behav['trial_correct'])
            f_neur=f['neural']
            trial_inputs=np.array(f_neur['trial_inputs'])

            trial_data=[]
            last_resp=float('NaN')

            for trial_idx in range(trial_rt.shape[1]):
                direction=np.where(trial_inputs[:,trial_idx]==np.max(trial_inputs[:,trial_idx]))[0][0]
                if direction==0:
                    direction=-1
                coherence=(trial_inputs[0,trial_idx]-mu_0)/(p_a*100.0)
                correct=int(trial_correct[0,trial_idx])
                resp=int(trial_resp[0,trial_idx])
                if resp==0:
                    resp=-1
                rt=trial_rt[0,trial_idx]

                if rt>100:
                    trial_data.append([trial_idx, direction, coherence, correct, resp, last_resp, rt])
                last_resp=resp
            trial_data=np.array(trial_data)
            outliers=mdm_outliers(trial_data[:,6])
            filtered_trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]
            for i in range(filtered_trial_data.shape[0]):
                out_file.write('%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (subj_id, condition,
                                                                             int(filtered_trial_data[i,0]),
                                                                             filtered_trial_data[i,1],
                                                                             filtered_trial_data[i,2],
                                                                             filtered_trial_data[i,4],
                                                                             filtered_trial_data[i,5],
                                                                             filtered_trial_data[i,3],
                                                                             filtered_trial_data[i,6]))



if __name__=='__main__':
#    export_behavioral_data_long(range(20),['control','depolarizing','hyperpolarizing'],
#        '/home/jbonaiuto/Projects/pySBI/data/rdmd','/home/jbonaiuto/Projects/pySBI/data/rdmd/behav_data_long.csv')
    export_behavioral_data_long([14],['control','depolarizing','hyperpolarizing'],
        '/home/jbonaiuto/Projects/pySBI/data/rdmd','/home/jbonaiuto/Projects/pySBI/data/rdmd/behav_data_long.csv')