from brian.stdunits import Hz
from brian.units import second, farad, siemens, volt
import h5py
import numpy as np
from pysbi.neglect.run import default_params

class FileInfo():
    def __init__(self, file_name):
        self.file_name=file_name
        f = h5py.File(file_name)
        param_group=f['parameters']
        self.background_input_size=int(param_group.attrs['background_input_size'])
        self.background_rate=float(param_group.attrs['background_rate'])*Hz
        self.vc_size=int(param_group.attrs['vc_size'])
        self.vc_rates=np.array(param_group.attrs['vc_rates'])
        self.trial_duration=float(param_group.attrs['trial_duration'])*second
        self.stim_start_time=float(param_group.attrs['stim_start_time'])*second
        self.stim_end_time=float(param_group.attrs['stim_end_time'])*second
        self.go_start_time=float(param_group.attrs['go_start_time'])*second
        self.go_end_time=float(param_group.attrs['go_end_time'])*second
        self.lip_size=int(param_group.attrs['lip_size'])


        self.params=default_params()
        self.params.C=float(param_group.attrs['C'])*farad
        self.params.gL=float(param_group.attrs['gL'])*siemens
        self.params.EL=float(param_group.attrs['EL'])*volt
        self.params.VT=float(param_group.attrs['VT'])*volt
        self.params.DeltaT=float(param_group.attrs['DeltaT'])*volt
        self.params.Mg=float(param_group.attrs['Mg'])
        self.params.E_ampa=float(param_group.attrs['E_ampa'])*volt
        self.params.E_nmda=float(param_group.attrs['E_nmda'])*volt
        self.params.E_gaba_a=float(param_group.attrs['E_gaba_a'])*volt
        self.params.E_gaba_b=float(param_group.attrs['E_gaba_b'])*volt
        self.params.tau_ampa=float(param_group.attrs['tau_ampa'])*second
        self.params.tau1_nmda=float(param_group.attrs['tau1_nmda'])*second
        self.params.tau2_nmda=float(param_group.attrs['tau2_nmda'])*second
        self.params.tau_gaba_a=float(param_group.attrs['tau_gaba_a'])*second
        self.params.tau1_gaba_b=float(param_group.attrs['tau1_gaba_b'])*second
        self.params.tau2_gaba_b=float(param_group.attrs['tau2_gaba_b'])*second

        self.params.w_ampa_min=float(param_group.attrs['w_ampa_min'])*siemens
        self.params.w_ampa_max=float(param_group.attrs['w_ampa_max'])*siemens
        self.params.w_nmda_min=float(param_group.attrs['w_nmda_min'])*siemens
        self.params.w_nmda_max=float(param_group.attrs['w_nmda_max'])*siemens
        self.params.w_gaba_a_min=float(param_group.attrs['w_gaba_a_min'])*siemens
        self.params.w_gaba_a_max=float(param_group.attrs['w_gaba_a_max'])*siemens
        self.params.w_gaba_b_min=float(param_group.attrs['w_gaba_b_min'])*siemens
        self.params.w_gaba_b_max=float(param_group.attrs['w_gaba_b_max'])*siemens
        self.params.p_g_e=float(param_group.attrs['p_g_e'])
        self.params.p_b_e=float(param_group.attrs['p_b_e'])
        self.params.p_v_ec_vis=float(param_group.attrs['p_v_ec_vis'])
        self.params.p_ec_mem_ec_mem=float(param_group.attrs['p_ec_mem_ec_mem'])
        self.params.p_ec_vis_ec_mem=float(param_group.attrs['p_ec_vis_ec_mem'])
        self.params.p_ii_ec=float(param_group.attrs['p_ii_ec'])
        self.params.p_ec_ei=float(param_group.attrs['p_ec_ei'])
        self.params.p_ei_mem_ei_mem=float(param_group.attrs['p_ei_mem_ei_mem'])
        self.params.p_ei_vis_ei_mem=float(param_group.attrs['p_ei_vis_ei_mem'])
        self.params.p_ic_ei=float(param_group.attrs['p_ic_ei'])
        self.params.p_ei_ii=float(param_group.attrs['p_ei_ii'])
        self.params.p_ec_ii=float(param_group.attrs['p_ec_ii'])
        self.params.p_ec_ic=float(param_group.attrs['p_ec_ic'])

        self.lfp_rec=None
        if 'lfp' in f:
            f_lfp=f['lfp']
            self.lfp_rec={'lfp': np.array(f_lfp['lfp'])}

        if 'voxel' in f:
            f_vox=f['voxel']
            self.voxel_params=voxel.default_params()
            self.voxel_params.eta=float(f_vox.attrs['eta'])*second
            self.voxel_params.G_base=float(f_vox.attrs['G_base'])*amp
            self.voxel_params.tau_f=float(f_vox.attrs['tau_f'])*second
            self.voxel_params.tau_s=float(f_vox.attrs['tau_s'])*second
            self.voxel_params.tau_o=float(f_vox.attrs['tau_o'])*second
            self.voxel_params.e_base=float(f_vox.attrs['e_base'])
            self.voxel_params.v_base=float(f_vox.attrs['v_base'])
            self.voxel_params.alpha=float(f_vox.attrs['alpha'])
            if 'T_2E' in f_vox.attrs:
                self.voxel_params.T_2E=float(f_vox.attrs['T_2E'])
            if 'T_2I' in f_vox.attrs:
                self.voxel_params.T_2I=float(f_vox.attrs['T_2I'])
            if 's_e_0' in f_vox.attrs:
                self.voxel_params.s_e_0=float(f_vox.attrs['s_e_0'])
            if 's_i_0' in f_vox.attrs:
                self.voxel_params.s_i_0=float(f_vox.attrs['s_i_0'])
            if 'B0' in f_vox.attrs:
                self.voxel_params.B0=float(f_vox.attrs['B0'])
            if 'TE' in f_vox.attrs:
                self.voxel_params.TE=float(f_vox.attrs['TE'])
            if 's_e' in f_vox.attrs:
                self.voxel_params.s_e=float(f_vox.attrs['s_e'])
            if 's_i' in f_vox.attrs:
                self.voxel_params.s_i=float(f_vox.attrs['s_i'])
            if 'beta' in f_vox.attrs:
                self.voxel_params.beta=float(f_vox.attrs['beta'])
            if 'k2' in f_vox.attrs:
                self.voxel_params.k2=float(f_vox.attrs['k2'])
            if 'k3' in f_vox.attrs:
                self.voxel_params.k3=float(f_vox.attrs['k3'])

            self.voxel_rec={}
            if 'G_total' in f_vox:
                self.voxel_rec['G_total']=np.array(f_vox['G_total'])
            if 'G_total' in f_vox:
                self.voxel_rec['s']=np.array(f_vox['s'])
            if 'G_total' in f_vox:
                self.voxel_rec['f_in']=np.array(f_vox['f_in'])
            if 'G_total' in f_vox:
                self.voxel_rec['v']=np.array(f_vox['v'])
            if 'G_total' in f_vox:
                self.voxel_rec['q']=np.array(f_vox['q'])
            if 'G_total' in f_vox:
                self.voxel_rec['y']=np.array(f_vox['y'])

        self.neural_state_rec=None
        if 'neuron_state' in f:
            f_state=f['neuron_state']
            self.neural_state_rec={'g_ampa_r': np.array(f_state['g_ampa_r']),
                                   'g_ampa_x': np.array(f_state['g_ampa_x']),
                                   'g_ampa_b': np.array(f_state['g_ampa_b']),
                                   'g_nmda':   np.array(f_state['g_nmda']),
                                   'g_gaba_a': np.array(f_state['g_gaba_a']),
                                   #'g_gaba_b': np.array(f_state['g_gaba_b']),
                                   'vm':       np.array(f_state['vm']),
                                   'record_idx': np.array(f_state['record_idx'])}

        self.e_firing_rates=None
        self.i_firing_rates=None
        self.rt=None
        if 'firing_rates' in f:
            f_rates=f['firing_rates']
            self.e_firing_rates=np.array(f_rates['e_rates'])
            self.i_firing_rates=np.array(f_rates['i_rates'])
            self.rt=get_response_time(self.e_firing_rates, self.stim_start_time)

        self.background_rate=None
        if 'background_rate' in f:
            b_rate=f['background_rate']
            self.background_rate=np.array(b_rate['firing_rate'])

        self.task_rates=None
        if 'task_rates' in f:
            t_rates=f['task_rates']
            self.task_rates=np.array(t_rates['firing_rates'])

        self.e_spike_neurons=None
        self.e_spike_times=None
        self.i_spike_neurons=None
        self.i_spike_times=None
        if 'spikes' in f:
            f_spikes=f['spikes']
            self.e_spike_neurons=[]
            self.e_spike_times=[]
            self.i_spike_neurons=[]
            self.i_spike_times=[]
            for idx in range(self.num_groups):
                if ('e.%d.spike_neurons' % idx) in f_spikes:
                    self.e_spike_neurons.append(np.array(f_spikes['e.%d.spike_neurons' % idx]))
                    self.e_spike_times.append(np.array(f_spikes['e.%d.spike_times' % idx]))
                if ('i.%d.spike_neurons' % idx) in f_spikes:
                    self.i_spike_neurons.append(np.array(f_spikes['i.%d.spike_neurons' % idx]))
                    self.i_spike_times.append(np.array(f_spikes['i.%d.spike_times' % idx]))

        f.close()
