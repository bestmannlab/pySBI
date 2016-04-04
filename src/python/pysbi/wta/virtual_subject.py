from brian import Clock, Hz, second, PoissonGroup, network_operation, pA, Network, nS
import h5py
from pysbi.wta.monitor import WTAMonitor, SessionMonitor
from pysbi.wta.network import default_params, pyr_params, inh_params, simulation_params, WTANetworkGroup
import numpy as np
import matplotlib.pyplot as plt

class VirtualSubject:
    def __init__(self, subj_id, wta_params=default_params(), pyr_params=pyr_params(), inh_params=inh_params(),
                 sim_params=simulation_params()):
        self.subj_id = subj_id
        self.wta_params = wta_params
        self.pyr_params = pyr_params
        self.inh_params = inh_params
        self.sim_params = sim_params

        self.simulation_clock = Clock(dt=self.sim_params.dt)
        self.input_update_clock = Clock(dt=1 / (self.wta_params.refresh_rate / Hz) * second)

        self.background_input = PoissonGroup(self.wta_params.background_input_size,
                                             rates=self.wta_params.background_freq,
                                             clock=self.simulation_clock)
        self.task_inputs = []
        for i in range(self.wta_params.num_groups):
            self.task_inputs.append(PoissonGroup(self.wta_params.task_input_size,
                                                 rates=self.wta_params.task_input_resting_rate,
                                                 clock=self.simulation_clock))

        # Create WTA network
        self.wta_network = WTANetworkGroup(params=self.wta_params, background_input=self.background_input,
                                           task_inputs=self.task_inputs, pyr_params=self.pyr_params,
                                           inh_params=self.inh_params, clock=self.simulation_clock)


        # Create network monitor
        self.wta_monitor = WTAMonitor(self.wta_network, None, None, self.sim_params, record_lfp=False,
                                      record_voxel=False, record_neuron_state=False, record_spikes=False,
                                      record_firing_rate=True, record_inputs=True, record_connections=None,
                                      save_summary_only=False, clock=self.simulation_clock)


        # Create Brian network and reset clock
        self.net = Network(self.background_input, self.task_inputs, self.wta_network,
                           self.wta_network.connections.values(), self.wta_monitor.monitors.values())


    def run_session(self, sim_params, output_file=None):

        session_monitor=SessionMonitor(self.wta_network, sim_params, {}, record_connections=[], conv_window=10,
                                       record_firing_rates=True)

        coherence_levels=[0.032, .064, .128, .256, .512]
        trials_per_level=20
        trial_inputs=np.zeros((trials_per_level*len(coherence_levels),2))
        for i in range(len(coherence_levels)):
            coherence=coherence_levels[i]
            # Left
            trial_inputs[i*trials_per_level:i*trials_per_level+trials_per_level/2,0]=self.wta_params.mu_0+self.wta_params.p_a*coherence*100.0
            trial_inputs[i*trials_per_level:i*trials_per_level+trials_per_level/2,1]=self.wta_params.mu_0-self.wta_params.p_b*coherence*100.0

            #Right
            trial_inputs[i*trials_per_level+trials_per_level/2:i*trials_per_level+trials_per_level,0]=self.wta_params.mu_0-self.wta_params.p_b*coherence*100.0
            trial_inputs[i*trials_per_level+trials_per_level/2:i*trials_per_level+trials_per_level,1]=self.wta_params.mu_0+self.wta_params.p_a*coherence*100.0

        trial_inputs=np.random.permutation(trial_inputs)

        for t in range(sim_params.ntrials):
            task_input_rates=trial_inputs[t,:]
            correct_input=np.where(task_input_rates==np.max(task_input_rates))[0]
            self.run_trial(sim_params, task_input_rates)
            session_monitor.record_trial(t, task_input_rates, correct_input, self.wta_network, self.wta_monitor)

        session_monitor.plot()
        if output_file is not None:
            session_monitor.write_output(output_file)

    def run_trial(self, sim_params, input_freq):
        self.wta_monitor.sim_params=sim_params
        self.net.reinit(states=False)

        @network_operation(when='start', clock=self.input_update_clock)
        def set_task_inputs():
            for idx in range(len(self.task_inputs)):
                rate = self.wta_params.task_input_resting_rate
                if sim_params.stim_start_time <= self.simulation_clock.t < sim_params.stim_end_time:
                    rate = input_freq[idx] * Hz + np.random.randn() * self.wta_params.input_var
                    if rate < self.wta_params.task_input_resting_rate:
                        rate = self.wta_params.task_input_resting_rate
                self.task_inputs[idx]._S[0, :] = rate

        @network_operation(clock=self.simulation_clock)
        def inject_current():
            if sim_params.dcs_start_time < self.simulation_clock.t <= sim_params.dcs_end_time:
                self.wta_network.group_e.I_dcs = sim_params.p_dcs
                self.wta_network.group_i.I_dcs = sim_params.i_dcs
            else:
                self.wta_network.group_e.I_dcs = 0 * pA
                self.wta_network.group_i.I_dcs = 0 * pA

        @network_operation(when='start', clock=self.simulation_clock)
        def inject_muscimol():
            if sim_params.muscimol_amount > 0:
                self.wta_network.groups_e[sim_params.injection_site].g_muscimol = sim_params.muscimol_amount

        self.net.add(set_task_inputs, inject_current, inject_muscimol)

        self.net.run(sim_params.trial_duration, report='text')

        #self.wta_monitor.plot()
        self.net.remove(set_task_inputs, inject_current, inject_muscimol)


if __name__=='__main__':
    for i in range(20):
        wta_params=default_params()
        wta_params.background_freq=950*Hz
        pyr_params=pyr_params()
        pyr_params.w_nmda=0.14*nS
        sim_params=simulation_params()
        sim_params.ntrials=100
        sim_params.dcs_end_time=sim_params.trial_duration
        subject=VirtualSubject(i, wta_params=wta_params, pyr_params=pyr_params, sim_params=sim_params)

        subject.run_session(sim_params, output_file='/home/jbonaiuto/Projects/pySBI/data/rdmd/subject.%d.control.h5' % i)
        #plt.show()

        sim_params.p_dcs=2*pA
        sim_params.i_dcs=-1*pA
        subject.run_session(sim_params, output_file='/home/jbonaiuto/Projects/pySBI/data/rdmd/subject.%d.depolarizing.h5' % i)
        #plt.show()

        sim_params.p_dcs=-2*pA
        sim_params.i_dcs=1*pA
        subject.run_session(sim_params, output_file='/home/jbonaiuto/Projects/pySBI/data/rdmd/subject.%d.hyperpolarizing.h5' % i)
        #plt.show()
