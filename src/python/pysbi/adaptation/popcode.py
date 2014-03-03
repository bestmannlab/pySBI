from brian import network_operation, defaultclock, Network, reinit_default_clock
from brian.equations import Equations
from brian.monitor import MultiStateMonitor
from brian.neurongroup import NeuronGroup, linked_var
from brian.stdunits import ms
from brian.tools.parameters import Parameters
import numpy as np
from pysbi.voxel import Voxel, get_bold_signal

default_params=Parameters(
    tune_width=10,
    tau_r1=5*ms,
    tau_r2=10*ms,
    tau_a=400*ms,
    tau_ar=50*ms,
    eta=2
)


class PopulationCode(NeuronGroup):

    def __init__(self, N, params=default_params):
        eqs=Equations('''
            dr/dt=x/tau_r1-r/tau_r2      : 1
            da/dt=(eta*r)/tau_ar-a/tau_a : 1
            e=1.0-a                      : 1
            tau_a                        : second
            tau_r1                       : second
            tau_r2                       : second
            tau_ar                       : second
            eta                          : 1
            total_e                      : 1
            total_r                      : 1
            x                            : 1
            ''')
        NeuronGroup.__init__(self, N, model=eqs, compile=True, freeze=True)

        self.N=N
        self.params=params
        self.tau_a=self.params.tau_a
        self.tau_r1=self.params.tau_r1
        self.tau_r2=self.params.tau_r2
        self.tau_ar=self.params.tau_ar
        self.eta=self.params.eta
        self.total_e=linked_var(self,'e',func=sum)
        self.total_r=linked_var(self,'r',func=sum)

    def get_population_function(self, x, var):
        pass


class ProbabilisticPopulationCode(PopulationCode):

    def __init__(self, N, params=default_params):
        PopulationCode.__init__(self, N, params=params)

    def get_population_function(self, x, var):
        return 1.0/ var *np.exp(-(np.array(range(self.N))-x)**2.0/(2.0*self.params.tune_width**2.0))


class SamplingPopulationCode(PopulationCode):

    def __init__(self, N, params=default_params):
        PopulationCode.__init__(self, N, params=params)

    def get_population_function(self, x, var):
        return 1.0/ var * np.exp( -(np.array(range(self.N)) - x)**2.0 / (2.0 * var**2.0))


def run_pop_code(pop_class, N, network_params, x_vec, var_vec, start_times, end_times, trial_duration):
    pop=pop_class(N,network_params)
    pop_monitor=MultiStateMonitor(pop, vars=['x','r','e','total_e','total_r'], record=True)
    voxel=Voxel()

    @network_operation(when='start')
    def get_pop_input():
        pop.x=0.0
        for i,(x,var) in enumerate(zip(x_vec,var_vec)):
            if start_times[i]<defaultclock.t<end_times[i]:
                pop.x=pop.get_population_function(x, var)

    net=Network(pop, pop_monitor, get_pop_input)
    reinit_default_clock()
    net.run(trial_duration)

    g_total=np.sum(np.clip(pop_monitor['e'].values,0,1) * pop_monitor['x'].values, axis=0)+0.1
    voxel_monitor=get_bold_signal(g_total, voxel.params, range(int(start_times[0]/defaultclock.dt)), trial_duration)

    return pop_monitor, voxel_monitor