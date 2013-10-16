from math import exp
from brian import Equations, NeuronGroup, Parameters, second, network_operation, defaultclock, Network, reinit_default_clock
from brian.monitor import MultiStateMonitor
from brian.neurongroup import linked_var
from brian.stdunits import nS

import numpy as np

default_params=Parameters(
    G_base=28670*nS,
    # synaptic efficacy (value from Zheng et al., 2002)
    eta=0.5 * second,
    # signal decay time constant (value from Zheng et al., 2002)
    tau_s=0.8 * second,
    # autoregulatory feedback time constant (value from Zheng et al., 2002)
    tau_f=0.4 * second,
    # Grubb's parameter
    alpha=0.2,
    # venous time constant (value from Friston et al., 2000)
    tau_o=1*second,
    # resting net oxygen extraction fraction by capillary bed (value
    # from Friston et al., 2000)
    e_base=0.8,
    # resting blood volume fraction (value from Friston et al., 2000)
    v_base=0.02,
    # resting intravascular transverse relaxation time (41.4ms at 4T, from
    # Yacoub et al, 2001)
    T_2E=.0414,
    # resting extravascular transverse relaxation time (23.5ms at 4T from
    # Yacoub et al, 2001)
    T_2I=.0235,
    # effective intravascular spin density (assumed to be equal to
    # extravascular spin density, from Behzadi & Liu, 2005)
    s_e_0=1,
    # effective extravascular spin density (assumed to be equal to
    # intravascular spin density, from Behzadi & Liu, 2005)
    s_i_0=1,
    # Main magnetic field strength
    B0=4.7,
    # Echo time
    TE=.02,
    # MR parameters (from Obata et al, 2004)
    computed_parameters = '''
        freq_offset=40.3*(B0/1.5)
        k1=4.3*freq_offset*e_base*TE
        r_0=25*(B0/1.5)**2
    ''')

zheng_params=Parameters(
    G_base=28670*nS,
    # synaptic efficacy (value from Zheng et al., 2002)
    eta=0.5 * second,
    # signal decay time constant (value from Zheng et al., 2002)
    tau_s=1 * second,
    # autoregulatory feedback time constant (value from Zheng et al., 2002)
    tau_f=1.11 * second,
    # arteriolar blood oxygen concentration (value from Zheng et al., 2002)
    c_ab=1.0,
    # resting net oxygen extraction fraction by capillary bed (value
    # from Friston et al., 2000)
    e_base=0.34,
    # capillary transit time (value from Zheng et al., 2002)
    transitTime=.2,
    # baseline ratio of tissue and arterial plasma oxygen concentration
    # (value from Zheng et al., 2002)
    g_0=0.1,
    # baseline mean oxygen concentration of the capillary (value from
    # Zheng et al,. 2002)
    cb_0=0.71,
    # ratio c_p/c_b (value from Zheng et al., 2002)
    r=0.01,
    # volume ratio (value from Zheng et al., 2002)
    v_ratio=50.0,
    # scaling factor (value from Zheng et al., 2002)
    j=8.0,
    # change in metabolic demand (value from Zheng et al, 2002)
    k=0.05,
    # venous time constant (value from Friston et al., 2000)
    tau_o=.3,
    # resting blood volume fraction (value from Friston et al., 2000)
    v_0=0.02,
    # Grubb's parameter
    alpha=0.38,
    # resting intravascular transverse relaxation time (41.4ms at 4T, from
    # Yacoub et al, 2001)
    T_2E=.0414,
    # resting extravascular transverse relaxation time (23.5ms at 4T from
    # Yacoub et al, 2001)
    T_2I=.0235,
    # effective intravascular spin density (assumed to be equal to
    # extravascular spin density, from Behzadi & Liu, 2005)
    s_e_0=1,
    # effective extravascular spin density (assumed to be equal to
    # intravascular spin density, from Behzadi & Liu, 2005)
    s_i_0=1,
    # Main magnetic field strength
    B0=4.7,
    # Echo time
    TE=.02,
    computed_parameters = '''
        phi=.15*transitTime
        freq_offset=40.3*(B0/1.5)
        k1=4.3*freq_offset*e_base*TE
        r_0=25*(B0/1.5)**2
    '''

)

class TestVoxel(NeuronGroup):
    def __init__(self, params=zheng_params, network=None):
        eqs=Equations('''
        G_total                                                       : siemens
        G_total_exc                                                   : siemens
        cmr_o                                                         : 1
        cb                                                            : 1
        g                                                             : 1
        c_ab                                                          : 1
        cb_0                                                          : 1
        g_0                                                           : 1
        ''')

        NeuronGroup.__init__(self, 1, model=eqs, compile=True, freeze=True)

        self.params=params
        self.c_ab=self.params.c_ab
        self.cb_0=self.params.cb_0
        self.g_0=self.params.g_0

        self.cb=self.cb_0
        self.g=self.g_0

        if network is not None:
            self.G_total = linked_var(network, 'g_syn', func=sum)
            self.G_total_exc = linked_var(network, 'g_syn_exc', func=sum)


class ZhengVoxel(NeuronGroup):
    def __init__(self, params=zheng_params, network=None):
        eqs=Equations('''
        G_total                                                       : siemens
        G_total_exc                                                   : siemens
        ds/dt=eta*(G_total-G_base)/G_base-s/tau_s-(f_in-1.0)/tau_f    : 1
        df_in/dt=s/second                                             : 1.0
        dv/dt=1/tau_o*(f_in-f_out)                                    : 1
        f_out=v**(1.0/alpha)                                          : 1
        do_e/dt=1.0/(phi/f_in)*(-o_e+(1.0-g)*(1.0-(1.0-e_base/(1.0-g_0))**(1.0/f_in))) : %.4f
        dcb/dt=1.0/(phi/f_in)*(-cb-(c_ab*o_e)/oe_log+c_ab*g)  : 1
        oe_log                            : 1
        cmr_o=(cb-g*c_ab)/(cb_0-g_0*c_ab) : 1
        dg/dt=1.0/(j*v_ratio*((r*transitTime)/e_base))*((cmr_o-1.0)-k*s)  : %.4f
        dq/dt=1/tau_o*((f_in*o_e/e_base)-f_out*q/v)                   : 1
        y=v_0*((k1+k2)*(1-q)-(k2+k3)*(1-v))                        : 1
        G_base                                                        : siemens
        eta                                                           : 1/second
        tau_s                                                         : second
        tau_f                                                         : second
        alpha                                                         : 1
        tau_o                                                         : second
        v_0                                                           : 1
        k1                                                            : 1
        k2                                                            : 1
        k3                                                            : 1
        phi                                                           : %.4f*second
        e_base                                                        : %.4f
        g_0                                                           : %.4f
        c_ab                                                          : 1
        cb_0                                                          : 1
        v_ratio                                                       : 1
        j                                                             : 1
        transitTime                                                   : second
        k                                                             : 1
        r                                                             : 1
        ''' % (params.e_base, params.g_0, params.phi, params.e_base, params.g_0))
        NeuronGroup.__init__(self, 1, model=eqs, compile=True, freeze=True)

        self.params=params
        self.G_base=params.G_base
        self.eta=params.eta
        self.tau_s=params.tau_s
        self.tau_f=params.tau_f
        self.alpha=params.alpha
        self.tau_o=params.tau_o
        self.e_base=params.e_base
        self.v_0=params.v_0
        self.k1=params.k1
        self.params.s_e=params.s_e_0*exp(-params.TE/params.T_2E)
        self.params.s_i=params.s_i_0*exp(-params.TE/params.T_2I)
        self.params.beta=self.params.s_e/self.params.s_i
        self.k2=self.params.beta*params.r_0*self.e_base*params.TE
        self.k3=self.params.beta-1.0
        self.c_ab=self.params.c_ab
        self.cb_0=self.params.cb_0
        self.g_0=self.params.g_0
        self.phi=self.params.phi
        self.v_ratio=self.params.v_ratio
        self.j=self.params.j
        self.transitTime=self.params.transitTime
        self.k=self.params.k
        self.r=self.params.r

        self.f_in=1.0
        self.s=0.0
        self.f_in=1.0
        self.f_out=1.0
        self.v=1.0
        self.o_e=self.e_base
        self.cb=self.cb_0
        self.g=self.g_0
        self.oe_log=np.log(1.0-self.o_e/(1.0-self.g))

        self.q=1.0
        self.y=0.0

        if network is not None:
            self.G_total = linked_var(network, 'g_syn', func=sum)
            self.G_total_exc = linked_var(network, 'g_syn_exc', func=sum)

class LFPSource(NeuronGroup):
    def __init__(self, pyramidal_group):
        eqs=Equations('''
         LFP : amp
        ''')
        NeuronGroup.__init__(self, 1, model=eqs, compile=True, freeze=True)
        self.LFP=linked_var(pyramidal_group, 'I_abs', func=sum)

class Voxel(NeuronGroup):
    def __init__(self, params=default_params, network=None):
        eqs=Equations('''
        G_total                                                       : siemens
        G_total_exc                                                   : siemens
        ds/dt=eta*(G_total-G_base)/G_base-s/tau_s-(f_in-1.0)/tau_f    : 1
        df_in/dt=s/second                                             : 1
        dv/dt=1/tau_o*(f_in-f_out)                                    : 1
        f_out=v**(1.0/alpha)                                          : 1
        o_e=1-(1-e_base)**(1/f_in)                                    : 1
        dq/dt=1/tau_o*((f_in*o_e/e_base)-f_out*q/v)                   : 1
        y=v_base*((k1+k2)*(1-q)-(k2+k3)*(1-v))                        : 1
        G_base                                                        : siemens
        eta                                                           : 1/second
        tau_s                                                         : second
        tau_f                                                         : second
        alpha                                                         : 1
        tau_o                                                         : second
        e_base                                                        : 1
        v_base                                                        : 1
        k1                                                            : 1
        k2                                                            : 1
        k3                                                            : 1
        ''')
        NeuronGroup.__init__(self, 1, model=eqs, compile=True, freeze=True)
        self.params=params
        self.G_base=params.G_base
        self.eta=params.eta
        self.tau_s=params.tau_s
        self.tau_f=params.tau_f
        self.alpha=params.alpha
        self.tau_o=params.tau_o
        self.e_base=params.e_base
        self.v_base=params.v_base
        self.k1=params.k1
        self.params.s_e=params.s_e_0*exp(-params.TE/params.T_2E)
        self.params.s_i=params.s_i_0*exp(-params.TE/params.T_2I)
        self.params.beta=self.params.s_e/self.params.s_i
        self.k2=self.params.beta*params.r_0*self.e_base*params.TE
        self.k3=self.params.beta-1

        self.f_in=1
        self.s=0
        self.f_in=1
        self.f_out=1
        self.v=1
        self.q=1
        self.y=0

        if network is not None:
            self.G_total = linked_var(network, 'g_syn', func=sum)
            self.G_total_exc = linked_var(network, 'g_syn_exc', func=sum)

def get_bold_signal(g_total, voxel_params, baseline_range, trial_duration):
    voxel=Voxel(params=voxel_params)
    voxel.G_base=g_total[baseline_range[0]:baseline_range[1]].mean()
    voxel_monitor = MultiStateMonitor(voxel, vars=['G_total','s','f_in','v','f_out','q','y'], record=True)

    @network_operation(when='start')
    def get_input():
        idx=int(defaultclock.t/defaultclock.dt)
        if idx<baseline_range[0]:
            voxel.G_total=voxel.G_base
        elif idx<len(g_total):
            voxel.G_total=g_total[idx]
        else:
            voxel.G_total=voxel.G_base

    net=Network(voxel, get_input, voxel_monitor)
    reinit_default_clock()
    bold_trial_duration=10*second
    if trial_duration+6*second>bold_trial_duration:
        bold_trial_duration=trial_duration+6*second
    net.run(bold_trial_duration)

    return voxel_monitor