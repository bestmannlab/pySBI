from scipy import optimize
from scipy.stats import ttest_ind
from brian import ms, second
from brian.connections.delayconnection import DelayConnection
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pylab as plt
import numpy as np
import rpy2.rlike.container as rlc
import rpy2.robjects as robjects

class Struct():
    def __init__(self):
        pass

colors={
    'control': 'b',
    'depolarizing': 'r',
    'hyperpolarizing': 'g'
}

def save_to_png(fig, output_file):
    fig.set_facecolor("#FFFFFF")
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(output_file, dpi=72)

def save_to_eps(fig, output_file):
    fig.set_facecolor("#FFFFFF")
    canvas = FigureCanvasAgg(fig)
    canvas.print_eps(output_file, dpi=72)

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

def init_rand_weight_connection(pop1, pop2, target_name, min_weight, max_weight, p, delay, allow_self_conn=True):
    """
    Initialize a connection between two populations
    pop1 = population sending projections
    pop2 = populations receiving projections
    target_name = name of synapse type to project to
    min_weight = min weight of connection
    max_weight = max weight of connection
    p = probability of connection between any two neurons
    delay = delay
    allow_self_conn = allow neuron to project to itself
    """
    W=min_weight+np.random.rand(len(pop1),len(pop2))*(max_weight-min_weight)
    conn=DelayConnection(pop1, pop2, target_name, sparseness=p, W=W, delay=delay)

    # Remove self-connections
    if not allow_self_conn and len(pop1)==len(pop2):
        for j in xrange(len(pop1)):
            conn[j,j]=0.0
            conn.delay[j,j]=0.0
            conn[j,j]=0.0
            conn.delay[j,j]=0.0
    return conn

def init_connection(pop1, pop2, target_name, weight, p, delay, allow_self_conn=True):
    """
    Initialize a connection between two populations
    pop1 = population sending projections
    pop2 = populations receiving projections
    target_name = name of synapse type to project to
    weight = weight of connection
    p = probability of connection between any two neurons
    delay = delay
    allow_self_conn = allow neuron to project to itself
    """
    conn=DelayConnection(pop1, pop2, target_name, sparseness=p, weight=weight, delay=delay)

    # Remove self-connections
    if not allow_self_conn and len(pop1)==len(pop2):
        for j in xrange(len(pop1)):
            conn[j,j]=0.0
            conn.delay[j,j]=0.0
            conn[j,j]=0.0
            conn.delay[j,j]=0.0
    return conn


def weibull(x, alpha, beta):
    return 1.0-0.5*np.exp(-(x/alpha)**beta)


def rt_function(x, a, k, tr):
    return a/(k*x)*np.tanh(a*k*x)+tr


def exp_decay(x, n, lam):
    return n*np.exp(-lam*x)

def get_response_time(e_firing_rates, stim_start_time, stim_end_time, upper_threshold=60, lower_threshold=None, dt=.1*ms):
    rate_1=e_firing_rates[0]
    rate_2=e_firing_rates[1]
    times=np.array(range(len(rate_1)))*(dt/second)
    rt=None
    decision_idx=-1
    for idx,time in enumerate(times):
        time=time*second
        if stim_start_time < time < stim_end_time:
            if rt is None:
                if rate_1[idx]>=upper_threshold and (lower_threshold is None or rate_2[idx]<=lower_threshold):
                    decision_idx=0
                    rt=(time-stim_start_time)/ms
                    break
                elif rate_2[idx]>=upper_threshold and (lower_threshold is None or rate_1[idx]<=lower_threshold):
                    decision_idx=1
                    rt=(time-stim_start_time)/ms
                    break
    return rt,decision_idx

def mdm_outliers(dist):
    c=1.1926
    medians=[]
    for idx,x in enumerate(dist):
        medians.append(np.median(np.abs(x-dist)))
    mdm=c*np.median(medians)
    outliers=[]
    for idx,x in enumerate(dist):
        if np.median(np.abs(x-dist))/mdm>3:
            outliers.append(idx)
    return outliers

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return data[s<m]

class _baseFunctionFit:
    """Not needed by most users except as a superclass for developping your own functions

    You must overide the eval and inverse methods and a good idea to overide the _initialGuess
    method aswell.
    """

    def __init__(self, xx, yy, sems=1.0, guess=None, display=1,
                 expectedMin=0.5):
        self.xx = np.asarray(xx)
        self.yy = np.asarray(yy)
        self.sems = np.asarray(sems)
        self.expectedMin = expectedMin
        self.display=display
        # for holding error calculations:
        self.ssq=0
        self.rms=0
        self.chi=0
        #initialise parameters
        if guess==None:
            self.params = self._initialGuess()
        else:
            self.params = guess

        #do the calculations:
        self._doFit()

    def _doFit(self):
        #get some useful variables to help choose starting fit vals
        self.params = optimize.fmin_powell(self._getErr, self.params, (self.xx,self.yy,self.sems),disp=self.display)
        #        self.params = optimize.fmin_bfgs(self._getErr, self.params, None, (self.xx,self.yy,self.sems),disp=self.display)
        self.ssq = self._getErr(self.params, self.xx, self.yy, 1.0)
        self.chi = self._getErr(self.params, self.xx, self.yy, self.sems)
        self.rms = self.ssq/len(self.xx)

    def _initialGuess(self):
        xMin = min(self.xx); xMax = max(self.xx)
        xRange=xMax-xMin; xMean= (xMax+xMin)/2.0
        guess=[xMean, xRange/5.0]
        return guess

    def _getErr(self, params, xx,yy,sems):
        mod = self.eval(xx, params)
        err = sum((yy-mod)**2/sems)
        return err

    def eval(self, xx=None, params=None):
        """Returns fitted yy for any given xx value(s).
        Uses the original xx values (from which fit was calculated)
        if none given.

        If params is specified this will override the current model params."""
        yy=xx
        return yy

    def inverse(self, yy, params=None):
        """Returns fitted xx for any given yy value(s).

        If params is specified this will override the current model params.
        """
        #define the inverse for your function here
        xx=yy
        return xx


class FitWeibull(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)"""
    def eval(self, xx=None, params=None):
        if params==None:  params=self.params #so the user can set params for this particular eval
        alpha = params[0];
        if alpha<=0: alpha=0.001
        beta = params[1]
        xx = np.asarray(xx)
        yy =  self.expectedMin + (1.0-self.expectedMin)*(1-np.exp( -(xx/alpha)**(beta) ))
        return yy
    def inverse(self, yy, params=None):
        if params==None: params=self.params #so the user can set params for this particular inv
        alpha = params[0]
        beta = params[1]
        xx = alpha * (-np.log((1.0-yy)/(1-self.expectedMin))) **(1.0/beta)
        return xx

class FitRT(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)"""
    def eval(self, xx=None, params=None):
        if params==None:  params=self.params #so the user can set params for this particular eval
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.asarray(xx)
        yy = a*np.tanh(k*xx)+tr
        return yy
    def inverse(self, yy, params=None):
        if params==None: params=self.params #so the user can set params for this particular inv
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.arctanh((yy-tr)/a)/k
        return xx

class FitSigmoid(_baseFunctionFit):
    def eval(self, xx=None, params=None):
        if params==None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        xx = np.asarray(xx)
        #yy = a+1.0/(1.0+np.exp(-k*xx))
        yy =1.0/(1.0+np.exp(-k*(xx-x0)))
        return yy

    def inverse(self, yy, params=None):
        if params==None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        #xx = -np.log((1/(yy-a))-1)/k
        xx = -np.log((1.0/yy)-1.0)/k+x0
        return xx


"""
file    twoway_interaction.py
author  Ernesto P. Adorio, Ph.D.
        ernesto.adorio@gmail.com
        UPDEPP at Clarkfield
desc    Performs an anova with interaction
        on replicated input data.
        Each block must have the same number
        of input values.
version 0.0.2 Sep 12, 2011

"""

def twoway_interaction(groups, first_factor_label, second_factor_label, format="html"):
    b = len(groups[0][0])
    a = len(groups)
    c = len(groups[0])
    groupsums = [0.0] * c

    #print "blocks, a, c=", b, a, c

    #print "Input groups:"
    v = 0.0   #total variation
    vs = 0.0  #subtotal variation
    vr = 0.0  #variation between rows
    GT = 0
    for i in range(a):
        vsx = 0.0
        vrx = 0.0
        for j in range(c):
            vsx = sum(groups[i][j])
            groupsums[j] += vsx
            #print "debug vsx", vsx
            vrx += vsx
            vs += vsx * vsx
            for k in range(b):
                x = groups[i][j][k]
                v += x * x
                GT += x
        vr += vrx* vrx

    #print "groupsums=", groupsums, vs

    totadjustment = GT*GT/(a * b * c)
    vs = vs/b - totadjustment
    vr = vr/(b * c)- totadjustment
    v  -= totadjustment
    vc = sum([x * x for x in groupsums])/ (a*b)-totadjustment
    vi = vs-vr -vc
    ve = v- (vr + vc + vi)
    #print "debug vs, vr, vc=", vs, vr, vc

    dfvr = (a-1)
    dfvc = (c-1.0)
    dfvi = ((a-1)*(c-1))
    dfve = (a * c* (b-1))
    dfvs = a*c - 1
    dfv  = (a * b * c -1)
    mvr = vr/(dfvr)
    mvc = vc/(dfvc)
    mvi = vi / dfvi
    mve = ve/dfve
    Fr = mvr/mve
    Fc = mvc/mve
    Fi = mvi/mve

    from scipy import stats

    pvalr = 1.0 - stats.f.cdf(Fr, dfvr, dfve)
    pvalc = 1.0 - stats.f.cdf(Fc, dfvc, dfve)
    pvali = 1.0 - stats.f.cdf(Fi, dfvi, dfve)


    if format=="html":
        output="""
    <table border="1">
    <tr><th>Variation  </th><th>Sum of Squares</th><th>  df</th><th>  Mean Sum of Squares</th><th>   F-value</th><th> p-value</th></tr>
    <td>Rows(%s) </td><td>%f</td><td>    %d</td><td>     %f</td> <td> %f</td> <td>%f</td></tr>
    <tr><td>Columns(%s)</td><td>%f</td><td>  %d</td><td>     %f</td> <td> %f</td> <td>%f</td></tr>
    <tr><td>Interaction</td><td>%f</td><td>  %d</td><td>     %f</td> <td> %f</td> <td>%f</td></tr>
    <tr><td>Subtotals </td><td> %f</td><td> %d</td></tr>
    <tr><td>Residuals(random)  </td><td>%f</td><td>  %d</td><td>%f</td></tr>
    <tr><td>Totals</td><td>%f.2 </td><td>%d </td></tr>
    </table>
    """ % (first_factor_label,vr, dfvr, mvr, mvr/mve, pvalr,
           second_factor_label,vc, dfvc, mvc, mvc/mve, pvalc,
           vi, dfvi, mvi, mvi/mve, pvali,
           vs, dfvs,
           ve, dfve, mve,
           v,  dfv)
    else:
        output=[[vr, dfvr, mvr, mvr/mve, pvalr],
                [vc, dfvc, mvc, mvc/mve, pvalc],
                [vi, dfvi, mvi, mvi/mve, pvali],
                [vs, dfvs],
                [ve, dfve, mve],
                [v,  dfv]]

    return output

def twoway_interaction_r(outcome, factors, data):
    od = rlc.OrdDict()
    od[outcome] = robjects.FloatVector([x[0] for x in data])
    od[factors[0]] = robjects.FloatVector([x[1] for x in data])
    od[factors[1]] = robjects.StrVector([x[2] for x in data])


    dataf = robjects.DataFrame(od)
    rcode = 'data = %s' % dataf.r_repr()
    res = robjects.r(rcode)
    anova_data = robjects.r('anova(lm(%s ~ %s*%s, data))' % (outcome,factors[0],factors[1]))

    fvals = anova_data[3]
    fprobs = anova_data[4]

    var_indices = {factors[0]:0, factors[1]:1, '%s+%s' % (factors[0],factors[1]):2}
    all_fvals = {}
    all_fprobs = {}

    for var_name,var_index in var_indices.iteritems():
        all_fvals[var_name] = fvals[var_index]
        all_fprobs[var_name] = fprobs[var_index]
    return (all_fvals,all_fprobs)

def pairwise_comparisons(measure_dict, factor_levels, comparison_factors):
    num_comparisons=len(factor_levels)
    pairwise={}
    for factor_value in factor_levels:
        (t,p)=ttest_ind(measure_dict[comparison_factors[0]][factor_value],measure_dict[comparison_factors[1]][factor_value])
        pairwise[factor_value]=(t,p*num_comparisons/2.0)
    return pairwise