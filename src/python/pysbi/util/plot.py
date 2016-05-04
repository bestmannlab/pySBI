from brian import ms, hertz
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure, subplot, ylim, legend, ylabel, xlabel, title
import numpy as np
from pysbi.util.utils import get_response_time, FitSigmoid


def plot_network_firing_rates(e_rates, sim_params, network_params, std_e_rates=None, i_rate=None, std_i_rate=None,
                              plt_title=None, labels=None):
    rt, choice = get_response_time(e_rates, sim_params.stim_start_time, sim_params.stim_end_time,
                                   upper_threshold = network_params.resp_threshold, dt = sim_params.dt)

    figure()
    max_rates=[network_params.resp_threshold+10]
    if i_rate is not None:
        max_rates.append(np.max(i_rate[500:]))
    for i in range(network_params.num_groups):
        max_rates.append(np.max(e_rates[i,500:]))
    max_rate=np.max(max_rates)

    if i_rate is not None:
        ax=subplot(211)
    else:
        ax=subplot(111)
    rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
        alpha=0.25, facecolor='yellow', edgecolor='none')
    ax.add_patch(rect)

    for idx in range(network_params.num_groups):
        label='e %d' % idx
        if labels is not None:
            label=labels[idx]
        time_ticks=(np.array(range(e_rates.shape[1]))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        baseline,=ax.plot(time_ticks, e_rates[idx,:], label=label)
        if std_e_rates is not None:
            ax.fill_between(time_ticks, e_rates[idx,:]-std_e_rates[idx,:], e_rates[idx,:]+std_e_rates[idx,:], alpha=0.5,
                facecolor=baseline.get_color())
    ylim(0,max_rate+5)
    ax.plot([0-sim_params.stim_start_time/ms, (sim_params.trial_duration-sim_params.stim_start_time)/ms],
        [network_params.resp_threshold/hertz, network_params.resp_threshold/hertz], 'k--')
    ax.plot([rt,rt],[0, max_rate+5],'k--')
    legend(loc='best')
    ylabel('Firing rate (Hz)')
    if plt_title is not None:
        title(plt_title)

    if i_rate is not None:
        ax=subplot(212)
        rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
            alpha=0.25, facecolor='yellow', edgecolor='none')
        ax.add_patch(rect)
        label='i'
        if labels is not None:
            label=labels[network_params.num_groups]
        time_ticks=(np.array(range(len(i_rate)))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        baseline,=ax.plot(time_ticks, i_rate, label=label)
        if std_i_rate is not None:
            ax.fill_between(time_ticks, i_rate-std_i_rate, i_rate+std_i_rate, alpha=0.5, facecolor=baseline.get_color())
        ylim(0,max_rate+5)
        ax.plot([rt,rt],[0, max_rate],'k--')
        ylabel('Firing rate (Hz)')
    xlabel('Time (ms)')


def plot_condition_choice_probability(ax, color, left_coherences, left_choice_probs, right_coherences, right_choice_probs, extra_label=''):
    acc_fit = FitSigmoid(left_coherences, left_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(left_coherences), max(left_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, '--%s' % color, label='L* %s' % extra_label)
    ax.plot(left_coherences, left_choice_probs, 'o%s' % color)
    acc_fit = FitSigmoid(right_coherences, right_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(right_coherences), max(right_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, color, label='R* %s' % extra_label)
    ax.plot(right_coherences, right_choice_probs, 'o%s' % color)
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% of Right Choices')