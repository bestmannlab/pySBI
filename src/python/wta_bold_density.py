from brian import *
import numpy as np
from playdoh import *
from wta import default_params, WTANetworkGroup, get_bold, get_voxel

def get_contrast(params):
    wta_params=default_params()
    wta_params.p_b_e=params[0]
    wta_params.p_x_e=params[1]
    wta_params.p_e_e=params[2]
    wta_params.p_e_i=params[3]
    wta_params.p_i_i=params[4]
    wta_params.p_i_e=params[5]

    high_contrast_input_freq=np.array([10, 10, 40])*Hz
    low_contrast_input_freq=np.array([20, 20, 20])*Hz
    network_group_size=4000
    num_groups=3

    network=WTANetworkGroup(network_group_size, num_groups, params=wta_params)
    voxel=get_voxel(28670*nA)
    voxel.G_total = linked_var(network, 'g_syn', func=sum)
    high_contrast_bold, high_contrast_pre=get_bold(high_contrast_input_freq, network, voxel, wta_params, num_groups)
    low_contrast_bold, low_contrast_pre=get_bold(low_contrast_input_freq, network, voxel, wta_params, num_groups)

    valid=True
    rate_1=high_contrast_pre[0].smooth_rate(width=5*ms,filter='gaussian')
    rate_2=high_contrast_pre[1].smooth_rate(width=5*ms,filter='gaussian')
    rate_3=high_contrast_pre[2].smooth_rate(width=5*ms,filter='gaussian')
    high_contrast_max=max(rate_3)
    if high_contrast_max<max(rate_1)+20 and high_contrast_max<max(rate_2)+20:
        valid=False

    if valid:
        rate_1=low_contrast_pre[0].smooth_rate(width=5*ms,filter='gaussian')
        rate_2=low_contrast_pre[1].smooth_rate(width=5*ms,filter='gaussian')
        rate_3=low_contrast_pre[2].smooth_rate(width=5*ms,filter='gaussian')
        low_contrast_maxes=[max(rate_1), max(rate_2), max(rate_3)]
        maxIdx=-1
        maxRate=0
        for i,low_contrast_max in enumerate(low_contrast_maxes):
            if low_contrast_max>maxRate:
                maxRate=low_contrast_max
                maxIdx=i
        for i,low_contrast_max in enumerate(low_contrast_maxes):
            if not i==maxIdx and maxRate<low_contrast_max+20:
                valid=False
                break

    contrast=max(high_contrast_bold)-max(low_contrast_bold)

    return (contrast, valid)

def get_bold_density(param_ranges, dist_size):
    p_b_e_dist=np.ones([dist_size])*1.0/float(dist_size)
    p_x_e_dist=np.ones([dist_size])*1.0/float(dist_size)
    p_e_e_dist=np.ones([dist_size])*1.0/float(dist_size)
    p_e_i_dist=np.ones([dist_size])*1.0/float(dist_size)
    p_i_i_dist=np.ones([dist_size])*1.0/float(dist_size)
    p_i_e_dist=np.ones([dist_size])*1.0/float(dist_size)

    contrast=np.zeros([dist_size])
    param_list=[]
    p_b_e_incr=(param_ranges['p_b_e'][1]-param_ranges['p_b_e'][0])/(dist_size-1)
    p_x_e_incr=(param_ranges['p_x_e'][1]-param_ranges['p_x_e'][0])/(dist_size-1)
    p_e_e_incr=(param_ranges['p_e_e'][1]-param_ranges['p_e_e'][0])/(dist_size-1)
    p_e_i_incr=(param_ranges['p_e_i'][1]-param_ranges['p_e_i'][0])/(dist_size-1)
    p_i_i_incr=(param_ranges['p_i_i'][1]-param_ranges['p_i_i'][0])/(dist_size-1)
    p_i_e_incr=(param_ranges['p_i_e'][1]-param_ranges['p_i_e'][0])/(dist_size-1)
    for p_b_e in [(x+1)*p_b_e_incr for x in range(dist_size)]:
        for p_x_e in [(x+1)*p_x_e_incr for x in range(dist_size)]:
            for p_e_e in [(x+1)*p_e_e_incr for x in range(dist_size)]:
                for p_e_i in [(x+1)*p_e_i_incr for x in range(dist_size)]:
                    for p_i_i in [(x+1)*p_i_i_incr for x in range(dist_size)]:
                        for p_i_e in [(x+1)*p_i_e_incr for x in range(dist_size)]:
                            param_list.append([p_b_e, p_x_e, p_e_e, p_e_i, p_i_i, p_i_e])

    print('Testing %d sets of parameter values' % len(param_list))
    #result=get_contrast(wta_params)
    map_results=map(get_contrast, param_list, machines=['localhost','gorilla','howler','bushbaby','bigfoot','orangutan'],
        codedependencies=['wta.py','wta_bold_density.py'])
    return map_results

