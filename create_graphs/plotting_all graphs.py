#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 12:41:30 2019

@author: jmw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:50:55 2018

@author: jmw
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import seaborn

title = '' # Annotates a title 
savename = 'N2 vs SWIM'
numbering_label = 10 # To control the label size of the numbers in plots
globalFont = 12 # To control the 
plt.rc('xtick',labelsize=numbering_label)
plt.rc('ytick',labelsize=numbering_label)
nameOfHeatmapKind = 'N2_FORWARDS' #This is the repeating element in your filenames to construct the 13x13 matrix from the 13x1 matrix

# Copy the names of the model/mutant filenames you want to compare for. This is just for ONE segment of 
# Cross correlation/mutual information. In the paper we plot it from the first segment 
with open('N2_SWIM0_CC_timelag_total.csv', newline='\n' ) as inputfile:
   cc = list(csv.reader(inputfile))  
with open('N2_SWIM0_CC_PEAK_total.csv', newline='\n' ) as inputfile:
   cc_peak = list(csv.reader(inputfile)) 
with open('N2_SWIM0_MI_total.csv', newline='\n' ) as inputfile:
   mi = list(csv.reader(inputfile))      
# Control filenames
with open('N2_CC_timelag_total.csv', newline='\n' ) as inputfile:
   control_cc = list(csv.reader(inputfile))
with open('N2_CC_PEAK_total.csv', newline='\n' ) as inputfile:
   control_cc_peak= list(csv.reader(inputfile))   
with open('N2_MI_total.csv', newline='\n' ) as inputfile:
   control_mi = list(csv.reader(inputfile)) 

def plot_all(a, y):
    for i in range(len(a)):
        plt.plot(a[i])
    plt.xlabel('segment #')
    plt.ylabel(y) 
    plt.title(title)
    plt.savefig(title + y + "individ.pdf")
    plt.show()

def normalize(a):
    maxx = np.max(a)
    for i in range(len(a)):
        a[i] = (a[i])/maxx
    return a

def normalize_multi(a):
    for i in range(len(a)):
        a[i] = normalize(a[i])
    return a

def matrix_float_model(a):
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append((float(a[0][i])))
    return new_matrix

def matrix_float(a):
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append((float(a[i][0])))
    return new_matrix

def matrix_avg(matrix):
    # Just averages trials together
    new_matrix_avg = []
    if len(np.shape(matrix)) <= 1:
        return matrix
    for j in range(len(matrix[0])):
        local_avg = []
        for i in range(len(matrix)):
            local_avg.append(matrix[i][j])
        new_matrix_avg.append(np.average(local_avg))
    return new_matrix_avg

def matrix_std(matrix):
    # Just finds of standard deviation of trials
    new_matrix_avg = []
    if len(np.shape(matrix)) <= 1:
        for i in range(len(matrix)):
            new_matrix_avg.append(0)
        return new_matrix_avg
    
    for j in range(len(matrix[0])):
        local_avg = []
        for i in range(len(matrix)):
            local_avg.append(matrix[i][j])
        new_matrix_avg.append(np.std(local_avg))
    return new_matrix_avg

def matrix_float_multi(a):
    new_matrix = [[0 for i in range(len(a[0]))] for j in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            if new_matrix[i][j] == 'NaN':
                new_matrix[i][j] = 'NaN'
            else:           
                new_matrix[i][j] = float(a[i][j])
    return new_matrix

def plot_compare(control, a, control_std, a_std,y, x_gray, y_gray):
    x_val = []
    for i in range(len(control)):
        x_val.append(i+1)
    ax1 = plt.axes(frameon=False)
    x_gray = x_gray
    y_gray = y_gray
    gray_std = 0
    plt.errorbar(x_val, control, yerr = control_std, fmt = '-o')
    plt.errorbar(x_val, a, yerr = a_std, fmt = '-o', color = 'orange')
    plt.errorbar(x_gray, y_gray, yerr = gray_std, fmt = '-o', color = 'gray')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))    
    plt.xlabel('Segment #', fontsize = globalFont)
    plt.ylabel(y, fontsize = globalFont) 
    plt.title(title)
    plt.savefig(savename + y + "comparison.pdf")
    plt.show()
    
def plot_compare_two_axis(control, a, control_std, a_std,y):
    x_val = []
    fig1 = plt.figure()
    for i in range(len(control)):
        x_val.append(i+1)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax1.set_ylabel(y, color='blue')
    plt.errorbar(x_val, control, yerr = control_std, fmt = '-o')
    xmin, xmax = ax2.get_xaxis().get_view_interval()
    ymin, ymax = ax2.get_yaxis().get_view_interval()
    plt.errorbar(x_val, a, yerr = a_std,  fmt = '-o')
    ax2.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='blue', linewidth=4)) 
    ax2.add_artist(Line2D((xmax, xmax), (ymin, ymax), color='orange', linewidth=4)) 
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Normalized neuron connections', color='orange')
    plt.title(title)
    ax1.set_xlabel('Segment #')
    plt.savefig(savename + "comparison.pdf")
    plt.show()

def plot_errorbars_single(control, control_std,y, save = False):
    x_val = []
    for i in range(len(control)):
        x_val.append(i +1)
    ax1 = plt.axes(frameon=False)
    plt.errorbar(x_val, control, yerr = control_std, fmt = '-o')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))    
    plt.xlabel('Segment #')
    plt.ylabel(y) 
    plt.title(title)
    if save == True:
        plt.savefig(savename + y + "single.pdf")
    plt.show()    
    return

def create_sine(sampling_rate, frequency, amplitude):
    Fs = sampling_rate ## sample rate
    f = frequency ## signal frequency
    sample = sampling_rate
    x = np.arange(1,sample +1)
    amp = amplitude ##amplitude
    y = amp*np.sin(2 * np.pi * f * x / Fs)
    plt.plot(x, y)
    plt.xlabel('sample(n)')
    plt.ylabel('voltage(V)')
    plt.show()
    return y

def add_noise(a, g_noise):
    # Check to make sure the noise and matrix are the same length
    if len(a) == len(g_noise):
        print('signal and noise same length')
    else:
        print('ERROR: signal and noise NOT same length')
    for i in range(len(a)):
        a[i] = a[i] +g_noise[i]
    return a

def plot_segs(a):
    for i in range(len(a)):
        plt.plot(a[i])
        plt.show()
     
def segment_difference(a,b):
    a_avg = np.mean(a)
    b_avg = np.mean(b)
    difference = b_avg - a_avg
    return difference 

def combine(a, b):
    total_len = len(a) + len(b)
    combined =[[0 for i in range(2)] for j in range(total_len)]
    for i in range(len(a)):
        combined[i] = a[i]
        combined[i] = a[i]
    for i in range(len(b)):
        index = i +len(a)
        combined[index] = b[i]
        combined[index] = b[i]
    return combined

def calc_experimental_mean(a, pos_1 = 4, pos_2 = 5):
    one = []
    two = []
    for i in range(len(a)):
        one.append(a[i][pos_1-1])
        two.append(a[i][pos_2-1])
    value = segment_difference(one,two)
    return value

def compile_data(a, pos_1 = 4, pos_2 = 5):
    one = []
    two = []
    for i in range(len(a)):
        one.append(a[i][pos_1-1])
        two.append(a[i][pos_2-1])
    combined = combine(one, two)
    return combined

def splice_data(a):
    ran_matrix = []
    for i in range(len(a)):
        ran = random.random()
        ran_matrix.append(ran)
    a.sort(key=dict(zip(a, ran_matrix)).get)
    half = int(len(a)/2)
    one = a[0:half]
    two = a[half:len(a)]
    curr_diff = segment_difference(one, two)  
    return curr_diff

def simulation(iterations = 1000, pos_1 = 4, pos_2 = 5):
    averages = []
    combined = compile_data(control_mi, pos_1 = pos_1, pos_2 = pos_2)
    for i in range(iterations):
        curr_avg = splice_data(combined)
        averages.append(curr_avg)
    return averages

def calc_p_value(avg_matrix, exp_mean):
    above_mean = 0
    for i in range(len(avg_matrix)):
        if avg_matrix[i]>= exp_mean:
            above_mean +=1
    p_value = above_mean/len(avg_matrix)
    return p_value

def interations(a, interations, I, J):
    for i in range(interations):
        curr_J = int(J + i)
        a[I][curr_J] = a[curr_J][I]
    return a

def matrix_symmetry(a):    
    tot_iterations = len(a)-1
    I_counter = 0
    J_counter = 1
    while tot_iterations > 0:
        a = interations(a, tot_iterations, I_counter, J_counter)
        J_counter+=1
        I_counter+=1
        tot_iterations = tot_iterations -1           
    return a

def heatmap_CC():
    MAXX = 1
    scale = []
    num_tics = 3
    for i in range(num_tics):
        curr_num = MAXX/num_tics*(i+1)
        scale.append(curr_num)    
    heatmap = []
    for i in range(13):
        heatmap.append(i)
    for i in range(13):
        input_name = nameOfHeatmapKind + str(i) +'_CC_PEAK_total.csv'
        with open(input_name, newline='\n' ) as inputfile:
            cc = list(csv.reader(inputfile))      
        cc = matrix_float_multi(cc)
        cc_avg = matrix_avg(cc)
        cc_avg = normalize(cc_avg)
        heatmap[i] = cc_avg
    heatmap = matrix_symmetry(heatmap)
        
    x_axis = []
    boolean = True
    for i in range(13):
        if boolean == True:
            x_axis.append(i+1) 
            boolean = False
        else: 
            x_axis.append(' ')
            boolean = True
    seaborn.heatmap(heatmap, cmap = 'Blues',vmax =MAXX,xticklabels = x_axis, yticklabels = x_axis, cbar_kws ={'ticks':scale})
    plt.xlabel('Segment number',fontsize = globalFont)  
    plt.ylabel('Segment number', fontsize = globalFont)         
    plt.savefig("heatmap_" + nameOfHeatmapKind+"_CC.pdf")
    plt.show()
    return heatmap

def heatmap_MI():
    MAXX = 1
    scale = []
    num_tics = 3
    for i in range(num_tics):
        curr_num = MAXX/num_tics*(i+1)
        scale.append(curr_num)
    heatmap = []
    for i in range(13):
        heatmap.append(i)
    for i in range(13):
        input_name = nameOfHeatmapKind + str(i) +'_MI_total.csv'
        with open(input_name, newline='\n' ) as inputfile:
            cc = list(csv.reader(inputfile))      
        cc = matrix_float_multi(cc)
        cc_avg = matrix_avg(cc)
        cc_avg = normalize(cc_avg)
        heatmap[i] = cc_avg
    heatmap = matrix_symmetry(heatmap)
    x_axis = []
    boolean = True
    for i in range(13):
        if boolean == True:
            x_axis.append(i+1) 
            boolean = False
        else: 
            x_axis.append(' ')
            boolean = True
    np.savetxt(savename+"2C_insert.csv", heatmap, delimiter=",")   
    seaborn.heatmap(heatmap, cmap = 'Blues',vmax =MAXX,xticklabels = x_axis, yticklabels = x_axis, cbar_kws ={'ticks':scale})
    plt.xlabel('Segment number',fontsize = globalFont)  
    plt.ylabel('Segment number', fontsize = globalFont)         
    plt.savefig("heatmap_" + nameOfHeatmapKind+"_MI.pdf")
    plt.show()
    return heatmap


cc = matrix_float_multi(cc)
cc_peak = matrix_float_multi(cc_peak)
mi = matrix_float_multi(mi)
cc_avg = matrix_avg(cc)
cc_std = matrix_std(cc)
mi_avg = matrix_avg(mi)
mi_std = matrix_std(mi)
cc_peak_avg = matrix_avg(cc_peak)
cc_peak_std = matrix_std(cc_peak)
control_mi = matrix_float_multi(control_mi)
control_cc = matrix_float_multi(control_cc)
control_cc_peak = matrix_float_multi(control_cc_peak)
control_cc_avg = matrix_avg(control_cc)
control_cc_peak_avg = matrix_avg(control_cc_peak)
control_mi_avg = matrix_avg(control_mi)
control_cc_std = matrix_std(control_cc)
control_cc_peak_std = matrix_std(control_cc_peak)
control_mi_std = matrix_std(control_mi)
plot_compare(control_cc_avg, cc_avg,control_cc_std, cc_std,y = 'Time lag to peak cross correlation (s)',x_gray=  1, y_gray=0)
plot_compare(control_cc_peak_avg, cc_peak_avg, control_cc_peak_std, cc_peak_std, y = 'Normalized peak cross correlation', x_gray = 1, y_gray =1 )
plot_compare(control_mi_avg, mi_avg, control_mi_std, mi_std, y = 'Normalized mutual information',x_gray= 1,y_gray =1)
heatmapp_CC = heatmap_CC()
heatmap_MI()
