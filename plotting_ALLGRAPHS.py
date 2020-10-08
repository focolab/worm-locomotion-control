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
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from math import atan2
from math import pi
from scipy import signal
from sklearn import metrics
from sklearn import feature_selection
from pyitlib import discrete_random_variable as drv
import h5py
import scipy.io as sio
from math import*
import math
from matplotlib.lines import Line2D
from six import iteritems
import random
import seaborn



########################################
#######################################
#           REPLACE 4things   #######
########################################
#########################################
title = ''
savename = 'N2_FORWARDS'
numbering_label = 15
globalFont = 20
plt.rc('xtick',labelsize=numbering_label)
plt.rc('ytick',labelsize=numbering_label)
nameOfHeatmapKind = 'N2_FORWARDS'
###############
###############
####     MUTANT  
###############
############### 

with open('modelcc_delay_model.csv', newline='\n' ) as inputfile:
   cc = list(csv.reader(inputfile))  
with open('N2 SWIMcc_peak_model.csv', newline='\n' ) as inputfile:
   cc_peak = list(csv.reader(inputfile)) 
with open('N2 SWIMmut_info_model.csv', newline='\n' ) as inputfile:
   mi = list(csv.reader(inputfile))   
   
###############
###############
####     CONTROL  
###############
############### 
  
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
#plot_all(cc, y = 'time lag to peak cross correlation')
#plot_all(mi, y = 'mutual information')
#plot_all(cc_peak, y = 'peak cross correlation')
def normalize(a):
    maxx = np.max(a)
    minn = np.min(a)
    mean = np.mean(a)

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
    ##compress the individual transitions for easier plotting
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
    ##compress the individual transitions for easier plotting
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

#cc = matrix_float(cc)
#cc_peak = matrix_float(cc_peak)
#mi = matrix_float(mi)

cc = matrix_float_multi(cc)
cc_peak = matrix_float_multi(cc_peak)
mi = matrix_float_multi(mi)
#cc = normalize_multi(cc)
#cc = matrix_float_model(cc)
#mi = matrix_float_model(mi)
#cc_peak = matrix_float_model(cc_peak)



cc_avg = matrix_avg(cc)
cc_std = matrix_std(cc)


mi_avg = matrix_avg(mi)
mi_std = matrix_std(mi)
#mi_avg = normalize(mi_avg)

cc_peak_avg = matrix_avg(cc_peak)
cc_peak_std = matrix_std(cc_peak)
#cc_peak_avg = normalize(cc_peak_avg)



control_mi = matrix_float_multi(control_mi)
control_cc = matrix_float_multi(control_cc)
control_cc_peak = matrix_float_multi(control_cc_peak)
#control_cc = normalize_multi(control_cc)


control_cc_avg = matrix_avg(control_cc)
#control_cc_avg = normalize(control_cc_avg)

control_cc_peak_avg = matrix_avg(control_cc_peak)
control_mi_avg = matrix_avg(control_mi)
#control_mi_avg = normalize(control_mi_avg)
#control_cc_peak_avg = normalize(control_cc_peak_avg)


control_cc_std = matrix_std(control_cc)
control_cc_peak_std = matrix_std(control_cc_peak)
control_mi_std = matrix_std(control_mi)


def plot_compare(control, a, control_std, a_std,y, x_gray, y_gray):
    x_val = []
    for i in range(len(control)):
        x_val.append(i+1)
    ax1 = plt.axes(frameon=False)
    x_gray = x_gray
    y_gray = y_gray
    gray_std = 0
    
    #control = control[1:len(control)]
    #control_std = control_std[1:len(control_std)]
    #a = a[1:len(a)]
    #a_std = a_std[1:len(a_std)]
    

    plt.errorbar(x_val, control, yerr = control_std, fmt = '-o')
    plt.errorbar(x_val, a, yerr = a_std, fmt = '-o', color = 'orange')
    plt.errorbar(x_gray, y_gray, yerr = gray_std, fmt = '-o', color = 'gray')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))    

    plt.xlabel('Segment #', fontsize = globalFont)
    #ax1.set_xticklabels(x_ticks, rotation=0, fontsize=80)
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
    #plt.ylabel(y) 
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

#plot_errorbars_single(control_cc_avg, control_cc_std, 'Time lag to peak cross correlation', save = True)
#plot_errorbars_single(control_cc_peak_avg, control_cc_peak_std, 'Normalized peak cross correlation', save = True)
#plot_errorbars_single(control_mi_avg, control_mi_std, 'Normalized peak mutual info', save = True)
    
    
    
plot_compare(control_cc_avg, cc_avg,control_cc_std, cc_std,y = 'Time lag to peak cross correlation (s)',x_gray=  1, y_gray=0)
plot_compare(control_cc_peak_avg, cc_peak_avg, control_cc_peak_std, cc_peak_std, y = 'Normalized peak cross correlation', x_gray = 1, y_gray =1 )
plot_compare(control_mi_avg, mi_avg, control_mi_std, mi_std, y = 'Normalized mutual information',x_gray= 1,y_gray =1)

#plot_compare_two_axis(control_mi_avg, mi_avg, control_mi_std, mi_std, y = 'Normalized mutual information')

freq = 5
sine_length = 8000
sine_freq = 5
noisyness =.2
delay_per_seg = 100
num_segments = 49 
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
    #check to make sure the noise and matrix are the same length
    if len(a) == len(g_noise):
        print('signal and noise same length')
    else:
        print('ERROR: signal and noise NOT same length')
    for i in range(len(a)):
        a[i] = a[i] +g_noise[i]
    #x = np.arange(1,len(a) +1)
    #plt.plot(x, a)
    #plt.xlabel('time')
    #plt.ylabel('cross correlation')
    #plt.show()
    return a
def delay_single_seg(a, delay_per_seg, curr_seg):
    b = []
    for i in range(len(a)):
        b.append(a[i])
    c = []
    for i in range(len(a)):
        c.append(a[i])
    #print(delay_per_seg*curr_seg)
    for i in range(delay_per_seg*curr_seg):
        b.append(a[i])
    ##copy the beginning of a to the end of b 'delay per seg' times'
    for i in range(len(a)):
        c[i] = b[i + delay_per_seg*curr_seg]
    #print(a[i])
    return c
def create_delay_segments(a, num_segments, delay_per_seg):
    segments = [0 for i in range(num_segments)]
    #create segments with the same freq 
    for i in range(len(segments)):
        segments[i] = a
    ### time delay each segment
    ### create and add noise for individual segments
    for i in range(len(segments)):
        segments[i] = delay_single_seg(segments[0], delay_per_seg, i)
        curr_noise = np.random.normal(0,noisyness, sine_length)
        segments[i] = add_noise(segments[i], curr_noise)
    return segments
def plot_segs(a):
    for i in range(len(a)):
        plt.plot(a[i])
        plt.show()
       
        
#sine = create_sine(sine_length, sine_freq, 1)
#noise = np.random.normal(0,noisyness, sine_length) #(mean of normal, std from normal, # elements)
#add_noise(sine, noise)
#ABBA = create_delay_segments(sine, num_segments, delay_per_seg)

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


#four_five_mean = calc_experimental_mean(control_mi)
#averages_4_5 = simulation(iterations = 10000000)
#p_val_4_5 = calc_p_value(averages_4_5, four_five_mean)


#nine_seven_mean = calc_experimental_mean(control_mi, pos_1 = 7, pos_2 = 9)
#nine_seven_avgs = simulation(iterations = 10000000, pos_1 = 7, pos_2 = 9)
#p_val_7_to_9 = calc_p_value(nine_seven_avgs, nine_seven_mean)
def heatmap_CC():
    MAXX = 1
    scale = []
    num_tics = 3
    for i in range(num_tics):
        curr_num = MAXX/num_tics*(i+1)
        scale.append(curr_num)
    print(scale)
    
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
        print(cc_avg)
       
        heatmap[i] = cc_avg
    x_axis = []
    boolean = True
    for i in range(13):
        if boolean == True:
            x_axis.append(i+1) 
            boolean = False
        else: 
            x_axis.append(' ')
            boolean = True
   # (..., cbar_kws={"ticks":[0.25,1]})        
    seaborn.heatmap(heatmap, vmax =MAXX,xticklabels = x_axis, yticklabels = x_axis, cbar_kws ={'ticks':scale})
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
    print(scale)
    
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
    x_axis = []
    boolean = True
    for i in range(13):
        if boolean == True:
            x_axis.append(i+1) 
            boolean = False
        else: 
            x_axis.append(' ')
            boolean = True
   # (..., cbar_kws={"ticks":[0.25,1]})        
    seaborn.heatmap(heatmap, vmax =MAXX,xticklabels = x_axis, yticklabels = x_axis, cbar_kws ={'ticks':scale})
    plt.xlabel('Segment number',fontsize = globalFont)  
    plt.ylabel('Segment number', fontsize = globalFont)         
       
    plt.savefig("heatmap_" + nameOfHeatmapKind+"_MI.pdf")
    plt.show()
    
    return heatmap

heatmapp_CC = heatmap_CC()
heatmap_MI()

#x_gray = 1
#y_gray = 1
#plt.scatter(x_gray, y_gray, c = 'grey')