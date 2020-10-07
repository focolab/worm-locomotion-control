#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:36:36 2018

@author: jmw
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from math import atan2
from math import pi
from math import ceil
from scipy import signal
from sklearn import metrics
from sklearn import feature_selection
from pyitlib import discrete_random_variable as drv
import h5py
import scipy.io as sio
from matplotlib.lines import Line2D
from six import iteritems
from scipy.fftpack import fft
from scipy.fftpack import fftfreq


#with open('skel10_x.csv', 'rU', newline='\n') as inputfile:
#   results_x = list(csv.reader(inputfile))
#with open('skel10_y.csv', 'rU', newline='\n' ) as inputfile:
#    results_y = list(csv.reader(inputfile))


###################################################################################
#NAME = 'wormskeleton_x.txt'
#with open('wormskelx.csv', newline='\n') as inputfile:
#    results_x = list(csv.reader(inputfile))
#with open('wormskely.csv', newline='\n' ) as inputfile:
#    results_y = list(csv.reader(inputfile))
###################################################################################
###################################################################################
###################################################################################
###################################################################################
sampling_rate = 30 ## not really relevant anymore. The sampling rate is now in the file you import
corr_div =1 ### how many divisions you want the time series divided into for analysis
corr_seg = 0 ## this is the segment # the MI and cross correlation compare to 
bins = 2 ## MI bin number
savename = 'SWim BEttwer' #in the saved filenames
interval = 3.69 #segment interval, to skip some segments 50/3.69 = 13 segments
instructions = 'n2_swim_cleanup_BEtter.csv' ##FILE with the macro-level time points
backwards = False  ##forwards or backwards

###################################################################################
###################################################################################
###################################################################################
###################################################################################

#These are the variables for cross correlation quality control

cc_ctrl_name = 'unc37_10.hdf5' 
START = 7748.4
STOP = 8800
#rate = 25.6
DURATION = int(STOP-START)
print('Start frame: ' + str(START))
print('Duration : ' + str(DURATION))
print('Sampling rate: ' + str(sampling_rate))
##########################################################################
##########################################################################


## These variables are for the sine generation in the head-first model
freq = 5
sine_length = 8000
sine_freq = 5
noisyness =0.1
delay_per_seg = 10
num_segments = 49
frame_rate = 300
##############################################################w
print('||||||global variables||||||')
print('mutual info bins: ' + str(bins))
print('video segments averaged: ' + str(corr_div))
print('sampling rate: ' + str(frame_rate))
print('segment to compare to: ' + str(corr_seg))
###########################################################
def absolute(a):
    ##returns a matrix of absolute numbers
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(np.absolute(a[i]))
    return new_matrix
def normalize_peak_cc(a):
    ##max-min normalization 
    minn = np.min(a)
    for i in range(len(a)):
        a[i] = a[i] + np.abs(minn)
    minn = np.min(a)
    maxx = np.max(a)   
    new_matrix = []
    for i in range(len(a)):
        num = a[i] - minn
        denom = maxx -minn
        if denom != 0:
            new_matrix.append(num/denom)
    return new_matrix
def import_h5py(name, start_frame, total_frames):
    #finds the skeleton timepoints in the .h5py file
    f = h5py.File(name, 'r')
    coord = f['coordinates']['skeletons']
    x = [0 for i in range(len(coord[0]))]
    y = [0 for i in range(len(coord[0]))]
    for i in range(len(x)):
        x[i] = []
        y[i] = []
    if len(coord) < total_frames:
        counter = len(coord)
    else:
        counter = total_frames 
    for i in range(len(coord[0])):
        for j in range(counter):
            x[i].append(coord[j + start_frame][i][0])        
            y[i].append(coord[j +start_frame][i][1]) 
    return x, y
def calc_worm_vel(a, Rate):
    ##finds worm velocity from the peak fourier transform
    x_axis = fftfreq(len(a),1/Rate)
    y_axis = fft(a)   
    ## can't have negative velocities for worm movement
    length = len(x_axis)
    midpt = int(length/2)
    x_axis = x_axis[0:midpt]
    y_axis = y_axis[0:midpt]
    #plt.plot(x_axis, y_axis)
    #plt.show()
    max_freq_index = np.argmax(y_axis)
    max_freq = x_axis[max_freq_index]
    
    return max_freq
def worm_vel_averaged(a, ratE):
    ##average the worm velocities
    
    vel_matrix = [0 for i in range(int(len(a)/interval))]    
    for i in range(int(len(a)/interval)):
        index = int(i*interval)
        vel_matrix[i] = calc_worm_vel(a[index], ratE)
    avg_vel = np.mean(vel_matrix)
    #print(vel_matrix)
    return avg_vel

def matlab_worm(name, start_frame, total_frames):
    ## an outdated way to import another file name
    
    data = sio.loadmat(name)
    coord = data['wormSegmentVector']
    x = [0 for i in range(len(coord[0][0]))]
    y = [0 for i in range(len(coord[0][0]))]
    for i in range(len(x)):
        x[i] = []
        y[i] = []
    counter = total_frames
    for i in range(len(coord[0][0])):
        for j in range(counter):
            x[i].append(coord[j + start_frame][0][i])
            y[i].append(coord[j + start_frame][1][i]) 
    return x,y

def matrix_float_multi(a):
   ## takes a nested matrix of string numbers and turns them into floats
    
    new_matrix = [[0 for i in range(len(a[0]))] for j in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            #if np.isnan(a[i][j]) == True:
           #     new_matrix[i][j] = nan
           # else:           
               a[i][j] = float(a[i][j])
    return a
def matrix_float(a):
    # Takes a matrix and returns floats
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append((float(a[i])))
    return new_matrix
def integer(a):
    #converts a matrix to integers
    new_matrix = []
    a = matrix_float(a)
    for i in range(len(a)):
        a[i] = round(a[i])
        new_matrix.append(int(a[i]))
    return new_matrix
def delete_beginning(a, amount = 1):
   #delete the first element in a matri
    for i in range(len(a)):
        a[i].pop(0)
    return a
def check_for_nan_multi(a):
    #delete 'NaN' in a nested matrix
    for i in range(len(a)):
        matrix = []
        for j in range(len(a[0])):
            if np.isnan(a[i][j]) == True:
                matrix.append(j)
        for j in range(len(matrix)):
            a = np.delete(a[i], matrix[j] -j)
    return a
def check_for_nan(a):
   # delete 'NaN' i a matrix
    matrix = []
    for i in range(len(a)):
        if np.isnan(a[i]) == True:
            matrix.append(i)
    for i in range(len(matrix)):
        a = np.delete(a, matrix[i] -i)
    return a  
def plot_worm(x, y, position = 0, save = False):
   #Plot the x,y coordiantes of the worm
   #position refers to the frame desired to plot
    curr_x = []
    curr_y = []
    for i in range(len(x)):
        curr_x.append(x[i][position])
    for i in range(len(y)):
        curr_y.append(y[i][position])
    #plt.plot(curr_x, curr_y)
    plt.axes(frameon=False)
    plt.scatter(curr_x, curr_y, color= 'black', s = 20)
    plt.axis('off')
    if save == True:
        plt.savefig('worm dots' + str(position))
    plt.show()
def calc_tangent(x, y, position = 0):
    #returns a matrix of tangents based on the x, y coordinates
    ## note: no longer used
    curr_x = []
    curr_y = []
    angles = []
    for i in range(len(x)):
        curr_x.append(x[i][position])
    for i in range(len(y)):
        curr_y.append(y[i][position])
    if np.isnan(curr_x[0]) == True:
        return 0
    for i in range(len(curr_x)-1):
        angles.append(atan2((curr_y[i]), (curr_x[i])))
    angle_mean = np.mean(angles)
    for i in range(len(angles)):
        angles[i] = angles[i] +angle_mean
    return angles
def calc_all_tangents(x,y):
   ##calculates the tagents for an x,y postion data series through time
   ##note no longer used
    angles = []
    for i in range(len(x[0])):
        angles.append(0)
    for i in range(len(angles)):
        angles[i] = calc_tangent(x, y, position = i)
    to_delete = []
    for i in range(len(angles)):
        if angles[i] == 0:
            to_delete.append(i)
    for i in range(len(to_delete)):
        angles = np.delete(angles, to_delete[i] -i)
    return angles
def destroyNAN(a):
    #delete the nans in a matrix
    to_delete = []
    new_matrix = []
    for i in range(len(a)):
       if np.isnan(a[i]) == True: 
           to_delete.append(i)
    for i in range(len(a)):
        if i not in to_delete:
            new_matrix.append(a[i])
    return new_matrix

def tangent(x,y):
    #calculate the tangent of an x,y data series
    for i in range(len(x)):
        x[i] = destroyNAN(x[i])
        y[i] = destroyNAN(y[i])
    angles = [[0 for i in range(len(x)-1)] for j in range(len(x[0]))]
    for j in range(len(x[0])):
        for i in range(len(x)-1):
            rise = y[i+1][j] -y[i][j]
            run = x[i +1][j] - x[i][j]
            angles[j][i] = atan2(rise, run)
    #length = len(angles)
    #skips = 0
    #for i in range(length):
    #    if np.isnan(angles[i-skips][0]) == True:
    #        angles.pop(i-skips) 
    #        skips +=1  
    for i in range(len(angles)):
        curr_mean = np.mean(angles[i])
        for j in range(len(angles[0])):
            angles[i][j] = angles[i][j] -curr_mean
    return angles  
def single_segment_series(a, segment = 0):
    #reformat the data so that a[i] represents a segments data through time
    time_series = []
    for i in range(len(a)):
        time_series.append(a[i][segment])
    return time_series
def total_segment_series(a):   
    #reformat the data so that a[i] represents a segments data through time
    
    time_series = []
    for i in range(len(a[0])):
        time_series.append(0)
    for i in range(len(a[0])):
        time_series[i] = single_segment_series(a, segment = i)
    return time_series
def time_create(seg_series, frame_rate):
    #create time matrix based on the frame rate and the length of the data points
    time = []
    for i in range(len(seg_series)*2-1):
        current_time = i /frame_rate -(len(seg_series)/frame_rate)
        time.append(current_time) 
    return time
def cross_correlate(time_series, frame_rate, interval=interval, corr_seg = corr_seg):
    #cross correlates. variables should be self-explanatory. interval and corr_seg are defined at the beginning of the program 
    # time _series is the tagents of each segment through time; frame rate is determined from the instructions file
    cross_matrix = []
    PEAK_cross = []
    time = time_create(time_series[0], frame_rate)
    #plt.plot(time)
    plt.show()
    #print('corr_seg' +str(corr_seg))
    #for i in range(len(time_series[0])+1):
    #    current_time = i/frame_rate
    #    time.append(current_time) 
    #for i in range(len(time_series[0])):
    #   current_time = -(len(time_series[0])/frame_rate) + i/frame_rate
    #   time.append(current_time) 
    compare_to = int(interval*corr_seg)
    if backwards == True:
        compare_to = int(len(time_series)/interval) - int(corr_seg*interval)
                        
    for i in range(int(len(time_series)/interval)):
        comparing = int(interval*i)
        #print('Compare_to'+ str(compare_to))
        #print('comparing' + str(comparing))
        if backwards == True:
            comparing = int((len(time_series)/interval)-comparing)
        s = signal.correlate(time_series[compare_to],time_series[comparing])
        # ss = s[len(s)/2:len(s)]
        #ss = absolute(ss)
        index = np.argmax(s)  ######### whole or just positive
        cross_matrix.append(time[index])
        #print(time[index])
        PEAK = np.correlate(time_series[compare_to], time_series[comparing])
        PEAK = PEAK[0]
        PEAK_cross.append(PEAK)
    return cross_matrix, PEAK_cross
def slice_time_segment(a, round = 0):
   #if corr_div is not equal to 1, this will slice up the time into the amount of divisions
    time_series = []
    seg_num = len(a)
    size = int(len(a[0])/corr_div)
    for i in range(seg_num):
        time_series.append(0)
    for i in range(seg_num):
        start = int(round*size)
        stop = int((round +1)*size)
        time_series[i] = a[i][start:stop]
    return time_series
def correlate_slice_calc(a, frame_rate, interval=interval, corr_seg = corr_seg):
    ## cross correlate every 
    
    corr_coef = [] #The time delay to peak cross correlation 
    PEAK_cross = [] #peak cross correlation
    for i in range(corr_div):
        corr_coef.append(0)
        PEAK_cross.append(0)
    for i in range(corr_div):
        curr_time_series = slice_time_segment(a, round = i)
        curr_corr_coef, curr_PEAK_cross = cross_correlate(curr_time_series, frame_rate, corr_seg = corr_seg)
        corr_coef[i] = curr_corr_coef
                 
        ##normalize         
        curr_PEAK_cross = normalize_peak_cc(curr_PEAK_cross)          
        PEAK_cross[i] = curr_PEAK_cross
    #print(PEAK_cross[0])
    corr_coef_avg = []
    PEAK_cross_avg = []
    for i in range(int(len(a)/interval)):
        running_total = 0
        running_peak_total = 0
        for j in range(len(corr_coef)):
            running_total = running_total + corr_coef[j][i]
            running_peak_total = running_peak_total + PEAK_cross[j][i]
        corr_coef_avg.append(running_total/corr_div)
        PEAK_cross_avg.append(running_peak_total/corr_div)
    
    return corr_coef_avg, PEAK_cross_avg
def calc_MI(x, y, bins):
    #calcuate the mutual info of a single segment to the 
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi
def mutual_info(a, bins = bins, interval = interval, corr_seg = corr_seg):
    #calculate the mutual info of all segments 
    
    score_matrix = []
    compare_to = int(interval*corr_seg)
    if backwards == True:
        compare_to = int(len(a)/interval) -int(corr_seg*interval)
    for i in range(int(len(a)/interval)):
        comparing = int(i*interval)
        if backwards == True:
            comparing = int((len(a)/interval) -comparing)
        curr_score = calc_MI(a[compare_to], a[comparing], bins)
        score_matrix.append(curr_score)
    return score_matrix
def plots_to_plot(a, y, save = False):
    time = []
    for i in range(len(a)):
        time.append(i +1)
    ax1 = plt.axes(frameon=False)
    plt.plot(time,a, '.-', markersize =10, linewidth  = 1)
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))    
    
    
    #plt.plot(time, a)
    #plt.ylim(-1,0)
    plt.xlabel('Segment number')
    plt.ylabel(y)  
    if save == True:
        plt.savefig(savename + y + ".png")
    plt.show()
def plots_to_plottt(a, y, save = False):
    time = []
    for i in range(len(a)):
        time.append(i +1)
    ax1 = plt.axes(frameon=False)
    plt.plot(time,a, '.-', markersize = 10, linewidth  = 1)
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    #plt.ylim(-1,0)
    plt.xlabel('Segment number')
    plt.ylabel(y)  
    if save == True:
        plt.savefig(savename + y + ".png")
    plt.show()    
def multiply_matrix(a, multiplier = 1000):
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j] = a[i][j]*multiplier
    return a
def mut_info_total(a, interval= interval, corr_seg = corr_seg):
    ##calculate the mutual info for every segment 
    
    corr_coef = []
    for i in range(corr_div):
        corr_coef.append(0)
    for i in range(corr_div):
        curr_time_series = slice_time_segment(a, round = i)
        curr_corr_coef = mutual_info(curr_time_series, corr_seg = corr_seg)
        corr_coef[i] = curr_corr_coef        
    corr_coef_avg = []
    for i in range(int(len(a)/interval)):
        running_total = 0
        for j in range(len(corr_coef)):
            running_total = running_total + corr_coef[j][i]
        corr_coef_avg.append(running_total/corr_div)
    ##normalize it
    maxx = np.max(corr_coef_avg)
    minn = np.min(corr_coef_avg)
    for i in range(len(corr_coef_avg)):
        corr_coef_avg[i] = (corr_coef_avg[i] - minn)/(maxx-minn)
    return corr_coef_avg
def plot_segments(a, divisions = 3):
   ##plot segments through time. divisons refers to how many equally-spaced segments to plot
    
    time = []
    for i in range(len(a)):
        current_time = i/sampling_rate
        time.append(current_time) 
    segments = []
    div_len = len(a[0])/divisions
    start = div_len /2
    print(segments)
    for i in range(divisions):
        value = int(i*div_len +start)
        segments.append(value)
    print(segments)
    for i in range(len(segments)):
        current_sinwave = []
        for j in range(len(a)):
            curr_seg = segments[i]
            current_sinwave.append(a[j][curr_seg])
        plt.plot(time, current_sinwave)
    plt.xlabel('time')
    plt.ylabel('radians')    
    plt.show()
# plot_segments(AAA)
def plot_cross_correlation(a, seg_num = 0):
    #plot the cross correlation of a given segment 
    time = []
    for i in range(len(a[0])*2):
        time.append(i)     
    for i in range(len(a)):       
        cross = signal.correlate(a[0], a[i])
        time.pop()
        plt.plot(cross)
        plt.show()
        print('seg num: ' + str(i))
        print(np.argmax(cross))
#plot_cross_correlation(segment_series)    
def plot_all_segs(a):
    ## plot all of the worm segments through time
    for i in range(len(a)):
        print('segment: '+str(i))
        plt.plot(a[i])
        #plt.ylim(-.1,.1)
        plt.show()
#plot_all_segs(segment_series)
def matrix_avg(matrix):
    ##compress the individual transitions for easier plotting
    if len(np.shape(matrix)) <= 1:
        return matrix
    new_matrix_avg = []
    for j in range(len(matrix[0])):
        local_avg = []
        for i in range(len(matrix)):
            local_avg.append(matrix[i][j])
        new_matrix_avg.append(np.average(local_avg))
    return new_matrix_avg
def extract_instructions(name):
    #import instructiona and turn them into seperate matrixes
    
    #def all_at_once(name)
    with open(name, newline='\n') as inputfile:
        instructions = list(csv.reader(inputfile))
        ## create a list of names
    name_list = []
    frame_rate_list = []
    for i in range(len(instructions)-1):
            name_list.append(instructions[i+1][0])
            frame_rate_list.append(instructions[i+1][3])
    ##create a list of start frames, convert it to floats
    start_list = []
    stop_list = []
    for i in range(len(instructions)-1):
        start_list.append(instructions[i+1][1])
        stop_list.append(instructions[i+1][2])        
    start_list = integer(start_list)  
    stop_list = integer(stop_list)
    frame_rate_list = integer(frame_rate_list)
    return name_list, start_list, stop_list, frame_rate_list
def correct_cc_lag_sign(a):
   #flip the signs on everything by mulitplyig by negative one
    for i in range(len(a)):
        a[i] = a[i]*-1
    return a
def normalize_cross_time_lag(a, vel):
    #normalize time deltay to peak cross correlation  
    for i in range(len(a)):
        a[i] = a[i]*vel
    return a
def run_once(name, start_frame, stop_frame, frame_rate, corr_seg = corr_seg):
    #run everything on a single worm 
    
    print('file name: ' + name)
    print('start frame: ' + str(start_frame))
    print('stop frame: ' + str(stop_frame))
    print('segment to compare to: '+ str(corr_seg))
    total_frames = stop_frame - start_frame
    if name[-1] == '5':
        results_x, results_y = import_h5py(name, start_frame, total_frames)
    else: 
        results_x, results_y = matlab_worm(name, start_frame, total_frames)
    pos_x = matrix_float_multi(results_x)
    pos_y = matrix_float_multi(results_y)
    #a = tangent(pos_x, pos_y)
    AAA = tangent(pos_x,pos_y)  
    segment_series = total_segment_series(AAA)
    #cross = cross_correlate(segment_series) 
    cross_avg,cross_peak_avg = correlate_slice_calc(segment_series, frame_rate, corr_seg = corr_seg)
    worm_vel = worm_vel_averaged(segment_series, ratE = frame_rate)
    cross_avg = correct_cc_lag_sign(cross_avg)
    cross_avg = normalize_cross_time_lag(cross_avg, worm_vel)
    #cross_peak_avg = normalize_peak_cc(cross_peak_avg)
    plots_to_plot(cross_avg, y = 'peak cross correlation time lag'+name, save = False)
    plots_to_plot(cross_peak_avg, y = 'peak cross correlation')
    mut_info_avg = mut_info_total(segment_series, corr_seg = corr_seg)
    plots_to_plot(mut_info_avg, y = 'normalized mutual info'+name, save = False)
    return cross_avg, mut_info_avg, cross_peak_avg, worm_vel
#run_once(name, start_frame, stop_frame)
#run_once(name, 500, 1000)
def combine_worms(name_list, start_list,stop_list, frame_rate_list, corr_seg=corr_seg):
    ## create cross correlation average matrix and mutual info 
    cross_total = [0 for i in range(len(name_list))]
    cross_peak_total = [0 for i in range(len(name_list))]
    mi_total =[0 for i in range(len(name_list))]
    vel_total =[0 for i in range(len(name_list))]
    ##check to make sure that the frames are the same
    total_frames = []
    for i in range(len(start_list)):
        total_frames.append(stop_list[i]-start_list[i])
    minimum_frames = np.min(total_frames)
    for i in range(len(total_frames)):
        if total_frames[i] > minimum_frames:
            stop_list[i] = start_list[i] +minimum_frames
    ## calculate the cross correlation and mutual info for all
    #the items in the list
    for i in range(len(name_list)):
        print('   ')
        print('list location: ' +str(i+1))
        local_cc, local_mi, local_cross_peak, local_vel = run_once(name_list[i], start_list[i], stop_list[i], frame_rate_list[i], corr_seg = corr_seg)         
        ## normalize the cross correlation
        #maxx = np.max(local_cc)
        #minn = np.min(local_cc)
        #for j in range(len(local_cc)):
        #    local_cc[j] = (local_cc[j] - minn)/(maxx-minn)
        #assign the local c. correlation to the larger matrix    
        cross_total[i] = local_cc
        mi_total[i] = local_mi 
        cross_peak_total[i] = local_cross_peak
        vel_total[i] = local_vel          
    cross_avg = matrix_avg(cross_total) # find the average matrix
    mi_avg = matrix_avg(mi_total) #
    cross_peak_avg = matrix_avg(cross_peak_total)
    print()
    print()
    print('totals: ')
    plots_to_plot(cross_avg, y = 'TOTAL peak cross correlation time lag', save = False)
    plots_to_plot(cross_peak_avg, y = 'TOTAL normalized peak cross correlation', save = False)
    plots_to_plot(mi_avg, y = 'TOTAL normalized mutual info', save = False)
    np.savetxt(savename+"_MI_total.csv", mi_total, delimiter=",")
    np.savetxt(savename+"_CC_timelag_total.csv", cross_total, delimiter=",")
    np.savetxt(savename+"_CC_PEAK_total.csv", cross_peak_total, delimiter=",")
    return {'cross_total':cross_total, 'mi_total':mi_total, 
            'cross_avg':cross_avg, 'mi_avg':mi_avg, 'cross_peak_total':cross_peak_total,
            'cross_peak_avg':cross_peak_avg, 'vel_total':vel_total}
def create_sine(sampling_rate, frequency, amplitude, show_plot = True):
    ##creates a sine wave by parameters used at the beginning of the program
    
    Fs = sampling_rate ## sample rate
    f = frequency ## signal frequency
    sample = sampling_rate
    x = np.arange(1,sample +1)
    amp = amplitude ##amplitude
    y = amp*np.sin(2 * np.pi * f * x / Fs)
    if show_plot == True:
        plt.plot(x, y, c = 'black')
        #plt.xlabel('sample(n)')
        #plt.ylabel('voltage(V)')
        plt.axis('off')
        plt.savefig('sine.png')
        plt.show()
    return y
def add_noise(a, g_noise):
    ##adds noise
    
    for i in range(len(a)):
        a[i] = a[i] +g_noise[i]
    return a
def delay_single_seg(a, delay_per_seg, curr_seg):
    ## delay a segment by delay_per_seg
    
    b = []
    new_matrix = []
    for i in range(len(a)):
        b.append(a[i])
    for i in range(len(a)-delay_per_seg):
        new_matrix.append(a[i])
    #print(delay_per_seg*curr_seg)
    #for i in range(delay_per_seg*curr_seg):
    #    b.append(a[i])
    ##copy the beginning of a to the end of b 'delay per seg' times'
    for i in range(len(b)-delay_per_seg):
        new_matrix[i] = b[i + delay_per_seg]
    return new_matrix
def adjust_len(a):
    max_len = len(a[-1])
    for i in range(len(a)):
       a[i] = a[i][0:max_len]
    return a
def create_delay_segments(a, num_segments, delay_per_seg):
    segments = [0 for i in range(num_segments)]
    #create segments with the same freq 
    segments[0] = a
    curr_noise = np.random.normal(0,noisyness, sine_length)
    segments[0] = add_noise(segments[0], curr_noise)
    ### time delay each segment
    ### create and add noise for individual segments
    for i in range(len(segments)-1):
        segments[i+1] = delay_single_seg(segments[i], delay_per_seg, i)
        curr_noise = np.random.normal(0,noisyness, sine_length)
        segments[i+1] = add_noise(segments[i+1], curr_noise)
    segments = adjust_len(segments)
    return segments
def plot_segs(a):
    for i in range(len(a)):
        plt.plot(a[i])
        plt.show()
def model_head_first():        
    sine = create_sine(sine_length, sine_freq, 1)
    #noise = np.random.normal(0,noisyness, sine_length) #(mean of normal, std from normal, # elements)
    #add_noise(sine, noise)
    ABBA = create_delay_segments(sine, num_segments, delay_per_seg)
    cross_avg, cross_peak_avg = correlate_slice_calc(ABBA, sampling_rate)
    plots_to_plottt(cross_avg, y = 'time delay to peak cross correlation', save = False)
    plots_to_plottt(cross_peak_avg, y = 'peak cross correlation', save = False)
    mut_info_avg = mut_info_total(ABBA)
    plots_to_plottt(mut_info_avg, y = 'normalized mutual info', save = False)
    np.savetxt(savename+"mut_info_model.csv", mut_info_avg, delimiter=",")   
    np.savetxt(savename+"cc_delay_model.csv", cross_avg, delimiter=",")   
    np.savetxt(savename+"cc_peak_model.csv", cross_peak_avg, delimiter=",")   



def repeat_model(cycles = 10):
    ##create empty variables 
    MI_total = [0 for i in range(cycles)]
    cross_timelag_total = [0 for i in range(cycles)]
    cross_peak_total = [0 for i in range(cycles)] 
    #run the thing for however many cycles 
    for i in range(cycles):
        sine = create_sine(sine_length, sine_freq, 1, show_plot = False)    
        ABBA = create_delay_segments(sine, num_segments, delay_per_seg)
        cross_timelag, cross_peak = correlate_slice_calc(ABBA, sampling_rate)
        mut_info = mut_info_total(ABBA)
        plots_to_plottt(cross_timelag, y = 'Time delay to peak cross correlation', save = False)
        plots_to_plottt(cross_peak, y = 'Normalized peak cross correlation', save = False)
        plots_to_plottt(mut_info, y = 'Normalized mutual info', save = False) 
        MI_total[i] = mut_info
        cross_timelag_total[i] = cross_timelag
        cross_peak_total[i] = cross_peak
    print(cross_timelag_total)                    
    cross_peak_avg = matrix_avg(cross_peak_total) # find the average matrix
    mut_info_avg = matrix_avg(MI_total) #
    cross_timelag_avg = matrix_avg(cross_timelag_total)  
    print(cross_timelag_avg)
    plots_to_plottt(mut_info_avg, y = str(cycles) +  ' Averaged normalized mutual info', save = True)    
    plots_to_plottt(cross_timelag_avg, y = str(cycles) + 'Averaged time delay to peak cross correlation', save = True)
    plots_to_plottt(cross_peak_avg, y = str(cycles) + 'Averaged peak cross correlation', save = True) 
    np.savetxt(savename+"mut_info_model.csv", MI_total, delimiter=",")   
    np.savetxt(savename+"cc_delay_model.csv", cross_timelag_total, delimiter=",")   
    np.savetxt(savename+"cc_peak_model.csv", cross_peak_total, delimiter=",")
#repeat_model(cycles = 1)

#sine = create_sine(sine_length, sine_freq, 1, show_plot = True) 
#ABBA = create_delay_segments(sine, num_segments, delay_per_seg)

#x_val = []
#for i in range(len(ABBA[0])):
#    x_val.append(i)

#plt.scatter(x_val, ABBA[10], c = 'black', s = .1)
#plt.axis('off')
#plt.savefig('messy sine.png')
#plt.show()


def normalizeTangent(a):
    for i in range(len(a)):
        maxx = np.max(a[i])
        minn = np.min(a[i])
        for j in range(len(a[0])):
            normalized = (a[i][j] - maxx)/(maxx-minn)
            a[i][j] = normalized
    return a 



#a = tangent(pos_x, pos_y)

#plot_segments(a)
#segment_series = total_segment_series(AAA)
#segment_series = normalizeTangent(segment_series)



#cross_avg,cross_peak_avg = correlate_slice_calc(segment_series, frame_rate, corr_seg=corr_seg)
#plots_to_plot(cross_avg, y = 'Peak cross correlation time lag')
#plots_to_plot(cross_peak_avg, y = 'Peak normalizedcross correlation')
#mut_info_avg = mut_info_total(segment_series, corr_seg = corr_seg)
#plots_to_plot(mut_info_avg, y = 'Normalized mutual info')

    
 
def time_4_segs(a):
    time = []
    for i in range(len(a)):
        time.append(i/sampling_rate)
    return time
def plot_segz(segment_series):
    x_axis = time_4_segs(segment_series[2])
    #plt.plot(time,a, markersize = 10, linewidth  = 1)

    ax1 = plt.axes(frameon=False)
    
    plt.plot(x_axis,segment_series[2], markersize = 10, linewidth  = 3)
    plt.plot(x_axis,segment_series[20], markersize = 10, linewidth  = 3)
    plt.plot(x_axis,segment_series[40], markersize = 10, linewidth  = 3)

    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 

    plt.xlabel('Time (s)')
    plt.ylabel('Tangent (radians)')    
    
    plt.savefig('tangents.png')
    plt.show()  
    return
def plot_individual_cross_correlations(segment_series):
    cc_time = time_create(segment_series[0], sampling_rate)
    for i in range(int(len(segment_series)/interval)):
        print('Segment number: ' + str(i+1))
        index = int(i*interval)
        cc_print = signal.correlate(segment_series[0], segment_series[index])
        
        
        
        #plt.plot(cc_time,cc_print, '-', markersize = 10, linewidth  = 1)
        ax1 = plt.axes(frameon=False)
        plt.plot(cc_time,cc_print, markersize = 10, linewidth  = 3)
        xmin, xmax = ax1.get_xaxis().get_view_interval()
        ymin, ymax = ax1.get_yaxis().get_view_interval()
        ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
        ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 

        plt.xlabel('Time (s)')
        plt.ylabel('Cross Correlation')
        plt.savefig('cross correlations.png')

        plt.show()

    return

        

def cc_quality_control():
    ### this function finds the 
    results_x, results_y = import_h5py(cc_ctrl_name, START,DURATION)  ##input the video to look at
    pos_x = matrix_float_multi(results_x)  ##
    pos_y = matrix_float_multi(results_y)  ##
    tangent_series = tangent(pos_x, pos_y)  ##   
    segment_series = total_segment_series(tangent_series)  ##
    
    plot_individual_cross_correlations(segment_series)
    ## plots all 13 cross correlations
    plot_segz(segment_series) ##plot the head, midbody, tail through time
    
             
    ## find worm velcoity
   # velocity = worm_vel_averaged(segment_series, ratE = rate)
         
    print('Start frame: ' + str(START))
    print('Stop frame : ' + str(STOP))
    print('File name: ' + str(cc_ctrl_name))
    return 0
#results_x, results_y = import_h5py(cc_ctrl_name, START,DURATION)

#vel = cc_quality_control()



#for i in range(13):
#    savename = 'n2_SWIMMIN' + str(i)
#    corr_seg = i
#    name_list, start_list, stop_list, frame_rate_list = extract_instructions(instructions)    
#    averaged = combine_worms(name_list, start_list, stop_list, frame_rate_list, corr_seg = i)


name_list, start_list, stop_list, frame_rate_list = extract_instructions(instructions)
averaged = combine_worms(name_list, start_list, stop_list, frame_rate_list, corr_seg = corr_seg)



#repeat_model(cycles = 1)

































#mid_seg = 24
#cc_print = signal.correlate(segment_series[0], segment_series[28])

#ax1 = plt.axes(frameon=False)

#frame1 = plt.plot(cc_time, cc_print)
#plt.plot(frame1)
#BBB = calc_all_tangents(pos_x, pos_y)
#blah = total_segment_series(BBB)

















#plt.plot(blah[0])
#plt.show(s)
#plt.plot(segment_series[0])


    
#xmin, xmax = ax1.get_xaxis().get_view_interval()
#ymin, ymax = ax1.get_yaxis().get_view_interval()
#ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
#frame1.axes.get_yaxis().set_visible(False)
#cur_axes = mplot.gca()

#ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
#plt.xlabel('Time (s)')
#plt.ylabel('Cross correlation')  
#plt.savefig('cc.png') 
#plt.show()

#plot_worm(pos_x, pos_y, position = 200, save = True)    
    
    
    
#cross_peak_avg = normalize_peak_cc(cross_peak_avg)    
#cross = cross_correlate(segment_series)   
    
    
    
## get the best frequency from def RUN_ONCE, export it downstream for further analysis    
    