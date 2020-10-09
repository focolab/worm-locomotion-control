#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:36:36 2018

@author: jmw
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from math import atan2
from scipy import signal
from sklearn import metrics
import h5py
import scipy.io as sio
from matplotlib.lines import Line2D
from scipy.fftpack import fft
from scipy.fftpack import fftfreq

###################################################################################
corr_div =1 # How many divisions you want the time series divided into for analysis
corr_seg = 0 # This is the segment # the MI and cross correlation compare to 
bins = 2 # MI bin number, recommended 2
savename_gross = 'N2_SWIM' # This is tacked on to the saved filenames
interval = 3.69 #segment interval, to skip some segments; for example: 50/3.69 = 13 segments
instructions = 'n2_swim_cleanup_BEtter.csv' ##FILE with the macro-level time points
backwards = False  # False means forward locomotion; True means backwards locomotion
###################################################################################
#These are the variables for cross correlation quality control
cc_ctrl_name = 'unc37_10.hdf5' 
START = 7748.4
STOP = 8800
DURATION = int(STOP-START)
sampling_rate = 30 # Sampling rate of the video file, this variable is only relevant to the cross correlation qualtiy control
print('Start frame: ' + str(START))
print('Duration : ' + str(DURATION))
############################################################
## These variables are for the sine generation in the head-first model
freq = 5
sine_length = 8000
sine_freq = 5
noisyness =0.1
delay_per_seg = 10
num_segments = 49
frame_rate = 300
###########################################################
print('||||||global variables||||||')
print('mutual info bins: ' + str(bins))
print('video segments averaged: ' + str(corr_div))
print('sampling rate: ' + str(frame_rate))
print('segment to compare to: ' + str(corr_seg))
###########################################################

def absolute(a):
    # Returns a matrix of absolute numbers
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
    max_freq_index = np.argmax(y_axis)
    max_freq = x_axis[max_freq_index]
    return max_freq

def worm_vel_averaged(a, ratE):
    ## Calulates average the worm velocities
    vel_matrix = [0 for i in range(int(len(a)/interval))]    
    for i in range(int(len(a)/interval)):
        index = int(i*interval)
        vel_matrix[i] = calc_worm_vel(a[index], ratE)
    avg_vel = np.mean(vel_matrix)
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
    # Takes a nested matrix of string numbers and turns them into floats
    for i in range(len(a)):
        for j in range(len(a[0])):         
            a[i][j] = float(a[i][j])
    return a

def matrix_float(a):
    # Takes a matrix and returns floats for all values
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
 
def plot_worm(x, y, position = 0, save = False):
    # Plot the x,y coordiantes of the worm
    # position refers to the frame desired to plot
    curr_x = []
    curr_y = []
    for i in range(len(x)):
        curr_x.append(x[i][position])
    for i in range(len(y)):
        curr_y.append(y[i][position])
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

def destroyNAN(a):
    # Delete the nans in a matrix
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
    # Calculate the tangent of an x,y data series
    for i in range(len(x)):
        x[i] = destroyNAN(x[i])
        y[i] = destroyNAN(y[i])
    angles = [[0 for i in range(len(x)-1)] for j in range(len(x[0]))]
    for j in range(len(x[0])):
        for i in range(len(x)-1):
            rise = y[i+1][j] -y[i][j]
            run = x[i +1][j] - x[i][j]
            angles[j][i] = atan2(rise, run) 
    for i in range(len(angles)):
        curr_mean = np.mean(angles[i])
        for j in range(len(angles[0])):
            angles[i][j] = angles[i][j] -curr_mean
    return angles  

def single_segment_series(a, segment = 0):
    # Reformat the data so that a signle segment matrix represents segmented data through time
    time_series = []
    for i in range(len(a)):
        time_series.append(a[i][segment])
    return time_series

def total_segment_series(a):   
    # Reformat the data so that the whole segment matrix represents a segmented data through time
    time_series = []
    for i in range(len(a[0])):
        time_series.append(0)
    for i in range(len(a[0])):
        time_series[i] = single_segment_series(a, segment = i)
    return time_series

def time_create(seg_series, frame_rate):
    # Create time matrix based on the frame rate and the length of the data points
    time = []
    for i in range(len(seg_series)*2-1):
        current_time = i /frame_rate -(len(seg_series)/frame_rate)
        time.append(current_time) 
    return time

def cross_correlate(time_series, frame_rate, interval=interval, corr_seg = corr_seg):
    # Cross correlates. variables should be self-explanatory. interval and corr_seg are defined at the beginning of the program 
    # Time _series is the tagents of each segment through time; frame rate is determined from the instructions file
    cross_matrix = []
    PEAK_cross = []
    time = time_create(time_series[0], frame_rate)
    plt.show()
    compare_to = int(interval*corr_seg)
    if backwards == True:
        compare_to = int(len(time_series)/interval) - int(corr_seg*interval)                 
    for i in range(int(len(time_series)/interval)):
        comparing = int(interval*i)
        if backwards == True:
            comparing = int((len(time_series)/interval)-comparing)
        s = signal.correlate(time_series[compare_to],time_series[comparing])
        index = np.argmax(s)  
        cross_matrix.append(time[index])
        PEAK = np.correlate(time_series[compare_to], time_series[comparing])
        PEAK = PEAK[0]
        PEAK_cross.append(PEAK)
    return cross_matrix, PEAK_cross

def slice_time_segment(a, round = 0):
    # If corr_div is not equal to 1, this will slice up the time into the amount of divisions
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
    # Cross correlates the corr_seg to all other segments
    corr_coef = [] # The time delay to peak cross correlation 
    PEAK_cross = [] # Peak cross correlation
    for i in range(corr_div):
        corr_coef.append(0)
        PEAK_cross.append(0)
    for i in range(corr_div):
        curr_time_series = slice_time_segment(a, round = i)
        curr_corr_coef, curr_PEAK_cross = cross_correlate(curr_time_series, frame_rate, corr_seg = corr_seg)
        corr_coef[i] = curr_corr_coef
        curr_PEAK_cross = normalize_peak_cc(curr_PEAK_cross) # Normalize the peak cross correlation so that the first value is 1      
        PEAK_cross[i] = curr_PEAK_cross
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
    # Calcuates the mutual info of a single segment to the 
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi

def mutual_info(a, bins = bins, interval = interval, corr_seg = corr_seg):
    # Calculates the mutual info of all segments 
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

    plt.xlabel('Segment number')
    plt.ylabel(y)  
    if save == True:
        plt.savefig(savename + y + ".pdf")
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
        plt.savefig(savename + y + ".pdf")
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

def plot_all_segs(a):
    # Plot all of the worm segments through time
    for i in range(len(a)):
        print('segment: '+str(i))
        plt.plot(a[i])
        #plt.ylim(-.1,.1)
        plt.show()

def matrix_avg(matrix):
    # Compress the individual transitions for easier plotting
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
    # Import instructions and turn them into seperate matrixes  
    with open(name, newline='\n') as inputfile:
        instructions = list(csv.reader(inputfile))
    name_list = []
    frame_rate_list = []
    for i in range(len(instructions)-1):
            name_list.append(instructions[i+1][0])
            frame_rate_list.append(instructions[i+1][3])
    # Creates a list of start frames, convert it to floats
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
    #Flip the signs on everything by mulitplyig by negative one
    for i in range(len(a)):
        a[i] = a[i]*-1
    return a

def normalize_cross_time_lag(a, vel):
    # Normalize time deltay to peak cross correlation  
    for i in range(len(a)):
        a[i] = a[i]*vel
    return a

def run_once(name, start_frame, stop_frame, frame_rate, corr_seg = corr_seg):
    # Run everything on a single worm 
    print('File name: ' + name)
    print('Start frame: ' + str(start_frame))
    print('Stop frame: ' + str(stop_frame))
    print('Segment to compare to: '+ str(corr_seg))
    total_frames = stop_frame - start_frame
    if name[-1] == '5':
        results_x, results_y = import_h5py(name, start_frame, total_frames)
    else: 
        results_x, results_y = matlab_worm(name, start_frame, total_frames)
    pos_x = matrix_float_multi(results_x)
    pos_y = matrix_float_multi(results_y)
    tangent_series = tangent(pos_x,pos_y)  
    segment_series = total_segment_series(tangent_series)
    cross_avg,cross_peak_avg = correlate_slice_calc(segment_series, frame_rate, corr_seg = corr_seg)
    worm_vel = worm_vel_averaged(segment_series, ratE = frame_rate)
    cross_avg = correct_cc_lag_sign(cross_avg)
    cross_avg = normalize_cross_time_lag(cross_avg, worm_vel)
    plots_to_plot(cross_avg, y = 'peak cross correlation time lag'+name, save = False)
    plots_to_plot(cross_peak_avg, y = 'peak cross correlation')
    mut_info_avg = mut_info_total(segment_series, corr_seg = corr_seg)
    plots_to_plot(mut_info_avg, y = 'normalized mutual info'+name, save = False)
    return cross_avg, mut_info_avg, cross_peak_avg, worm_vel

def combine_worms(name_list, start_list,stop_list, frame_rate_list, corr_seg=corr_seg):
    # Create cross correlation average matrix and mutual info 
    cross_total = [0 for i in range(len(name_list))] # Time lage for peak cross correlation
    cross_peak_total = [0 for i in range(len(name_list))] #Peak cross correlation
    mi_total =[0 for i in range(len(name_list))]
    vel_total =[0 for i in range(len(name_list))]
    # Check to make sure that the frames are the same
    total_frames = []
    for i in range(len(start_list)):
        total_frames.append(stop_list[i]-start_list[i])
    minimum_frames = np.min(total_frames)
    for i in range(len(total_frames)):
        if total_frames[i] > minimum_frames:
            stop_list[i] = start_list[i] +minimum_frames
    # Calculate the cross correlation and mutual info for all items in the list
    for i in range(len(name_list)):
        print('   ')
        print('list location: ' +str(i+1))
        local_cc, local_mi, local_cross_peak, local_vel = run_once(name_list[i], start_list[i], stop_list[i], frame_rate_list[i], corr_seg = corr_seg)            
        cross_total[i] = local_cc
        mi_total[i] = local_mi 
        cross_peak_total[i] = local_cross_peak
        vel_total[i] = local_vel          
    cross_avg = matrix_avg(cross_total) # Find the average matrix
    mi_avg = matrix_avg(mi_total) #
    cross_peak_avg = matrix_avg(cross_peak_total)
    print()
    print()
    print('totals: ')
    plots_to_plot(cross_avg, y = 'Total peak cross correlation time lag', save = True)
    plots_to_plot(cross_peak_avg, y = 'Total normalized peak cross correlation', save = True)
    plots_to_plot(mi_avg, y = 'Total normalized mutual info', save = True)
    np.savetxt(savename+"_MI_total.csv", mi_total, delimiter=",")
    np.savetxt(savename+"_CC_timelag_total.csv", cross_total, delimiter=",")
    np.savetxt(savename+"_CC_Peak_total.csv", cross_peak_total, delimiter=",")
    return {'cross_total':cross_total, 'mi_total':mi_total, 
            'cross_avg':cross_avg, 'mi_avg':mi_avg, 'cross_peak_total':cross_peak_total,
            'cross_peak_avg':cross_peak_avg, 'vel_total':vel_total}

def create_sine(sampling_rate, frequency, amplitude, show_plot = True):
    # Creates a sine wave for the by parameters used at the beginning of the program
    Fs = sampling_rate # Sampling rate
    f = frequency # Signal frequency
    x = np.arange(1,Fs +1)
    amp = amplitude # Amplitude of sine wave
    y = amp*np.sin(2 * np.pi * f * x / Fs)
    if show_plot == True:
        plt.plot(x, y, c = 'black')
        plt.axis('off')
        plt.savefig('sine.pdf')
        plt.show()
    return y

def add_noise(a, g_noise):
    # Adds noise
    for i in range(len(a)):
        a[i] = a[i] +g_noise[i]
    return a

def delay_single_seg(a, delay_per_seg, curr_seg):
    # Delay a segment by delay_per_seg
    b = []
    new_matrix = []
    for i in range(len(a)):
        b.append(a[i])
    for i in range(len(a)-delay_per_seg):
        new_matrix.append(a[i])
    for i in range(len(b)-delay_per_seg):
        new_matrix[i] = b[i + delay_per_seg]
    return new_matrix

def adjust_len(a):
    max_len = len(a[-1])
    for i in range(len(a)):
       a[i] = a[i][0:max_len]
    return a

def create_delay_segments(a, num_segments, delay_per_seg):
    # Creates segments with the same freq     
    segments = [0 for i in range(num_segments)]
    segments[0] = a
    curr_noise = np.random.normal(0,noisyness, sine_length)
    segments[0] = add_noise(segments[0], curr_noise)
    # Time delays each segment
    # Creates and add noise for individual segments
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
    delayed_segments = create_delay_segments(sine, num_segments, delay_per_seg)
    cross_avg, cross_peak_avg = correlate_slice_calc(delayed_segments, sampling_rate)
    plots_to_plottt(cross_avg, y = 'time delay to peak cross correlation', save = False)
    plots_to_plottt(cross_peak_avg, y = 'peak cross correlation', save = False)
    mut_info_avg = mut_info_total(delayed_segments)
    plots_to_plottt(mut_info_avg, y = 'normalized mutual info', save = False)
    np.savetxt(savename+"mut_info_model.csv", mut_info_avg, delimiter=",")   
    np.savetxt(savename+"cc_delay_model.csv", cross_avg, delimiter=",")   
    np.savetxt(savename+"cc_peak_model.csv", cross_peak_avg, delimiter=",")   

def repeat_model(cycles = 10):
    ##create empty variables 
    MI_total = [0 for i in range(cycles)]
    cross_timelag_total = [0 for i in range(cycles)]
    cross_peak_total = [0 for i in range(cycles)] 
    # Run the model for however many cycles 
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

def time_4_segs(a):
    # Creates the x axis to 
    time = []
    for i in range(len(a)):
        time.append(i/sampling_rate)
    return time

def plot_segz(segment_series):
    x_axis = time_4_segs(segment_series[2])
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
    # This function finds the first, middle, end segments and plots the cross correlation to gut check the code is working
    results_x, results_y = import_h5py(cc_ctrl_name, START,DURATION)  # Open video file, extract worm x & y coordinates
    pos_x = matrix_float_multi(results_x)  # Turn the x coordinate into a float
    pos_y = matrix_float_multi(results_y)  # Turn the y coordinate into a float
    tangent_series = tangent(pos_x, pos_y)  # Find the    
    segment_series = total_segment_series(tangent_series)  #
    plot_individual_cross_correlations(segment_series)
    plot_segz(segment_series) ##plot the head, midbody, tail through time        
    print('Start frame: ' + str(START))
    print('Stop frame : ' + str(STOP))
    print('File name: ' + str(cc_ctrl_name))
    return 0




for i in range(13):
    savename = savename_gross + str(i)
    corr_seg = i
    name_list, start_list, stop_list, frame_rate_list = extract_instructions(instructions)    
    averaged = combine_worms(name_list, start_list, stop_list, frame_rate_list, corr_seg = i)

name_list, start_list, stop_list, frame_rate_list = extract_instructions(instructions)
averaged = combine_worms(name_list, start_list, stop_list, frame_rate_list, corr_seg = corr_seg)


    