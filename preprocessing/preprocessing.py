# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:07:43 2020

@author: dilpreet
"""
import os
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from python_speech_features import logfbank, mfcc
from tqdm import tqdm

instrument_list = [
        'Kamancheh',
        'Ney',
        'Santur',
        'Setar',
        'Tar',
        'Ud'
    ]


def pie_chart(class_dist):
    fig, ax = plt.subplots()
    ax.set_title("Class Distribution",y=1.08)
    ax.pie(class_dist,labels=class_dist.index, autopct="%1.1f%%",shadow=False, startangle=90)
    ax.axis("equal")
    plt.show()


def plot_signals(signals):
    fig,axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=False, figsize=(20,10))
    fig.suptitle("Time Series", size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.show()

def plot_fft(fft):
    fig,axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=False, figsize=(20,10))
    fig.suptitle("Fourier Transforms", size=16)
    i=0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq,Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.show()

def plot_fbank(fbank):
    fig,axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=False, figsize=(20,10))
    fig.suptitle("Filter Bank Coefficients", size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],cmap="hot", interpolation="nearest")
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.show()

def plot_mfccs(mfccs):
    fig,axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=False, figsize=(20,10))
    fig.suptitle("Mel Frequency cepstrum coefficients", size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],cmap="hot", interpolation="nearest")
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    plt.show()

def calc_fft(signal,rate):
    freq = np.fft.rfftfreq(len(signal),d=(1/rate))
    mag = abs(np.fft.rfft(signal)/len(signal))
    return (mag,freq)


    
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

# def down_sample(dataset,threshold):
#     for f in tqdm(dataset["file_names"]):
#         signal,rate = librosa.load(f, sr=16000)
#         mask,y_mean = envelope(signal, rate, threshold)
#         wavfile.write(filename=f, rate=rate, data=signal[mask])
            


def get_data():
    data = []
    for instrument_type in instrument_list:
            filenames = [i for i in os.walk(f"wavfiles/{instrument_type}")][0][2]
            for names in tqdm(filenames):
                y, sr = librosa.load(f"wavfiles/{instrument_type}/{names}", sr=16000)
                data.append([y,sr,instrument_type])
                
    signal_length = []
    for i in data:
        signal_length.append(len(i[0]))
    
    dataset = pd.DataFrame(data,columns=["signal","rate","name"])
    dataset["signal_length"] = signal_length
    class_dist = dataset.groupby(["name"])["signal_length"].mean()
    
    return dataset, class_dist


def process_data(dataset):
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    
    for c in instrument_list:
        signal = dataset[dataset["name"] == c]["signal"].iloc[0]
        rate = dataset[dataset["name"] == c]["rate"].iloc[0]
                
        signals[c] = signal
        
        fft[c] = calc_fft(signal,rate)
        
        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        fbank[c] = bank
        
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel
        
    return class_dist,signals,fft,fbank,mfccs
    


def get_charts(class_dist,signals,fft,fbank,mfccs):
    pie_chart(class_dist)
    
    plot_signals(signals)
    
    plot_fft(fft)
    
    plot_fbank(fbank)
    
    plot_mfccs(mfccs)



if __name__ == "__main__":
        
    dataset,class_dist = get_data()
    
    class_dist,signals,fft,fbank,mfccs = process_data(dataset)
    
    get_charts(class_dist,signals,fft,fbank,mfccs)
    
    # threshold = 0.0005
    
    # down_sample(dataset,threshold)
