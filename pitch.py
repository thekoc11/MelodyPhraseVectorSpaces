from __future__ import print_function
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow
import numpy as np
from essentia.standard import *
import  essentia as es
from statsmodels.graphics.tsaplots import  plot_acf
import networkx as net
from math import *


def autocorrelate(data):
    data = np.array(data)
    avg = data.mean(axis=0)
    new_data = (data - np.mean(avg))/np.std(avg)
    for count in range(1, 30):
        i = np.arange(len(new_data) - count)
        yield np.multiply(new_data[i], new_data[i+count]).sum(axis=0).mean()




if __name__== '__main__':
    loader = EqloudLoader(filename='Audio/19.m4a')
    audio = loader()
    print("Duration of audio: ", len(audio)/44100)
    

    tonic_identifier = TonicIndianArtMusic()
    tonic = tonic_identifier(audio)
    pitch_extractor = PredominantPitchMelodia(frameSize=2048, hopSize=196, referenceFrequency=tonic)
    pitch_values, pitch_confidence = pitch_extractor(audio)

    pitch_times = np.linspace(0.0, len(audio) / 44100.0, len(pitch_values))
    print("Tonic:", tonic)

    # f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].plot(pitch_times, pitch_values)
    # axarr[0].set_title('Estimated Pitch (Hz)')
    # axarr[1].plot(pitch_times, pitch_confidence)
    # axarr[1].set_title('Pitch Confidences')
    # plt.show()

    resampler = Resample(outputSampleRate=450)

    pitch_values_resampled = resampler(pitch_values)
    pitch_times_resampled = np.linspace(0.0, len(audio)/44100.0, len(pitch_values_resampled))

    # f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].plot(pitch_times, pitch_values)
    # axarr[0].set_title('Estimated Pitch (Hz)')
    # axarr[1].plot(pitch_times_resampled, pitch_values_resampled)
    # axarr[1].set_title('Resampled')
    # plt.show()

    print(pitch_times_resampled)

    # plot_acf(pitch_segments[87], lags=30)
    # plt.show()





