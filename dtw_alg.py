import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display
import pandas as pd


if __name__=='__main__':
    y_not, sr_not = librosa.load('Audio/19.m4a')
    y_twi, sr_twi = librosa.load('Audio/twi.m4a')

    idx = tuple([slice(None), slice(*list(librosa.time_to_frames([45, 60])))])

    y_not = librosa.effects.harmonic(y=y_not, margin=8)

    chroma_not = librosa.feature.chroma_stft(y=y_not, sr=sr_not)
    chroma_twi = librosa.feature.chroma_stft(y=y_twi, sr=sr_twi)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(chroma_not, y_axis='chroma')
    plt.ylabel('Nottusvara')
    plt.colorbar()
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    librosa.display.specshow(chroma_twi, y_axis='chroma')
    plt.ylabel('Twinkle')
    plt.colorbar()
    plt.tight_layout()
    plt.show()