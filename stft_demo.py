import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils import *

for i in (1, 3, 5, 7, 9):
    train_data, test_data = get_data(
        trace_path='./traces/trace.log',
        source_id=i,
        obs_ord=2,
        scale=1e8,
    )
    sig = test_data[:2048, 0]
    fft = librosa.stft(sig)
    mag, pha = np.abs(fft), np.angle(fft)
    print(mag.shape)
    plt.imshow(mag)
    plt.savefig(f'./temp_node{i}.jpg')
    plt.close()