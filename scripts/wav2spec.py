import librosa
import librosa.display as display
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--sr', type=int, default=None)
    args = parser.parse_args()

    n_fft = args.n_fft
    hop_length = args.hop_length

    y, sr = librosa.load(args.wav, sr=args.sr)
    spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)))
    display.specshow(spec, y_axis='linear', x_axis='time', cmap='magma',
                     sr=sr, hop_length=hop_length, n_fft=n_fft)
    plt.savefig(args.out)
