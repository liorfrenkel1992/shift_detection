import warnings
import numpy as np

import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)
torchaudio.set_audio_backend("sox_io")


def normalize(x):
    if np.max(np.abs(x)) != 0:
        x_norm = x / 1.1 / np.max(np.abs(x))
    else:
        x_norm = x
    
    return x_norm

class AudioPreProcessing():
    def __init__(self, cfg):
        self.cfg = cfg

    def preprocess(self, mixed, cut_length=2, normalize_noisy=False, cut_signal=True, mode='train'):
        cfg = self.cfg

        noisy = mixed[0, :]
        noise = mixed[1, :]
        clean = mixed[2, :]

        window_size = cfg.window_size

        duration_sec = cfg.fs * cut_length

        if cut_signal:
            if mixed.shape[-1] > duration_sec:
                if mode == 'val' or mode == 'test':
                    start_point = 0
                else:
                    start_point = torch.randint(0, mixed.shape[-1] - duration_sec, (1,)).item()
                mixed = mixed[..., start_point:start_point + duration_sec]
            elif mixed.shape[-1] < duration_sec:
                mixed = np.concatenate((mixed, np.zeros((mixed.shape[-2], duration_sec - mixed.shape[-1]))), axis=1)

        noisy = mixed[0, :]
        noise = mixed[1, :]
        clean = mixed[2, :]
        
        if normalize_noisy:
            noisy = normalize(noisy)
            mixed[0, :] = noisy

        S_stft_complex = torch.stft(torch.from_numpy(clean), window_size, int(window_size * cfg.overlap), window=torch.hamming_window(window_size), return_complex=True)
        V_stft_complex = torch.stft(torch.from_numpy(noise), window_size, int(window_size * cfg.overlap), window=torch.hamming_window(window_size), return_complex=True)
        Z_stft_complex = torch.stft(torch.from_numpy(noisy), window_size, int(window_size * cfg.overlap), window=torch.hamming_window(window_size), return_complex=True)
        
        Z_stft_complex = torch.tensor(Z_stft_complex, dtype=torch.complex128)
        S_stft_complex = torch.tensor(S_stft_complex, dtype=torch.complex128)            
        
        real_target = S_stft_complex.real
        imag_target = S_stft_complex.imag
        real_noise_target = V_stft_complex.real
        imag_noise_target = V_stft_complex.imag
        V_out = torch.stack((real_noise_target, imag_noise_target), dim=0)
        target = torch.stack((real_target, imag_target), dim=0)
        
        real_input_map = Z_stft_complex.real
        imag_input_map = Z_stft_complex.imag
        
        input_map = torch.stack((real_input_map, imag_input_map), dim=0)
            
        return input_map, target, Z_stft_complex, S_stft_complex, V_out, mixed
    