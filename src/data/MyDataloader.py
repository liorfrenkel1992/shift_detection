import glob
import warnings

import librosa

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.data_utils import AudioPreProcessing

warnings.filterwarnings("ignore", category=UserWarning)


class BasicDataset(Dataset):
    """Generated Input and Output to the model"""

    def __init__(self, cfg, data_path, speech_type='librispeech', noise_type='wham', mode='train'):
        """ 
        Args:
            cfg (yaml): Config file
            data_path (string): Data directory
        """
        self.cfg = cfg
        self.mode = mode
        self.num_of_chosen_sec = 5
        self.audio_pre_process = AudioPreProcessing(cfg)

        self.cut_signal = True

        data_path = f'{data_path}/{speech_type}/{noise_type}/{mode}'

        self.waves_list = glob.glob(f'{data_path}/*.wav')

    def __len__(self):
        return len(self.waves_list)

    def get_inputs_outputs(self, i):
        mixed, fs = librosa.load(self.waves_list[i], sr=16000, mono=False)

        input_map, target, noisy_stft, _, _, mixed = self.audio_pre_process.preprocess(mixed, cut_length=self.num_of_chosen_sec, normalize_noisy=True, cut_signal=self.cut_signal, mode=self.mode)

        target = target.squeeze(0).float()
        input_map = input_map.squeeze(0).float()
        
        input_map = input_map[:2, ...]
        target = target[:2, ...]
        input_map[:, 0:3, :] = input_map[:, 0:3, :] * 0.001
        
        mx_mix = torch.max(torch.max(torch.abs(input_map[0, ...]), torch.max(torch.abs(input_map[1, ...]))))
        input_map /= mx_mix

        return input_map, target, mixed, noisy_stft
                
    def pre_process(self, i):
        
        wave_file = self.waves_list[i].split('/')[-1]
        
        input_map, target, mixed, noisy_stft = self.get_inputs_outputs(i)
            
        mixed = mixed[:3, ...]

        pre_processed_data = [input_map, target, mixed, noisy_stft, wave_file]
       
        return tuple(pre_processed_data)
        
    def __getitem__(self, i):

        # ======== load audio/img file ====================
        
        pre_processed_data = self.pre_process(i)

        return pre_processed_data
          

class EnhancementDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.data_type = cfg.speech_type
        self.noise_type = cfg.noise_type
        self.number_of_gpus = 1
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        
    def setup(self, stage=None):
        self.train_set = BasicDataset(self.cfg, data_path=self.data_path, data_type=self.data_type, sub_data_type=self.noise_type, mode='train')
        self.val_set = BasicDataset(self.cfg, data_path=self.data_path, data_type=self.data_type, sub_data_type=self.noise_type, mode='val')
        
    def train_dataloader(self):
        self.train_sampler = DistributedSampler(self.train_set, shuffle=True) if self.number_of_gpus > 1 else None
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, 
                          sampler=self.train_sampler, shuffle=False if (self.number_of_gpus > 1) else True)

    def val_dataloader(self):
        self.val_sampler = DistributedSampler(self.val_set, shuffle=False) if self.number_of_gpus > 1 else None
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers =self.num_workers, 
                          sampler=self.val_sampler, shuffle=False)
    

class TestDataset(Dataset):
    """Generated Input and Output to the model"""

    def __init__(self, cfg, speech_type='librispeech', noise_type='wham'):
        
        self.cfg = cfg
        self.num_of_chosen_sec = 5

        self.audio_pre_process = AudioPreProcessing(cfg)
        
        self.cut_signal = True # if cfg.data_type != 'nisqa' else False

        data_path = f'{cfg.data_path}/{speech_type}/{noise_type}/test'

        self.waves_list = glob.glob(f'{data_path}/*.wav')

    def __len__(self):
        return len(self.waves_list)
    
    def get_inputs_outputs(self, i):
        mixed, fs = librosa.load(self.waves_list[i], sr=self.cfg.fs, mono=False)

        input_map, target, noisy_stft, _, _, mixed = self.audio_pre_process.preprocess(mixed, cut_length=self.num_of_chosen_sec, normalize_noisy=True, cut_signal=self.cut_signal, mode='test')

        target = target.squeeze(0).float()
        input_map = input_map.squeeze(0).float()
        
        input_map = input_map[:2, ...]
        target = target[:2, ...]
        input_map[:, 0:3, :] = input_map[:, 0:3, :] * 0.001
        
        mx_mix = torch.max(torch.max(torch.abs(input_map[0, ...]), torch.max(torch.abs(input_map[1, ...]))))
        input_map /= mx_mix

        return input_map, target, mixed, noisy_stft
                
    def pre_process(self, i):
        wave_file = self.waves_list[i].split('/')[-1]

        snr = int(wave_file.split("_")[-1].split(".")[0])
        
        input_map, target, mixed, noisy_stft = self.get_inputs_outputs(i)
            
        mixed = mixed[:3, ...]

        pre_processed_data = [input_map, target, mixed, noisy_stft, snr, wave_file]
       
        return tuple(pre_processed_data)

    def __getitem__(self, idx):
        pre_processed_data = self.pre_process(idx)

        input_map, target, raw, noisy_stft_complex, snr, name = pre_processed_data

        raw = torch.from_numpy(raw).unsqueeze(0)

        return input_map, raw, noisy_stft_complex, snr

