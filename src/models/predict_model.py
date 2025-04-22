import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import hydra
from matplotlib.ticker import FormatStrFormatter

cwd_path = os.getcwd().split("src")[0]
cwd_msg = f"\nWorking Directory Change: {os.getcwd()}   ->  {cwd_path} \n"
print(cwd_msg)
os.chdir(cwd_path)
sys.path.append(cwd_path)

from src.data.MyDataloader import TestDataset
from eval_utils import *
from model_def import Pl_UNet_enhance_reg

Hydra_path = '../conf/'
@hydra.main(config_path=Hydra_path, config_name='prediction_cfg')
def main(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    model = Pl_UNet_enhance_reg
    if cfg.supervised_target:
        model_target = Pl_UNet_enhance_reg

    model = load_model(cfg, model, device, speech_type=cfg.speech_type, noise_type=cfg.noise_type)
    if cfg.supervised_target:
        model_target = load_model(cfg, model_target, device, speech_type=cfg.speech_type_target, noise_type=cfg.noise_type_target)

    # Source domain test set using source model

    test_set = TestDataset(cfg, speech_type=cfg.speech_type_test, noise_type=cfg.noise_type_test)

    test_dataloader = DataLoader(test_set, batch_size = 50, shuffle = False, num_workers = cfg.num_workers, pin_memory = False)
    iterator = iter(test_dataloader)
    predicted_sisdr_var_dict = {-10: [], -5: [], 0: [], 5: [], 10: []}
    predicted_sisdr_dict = {-10: [], -5: [], 0: [], 5: [], 10: []}
    enhanced_dict = {-10: [], -5: [], 0: [], 5: [], 10: []}
    clean_dict = {-10: [], -5: [], 0: [], 5: [], 10: []}
    noisy_dict = {-10: [], -5: [], 0: [], 5: [], 10: []}
    noise_dict = {-10: [], -5: [], 0: [], 5: [], 10: []}
    real_sisdr_dict = {-10: [], -5: [], 0: [], 5: [], 10: []}

    model = model.to('cuda')
    model.eval()
    inx_iter = 0

    while True:
        try:
            print(f'Iter #{inx_iter}')
            batch = next(iterator)
            x = any(item is None for item in batch)

            input_map, raw, noisy_stft_complex, snr = batch

            input_map = input_map.to(model.device).squeeze(1)

            with torch.no_grad():
                enhanced_spec = model.model(input_map)

            enhanced_spec, enhanced_var_spec = enhance_postprocess_short_sisdr(enhanced_spec)
            enhanced, clean, noisy, noise = enhance_postprocess_sisdr(cfg, raw, input_map, enhanced_spec, enhanced_spec.device)
            enhanced, clean, noisy, noise = torch.from_numpy(enhanced), torch.from_numpy(clean), torch.from_numpy(noisy), torch.from_numpy(noise)

            predicted_sisdr_var = enhanced_var_spec[1]
            predicted_sisdr = enhanced_var_spec[0]
            
            for inx, (snr_single, predicted_sisdr_single, clean_single, enhanced_single, predicted_sisdr_var_single, noisy_single) in enumerate(zip(snr, predicted_sisdr, clean, enhanced, predicted_sisdr_var, noisy)):
                snr_single = int(snr_single)
                predicted_sisdr_dict[snr_single].append(predicted_sisdr_single.item())
                enhanced_dict[snr_single].append(enhanced_single)
                clean_dict[snr_single].append(clean_single)
                noisy_dict[snr_single].append(noisy_single)
                noise_dict[snr_single].append(noise[inx])
            
                sisdr_real_single = si_sdr_torchaudio_calc(enhanced_single.unsqueeze(0), clean_single.unsqueeze(0)).item()
                real_sisdr_dict[snr_single].append(sisdr_real_single)

                predicted_sisdr_var_dict[snr_single].append(predicted_sisdr_var_single.item())

            inx_iter += 1

        except StopIteration:
            print('End iterations')
            break
            # continue #break

    # Target domain test set using source model
    
    if cfg.speech_type_target_test != '':
        test_set_target = TestDataset(cfg, speech_type=cfg.speech_type_target_test, noise_type=cfg.noise_type_target_test)

        test_target_dataloader = torch.utils.data.DataLoader(test_set_target, batch_size = 50, shuffle = False, num_workers = cfg.num_workers, pin_memory = False)
        iterator = iter(test_target_dataloader)
        predicted_sisdr_var_dict2 = {-10: [], -5: [], 0: [], 5: [], 10: []}
        predicted_sisdr_dict2 = {-10: [], -5: [], 0: [], 5: [], 10: []}
        enhanced_dict2 = {-10: [], -5: [], 0: [], 5: [], 10: []}
        clean_dict2 = {-10: [], -5: [], 0: [], 5: [], 10: []}
        noise_dict2 = {-10: [], -5: [], 0: [], 5: [], 10: []}
        noisy_dict2 = {-10: [], -5: [], 0: [], 5: [], 10: []}
        real_sisdr_dict2 = {-10: [], -5: [], 0: [], 5: [], 10: []}
        real_sisdr_dict2_target = {-10: [], -5: [], 0: [], 5: [], 10: []}

        inx_iter = 0

        while True:
            try:
                print(f'Iter #{inx_iter}')

                batch = next(iterator)
                x = any(item is None for item in batch)

                input_map, raw, noisy_stft_complex, snr = batch

                input_map = input_map.to(model.device).squeeze(1)

                with torch.no_grad():
                    enhanced_spec = model.model(input_map)
                    if cfg.supervised_target:
                        enhanced_spec_target = model_target.model(input_map)

                enhanced_spec, enhanced_var_spec = enhance_postprocess_short_sisdr(enhanced_spec)
                enhanced, clean, noisy, noise = enhance_postprocess_sisdr(cfg, raw, input_map, enhanced_spec, enhanced_spec.device)
                enhanced, clean, noisy, noise = torch.from_numpy(enhanced), torch.from_numpy(clean), torch.from_numpy(noisy), torch.from_numpy(noise)

                if cfg.supervised_target:
                    enhanced_spec_target, _ = enhance_postprocess_short_sisdr(enhanced_spec_target)
                    enhanced_target, clean_target, noisy_target = enhance_postprocess_sisdr(cfg, raw, input_map, enhanced_spec_target, enhanced_spec.device)
                    enhanced_target, clean_target, noisy_target = torch.from_numpy(enhanced_target), torch.from_numpy(clean_target), torch.from_numpy(noisy_target)
                else:
                    enhanced_target, clean_target, noisy_target = enhanced, clean, noisy

                predicted_sisdr_var = enhanced_var_spec[1]
                predicted_sisdr = enhanced_var_spec[0]
            
                for inx, (snr_single, predicted_sisdr_single, clean_single, enhanced_single, predicted_sisdr_var_single, enhanced_target_single, clean_target_single, noisy_single) in enumerate(zip(snr, predicted_sisdr, clean, enhanced, predicted_sisdr_var, enhanced_target, clean_target, noisy)):
                    snr_single = int(snr_single)
                    predicted_sisdr_dict2[snr_single].append(predicted_sisdr_single.item())
                    enhanced_dict2[snr_single].append(enhanced_single)
                    clean_dict2[snr_single].append(clean_single)
                    noisy_dict2[snr_single].append(noisy_single)
                    noise_dict2[snr_single].append(noise[inx])
                    
                    sisdr_real_single = si_sdr_torchaudio_calc(enhanced_single.unsqueeze(0), clean_single.unsqueeze(0)).item()

                    real_sisdr_dict2[snr_single].append(sisdr_real_single)
                    if cfg.supervised_target:
                        sisdr_real_single_target = si_sdr_torchaudio_calc(enhanced_target_single.unsqueeze(0), clean_target_single.unsqueeze(0)).item()

                        real_sisdr_dict2_target[snr_single].append(sisdr_real_single_target)
                    
                    predicted_sisdr_var_dict2[snr_single].append(predicted_sisdr_var_single.item())
                
                inx_iter += 1

            except StopIteration:
                print('End iterations')
                break
                # continue #break

    predicted_sisdr_var_list = []
    predicted_sisdr_error_list = []

    if cfg.speech_type_target_test != '':
        predicted_sisdr_var_list2 = []
        predicted_sisdr_error_list2 = []

    snr_list = [-5, 0, 5, 10]
    
    max_k = cfg.k_value
    zero_counter_every_list = list(np.arange(1, max_k + 1))
    zero_counter_every_constant = cfg.k_value
    enhanced_dict_means = {zero_counter: {-5: [], 0: [], 5: [], 10: []} for zero_counter in zero_counter_every_list}
    enhanced_dict_mean = {zero_counter: [] for zero_counter in zero_counter_every_list}
    enhanced_dict_std = {zero_counter: [] for zero_counter in zero_counter_every_list}

    sisdr_dict_means = {-5: [], 0: [], 5: [], 10: []}
    sisdr_dict_mean = []
    sisdr_dict_std = []

    enhanced_dict_means2 = {zero_counter: {-5: [], 0: [], 5: [], 10: []} for zero_counter in zero_counter_every_list}
    enhanced_dict_mean2 = {zero_counter: [] for zero_counter in zero_counter_every_list}
    enhanced_dict_std2 = {zero_counter: [] for zero_counter in zero_counter_every_list}
    sisdr_dict_means2 = {-5: [], 0: [], 5: [], 10: []}
    sisdr_dict_mean2 = []
    sisdr_dict_std2 = []
    sisdr_dict_means2_target = {-5: [], 0: [], 5: [], 10: []}
    sisdr_dict_mean2_target = []
    sisdr_dict_std2_target = []
    
    for snr_single in snr_list:
        
        predicted_sisdr = torch.tensor(predicted_sisdr_dict[snr_single])
        
        for zero_counter_every in zero_counter_every_list:

            # Calculate the mean for each group of 20 elements
            for i in range(0, len(predicted_sisdr_var_dict[snr_single]), zero_counter_every):
                # On source test set
                group = predicted_sisdr_var_dict[snr_single][i:i+zero_counter_every]
                group_mean = sum(group) / len(group)
                enhanced_dict_means[zero_counter_every][snr_single].append(group_mean)
                if zero_counter_every == zero_counter_every_constant:
                    group_sisdr = real_sisdr_dict[snr_single][i:i+zero_counter_every_constant]
                    group_sisdr_mean = sum(group_sisdr) / len(group_sisdr)
                    sisdr_dict_means[snr_single].append(group_sisdr_mean)

            enhanced_dict_mean[zero_counter_every].append(torch.mean(torch.tensor(enhanced_dict_means[zero_counter_every][snr_single])).item())
            enhanced_dict_std[zero_counter_every].append(torch.std(torch.tensor(enhanced_dict_means[zero_counter_every][snr_single])).item())
            if zero_counter_every == zero_counter_every_constant:
                sisdr_dict_mean.append(torch.mean(torch.tensor(sisdr_dict_means[snr_single])).item())
                sisdr_dict_std.append(torch.std(torch.tensor(sisdr_dict_means[snr_single])).item())
        
        predicted_sisdr_var = torch.tensor(predicted_sisdr_var_dict[snr_single])

        predicted_sisdr_var_list.append(predicted_sisdr_var.mean().item())
        predicted_sisdr_error_list.append(predicted_sisdr_var.std().item())

        if cfg.speech_type_target_test != '':

            for zero_counter_every in zero_counter_every_list:
            
                # Calculate the mean for each group of 20 elements
                for i in range(0, len(predicted_sisdr_var_dict2[snr_single]), zero_counter_every):
                    group = predicted_sisdr_var_dict2[snr_single][i:i+zero_counter_every]
                    group_mean = sum(group) / len(group)
                    enhanced_dict_means2[zero_counter_every][snr_single].append(group_mean)
                    
                    if zero_counter_every == zero_counter_every_constant:
                        group_sisdr = real_sisdr_dict2[snr_single][i:i+zero_counter_every_constant]
                        group_sisdr_mean2 = sum(group_sisdr) / len(group_sisdr)
                        sisdr_dict_means2[snr_single].append(group_sisdr_mean2)

                        if cfg.supervised_target:
                            group_sisdr_target = real_sisdr_dict2_target[snr_single][i:i+zero_counter_every_constant]
                            group_sisdr_mean2_target = sum(group_sisdr_target) / len(group_sisdr_target)
                            sisdr_dict_means2_target[snr_single].append(group_sisdr_mean2_target)

                enhanced_dict_mean2[zero_counter_every].append(torch.mean(torch.tensor(enhanced_dict_means2[zero_counter_every][snr_single])).item())
                enhanced_dict_std2[zero_counter_every].append(torch.std(torch.tensor(enhanced_dict_means2[zero_counter_every][snr_single])).item())
                if zero_counter_every == zero_counter_every_constant:
                    sisdr_dict_mean2.append(torch.mean(torch.tensor(sisdr_dict_means2[snr_single])).item())
                    sisdr_dict_std2.append(torch.std(torch.tensor(sisdr_dict_means2[snr_single])).item())
                    if cfg.supervised_target:
                        sisdr_dict_mean2_target.append(torch.mean(torch.tensor(sisdr_dict_means2_target[snr_single])).item())
                        sisdr_dict_std2_target.append(torch.std(torch.tensor(sisdr_dict_means2_target[snr_single])).item())

            predicted_sisdr_var2 = torch.tensor(predicted_sisdr_var_dict2[snr_single])
            
            predicted_sisdr_var_list2.append(predicted_sisdr_var2.mean().item())
            predicted_sisdr_error_list2.append(predicted_sisdr_var2.std().item())

    # Plot figures
   
    if cfg.speech_type_test == 'librispeech' and cfg.noise_type_test == 'wham':
        suffix = '+WHAM!'
    else:
        suffix = '+MUSAN' if cfg.noise_type_test == 'musan' else ''

    if cfg.speech_type_target_test == 'librispeech' and cfg.noise_type_target_test == 'wham':
        suffix2 = '+WHAM!'
    else:
        suffix2 = '+MUSAN' if cfg.noise_type_target_test == 'musan' else ''

    if cfg.speech_type_test == 'librispeech':
        speech_type_test = 'LibriSpeech'
    else:
        speech_type_test = 'DNS'

    if cfg.speech_type_target_test == 'librispeech':
        speech_type_target_test = 'LibriSpeech'
    else:
        speech_type_target_test = 'DNS'
   
    # Plot SI-SDR comparison between enhancement source / target models on source / target data
    
    if cfg.supervised_target:
        plt.figure()
        bar_width = 0.2
        # Set positions of the bars on the x-axis
        r1 = np.arange(len(snr_list))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]
        plt.bar(r1, sisdr_dict_mean, width=bar_width, label=f'{speech_type_test}{suffix}_s')
        plt.bar(r2, sisdr_dict_mean2, width=bar_width, label=f'{speech_type_target_test}{suffix2}_s')
        plt.bar(r3, sisdr_dict_mean2_target, width=bar_width, label=f'{speech_type_target_test}{suffix2}_t')
        
        plt.xlabel('SNR', fontsize=12)
        plt.ylabel('SI-SDR', fontsize=12)
        plt.yticks(fontsize=14)
        plt.xticks([r + bar_width for r in range(len(snr_list))], snr_list, fontsize=14)
        plt.legend()
        plt.savefig(f'reports/detection_results/sisdr_snr_{cfg.speech_type}{suffix}_{cfg.speech_type_target_test}{suffix2}_{max_k}_k.pdf', dpi=40)
        plt.close()

    # Plot errorbars for predicted log-variance of source and target data
    
    plt.figure()
    plt.errorbar(snr_list, enhanced_dict_mean[zero_counter_every_constant], enhanced_dict_std[zero_counter_every_constant], capsize=5, linestyle='None', marker='o', label=f'{speech_type_test}{suffix}')
    plt.errorbar(snr_list, enhanced_dict_mean2[zero_counter_every_constant], enhanced_dict_std2[zero_counter_every_constant], capsize=5, linestyle='None', marker='o', label=f'{speech_type_target_test}{suffix2}')
    plt.xlabel('SNR', fontsize=12)
    plt.ylabel('Mean Log-Variance', fontsize=12)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.yticks(fontsize=14)
    plt.xticks(ticks=snr_list, labels=snr_list, fontsize=14)
    sub_suffix = f'_musan' if cfg.noise_type_test == 'musan' else '_wham'
    sub_suffix2 = f'_musan' if cfg.noise_type_target_test == 'musan' else '_wham'
    suffix_both = f'_{cfg.speech_type_target_test}{sub_suffix2}'
    plt.legend()
    plt.savefig(f'reports/detection_results/mean_log_var_snr_{cfg.speech_type_test}{sub_suffix}{suffix_both}_std_{max_k}_k.pdf', dpi=40)        
    plt.close()

    # Plot detection accuracy vs. k values
    
    stds = 3
    
    false_alarms = {zero_counter: 0.0 for zero_counter in zero_counter_every_list}
    for zero_counter_inx, source_samples in enhanced_dict_means.items():
        snr_count = 0
        samples_counter_all_snrs = 0
        for mean_log_var_snr in source_samples.values():
            for source_sample in mean_log_var_snr:
                source_target_threshold = enhanced_dict_mean[zero_counter_inx][snr_count] + stds * enhanced_dict_std[zero_counter_inx][snr_count]
                samples_counter_all_snrs += 1
                if source_sample > source_target_threshold:
                    false_alarms[zero_counter_inx] += 1
            snr_count +=1 
        false_alarms[zero_counter_inx] /= samples_counter_all_snrs

    miss_detections = {zero_counter: 0.0 for zero_counter in zero_counter_every_list}
    for zero_counter_inx, target_samples in enhanced_dict_means2.items():
        snr_count = 0
        samples_counter_all_snrs = 0
        for mean_log_var_snr in target_samples.values():
            for target_sample in mean_log_var_snr:
                source_target_threshold = enhanced_dict_mean[zero_counter_inx][snr_count] + stds * enhanced_dict_std[zero_counter_inx][snr_count]
                samples_counter_all_snrs += 1
                if target_sample > source_target_threshold:
                    miss_detections[zero_counter_inx] += 1
            snr_count +=1 
        miss_detections[zero_counter_inx] /= samples_counter_all_snrs
    

    false_alarms = [fa for fa in false_alarms.values()]
    miss_detections = [md for md in miss_detections.values()]

    plt.figure()
    plt.plot(zero_counter_every_list, false_alarms, label=f'{speech_type_test}{suffix}', linewidth=3.0) # False alarms
    plt.plot(zero_counter_every_list, miss_detections, label=f'{speech_type_target_test}{suffix2}', linewidth=3.0) # Miss detections
    plt.ylim(bottom=-0.02, top=1.02)
    plt.xlabel('k', fontsize=19)
    plt.ylabel('Detection', fontsize=19)
    plt.yticks(fontsize=15)
    k_list = [1] + list(range(5, max_k + 1, 5))
    plt.xticks(ticks=k_list, labels=k_list, fontsize=15)
    sub_suffix = f'_musan' if cfg.noise_type_test == 'musan' else '_wham'
    sub_suffix2 = f'_musan' if cfg.noise_type_target_test == 'musan' else '_wham'
    suffix_both = f'_{cfg.speech_type_target_test}{sub_suffix2}_both'
    plt.legend(fontsize=14, loc='center right')
    plt.tight_layout()
    plt.savefig(f'reports/detection_results/detect_accuracy_{cfg.speech_type_test}{sub_suffix}{suffix_both}_{stds}_stds_{max_k}_k.pdf', dpi=40)        
    plt.close()
    
   
if __name__ == "__main__":
    main()