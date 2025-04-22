import os
import torch

def enhance_postprocess_short_sisdr(enhanced_spec):

    enhanced_var_spec = []

    enhanced_target_origin = enhanced_spec
    enhanced_spec = enhanced_target_origin[0]
    enhanced_var_spec = enhanced_target_origin[1]
    enhanced_sisdr_var = enhanced_target_origin[2]

    enhanced_var_spec = [enhanced_var_spec, enhanced_sisdr_var]

    enhanced_target_origin = enhanced_spec
    enhanced_real = enhanced_target_origin[:, 0, :, :]
    enhanced_imag = enhanced_target_origin[:, 1, :, :]
    enhanced_spec = enhanced_real + 1j*enhanced_imag

    return enhanced_spec, enhanced_var_spec

def enhance_postprocess_sisdr(cfg, raw, input_map, enhanced_spec, device):

    window_size = cfg.window_size

    clean = raw[:, 0, 2, :].numpy()
    noise = raw[:, 0, 1, :].numpy()
    noisy = raw[:, 0, 0, :].numpy()

    enhanced_spec = enhanced_spec.squeeze()[..., :input_map.shape[-1]]
    
    enhanced = torch.istft(enhanced_spec.squeeze(), n_fft=window_size, window=torch.hamming_window(window_size).to(device), hop_length=int(window_size * cfg.overlap), length=noisy.shape[-1])

    if len(enhanced) > len(clean):
        enhanced = enhanced[:len(clean)]
    else:
        clean = clean[:len(enhanced)]
        noise = noise[:len(enhanced)]
    
    noisy = noisy[:len(enhanced)]
        
    enhanced = enhanced.detach().cpu().numpy()

    return enhanced, clean, noisy, noise

def load_model(cfg, model, device, speech_type='librispeech', noise_type='wham'):
    if speech_type == 'librispeech':
        if noise_type == 'musan':
            path_to_ckpt = 'models/detection_model/librispeech_musan_source/epoch-epoch=19-val-loss-val_loss=1.301.ckpt'
        else: # WHAM
            path_to_ckpt = 'models/detection_model/librispeech_wham_source/epoch-epoch=26-val-loss-val_loss=0.277.ckpt'
    else: # DNS
        path_to_ckpt = 'models/detection_model/dns_dns_source/epoch-epoch=12-val-loss-val_loss=0.995.ckpt'

    os.chdir('../../../')
    
    model = model.load_from_checkpoint(
    checkpoint_path=path_to_ckpt,
    cfg=cfg
    ).to(device)

    return model

def si_sdri_calc(mix, estimated, original, mean_value=True):
    sisdri_enhanced = si_sdr_torchaudio_calc(estimated, original, mean_value=mean_value)
    return sisdri_enhanced - si_sdr_torchaudio_calc(mix, original), sisdri_enhanced

####========torchaudio si_sdr===========######

def si_sdr_torchaudio_calc(estimate, reference, epsilon=1e-8, mean_value=True):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    sisdr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    if mean_value:
        return torch.mean(sisdr)
    else:
        return sisdr

def si_sdri_loss(enhanced_spec, clean, noisy, device, mean_value=True):

    window_size = 512
    overlap = 0.5
    
    enhanced = torch.istft(enhanced_spec.squeeze(), n_fft=window_size, window=torch.hamming_window(window_size).to(device), hop_length=int(window_size * overlap), length=clean.shape[-1])
    
    si_sdri, sisdri_enhanced = si_sdri_calc(noisy, enhanced, clean, mean_value=mean_value)

    return -si_sdri, sisdri_enhanced



