import torch
import pytorch_lightning as pl

from model_parts import *
from eval_utils import si_sdri_loss

def nll_criterion_gaussian(mu, logvar, target, reduction='mean'):
    loss = torch.exp(-logvar) * torch.pow(target-mu, 2).mean(dim=-1) + logvar
    return loss.mean() if reduction == 'mean' else loss.sum()


class UNet_Down(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet_Down, self).__init__()

        last_down_factor = (32, 2)
        self.eps = 1e-6

        in_channels = 2
        out_channels = 2

        var_out_dim = 1
        var_out_dim_spec = 1

        pred_cnn_input_size = 2

        time_squeeze_inp_size = 19

        factor = 2 if bilinear else 1
        hidden_size = 64

        down3_dim = hidden_size * 8


        self.inc = DoubleConv(in_channels, hidden_size)
        self.down1 = Down(hidden_size, hidden_size * 2)
        self.down2 = Down(hidden_size * 2, hidden_size * 4)
        self.down3 = Down(hidden_size * 4, hidden_size * 8)
        self.down4 = Down(down3_dim, hidden_size * 16 // factor)
       
        self.up1 = Up(hidden_size * 16, hidden_size * 8 // factor)
        self.up2 = Up(hidden_size * 8, hidden_size * 4 // factor)
        self.up3 = Up(hidden_size * 4, hidden_size * 2 // factor)
        self.up4 = Up(hidden_size * 2, hidden_size)
        self.outputs = OutConv(hidden_size, out_channels)

        self.outputs_var = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(hidden_size // 2, var_out_dim, kernel_size=1)
        )

        self.outputs_spec = OutConv(hidden_size, var_out_dim_spec)

        self.pred_cnn = nn.Sequential(
            DoubleConv(pred_cnn_input_size, hidden_size), 
            Down(hidden_size, hidden_size * 2),
            Down(hidden_size * 2, hidden_size * 4),
            Down(hidden_size * 4, hidden_size * 8),
            Down(hidden_size * 8, hidden_size * 16 // factor, down_factor=last_down_factor)
        )

        d_model = 512
        attn_n_heads = 8
        attn_n_layers = 1
        
        encod_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=attn_n_heads, batch_first=True)
        if attn_n_layers > 1:
            self.encod_attn = nn.TransformerEncoder(encod_layer, num_layers=attn_n_layers)
        else:
            self.encod_attn = encod_layer

        feature_squeeze_in = d_model
        
        self.feature_squeeze_linear = nn.Sequential(
            nn.Linear(feature_squeeze_in, feature_squeeze_in // 2),
            nn.PReLU(),
            nn.Linear(feature_squeeze_in // 2, 1)
        )

        self.time_squeeze_linear = nn.Sequential(
            nn.Linear(time_squeeze_inp_size, time_squeeze_inp_size // 2),
            nn.PReLU(),
            nn.Linear(time_squeeze_inp_size // 2, 1)
        )
        
        encod_layer_var = nn.TransformerEncoderLayer(d_model=d_model, nhead=attn_n_heads, batch_first=True)
        if attn_n_layers > 1:
            self.encod_attn_var = nn.TransformerEncoder(encod_layer_var, num_layers=attn_n_layers)
        else:
            self.encod_attn_var = encod_layer_var
        
        self.feature_squeeze_linear_var = nn.Sequential(
            nn.Linear(feature_squeeze_in, feature_squeeze_in // 2),
            nn.PReLU(),
            nn.Linear(feature_squeeze_in // 2, 1)
        )

        self.time_squeeze_linear_var = nn.Sequential(
            nn.Linear(time_squeeze_inp_size, time_squeeze_inp_size // 2),
            nn.PReLU(),
            nn.Linear(time_squeeze_inp_size // 2, 1)
        )

    def forward(self, x):
        noisy = x.clone()
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outputs(x)

        out_var = self.outputs_var(x)

        out_spec = self.outputs_spec(x)

        input_real = noisy[:, 0, ...]
        input_imag = noisy[:, 1, ...]
        input_comlex = input_real + 1j*input_imag
        input_spec = torch.log10(torch.abs(input_comlex) + self.eps).unsqueeze(1)

        out_var = out_var.detach()
        out_spec = out_spec.detach()

        out_spec_norm = out_spec
        input_spec_norm = input_spec

        channels_concat = torch.cat((out_spec_norm, input_spec_norm), dim=1)
        
        pred_var_concat_inp = self.pred_cnn(channels_concat)

        pred_var_concat = torch.flatten(pred_var_concat_inp, start_dim=1, end_dim=2).permute(0, 2, 1)
        pred_var_concat = self.encod_attn(pred_var_concat)

        features_squeezed = self.feature_squeeze_linear(pred_var_concat).squeeze(-1)
        pred = self.time_squeeze_linear(features_squeezed).squeeze(-1)

        pred_var_concat_2 = torch.flatten(pred_var_concat_inp, start_dim=1, end_dim=2).permute(0, 2, 1)
        pred_var_concat_2 = self.encod_attn_var(pred_var_concat_2)

        features_squeezed_var = self.feature_squeeze_linear_var(pred_var_concat_2).squeeze(-1)
        pred_var = self.time_squeeze_linear_var(features_squeezed_var).squeeze(-1)

        return out, pred, pred_var


class Pl_UNet_enhance_reg(pl.LightningModule):
    def __init__(self, cfg, speech_type='librispeech', noise_type='wham'):
        super().__init__()

        self.model = UNet_Down()

        if speech_type == 'librispeech':
            if noise_type == 'musan':
                bestri_sisdr_path = 'models/enhancement/librispeech/musan/epoch-epoch=413-val-loss-val_loss=-14.828.ckpt'
            else: # WHAM!
                bestri_sisdr_path = 'models/enhancement/librispeech/wham/epoch-epoch=258-val-loss-val_loss=-10.605.ckpt'
        else: # DNS
            if noise_type == 'musan':
                bestri_sisdr_path = 'models/enhancement/dns/musan/epoch-epoch=413-val-loss-val_loss=-13.045.ckpt'
            else: # DNS noises
                bestri_sisdr_path = 'models/enhancement/dns/dns_noises/epoch-epoch=325-val-loss-val_loss=-12.021.ckpt'
        
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in torch.load(bestri_sisdr_path)["state_dict"].items() if k.split('.')[0] == 'model'}
        model_dict = self.model.state_dict()

        for k, v in pretrained_dict.items():
            model_dict[k] = v

        self.model.load_state_dict(model_dict)

        self.eps = 1e-6
        
        for name, param in self.model.named_parameters():
            if 'cnn' not in name.split('.')[0].split('_') and 'linear' not in name.split('.')[0].split('_') and 'attn' not in name.split('.')[0].split('_'):
                param.requires_grad = False

    def forward_train(self,x):
        noisy = x.clone()
        self.model.eval()
        x1 = self.model._modules["inc"](x)
        x2 = self.model._modules["down1"](x1)
        x3 = self.model._modules["down2"](x2)
        x4 = self.model._modules["down3"](x3)
        x5 = self.model._modules["down4"](x4)

        x = self.model._modules["up1"](x5, x4)
        x = self.model._modules["up2"](x, x3)
        x = self.model._modules["up3"](x, x2)
        x = self.model._modules["up4"](x, x1)

        out = self.model._modules["outputs"](x)

        out_var = self.model._modules["outputs_var"](x)
        out_spec = self.model._modules["outputs_spec"](x)

        self.model.train()

        input_real = noisy[:, 0, ...]
        input_imag = noisy[:, 1, ...]
        input_comlex = input_real + 1j*input_imag
        input_spec = torch.log10(torch.abs(input_comlex) + self.eps).unsqueeze(1)

        out_var = out_var.detach()
        out_spec = out_spec.detach()

        out_spec_norm = out_spec
        input_spec_norm = input_spec

        channels_concat = torch.cat((out_spec_norm, input_spec_norm), dim=1)
       
        pred_var_concat_inp = self.model._modules["pred_cnn"](channels_concat)

        pred_var_concat = torch.flatten(pred_var_concat_inp, start_dim=1, end_dim=2).permute(0, 2, 1)
        pred_var_concat = self.model._modules["encod_attn"](pred_var_concat)

        features_squeezed = self.model._modules["feature_squeeze_linear"](pred_var_concat).squeeze(-1)
        
        pred = self.model._modules["time_squeeze_linear"](features_squeezed).squeeze(-1)

        pred_var_concat_2 = torch.flatten(pred_var_concat_inp, start_dim=1, end_dim=2).permute(0, 2, 1)
        pred_var_concat_2 = self.model._modules["encod_attn_var"](pred_var_concat_2)

        features_squeezed_var = self.model._modules["feature_squeeze_linear_var"](pred_var_concat_2).squeeze(-1)
        pred_var = self.model._modules["time_squeeze_linear_var"](features_squeezed_var).squeeze(-1)

        return out, pred, pred_var

    
    def forward(self, x):
        y = self.model(x)

        return y

    def predict_sisdr(self, noisy, enhanced):
        input_real = noisy[:, 0, ...]
        input_imag = noisy[:, 1, ...]
        input_complex = input_real + 1j*input_imag
        input_spec = torch.log10(torch.abs(input_complex) + self.eps).unsqueeze(1)

        enhanced_real = enhanced[:, 0, ...]
        enhanced_imag = enhanced[:, 1, ...]
        enhanced_complex = enhanced_real + 1j*enhanced_imag
        out_spec = torch.log10(torch.abs(enhanced_complex) + self.eps).unsqueeze(1)

        out_spec = out_spec.detach()

        channels_concat = torch.cat((out_spec, input_spec), dim=1)
        pred_var_concat_inp = self.model._modules["pred_cnn"](channels_concat)

        pred_var_concat = torch.flatten(pred_var_concat_inp, start_dim=1, end_dim=2).permute(0, 2, 1)
        pred_var_concat = self.model._modules["encod_attn"](pred_var_concat)

        features_squeezed = self.model._modules["feature_squeeze_linear"](pred_var_concat).squeeze(-1)

        pred = self.model._modules["time_squeeze_linear"](features_squeezed).squeeze(-1)

        pred_var_concat_2 = torch.flatten(pred_var_concat_inp, start_dim=1, end_dim=2).permute(0, 2, 1)
        pred_var_concat_2 = self.model._modules["encod_attn_var"](pred_var_concat_2)

        features_squeezed_var = self.model._modules["feature_squeeze_linear_var"](pred_var_concat_2).squeeze(-1)
        pred_var = self.model._modules["time_squeeze_linear_var"](features_squeezed_var).squeeze(-1)

        return enhanced, pred, pred_var
    
    def training_step(self, batch, batch_idx):
        input_map, _, mixed, _, _ = batch

        enhanced_pred, pred, pred_var = self.forward_train(input_map)
        
        clean_raw = mixed[:, 2]
        noisy_raw = mixed[:, 0]

        pred_real = enhanced_pred[:, 0, :, :].float()
        pred_imag = enhanced_pred[:, 1, :, :].float()

        enhanced_spec = pred_real + 1j*pred_imag

        _, sisdri_enhanced = si_sdri_loss(enhanced_spec, clean_raw, noisy_raw, enhanced_spec.device, mean_value=False)
    
        loss = nll_criterion_gaussian(pred, pred_var, sisdri_enhanced)
        
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_map, _, mixed, _, _ = batch

        enhanced_pred, pred, pred_var = self(input_map)
        
        clean_raw = mixed[:, 2]
        noisy_raw = mixed[:, 0]

        pred_real = enhanced_pred[:, 0, :, :].float()
        pred_imag = enhanced_pred[:, 1, :, :].float()

        enhanced_spec = pred_real + 1j*pred_imag

        _, sisdri_enhanced = si_sdri_loss(enhanced_spec, clean_raw, noisy_raw, enhanced_spec.device, mean_value=False)
    
        loss = nll_criterion_gaussian(pred, pred_var, sisdri_enhanced)
      
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))

        return optimizer