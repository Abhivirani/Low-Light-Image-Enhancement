import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, num_channels, reduction_ratio=8):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_squeeze = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.fc_excite  = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.shape
        # Squeeze
        attn_weights = self.global_avg_pool(x).view(batch_size, num_channels)
        # Excitation
        attn_weights = self.relu(self.fc_squeeze(attn_weights))
        attn_weights = self.sigmoid(self.fc_excite(attn_weights))
        attn_weights = attn_weights.view(batch_size, num_channels, 1, 1)
        return x * attn_weights


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # Quantization-safe add
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class LLIE_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.quant   = QuantStub()
        self.dequant = DeQuantStub()
        self.use_attention = True   
        self.initial = nn.Conv2d(3, 64, 3, padding=1)

        self.enc1  = ResidualBlock(64)
        self.down1 = nn.Conv2d(64, 96, 3, stride=2, padding=1)

        self.enc2  = ResidualBlock(96)
        self.down2 = nn.Conv2d(96, 128, 3, stride=2, padding=1)

        self.enc3  = ResidualBlock(128)
        self.down3 = nn.Conv2d(128, 196, 3, stride=2, padding=1)

        # NEW: bottleneck + channel attention
        self.bottleneck         = ResidualBlock(196)
        self.bottleneck_attn    = ChannelAttention(196, reduction_ratio=8)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(196, 128, 3, padding=1)
        )
        self.dec3 = ResidualBlock(128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, 3, padding=1)
        )
        self.dec2 = ResidualBlock(96)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, 3, padding=1)
        )
        self.dec1 = ResidualBlock(64)

        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)
        # NEW: Sigmoid instead of Tanh — output directly in [0,1]
        self.final_act  = nn.Sigmoid()

        # Quantization-safe skip-connection adds
        self.skip_add1 = nn.quantized.FloatFunctional()
        self.skip_add2 = nn.quantized.FloatFunctional()
        self.skip_add3 = nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.quant(x)

        feat_init = self.initial(x)

        feat_enc1  = self.enc1(feat_init)
        feat_down1 = self.down1(feat_enc1)

        feat_enc2  = self.enc2(feat_down1)
        feat_down2 = self.down2(feat_enc2)

        feat_enc3  = self.enc3(feat_down2)
        feat_down3 = self.down3(feat_enc3)

        # Bottleneck + attention
        feat_bottleneck = self.bottleneck(feat_down3)
        if self.use_attention:
            feat_bottleneck = self.bottleneck_attn(feat_bottleneck)

        # Decoder with skip connections
        feat_up3   = self.skip_add1.add(self.up3(feat_bottleneck), feat_enc3)
        feat_dec3  = self.dec3(feat_up3)

        feat_up2   = self.skip_add2.add(self.up2(feat_dec3), feat_enc2)
        feat_dec2  = self.dec2(feat_up2)

        feat_up1   = self.skip_add3.add(self.up1(feat_dec2), feat_enc1)
        feat_dec1  = self.dec1(feat_up1)

        out = self.final_conv(feat_dec1)
        out = self.final_act(out)     # NEW: Sigmoid → [0,1] directly

        out = self.dequant(out)
        return out


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using relu3_3 features."""
    def __init__(self):
        super().__init__()
        vgg_features = tv_models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg_features.parameters():
            param.requires_grad = False
        self.vgg_feature_extractor = vgg_features.to(device)

    def forward(self, pred, target):
        pred_features   = self.vgg_feature_extractor(pred)
        target_features = self.vgg_feature_extractor(target)
        return nn.functional.l1_loss(pred_features, target_features)


class FFTFrequencyLoss(nn.Module):
    """Frequency-domain L1 loss — enforces sharpness and edge fidelity."""
    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return nn.functional.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


# Instantiate all loss functions
l1_criterion          = nn.L1Loss()
perceptual_criterion  = PerceptualLoss()
fft_criterion         = FFTFrequencyLoss()

# Loss weights
LAMBDA_L1          = 1.0
LAMBDA_PERCEPTUAL  = 0.1
LAMBDA_FFT         = 0.05

def compute_combined_loss(pred, target):
    loss_l1         = l1_criterion(pred, target)
    loss_perceptual = perceptual_criterion(pred, target)
    loss_fft        = fft_criterion(pred, target)
    total_loss = (LAMBDA_L1         * loss_l1
                + LAMBDA_PERCEPTUAL * loss_perceptual
                + LAMBDA_FFT        * loss_fft)
    return total_loss, loss_l1, loss_perceptual, loss_fft


