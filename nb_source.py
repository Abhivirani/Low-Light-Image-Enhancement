
# --- CELL 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import time
from torch.utils.data import Dataset, DataLoader
from torch.ao.quantization import QuantStub, DeQuantStub
import torch.ao.quantization as tq
import torchvision.models as tv_models

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --- CELL 3 ---
BASE_PATH = "/kaggle/input/datasets/tanhyml/lol-v2-dataset/LOL-v2/Real_captured"
TRAIN_LOW  = BASE_PATH + "/Train/Low"
TRAIN_HIGH = BASE_PATH + "/Train/Normal"

TEST_LOW   = BASE_PATH + "/Test/Low"
TEST_HIGH  = BASE_PATH + "/Test/Normal"

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --- CELL 5 ---
class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, img_size=384, augment=False):
        self.low_dir  = low_dir
        self.high_dir = high_dir
        self.img_size = img_size
        self.augment  = augment          # NEW: augmentation flag

        self.low_images  = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        assert len(self.low_images) == len(self.high_images)

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path  = os.path.join(self.low_dir,  self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low_img  = cv2.imread(low_path)
        high_img = cv2.imread(high_path)

        low_img  = cv2.cvtColor(low_img,  cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        # NEW: higher resolution for more detail
        low_img  = cv2.resize(low_img,  (self.img_size, self.img_size))
        high_img = cv2.resize(high_img, (self.img_size, self.img_size))

        low_tensor  = torch.tensor(low_img  / 255.0).permute(2, 0, 1).float()
        high_tensor = torch.tensor(high_img / 255.0).permute(2, 0, 1).float()

        # NEW: paired augmentation (same transform on both images)
        if self.augment:
            if random.random() > 0.5:
                low_tensor  = torch.flip(low_tensor,  [2])   # horizontal flip
                high_tensor = torch.flip(high_tensor, [2])
            if random.random() > 0.5:
                low_tensor  = torch.flip(low_tensor,  [1])   # vertical flip
                high_tensor = torch.flip(high_tensor, [1])

        return low_tensor, high_tensor


# --- CELL 7 ---
IMG_SIZE   = 384   # NEW: was 256
BATCH_SIZE = 4     # NEW: was 8 (reduced to fit 384×384 in GPU memory)

train_dataset = LowLightDataset(TRAIN_LOW, TRAIN_HIGH, img_size=IMG_SIZE, augment=True)
test_dataset  = LowLightDataset(TEST_LOW,  TEST_HIGH,  img_size=IMG_SIZE, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print("Train samples:", len(train_dataset))
print("Test  samples:", len(test_dataset))


# --- CELL 9 ---
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


# --- CELL 10 ---
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


# --- CELL 13 ---
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


# --- CELL 15 ---
model = LLIE_Model().to(device)
print("Model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))


# --- CELL 17 ---
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


# --- CELL 19 ---
NUM_EPOCHS = 30

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

# NEW: Cosine annealing with warm restarts
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

best_val_loss   = float('inf')
loss_history    = []

# Checkpoint helpers
def save_checkpoint(epoch, model, optimizer, lr_scheduler, loss, filepath):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'loss':                 loss,
    }, filepath)
    print(f"  ✅ Checkpoint saved → {filepath}")

def load_checkpoint(filepath, model, optimizer=None, lr_scheduler=None):
    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f"  ✅ Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return ckpt['epoch'], ckpt['loss']


# --- CELL 21 ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_total_loss = 0.0

    for low_batch, high_batch in train_loader:
        low_batch  = low_batch.to(device)
        high_batch = high_batch.to(device)

        optimizer.zero_grad()

        # Forward — output already in [0,1] via Sigmoid (no shift needed)
        pred_batch = model(low_batch)

        # Combined loss
        total_loss, loss_l1, loss_perceptual, loss_fft = compute_combined_loss(
            pred_batch, high_batch
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_total_loss += total_loss.item()

    epoch_loss = running_total_loss / len(train_loader)
    loss_history.append(epoch_loss)

    # Step LR scheduler
    lr_scheduler.step(epoch)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | Loss: {epoch_loss:.4f} | LR: {current_lr:.2e}")

    # ── Save checkpoint every 5 epochs ──
    if (epoch + 1) % 5 == 0:
        periodic_ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1:03d}.pth"
        )
        save_checkpoint(epoch + 1, model, optimizer, lr_scheduler,
                        epoch_loss, periodic_ckpt_path)

    # ── Save best model ──
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        best_model_path = os.path.join(CHECKPOINT_DIR, "model_best.pth")
        save_checkpoint(epoch + 1, model, optimizer, lr_scheduler,
                        epoch_loss, best_model_path)
        torch.save(model.state_dict(), "/kaggle/working/model_fp32.pth")

print("\nTraining complete. Best loss:", best_val_loss)


# --- CELL 23 ---
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Combined Loss")
plt.title("Training Loss History")
plt.grid(True)
plt.tight_layout()
plt.savefig("/kaggle/working/loss_curve.png", dpi=150)
plt.show()


# --- CELL 25 ---
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * torch.log10(1.0 / mse)


# --- CELL 27 ---
# Load best checkpoint weights into model
model.load_state_dict(torch.load("/kaggle/working/model_fp32.pth"))
model.eval()

psnr_fp32_total = 0.0

with torch.no_grad():
    for low_img, high_img in test_loader:
        low_img  = low_img.to(device)
        high_img = high_img.to(device)

        # Output already in [0,1] — no shift needed
        pred_img = model(low_img)

        psnr_fp32_total += compute_psnr(pred_img, high_img).item()

avg_psnr_fp32 = psnr_fp32_total / len(test_loader)
print("FP32 PSNR:", avg_psnr_fp32)


# --- CELL 29 ---
model_fp32_for_quant = LLIE_Model()
model_fp32_for_quant.load_state_dict(torch.load("/kaggle/working/model_fp32.pth"))
model_fp32_for_quant.eval()
model_fp32_for_quant.cpu()


# --- CELL 30 ---
import torch.ao.quantization as tq

torch.backends.quantized.engine = 'fbgemm'
model_fp32_for_quant.qconfig = tq.get_default_qconfig('fbgemm')

def fuse_model_layers(model_to_fuse):
    for module in model_to_fuse.modules():
        if isinstance(module, ResidualBlock):
            tq.fuse_modules(module, ['conv1', 'relu'], inplace=True)

fuse_model_layers(model_fp32_for_quant)
tq.prepare(model_fp32_for_quant, inplace=True)

# Calibration pass
with torch.no_grad():
    for calib_idx, (low_calib, _) in enumerate(train_loader):
        model_fp32_for_quant(low_calib)
        if calib_idx > 50:
            break

print("Calibration complete.")


# --- CELL 31 ---
model_int8 = tq.convert(model_fp32_for_quant, inplace=False)
model_int8.eval()
model_int8.use_attention = False   
print("INT8 model ready.")

# --- CELL 32 ---
psnr_int8_total = 0.0

with torch.no_grad():
    for low_img, high_img in test_loader:
        pred_int8 = model_int8(low_img.cpu())
        psnr_int8_total += compute_psnr(pred_int8, high_img.cpu()).item()

avg_psnr_int8 = psnr_int8_total / len(test_loader)

print("FP32 PSNR:", avg_psnr_fp32)
print("INT8 PSNR:", avg_psnr_int8)
print("PSNR Drop:", avg_psnr_fp32 - avg_psnr_int8)


# --- CELL 33 ---
torch.save(model_int8.state_dict(), "model_int8_real.pth")

import os
print("INT8 Size (MB):", os.path.getsize("model_int8_real.pth") / (1024 * 1024))


# --- CELL 35 ---
model_fp32_for_quant.eval().cpu()
model_int8.eval().cpu()

for sample_idx in range(7):
    low_sample, high_sample = test_dataset[sample_idx]
    low_input = low_sample.unsqueeze(0).cpu()

    with torch.no_grad():
        pred_fp32 = model_fp32_for_quant(low_input)
        pred_int8 = model_int8(low_input)

    low_vis   = low_sample.permute(1, 2, 0).numpy()
    high_vis  = high_sample.permute(1, 2, 0).numpy()
    fp32_vis  = pred_fp32.squeeze().permute(1, 2, 0).numpy()
    int8_vis  = pred_int8.squeeze().permute(1, 2, 0).numpy()

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(low_vis)
    plt.title("Low Light Input")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(fp32_vis)
    plt.title("FP32 Output")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(int8_vis)
    plt.title("INT8 Output")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(high_vis)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# --- CELL 37 ---
import time, os

model.eval().to(device)
model_int8.eval().cpu()

NUM_BENCHMARK_IMGS = 100

# ── FP32 GPU speed ──
start_fp32_gpu = time.time()
with torch.no_grad():
    for bench_idx in range(NUM_BENCHMARK_IMGS):
        low_bench, _ = test_dataset[bench_idx]
        low_bench = low_bench.unsqueeze(0).to(device)
        _ = model(low_bench)
if device.type == "cuda":
    torch.cuda.synchronize()
end_fp32_gpu = time.time()

fp32_gpu_time = (end_fp32_gpu - start_fp32_gpu) / NUM_BENCHMARK_IMGS
fp32_gpu_fps  = 1.0 / fp32_gpu_time

# ── FP32 CPU speed ──
model_fp32_cpu = LLIE_Model()
model_fp32_cpu.load_state_dict(torch.load("/kaggle/working/model_fp32.pth"))
model_fp32_cpu.eval().cpu()

start_fp32_cpu = time.time()
with torch.no_grad():
    for bench_idx in range(NUM_BENCHMARK_IMGS):
        low_bench, _ = test_dataset[bench_idx]
        _ = model_fp32_cpu(low_bench.unsqueeze(0).cpu())
end_fp32_cpu = time.time()

fp32_cpu_time = (end_fp32_cpu - start_fp32_cpu) / NUM_BENCHMARK_IMGS
fp32_cpu_fps  = 1.0 / fp32_cpu_time

# ── INT8 CPU speed ──
start_int8 = time.time()
with torch.no_grad():
    for bench_idx in range(NUM_BENCHMARK_IMGS):
        low_bench, _ = test_dataset[bench_idx]
        _ = model_int8(low_bench.unsqueeze(0).cpu())
end_int8 = time.time()

int8_cpu_time = (end_int8 - start_int8) / NUM_BENCHMARK_IMGS
int8_cpu_fps  = 1.0 / int8_cpu_time

# ── Model sizes ──
torch.save(model.state_dict(), "fp32_model.pth")
torch.save(model_int8.state_dict(), "int8_model.pth")

fp32_size_mb = os.path.getsize("fp32_model.pth") / (1024 * 1024)
int8_size_mb = os.path.getsize("int8_model.pth") / (1024 * 1024)
size_reduction_pct = ((fp32_size_mb - int8_size_mb) / fp32_size_mb) * 100

print("\n========== FINAL RESULTS ==========\n")
print(" QUALITY:")
print(f"  FP32 PSNR : {avg_psnr_fp32:.4f} dB")
print(f"  INT8 PSNR : {avg_psnr_int8:.4f} dB")
print(f"  PSNR Drop : {avg_psnr_fp32 - avg_psnr_int8:.4f} dB")
print("\n SPEED:")
print(f"  FP32 GPU FPS : {fp32_gpu_fps:.2f}")
print(f"  FP32 CPU FPS : {fp32_cpu_fps:.2f}")
print(f"  INT8 CPU FPS : {int8_cpu_fps:.2f}")
print("\n MODEL SIZE:")
print(f"  FP32 : {fp32_size_mb:.2f} MB")
print(f"  INT8 : {int8_size_mb:.2f} MB")
print(f"  Reduction : {size_reduction_pct:.2f}%")
print("\n===================================\n")


# --- CELL 39 ---
!pip install lpips scikit-image

# --- CELL 40 ---
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
lpips_loss_fn.eval()


# --- CELL 41 ---
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * torch.log10(1.0 / mse)

def compute_ssim(img1, img2):
    img1_np = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)

def compute_lpips(img1, img2):
    # LPIPS expects inputs in [-1, 1]
    img1_scaled = img1 * 2 - 1
    img2_scaled = img2 * 2 - 1
    return lpips_loss_fn(img1_scaled, img2_scaled).item()


# --- CELL 42 ---
psnr_metric_total  = 0.0
ssim_metric_total  = 0.0
lpips_metric_total = 0.0

model.eval()

with torch.no_grad():
    for low_eval, high_eval in test_loader:
        low_eval  = low_eval.to(device)
        high_eval = high_eval.to(device)

        pred_eval = model(low_eval)
        # Output already in [0,1] via Sigmoid

        psnr_metric_total  += compute_psnr(pred_eval, high_eval).item()
        ssim_metric_total  += compute_ssim(pred_eval, high_eval)
        lpips_metric_total += compute_lpips(pred_eval, high_eval)

avg_psnr_metric  = psnr_metric_total  / len(test_loader)
avg_ssim_metric  = ssim_metric_total  / len(test_loader)
avg_lpips_metric = lpips_metric_total / len(test_loader)

print("===== EXTENDED METRICS =====")
print(f"PSNR  : {avg_psnr_metric:.4f} dB")
print(f"SSIM  : {avg_ssim_metric:.4f}")
print(f"LPIPS : {avg_lpips_metric:.4f}  (lower is better)")


# --- CELL 44 ---
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

demo_model = LLIE_Model().to(device)
demo_model.load_state_dict(torch.load("/kaggle/working/model_fp32.pth"))
demo_model.eval()


# --- CELL 45 ---
def enhance_image(image_path, img_size=384):
    """Enhance a single low-light image using the trained model."""
    raw_img   = cv2.imread(image_path)
    rgb_img   = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(rgb_img, (img_size, img_size))
    normalised = resized / 255.0

    input_tensor = (torch.tensor(normalised)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(device))

    with torch.no_grad():
        # Output already in [0,1]
        enhanced_tensor = demo_model(input_tensor)

    enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_np = np.clip(enhanced_np, 0, 1)

    return rgb_img, resized, enhanced_np


# --- CELL 46 ---
low_img_demo, resized_demo, enhanced_demo = enhance_image(
    "/kaggle/input/datasets/abhivirani2005/dataset1/WhatsApp Image 2026-04-05 at 4.02.44 PM.jpeg"
)


# --- CELL 47 ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(low_img_demo)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(resized_demo)
plt.title("Low Light (Resized)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(enhanced_demo)
plt.title("Enhanced Output")
plt.axis('off')

plt.tight_layout()
plt.show()


# --- CELL 48 ---
# Save enhanced output
enhanced_bgr = cv2.cvtColor(
    (enhanced_demo * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
)
cv2.imwrite("/kaggle/working/enhanced_output.jpg", enhanced_bgr)
print("Saved enhanced image at: /kaggle/working/enhanced_output.jpg")

