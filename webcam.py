import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
import time
import argparse

# ---------------------------------------------------------------------------
# MODEL DEFINITION (Copied from notebook)
# ---------------------------------------------------------------------------
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
        attn_weights = self.global_avg_pool(x).view(batch_size, num_channels)
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
        self.final_act  = nn.Sigmoid()

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

        feat_bottleneck = self.bottleneck(feat_down3)
        if self.use_attention:
            feat_bottleneck = self.bottleneck_attn(feat_bottleneck)

        feat_up3   = self.skip_add1.add(self.up3(feat_bottleneck), feat_enc3)
        feat_dec3  = self.dec3(feat_up3)

        feat_up2   = self.skip_add2.add(self.up2(feat_dec3), feat_enc2)
        feat_dec2  = self.dec2(feat_up2)

        feat_up1   = self.skip_add3.add(self.up1(feat_dec2), feat_enc1)
        feat_dec1  = self.dec1(feat_up1)

        out = self.final_conv(feat_dec1)
        out = self.final_act(out)
        out = self.dequant(out)
        return out


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------
def fuse_model_layers(model_to_fuse):
    for module in model_to_fuse.modules():
        if isinstance(module, ResidualBlock):
            tq.fuse_modules(module, ['conv1', 'relu'], inplace=True)

def load_int8_model(model_path):
    print("Loading INT8 model...")
    supported_engines = torch.backends.quantized.supported_engines
    if 'fbgemm' in supported_engines:
        torch.backends.quantized.engine = 'fbgemm'
    elif 'onednn' in supported_engines:
        torch.backends.quantized.engine = 'onednn'
    elif 'qnnpack' in supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
    else:
        torch.backends.quantized.engine = supported_engines[0]
    model = LLIE_Model()
    model.eval()
    fuse_model_layers(model)
    model.qconfig = tq.get_default_qconfig('fbgemm')
    tq.prepare(model, inplace=True)
    model_int8 = tq.convert(model, inplace=False)
    model_int8.use_attention = False # As per training notebook notebooke3d7a332db
    model_int8.load_state_dict(torch.load(model_path, map_location='cpu'))
    model_int8.eval()
    return model_int8

def load_fp32_model(model_path, device):
    print("Loading FP32 model...")
    model = LLIE_Model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# ---------------------------------------------------------------------------
# WEBCAM LOOP
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Webcam Low-Light Image Enhancement")
    parser.add_argument("--model", type=str, choices=["int8", "fp32"], default="int8", help="Model type to use")
    parser.add_argument("--size", type=int, default=384, help="Input size for the model (square)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "int8":
        model = load_int8_model("int8_model.pth")
        device = torch.device("cpu") # INT8 fbgemm requires CPU
    else:
        model = load_fp32_model("fp32_model.pth", device)

    print(f"Model {args.model} loaded. Starting webcam...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Frame properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {orig_w}x{orig_h}")
    
    # GUI config
    window_name = "Low-Light Enhancement (Press 'q' to quit, 't' to toggle enhancement, 's' to save frame)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, orig_w * 2, orig_h)
    
    is_enhanced = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        start_time = time.time()
        
        # Keep copy of original for display
        display_frame = frame.copy()
        
        if is_enhanced:
            # 1. Resize and normalize
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_img, (args.size, args.size))
            input_tensor = torch.tensor(resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # 2. Inference
            with torch.no_grad():
                enhanced_tensor = model(input_tensor)
                
            # 3. Post-process and resize back to original resolution
            enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
            enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
            enhanced_bgr = cv2.resize(enhanced_bgr, (orig_w, orig_h))
            
            output_frame = enhanced_bgr
            
            # Annotate fps
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(output_frame, f"Enhanced ({args.model.upper()}) FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            output_frame = frame
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(output_frame, f"Original FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Place side by side (Original | Enhanced)
        cv2.putText(display_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        side_by_side = np.hstack((display_frame, output_frame))

        cv2.imshow(window_name, side_by_side)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            is_enhanced = not is_enhanced
        elif key == ord('s'):
            filename = f"saved_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, side_by_side)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
