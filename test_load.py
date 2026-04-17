import torch
from model import LLIENet

try:
    print("Testing FP32 model load...")
    fp32_model = LLIENet()
    fp32_model.load_state_dict(torch.load('fp32_model.pth', map_location='cpu'))
    print("FP32 model loaded successfully!")
except Exception as e:
    print(f"Failed FP32: {e}")

try:
    print("\nTesting INT8 model load (fbgemm)...")
    int8_model = LLIENet()
    int8_model.eval()
    int8_model.fuse_model()
    int8_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(int8_model, inplace=True)
    torch.quantization.convert(int8_model, inplace=True)
    int8_model.load_state_dict(torch.load('int8_model.pth', map_location='cpu'))
    print("INT8 model (fbgemm) loaded successfully!")
except Exception as e:
    print(f"Failed INT8 (fbgemm): {e}")

try:
    print("\nTesting INT8 model load (qnnpack)...")
    int8_model = LLIENet()
    int8_model.eval()
    int8_model.fuse_model()
    int8_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(int8_model, inplace=True)
    torch.quantization.convert(int8_model, inplace=True)
    int8_model.load_state_dict(torch.load('int8_model.pth', map_location='cpu'))
    print("INT8 model (qnnpack) loaded successfully!")
except Exception as e:
    print(f"Failed INT8 (qnnpack): {e}")
