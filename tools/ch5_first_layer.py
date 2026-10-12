"""ข้อมูลบทที่ 5 (convolution) -> data/ch5_first_layer.json

ดึงชั้น Conv ชั้นแรกของ yolo11n-seg.pt: weight ของ kernel 16 ตัว (16, 3, 3, 3) + ค่า BatchNorm
และ shape/จำนวน parameter ของ Conv 5 ชั้นแรกใน backbone เมื่อป้อนภาพ (1, 3, 640, 640)

ถ้าใส่ไฟล์ภาพ PNG ขนาด 640x640 เป็นอาร์กิวเมนต์ จะรันชั้นแรกจริงด้วย PyTorch แล้วพิมพ์ค่าเอาต์พุตบางตำแหน่ง
ไว้เทียบกับที่เว็บคำนวณด้วย JavaScript:
    cd <temp> && python D:/course-plan-YOLO/tools/ch5_first_layer.py [ภาพ.png]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

OUT = Path(__file__).resolve().parent.parent / "data" / "ch5_first_layer.json"

model = YOLO("yolo11n-seg.pt").model.float().eval()
layers = model.model
conv0 = layers[0]

# shape ของเอาต์พุตแต่ละชั้นด้วย forward hook
shapes = {}
hooks = [m.register_forward_hook(lambda m, i, o, k=k: shapes.__setitem__(k, list(o.shape))) for k, m in enumerate(layers[:5])]
with torch.no_grad():
    model(torch.zeros(1, 3, 640, 640))
for h in hooks:
    h.remove()

convs = []
for k in (0, 1, 3):  # ชั้นที่เป็น Conv ธรรมดาใน 5 ชั้นแรก
    c = layers[k].conv
    convs.append({
        "index": k, "c_in": c.in_channels, "c_out": c.out_channels, "k": c.kernel_size[0],
        "stride": c.stride[0], "padding": c.padding[0], "bias": c.bias is not None,
        "conv_params": sum(p.numel() for p in c.parameters()),
        "bn_params": sum(p.numel() for p in layers[k].bn.parameters()),
        "act": type(layers[k].act).__name__, "out_shape": shapes[k],
    })

bn = conv0.bn
data = {
    "ultralytics": ultralytics.__version__,
    "input_shape": [1, 3, 640, 640],
    "convs": convs,
    "layer_shapes": {str(k): v for k, v in shapes.items()},
    "weight": np.round(conv0.conv.weight.detach().numpy(), 6).tolist(),  # (16, 3, 3, 3) ช่องอินพุตเรียง R, G, B
    "bn": {"gamma": bn.weight.tolist(), "beta": bn.bias.tolist(), "mean": bn.running_mean.tolist(),
           "var": bn.running_var.tolist(), "eps": bn.eps},
}
OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
print(json.dumps(convs, indent=1))

if len(sys.argv) > 1:
    from PIL import Image
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float32) / 255  # HWC, RGB
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = conv0(x)[0].numpy()  # conv → bn → SiLU
    print("output", y.shape)
    for ch, r, c in [(0, 0, 0), (0, 160, 160), (3, 100, 200), (7, 250, 90), (15, 319, 319), (11, 140, 175)]:
        print(f"ch {ch} row {r} col {c}: {y[ch, r, c]:.5f}")
