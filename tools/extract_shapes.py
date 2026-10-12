"""ข้อมูลสถาปัตยกรรม YOLO11-seg จากการรันจริง -> data/yolo11-seg_shapes.json และ data/ch6_maps/*.png

1) ทุกรุ่น (n, s, m, l, x): ลำดับชั้น, from, จำนวนบล็อก, ชนิดโมดูล, args, shape เอาต์พุต, จำนวนพารามิเตอร์
   (ป้อน tensor หลอก (1, 3, 640, 640) แล้วเก็บด้วย forward hook)
2) รุ่น n: โครงสร้างภายใน C3k2 ทุกตัว (ช่องซ่อน c, ชนิดบล็อกภายใน, มี shortcut หรือไม่)
3) รุ่น n (yolo11n-seg.pt): receptive field ทางทฤษฎีของเซลล์กลางภาพในแต่ละชั้นของ Backbone
   วัดจากบริเวณบนภาพอินพุตที่ gradient ไม่เป็นศูนย์ (สลับ MaxPool เป็น AvgPool ขนาดเดียวกันชั่วคราว
   เพราะ gradient ของ max ไหลผ่านแค่ตำแหน่งที่ชนะ ทำให้วัดเล็กเกินจริง)
4) ถ้าใส่ภาพ PNG 640x640 (ภาพผลไม้หลัง letterbox) จะบันทึก feature map จริงของชั้น 0–10 เป็น PNG

รันในโฟลเดอร์ชั่วคราว:
    cd <temp> && python D:/course-plan-YOLO/tools/extract_shapes.py D:/course-plan-YOLO/data/fruit_letterbox_640.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yolo11-seg_shapes.json"
MAPS = ROOT / "data" / "ch6_maps"


def layer_table(model):
    shapes = {}
    hooks = [m.register_forward_hook(lambda m, i, o, k=k: shapes.__setitem__(k, list(o.shape) if torch.is_tensor(o) else None))
             for k, m in enumerate(model.model)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    for h in hooks:
        h.remove()
    rows = []
    for k, m in enumerate(model.model):
        rows.append({"i": k, "from": m.f, "n": len(m.m) if hasattr(m, "m") and isinstance(m.m, (nn.ModuleList, nn.Sequential)) else 1,
                     "type": m.type.split(".")[-1], "args": [a if isinstance(a, (int, float, bool, str)) or a is None else str(a) for a in m.args] if hasattr(m, "args") else None,
                     "params": sum(p.numel() for p in m.parameters()), "out": shapes.get(k)})
    return rows


data = {"ultralytics": ultralytics.__version__, "input": [1, 3, 640, 640], "scales": {}}
for s in "nsmlx":
    m = SegmentationModel(f"yolo11{s}-seg.yaml", verbose=False)
    rows = layer_table(m)
    data["scales"][s] = {"layers": rows, "params": sum(p.numel() for p in m.parameters())}
    print(s, data["scales"][s]["params"])

pt = YOLO("yolo11n-seg.pt").model.float().eval()
assert sum(p.numel() for p in pt.parameters()) == data["scales"]["n"]["params"]

# โครงสร้างภายใน C3k2 ของรุ่น n
inner = []
for k, m in enumerate(pt.model):
    if type(m).__name__ == "C3k2":
        b = m.m[0]
        inner.append({"i": k, "c_in": m.cv1.conv.in_channels, "c_out": m.cv2.conv.out_channels, "hidden": m.c, "blocks": len(m.m),
                      "block_type": type(b).__name__, "cv1_out": m.cv1.conv.out_channels, "cv2_in": m.cv2.conv.in_channels,
                      "bottleneck_mid": b.cv1.conv.out_channels if hasattr(b, "cv1") and type(b).__name__ == "Bottleneck" else None,
                      "shortcut": bool(getattr(b, "add", False)) if type(b).__name__ == "Bottleneck" else None})
data["c3k2_n"] = inner

# receptive field ของ Backbone (ชั้น 0–10) ในรุ่น n
for mod in pt.modules():
    for name, ch in list(mod.named_children()):
        if isinstance(ch, nn.MaxPool2d):
            setattr(mod, name, nn.AvgPool2d(ch.kernel_size, ch.stride, ch.padding, count_include_pad=False))
rf = []
for L in range(11):
    x = torch.zeros(1, 3, 640, 640, requires_grad=True)
    y, outs = x, []
    for k in range(L + 1):
        m = pt.model[k]
        y = m(y)
    H, W = y.shape[2:]
    y[0, :, H // 2, W // 2].sum().backward()
    g = (x.grad[0].abs().sum(0) > 0).numpy()
    ys, xs = np.where(g)
    rf.append({"layer": L, "out_hw": [int(H), int(W)], "stride": 640 // int(H), "cell": [H // 2, W // 2],
               "y0": int(ys.min()), "y1": int(ys.max()), "x0": int(xs.min()), "x1": int(xs.max()),
               "size": int(xs.max() - xs.min() + 1)})
    print("RF layer", L, rf[-1]["size"], rf[-1]["out_hw"])
data["rf_n"] = rf
OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)

if len(sys.argv) > 1:
    from PIL import Image
    pt = YOLO("yolo11n-seg.pt").model.float().eval()   # โหลดใหม่ (ไม่สลับ pool)
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float32) / 255
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    MAPS.mkdir(exist_ok=True)
    info = []
    with torch.no_grad():
        y = x
        for k in range(11):
            y = pt.model[k](y)
            a = y[0].numpy()
            # ภาพเฉลี่ยทุกช่อง และ 4 ช่องที่มีความแปรปรวนสูงสุด
            def save(arr, name):
                lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
                Image.fromarray((np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)).save(MAPS / name)
            save(a.mean(0), f"L{k}_mean.png")
            top = np.argsort(a.reshape(a.shape[0], -1).var(1))[::-1][:4].tolist()
            for j, c in enumerate(top):
                save(a[c], f"L{k}_top{j}.png")
            info.append({"layer": k, "shape": list(y.shape), "top_channels": top})
    (MAPS / "index.json").write_text(json.dumps({"ultralytics": ultralytics.__version__, "input": Path(sys.argv[1]).name, "maps": info}), encoding="utf-8")
    print("wrote", MAPS)
