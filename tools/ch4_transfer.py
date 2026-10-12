"""ข้อมูลบทที่ 4 (transfer learning) -> data/ch4_transfer.json

1) นับว่าน้ำหนักจาก yolo11n-seg.pt ถูกโอนเข้าโมเดลใหม่ได้กี่ชิ้น เมื่อ nc = 80 (เท่าเดิม) และ nc = 3 (ผลไม้ 3 คลาส)
   ใช้ intersect_dicts แบบเดียวกับ Model.load() ซึ่งพิมพ์ "Transferred X/Y items"
2) เทรน coco8-seg 30 epoch สองแบบ: เริ่มจาก yolo11n-seg.pt กับ yolo11n-seg.yaml (สุ่ม) แล้วเก็บ results.csv

รันในโฟลเดอร์ชั่วคราว (ultralytics จะดาวน์โหลด .pt และสร้างโฟลเดอร์ผลเทรนที่นั่น):
    cd <temp> && python D:/course-plan-YOLO/tools/ch4_transfer.py
"""
import csv
import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import intersect_dicts

OUT = Path(__file__).resolve().parent.parent / "data" / "ch4_transfer.json"
EPOCHS = 30
COLS = ["train/seg_loss", "val/seg_loss", "metrics/mAP50(B)", "metrics/mAP50-95(M)"]

csd = YOLO("yolo11n-seg.pt").model.float().state_dict()
transfer = {}
for nc in (80, 3):
    m = SegmentationModel("yolo11n-seg.yaml", nc=nc, verbose=False)
    sd, pnames = m.state_dict(), dict(m.named_parameters())
    upd = intersect_dicts(csd, sd)
    transfer[nc] = {
        "items": len(upd),
        "items_total": len(sd),
        "params": sum(v.numel() for k, v in upd.items() if k in pnames),
        "params_total": sum(p.numel() for p in m.parameters()),
        "not_transferred_modules": sorted({".".join(k.split(".")[:4]) for k in sd if k not in upd}),
    }

runs = {}
for name, src in (("pt", "yolo11n-seg.pt"), ("yaml", "yolo11n-seg.yaml")):
    YOLO(src).train(data="coco8-seg.yaml", epochs=EPOCHS, imgsz=640, batch=16, device="cpu", workers=0,
                    project=".", name=name, exist_ok=True, plots=False, verbose=False, seed=0)
    rows = list(csv.DictReader(open(Path(name) / "results.csv")))
    runs[name] = {c: [round(float(r[c]), 5) for r in rows] for c in ["epoch", *COLS]}

OUT.write_text(json.dumps({"ultralytics": ultralytics.__version__, "epochs": EPOCHS,
                           "transfer": transfer, "runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
