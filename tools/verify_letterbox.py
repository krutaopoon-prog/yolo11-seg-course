"""ตรวจตัวเลข letterbox จาก ultralytics จริง เพื่อเทียบกับซิมมูเลชั่น 1.2 ในบทที่ 1

รัน:  python tools/verify_letterbox.py
"""

import numpy as np
import ultralytics
from ultralytics.data.augment import LetterBox

SIZES = [(1280, 720), (640, 480), (1080, 1920), (500, 500), (1000, 333)]  # (W, H)


def pad_of(out, img_w, img_h):
    """หา pad บน/ล่าง/ซ้าย/ขวา จากภาพผลลัพธ์ (ขอบที่เติมมีค่า 114 ภาพต้นฉบับเป็น 0 ทั้งหมด)"""
    content = np.argwhere(out[..., 0] != 114)
    (y0, x0), (y1, x1) = content.min(0), content.max(0)
    return int(y0), int(out.shape[0] - 1 - y1), int(x0), int(out.shape[1] - 1 - x1)


print(f"ultralytics {ultralytics.__version__}\n")
for auto in (False, True):
    title = "auto=True  (model.predict กับไฟล์ .pt: สี่เหลี่ยมเล็กสุดที่เป็นพหุคูณของ 32)" if auto else \
            "auto=False (สี่เหลี่ยมจัตุรัส 640x640)"
    print(title)
    for w, h in SIZES:
        img = np.zeros((h, w, 3), dtype=np.uint8)  # ภาพดำล้วน เพื่อแยกขอบ 114 ออกได้ง่าย
        out = LetterBox(new_shape=(640, 640), auto=auto, stride=32)(image=img)
        r = min(640 / h, 640 / w)
        t, b, l, rt = pad_of(out, w, h)
        print(f"  {w:>4}x{h:<4} -> r={r:.4f}  ย่อเป็น {round(w * r)}x{round(h * r)}"
              f"  ผลลัพธ์ {out.shape[1]}x{out.shape[0]}  pad บน/ล่าง/ซ้าย/ขวา = {t}/{b}/{l}/{rt}")
    print()

print("ค่าสีขอบ (padding_value) =", LetterBox().padding_value)
