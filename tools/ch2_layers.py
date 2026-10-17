"""ตัวเลขของหัวข้อ 2.3 (ต่อนิวรอนเป็นชั้น) และ 2.4 (นิวรอนจริงของ YOLO11)

2.3  เครือข่าย 2-2-1 ที่ "หนีบ" ส้มไว้ด้วยสองเส้นแนวนอนบนแกน G และกรณีตัด activation ออก
     ตัวอย่างการยุบรวมเป็นเส้นตรงของซิม 2.3 และเนินจาก ReLU 3 ตัว
2.4  นิวรอนหนึ่งตัวของชั้นแรก yolo11n-seg (kernel 3x3x3 = 27 อินพุต) คำนวณด้วยมือ เทียบกับ PyTorch
     แบบแยก BN และแบบรวม BN เข้า weight (fuse) และนับจำนวนครั้งที่ SiLU ถูกคำนวณต่อภาพ 640x640

รันในโฟลเดอร์ชั่วคราว (ultralytics จะดาวน์โหลด yolo11n-seg.pt ที่นั่น):
    cd <temp> && python D:/course-plan-YOLO/tools/ch2_layers.py
"""
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv

ROOT = Path(__file__).resolve().parent.parent
sig = lambda z: 1 / (1 + np.exp(-z))

# ---------------- 2.3 เครือข่ายหนีบส้ม ----------------
W = np.array([[0, 20], [0, -20]], float)   # แถว = นิวรอน, คอลัมน์ = (R, G)
b = np.array([-8, 15], float)
v = np.array([10, 10], float)
c = -15.0
print("== 2.3 เครือข่าย 2-2-1 หนีบส้ม ==")
for name, x in [("ส้ม", (0.941, 0.545)), ("แอปเปิลแดง", (0.863, 0.184)), ("กล้วย", (0.953, 0.812)),
                ("ส้ม G ต่ำสุด", (0.941, 0.478)), ("ส้ม G สูงสุด", (0.941, 0.678)), ("แดง G สูงสุด", (0.863, 0.353))]:
    x = np.array(x)
    z = W @ x + b
    h = sig(z)
    zo = v @ h + c
    print(f"{name:12s} x={x} z={np.round(z, 3)} h={np.round(h, 4)} z_out={zo:.3f} y={sig(zo):.4f}")
A = v @ W
B = v @ b + c
print(f"ไม่มี activation: y = sigma({A} . x + {B}) = sigma({B}) = {sig(B):.6f} ทุกพิกเซล")

print("\n== 2.3 ซิม 2.3 ค่าเริ่ม (ไม่มี activation) ==")
n = [(1, 0.5, 1), (-0.8, 0.4, 0.7), (0.6, -1, -1.2)]
A1 = sum(vv * w for w, bb, vv in n)
B1 = sum(vv * bb for w, bb, vv in n)
xx = 1.0
hs = [w * xx + bb for w, bb, vv in n]
print(f"A = {A1:.4f}, B = {B1:.4f}; ที่ x = 1: h = {hs}, y = {sum(vv * h for (w, bb, vv), h in zip(n, hs)):.4f}, A*1+B = {A1 + B1:.4f}")

print("\n== 2.3 เนินจาก ReLU 3 ตัว: y = ReLU(x+1) - 2 ReLU(x) + ReLU(x-1) ==")
relu = lambda t: max(0.0, t)
for xx in (-2, -1, -0.5, 0, 0.5, 1, 2):
    h = (relu(xx + 1), relu(xx), relu(xx - 1))
    print(f"x={xx:5}: h={h} y={h[0] - 2 * h[1] + h[2]}")

print("\nparams 2-2-1 =", (2 * 2 + 2) + (2 * 1 + 1), " 2-3-1 =", (2 * 3 + 3) + (3 * 1 + 1))

# ---------------- 2.4 นิวรอนจริงของ YOLO11 ----------------
print(f"\n== 2.4 ultralytics {ultralytics.__version__} ==")
model = YOLO("yolo11n-seg.pt").model.float().eval()
conv0 = model.model[0]
img = np.asarray(Image.open(ROOT / "data" / "fruit_letterbox_640.png").convert("RGB"), dtype=np.float32) / 255
x = torch.from_numpy(img).permute(2, 0, 1)[None]
with torch.no_grad():
    out = conv0(x)[0].numpy()  # (16, 320, 320) หลัง conv → bn → SiLU

# จุดกลางของแอปเปิลแดงและแอปเปิลเขียวในภาพ (หาจากสี)
R, G, Bc = (img[..., k] * 255 for k in range(3))
for name, mask in [("แอปเปิลแดง", (R > 190) & (G < 60) & (Bc < 70)), ("แอปเปิลเขียว", (G > 170) & (R < 140) & (Bc < 80))]:
    ys, xs = np.nonzero(mask)
    print(f"{name}: จำนวนพิกเซล {len(ys)}, median (row, col) = ({int(np.median(ys))}, {int(np.median(xs))})")

w = conv0.conv.weight.detach().numpy()  # (16, 3, 3, 3)
bn = conv0.bn
g_, be, mu, var, eps = (bn.weight.detach().numpy(), bn.bias.detach().numpy(), bn.running_mean.numpy(),
                        bn.running_var.numpy(), bn.eps)
scale = g_ / np.sqrt(var + eps)
wf = w * scale[:, None, None, None]
bf = be - mu * scale
xp = np.pad(img.transpose(2, 0, 1), ((0, 0), (1, 1), (1, 1)))  # padding 1


def neuron(ch, i, j):
    patch = xp[:, 2 * i:2 * i + 3, 2 * j:2 * j + 3]  # stride 2: เอาต์พุต (i, j) ใช้อินพุตแถว 2i-1..2i+1 (หลัง pad คือ 2i..2i+2)
    z = float((w[ch] * patch).sum())
    per = [(w[ch, k] * patch[k]).sum() for k in range(3)]
    zb = (z - mu[ch]) * scale[ch] + be[ch]
    zf = float((wf[ch] * patch).sum() + bf[ch])
    y = zb * sig(zb)
    return patch, z, per, zb, zf, y


pts = {}
for name, mask in [("แอปเปิลแดง", (R > 190) & (G < 60) & (Bc < 70)), ("แอปเปิลเขียว", (G > 170) & (R < 140) & (Bc < 80))]:
    ys, xs = np.nonzero(mask)
    pts[name] = (int(np.median(ys)) // 2, int(np.median(xs)) // 2)
for ch in (15, 12):
    print(f"\n-- ช่อง {ch}: gamma={g_[ch]:.4f} beta={be[ch]:.4f} mean={mu[ch]:.4f} var={var[ch]:.6f} eps={eps}"
          f" scale={scale[ch]:.4f} b'={bf[ch]:.4f}")
    print("weight R\n", np.round(w[ch, 0], 4), "\nweight G\n", np.round(w[ch, 1], 4), "\nweight B\n", np.round(w[ch, 2], 4))
    for name, (i, j) in pts.items():
        patch, z, per, zb, zf, y = neuron(ch, i, j)
        print(f"{name} out(i,j)=({i},{j}) patch R\n{np.round(patch[0], 3)}\n G\n{np.round(patch[1], 3)}\n B\n{np.round(patch[2], 3)}")
        print(f"  z = sum(w*x) = {z:.4f} (R {per[0]:.4f}, G {per[1]:.4f}, B {per[2]:.4f})"
              f"  BN -> {zb:.4f}  fused w'.x+b' = {zf:.4f}  SiLU -> {y:.4f}  PyTorch = {out[ch, i, j]:.4f}")
print("\nmax |manual - pytorch| ทั้งแผนที่ช่อง 15:", end=" ")
zz = np.zeros((320, 320))
for k in range(3):
    for di in range(3):
        for dj in range(3):
            zz += wf[15, k, di, dj] * xp[k, di:di + 640:2, dj:dj + 640:2]
zz += bf[15]
print(float(np.abs(zz * sig(zz) - out[15]).max()))

# ---------------- จำนวน SiLU ต่อภาพ ----------------
cnt = {"silu_blocks": 0, "silu_values": 0, "conv_blocks": 0}


def hook(m, i, o):
    cnt["conv_blocks"] += 1
    if isinstance(m.act, torch.nn.SiLU):
        cnt["silu_blocks"] += 1
        cnt["silu_values"] += o.numel()


hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, Conv)]
with torch.no_grad():
    model(torch.zeros(1, 3, 640, 640))
for h in hs:
    h.remove()
print("\nต่อภาพ 640x640:", cnt, " เฉพาะชั้น 0:", 16 * 320 * 320, " คูณ-บวกชั้น 0:", 16 * 320 * 320 * 27)
