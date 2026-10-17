"""ตัวเลขของหัวข้อ 2.2 (ฟังก์ชันกระตุ้น 4 แบบ) คำนวณด้วย PyTorch จริง

1) f(z) และความชัน f'(z) (จาก autograd) ของ Sigmoid, ReLU, SiLU ที่ค่า z ที่ใช้ในบท
2) จุดต่ำสุดของ SiLU, ความชันสูงสุดของ sigmoid, 0.25^n
3) ความชันของ ReLU ที่ z = 0 ตามที่ PyTorch ใช้จริง
4) ยืนยันว่า Conv.default_act ของ ultralytics ที่ติดตั้งคือ nn.SiLU

    python tools/ch2_activations.py
"""
import torch
import ultralytics
from ultralytics.nn.modules.conv import Conv

torch.set_printoptions(precision=6)
ZS = [-4, -2.79, -2, -1.28, -1, -0.5, 0, 0.5, 1, 1.4, 2, 4]
ACTS = {"sigmoid": torch.sigmoid, "relu": torch.relu, "silu": torch.nn.SiLU()}


def value_and_grad(fn, zs):
    z = torch.tensor(zs, dtype=torch.float64, requires_grad=True)
    y = fn(z)
    y.sum().backward()
    return y.detach().tolist(), z.grad.tolist()


print(f"torch {torch.__version__}, ultralytics {ultralytics.__version__}")
print(f"Conv.default_act = {Conv.default_act!r}")
print()
print(f"{'z':>7} | " + " | ".join(f"{k:>8}(z) {k:>6}'" for k in ACTS))
res = {k: value_and_grad(f, ZS) for k, f in ACTS.items()}
for i, z in enumerate(ZS):
    print(f"{z:7.2f} | " + " | ".join(f"{res[k][0][i]:11.4f} {res[k][1][i]:7.4f}" for k in ACTS))

# ความชันแบบประมาณ [f(z+h) - f(z)] / h ของ sigmoid
print("\nsigmoid secant [f(z+h)-f(z)]/h")
for z in (0.0, 4.0):
    for h in (1.0, 0.1, 0.01):
        s = (torch.sigmoid(torch.tensor(z + h, dtype=torch.float64)) - torch.sigmoid(torch.tensor(z, dtype=torch.float64))) / h
        print(f"  z={z:4.1f} h={h:5.2f}: {s.item():.5f}  f(z)={torch.sigmoid(torch.tensor(z)).item():.5f} f(z+h)={torch.sigmoid(torch.tensor(z + h)).item():.5f}")

# จุดต่ำสุดของ SiLU
zz = torch.linspace(-4, 0, 400001, dtype=torch.float64)
yy = torch.nn.functional.silu(zz)
i = int(torch.argmin(yy))
print(f"\nSiLU min: z = {zz[i].item():.5f}, SiLU = {yy[i].item():.5f}")

# ความชันสูงสุดของ sigmoid
zz = torch.linspace(-6, 6, 120001, dtype=torch.float64, requires_grad=True)
torch.sigmoid(zz).sum().backward()
j = int(torch.argmax(zz.grad))
print(f"sigmoid' max: {zz.grad[j].item():.6f} at z = {zz[j].item():.4f}")
# ความชันสูงสุดของ SiLU (ช่วง -6..6)
zz2 = zz.detach().clone().requires_grad_(True)
torch.nn.functional.silu(zz2).sum().backward()
k = int(torch.argmax(zz2.grad))
print(f"SiLU' max: {zz2.grad[k].item():.6f} at z = {zz2[k].item():.4f}, SiLU'(0) = 0.5")

print("\n0.25^n:", ", ".join(f"n={n}: {0.25 ** n:.3g}" for n in range(1, 11)))

z0 = torch.tensor([0.0], requires_grad=True)
torch.relu(z0).backward()
print(f"ReLU'(0) ใน PyTorch = {z0.grad.item()}")

# ความชันต่ำสุด (ติดลบมากสุด) ของ SiLU และ SiLU'(z) + SiLU'(-z) = 1
m = int(torch.argmin(zz2.grad))
print(f"SiLU' min: {zz2.grad[m].item():.6f} at z = {zz2[m].item():.4f}")
z = torch.tensor([4.0, -4.0, 1.4, -1.4], dtype=torch.float64, requires_grad=True)
torch.nn.functional.silu(z).sum().backward()
print(f"SiLU'(4) + SiLU'(-4) = {z.grad[0].item() + z.grad[1].item():.6f}, SiLU'(1.4) + SiLU'(-1.4) = {z.grad[2].item() + z.grad[3].item():.6f}")

# Step ไม่บอกความมั่นใจ: z = 0.01 กับ z = 3 / ขยับ w1 จาก 4 เป็น 4.1 กับพิกเซลแอปเปิลแดง (R = 0.85, G = 0.20, w2 = -5, b = -1)
for zv in (0.01, 3.0):
    print(f"z = {zv}: step = {int(zv >= 0)}, sigmoid = {torch.sigmoid(torch.tensor(zv, dtype=torch.float64)).item():.4f}")
for w1 in (4.0, 4.1):
    zv = w1 * 0.85 - 5 * 0.20 - 1
    print(f"w1 = {w1}: z = {zv:.3f}, step = {int(zv >= 0)}, sigmoid = {torch.sigmoid(torch.tensor(zv, dtype=torch.float64)).item():.4f}")

# ความสมมาตร sigmoid(-z) = 1 - sigmoid(z)
z = torch.tensor([1.4, -2.79], dtype=torch.float64)
print(f"sigmoid(-z) = {torch.sigmoid(-z).tolist()}, 1 - sigmoid(z) = {(1 - torch.sigmoid(z)).tolist()}")
