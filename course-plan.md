# แผนรายวิชา: เข้าใจ YOLO11-seg ตั้งแต่ศูนย์
### (Understanding YOLO11 Instance Segmentation from Scratch)

> ไฟล์นี้เป็น **แผนเนื้อหา** ของรายวิชาใหม่ในเว็บ WBI
> ใช้คู่กับ `SKILL.md` ซึ่งเป็นตัวกำหนด **โครงสร้างไฟล์ การตั้งชื่อ และเทมเพลตหน้าเว็บ**

---

## 0. Instructions for the AI coding agent (Claude Code)

Read this section before building any lesson.

1. **Read `SKILL.md` first, every session.** It defines folder structure, naming conventions, page templates, quiz and simulation formats. If this plan and `SKILL.md` disagree about *structure*, follow `SKILL.md`. This plan is authoritative only about *content*.
2. **Build one lesson per request** (e.g., "สร้างบทที่ 3"). Do not build ahead. After finishing a lesson, stop and summarize what was built so the learner can study and review it before the next lesson.
3. **Language:** lesson text in Thai. On first use of a technical term, give the English term in parentheses. Keep code, variable names, and formulas in English/math notation.
4. **Tech constraints:** pure HTML/CSS/JavaScript, no build step, no server, no backend. Simulations use Canvas or SVG with vanilla JS. Everything must run on GitHub Pages and on a phone-width screen.
5. **Accuracy rules (important):**
   - Any number describing YOLO11 internals (channel counts, tensor shapes, number of parameters, default hyperparameters such as `conf`, `iou`, loss gains `box`/`cls`/`dfl`) must be **obtained by running code** against the installed `ultralytics` package, not written from memory. Use the extraction script described in Lesson 9 and store results as JSON in the lesson's data folder.
   - Record the `ultralytics` version used (`pip show ultralytics`) in the footer of every lesson that shows such numbers.
   - When explaining a module (C3k2, C2PSA, SPPF, Segment, Proto, DFL), read its source in `ultralytics/nn/modules/` and explain what the code actually does.
6. **References:** cite in APA 7 using the keys in Section 4 (e.g. `[Redmon2016]`). Each lesson ends with a "เอกสารอ้างอิง" list containing full APA entries for the keys it uses. If a lesson needs a source not in Section 4, add it but mark it `⚠ ต้องตรวจสอบ` so the learner verifies it.
7. **Examples are generic.** Use the course's running example (Section 1) and public datasets only. Do not reference any personal project of the learner.
8. **Every simulation must have:** a one-line goal ("ลองทำ…แล้วสังเกต…"), a reset button, sensible defaults, and a short "สิ่งที่ควรสังเกต" box underneath.

---

## 1. ภาพรวมรายวิชา

**ผู้เรียน:** ผู้เริ่มต้นที่ต้องการเข้าใจการทำงานของ YOLO11-seg อย่างแท้จริง ตั้งแต่ตัวเลขในภาพจนถึง mask ของวัตถุแต่ละชิ้น ไม่ใช่แค่รันคำสั่งได้

**ผลลัพธ์การเรียนรู้ของรายวิชา:** เมื่อเรียนจบ ผู้เรียนสามารถ
1. อธิบายว่าภาพถูกแปลงเป็นตัวเลขและผ่านเครือข่ายประสาทเทียมอย่างไร
2. อธิบายกลไกการเรียนรู้ (loss, gradient descent, backpropagation) และปัญหา overfitting
3. อธิบายการทำงานของ convolution และ CNN พร้อมคำนวณขนาด feature map ได้
4. อธิบาย IoU, NMS, Precision, Recall, mAP และผลของการตั้งค่า threshold
5. อธิบายสถาปัตยกรรม YOLO11-seg (Backbone, Neck, Head, Proto) และ shape ของ tensor ทุกช่วง
6. อ่านและตีความผลการเทรน YOLO11-seg ได้ และอธิบายแต่ละขั้นด้วยคำพูดของตนเอง

**โครงสร้าง 4 หน่วย 12 บท**

| หน่วย | บท | ชื่อบท |
|---|---|---|
| 1. พื้นฐานการเรียนรู้ของเครื่อง | 1 | ภาพดิจิทัลคือตัวเลข |
| | 2 | นิวรอนและฟังก์ชันกระตุ้น |
| | 3 | Loss และ Gradient Descent |
| | 4 | การเทรนเครือข่าย: Backpropagation และ Overfitting |
| 2. CNN | 5 | Convolution |
| | 6 | จาก Convolution สู่ CNN ทั้งเครือข่าย |
| 3. Object Detection | 7 | Bounding Box, IoU และ NMS |
| | 8 | การวัดผล: Precision, Recall, mAP |
| 4. YOLO11-seg | 9 | สถาปัตยกรรม YOLO11: Backbone, Neck, Head |
| | 10 | Segmentation Head: Prototype × Coefficient |
| | 11 | การเทรน: Label Assignment, Loss และการอ่านผล |
| | 12 | ไล่ภาพหนึ่งภาพผ่านทั้งระบบ และทบทวนรวบยอด |

**ตัวอย่างประจำรายวิชา (running example):** ภาพ "ผลไม้ในตะกร้า" ที่วาดด้วย canvas มีผลไม้หลายลูกวางซ้อนกันบางส่วน ใช้ซ้ำในทุกบทเพื่อให้ผู้เรียนเห็นภาพเดียวกันผ่านทุกขั้นตอน และเห็นปัญหาวัตถุซ้อนกันที่ instance segmentation ต้องแก้

**เส้นเรื่องที่ร้อยทุกบทเข้าด้วยกัน:** ทุกบทต้องมีกล่อง "🔗 อยู่ตรงไหนใน YOLO11-seg" ที่ชี้ตำแหน่งของเนื้อหาบทนั้นบนผังนี้ (ไฮไลต์ขั้นที่เกี่ยวข้อง)

```
ภาพ → Letterbox 640×640 → Backbone → Neck → Head + Proto
    → ถอดรหัส (decode) → กรอง conf → NMS → Mask ของวัตถุแต่ละชิ้น
```

**ชุดข้อมูลสำหรับฝึกปฏิบัติ:** `coco8-seg` ของ Ultralytics (ชุดข้อมูลตัวอย่างขนาดเล็ก) เทรนได้ในไม่กี่นาทีแม้บน CPU ใช้ในบทที่ 9–12 เพื่อให้ผู้เรียนมีไฟล์ผลเทรนจริงไว้ศึกษา

---

## 2. โครงสร้างมาตรฐานของแต่ละบท

ใช้เทมเพลตหน้าบทเรียนตาม `SKILL.md` โดยมีส่วนต่อไปนี้ตามลำดับ

1. **วัตถุประสงค์การเรียนรู้** (3–5 ข้อ เขียนแบบวัดได้ เช่น "คำนวณ…ได้", "อธิบาย…ได้")
2. **ทบทวนความรู้เดิม** (1 ย่อหน้า เชื่อมจากบทก่อน)
3. **เนื้อหา** แบ่งเป็นหัวข้อย่อย มีแผนภาพ SVG ประกอบ
4. **ซิมมูเลชั่น** (ตามสเปกในแต่ละบท)
5. **🔗 อยู่ตรงไหนใน YOLO11-seg**
6. **สรุปท้ายบท** (5–7 bullet)
7. **แบบทดสอบท้ายบท** (5–8 ข้อ ปรนัย พร้อมเฉลยและคำอธิบาย)
8. **เอกสารอ้างอิง** (APA 7)

---

## 3. รายละเอียดรายบท

### บทที่ 1 ภาพดิจิทัลคือตัวเลข

**วัตถุประสงค์**
- อธิบายว่าภาพสีคือ tensor 3 มิติ ขนาด (C, H, W)
- อธิบายมิติ batch (N, C, H, W)
- อธิบายการ normalize ค่า 0–255 เป็น 0–1
- อธิบายการ resize แบบ letterbox ให้เป็น 640×640 โดยไม่บิดสัดส่วนภาพ

**เนื้อหา**
- พิกเซล ช่องสี RGB ความลึกบิต (8 bit = 0–255)
- Tensor และ shape; ความต่างระหว่าง HWC (ภาพทั่วไป/OpenCV) กับ CHW (PyTorch)
- Normalization
- Letterbox: ย่อภาพตามสเกลแล้วเติมขอบให้เป็นสี่เหลี่ยมจัตุรัส (ตรวจค่าสีขอบและพฤติกรรมจริงจาก source code ของ Ultralytics)

**ซิมมูเลชั่น**
- **1.1 ส่องพิกเซล:** แสดงภาพตัวอย่างประจำรายวิชา เลื่อนเมาส์/แตะที่ภาพแล้วแสดงค่า R, G, B ณ จุดนั้น พร้อมกรอบซูมตาราง 8×8 ที่แสดงตัวเลขจริง มีปุ่มสลับดูทีละช่องสี R / G / B เป็นภาพขาวดำ
- **1.2 Letterbox:** slider ปรับขนาดภาพต้นฉบับ (เช่น 1280×720, 640×480, 1080×1920) แสดงภาพที่ถูกย่อและขอบที่เติม พร้อมตัวเลข scale และ padding บน/ล่าง/ซ้าย/ขวา

**🔗 อยู่ตรงไหน:** ขั้นแรกสุดก่อนเข้าโมเดล และพิกัดผลลัพธ์ต้อง "ย้อน letterbox" กลับไปยังภาพจริงในขั้นสุดท้าย

**แบบทดสอบ (ประเด็น):** shape ของภาพ RGB 640×640, ค่าหลัง normalize, ทำไมไม่ยืดภาพตรง ๆ, จำนวนตัวเลขในภาพหนึ่งภาพ

**อ้างอิง:** [Goodfellow2016], [Zhang2023], [UltralyticsDocs]

---

### บทที่ 2 นิวรอนและฟังก์ชันกระตุ้น

**วัตถุประสงค์**
- อธิบายการคำนวณของนิวรอน y = f(w·x + b)
- เปรียบเทียบ activation function: Step, Sigmoid, ReLU, SiLU
- อธิบายว่าเหตุใดต้องมี activation ที่ไม่เป็นเชิงเส้น

**เนื้อหา**
- Weight, bias, weighted sum
- Activation function และกราฟ; SiLU(x) = x·σ(x) ซึ่งเป็น activation ที่ YOLO11 ใช้ (ยืนยันจาก source code)
- การต่อนิวรอนเป็นชั้น (layer) และเครือข่าย
- ถ้าไม่มี activation เครือข่ายหลายชั้นจะยุบรวมเหลือฟังก์ชันเชิงเส้นชั้นเดียว

**ซิมมูเลชั่น**
- **2.1 นิวรอนตัวเดียว:** 2 อินพุต slider ปรับ w1, w2, b แสดงเส้นแบ่งบนระนาบ 2 มิติที่มีจุดสองกลุ่ม (สีต่างกัน) และนับจำนวนจุดที่แบ่งถูก
- **2.2 สนามกราฟ activation:** เลือก activation แล้วเห็นกราฟ และลากจุด x เพื่อดูค่า f(x)

**🔗 อยู่ตรงไหน:** บล็อก `Conv` ทุกตัวใน YOLO11 คือ Conv2d → BatchNorm → SiLU

**อ้างอิง:** [Goodfellow2016], [Elfwing2018], [Sanderson2017]

---

### บทที่ 3 Loss และ Gradient Descent

**วัตถุประสงค์**
- อธิบายความหมายของ loss function
- อธิบาย gradient ในฐานะ "ทิศที่ loss เพิ่มเร็วที่สุด"
- อธิบายผลของ learning rate ที่เล็กไป เหมาะสม และใหญ่ไป

**เนื้อหา**
- Loss ตัวอย่าง: MSE (สำหรับปัญหา regression) และ BCE (สำหรับปัญหาใช่/ไม่ใช่)
- อนุพันธ์ในเชิงแนวคิด (ความชัน) โดยไม่ต้องคำนวณซับซ้อน
- กฎการปรับค่า: w ← w − η · ∂L/∂w
- SGD, Momentum, Adam/AdamW แบบแนวคิด

**ซิมมูเลชั่น**
- **3.1 ฟิตเส้นตรง:** จุดข้อมูลกระจาย ผู้เรียนปรับ slope/intercept เองแล้วเห็นค่า MSE จากนั้นกดปุ่ม "Step" หรือ "Auto" ให้ gradient descent ปรับให้ พร้อมกราฟ loss ตามรอบ
- **3.2 ลูกบอลกลิ้งลงหุบ:** กราฟ loss 1 มิติ (มีหุบหลายหุบ) slider learning rate แล้วดูลูกบอลลู่เข้า แกว่ง หรือกระเด็นออก

**🔗 อยู่ตรงไหน:** พารามิเตอร์ `lr0`, `optimizer` ตอนสั่งเทรน และค่า loss ที่ Ultralytics บันทึกทุก epoch

**อ้างอิง:** [Rumelhart1986], [Goodfellow2016], [Kingma2015], [Loshchilov2019], [Sanderson2017]

---

### บทที่ 4 การเทรนเครือข่าย: Backpropagation และ Overfitting

**วัตถุประสงค์**
- อธิบาย backpropagation ด้วยกฎลูกโซ่ (chain rule) ในเชิงแนวคิด
- อธิบาย epoch, batch, iteration และคำนวณจำนวน iteration ต่อ epoch ได้
- อธิบาย train/val/test split, overfitting, data augmentation และ transfer learning
- อธิบายความหมายและสาเหตุของ data leakage

**เนื้อหา**
- Forward pass และ backward pass
- Epoch / batch size / iteration
- Train / Validation / Test
- Overfitting vs underfitting และสัญญาณบนกราฟ
- Data augmentation
- **Transfer learning:** ความต่างระหว่างเริ่มจาก `yolo11n-seg.pt` (มีน้ำหนักที่เทรนจาก COCO แล้ว) กับ `yolo11n-seg.yaml` (เริ่มจากสุ่ม)
- **Data leakage:** เมื่อข้อมูลชุดทดสอบ "รั่ว" ไปอยู่ในชุดเทรน เช่น ภาพที่เกือบซ้ำกัน (ภาพถ่ายต่อเนื่อง เฟรมวิดีโอที่ติดกัน) ถูกแยกไปอยู่คนละชุด ทำให้คะแนนสูงเกินจริง

**ซิมมูเลชั่น**
- **4.1 กราฟคำนวณ:** x → ×w → +b → loss แสดงตัวเลขไหลไปข้างหน้า กดปุ่มแล้วเห็น gradient ไหลย้อนกลับทีละโหนด
- **4.2 Overfitting:** จุดข้อมูลมีสัญญาณรบกวน slider ปรับ degree ของพหุนาม แสดงเส้นที่ฟิต และกราฟ train error กับ val error แยกกัน
- **4.3 Data leakage:** ลำดับภาพ 100 ภาพที่ภาพติดกันคล้ายกัน ให้เลือกวิธีแบ่ง "สุ่มทีละภาพ" กับ "แบ่งเป็นช่วง" แล้วแสดงเป็นสีว่าภาพ val แต่ละภาพมีภาพเกือบซ้ำอยู่ใน train หรือไม่
- **4.4 เครื่องคิดเลข iteration:** กรอกจำนวนภาพ, batch → iteration/epoch และจำนวน iteration ทั้งหมด

**🔗 อยู่ตรงไหน:** ขั้นเตรียมข้อมูลและการเทรนทั้งหมด (`model.train(...)`)

**อ้างอิง:** [Rumelhart1986], [Goodfellow2016], [Zhang2023]

---

### บทที่ 5 Convolution

**วัตถุประสงค์**
- อธิบายการเลื่อน kernel และการคำนวณผลรวมของผลคูณ
- คำนวณขนาด output ด้วยสูตร O = ⌊(I + 2P − K) / S⌋ + 1
- อธิบาย convolution หลายช่อง (multi-channel) และคำนวณจำนวนพารามิเตอร์ K×K×C_in×C_out (+ bias)

**เนื้อหา**
- Kernel/filter, stride, padding, feature map
- Kernel ตัวอย่าง: ตรวจขอบแนวตั้ง แนวนอน เบลอ เพิ่มความคม
- Multi-channel: kernel มีความลึกเท่าจำนวนช่องอินพุต; จำนวน kernel = จำนวนช่องเอาต์พุต
- ในโครงข่ายจริง kernel ไม่ได้ออกแบบด้วยมือ แต่ "เรียนรู้" ด้วย gradient descent (เชื่อมบทที่ 3–4)

**ซิมมูเลชั่น**
- **5.1 เลื่อน kernel ทีละช่อง:** ตาราง 7×7 kernel 3×3 กด Step แล้วเห็นการคูณทีละช่องและผลรวม ปรับ stride (1, 2) padding (0, 1) และเลือก kernel ตัวอย่าง สูตรขนาด output คำนวณสดให้ดู
- **5.2 กรองภาพจริง:** ใช้ kernel กับภาพตัวอย่างประจำรายวิชาแล้วเห็นภาพขอบ
- **5.3 นับพารามิเตอร์:** กรอก K, C_in, C_out แล้วได้จำนวนพารามิเตอร์

**🔗 อยู่ตรงไหน:** ชั้นแรกของ YOLO11 เป็น Conv kernel 3 stride 2 ทำให้ 640×640 เหลือ 320×320 (ยืนยันค่าจาก yaml)

**อ้างอิง:** [Dumoulin2016], [LeCun1998], [CS231n]

---

### บทที่ 6 จาก Convolution สู่ CNN ทั้งเครือข่าย

**วัตถุประสงค์**
- อธิบายการซ้อนชั้นที่ทำให้ภาพเล็กลงแต่จำนวนช่องเพิ่มขึ้น
- อธิบายระดับของ feature (ขอบ → รูปทรง → วัตถุ)
- อธิบาย receptive field, max pooling, Batch Normalization และ residual connection
- อธิบายแนวคิด CSP (แบ่งช่องเป็นสองทางแล้วรวม) ที่เป็นรากของบล็อก C3k2

**เนื้อหา**
- ลำดับชั้นของ feature และสัญลักษณ์ P1–P5 (stride 2, 4, 8, 16, 32)
- Receptive field
- Max pooling
- BatchNorm
- Residual/skip connection และ bottleneck
- CSP แบบแนวคิด

**ซิมมูเลชั่น**
- **6.1 พีระมิด feature:** แสดงขนาด 640 → 320 → 160 → 80 → 40 → 20 พร้อมจำนวนช่อง (ใช้ค่าจริงของ yolo11n-seg จาก JSON ในบทที่ 9) และป้าย P1–P5
- **6.2 Receptive field:** คลิกเซลล์ในชั้นลึก แล้วไฮไลต์พื้นที่บนภาพอินพุตที่เซลล์นั้น "มองเห็น"
- **6.3 Max pooling:** ตาราง 4×4 → 2×2 แบบทีละขั้น
- **6.4 Residual:** แผนภาพที่ปิด/เปิดเส้นทางลัดแล้วเห็นว่าข้อมูลไหลอย่างไร

**🔗 อยู่ตรงไหน:** ส่วน Backbone; วัตถุขนาดเล็กตรวจได้ดีที่ระดับ P3 วัตถุใหญ่ใช้ P4/P5

**อ้างอิง:** [He2016], [Ioffe2015], [Wang2020], [CS231n]

---

### บทที่ 7 Bounding Box, IoU และ NMS

**วัตถุประสงค์**
- แยกงาน classification, object detection, semantic segmentation และ instance segmentation
- คำนวณ IoU ได้
- อธิบายขั้นตอน NMS และผลของ IoU threshold ต่อวัตถุที่ซ้อนกัน
- อธิบายความต่างของ one-stage/two-stage และ anchor-based/anchor-free

**เนื้อหา**
- เปรียบเทียบงาน 4 แบบบนภาพเดียวกัน
- รูปแบบกล่อง xyxy, xywh, แบบ normalized
- Confidence score
- IoU
- NMS: เรียงคะแนน → เก็บกล่องสูงสุด → ลบกล่องที่ IoU เกินเกณฑ์ → ทำซ้ำ
- ประวัติ YOLO โดยย่อ: v1 (แนวคิด grid, one-stage) → รุ่นหลายสเกล → รุ่น anchor-free ของ Ultralytics → YOLO11 (กล่าวถึงว่ามีรุ่นใหม่กว่าออกมาแล้วโดยย่อ)

**ซิมมูเลชั่น**
- **7.1 IoU:** ลากกล่อง 2 กล่อง แสดงพื้นที่ intersection และ union เป็นสี และค่า IoU สด
- **7.2 NMS กับวัตถุซ้อนกัน (บทเด่น):** ภาพผลไม้ในตะกร้า มีกล่องผู้สมัคร (candidate) หลายกล่องพร้อมคะแนน slider ปรับ `conf` และ `iou` threshold ปุ่มเดินทีละขั้นของ NMS แสดง "จำนวนวัตถุที่เหลือ" เทียบกับ "จำนวนวัตถุจริง" ให้เห็นว่า
  - iou threshold ต่ำเกินไป → วัตถุจริงที่ซ้อนกันถูกลบทิ้ง
  - iou threshold สูงเกินไป → กล่องซ้ำไม่ถูกลบ
- **7.3 งาน 4 แบบ:** สลับดูผลลัพธ์ของงานแต่ละแบบบนภาพเดียวกัน

**🔗 อยู่ตรงไหน:** ขั้น post-processing หลัง Head; ค่า `conf` และ `iou` ตอน predict

**อ้างอิง:** [Redmon2016], [Neubeck2006], [He2017], [UltralyticsDocs]

---

### บทที่ 8 การวัดผล: Precision, Recall, mAP

**วัตถุประสงค์**
- อธิบายการจับคู่ผลทำนายกับเฉลยด้วย IoU และนับ TP, FP, FN
- คำนวณ Precision, Recall, F1
- อธิบาย PR curve, AP, mAP50 และ mAP50-95
- แยกความหมายของเมตริก Box (B) กับ Mask (M)

**เนื้อหา**
- TP / FP / FN (detection ไม่มี TN ที่มีความหมาย)
- Precision, Recall, F1
- PR curve และพื้นที่ใต้กราฟ (AP)
- mAP50 (แนว PASCAL VOC) และ mAP50-95 (แนว COCO)
- Mask IoU vs Box IoU
- Confusion matrix
- เมตริกขึ้นกับเป้าหมายของงาน: บางงานต้องการ recall สูง บางงานต้องการ precision สูง

**ซิมมูเลชั่น**
- **8.1 ตาราง detection:** รายการผลทำนายเรียงตามคะแนน slider conf threshold แล้วตาราง TP/FP/FN, ค่า P, R อัปเดต และวาด PR curve พร้อมแรเงาพื้นที่ AP
- **8.2 IoU threshold หลายระดับ:** เปลี่ยน IoU เกณฑ์ 0.50 → 0.95 แล้วดู AP แต่ละระดับลดลง และค่าเฉลี่ยที่กลายเป็น mAP50-95
- **8.3 Box vs Mask IoU:** กล่องที่ทับกันดีแต่ mask ทับกันไม่ดี (และกลับกัน) เพื่อให้เห็นว่าทำไมเมตริก B กับ M ต่างกัน

**🔗 อยู่ตรงไหน:** ขั้น validation; คอลัมน์ `metrics/mAP50(B)`, `metrics/mAP50-95(M)` ฯลฯ ในไฟล์ `results.csv`

**อ้างอิง:** [Everingham2010], [Lin2014], [Padilla2020]

---

### บทที่ 9 สถาปัตยกรรม YOLO11: Backbone, Neck, Head

**วัตถุประสงค์**
- อ่านไฟล์ `yolo11-seg.yaml` ได้ (รูปแบบ `[from, repeats, module, args]`)
- อธิบายหน้าที่ของ Backbone, Neck, Head
- อธิบายบล็อก Conv, C3k2, SPPF, C2PSA, Upsample, Concat
- อธิบาย FPN (บนลงล่าง) + PAN (ล่างขึ้นบน)
- บอก shape ของ tensor ในแต่ละช่วงสำหรับอินพุต 640×640 และอธิบายที่มาของตำแหน่งทำนาย 80² + 40² + 20² = 8,400 จุด
- อธิบายตัวคูณสเกล (depth, width, max_channels) ของรุ่น n/s/m/l/x

**เนื้อหา**
- ภาพรวม 3 ส่วน
- การอ่าน yaml ทีละบรรทัด
- **อธิบายแต่ละโมดูลจาก source code จริง** ใน `ultralytics/nn/modules/block.py` และ `conv.py`
- SPPF: max pooling ต่อกันหลายครั้งเพื่อรวมบริบทหลายขนาด
- C2PSA: บล็อกที่มี attention เชิงตำแหน่ง (เชื่อมแนวคิด attention แบบย่อ)
- Neck: Upsample + Concat (FPN) แล้ว Conv stride 2 + Concat (PAN)
- Head 3 สเกล: P3/8, P4/16, P5/32

**งานเตรียมข้อมูล (สำหรับ agent):** เขียนสคริปต์ `tools/extract_shapes.py` (รันในเครื่องครั้งเดียว ไม่ได้รันบนเว็บ) ที่
1. สร้างโมเดลจาก `yolo11n-seg.yaml` (ใช้จำนวนคลาสตามค่าเริ่มต้นของไฟล์ทางการ)
2. ลงทะเบียน forward hook ทุกชั้น
3. ป้อน tensor หลอกขนาด (1, 3, 640, 640)
4. บันทึก JSON: ลำดับชั้น, ชื่อโมดูล, from, args, shape เอาต์พุต, จำนวนพารามิเตอร์ ของรุ่น n (และรุ่นอื่นถ้าต้องการ)
5. บันทึกเวอร์ชัน ultralytics ลงใน JSON

ซิมมูเลชั่นทุกตัวในบท 6, 9, 10 อ่านตัวเลขจาก JSON นี้ ห้ามพิมพ์ตัวเลขเอง

**ซิมมูเลชั่น**
- **9.1 ผังสถาปัตยกรรมคลิกได้:** คลิกบล็อกแล้วแสดงคำอธิบาย บรรทัด yaml ที่ตรงกัน shape อินพุต/เอาต์พุต และจำนวนพารามิเตอร์
- **9.2 เลือกขนาดรุ่น:** สลับ n/s/m/l/x แล้วเห็นจำนวนช่องและพารามิเตอร์เปลี่ยน
- **9.3 SPPF:** แสดงการ max pool ซ้ำหลายครั้งบนตารางเล็ก แล้วต่อ (concat) ผลทั้งหมด
- **9.4 การไหลของ FPN + PAN:** แอนิเมชันลูกศรข้อมูลไหลลงแล้วไหลขึ้น

**แบบฝึกปฏิบัติ:** โหลด `yolo11n-seg.pt` แล้วรัน `model.info()` เทียบกับผังบนเว็บ

**🔗 อยู่ตรงไหน:** Backbone + Neck + Head ทั้งหมด

**อ้างอิง:** [Jocher2024], [Khanam2024], [Lin2017], [Liu2018], [He2015], [Wang2020], [Vaswani2017]

---

### บทที่ 10 Segmentation Head: Prototype × Coefficient

**วัตถุประสงค์**
- อธิบายเอาต์พุตของ head สาขา box, class และ mask coefficient
- อธิบาย DFL: การทำนายระยะขอบกล่องเป็นการแจกแจงความน่าจะเป็นแล้วหาค่าคาดหมาย
- อธิบายการสร้าง mask: coefficient · prototypes → sigmoid → crop ด้วยกล่อง → ขยายขนาด → threshold
- ไล่ขั้นตอน post-processing ตั้งแต่เอาต์พุตดิบจนได้ mask ของวัตถุแต่ละชิ้น

**เนื้อหา**
- Anchor point (จุดศูนย์กลางเซลล์ของทั้ง 3 สเกล) และระยะ l, t, r, b × stride
- DFL และ `reg_max` (ยืนยันค่าจาก source)
- Proto module: สร้าง prototype masks จาก feature ความละเอียดสูง (ยืนยันจำนวน prototype และความละเอียดจาก JSON)
- Mask coefficient ของแต่ละ instance
- แนวคิดจาก YOLACT
- Shape เอาต์พุตสุดท้าย (จำนวนช่อง = 4 + จำนวนคลาส + จำนวน prototype) ดึงจาก JSON
- ลำดับ post-processing: decode → กรอง conf → NMS → สร้าง mask เฉพาะกล่องที่รอด

**ซิมมูเลชั่น**
- **10.1 ผสม mask (บทเด่น):** แสดง prototype สังเคราะห์ 6–8 แบบ (รูปไล่ระดับ, วงรี, ขอบ) slider ปรับ coefficient แต่ละตัว (−1 ถึง 1) แล้วเห็น mask รวมเปลี่ยนสด ๆ จากนั้นปุ่ม crop ด้วยกล่อง และ slider threshold
- **10.2 DFL:** histogram ตามจำนวนช่อง `reg_max` จริง ลากแท่งปรับความน่าจะเป็น แล้วเห็นค่าคาดหมายของระยะ และขอบกล่องเคลื่อนตาม
- **10.3 Anchor point 3 สเกล:** แสดงจุดบนภาพ 640×640 สลับดูทีละสเกล นับรวม 8,400 จุด
- **10.4 Pipeline ทีละขั้น:** ปุ่ม Next พาจากเอาต์พุตดิบ → ผ่าน conf → ผ่าน NMS → mask ของวัตถุแต่ละชิ้น
- *(ทางเลือก)* สคริปต์ `tools/export_protos.py` ดึง prototype จริงจาก `yolo11n-seg.pt` กับภาพตัวอย่างสาธารณะ แล้วบันทึกเป็น PNG ให้ซิมมูเลชั่น 10.1 มีโหมด "prototype จริง"

**🔗 อยู่ตรงไหน:** Head (สาขา mask coefficient) + Proto + ขั้นสร้าง mask; ผลไม้ที่ซ้อนกันได้ mask แยกกันเพราะแต่ละ instance มีชุด coefficient ของตัวเอง

**อ้างอิง:** [Bolya2019], [Li2020], [Jocher2024]

---

### บทที่ 11 การเทรน: Label Assignment, Loss และการอ่านผล

**วัตถุประสงค์**
- อธิบายรูปแบบไฟล์ label ของ YOLO-seg (class + พิกัด polygon แบบ normalized)
- อธิบายแนวคิด label assignment (Task-Aligned Assigner) ว่าจุดทำนายใดรับผิดชอบวัตถุใด
- อธิบาย loss ทั้ง 4 ตัว: box (CIoU), seg, cls (BCE), dfl และค่าน้ำหนักของแต่ละตัว
- อธิบาย augmentation (mosaic และการปิด mosaic ช่วงท้าย), warmup, learning rate schedule
- อ่านและตีความ `results.csv`, `results.png`, confusion matrix และ PR curve

**เนื้อหา**
- ไฟล์ label `.txt` และ `data.yaml` (ใช้ `coco8-seg` เป็นตัวอย่าง)
- Task-Aligned Assigner (แนวคิดจาก TOOD)
- CIoU = IoU − (ระยะศูนย์กลาง) − (ความต่างสัดส่วน)
- Seg loss บน mask
- ค่าน้ำหนัก loss และ hyperparameter อื่น **ดึงจาก `args.yaml` ของการเทรนตัวอย่างและค่า default ในเวอร์ชันที่ติดตั้ง**
- Mosaic, close_mosaic, warmup, lr schedule
- การอ่านกราฟ: train loss ลด แต่ val loss เริ่มขึ้น = overfitting (เชื่อมบทที่ 4)
- คุณภาพของเฉลย (ground truth) กำหนดเพดานของโมเดล

**ซิมมูเลชั่น**
- **11.1 เครื่องคิด CIoU:** ลากกล่องทำนายและกล่องเฉลย แสดงแต่ละพจน์ของ CIoU แยกกัน
- **11.2 Mosaic:** รวม 4 ภาพเป็นภาพเดียวพร้อมกล่อง/polygon ที่ปรับพิกัดตาม สุ่มใหม่ได้
- **11.3 ตัวอ่านผลเทรน:** ผู้เรียนเลือกไฟล์ `results.csv` จากเครื่อง (อ่านด้วย FileReader ในเบราว์เซอร์ ไม่อัปโหลดขึ้นที่ไหน) แล้ววาดกราฟ loss 4 ตัวและ mAP พร้อมคำอธิบายกำกับ มีไฟล์ตัวอย่างจากการเทรน `coco8-seg` ให้ลองถ้ายังไม่เลือกไฟล์

**แบบฝึกปฏิบัติ:** เทรน `yolo11n-seg.pt` บน `coco8-seg` เป็นเวลาสั้น ๆ แล้วเปิดโฟลเดอร์ผลลัพธ์ อธิบายทุกไฟล์ที่ได้

**🔗 อยู่ตรงไหน:** วงจรการเทรนทั้งหมด

**อ้างอิง:** [Feng2021], [Zheng2020], [Bochkovskiy2020], [Loshchilov2019], [Jocher2024], [UltralyticsDocs]

---

### บทที่ 12 ไล่ภาพหนึ่งภาพผ่านทั้งระบบ และทบทวนรวบยอด

**วัตถุประสงค์**
- ไล่ภาพหนึ่งภาพผ่าน YOLO11-seg ตั้งแต่พิกเซลจนเป็น mask และอธิบายทุกขั้นได้
- เชื่อมโยงความรู้จากบทที่ 1–11 เป็นภาพเดียว
- อธิบายการทำงานของ YOLO11-seg ด้วยคำพูดของตนเอง

**เนื้อหา**
- ทบทวนผังทั้งระบบพร้อมชี้กลับไปบทที่เกี่ยวข้องทุกขั้น
- แผนผังความคิด (concept map) ของรายวิชา
- ข้อจำกัดของ YOLO11-seg และแนวทางพัฒนาต่อ (เช่น วัตถุซ้อนกันมาก วัตถุเล็กมาก)

**กิจกรรม**
- **12.1 Pipeline stepper ทั้งระบบ:** ภาพตัวอย่างประจำรายวิชาเดินผ่านทุกขั้น (letterbox → tensor → feature P3/P4/P5 → เอาต์พุตดิบ → conf → NMS → mask) แต่ละขั้นแสดง shape และลิงก์กลับไปบทที่สอนเรื่องนั้น
- **12.2 เรียงขั้นตอน:** ลากกล่องขั้นตอนมาเรียงให้ถูกลำดับ
- **12.3 การ์ด "อธิบายด้วยคำพูดตัวเอง":** ด้านหน้าคำถาม ด้านหลังแนวคำตอบและลิงก์ไปบทที่เกี่ยวข้อง ตัวอย่าง
  1. Instance segmentation ต่างจาก object detection และ semantic segmentation อย่างไร
  2. ภาพ 640×640 กลายเป็นตำแหน่งทำนาย 8,400 จุดได้อย่างไร
  3. Backbone, Neck, Head ทำหน้าที่อะไร
  4. ทำไมต้องมีหลายสเกล (P3, P4, P5)
  5. Prototype กับ coefficient รวมกันเป็น mask อย่างไร
  6. NMS ทำอะไร ค่า iou threshold ส่งผลอย่างไร
  7. loss แต่ละตัวในกราฟผลเทรนวัดอะไร
  8. mAP50 กับ mAP50-95 ของ Box และ Mask ต่างกันอย่างไร
  9. เริ่มเทรนจากน้ำหนักที่เทรนมาแล้ว (pretrained) ดีกว่าเริ่มจากสุ่มอย่างไร
  10. จะรู้ได้อย่างไรว่าโมเดล overfit หรือเกิด data leakage
- **12.4 แบบประเมินตนเอง (rubric):** ตอบไม่ได้ / ตอบได้บางส่วน / อธิบายพร้อมเหตุผลได้
- **แบบทดสอบรวบยอด** 15–20 ข้อ ครอบคลุมทุกบท

**อ้างอิง:** รวมจากบทที่เกี่ยวข้อง

---

## 4. เอกสารอ้างอิง (APA 7)

> ตรวจ DOI/URL ทุกรายการก่อนนำไปอ้างในงานวิชาการ

- **[Bochkovskiy2020]** Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). *YOLOv4: Optimal speed and accuracy of object detection* (arXiv:2004.10934). arXiv. https://doi.org/10.48550/arXiv.2004.10934
- **[Bolya2019]** Bolya, D., Zhou, C., Xiao, F., & Lee, Y. J. (2019). YOLACT: Real-time instance segmentation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 9157–9166). https://doi.org/10.1109/ICCV.2019.00925
- **[CS231n]** Stanford University. (n.d.). *CS231n: Deep learning for computer vision* [Course notes]. https://cs231n.github.io
- **[Dumoulin2016]** Dumoulin, V., & Visin, F. (2016). *A guide to convolution arithmetic for deep learning* (arXiv:1603.07285). arXiv. https://doi.org/10.48550/arXiv.1603.07285
- **[Elfwing2018]** Elfwing, S., Uchibe, E., & Doya, K. (2018). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. *Neural Networks, 107*, 3–11. https://doi.org/10.1016/j.neunet.2017.12.012
- **[Everingham2010]** Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL visual object classes (VOC) challenge. *International Journal of Computer Vision, 88*(2), 303–338. https://doi.org/10.1007/s11263-009-0275-4
- **[Feng2021]** Feng, C., Zhong, Y., Gao, Y., Scott, M. R., & Huang, W. (2021). TOOD: Task-aligned one-stage object detection. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 3490–3499). https://doi.org/10.1109/ICCV48922.2021.00349
- **[Goodfellow2016]** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org
- **[He2015]** He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial pyramid pooling in deep convolutional networks for visual recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 37*(9), 1904–1916. https://doi.org/10.1109/TPAMI.2015.2389824
- **[He2016]** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90
- **[He2017]** He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 2961–2969). https://doi.org/10.1109/ICCV.2017.322
- **[Ioffe2015]** Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 448–456). PMLR. https://proceedings.mlr.press/v37/ioffe15.html
- **[Jocher2024]** Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11* [Computer software]. Ultralytics. https://github.com/ultralytics/ultralytics
  *(หมายเหตุ: Ultralytics ไม่ได้ตีพิมพ์บทความวิจัยของ YOLO11 จึงอ้างเป็นซอฟต์แวร์ตามรูปแบบที่ Ultralytics แนะนำ)*
- **[Khanam2024]** Khanam, R., & Hussain, M. (2024). *YOLOv11: An overview of the key architectural enhancements* (arXiv:2410.17725). arXiv. https://doi.org/10.48550/arXiv.2410.17725
- **[Kingma2015]** Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *3rd International Conference on Learning Representations*. https://doi.org/10.48550/arXiv.1412.6980
- **[LeCun1998]** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324. https://doi.org/10.1109/5.726791
- **[Li2020]** Li, X., Wang, W., Wu, L., Chen, S., Hu, X., Li, J., Tang, J., & Yang, J. (2020). Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 21002–21012). https://doi.org/10.48550/arXiv.2006.04388
- **[Lin2014]** Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In *Computer Vision – ECCV 2014* (pp. 740–755). Springer. https://doi.org/10.1007/978-3-319-10602-1_48
- **[Lin2017]** Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2117–2125). https://doi.org/10.1109/CVPR.2017.106
- **[Liu2018]** Liu, S., Qi, L., Qin, H., Shi, J., & Jia, J. (2018). Path aggregation network for instance segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 8759–8768). https://doi.org/10.1109/CVPR.2018.00913
- **[Loshchilov2019]** Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *7th International Conference on Learning Representations*. https://doi.org/10.48550/arXiv.1711.05101
- **[Neubeck2006]** Neubeck, A., & Van Gool, L. (2006). Efficient non-maximum suppression. In *18th International Conference on Pattern Recognition* (Vol. 3, pp. 850–855). IEEE. https://doi.org/10.1109/ICPR.2006.479
- **[Padilla2020]** Padilla, R., Netto, S. L., & da Silva, E. A. B. (2020). A survey on performance metrics for object-detection algorithms. In *2020 International Conference on Systems, Signals and Image Processing* (pp. 237–242). IEEE. https://doi.org/10.1109/IWSSIP48289.2020.9145130
- **[Redmon2016]** Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 779–788). https://doi.org/10.1109/CVPR.2016.91
- **[Rumelhart1986]** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536. https://doi.org/10.1038/323533a0
- **[Sanderson2017]** Sanderson, G. (2017). *Neural networks* [Video series]. 3Blue1Brown. https://www.3blue1brown.com/topics/neural-networks
- **[UltralyticsDocs]** Ultralytics. (n.d.). *Ultralytics YOLO docs*. https://docs.ultralytics.com
- **[Vaswani2017]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (Vol. 30). https://doi.org/10.48550/arXiv.1706.03762
- **[Wang2020]** Wang, C.-Y., Liao, H.-Y. M., Wu, Y.-H., Chen, P.-Y., Hsieh, J.-W., & Yeh, I.-H. (2020). CSPNet: A new backbone that can enhance learning capability of CNN. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops* (pp. 390–391). https://doi.org/10.1109/CVPRW50498.2020.00203
- **[Zhang2023]** Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into deep learning*. Cambridge University Press. https://d2l.ai
- **[Zheng2020]** Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). Distance-IoU loss: Faster and better learning for bounding box regression. *Proceedings of the AAAI Conference on Artificial Intelligence, 34*(07), 12993–13000. https://doi.org/10.1609/aaai.v34i07.6999

---

## 5. สิ่งที่ต้องเตรียมในเครื่อง

- Python และ `pip install ultralytics`
- จดเวอร์ชัน: `pip show ultralytics`
- น้ำหนักทางการ `yolo11n-seg.pt` (Ultralytics ดาวน์โหลดให้อัตโนมัติเมื่อเรียกใช้ครั้งแรก)
- ชุดข้อมูล `coco8-seg` (ดาวน์โหลดอัตโนมัติเมื่อสั่งเทรนด้วย `data="coco8-seg.yaml"`)
- ไฟล์น้ำหนัก (`*.pt`) และโฟลเดอร์ `runs/` **ไม่ต้อง commit** ขึ้น GitHub ให้เพิ่มใน `.gitignore`

---

## 6. ขั้นตอนการทำงานกับ Claude Code

**รอบละ 1 บท:**
1. สั่ง: `อ่าน SKILL.md และ course-plan.md แล้วสร้างบทที่ N`
2. เปิดดูในเครื่อง: `python -m http.server 8000` แล้วเข้า `http://localhost:8000`
3. **เรียนบทนั้นจริง** เล่นซิมมูเลชั่น ทำแบบทดสอบ จดสิ่งที่ยังไม่เข้าใจ
4. สั่งแก้: คำอธิบายที่ยังงง ซิมมูเลชั่นที่ยังไม่ชัด
5. ตรวจเอกสารอ้างอิงของบทนั้น
6. `git commit` และ `git push`

**ลำดับที่แนะนำ:** บท 1 → 12 ตามลำดับ ยกเว้นสคริปต์ `tools/extract_shapes.py` (บทที่ 9) ซึ่งทำก่อนบทที่ 6 ได้ เพราะบทที่ 6 ใช้ตัวเลขจาก JSON นี้

**Checklist ก่อน push แต่ละบท**
- [ ] ตรงตามเทมเพลตใน `SKILL.md`
- [ ] ครบ 8 ส่วนตาม Section 2
- [ ] ซิมมูเลชั่นใช้งานได้บนมือถือ และมีปุ่มรีเซ็ต
- [ ] ตัวเลขสถาปัตยกรรมมาจาก JSON/โค้ดจริง และระบุเวอร์ชัน ultralytics
- [ ] เอกสารอ้างอิงตรวจแล้ว
- [ ] ไม่มีไฟล์ `*.pt` หรือโฟลเดอร์ `runs/` ถูก commit
- [ ] **อธิบายเนื้อหาบทนี้ด้วยคำพูดตัวเองได้**
