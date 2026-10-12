# เข้าใจ YOLO11-seg ตั้งแต่ศูนย์ (เว็บอ่านเองส่วนตัว)

- อ่าน `course-plan.md` ก่อนทุกครั้ง (เนื้อหา 12 บท) — **ไม่ใช้** SKILL.md / repo WBI
- เป้าหมายคือผู้เรียน "เข้าใจทุกอย่าง" ไม่เน้นความสวย: เน้นคำอธิบายชัด ซิมมูเลชั่นที่ทำให้เห็นภาพ และความถูกต้อง
- ผู้เรียนอยากอ่านแบบ "เห็นภาพ" มากกว่าอ่านโค้ด: ใช้แผนภาพ/ซิม/ตัวเลขตัวอย่างเป็นหลัก โค้ดเท่าที่จำเป็นและสั้นมาก (ซ่อนใน details)
- สร้าง **ทีละบท** ตามที่สั่ง แล้วหยุดให้ผู้เรียนอ่าน ไม่ต้องย้อนแก้บทเก่าเว้นแต่สั่ง; commit เฉพาะเมื่อผู้เรียนสั่ง (repo local ยังไม่มี remote)
- โครงสร้าง: `chapterN/index.html` หน้าเดียวรวม เนื้อหา + ซิม + แบบทดสอบ (8 ส่วนตาม course-plan §2); สคริปต์ Python ใน `tools/`, JSON ใน `data/`; อัปเดตลิงก์ใน `index.html` และ footer ของบทก่อนหน้าเมื่อเพิ่มบท
- ตัวเลขภายใน YOLO11 ต้องได้จากการรันโค้ด/อ่าน source ของ `ultralytics` ที่ติดตั้ง (ตอนนี้ 8.4.2) ห้ามเขียนจากความจำ และระบุเวอร์ชันใน footer
- ดูในเครื่อง: `python -m http.server 8000` → http://localhost:8000

## ความคืบหน้า
- เสร็จแล้ว: บทที่ 1 (ภาพคือตัวเลข, letterbox), 2 (นิวรอน, activation), 3 (loss, gradient descent, optimizer), 4 (backprop, epoch/batch, overfitting, augmentation, transfer, leakage), 5 (convolution, stride/padding, multi-channel, kernel จริงชั้นแรก), 6 (P1–P5, receptive field, pooling, BN, residual, CSP/C3k2) — บทถัดไปคือ **บทที่ 7**
- ตัวอย่างประจำวิชา "ผลไม้ในตะกร้า" และโจทย์พิกเซลผลไม้ (แอปเปิลแดง/เขียว, ส้ม vs แอปเปิล+กล้วย) ใช้ต่อเนื่องจากบท 2→3 ควรใช้ซ้ำเพื่อร้อยเรื่อง

## `assets/common.js` (ห้ามประกาศชื่อเหล่านี้ซ้ำในหน้าบท — จะชน)
`pyRound, letterbox, rng, FRUITS, sceneFrame, drawFruitBasket, makeFruitImage, fruitPixels, logistic, setupCanvas, canvasXY, PIPELINE_STEPS, renderPipeline, initQuiz`

## `assets/tensor3d.js` (ฉาก 3D ใช้ร่วม: class `Scene3D`)
โหลด three.js r128 จาก cdnjs ก่อน common.js แล้วตามด้วย tensor3d.js · มี boxes/place (InstancedMesh), outline, line/setLine, setLabels, setBounds (จัดกล้องอัตโนมัติ), bindButtons (data-view3d, data-zoom3d), onPick (คลิกไม่ลาก) · ใช้ในบท 1 (ซิม 1.4), 3 (ซิม 3.3), 5 (ซิม 5.3, 5.4)

## ข้อเท็จจริงที่ตรวจแล้ว (ultralytics 8.4.2) ไว้อ้างในบทถัดไป
- predict กับ .pt ใช้ letterbox auto=True (1280×720 → (1,3,384,640)); pad 114; หาร 255 อย่างเดียว
- yolo11n-seg: 90 บล็อก Conv (86 ใช้ SiLU, 4 ใน C2PSA ไม่มี act), ชั้นสุดท้ายของ Head เป็น Conv2d ไม่มี act, คะแนนคลาสผ่าน sigmoid, params 2,876,848
- optimizer=auto ไม่ใช้ lr0/momentum: ก้าวรวม > 10,000 → SGD lr 0.01; ไม่งั้น MuSGD lr = 0.01/(4+nc) (coco8-seg: 0.000119); cls loss = BCEWithLogitsLoss; gain box 7.5, seg 7.5 (ใช้ hyp.box), cls 0.5, dfl 1.5; results.csv มี box/seg/cls/dfl/sem_loss (sem_loss = 0 สำหรับ yolo11-seg)

- บท 4: train ใช้ batch 16, ไม่ drop batch สุดท้าย (เว้น compile); nbs 64 → accumulate = round(64/batch) หลัง warmup; warmup = max(round(3×nb),100) iteration; patience 100; fitness = mAP50-95 อย่างเดียว (seg = B + M); val ใช้ augment=False
- augmentation default: fliplr 0.5, scale 0.5 (×0.5–1.5), translate 0.1, hsv_h 0.015 (บวก ±5.4°), hsv_s 0.7, hsv_v 0.4 (คูณ), mosaic 1.0, close_mosaic 10, degrees/mixup/copy_paste 0; border 114
- transfer จาก yolo11n-seg.pt: nc=80 โอน 561/561; nc=3 โอน 510/561 (98.4% params) ไม่โอน model.23.cv3.* เพราะ c3 = max(ch0, min(nc,100)); coco8-seg = 8 ภาพแรกของ COCO train2017 (val รั่วกับ pretrained) ข้อมูลอยู่ใน `data/ch4_transfer.json` (สร้างโดย `tools/ch4_transfer.py`)

- บท 5: Conv = Conv2d(bias=False) → BN → SiLU, autopad P = k//2; ชั้น 0: 3→16 k3 s2 (432 weight) → (1,16,320,320); ชั้น 1: 16→32 (4,608) → (1,32,160,160); ชั้น 3: 64→64 (36,864) → (1,64,80,80); shape ชั้น 0–4 อยู่ใน `data/ch5_first_layer.json` (มี weight+BN ชั้นแรก) สร้างโดย `tools/ch5_first_layer.py` ซึ่งเทียบผล JS กับ PyTorch ได้ (ตรง 5 ตำแหน่งทศนิยม)
- ผู้เรียนชอบภาพ 3D สำหรับเรื่อง tensor/ช่อง/พื้นผิว loss: บท 6, 9, 10 ควรใช้ Scene3D (พีระมิด feature, Concat/Upsample, prototype 32 แผ่น)
- ข้อควรระวัง three r128: Box3 ไม่นับ instance (ใช้ setBounds เอง), เส้นที่เปลี่ยนจุดต้อง computeBoundingSphere (setLine ทำให้แล้ว), พื้นผิวโค้งหงายทำให้เส้นบนพื้นผิวจมใต้ตาข่าย → depthTest false
- heredoc ใน Bash tool พังเมื่อเนื้อหามี backtick/quote ซับซ้อน → เขียนสคริปต์แก้ไฟล์ด้วย Write แล้วค่อยรัน
- three.js: วัตถุโปร่งใสถูกวาดหลังวัตถุทึบเสมอ เส้นที่ต้องเห็นทับแผ่นโปร่งใสต้องตั้ง transparent=true + depthTest=false + renderOrder สูง; Scene3D.pick เลือกเฉพาะ Mesh (เส้นมี threshold กว้าง 1 หน่วย)
- console 404 ในทุกหน้าคือ favicon เท่านั้น (ตรวจด้วย context response listener แล้ว) · เช็กความกว้างมือถือ: scrollWidth ต้อง = 375 (URL ยาวในอ้างอิงเคยดันเกิน แก้ด้วย overflow-wrap ใน style.css แล้ว)
- kernel ชั้นแรกจริง: 6/3 = ขอบแนวตั้ง, 8/7 = ขอบแนวนอน, 12 = เขียว, 15 = แดง, 5 = องุ่น/แดง (G ลบ), 13 แทบ 0 กับภาพผลไม้

- บท 6: `tools/extract_shapes.py` → `data/yolo11-seg_shapes.json` (ทุกชั้นทั้ง 5 รุ่น: type, from, params, out shape; params รวมตรงกับ yaml) + `c3k2_n` (โครงสร้างภายใน C3k2) + `rf_n` (receptive field วัดด้วย gradient, สลับ MaxPool→AvgPool) และ `data/ch6_maps/` (feature map จริงชั้น 0–10 ของ `data/fruit_letterbox_640.png`) — บท 9, 10 ใช้ JSON นี้ต่อได้เลย
- shape yolo11n-seg: 0:(16,320²) 1:(32,160²) 2:(64,160²) 3:(64,80²) 4:(128,80²) 5:(128,40²) 6:(128,40²) 7:(256,20²) 8–10:(256,20²); Neck 11–22; Segment ชั้น 23 รับ [16,19,22]
- RF เซลล์กลาง: 3, 7, 23, 31, 63, 79, 207, 239, 495, ทั้งภาพ (SPPF), ทั้งภาพ (C2PSA) = สูตร +(K−1)×stride ตรงทุกชั้น
- C3k2 = C2f: cv1 1×1 → chunk(2) → Bottleneck(e=0.5, shortcut) หรือ C3k (ชั้น 6, 8, 22 เพราะ yaml ตั้ง True; รุ่น m/l/x parse_model บังคับ c3k=True ทุกตัว) → cat → cv2; ชั้น 2: 32→32=16+16, Bottleneck 16→8→16, cat 48 → 64, params 6,640

## บทเรียนจากการทำงาน
- ทุกข้อความใน "สิ่งที่ควรสังเกต" ต้องตรวจด้วยการรันซิมจริง (กดปุ่มผ่าน JS แล้วอ่านผล) ก่อนส่ง — เคยเจอข้อความไม่ตรงกับพฤติกรรมจริงหลายครั้ง
- screenshot ของ claude-in-chrome มัก timeout ในแท็บเดิม → เปิดแท็บใหม่แล้วถ่าย
- ถ้า claude-in-chrome ไม่เชื่อมต่อ: ใช้ Playwright (ติดตั้งแล้วใน Python) กับ Chrome ในเครื่อง `chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")` กดปุ่ม/อ่าน innerHTML/ถ่าย screenshot ได้ครบ (ตั้ง PYTHONIOENCODING=utf-8)
- `yolo11n-seg.pt` ไม่อยู่ใน repo (ultralytics ดาวน์โหลดเองเมื่อเรียกใช้; ให้รันในโฟลเดอร์ชั่วคราว) ชุดข้อมูลอยู่ที่ `D:\yolov11\datasets`
