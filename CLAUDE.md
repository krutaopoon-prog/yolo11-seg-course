# เข้าใจ YOLO11-seg ตั้งแต่ศูนย์ (เว็บอ่านเองส่วนตัว)

- อ่าน `course-plan.md` ก่อนทุกครั้ง (เนื้อหา 12 บท) — **ไม่ใช้** SKILL.md / repo WBI
- เป้าหมายคือผู้เรียน "เข้าใจทุกอย่าง" ไม่เน้นความสวย: เน้นคำอธิบายชัด ซิมมูเลชั่นที่ทำให้เห็นภาพ และความถูกต้อง
- ผู้เรียนอยากอ่านแบบ "เห็นภาพ" มากกว่าอ่านโค้ด: ใช้แผนภาพ/ซิม/ตัวเลขตัวอย่างเป็นหลัก โค้ดเท่าที่จำเป็นและสั้นมาก (ซ่อนใน details)
- สร้าง **ทีละบท** ตามที่สั่ง แล้วหยุดให้ผู้เรียนอ่าน ไม่ต้องย้อนแก้บทเก่าเว้นแต่สั่ง; commit เฉพาะเมื่อผู้เรียนสั่ง (repo local ยังไม่มี remote)
- โครงสร้าง: `chapterN/index.html` หน้าเดียวรวม เนื้อหา + ซิม + แบบทดสอบ (8 ส่วนตาม course-plan §2); สคริปต์ Python ใน `tools/`, JSON ใน `data/`; อัปเดตลิงก์ใน `index.html` และ footer ของบทก่อนหน้าเมื่อเพิ่มบท
- ตัวเลขภายใน YOLO11 ต้องได้จากการรันโค้ด/อ่าน source ของ `ultralytics` ที่ติดตั้ง (ตอนนี้ 8.4.2) ห้ามเขียนจากความจำ และระบุเวอร์ชันใน footer
- ดูในเครื่อง: `python -m http.server 8000` → http://localhost:8000

## ความคืบหน้า
- เสร็จแล้ว: บทที่ 1 (ภาพคือตัวเลข, letterbox), 2 (นิวรอน, activation), 3 (loss, gradient descent, optimizer) — บทถัดไปคือ **บทที่ 4**
- ตัวอย่างประจำวิชา "ผลไม้ในตะกร้า" และโจทย์พิกเซลผลไม้ (แอปเปิลแดง/เขียว, ส้ม vs แอปเปิล+กล้วย) ใช้ต่อเนื่องจากบท 2→3 ควรใช้ซ้ำเพื่อร้อยเรื่อง

## `assets/common.js` (ห้ามประกาศชื่อเหล่านี้ซ้ำในหน้าบท — จะชน)
`pyRound, letterbox, rng, FRUITS, sceneFrame, drawFruitBasket, makeFruitImage, fruitPixels, logistic, setupCanvas, canvasXY, PIPELINE_STEPS, renderPipeline, initQuiz`

## ข้อเท็จจริงที่ตรวจแล้ว (ultralytics 8.4.2) ไว้อ้างในบทถัดไป
- predict กับ .pt ใช้ letterbox auto=True (1280×720 → (1,3,384,640)); pad 114; หาร 255 อย่างเดียว
- yolo11n-seg: 90 บล็อก Conv (86 ใช้ SiLU, 4 ใน C2PSA ไม่มี act), ชั้นสุดท้ายของ Head เป็น Conv2d ไม่มี act, คะแนนคลาสผ่าน sigmoid, params 2,876,848
- optimizer=auto ไม่ใช้ lr0/momentum: ก้าวรวม > 10,000 → SGD lr 0.01; ไม่งั้น MuSGD lr = 0.01/(4+nc) (coco8-seg: 0.000119); cls loss = BCEWithLogitsLoss; gain box 7.5, seg 7.5 (ใช้ hyp.box), cls 0.5, dfl 1.5; results.csv มี box/seg/cls/dfl/sem_loss (sem_loss = 0 สำหรับ yolo11-seg)

## บทเรียนจากการทำงาน
- ทุกข้อความใน "สิ่งที่ควรสังเกต" ต้องตรวจด้วยการรันซิมจริง (กดปุ่มผ่าน JS แล้วอ่านผล) ก่อนส่ง — เคยเจอข้อความไม่ตรงกับพฤติกรรมจริงหลายครั้ง
- screenshot ของ claude-in-chrome มัก timeout ในแท็บเดิม → เปิดแท็บใหม่แล้วถ่าย
- `yolo11n-seg.pt` ไม่อยู่ใน repo (ultralytics ดาวน์โหลดเองเมื่อเรียกใช้; ให้รันในโฟลเดอร์ชั่วคราว) ชุดข้อมูลอยู่ที่ `D:\yolov11\datasets`
